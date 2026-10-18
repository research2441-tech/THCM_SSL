"""Train THCM-SSL using only chronological train/validation/calibration partitions."""
import argparse,json,random,time,platform,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from thcm_ssl import THCMSSL,TokenVocabulary,TemporalHypergraph,batch_adjacency,ContinualMemory,TemperatureCalibrator

class TextSet(Dataset):
    def __init__(self,frame,text_col='text',label_col='label'):self.texts=frame[text_col].astype(str).tolist();self.labels=frame[label_col].astype(int).tolist()
    def __len__(self):return len(self.texts)
    def __getitem__(self,i):return self.texts[i],self.labels[i]

def seed_everything(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def loader(frame,vocab,hg,batch_size,max_len,device='cpu',shuffle=False):
    def collate(rows):
        texts,ys=zip(*rows);ids,mask,adj=batch_adjacency(texts,vocab,hg,max_len,'cpu')
        return ids,mask,adj,torch.tensor(ys,dtype=torch.long)
    return DataLoader(TextSet(frame),batch_size=batch_size,shuffle=shuffle,collate_fn=collate)

@torch.no_grad()
def infer(model,data,device='cpu'):
    model.eval();ys=[];logits=[];rep=[]
    for ids,mask,adj,y in data:
        z,_,h=model(ids.to(device),mask.to(device),adj.to(device),return_features=True)
        ys.extend(y.tolist());logits.append(z.cpu());rep.append(h.cpu())
    if not logits:raise ValueError('Empty dataset')
    return np.array(ys),torch.cat(logits),torch.cat(rep)

def run(cfg,input_dir='.',output_dir='outputs',seed=11,epochs=None):
    seed_everything(seed);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    dev=cfg['experiment']['device'];dev=('cuda' if torch.cuda.is_available() else 'cpu') if dev=='auto' else dev
    parts={k:pd.read_csv(Path(input_dir)/f'{k}.csv') for k in ('train','validation','calibration','test')}
    # Label encoder fitted strictly to training partition; unseen class is rejected rather than remapped.
    enc=LabelEncoder().fit(parts['train']['label'].astype(str))
    for name,df in parts.items():
        labels=df['label'].astype(str)
        if not set(labels).issubset(set(enc.classes_)):raise ValueError(f'Unseen labels in {name}; explicit class expansion required')
        df['label']=enc.transform(labels)
    dat=cfg['data'];hp=cfg['hypergraph'];tc=cfg['training'];mc=cfg['model'];cc=cfg['continual']
    vocab=TokenVocabulary(dat['max_vocab']).fit(parts['train']['text'])
    hg=TemporalHypergraph(span=hp['token_span'],max_edges=hp['max_edges'],max_nodes=hp['max_nodes'],tau=hp['tau'],alpha=hp['alpha'],beta=hp['beta'],gamma=hp['gamma'],prune_quantile=hp['prune_quantile'],min_support=hp['min_support'])
    # No validation or test content is used to compute graph statistics.
    hg.update(parts['train']['text'].astype(str).tolist(),parts['train']['label'].tolist())
    model=THCMSSL(len(vocab.stoi),len(enc.classes_),mc['embedding_dim'],mc['hidden_dim'],mc['ssm_layers'],mc['ssm_state_dim'],mc['dropout']).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    memory=ContinualMemory(cc['prototype_eta'],cc['retention_coefficient'])
    tr=loader(parts['train'],vocab,hg,tc['batch_size'],dat['max_length'],shuffle=True)
    va=loader(parts['validation'],vocab,hg,tc['batch_size'],dat['max_length'])
    best=-1.;bad=0;history=[];started=time.time()
    for epoch in range(epochs or tc['epochs']):
        model.train();losses=[]
        temp=mc['gate_temp_start']+(mc['gate_temp_end']-mc['gate_temp_start'])*epoch/max((epochs or tc['epochs'])-1,1)
        for ids,mask,adj,y in tr:
            ids,mask,adj,y=[x.to(dev) for x in (ids,mask,adj,y)]
            opt.zero_grad(set_to_none=True)
            logits,gate=model(ids,mask,adj,gate_temp=temp,stochastic_gate=True)
            loss=torch.nn.functional.cross_entropy(logits,y)+mc['gate_sparsity']*gate+memory.penalty(model)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();losses.append(float(loss.detach()))
        vy,vlog,_=infer(model,va,dev)
        score=f1_score(vy,vlog.argmax(-1).numpy(),average='macro',zero_division=0)
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_macro_f1':float(score)})
        if score>best+1e-6:
            best=score;bad=0;torch.save(model.state_dict(),out/'best_model.pt')
        else:
            bad+=1
            if bad>=tc['patience']:break
    model.load_state_dict(torch.load(out/'best_model.pt',map_location=dev,weights_only=True))
    cal=loader(parts['calibration'],vocab,hg,tc['batch_size'],dat['max_length'])
    cy,clog,_=infer(model,cal,dev)
    temperature=TemperatureCalibrator();temperature.fit(clog,cy)
    ty,tlog,tfeats=infer(model,loader(parts['test'],vocab,hg,tc['batch_size'],dat['max_length']),dev)
    probs=temperature.probabilities(tlog).numpy()
    pred=probs.argmax(axis=1)
    pd.DataFrame({'target':ty,'prediction':pred,'confidence':probs.max(axis=1),'seed':seed,**{f'p_{str(c)}':probs[:,i] for i,c in enumerate(enc.classes_)}}).to_csv(out/'predictions.csv',index=False)
    memory.update_prototypes(tfeats,torch.tensor(ty)) # evaluation example only; never fed back into training
    metadata={'seed':seed,'config':cfg,'classes':list(enc.classes_),'vocabulary':vocab.stoi,'graph_statistics':hg.stats(),'temperature':temperature.temperature,'training_seconds':time.time()-started,'model_parameters':sum(p.numel() for p in model.parameters()),'device':dev,'torch_version':torch.__version__,'python_version':sys.version,'validation_macro_f1':best,'test_evaluated_once':True,'implementation':'reference selective diagonal SSM; not official Mamba'}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2,default=str));(out/'history.json').write_text(json.dumps(history,indent=2))
    return metadata

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='config.yaml');ap.add_argument('--input-dir',default='.');ap.add_argument('--output-dir',default='outputs');ap.add_argument('--seed',type=int,default=11);ap.add_argument('--epochs',type=int);args=ap.parse_args()
    with open(args.config) as f:cfg=yaml.safe_load(f)
    print(json.dumps({k:v for k,v in run(cfg,args.input_dir,args.output_dir,args.seed,args.epochs).items() if k not in ('vocabulary','config')},indent=2,default=str))
if __name__=='__main__':main()
