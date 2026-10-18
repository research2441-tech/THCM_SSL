"""Reviewer-oriented executable sensitivity and comparison building blocks.
Run each configuration with the same training split/seeds; do not assert unmeasured superiority.
"""
import argparse,itertools,json
from pathlib import Path
import pandas as pd,numpy as np,yaml
from thcm_ssl import TemporalHypergraph
from baselines import make_hypergraph_variant

def hypergraph_sensitivity(train_df,config,grid=None):
    hp=config['hypergraph'].copy()
    grid=grid or {'tau':[.5,1.,3.,6.,12.],'prune_quantile':[0.,.1,.2,.3,.4],'mode':['temporal_hypergraph','pairwise_graph','static_hypergraph','no_class_weight']}
    rows=[]
    for tau,pct,mode in itertools.product(grid['tau'],grid['prune_quantile'],grid['mode']):
        options=dict(span=hp['token_span'],tau=tau,alpha=hp['alpha'],beta=hp['beta'],gamma=hp['gamma'],prune_quantile=pct,max_edges=hp['max_edges'],min_support=hp['min_support'])
        hg=make_hypergraph_variant(mode,**options)
        # Model selection never uses future-time test posts.
        hg.update(train_df.text.astype(str).tolist(),list(pd.factorize(train_df.label)[0]))
        rows.append({'tau':tau,'prune_quantile':pct,'mode':mode,**hg.stats()})
    return pd.DataFrame(rows)
def mix_grid(step=.25):
    count=int(round(1/step))
    return [{'alpha':i*step,'beta':j*step,'gamma':(count-i-j)*step} for i in range(count+1) for j in range(count+1-i)]
def shifts(train,test):
    """Deterministic shift subsets without viewing labels or tuning model on test."""
    from thcm_ssl import tokenize
    train_text=' '.join(train.text.astype(str))
    seen=set(tokenize(train_text));train_platforms=set(train.platform.astype(str)) if 'platform' in train else set()
    result=[]
    for i,r in test.iterrows():
        tok=tokenize(str(r.text));hashtags=[x for x in tok if x.startswith('#')]
        unseen=any(x not in seen for x in hashtags)
        new_platform='platform' in test and str(r['platform']) not in train_platforms
        result.append({'index':i,'unseen_hashtag':unseen,'new_platform':new_platform,'new_vocabulary_fraction':sum(x not in seen for x in tok)/max(len(tok),1)})
    return pd.DataFrame(result)
def main():
    p=argparse.ArgumentParser();p.add_argument('--train',default='train.csv');p.add_argument('--test',default='test.csv');p.add_argument('--config',default='config.yaml');p.add_argument('--out',default='outputs');a=p.parse_args()
    train=pd.read_csv(a.train);test=pd.read_csv(a.test);cfg=yaml.safe_load(open(a.config));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    hypergraph_sensitivity(train,cfg).to_csv(out/'graph_sensitivity.csv',index=False)
    pd.DataFrame(mix_grid()).to_csv(out/'mixture_candidate_grid.csv',index=False)
    shifts(train,test).to_csv(out/'shift_masks.csv',index=False)
    print('Saved measured graph-density sensitivity, candidate weights and test-shift masks. Predictive sensitivity requires retraining each configuration; these CSVs do NOT claim F1 results.')
if __name__=='__main__':main()
