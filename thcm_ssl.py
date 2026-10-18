"""THCM-SSL reference implementation. Experimental reimplementation, not original author source."""
from __future__ import annotations
import math, re, collections
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TOKEN_RE = re.compile(r'#[\w_]+|@[\w_]+|[\U0001F300-\U0001FAFF]|\w+(?:[\'’]\w+)?|[^\w\s]', re.UNICODE)
def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())
def node_type(token):
    if token.startswith('#'): return 'hashtag'
    if token.startswith('@'): return 'mention'
    if any(ord(c) > 0x1F000 for c in token): return 'emoji'
    return 'word'

@dataclass
class Hyperedge:
    nodes: tuple
    weight: float
    last_seen: int
    support: int

class TemporalHypergraph:
    """Causal edge memory. Only current/past training examples may update statistics."""
    def __init__(self, span=8, max_edges=15000, max_nodes=30000, tau=3., alpha=.4, beta=.35, gamma=.25, prune_quantile=.2, min_support=1, pairwise=False, decay=True, class_weight=True):
        assert span >= 2 and tau > 0 and abs(alpha+beta+gamma-1) < 1e-6
        self.span,self.max_edges,self.max_nodes=span,max_edges,max_nodes
        self.tau,self.alpha,self.beta,self.gamma=tau,alpha,beta,gamma
        self.prune_quantile,self.min_support=prune_quantile,min_support
        self.pairwise,self.decay,self.class_weight=pairwise,decay,class_weight
        self.edges={}; self.counts=collections.Counter(); self.docs=collections.Counter(); self.label_counts=collections.defaultdict(collections.Counter); self.steps=0
        self.last_pruned=[]
    def _candidates(self,tokens):
        result=[]
        for i in range(0,len(tokens),self.span):
            window=tokens[i:i+self.span]
            unique=list(dict.fromkeys(window))
            if len(unique)<2: continue
            if self.pairwise:
                result.extend(tuple(sorted((a,b))) for j,a in enumerate(unique) for b in unique[j+1:])
            else: result.append(tuple(sorted(unique)))
        return result
    def update(self, texts, labels=None):
        self.steps+=1
        local=collections.Counter()
        for k,txt in enumerate(texts):
            toks=tokenize(txt)
            self.counts.update(toks)
            self.docs.update(set(toks))
            if labels is not None:
                for t in set(toks): self.label_counts[t][int(labels[k])]+=1
            local.update(self._candidates(toks))
        total=max(sum(self.counts.values()),1)
        max_count=max(self.counts.values(),default=1)
        for key,support in local.items():
            if support<self.min_support: continue
            salience=float(np.mean([(self.counts[t]/total)*math.log((self.steps*max(1,len(texts))+1)/(self.docs[t]+1)+1) for t in key]))
            coherence= support/(support+max(1,np.mean([self.counts[t] for t in key])))
            relevance=0.
            if self.class_weight and labels is not None:
                relevance=float(np.mean([max(self.label_counts[t].values(),default=0)/max(sum(self.label_counts[t].values()),1) for t in key]))
            # Monotone bounded normalization prevents a frequent term from exploding weights.
            raw=self.alpha*min(salience*max_count,1.)+self.beta*coherence+self.gamma*relevance
            old=self.edges.get(key)
            if old:
                aged=old.weight * (math.exp(-(self.steps-old.last_seen)/self.tau) if self.decay else 1.)
                raw=max(raw, aged)
            self.edges[key]=Hyperedge(key,float(max(raw,1e-8)),self.steps,(old.support if old else 0)+support)
        for key,edge in list(self.edges.items()):
            if edge.last_seen!=self.steps and self.decay:
                self.edges[key]=Hyperedge(edge.nodes,edge.weight*math.exp(-(self.steps-edge.last_seen)/self.tau),edge.last_seen,edge.support)
        if self.edges:
            thresh=float(np.quantile([e.weight for e in self.edges.values()],self.prune_quantile))
            remove=[k for k,e in self.edges.items() if e.weight<thresh or e.support<self.min_support]
            for k in remove: del self.edges[k]
            if len(self.edges)>self.max_edges:
                remove += [k for k,e in sorted(self.edges.items(),key=lambda kv:kv[1].weight)[:len(self.edges)-self.max_edges]]
                for k in remove: del self.edges[k]
            self.last_pruned=remove
    def token_context(self,tokens):
        # Sparse bounded-size local weighted incidence propagation (two-hop
        # token->hyperedge->token); avoids ever materializing a dense VxV matrix.
        lookup=collections.defaultdict(list)
        for e in self.edges.values():
            for t in e.nodes: lookup[t].append(e)
        context=[]
        for t in tokens:
            near=collections.Counter()
            for edge in lookup[t][:64]:
                for other in edge.nodes:
                    if other!=t: near[other]+=edge.weight/max(len(edge.nodes),1)
            context.append(near.most_common(8))
        return context
    def stats(self):
        return {'window':self.steps,'active_edges':len(self.edges),'observed_nodes':len(self.counts),'pruned_last_window':len(self.last_pruned)}

class TokenVocabulary:
    def __init__(self,max_size=30000): self.max_size=max_size;self.stoi={'<pad>':0,'<unk>':1}
    def fit(self,texts):
        counts=collections.Counter(t for s in texts for t in tokenize(s))
        for token,_ in counts.most_common(self.max_size-2): self.stoi.setdefault(token,len(self.stoi))
        return self
    def encode(self,text,max_len=128):
        values=[self.stoi.get(t,1) for t in tokenize(text)[:max_len]]
        return values or [1]
    def batch(self,texts,max_len=128,device='cpu'):
        tokens=[self.encode(t,max_len) for t in texts]
        width=max(len(t) for t in tokens)
        ids=torch.zeros(len(tokens),width,dtype=torch.long,device=device)
        mask=torch.zeros_like(ids,dtype=torch.bool)
        for i,t in enumerate(tokens): ids[i,:len(t)]=torch.tensor(t,device=device);mask[i,:len(t)]=True
        return ids,mask

class SparseFeatureGate(nn.Module):
    def __init__(self,dim):
        super().__init__();self.linear=nn.Linear(dim,dim)
    def forward(self,x,temperature=1.,stochastic=False):
        logits=self.linear(x)
        if stochastic and self.training:
            u=torch.rand_like(logits).clamp(1e-6,1-1e-6)
            logits=logits+torch.log(u)-torch.log1p(-u)
        gate=torch.sigmoid(logits/max(temperature,0.01))
        return x*gate, gate.mean()

class SelectiveSSMBlock(nn.Module):
    """Input-dependent diagonal selective recurrence; not a fused official Mamba kernel."""
    def __init__(self,dim,state_dim=64,dropout=.1):
        super().__init__();self.dim=dim;self.state_dim=state_dim
        self.norm=nn.LayerNorm(dim);self.dt=nn.Linear(dim,dim)
        self.b=nn.Linear(dim,dim);self.c=nn.Linear(dim,dim)
        self.a_log=nn.Parameter(torch.zeros(dim));self.out=nn.Linear(dim,dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        # channel-wise diagonal SSM, state_dim is documented reserved parameter
        # for compatibility; effective recurrent state is exactly `dim`.
        z=self.norm(x);state=torch.zeros_like(z[:,0]);out=[]
        a=-torch.exp(self.a_log).unsqueeze(0)
        for t in range(z.shape[1]):
            dt=F.softplus(self.dt(z[:,t])).clamp(max=8)
            decay=torch.exp(a*dt)
            new=decay*state+(1-decay)*self.b(z[:,t])
            if mask is not None: state=torch.where(mask[:,t,None],new,state)
            else: state=new
            out.append(state*torch.sigmoid(self.c(z[:,t])))
        y=self.dropout(self.out(torch.stack(out,dim=1)))
        return x+y

class THCMSSL(nn.Module):
    def __init__(self,vocab_size,num_classes,embedding_dim=300,hidden_dim=256,layers=4,ssm_state_dim=64,dropout=.1,use_gate=True,use_hypergraph=True,encoder='ssm'):
        super().__init__();self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.proj=nn.Linear(embedding_dim,hidden_dim)
        self.hg1=nn.Linear(hidden_dim,hidden_dim);self.hg2=nn.Linear(hidden_dim,hidden_dim)
        self.gate=SparseFeatureGate(hidden_dim);self.use_gate=use_gate;self.use_hypergraph=use_hypergraph
        self.encoder_type=encoder
        if encoder=='ssm': self.blocks=nn.ModuleList([SelectiveSSMBlock(hidden_dim,ssm_state_dim,dropout) for _ in range(layers)])
        elif encoder=='bilstm': self.blocks=nn.LSTM(hidden_dim,hidden_dim//2,num_layers=max(1,layers//2),batch_first=True,bidirectional=True)
        elif encoder=='transformer': self.blocks=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim,4,hidden_dim*4,dropout,batch_first=True),layers)
        else: raise ValueError('unknown encoder')
        self.classifier=nn.Linear(hidden_dim,num_classes)
    def forward(self,ids,mask,adj=None,gate_temp=1.,stochastic_gate=False,return_features=False):
        x=self.proj(self.embedding(ids))
        if self.use_hypergraph:
            # adj [B,L,L] weighted, constructed causally from trained HG memory.
            if adj is None: adj=torch.eye(ids.shape[1],device=ids.device)[None].expand(ids.shape[0],-1,-1)
            x=F.gelu(self.hg1(torch.bmm(adj,x)))
            x=F.gelu(self.hg2(torch.bmm(adj,x)))
        if self.use_gate: x,sparsity=self.gate(x,gate_temp,stochastic_gate)
        else: sparsity=x.new_zeros(())
        if self.encoder_type=='ssm':
            for block in self.blocks: x=block(x,mask)
        elif self.encoder_type=='bilstm': x,_=self.blocks(x)
        else: x=self.blocks(x,src_key_padding_mask=~mask)
        pooled=(x*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)
        logits=self.classifier(pooled)
        return (logits,sparsity,pooled) if return_features else (logits,sparsity)

def batch_adjacency(texts,vocab,hypergraph,max_len,device='cpu'):
    ids,mask=vocab.batch(texts,max_len,device)
    b,n=ids.shape; adj=torch.eye(n,device=device).unsqueeze(0).repeat(b,1,1)
    for i,text in enumerate(texts):
        tokens=tokenize(text)[:max_len] or ['<unk>']
        token2pos=collections.defaultdict(list)
        for j,t in enumerate(tokens):token2pos[t].append(j)
        for edge in hypergraph.edges.values():
            positions=[p for t in edge.nodes for p in token2pos.get(t,[])]
            if len(positions)<2:continue
            scale=edge.weight/len(positions)
            for p in positions:
                for q in positions:adj[i,p,q]+=scale
    adj=adj*mask.unsqueeze(1)*mask.unsqueeze(2)
    return ids,mask,adj/adj.sum(-1,keepdim=True).clamp_min(1e-8)

class ContinualMemory:
    def __init__(self,eta=.1,lambda_retain=.4):
        self.eta=eta;self.lambda_retain=lambda_retain
        self.prototypes={};self.theta={};self.fisher={}
    def update_prototypes(self,features,labels,confidence=None,threshold=0.):
        x=features.detach().cpu();ys=labels.detach().cpu().tolist()
        for j,c in enumerate(ys):
            if confidence is not None and confidence[j]<threshold:continue
            z=x[j]
            self.prototypes[c]=(1-self.eta)*self.prototypes[c]+self.eta*z if c in self.prototypes else z.clone()
    def consolidate(self,model,loader,device='cpu',max_batches=30):
        self.theta={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
        fisher={k:torch.zeros_like(p) for k,p in model.named_parameters() if p.requires_grad}
        count=0;model.eval()
        for batch in loader:
            ids,mask,adj,y=batch
            ids,mask,adj,y=[v.to(device) for v in (ids,mask,adj,y)]
            for i in range(len(y)):
                model.zero_grad(set_to_none=True)
                logits,_=model(ids[i:i+1],mask[i:i+1],adj[i:i+1])
                F.cross_entropy(logits,y[i:i+1]).backward()
                for k,p in model.named_parameters():
                    if p.grad is not None and k in fisher:fisher[k]+=p.grad.detach()**2
                count+=1
            if count>=max_batches:break
        for k in fisher:fisher[k]/=max(count,1)
        if not self.fisher:self.fisher=fisher
        else:
            for k in fisher:self.fisher[k]=self.fisher.get(k,torch.zeros_like(fisher[k]))+fisher[k]
        model.zero_grad(set_to_none=True)
    def penalty(self,model):
        val=None
        for k,p in model.named_parameters():
            if k in self.theta and k in self.fisher:
                term=(self.fisher[k]*(p-self.theta[k]).square()).sum()
                val=term if val is None else val+term
        return self.lambda_retain*.5*val if val is not None else next(model.parameters()).new_zeros(())

class TemperatureCalibrator:
    def __init__(self):self.temperature=1.
    def fit(self,logits,labels):
        logits=torch.as_tensor(logits,dtype=torch.float32);labels=torch.as_tensor(labels,dtype=torch.long)
        log_t=torch.tensor(0.,requires_grad=True)
        opt=torch.optim.LBFGS([log_t],lr=.1,max_iter=60)
        def closure():
            opt.zero_grad();loss=F.cross_entropy(logits/log_t.exp().clamp(.05,100),labels);loss.backward();return loss
        opt.step(closure);self.temperature=float(log_t.exp().clamp(.05,100).detach())
        return self.temperature
    def probabilities(self,logits):return torch.softmax(torch.as_tensor(logits)/self.temperature,dim=-1)
