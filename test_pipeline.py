"""Tests: pytest -q test_pipeline.py"""
import pandas as pd,numpy as np,torch
from thcm_ssl import tokenize,TokenVocabulary,TemporalHypergraph,batch_adjacency,THCMSSL,ContinualMemory,TemperatureCalibrator
from prepare_data import prepare
from evaluate import metrics,risk_coverage
from statistics_tests import holm,paired_bootstrap

def test_tokenization():assert '#launch' in tokenize('Love #Launch 😀')
def test_hypergraph_pruning():
    g=TemporalHypergraph(span=3,prune_quantile=.25,tau=1.)
    g.update(['#test good 😀','#test bad 😢','amazing product now'],[1,0,1])
    assert len(g.edges)>0
    before=set(g.edges);g.update(['new issue today'],[0]);assert g.steps==2
    assert len(g.edges)<=g.max_edges

def test_differentiable_model():
    vocab=TokenVocabulary().fit(['good #deal 😀','bad launch 😡'])
    g=TemporalHypergraph();g.update(['good #deal 😀','bad launch 😡'],[1,0])
    ids,mask,adj=batch_adjacency(['good #deal 😀','bad launch 😡'],vocab,g,64)
    model=THCMSSL(len(vocab.stoi),2,embedding_dim=24,hidden_dim=16,layers=1,ssm_state_dim=8)
    logits,gate=model(ids,mask,adj,stochastic_gate=True)
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([1,0]))+.01*gate
    loss.backward()
    assert torch.isfinite(loss)
    assert model.gate.linear.weight.grad is not None

def test_calibration():
    c=TemperatureCalibrator();t=c.fit(torch.tensor([[3.,1.],[1.,3.],[2.,1.],[1.,2.]]),torch.tensor([0,1,0,1]));assert t>0
    assert np.allclose(c.probabilities(torch.zeros(2,2)).numpy().sum(axis=1),1.)
def test_temporal_split_no_leak():
    df=pd.DataFrame({'text':[f'token{i} unique sentence' for i in range(40)],'label':[i%2 for i in range(40)],'timestamp':pd.date_range('2021-01-01',periods=40,freq='D')})
    parts,audit=prepare(df)
    assert max(pd.to_datetime(parts['train'].timestamp))<min(pd.to_datetime(parts['validation'].timestamp))
    assert max(pd.to_datetime(parts['calibration'].timestamp))<min(pd.to_datetime(parts['test'].timestamp))
    assert sum(len(x) for x in parts.values())==40

def test_metrics():
    p=np.array([[.9,.1],[.2,.8],[.6,.4]]);result=metrics([0,1,0],p)
    assert result['accuracy']==1
    assert len(risk_coverage([0,1,0],p))==3

def test_statistics():
    assert holm([.01,.02,.5])[0]<=.05
    r=paired_bootstrap([0,1,0,1],[0,1,0,1],[1,1,0,0],n_boot=20)
    assert r['observed_delta_macro_f1']>0

def test_fisher_penalty():
    m=THCMSSL(7,2,embedding_dim=8,hidden_dim=8,layers=1)
    mem=ContinualMemory();mem.theta={k:p.detach().clone() for k,p in m.named_parameters()};mem.fisher={k:torch.ones_like(p) for k,p in m.named_parameters()}
    assert mem.penalty(m)==0
    with torch.no_grad():m.classifier.bias.add_(1.)
    assert mem.penalty(m)>0
