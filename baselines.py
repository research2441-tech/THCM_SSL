"""Runnable classical baseline and controlled neural architectural variants."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from thcm_ssl import THCMSSL,TemporalHypergraph

def fisher_tfidf_svm(train_texts,train_labels,test_texts):
    model=make_pipeline(TfidfVectorizer(token_pattern=r'(?u)\b\w+\b|#[\w]+',ngram_range=(1,2),min_df=1),LinearSVC())
    model.fit(train_texts,train_labels)
    return model.predict(test_texts)
def make_variant(name,vocab_size,classes,embedding_dim=300,hidden_dim=256,layers=4):
    options={'full':(True,True,'ssm'),'no_hypergraph':(True,False,'ssm'),'no_gate':(False,True,'ssm'),'bilstm':(True,True,'bilstm'),'transformer':(True,True,'transformer'),'ssm_only':(False,False,'ssm')}
    if name not in options:raise ValueError(f'Unknown variant {name}; choices: {list(options)}')
    gated,hypergraph,encoder=options[name]
    return THCMSSL(vocab_size,classes,embedding_dim,hidden_dim,layers,use_gate=gated,use_hypergraph=hypergraph,encoder=encoder)
def make_hypergraph_variant(name,**kwargs):
    variations={'temporal_hypergraph':{},'static_hypergraph':{'decay':False},'pairwise_graph':{'pairwise':True},'no_class_weight':{'class_weight':False},'no_decay':{'decay':False}}
    if name not in variations:raise ValueError(name)
    return TemporalHypergraph(**{**kwargs,**variations[name]})
