"""Paired test-instance bootstrap, seed summaries and Holm stepdown control."""
import argparse,json
import numpy as np,pandas as pd
from scipy.stats import t
from sklearn.metrics import f1_score

def paired_bootstrap(y,pred_a,pred_b,n_boot=2000,seed=11,clusters=None):
    y=np.asarray(y);a=np.asarray(pred_a);b=np.asarray(pred_b)
    if len(y)!=len(a) or len(y)!=len(b):raise ValueError('Paired predictions must align')
    rng=np.random.default_rng(seed)
    if clusters is None:clusters=np.arange(len(y))
    clusters=np.asarray(clusters);groups=np.unique(clusters)
    positions={g:np.flatnonzero(clusters==g) for g in groups}
    observed=f1_score(y,a,average='macro',zero_division=0)-f1_score(y,b,average='macro',zero_division=0)
    diffs=[]
    for _ in range(n_boot):
        selected=rng.choice(groups,size=len(groups),replace=True)
        idx=np.concatenate([positions[g] for g in selected])
        diffs.append(f1_score(y[idx],a[idx],average='macro',zero_division=0)-f1_score(y[idx],b[idx],average='macro',zero_division=0))
    ci=np.quantile(diffs,[.025,.975]);p=min(1.,2*min(np.mean(np.asarray(diffs)<=0),np.mean(np.asarray(diffs)>=0)))
    return {'observed_delta_macro_f1':float(observed),'bootstrap_ci_95':[float(ci[0]),float(ci[1])],'bootstrap_p_approx':float(p),'bootstrap_replicates':n_boot,'resampling_units':len(groups)}
def seed_summary(values):
    x=np.asarray(values,dtype=float);n=len(x)
    if n<2:raise ValueError('At least two seeds required')
    sem=x.std(ddof=1)/np.sqrt(n);m=x.mean();margin=t.ppf(.975,n-1)*sem
    return {'mean':float(m),'sd':float(x.std(ddof=1)),'ci95':[float(m-margin),float(m+margin)],'n_seeds':n,'note':'This interval measures seed variability, not test-instance uncertainty.'}
def holm(pvalues):
    vals=np.array(pvalues,dtype=float);m=len(vals);order=np.argsort(vals);adj=np.zeros(m);highest=0.
    for rank,i in enumerate(order):highest=max(highest,(m-rank)*vals[i]);adj[i]=min(1.,highest)
    return adj.tolist()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paired-a');ap.add_argument('--paired-b');ap.add_argument('--bootstrap',type=int,default=2000);args=ap.parse_args()
    if not args.paired_a or not args.paired_b:
        print('Supply --paired-a and --paired-b with aligned per-instance prediction CSVs');return
    a=pd.read_csv(args.paired_a);b=pd.read_csv(args.paired_b)
    if not np.array_equal(a.target,b.target):raise ValueError('Target labels or row order differ')
    print(json.dumps(paired_bootstrap(a.target,a.prediction,b.prediction,args.bootstrap),indent=2))
if __name__=='__main__':main()
