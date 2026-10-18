"""Metric computation from actual per-instance predictions; no hardcoded paper results."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score,brier_score_loss

def expected_calibration_error(y,probs,bins=10):
    y=np.asarray(y);probs=np.asarray(probs);conf=probs.max(1);pred=probs.argmax(1)
    ece=0.
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        sel=(conf>=lo)&((conf<=hi) if hi==1 else (conf<hi))
        if sel.any():ece+=sel.mean()*abs((pred[sel]==y[sel]).mean()-conf[sel].mean())
    return float(ece)
def risk_coverage(y,probs):
    y=np.asarray(y);p=np.asarray(probs);confidence=p.max(1);correct=p.argmax(1)==y
    order=np.argsort(-confidence,kind='stable');ranks=np.arange(1,len(y)+1)
    return pd.DataFrame({'coverage':ranks/len(y),'selective_risk':1-np.cumsum(correct[order])/ranks,'threshold':confidence[order]})
def metrics(y,probs):
    y=np.asarray(y).astype(int);p=np.asarray(probs);pred=p.argmax(1);k=p.shape[1]
    one_hot=np.eye(k)[y]
    curve=risk_coverage(y,p)
    ix=np.argmin(abs(curve.coverage-.9))
    return {'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_precision':float(precision_score(y,pred,average='macro',zero_division=0)),'macro_recall':float(recall_score(y,pred,average='macro',zero_division=0)),'ece_10bin':expected_calibration_error(y,p),'brier_multiclass':float(np.mean(np.sum((p-one_hot)**2,axis=1))),'selective_risk_90_coverage':float(curve.iloc[ix].selective_risk),'aurc':float(np.trapz(curve.selective_risk,curve.coverage))}
def read_predictions(path):
    df=pd.read_csv(path);cols=sorted([x for x in df if x.startswith('p_')])
    if not cols:raise ValueError('Missing probability columns (p_*)')
    return df.target.to_numpy(),df[cols].to_numpy()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',default='outputs/predictions.csv');ap.add_argument('--output',default='outputs/metrics.json');args=ap.parse_args()
    y,p=read_predictions(args.predictions);result=metrics(y,p)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    risk_coverage(y,p).to_csv(out.with_name('risk_coverage.csv'),index=False)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
