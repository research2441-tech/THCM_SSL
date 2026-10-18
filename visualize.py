"""Figures generated only from measured saved outputs."""
import argparse,json
from pathlib import Path
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def figures(out='outputs'):
    out=Path(out);result=[]
    rc=out/'risk_coverage.csv'
    if rc.exists():
        frame=pd.read_csv(rc);fig,ax=plt.subplots();ax.plot(frame.coverage,frame.selective_risk);ax.set(xlabel='Coverage',ylabel='Selective risk',title='Risk–coverage curve');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'risk_coverage.png',dpi=200);plt.close(fig);result.append('risk_coverage.png')
    hist=out/'history.json'
    if hist.exists():
        frame=pd.DataFrame(json.loads(hist.read_text()));fig,ax=plt.subplots();ax.plot(frame.epoch,frame.validation_macro_f1);ax.set(xlabel='Epoch',ylabel='Validation macro-F1',title='Training convergence');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'convergence.png',dpi=200);plt.close(fig);result.append('convergence.png')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default='outputs');a=p.parse_args();print(figures(a.output_dir))
