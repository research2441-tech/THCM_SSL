"""Reproduction driver. Pass original labeled, timestamped corpus CSV."""
import argparse,subprocess,sys
from pathlib import Path

def command(*args):
    print('RUN:',sys.executable,*args,flush=True)
    subprocess.run([sys.executable,*map(str,args)],check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--config',default='config.yaml');p.add_argument('--out',default='outputs');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    command('prepare_data.py','--input',a.data,'--out',out)
    command('train.py','--config',a.config,'--input-dir',out,'--output-dir',out,'--epochs',2 if a.smoke else 40)
    command('evaluate.py','--predictions',out/'predictions.csv','--output',out/'metrics.json')
    command('reviewer_experiments.py','--train',out/'train.csv','--test',out/'test.csv','--config',a.config,'--out',out)
    command('visualize.py','--output-dir',out)
    print('Finished. Outputs are generated from supplied data, not the values printed in the paper.')
if __name__=='__main__':main()
