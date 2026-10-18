"""Strict chronological split and leakage audit; emits canonical train/val/calibration/test CSVs."""
import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd
from thcm_ssl import tokenize

def canonical(s):return ' '.join(tokenize(str(s))).strip()
def near_signature(s,ngram=3):
    t=canonical(s).split()
    if len(t)<ngram:return {tuple(t)}
    return {tuple(t[i:i+ngram]) for i in range(len(t)-ngram+1)}
def prepare(frame,text='text',label='label',timestamp='timestamp',user='user_id',train=.7,val=.15,calibration=.5):
    if not 0<train<1 or not 0<val<1-train:raise ValueError('Invalid split proportions')
    missing=[x for x in (text,label,timestamp) if x not in frame]
    if missing:raise ValueError(f'Missing required columns: {missing}')
    df=frame.copy();df[timestamp]=pd.to_datetime(df[timestamp],utc=True,errors='coerce')
    if df[timestamp].isna().any():raise ValueError('Missing/unparseable timestamps: chronological claims unsupported')
    if df[[text,label]].isna().any().any():raise ValueError('Missing text/labels')
    df['_canon']=df[text].map(canonical)
    df=df.loc[df['_canon'].str.len()>0].sort_values(timestamp,kind='stable')
    df=df.drop_duplicates('_canon',keep='first').reset_index(drop=True)
    n=len(df);a=int(train*n);b=int((train+val)*n)
    if a<2 or b<=a or b>=n:raise ValueError('Insufficient examples for chronological partitions')
    cut1=df.iloc[a][timestamp];cut2=df.iloc[b][timestamp]
    # Strictly separate same-timestamp posts: ties are assigned to the later partition.
    tr=df[df[timestamp]<cut1].copy()
    va=df[(df[timestamp]>=cut1)&(df[timestamp]<cut2)].copy()
    te=df[df[timestamp]>=cut2].copy()
    if any(len(x)==0 for x in (tr,va,te)):raise ValueError('Cannot construct disjoint temporal intervals')
    if user in df.columns:
        # Evaluate over previously unseen users when identifiers are supplied.
        seen=set(tr[user].dropna().astype(str));va=va[~va[user].astype(str).isin(seen)]
        seen.update(va[user].dropna().astype(str));te=te[~te[user].astype(str).isin(seen)]
    # Cross-partition near-duplicate audit removes suspect later observations.
    def remove_overlap(source,target,threshold=.9):
        old=[near_signature(s) for s in source['_canon']]
        if not old:return target,0
        keep=[];removed=0
        for _,r in target.iterrows():
            cur=near_signature(r['_canon']);overlap=False
            for sig in old:
                union=len(cur|sig)
                if union and len(cur&sig)/union>=threshold:overlap=True;break
            keep.append(not overlap);removed+=int(overlap)
        return target.loc[keep].copy(),removed
    va,rmva=remove_overlap(tr,va);te,rmte=remove_overlap(pd.concat([tr,va]),te)
    va=va.sort_values(timestamp)
    vcut=max(1,int(len(va)*(1-calibration)))
    valid,cal=va.iloc[:vcut],va.iloc[vcut:]
    if any(len(x)==0 for x in (tr,valid,cal,te)):raise ValueError('Split filtering left empty partition')
    audit={'counts':{'train':len(tr),'validation':len(valid),'calibration':len(cal),'test':len(te)},'removed_near_duplicates':{'validation':rmva,'test':rmte},'boundaries':{k:[str(v[timestamp].min()),str(v[timestamp].max())] for k,v in [('train',tr),('validation',valid),('calibration',cal),('test',te)]}}
    return {k:part.drop(columns=['_canon']) for k,part in [('train',tr),('validation',valid),('calibration',cal),('test',te)]},audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',default='.');ap.add_argument('--train',type=float,default=.7);ap.add_argument('--val',type=float,default=.15);args=ap.parse_args()
    parts,audit=prepare(pd.read_csv(args.input),train=args.train,val=args.val)
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    for k,v in parts.items():v.to_csv(dest/f'{k}.csv',index=False)
    (dest/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
