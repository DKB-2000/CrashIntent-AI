"""Describe right-turn losses from the frozen transition experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_stage3_straight_sample_control import sha

P=Path(__file__).resolve().parents[1]
REF=P/'artifacts/stage3-representation-20260910'
EXP=P/'artifacts/stage3-transition-matched-control-20260914'
OUT=P/'artifacts/stage3-transition-right-losses-20260914'

def main():
    OUT.mkdir(exist_ok=False)
    report=json.loads((EXP/'final-review.json').read_text());assert report['status']=='COMPLETE_VALIDATED'
    f=pd.read_csv(REF/'validation-samples.csv');rows=[]
    f['stable_right11']=False
    for sid,g in f.groupby('ID'):
        assert np.all(np.diff(g.endpoint)==1)
        f.loc[g.index,'stable_right11']=((g.steer==2)&(g.accel!=3)).astype(int).rolling(11,center=True,min_periods=11).sum().eq(11).to_numpy()
    summaries=[]
    for seed in [20260910,20260911,20260912]:
        predictions=[]
        for folder in [REF/f'{seed}-motion',EXP/'add_transition'/str(seed)]:
            path=folder/'validation-logits.npz';assert sha(path)==report['output_sha256'][str(path.relative_to(P))]
            with np.load(path) as z:predictions.append(z['steer'].argmax(1))
        a,b=predictions;right=(f.steer==2)&(f.accel!=3)
        g=f[right].copy();g['seed']=seed;g['original_prediction']=a[right];g['candidate_prediction']=b[right]
        g['lost']=(g.original_prediction==2)&(g.candidate_prediction!=2);g['gained']=(g.original_prediction!=2)&(g.candidate_prediction==2)
        rows.append(g)
        for dimension in ['ALL','route','stable_right11']:
            groups=[('ALL',g)] if dimension=='ALL' else g.groupby(dimension)
            for name,h in groups:
                lost=h[h.lost]
                summaries.append(dict(seed=seed,dimension=dimension,value=str(name),rows=len(h),original_correct=int((h.original_prediction==2).sum()),lost=len(lost),gained=int(h.gained.sum()),lost_to_straight=int((lost.candidate_prediction==1).sum()),lost_to_left=int((lost.candidate_prediction==0).sum())))
    pd.concat(rows,ignore_index=True).to_csv(OUT/'right-predictions.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    result=dict(status='COMPLETE_VALIDATED',summary=summaries,scope='Reused development sensor right labels; stability is label persistence, not human or road geometry.',inputs_sha256={str(p.relative_to(P)):sha(p) for p in [Path(__file__),REF/'validation-samples.csv',EXP/'final-review.json']})
    (OUT/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).query("dimension != 'route'").to_string(index=False))

if __name__=='__main__':main()
