"""Low-speed coverage and straight-error diagnostics without changing labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
L=P/'artifacts/stage3-civic-holdout-labels-20260914'
S=P/'artifacts/stage3-civic-holdout-score-20260914'
A=P/'artifacts/stage3-civic-holdout-acquisition-20260914'
OUT=P/'artifacts/stage3-low-speed-diagnostic-20260914'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    lr=json.loads((L/'report.json').read_text());sr=json.loads((S/'report.json').read_text());ar=json.loads((A/'validation.json').read_text())
    assert sha(L/'evaluation_partial_labels.csv')==lr['outputs_sha256']['evaluation_partial_labels.csv']
    assert sha(S/'predictions.csv')==sr['prediction_sha256']
    f=pd.read_csv(S/'predictions.csv')
    assert len(f)==5310
    f['interior_can']=False;f['confidence']=0.;f['horizon_flow']=0.
    episodes=[]
    for sid,g in f.groupby('ID'):
        root=P/'data_raw/comma2k19/civic-holdout-20260914'/sid
        ft=np.load(root/'global_pose/frame_times',allow_pickle=False).reshape(-1)
        v=next(s for s in ar['segments'] if s['ID']==sid)
        lo=max(s['start'] for s in v['sensors'].values());hi=min(s['end'] for s in v['sensors'].values())
        f.loc[g.index,'interior_can']=(g.timestamp>=max(lo,ft[0]+1))&(g.timestamp<=min(hi,ft[-1]-1))
        path=S/(sid+'.npz');pv=next(s for s in sr['provenance'] if s['ID']==sid);assert sha(path)==pv['cache_sha256']
        with np.load(path) as z:
            x=z['steer'];prob=np.exp(x-x.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
            np.testing.assert_array_equal(np.array(['LEFT','STRAIGHT','RIGHT'])[x.argmax(1)],g.prediction)
            f.loc[g.index,'confidence']=prob.max(1)
            f.loc[g.index,'horizon_flow']=z['features'][:,130]
        error=(g.partial_valid_steer&(g.partial_steer_label=='STRAIGHT')&(g.prediction!='STRAIGHT')).to_numpy()
        ids=np.flatnonzero(error)
        for run in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
            if len(run):episodes.append(dict(ID=sid,start=int(g.iloc[run[0]].sample_index),end=int(g.iloc[run[-1]].sample_index),errors=len(run),speed_mean=float(g.iloc[run].speed_mps.mean())))
    bins=[]
    for name,lo,hi in [('stopped',0,.5),('creeping',.5,2),('slow',2,5),('moving',5,35.00001)]:
        g=f[(f.speed_mps>=lo)&(f.speed_mps<hi)];valid=g[g.interior_can]
        strong=valid[valid.pose_yaw_left_rad_s.abs()>.012]
        straight=valid[valid.pose_yaw_left_rad_s.abs()<.003]
        angle=strong.steering_corrected_deg.to_numpy();yaw=strong.pose_yaw_left_rad_s.to_numpy()
        bins.append(dict(bin=name,rows=len(g),interior_rows=len(valid),strong_pose_rows=len(strong),
             steer_pose_direction_agreement=float(np.mean(np.sign(angle)==np.sign(yaw))) if len(strong) else None,
             near_straight_pose_rows=len(straight),near_straight_large_steer_rows=int((straight.steering_corrected_deg.abs()>1.5).sum()),
             predictions=g.prediction.value_counts().to_dict()))
    straight=f[f.partial_valid_steer&(f.partial_steer_label=='STRAIGHT')].copy()
    straight['error']=straight.prediction!='STRAIGHT'
    comparisons=[]
    for error,g in straight.groupby('error'):
        comparisons.append(dict(error=bool(error),rows=len(g),confidence_median=float(g.confidence.median()),
             confidence_ge_0p9=int((g.confidence>=.9).sum()),speed_median=float(g.speed_mps.median()),flow_p90_median=float(g.horizon_flow.median())))
    assert sum(e['errors'] for e in episodes)==52
    straight.to_csv(OUT/'straight_samples.csv',index=False)
    pd.DataFrame(episodes).to_csv(OUT/'straight_error_episodes.csv',index=False)
    f[f.speed_mps<5].to_csv(OUT/'low_speed_samples.csv',index=False)
    result=dict(status='COMPLETE_VALIDATED',speed_bins=bins,straight_comparisons=comparisons,straight_error_episodes=episodes,
         scope='Sensor diagnostic only. Pose yaw from horizontal velocity is unreliable near standstill; direction agreement is not true accuracy. No new labels, rule fitting or model changes.',
         inputs_sha256={str(p.relative_to(P)):sha(p) for p in [L/'evaluation_partial_labels.csv',S/'predictions.csv',Path(__file__)]},
         outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
