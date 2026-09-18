"""Read-only regional flow comparison; grid cells are central-crop coordinates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
OUT=P/'artifacts/stage3-flow-regions-20260914'
DEV=P/'artifacts/stage3-representation-20260910'
CIVIC=P/'artifacts/stage3-civic-holdout-score-20260914'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def regions(feature):
    grid=feature[:128].reshape(8,8,2)
    magnitude=np.linalg.norm(grid,axis=2)
    left=magnitude[:,:2].mean();right=magnitude[:,6:].mean();center=magnitude[:,2:6].mean()
    return dict(side_magnitude=float((left+right)/2),center_magnitude=float(center),
                side_center_ratio=float((left+right)/2/max(center,1e-6)),
                left_magnitude=float(left),right_magnitude=float(right),
                side_asymmetry=float((right-left)/max(right+left,1e-6)),
                mean_horizontal=float(grid[:,:,0].mean()),mean_vertical=float(grid[:,:,1].mean()),
                lower_magnitude=float(magnitude[4:].mean()),upper_magnitude=float(magnitude[:4].mean()))


def main():
    OUT.mkdir(exist_ok=False)
    inputs={};rows=[]
    cp=P/'artifacts/stage3-low-speed-diagnostic-20260914/straight_samples.csv'
    cr=json.loads((cp.parent/'report.json').read_text());assert sha(cp)==cr['outputs_sha256'][cp.name]
    civic=pd.read_csv(cp)
    score=json.loads((CIVIC/'report.json').read_text())
    for sid,g in civic.groupby('ID'):
        path=CIVIC/(sid+'.npz');expected=next(r['cache_sha256'] for r in score['provenance'] if r['ID']==sid)
        assert sha(path)==expected;inputs[str(path.relative_to(P))]=expected
        with np.load(path) as z:x=z['features']
        for r in g.itertuples():rows.append(dict(dataset='civic',ID=sid,sample_index=r.sample_index,error=bool(r.error),prediction=r.prediction,speed_mps=r.speed_mps,**regions(x[r.sample_index])))
    f=pd.read_csv(DEV/'validation-samples.csv')
    pred=pd.read_csv(DEV/'20260910-motion/validation-predictions.csv')
    assert np.array_equal(f.ID,pred.ID) and np.array_equal(f.endpoint,pred.sample_index)
    f['prediction']=pred.steer_label
    signals=P/'artifacts/stage3-comma-chunk1-calibrated-v1/signals_and_candidates.csv'
    speed=pd.read_csv(signals,usecols=['ID','frame_index','speed_mps'])
    f=f.merge(speed,left_on=['ID','endpoint'],right_on=['ID','frame_index'],validate='one_to_one')
    f=f[(f.accel!=3)&(f.steer_label=='STRAIGHT')&(f.speed_mps>=10)&(f.speed_mps<=25)]
    for sid,g in f.groupby('ID'):
        path=DEV/'cache'/(sid+'.npz');metadata=json.loads(path.with_suffix('.json').read_text());assert sha(path)==metadata['features_sha256']
        inputs[str(path.relative_to(P))]=sha(path)
        with np.load(path) as z:
            lookup={int(v):i for i,v in enumerate(z['endpoints'])};x=z['features'][:,1280:]
        for r in g.itertuples():rows.append(dict(dataset='development',ID=sid,sample_index=r.endpoint,error=r.prediction!='STRAIGHT',prediction=r.prediction,speed_mps=r.speed_mps,**regions(x[lookup[r.endpoint]])))
    data=pd.DataFrame(rows);data=data[(data.speed_mps>=10)&(data.speed_mps<=25)]
    assert len(data[(data.dataset=='civic')&data.error])==52
    summaries=[]
    fields=['speed_mps','side_magnitude','center_magnitude','side_center_ratio','side_asymmetry','mean_horizontal','lower_magnitude','upper_magnitude']
    for (dataset,error),g in data.groupby(['dataset','error']):
        summaries.append(dict(dataset=dataset,error=bool(error),rows=len(g),videos=int(g.ID.nunique()),
                              medians={c:float(g[c].median()) for c in fields}))
    paired=[]
    for (dataset,sid),g in data.groupby(['dataset','ID']):
        if g.error.nunique()<2:continue
        good=g[~g.error];bad=g[g.error]
        paired.append(dict(dataset=dataset,ID=sid,correct=len(good),errors=len(bad),
            side_difference=float(bad.side_magnitude.median()-good.side_magnitude.median()),
            center_difference=float(bad.center_magnitude.median()-good.center_magnitude.median()),
            ratio_difference=float(bad.side_center_ratio.median()-good.side_center_ratio.median())))
    data.to_csv(OUT/'samples.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'within_video.csv',index=False)
    inputs.update({str(p.relative_to(P)):sha(p) for p in [cp,DEV/'validation-samples.csv',DEV/'20260910-motion/validation-predictions.csv',signals,Path(__file__)]})
    result=dict(status='COMPLETE_VALIDATED',summaries=summaries,within_video=paired,
         definition='Last flow8x8 central crop grid; sides columns0:2 and6:8, center2:6; means of grid-vector magnitude. Ratios capped denominator1e-6.',
         limits='Different sensor label selection and vehicle/route; speed10-25m/s restriction is coarse matching only. No object tracking, depth, camera-motion separation, pixel intervention or retraining. Frame medians are correlated observations.',
         inputs_sha256=inputs,outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    for path,h in inputs.items():assert sha(P/path)==h
    (OUT/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','summaries','within_video']},indent=2))

if __name__=='__main__':main()
