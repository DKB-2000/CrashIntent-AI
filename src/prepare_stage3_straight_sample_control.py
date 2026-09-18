"""Prepare a fixed-label, matched-composition straight sampling control."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
OUT=P/'artifacts/stage3-straight-sample-control-20260914'
DATA=P/'artifacts/stage3-comma-chunk1-calibrated-v1'
REF=P/'artifacts/stage3-representation-20260910'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    samples=pd.read_csv(REF/'train-samples.csv');signals=pd.read_csv(DATA/'signals_and_candidates.csv')
    labels=pd.read_csv(DATA/'labels_train_candidate.csv');split=pd.read_csv(DATA/'split_manifest.csv')
    meta=json.loads((DATA/'label_version.json').read_text())
    train=signals[signals.split=='train'].copy()
    train=train.drop(columns=['steer_label']).merge(labels[['ID','frame_index','steer_label']],on=['ID','frame_index'],validate='one_to_one')
    train['angle']=train.steering_angle_deg-meta['steer_offset_deg']
    train['speed_bin']=pd.cut(train.speed_mps,[-np.inf,5,15,25,np.inf],right=False,labels=['under5','5to15','15to25','25plus']).astype(str)
    train['stable11']=False
    for sid,g in train.groupby('ID'):
        g=g.sort_values('frame_index')
        mask=(g.steer_label=='STRAIGHT').astype(int).rolling(11,center=True,min_periods=11).sum().eq(11)
        train.loc[g.index,'stable11']=mask.to_numpy()
    train['clear_straight']=(train.steer_label=='STRAIGHT')&(train.speed_mps>=5)&(train.pose_yaw_left_rad_s.abs()<.003)&train.stable11&(np.abs(np.abs(train.angle)-meta['steer_deadzone_deg'])>.25)&(train.frame_index>=15)
    train['accel']=train.accel_label.map({'ACCELERATING':0,'DECELERATING':1,'CONSTANT':2,'STOPPED':3})
    baseline=samples.merge(train[['ID','frame_index','speed_bin','clear_straight']],left_on=['ID','endpoint'],right_on=['ID','frame_index'],validate='one_to_one')
    baseline['slot']=np.arange(len(baseline));baseline=baseline.drop(columns=['record','frame_index'])
    candidate=baseline.copy();candidate['replaced']=False
    # Same route x accel group x speed bin; replace at most half per cell.
    rng=np.random.default_rng(20260917);used=set(zip(baseline.ID,baseline.endpoint));availability=[]
    for (route,accel,speed_bin),g in baseline[(baseline.steer==1)&(baseline.accel!=3)].groupby(['route','accel','speed_bin'],sort=True):
        pool=train[(train.route==route)&(train.accel==accel)&(train.speed_bin==speed_bin)&train.clear_straight]
        pool=pool[[ (r.ID,int(r.frame_index)) not in used for r in pool.itertuples()]] if len(pool) else pool
        old=g[~g.clear_straight]
        n=min(len(old),len(pool),len(g)//2)
        availability.append(dict(route=route,accel=int(accel),speed_bin=speed_bin,baseline=len(g),baseline_clear=int(g.clear_straight.sum()),available_unused_clear=len(pool),replacements=n))
        if n:
            oldids=rng.choice(old.index.to_numpy(),n,replace=False);newids=rng.choice(pool.index.to_numpy(),n,replace=False)
            for idx,j in zip(oldids,newids):
                row=pool.loc[j];used.add((row.ID,int(row.frame_index)))
                candidate.loc[idx,'ID']=row.ID;candidate.loc[idx,'endpoint']=int(row.frame_index)
                candidate.loc[idx,'video']=split.set_index('ID').loc[row.ID,'video']
                candidate.loc[idx,'clear_straight']=True;candidate.loc[idx,'replaced']=True
    keys=['route','accel','steer','group','speed_bin']
    pd.testing.assert_series_equal(baseline.groupby(keys).size(),candidate.groupby(keys).size())
    assert len(candidate)==1000 and not candidate[['ID','endpoint']].duplicated().any()
    assert not candidate.loc[candidate.replaced,'accel'].eq(3).any() and candidate.loc[candidate.replaced,'steer'].eq(1).all()
    check=candidate.merge(train[['ID','frame_index','steer_label','accel','route','speed_bin','clear_straight']],left_on=['ID','endpoint'],right_on=['ID','frame_index'],validate='one_to_one',suffixes=('','_check'))
    for key in ['accel','route','speed_bin','clear_straight']:assert np.array_equal(check[key],check[key+'_check'])
    assert np.array_equal(check.steer_label,np.array(['LEFT','STRAIGHT','RIGHT'])[check.steer])
    baseline.to_csv(OUT/'baseline-samples.csv',index=False);candidate.to_csv(OUT/'candidate-samples.csv',index=False)
    pd.DataFrame(availability).to_csv(OUT/'availability.csv',index=False)
    report=dict(status='MATCHED_SAMPLE_PLAN_READY_NO_TRAINING',total=1000,replaced=int(candidate.replaced.sum()),
        straight_clear_before=int(baseline[(baseline.steer==1)&(baseline.accel!=3)].clear_straight.sum()),
        straight_clear_after=int(candidate[(candidate.steer==1)&(candidate.accel!=3)].clear_straight.sum()),
        replacement_policy='At most floor(n/2) of each route x accel x speed-bin straight cell; only non-clear to unused clear; seed20260917. Fixed original label, no relabeling.',
        clear_definition='Straight fixed label, speed>=5,abs(pose yaw)<.003,11same straight labels,angle distance to threshold>.25deg,endpoint>=15.',
        matched='Exact1000samples,10group counts,route x accel x steer x speed-bin counts; all stopped/turning samples and existing clear samples preserved.',
        proposed_training='Paired fixed3seeds/1200updates/same10group balanced batches; existing architecture/preprocessing/labels. No training yet.',
        proposed_validation='Existing4development routes, entire moving set and per-route/per-class, stable vs transitions; no new-Civic tuning. Candidate only if3seedF1 positive,>=3/4route means positive,no class recall loss>2pp; report transition misses/delays too.',
        caveat='Selection based on sensor stability may reduce transition diversity; speed bin matched but not exact speed or video identity. Diagnostic development comparison, not guaranteed improvement.',
        input_sha256={str(p.relative_to(P)):sha(p) for p in [Path(__file__),REF/'train-samples.csv',DATA/'signals_and_candidates.csv',DATA/'labels_train_candidate.csv',DATA/'label_version.json',DATA/'split_manifest.csv']},
        output_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
