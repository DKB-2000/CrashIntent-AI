"""Describe lost straight predictions and removed training samples; no training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from diagnose_stage3_flow_regions import regions, sha

P=Path(__file__).resolve().parents[1]
REF=P/'artifacts/stage3-representation-20260910'
EXP=P/'artifacts/stage3-straight-sample-control-20260914'
DATA=P/'artifacts/stage3-comma-chunk1-calibrated-v1'
OUT=P/'artifacts/stage3-sample-loss-diagnostic-20260914'

def main():
    OUT.mkdir(exist_ok=False);hashes={}
    def read(path):
        hashes[str(path.relative_to(P))]=sha(path)
        return pd.read_csv(path)
    final=json.loads((EXP/'final-review.json').read_text());assert final['status']=='COMPLETE_VALIDATED'
    valid=read(REF/'validation-samples.csv');old=read(EXP/'baseline-samples.csv');new=read(EXP/'candidate-samples.csv')
    signals=read(DATA/'signals_and_candidates.csv')
    meta=json.loads((DATA/'label_version.json').read_text())
    signals['angle']=signals.steering_angle_deg-meta['steer_offset_deg']
    signals['fixed']=np.where(signals.angle>meta['steer_deadzone_deg'],0,np.where(signals.angle< -meta['steer_deadzone_deg'],2,1))
    # Use full fixed-label sequences; never infer event distance from sampled clips.
    signals['since_turn_s']=np.nan;signals['until_turn_s']=np.nan;signals['stable11']=False
    for sid,g in signals.groupby('ID'):
        g=g.sort_values('frame_index');frame=g.frame_index.to_numpy();turn=g.fixed.to_numpy()!=1
        last=np.maximum.accumulate(np.where(turn,frame,-100000))
        nxt=np.minimum.accumulate(np.where(turn,frame,100000)[::-1])[::-1]
        signals.loc[g.index,'since_turn_s']=(frame-last)/10
        signals.loc[g.index,'until_turn_s']=(nxt-frame)/10
        signals.loc[g.index,'stable11']=(g.fixed==1).astype(int).rolling(11,center=True,min_periods=11).sum().eq(11).to_numpy()
    def enrich(frame):
        cols=['ID','frame_index','speed_mps','angle','fixed','pose_yaw_left_rad_s','since_turn_s','until_turn_s','stable11']
        f=frame.merge(signals[cols],left_on=['ID','endpoint'],right_on=['ID','frame_index'],validate='one_to_one')
        assert len(f)==len(frame) and np.array_equal(f.steer,f.fixed)
        f['speed_bin']=pd.cut(f.speed_mps,[-np.inf,5,15,25,np.inf],right=False,labels=['under5','5to15','15to25','25plus']).astype(str)
        f['pose_bin']=np.where(f.speed_mps<5,'low_speed',np.where(f.pose_yaw_left_rad_s.abs()<.003,'near_straight',np.where(f.pose_yaw_left_rad_s.abs()>.012,'strong_turn','intermediate')))
        f['near_turn_1s']=(f.since_turn_s<=1)|(f.until_turn_s<=1)
        f['after_turn_1s']=f.since_turn_s<=1
        f['near_angle_threshold']=np.abs(np.abs(f.angle)-meta['steer_deadzone_deg'])<=.25
        rows=[]
        for sid,g in f.groupby('ID'):
            path=REF/'cache'/(sid+'.npz');m=json.loads(path.with_suffix('.json').read_text())
            assert sha(path)==m['features_sha256'];hashes[str(path.relative_to(P))]=sha(path)
            with np.load(path) as z:lookup={int(e):i for i,e in enumerate(z['endpoints'])};x=z['features'][:,1280:]
            for idx,r in g.iterrows():rows.append(dict(index=idx,flow_p90=float(x[lookup[int(r.endpoint)],130]),**regions(x[lookup[int(r.endpoint)]])))
        return f.join(pd.DataFrame(rows).set_index('index'))
    f=enrich(valid);f['loss_votes']=0;f['gain_votes']=0;f['old_correct_votes']=0;f['new_correct_votes']=0
    perseed=[]
    for seed in [20260910,20260911,20260912]:
        preds=[]
        for arm in ['baseline','candidate']:
            path=EXP/arm/f'{seed}-motion/validation-predictions.csv'
            assert sha(path)==final['hashes'][str(path.relative_to(EXP))]
            pred=read(path);assert np.array_equal(pred.ID,f.ID) and np.array_equal(pred.sample_index,f.endpoint)
            preds.append(pred.steer_label.eq('STRAIGHT').to_numpy())
        a,b=preds;f['loss_votes']+=(a&~b);f['gain_votes']+=(~a&b);f['old_correct_votes']+=a;f['new_correct_votes']+=b
        use=(f.steer==1)&(f.accel!=3)
        perseed.append(dict(seed=seed,lost=int((a&~b&use).sum()),gained=int((~a&b&use).sum())))
    straight=f[(f.steer==1)&(f.accel!=3)].copy()
    straight['cohort']=np.where(straight.loss_votes>=2,'lost_2plus',np.where(straight.gain_votes>=2,'gained_2plus',np.where((straight.old_correct_votes==3)&(straight.new_correct_votes==3),'correct_all6','other')))
    train=enrich(old.drop(columns=['speed_bin']))
    train=train[(train.steer==1)&(train.accel!=3)].copy()
    changed=(old.ID!=new.ID)|(old.endpoint!=new.endpoint)
    train['cohort']=np.where(train.slot.isin(old.loc[changed,'slot']),'removed84','retained216')
    summaries=[]
    def describe(dataset,g,name):
        return dict(dataset=dataset,cohort=name,rows=len(g),videos=int(g.ID.nunique()),
            speed_counts=g.speed_bin.value_counts().to_dict(),pose_counts=g.pose_bin.value_counts().to_dict(),
            rates={c:float(g[c].mean()) for c in ['near_turn_1s','after_turn_1s','near_angle_threshold','stable11']},
            medians={c:float(g[c].median()) for c in ['speed_mps','flow_p90','side_magnitude','center_magnitude','side_center_ratio']})
    for name,g in straight.groupby('cohort'):summaries.append(describe('validation',g,name))
    for name,g in train.groupby('cohort'):summaries.append(describe('training',g,name))
    stratified=[]
    for column in ['route','speed_bin','pose_bin','near_turn_1s','after_turn_1s','near_angle_threshold']:
        for name,g in straight.groupby(column):
            at_risk=int(g.old_correct_votes.sum())
            stratified.append(dict(field=column,value=str(name),frames=len(g),old_correct_predictions=at_risk,lost=int(g.loss_votes.sum()),gained=int(g.gain_votes.sum()),loss_fraction_of_old_correct=float(g.loss_votes.sum()/at_risk) if at_risk else None))
    runs=[]
    for sid,g in straight.groupby('ID'):
        g=g.sort_values('endpoint');bad=g[g.loss_votes>=2];groups=(bad.endpoint.diff()!=1).cumsum()
        for _,r in bad.groupby(groups):runs.append(dict(ID=sid,route=r.route.iloc[0],start=int(r.endpoint.min()),end=int(r.endpoint.max()),frames=len(r),mean_loss_votes=float(r.loss_votes.mean()),median_speed=float(r.speed_mps.median()),near_turn_fraction=float(r.near_turn_1s.mean()),pose_bin=r.pose_bin.mode().iloc[0]))
    straight.to_csv(OUT/'validation-straight.csv',index=False);train.to_csv(OUT/'training-straight.csv',index=False)
    pd.DataFrame(stratified).to_csv(OUT/'stratified-losses.csv',index=False)
    pd.DataFrame(runs).sort_values(['frames','ID'],ascending=[False,True]).to_csv(OUT/'loss-runs.csv',index=False)
    assert int(straight.loss_votes.sum())==sum(r['lost'] for r in perseed)
    assert int(straight.gain_votes.sum())==sum(r['gained'] for r in perseed)
    assert train.cohort.value_counts().to_dict()=={'retained216':216,'removed84':84}
    for path,h in hashes.items():assert sha(P/path)==h
    report=dict(status='COMPLETE_VALIDATED',per_seed=perseed,summaries=summaries,loss_runs=len(runs),
        interpretation_limits='Descriptive association, correlated frames. Flow regions are not object identification. Angle and pose categories do not establish road geometry. Weight training and normalization both change; cannot attribute loss solely to removed samples.',
        inputs_sha256=hashes,outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','per_seed','summaries','loss_runs']},indent=2))

if __name__=='__main__':main()
