"""Audit selected training labels and route/speed coverage; no mutation/training."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
REF=P/'artifacts/stage3-representation-20260910'
DATA=P/'artifacts/stage3-comma-chunk1-calibrated-v1'
OUT=P/'artifacts/stage3-training-label-audit-20260914'
CLASSES=['LEFT','STRAIGHT','RIGHT']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    paths=[REF/'train-samples.csv',DATA/'signals_and_candidates.csv',DATA/'labels_train_candidate.csv',DATA/'label_version.json']
    hashes={str(p.relative_to(P)):sha(p) for p in paths}
    samples=pd.read_csv(paths[0]);signals=pd.read_csv(paths[1]);labels=pd.read_csv(paths[2]);meta=json.loads(paths[3].read_text())
    assert len(samples)==1000 and not samples[['ID','endpoint']].duplicated().any()
    corrected=signals.steering_angle_deg-meta['steer_offset_deg'];width=meta['steer_deadzone_deg']
    signals['fixed_label']=np.where(corrected>width,'LEFT',np.where(corrected< -width,'RIGHT','STRAIGHT'))
    signals['margin_deg']=np.abs(np.abs(corrected)-width)
    train=signals[signals.split=='train'].copy()
    check=train.merge(labels,on=['ID','frame_index'],validate='one_to_one',suffixes=('_signal','_label'))
    assert len(check)==len(train)==len(labels) and np.array_equal(check.fixed_label,check.steer_label_label)
    selected=samples.merge(train,left_on=['ID','endpoint'],right_on=['ID','frame_index'],validate='one_to_one',suffixes=('','_signal'))
    assert len(selected)==1000 and (selected.route==selected.route_signal).all()
    assert np.array_equal(selected.fixed_label,np.array(CLASSES)[selected.steer])
    selected['moving']=selected.accel!=3
    selected['pose_category']=np.where(selected.speed_mps<5,'LOW_SPEED_UNRELIABLE',np.where(selected.pose_yaw_left_rad_s.abs()<.003,'NEAR_STRAIGHT',np.where(selected.pose_yaw_left_rad_s.abs()>.012,np.where(selected.pose_yaw_left_rad_s>0,'STRONG_LEFT','STRONG_RIGHT'),'INTERMEDIATE')))
    # Label persistence around selected endpoint is diagnostic, not automatic rejection.
    indexed={sid:g.sort_values('frame_index').set_index('frame_index') for sid,g in train.groupby('ID')}
    persistence=[]
    for r in selected.itertuples():
        g=indexed[r.ID];window=g.loc[max(0,r.endpoint-5):r.endpoint+5]
        persistence.append(len(window)==11 and (window.fixed_label==r.fixed_label).all())
    selected['stable_11']=persistence
    selected.to_csv(OUT/'selected-sample-audit.csv',index=False)
    moving=selected[selected.moving];straight=moving[moving.fixed_label=='STRAIGHT']
    coverage=[]
    for route,g in train.groupby('route'):
        eligible=g[(g.accel_label!='STOPPED')&(g.fixed_label=='STRAIGHT')]
        chosen=straight[straight.route==route]
        coverage.append(dict(route=route,available_straight=len(eligible),selected_straight=len(chosen),
            selected_near_straight=int((chosen.pose_category=='NEAR_STRAIGHT').sum()),selected_low_speed=int((chosen.speed_mps<5).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'straight-route-coverage.csv',index=False)
    summaries=[]
    for dataset,f in [('selected',moving),('available',train[train.accel_label!='STOPPED'])]:
        for name,lo,hi in [('under5',0,5),('5to15',5,15),('15to25',15,25),('25plus',25,float('inf'))]:
            g=f[(f.speed_mps>=lo)&(f.speed_mps<hi)]
            summaries.append(dict(dataset=dataset,speed_bin=name,counts=g.fixed_label.value_counts().to_dict()))
    audit=[]
    for label,g in moving.groupby('fixed_label'):
        reliable=g[g.speed_mps>=5];strong=reliable[reliable.pose_yaw_left_rad_s.abs()>.012]
        opposite=strong[(strong.pose_yaw_left_rad_s<0)&(label=='LEFT') | (strong.pose_yaw_left_rad_s>0)&(label=='RIGHT')]
        audit.append(dict(label=label,rows=len(g),routes=int(g.route.nunique()),pose_categories=g.pose_category.value_counts().to_dict(),
             within_0p25deg_of_threshold=int((g.margin_deg<=.25).sum()),stable_11=int(g.stable_11.sum()),
             opposite_strong_turn=len(opposite),median_speed_mps=float(g.speed_mps.median())))
    suspected=selected[selected.moving&(((selected.fixed_label=='STRAIGHT')&selected.pose_category.isin(['STRONG_LEFT','STRONG_RIGHT']))|((selected.fixed_label=='LEFT')&(selected.pose_category=='STRONG_RIGHT'))|((selected.fixed_label=='RIGHT')&(selected.pose_category=='STRONG_LEFT')))]
    suspected.to_csv(OUT/'sensor-conflict-candidates.csv',index=False)
    counts=straight.route.value_counts()
    report=dict(status='AUDIT_COMPLETE',samples=len(selected),moving=len(moving),stopped=int((~selected.moving).sum()),
         class_counts=moving.fixed_label.value_counts().to_dict(),group_counts=selected.group.value_counts().sort_index().to_dict(),
         straight_routes=len(counts),top3_straight_route_share=float(counts.head(3).sum()/len(straight)),
         missing_straight_routes=[r for r in coverage if r['available_straight']>0 and r['selected_straight']==0],
         class_audit=audit,speed_coverage=summaries,sensor_conflict_candidates=len(suspected),
         provenance='Fixed labels reconstructed from raw steering angle and recorded calibration metadata; compared every93602training row and selected1000labels. No official equivalence claim.',
         limitations='Pose is a sensor diagnostic, not human truth; selected points represent16frame clips; class balance does not ensure route/speed diversity. No label edits or retraining.',
         inputs_sha256=hashes,outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    for p,h in hashes.items():assert sha(P/p)==h
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
