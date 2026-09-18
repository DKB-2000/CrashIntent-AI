"""Conservative consensus labels for a diagnostic subset, not full-domain truth."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
SOURCE=P/'artifacts/stage3-civic-calibration-20260914'
OUT=P/'artifacts/stage3-civic-partial-labels-20260914'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    prior=json.loads((SOURCE/'report.json').read_text())
    assert sha(SOURCE/'signals.csv')==prior['outputs_sha256']['signals.csv']
    split=json.loads((SOURCE/'plan.json').read_text())
    offset=prior['offset']['offset_deg']
    plan=dict(status='FIXED_BEFORE_SELECTION',offset_deg=offset,
              straight='abs(corrected steering)<=0.5deg AND abs(pose yaw)<0.003rad/s',
              left='corrected steering>1.5deg AND pose yaw>0.012rad/s',
              right='corrected steering< -1.5deg AND pose yaw< -0.012rad/s',
              domain='valid_sensor AND 5<=speed<=35m/s AND abs(pose yaw)<0.12rad/s',
              persistence='Same non-UNKNOWN preliminary label across centered11samples (5past,5future). Offline labels only.',
              splits=split,gyro='Not used for labels: route-specific bias diagnosed; no evaluation-route bias fitting.',
              semantics='Provisional sensor consensus diagnostic subset, not official/human truth or unbiased whole-video evaluation.',
              provenance='Fixed existing calibration offset and existing threshold values; no tuning by retained counts or model predictions.',
              inputs_sha256={str(p.relative_to(P)):sha(p) for p in [Path(__file__),SOURCE/'signals.csv',SOURCE/'report.json',SOURCE/'plan.json']})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    data=pd.read_csv(SOURCE/'signals.csv')
    angle=data.steering_angle_deg.to_numpy()-offset;yaw=data.pose_yaw_left_rad_s.to_numpy()
    domain=data.valid_sensor.to_numpy()&(data.speed_mps.to_numpy()>=5)&(data.speed_mps.to_numpy()<=35)&(np.abs(yaw)<.12)
    labels=np.full(len(data),'UNKNOWN',dtype='<U8')
    labels[domain&(np.abs(angle)<=.5)&(np.abs(yaw)<.003)]='STRAIGHT'
    labels[domain&(angle>1.5)&(yaw>.012)]='LEFT'
    labels[domain&(angle< -1.5)&(yaw< -.012)]='RIGHT'
    stable=np.zeros(len(data),bool)
    for sid,g in data.groupby('ID',sort=False):
        idx=g.index.to_numpy();assert np.all(np.diff(g.sample_index)==1)
        for j in range(5,len(idx)-5):
            window=idx[j-5:j+6]
            stable[idx[j]]=labels[idx[j]]!='UNKNOWN' and np.all(labels[window]==labels[idx[j]])
    data['partial_steer_label']=np.where(stable,labels,'UNKNOWN')
    data['partial_valid_steer']=stable
    data['exclusion_reason']=np.where(~domain,'OUTSIDE_DOMAIN',np.where(labels=='UNKNOWN','SENSOR_DISAGREEMENT_OR_AMBIGUOUS',np.where(~stable,'NOT_STABLE_11_SAMPLES','RETAINED')))
    data.to_csv(OUT/'all_samples.csv',index=False)
    evaluation=data[data.split=='evaluation'].copy()
    columns=['ID','source_frame_index','sample_index','timestamp','partial_steer_label','partial_valid_steer','exclusion_reason','route']
    evaluation[columns].to_csv(OUT/'evaluation_partial_labels.csv',index=False)
    summaries=[]
    for (part,route),g in data.groupby(['split','route']):
        kept=g[g.partial_valid_steer]
        summaries.append(dict(split=part,route=route,total=len(g),retained=len(kept),excluded=len(g)-len(kept),
                              retained_fraction=len(kept)/len(g),counts=kept.partial_steer_label.value_counts().to_dict(),
                              reasons=g.exclusion_reason.value_counts().to_dict()))
    # Independent scalar CSV checks against saved physical signals, plus per-video persistence.
    with (OUT/'all_samples.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    groups={}
    for row in rows:groups.setdefault(row['ID'],[]).append(row)
    for group in groups.values():
        def preliminary(r):
            a=float(r['steering_angle_deg'])-offset;y=float(r['pose_yaw_left_rad_s']);v=float(r['speed_mps'])
            if r['valid_sensor']!='True' or not 5<=v<=35 or abs(y)>=.12:return 'UNKNOWN'
            if abs(a)<=.5 and abs(y)<.003:return 'STRAIGHT'
            if a>1.5 and y>.012:return 'LEFT'
            if a< -1.5 and y< -.012:return 'RIGHT'
            return 'UNKNOWN'
        initial=[preliminary(r) for r in group]
        for j,r in enumerate(group):
            keep=5<=j<len(group)-5 and initial[j]!='UNKNOWN' and all(x==initial[j] for x in initial[j-5:j+6])
            assert r['partial_valid_steer']==str(keep)
            assert r['partial_steer_label']==(initial[j] if keep else 'UNKNOWN')
            assert int(r['source_frame_index'])==2*int(r['sample_index'])
    assert set(evaluation.route)==set(split['evaluation_routes'])
    assert set(evaluation.route).isdisjoint(split['calibration_routes'])
    selected=evaluation[evaluation.partial_valid_steer]
    episodes=[]
    for sid,g in evaluation.groupby('ID'):
        block=(g.partial_steer_label!=g.partial_steer_label.shift()).cumsum()
        for _,clip in g.groupby(block):
            if clip.partial_steer_label.iloc[0]=='UNKNOWN':continue
            episodes.append(dict(ID=sid,label=clip.partial_steer_label.iloc[0],first_sample=int(clip.sample_index.iloc[0]),last_sample=int(clip.sample_index.iloc[-1]),samples=len(clip)))
    pd.DataFrame(episodes,columns=['ID','label','first_sample','last_sample','samples']).to_csv(OUT/'evaluation_episodes.csv',index=False)
    result=dict(status='PARTIAL_PROXY_LABELS_READY',validation='PASS: scalar label/persistence recomputation, source-frame mapping, fixed route split, input/output hashes',
                evaluation_total=len(evaluation),evaluation_retained=len(selected),evaluation_excluded=len(evaluation)-len(selected),
                evaluation_excluded_fraction=1-len(selected)/len(evaluation),evaluation_class_counts=selected.partial_steer_label.value_counts().to_dict(),
                evaluation_episode_count=len(episodes),routes=summaries,
                scope='Consensus-selected, temporally stable, speed5-35m/s subset. Correlated frames,2routes, excludes ambiguity; not full-domain model performance. Gyro not a labeling source. No model predictions.',
                outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    for p,h in plan['inputs_sha256'].items():assert sha(P/p)==h
    (OUT/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
