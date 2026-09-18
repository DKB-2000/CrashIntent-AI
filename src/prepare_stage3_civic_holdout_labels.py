"""Apply frozen pilot consensus rules to newly reserved recording routes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_stage3_comma2k19 import _smooth, _deduplicate_times

P=Path(__file__).resolve().parents[1]
A=P/'artifacts/stage3-civic-holdout-acquisition-20260914'
OLD=P/'artifacts/stage3-civic-partial-labels-20260914'
OUT=P/'artifacts/stage3-civic-holdout-labels-20260914'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    plan=json.loads((A/'plan.json').read_text());validation=json.loads((A/'validation.json').read_text())
    rules=json.loads((OLD/'plan.json').read_text());offset=rules['offset_deg']
    prior=json.loads((P/'artifacts/stage3-civic-acquisition-20260914/plan.json').read_text())
    assert set(plan['routes']).isdisjoint(prior['routes']) and validation['status']=='PASS'
    frozen=dict(offset_deg=offset,rules=rules,source_plan_sha256=sha(A/'plan.json'),rules_sha256=sha(OLD/'plan.json'),
                script_sha256=sha(Path(__file__)),scope='All9segments of5reserved routes; frozen pilot rules; no fitting or model predictions')
    (OUT/'plan.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    for f in validation['files']:assert sha(P/f['local'])==f['sha256']
    rows=[]
    for i,segment in enumerate(plan['segments']):
        sid=f'CIVIC_HOLDOUT_{i:03}';root=P/'data_raw/comma2k19/civic-holdout-20260914'/sid
        def load(n):return np.asarray(np.load(root/n,allow_pickle=False))
        ft=load('global_pose/frame_times').reshape(-1);t=ft[::2]
        speedkey='speed' if (root/'processed_log/CAN/speed').exists() else 'car_speed'
        st,sv=_deduplicate_times('speed',load(f'processed_log/CAN/{speedkey}/t').reshape(-1),load(f'processed_log/CAN/{speedkey}/value').reshape(-1))
        at,av=_deduplicate_times('steering',load('processed_log/CAN/steering_angle/t').reshape(-1),load('processed_log/CAN/steering_angle/value').reshape(-1))
        speed=np.interp(t,st,sv);angle=np.interp(t,at,av)-offset
        pos=load('global_pose/frame_positions');vel=load('global_pose/frame_velocities')
        assert pos.shape==vel.shape==(len(ft),3) and np.isfinite(pos).all() and np.isfinite(vel).all()
        up=pos/np.linalg.norm(pos,axis=1,keepdims=True)
        h=vel-(vel*up).sum(1,keepdims=True)*up
        h=np.column_stack([_smooth(h[:,j],21) for j in range(3)])
        yaw=np.sum(np.cross(h,np.gradient(h,ft,axis=0))*up,1)/np.maximum(np.sum(h*h,axis=1),1.)
        yaw=np.interp(t,ft,_smooth(yaw,21))
        valid=(t>=max(st[0],at[0]))&(t<=min(st[-1],at[-1]))&(t>=ft[0]+1)&(t<=ft[-1]-1)
        domain=valid&(speed>=5)&(speed<=35)&(abs(yaw)<.12)
        preliminary=np.full(len(t),'UNKNOWN',dtype='<U8')
        preliminary[domain&(abs(angle)<=.5)&(abs(yaw)<.003)]='STRAIGHT'
        preliminary[domain&(angle>1.5)&(yaw>.012)]='LEFT'
        preliminary[domain&(angle< -1.5)&(yaw< -.012)]='RIGHT'
        keep=np.zeros(len(t),bool)
        for j in range(5,len(t)-5):keep[j]=preliminary[j]!='UNKNOWN' and np.all(preliminary[j-5:j+6]==preliminary[j])
        labels=np.where(keep,preliminary,'UNKNOWN')
        # Independent scalar rules and temporal checks for every point.
        scalar=[]
        for j in range(len(t)):
            v=valid[j] and 5<=speed[j]<=35 and abs(yaw[j])<.12
            scalar.append('STRAIGHT' if v and abs(angle[j])<=.5 and abs(yaw[j])<.003 else 'LEFT' if v and angle[j]>1.5 and yaw[j]>.012 else 'RIGHT' if v and angle[j]<-1.5 and yaw[j]<-.012 else 'UNKNOWN')
        for j in range(len(t)):
            expected=scalar[j] if 5<=j<len(t)-5 and scalar[j]!='UNKNOWN' and all(x==scalar[j] for x in scalar[j-5:j+6]) else 'UNKNOWN'
            assert labels[j]==expected
        rows.append(pd.DataFrame(dict(ID=sid,source_frame_index=np.arange(len(t))*2,sample_index=np.arange(len(t)),timestamp=t,
          partial_steer_label=labels,partial_valid_steer=keep,route=segment.rsplit('/',1)[0],speed_mps=speed,
          steering_corrected_deg=angle,pose_yaw_left_rad_s=yaw,
          exclusion_reason=np.where(~domain,'OUTSIDE_DOMAIN',np.where(preliminary=='UNKNOWN','SENSOR_DISAGREEMENT_OR_AMBIGUOUS',np.where(~keep,'NOT_STABLE_11_SAMPLES','RETAINED'))))))
    df=pd.concat(rows,ignore_index=True);df.to_csv(OUT/'evaluation_partial_labels.csv',index=False)
    kept=df[df.partial_valid_steer]
    result=dict(status='PARTIAL_PROXY_LABELS_READY',validation='PASS:99source hashes,disjoint recording routes,frozen offset/rules,scalar rule recomputation',
         evaluation_total=len(df),evaluation_retained=len(kept),evaluation_excluded=len(df)-len(kept),evaluation_excluded_fraction=1-len(kept)/len(df),
         evaluation_class_counts=kept.partial_steer_label.value_counts().to_dict(),
         routes=[dict(route=r,total=len(g),retained=int(g.partial_valid_steer.sum()),counts=g[g.partial_valid_steer].partial_steer_label.value_counts().to_dict()) for r,g in df.groupby('route')],
         outputs_sha256={'evaluation_partial_labels.csv':sha(OUT/'evaluation_partial_labels.csv')})
    (OUT/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
