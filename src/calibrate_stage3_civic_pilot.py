"""Route-separated sensor proxy calibration; no model predictions or training."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
from prepare_stage3_comma2k19 import _smooth, _make_labels, _deduplicate_times
from review_stage3_comma_calibration import robust_fit

P=Path(__file__).resolve().parents[1]
SOURCE=P/'artifacts/stage3-civic-acquisition-20260914'
ROOT=P/'artifacts/stage3-civic-calibration-20260914'


def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def main():
    ROOT.mkdir(exist_ok=False)
    plan=json.loads((SOURCE/'plan.json').read_text(encoding='utf-8'))
    acquired=json.loads((SOURCE/'validation.json').read_text(encoding='utf-8'))
    assert acquired['status']=='PASS'
    routes=sorted(plan['routes']); random.Random(20260915).shuffle(routes)
    calibration=set(routes[:3]); evaluation=set(routes[3:])
    save('plan.json',dict(calibration_routes=sorted(calibration),evaluation_routes=sorted(evaluation),
         offset='Robust pose yaw/speed + yaw*speed regression intercept on calibration routes only, >=5 <=35m/s, every1s, excluding first/last1s',
         threshold_grid=[.5,1,1.5,2],gate='>=30 near-straight and >=30 strong-turn calibration rows; >=90% straight and >=97% turn agreement; positive fitted curvature response; pose/gyro sign diagnostics only',
         boundaries='Exclude first/last1s and target points outside CAN range; do not shift timestamps or infer official labels',
         purpose='Provisional sensor proxy evaluation labels; neither official nor human ground truth',
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for f in acquired['files']:
        data=(P/f['local']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==f['sha256']
    frames=[]
    for i,segment in enumerate(plan['segments']):
        sid=f'CIVIC_PILOT_{i:03}'; folder=P/'data_raw/comma2k19/civic-pilot-20260914'/sid
        def load(k):return np.asarray(np.load(folder/k,allow_pickle=False))
        ft=load('global_pose/frame_times').reshape(-1);target=ft[::2]
        signals={}
        for key in ['car_speed','speed','steering_angle']:
            if not (folder/'processed_log/CAN'/key).exists():continue
            t=load(f'processed_log/CAN/{key}/t').reshape(-1);v=load(f'processed_log/CAN/{key}/value').reshape(-1)
            signals[key]=_deduplicate_times(key,t,v)
        st,sv=signals.get('speed',signals.get('car_speed'));at,av=signals['steering_angle']
        f=_make_labels(sid,target,st,sv,at,av,smoothing_window=7,stop_speed=.5,accel_threshold=.2,
                       steer_deadzone=.5,steer_offset=0,positive_steer_label='LEFT')
        pos=load('global_pose/frame_positions');vel=load('global_pose/frame_velocities')
        assert pos.shape==vel.shape==(len(ft),3) and np.isfinite(pos).all() and np.isfinite(vel).all()
        up=pos/np.linalg.norm(pos,axis=1,keepdims=True)
        horizontal=vel-(vel*up).sum(1,keepdims=True)*up
        horizontal=np.column_stack([_smooth(horizontal[:,j],21) for j in range(3)])
        derivative=np.gradient(horizontal,ft,axis=0)
        yaw=np.sum(np.cross(horizontal,derivative)*up,1)/np.maximum(np.sum(horizontal**2,axis=1),1.)
        f['pose_yaw_left_rad_s']=np.interp(target,ft,_smooth(yaw,21))
        gt=load('processed_log/IMU/gyro/t').reshape(-1);gv=load('processed_log/IMU/gyro/value')
        assert gv.shape==(len(gt),3) and np.isfinite(gv).all() and np.isfinite(gt).all() and np.all(np.diff(gt)>=0)
        keep=np.r_[True,np.diff(gt)>0];gt=gt[keep];gv=gv[keep]
        f['gyro_yaw_left_rad_s']=_smooth(np.interp(target,gt,-gv[:,2]),11)
        f['gyro_valid']=(target>=gt[0])&(target<=gt[-1])
        route=segment.rsplit('/',1)[0]
        f['route']=route;f['split']='calibration' if route in calibration else 'evaluation'
        f['source_frame_index']=np.arange(len(target))*2
        f['valid_sensor']=(target>=max(st[0],at[0])+0)&(target<=min(st[-1],at[-1]))&(target>=ft[0]+1)&(target<=ft[-1]-1)
        frames.append(f)
    data=pd.concat(frames,ignore_index=True)
    assert set(data[data.split=='calibration'].route).isdisjoint(data[data.split=='evaluation'].route)
    valid=data.valid_sensor&(data.speed_mps>=5)&(data.speed_mps<=35)&(data.pose_yaw_left_rad_s.abs()<.12)
    fit=data[valid&(data.split=='calibration')&(data.frame_index%10==0)]
    beta,stats=robust_fit(fit,'pose_yaw_left_rad_s');offset=float(beta[0])
    c=data[valid&(data.split=='calibration')]
    straight=c.pose_yaw_left_rad_s.abs()<.003;turn=c.pose_yaw_left_rad_s.abs()>.012
    positive=np.all(beta[1]/c.speed_mps.to_numpy()*1000+beta[2]*c.speed_mps.to_numpy()>0)
    sensitivity=[]
    for width in [.5,1,1.5,2]:
        angle=c.steering_angle_deg.to_numpy()-offset
        pred=np.where(angle>width,'LEFT',np.where(angle< -width,'RIGHT','STRAIGHT'))
        target=np.where(c.pose_yaw_left_rad_s>0,'LEFT','RIGHT')
        sensitivity.append(dict(width=width,straight_rows=int(straight.sum()),turn_rows=int(turn.sum()),
             straight_agreement=float(np.mean(pred[straight]=='STRAIGHT')) if straight.any() else None,
             turn_agreement=float(np.mean(pred[turn]==target[turn])) if turn.any() else None))
    eligible=[r for r in sensitivity if r['straight_rows']>=30 and r['turn_rows']>=30 and r['straight_agreement']>=.9 and r['turn_agreement']>=.97 and positive]
    width=eligible[0]['width'] if eligible else None
    diagnostics=[]
    for route,g in data[data.valid_sensor].groupby('route'):
        strong=g[(g.speed_mps>=5)&(g.pose_yaw_left_rad_s.abs()>.012)]
        gy=strong[strong.gyro_valid]
        diagnostics.append(dict(route=route,split=g.split.iloc[0],rows=len(g),strong_turn_rows=len(strong),
           gyro_covered_rows=int(g.gyro_valid.sum()),gyro_pose_sign_agreement=float(np.mean(np.sign(gy.gyro_yaw_left_rad_s)==np.sign(gy.pose_yaw_left_rad_s))) if len(gy) else None,
           raw_steer_pose_sign_agreement=float(np.mean(np.sign(strong.steering_angle_deg-offset)==np.sign(strong.pose_yaw_left_rad_s))) if len(strong) else None))
    data['steering_corrected_deg']=data.steering_angle_deg-offset
    data['steer_label']='UNKNOWN'
    if width is not None:
        mask=data.valid_sensor
        angle=data.loc[mask,'steering_corrected_deg']
        data.loc[mask,'steer_label']=np.where(angle>width,'LEFT',np.where(angle< -width,'RIGHT','STRAIGHT'))
    data['valid_steer']=data.valid_sensor&(data.accel_label!='STOPPED')&(data.steer_label!='UNKNOWN')
    data.to_csv(ROOT/'signals.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(ROOT/'threshold_sensitivity_calibration.csv',index=False)
    labels=data[(data.split=='evaluation')&data.valid_steer][['ID','source_frame_index','sample_index','timestamp','accel_label','steer_label','route']]
    labels.to_csv(ROOT/'evaluation_labels_proxy.csv',index=False)
    pd.testing.assert_frame_equal(pd.read_csv(ROOT/'evaluation_labels_proxy.csv'),labels.reset_index(drop=True),check_dtype=False)
    result=dict(status='PROXY_LABELS_READY' if width is not None else 'CALIBRATION_GATE_FAILED_NO_LABELS',
                offset=stats,selected_width=width,positive_curvature_response=bool(positive),sensitivity=sensitivity,
                routes=diagnostics,rows=len(data),excluded_boundary_rows=int((~data.valid_sensor).sum()),
                evaluation_rows=len(labels),evaluation_class_counts=labels.steer_label.value_counts().to_dict(),
                limitations='Small3route calibration/2route evaluation, same highway, sensor-derived proxy only; no official labels, manual review, model fitting or predictions.',
                outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.csv')})
    save('report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
