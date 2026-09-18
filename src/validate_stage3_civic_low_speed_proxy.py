"""Calibrate a fixed low-speed sensor-consensus gate before touching held signals."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_stage3_comma2k19 import _deduplicate_times, _smooth
from review_stage3_comma_calibration import robust_fit

P = Path(__file__).resolve().parents[1]
ACQ = P/'artifacts/stage3-civic-low-speed-acquisition-20260916'
RAW = P/'data_raw/comma2k19/civic-low-speed-20260916'
OUT = P/'artifacts/stage3-civic-low-speed-proxy-20260916'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def signals(sid, route, split):
    folder = RAW/sid
    def load(name): return np.asarray(np.load(folder/name,allow_pickle=False))
    ft = load('global_pose/frame_times').reshape(-1); t = ft[::2]
    speedkey = 'speed' if (folder/'processed_log/CAN/speed').exists() else 'car_speed'
    st,sv = _deduplicate_times('speed', load(f'processed_log/CAN/{speedkey}/t').reshape(-1),
                               load(f'processed_log/CAN/{speedkey}/value').reshape(-1))
    at,av = _deduplicate_times('steering', load('processed_log/CAN/steering_angle/t').reshape(-1),
                               load('processed_log/CAN/steering_angle/value').reshape(-1))
    pos = load('global_pose/frame_positions');vel = load('global_pose/frame_velocities')
    assert pos.shape == vel.shape == (len(ft),3)
    up = pos/np.linalg.norm(pos,axis=1,keepdims=True)
    horizontal = vel-(vel*up).sum(1,keepdims=True)*up
    horizontal = np.column_stack([_smooth(horizontal[:,j],21) for j in range(3)])
    yaw = np.sum(np.cross(horizontal,np.gradient(horizontal,ft,axis=0))*up,axis=1)/np.maximum(np.sum(horizontal*horizontal,axis=1),1.)
    yaw = np.interp(t,ft,_smooth(yaw,21))
    gt = load('processed_log/IMU/gyro/t').reshape(-1)
    gv = load('processed_log/IMU/gyro/value')
    assert gv.shape == (len(gt),3) and np.isfinite(gv).all()
    mask = np.r_[True,np.diff(gt)>0];gt=gt[mask];gv=gv[mask]
    gyro = _smooth(np.interp(t,gt,-gv[:,2]),11)
    valid = (t>=max(st[0],at[0])) & (t<=min(st[-1],at[-1])) & (t>=ft[0]+1) & (t<=ft[-1]-1)
    gvalid = (t>=gt[0]) & (t<=gt[-1])
    frame = pd.DataFrame(dict(ID=sid,route=route,split=split,sample_index=np.arange(len(t)),timestamp=t,
                              speed_mps=np.interp(t,st,sv),steering_angle_deg=np.interp(t,at,av),
                              pose_yaw_left_rad_s=yaw,gyro_yaw_left_rad_s=gyro,valid_sensor=valid,gyro_valid=gvalid))
    assert len(frame)>=100 and np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.all(np.diff(frame.timestamp)>0)
    return frame


def stable(labels):
    keep=np.zeros(len(labels),bool)
    for i in range(5,len(labels)-5):
        keep[i]=labels[i]!='UNKNOWN' and np.all(labels[i-5:i+6]==labels[i])
    return keep


def candidate(frame, offset, bias):
    speed=frame.speed_mps.to_numpy();angle=frame.steering_angle_deg.to_numpy()-offset
    yaw=frame.pose_yaw_left_rad_s.to_numpy();gyro=frame.gyro_yaw_left_rad_s.to_numpy()-bias
    domain=frame.valid_sensor.to_numpy() & frame.gyro_valid.to_numpy() & (speed>=2) & (speed<5) & (np.abs(yaw)<.12)
    label=np.full(len(frame),'UNKNOWN',dtype='<U8')
    label[domain & (np.abs(angle)<=.5) & (np.abs(yaw)<.003) & (np.abs(gyro)<.006)]='STRAIGHT'
    label[domain & (angle>2) & (yaw>.02) & (gyro>.02)]='LEFT'
    label[domain & (angle< -2) & (yaw< -.02) & (gyro< -.02)]='RIGHT'
    keep=stable(label)
    return np.where(keep,label,'UNKNOWN'), keep


def main():
    assert not OUT.exists(), 'Existing proxy result; no silent rerun'
    plan=json.loads((ACQ/'plan.json').read_text(encoding='utf-8'))
    validation=json.loads((ACQ/'validation.json').read_text(encoding='utf-8'))
    status=json.loads((ACQ/'status.json').read_text(encoding='utf-8'))
    assert validation['status']=='PASS' and status['status']=='ACQUIRED_VALIDATED_UNCALIBRATED'
    assert len(plan['segments'])==10 and len(plan['route_splits'])==5
    assert list(plan['route_splits'].values()).count('calibration')==3
    assert list(plan['route_splits'].values()).count('evaluation')==2
    assert len(validation['segments'])==10 and all(s['sensor_coverage_pass'] for s in validation['segments'])
    OUT.mkdir()
    cfg=dict(status='FROZEN_BEFORE_EVALUATION',domain='2<=speed<5m/s,valid CAN/pose/gyro,abs(pose yaw)<.12',
             labels='STRAIGHT abs(angle)<=.5,abs(pose)<.003,abs(debiased gyro)<.006; LEFT/RIGHT abs(angle)>2,abs(pose)> .02,abs(debiased gyro)>.02 and signs agree; centered11 persistence',
             gyro_bias='From other same-route segment, speed5-35 valid points, median(gyro-pose); no model output used',
             gate='Each calibration route >=30 valid low-speed strong turns; transferred gyro-pose sign>=97% on strong turns; high-speed transferred MAE<=.005rad/s; each candidate class>=30 and >=2classes/route. Apply held routes only if all pass.',
             acquisition_plan_sha256=sha(ACQ/'plan.json'),acquisition_validation_sha256=sha(ACQ/'validation.json'),
             script_sha256=sha(Path(__file__)),scope='Calibration3routes; evaluation2routes untouched until gate passes. Proxy labels, not official truth.')
    save('plan.json',cfg)
    raw=[]
    for i,seg in enumerate(plan['segments']):
        route=seg.rsplit('/',1)[0]
        if plan['route_splits'][route]=='calibration':
            raw.append(signals(f'CIVIC_LOW_{i:03}',route,'calibration'))
    calib=pd.concat(raw,ignore_index=True)
    assert calib.route.nunique()==3 and calib.ID.nunique()==6
    high=calib[calib.valid_sensor & (calib.speed_mps>=5) & (calib.speed_mps<=35) & (calib.pose_yaw_left_rad_s.abs()<.12) & (calib.sample_index%10==0)]
    assert len(high)>=100
    fit=high.rename(columns={'sample_index':'frame_index'})
    beta,stats=robust_fit(fit,'pose_yaw_left_rad_s')
    offset=float(beta[0])
    records=[]; transfers=[]; labeled=[]
    for route,group in calib.groupby('route'):
        ids=sorted(group.ID.unique());assert len(ids)==2
        for target in ids:
            source=next(x for x in ids if x!=target)
            a=group[(group.ID==source)&group.valid_sensor&group.gyro_valid&(group.speed_mps>=5)&(group.speed_mps<=35)]
            assert len(a)>=30
            bias=float(np.median(a.gyro_yaw_left_rad_s-a.pose_yaw_left_rad_s))
            b=group[(group.ID==target)&group.valid_sensor&group.gyro_valid&(group.speed_mps>=5)&(group.speed_mps<=35)]
            high_mae=float(np.mean(np.abs(b.gyro_yaw_left_rad_s-bias-b.pose_yaw_left_rad_s)))
            g=group[group.ID==target].copy()
            low=g[g.valid_sensor&g.gyro_valid&(g.speed_mps>=2)&(g.speed_mps<5)]
            strong=low[low.pose_yaw_left_rad_s.abs()>.02]
            centered=strong.gyro_yaw_left_rad_s.to_numpy()-bias
            agreement=float(np.mean(np.sign(centered)==np.sign(strong.pose_yaw_left_rad_s))) if len(strong) else None
            transfers.append(dict(route=route,source=source,target=target,bias=bias,high_mae=high_mae,
                                  low_valid=len(low),low_strong=len(strong),strong_sign_agreement=agreement))
            labels,keep=candidate(g,offset,bias)
            g['proxy_label']=labels;g['proxy_valid']=keep
            labeled.append(g)
            records.append(dict(route=route,ID=target,low_valid=len(low),retained=int(keep.sum()),classes=g[g.proxy_valid].proxy_label.value_counts().to_dict()))
    cal=pd.concat(labeled,ignore_index=True)
    counts=cal[cal.proxy_valid].proxy_label.value_counts().to_dict()
    gate=all(sum(x['low_strong'] for x in transfers if x['route']==r)>=30 for r in cal.route.unique())
    gate &= all(x['strong_sign_agreement'] is not None and x['strong_sign_agreement']>=.97 and x['high_mae']<=.005 for x in transfers)
    gate &= all(counts.get(k,0)>=30 for k in ('LEFT','STRAIGHT','RIGHT'))
    gate &= all(sum(v>0 for v in cal[(cal.route==r)&cal.proxy_valid].proxy_label.value_counts().to_dict().values())>=2 for r in cal.route.unique())
    cal.to_csv(OUT/'calibration-signals-and-proxy.csv',index=False)
    result=dict(status='CALIBRATION_VALIDATED',decision='EVALUATE_FROZEN_PROXY' if gate else 'NO_LOW_SPEED_EVALUATION_LABELS',
                gate_passed=bool(gate),offset_deg=offset,offset_fit=stats,calibration_rows=len(cal),
                calibration_low_valid=int((cal.valid_sensor & cal.gyro_valid & (cal.speed_mps>=2) & (cal.speed_mps<5)).sum()),candidate_class_counts=counts,
                segments=records,gyro_transfer=transfers,
                validation='Acquisition PASS, fixed route split, calibration-only threshold/offset/bias evaluation; held raw signals not read before gate.',
                output_sha256={'calibration-signals-and-proxy.csv':sha(OUT/'calibration-signals-and-proxy.csv')})
    save('calibration-review.json',result)
    if not gate:
        print(json.dumps({k:result[k] for k in ('status','decision','gate_passed','candidate_class_counts')},indent=2))
        return
    # Gate passed: apply the already frozen thresholds and sibling-segment bias method once.
    held=[]
    for i,seg in enumerate(plan['segments']):
        route=seg.rsplit('/',1)[0]
        if plan['route_splits'][route]=='evaluation':
            held.append(signals(f'CIVIC_LOW_{i:03}',route,'evaluation'))
    evalframe=pd.concat(held,ignore_index=True)
    outputs=[]
    for route,group in evalframe.groupby('route'):
        ids=sorted(group.ID.unique());assert len(ids)==2
        for target in ids:
            source=next(x for x in ids if x!=target)
            a=group[(group.ID==source)&group.valid_sensor&group.gyro_valid&(group.speed_mps>=5)&(group.speed_mps<=35)]
            assert len(a)>=30
            bias=float(np.median(a.gyro_yaw_left_rad_s-a.pose_yaw_left_rad_s))
            g=group[group.ID==target].copy();labels,keep=candidate(g,offset,bias)
            g['proxy_label']=labels;g['proxy_valid']=keep;outputs.append(g)
    out=pd.concat(outputs,ignore_index=True)
    out.to_csv(OUT/'evaluation-proxy-labels.csv',index=False)
    result['status']='COMPLETE_VALIDATED';result['evaluation_total']=len(out)
    result['evaluation_retained']=int(out.proxy_valid.sum())
    result['evaluation_class_counts']=out[out.proxy_valid].proxy_label.value_counts().to_dict()
    result['output_sha256']['evaluation-proxy-labels.csv']=sha(OUT/'evaluation-proxy-labels.csv')
    save('final-review.json',result)
    print(json.dumps({k:result[k] for k in ('status','decision','evaluation_total','evaluation_retained','evaluation_class_counts')},indent=2))


if __name__=='__main__':main()
