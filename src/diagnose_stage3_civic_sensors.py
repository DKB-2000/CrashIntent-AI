"""Calibration-route-only gyro and speed diagnostics, without relabeling."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from review_stage3_comma_calibration import robust_fit

P=Path(__file__).resolve().parents[1]
SOURCE=P/'artifacts/stage3-civic-calibration-20260914'
OUT=P/'artifacts/stage3-civic-sensor-diagnostic-20260914'


def correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-9 and np.std(b)>1e-9 else None


def compare(g):
    x=g.pose_yaw_left_rad_s.to_numpy();y=g.gyro_yaw_left_rad_s.to_numpy()
    bias=float(np.median(y-x)); strong=np.abs(x)>.012
    return dict(rows=len(g),correlation=correlation(x,y),median_bias_rad_s=bias,
                raw_mae=float(np.mean(np.abs(y-x))),debiased_mae=float(np.mean(np.abs(y-bias-x))),
                strong_rows=int(strong.sum()),raw_sign=float(np.mean(np.sign(y[strong])==np.sign(x[strong]))) if strong.any() else None,
                centered_sign=float(np.mean(np.sign((y-bias)[strong])==np.sign(x[strong]))) if strong.any() else None)


def main():
    OUT.mkdir(exist_ok=False)
    report=json.loads((SOURCE/'report.json').read_text())
    path=SOURCE/'signals.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==report['outputs_sha256']['signals.csv']
    data=pd.read_csv(path)
    data=data[(data.split=='calibration')&data.valid_sensor&data.gyro_valid&(data.speed_mps>=5)].copy()
    assert data.route.nunique()==3 and data.ID.nunique()==6
    plan=dict(scope='Only3calibration routes/6segments; no evaluation metrics, label creation or model predictions',
              lag_grid_seconds=[i/10 for i in range(-10,11)],
              bias='Median gyro minus pose; same-segment correction diagnostic only; cross-segment bias check reported separately',
              speed='Fixed speed bins5/10/20/30/40m/s; leave-one-route-out angle regression: intercept+yaw versus intercept+yaw/speed*1000+yaw*speed',
              signals_sha256=report['outputs_sha256']['signals.csv'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    segments=[];lags=[];transfer=[];speed=[];cross=[]
    for sid,g in data.groupby('ID'):
        g=g.sort_values('timestamp');row=dict(ID=sid,route=g.route.iloc[0],**compare(g));segments.append(row)
        t=g.timestamp.to_numpy();x=g.pose_yaw_left_rad_s.to_numpy();y=g.gyro_yaw_left_rad_s.to_numpy()
        for lag in plan['lag_grid_seconds']:
            mask=(t+lag>=t[0])&(t+lag<=t[-1])
            shifted=np.interp(t[mask]+lag,t,y);truth=x[mask]
            bias=np.median(shifted-truth)
            lags.append(dict(ID=sid,lag=lag,correlation=correlation(truth,shifted),centered_mae=float(np.mean(np.abs(shifted-bias-truth)))))
    for route,g in data.groupby('route'):
        ids=sorted(g.ID.unique())
        for source in ids:
            a=g[g.ID==source];bias=float(np.median(a.gyro_yaw_left_rad_s-a.pose_yaw_left_rad_s))
            target=next(s for s in ids if s!=source);b=g[g.ID==target]
            mask=b.pose_yaw_left_rad_s.abs()>.012
            transfer.append(dict(route=route,source=source,target=target,bias=bias,
               mae=float(np.mean(np.abs(b.gyro_yaw_left_rad_s-bias-b.pose_yaw_left_rad_s))),
               strong_rows=int(mask.sum()),sign_agreement=float(np.mean(np.sign(b.loc[mask,'gyro_yaw_left_rad_s']-bias)==np.sign(b.loc[mask,'pose_yaw_left_rad_s']))) if mask.any() else None))
    for lo,hi in [(5,10),(10,20),(20,30),(30,40)]:
        g=data[(data.speed_mps>=lo)&(data.speed_mps<hi)]
        if not len(g):continue
        angle=g.steering_angle_deg-report['offset']['offset_deg']
        straight=g.pose_yaw_left_rad_s.abs()<.003;turn=g.pose_yaw_left_rad_s.abs()>.012
        pred=np.where(angle>1.5,'LEFT',np.where(angle< -1.5,'RIGHT','STRAIGHT'))
        truth=np.where(g.pose_yaw_left_rad_s>0,'LEFT','RIGHT')
        speed.append(dict(speed_min=lo,speed_max=hi,rows=len(g),straight_rows=int(straight.sum()),turn_rows=int(turn.sum()),
            straight_agreement=float(np.mean(pred[straight]=='STRAIGHT')) if straight.any() else None,
            turn_agreement=float(np.mean(pred[turn]==truth[turn])) if turn.any() else None))
    fit=data[(data.speed_mps<=35)&(data.pose_yaw_left_rad_s.abs()<.12)&(data.frame_index%10==0)]
    for route in sorted(fit.route.unique()):
        train=fit[fit.route!=route];test=fit[fit.route==route]
        beta,_=robust_fit(train,'pose_yaw_left_rad_s')
        def design(g):return np.column_stack([np.ones(len(g)),g.pose_yaw_left_rad_s/g.speed_mps*1000,g.pose_yaw_left_rad_s*g.speed_mps])
        # Same robust solver iterations for the yaw-only reference.
        x=np.column_stack([np.ones(len(train)),train.pose_yaw_left_rad_s]);y=train.steering_angle_deg.to_numpy();weights=np.ones(len(y))
        for _ in range(15):
            base=np.linalg.lstsq(x*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0]
            residual=y-x@base;scale=max(1.4826*np.median(np.abs(residual-np.median(residual))),.05)
            weights=np.minimum(1,1.5*scale/np.maximum(np.abs(residual),1e-9))
        target=test.steering_angle_deg.to_numpy()
        cross.append(dict(held_route=route,train_rows=len(train),test_rows=len(test),
             yaw_only_mae=float(np.mean(np.abs(target-np.column_stack([np.ones(len(test)),test.pose_yaw_left_rad_s])@base))),
             speed_terms_mae=float(np.mean(np.abs(target-design(test)@beta))),
             speed_terms_coefficients=beta.tolist()))
    for name,rows in [('segments',segments),('lag_scan',lags),('bias_transfer',transfer),('speed_bins',speed),('route_regression',cross)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    result=dict(status='DIAGNOSTIC_COMPLETE',segments=segments,bias_transfer=transfer,speed_bins=speed,route_regression=cross,
         best_centered_lags=[min([r for r in lags if r['ID']==sid],key=lambda r:r['centered_mae']) for sid in sorted(data.ID.unique())],
         limits='Bias and lag fits are diagnostics, not physical certification. Cross-segment bias checks are within3calibration routes.6segments are correlated. No timestamps, raw files or labels modified.',
         outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')})
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
