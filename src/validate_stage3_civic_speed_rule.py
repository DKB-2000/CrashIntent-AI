"""Leave-one-calibration-route-out validation of a speed-aware sensor rule."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from review_stage3_comma_calibration import robust_fit

P=Path(__file__).resolve().parents[1]
SOURCE=P/'artifacts/stage3-civic-calibration-20260914'
OUT=P/'artifacts/stage3-civic-speed-rule-20260914'
GRID=[.003,.006,.009,.012]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(frame,beta):
    gain=beta[1]*1000/frame.speed_mps.to_numpy()+beta[2]*frame.speed_mps.to_numpy()
    assert np.isfinite(gain).all() and np.all(gain>0)
    return (frame.steering_angle_deg.to_numpy()-beta[0])/gain


def metrics(frame,pred):
    straight=frame.pose_yaw_left_rad_s.abs().to_numpy()<.003
    turn=frame.pose_yaw_left_rad_s.abs().to_numpy()>.012
    target=np.where(frame.pose_yaw_left_rad_s>0,'LEFT','RIGHT')
    a=float(np.mean(pred[straight]=='STRAIGHT')) if straight.any() else None
    b=float(np.mean(pred[turn]==target[turn])) if turn.any() else None
    return dict(rows=len(frame),straight_rows=int(straight.sum()),turn_rows=int(turn.sum()),
                intermediate_rows=int((~straight&~turn).sum()),straight_agreement=a,turn_agreement=b,
                passed=bool(straight.sum()>=30 and turn.sum()>=30 and a>=.9 and b>=.97))


def predict(yaw,width):return np.where(yaw>width,'LEFT',np.where(yaw< -width,'RIGHT','STRAIGHT'))


def main():
    OUT.mkdir(exist_ok=False)
    previous=json.loads((SOURCE/'report.json').read_text())
    assert sha(SOURCE/'signals.csv')==previous['outputs_sha256']['signals.csv']
    split=json.loads((SOURCE/'plan.json').read_text())
    plan=dict(routes=split['calibration_routes'],excluded_evaluation_routes=split['evaluation_routes'],
              yaw_threshold_grid_rad_s=GRID,selection='Smallest threshold passing >=90% near-straight and >=97% strong-turn agreement, at least30 rows each, on fitting routes only.',
              gate='All3held calibration routes must pass independently; absent train threshold fails fold. No selection on held routes.',
              fit='Previous robust pose regression,1s cadence; angle=offset+(1000*b1/speed+b2*speed)*yaw',
              domain='Valid sensor boundary,5<=speed<=35m/s,abs(pose_yaw)<.12; outside domain excluded, no deployment claim.',
              scope='Repeated calibration-development study, not untouched test or video model accuracy. Pose intermediate zone excluded from agreement denominators and counted.',
              inputs_sha256={str(p.relative_to(P)):sha(p) for p in [Path(__file__),SOURCE/'signals.csv',SOURCE/'plan.json',P/'src/review_stage3_comma_calibration.py']})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    all_data=pd.read_csv(SOURCE/'signals.csv')
    cal=all_data[all_data.split=='calibration'].copy()
    domain=cal.valid_sensor&(cal.speed_mps>=5)&(cal.speed_mps<=35)&(cal.pose_yaw_left_rad_s.abs()<.12)
    data=cal[domain].copy()
    assert set(data.route)==set(plan['routes']) and set(data.route).isdisjoint(plan['excluded_evaluation_routes'])
    folds=[];tables=[]
    for held in sorted(data.route.unique()):
        train=data[data.route!=held];test=data[data.route==held]
        fit=train[train.frame_index%10==0]
        beta,stats=robust_fit(fit,'pose_yaw_left_rad_s')
        yaw_train=estimate(train,beta);yaw_test=estimate(test,beta)
        candidates=[dict(width=w,**metrics(train,predict(yaw_train,w))) for w in GRID]
        choices=[r['width'] for r in candidates if r['passed']]
        width=min(choices) if choices else None
        pred=predict(yaw_test,width) if width is not None else np.full(len(test),'UNKNOWN')
        result=metrics(test,pred)
        baseline=metrics(test,np.where(test.steering_angle_deg.to_numpy()-beta[0]>1.5,'LEFT',np.where(test.steering_angle_deg.to_numpy()-beta[0]<-1.5,'RIGHT','STRAIGHT')))
        folds.append(dict(held_route=held,fit_routes=sorted(fit.route.unique()),fit_rows=len(fit),coefficients=beta.tolist(),
             calibration_fit=stats,train_candidates=candidates,selected_width=width,held_metrics=result,
             fixed_1p5_same_fold_offset=baseline,passed=bool(width is not None and result['passed'])))
        table=test[['ID','route','frame_index','speed_mps','steering_angle_deg','pose_yaw_left_rad_s']].copy()
        table['estimated_yaw']=yaw_test;table['prediction']=pred;table['selected_width']=width
        tables.append(table)
    oof=pd.concat(tables).sort_index();assert len(oof)==len(data) and not oof[['ID','frame_index']].duplicated().any()
    oof.to_csv(OUT/'calibration_oof.csv',index=False)
    # Re-read saved output and independently calculate exact agreement counts.
    import csv
    counters={r:{'straight':0,'straight_ok':0,'turn':0,'turn_ok':0} for r in plan['routes']}
    with (OUT/'calibration_oof.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            y=float(row['pose_yaw_left_rad_s']);c=counters[row['route']]
            if abs(y)<.003:c['straight']+=1;c['straight_ok']+=int(row['prediction']=='STRAIGHT')
            if abs(y)>.012:c['turn']+=1;c['turn_ok']+=int(row['prediction']==('LEFT' if y>0 else 'RIGHT'))
    for f in folds:
        c=counters[f['held_route']];m=f['held_metrics']
        assert c['straight']==m['straight_rows'] and c['turn']==m['turn_rows']
        for n in ['straight','turn']:
            if c[n]:assert abs(c[n+'_ok']/c[n]-m[n+'_agreement'])<1e-12
    result=dict(status='PASS_FOLLOWUP_ONLY' if all(f['passed'] for f in folds) else 'FAILED_NO_EVALUATION_LABELS',
                folds=folds,calibration_rows=len(cal),in_domain_rows=len(data),excluded_rows=len(cal)-len(data),
                pooled_diagnostic=metrics(data.loc[oof.index],oof.prediction.to_numpy()),
                independent_csv_counts=counters,validation='PASS: route disjointness, complete unique OOF rows and independent CSV counts',
                oof_sha256=sha(OUT/'calibration_oof.csv'),scope=plan['scope'])
    for p,h in plan['inputs_sha256'].items():assert sha(P/p)==h
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
