"""Read-only error episodes and synchronized review material for Civic June11."""
import hashlib
import html
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
S=P/'artifacts/stage3-civic-partial-score-20260914'
L=P/'artifacts/stage3-civic-partial-labels-20260914'
OUT=P/'artifacts/stage3-civic-error-review-20260914'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(1)
    score=json.loads((S/'report.json').read_text());label=json.loads((L/'report.json').read_text())
    assert sha(S/'predictions.csv')==score['prediction_sha256']
    assert sha(L/'all_samples.csv')==label['outputs_sha256']['all_samples.csv']
    pred=pd.read_csv(S/'predictions.csv');signals=pd.read_csv(L/'all_samples.csv')
    f=pred.merge(signals[['ID','sample_index','speed_mps','steering_corrected_deg','pose_yaw_left_rad_s','gyro_yaw_left_rad_s']],on=['ID','sample_index'],validate='one_to_one')
    f=f[f.route.str.contains('2018-06-11')].copy()
    f['error']=f.partial_valid_steer&(f.partial_steer_label!=f.prediction)
    assert f.error.sum()==72 and f.partial_valid_steer.sum()==428
    episodes=[];cards=[];cache_metrics=[]
    for sid,g in f.groupby('ID'):
        g=g.sort_values('sample_index').reset_index(drop=True)
        provenance=next(v for v in score['provenance'] if v['ID']==sid)
        video=S/(sid+'.avi');cache=S/(sid+'.npz')
        assert sha(video)==provenance['converted_sha256'] and sha(cache)==provenance['cache_sha256']
        with np.load(cache) as z:
            logits=z['steer'];features=z['features']
        prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        assert np.array_equal(np.array(['LEFT','STRAIGHT','RIGHT'])[logits.argmax(1)],g.prediction)
        g['confidence']=prob.max(1)
        # Last flow feature layout: grid128 + magnitude percentiles4 + global mean2.
        g['flow_p90']=features[:,130];g['flow_mean_x']=features[:,132];g['flow_mean_y']=features[:,133]
        g.to_csv(OUT/(sid+'-timeline.csv'),index=False)
        ids=np.flatnonzero(g.error.to_numpy());runs=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
        cap=cv2.VideoCapture(str(video))
        for run in runs:
            if not len(run):continue
            start,end=int(run[0]),int(run[-1]);r=g.iloc[run]
            context_start=max(0,start-10);context_end=min(len(g)-1,end+10)
            points=np.linspace(context_start,context_end,6).round().astype(int)
            tiles=[]
            for t in points:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(t));ok,bgr=cap.read();assert ok
                tile=cv2.resize(bgr,(480,270));row=g.iloc[t]
                cv2.rectangle(tile,(0,0),(480,48),(0,0,0),-1)
                cv2.putText(tile,f'{sid} {t/10:.1f}s proxy:{row.partial_steer_label} pred:{row.prediction}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
                cv2.putText(tile,f'{row.speed_mps*3.6:.1f}km/h angle:{row.steering_corrected_deg:+.2f} yaw:{row.pose_yaw_left_rad_s:+.4f}',(5,38),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                tiles.append(tile)
            number=len(episodes)+1;name=f'episode-{number:02d}.jpg'
            assert cv2.imwrite(str(OUT/name),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
            record=dict(episode=number,ID=sid,start_sample=start,end_sample=end,error_frames=len(r),
                proxy_counts=r.partial_steer_label.value_counts().to_dict(),prediction_counts=r.prediction.value_counts().to_dict(),
                mean_speed_kmh=float(r.speed_mps.mean()*3.6),angle_min=float(r.steering_corrected_deg.min()),angle_max=float(r.steering_corrected_deg.max()),
                yaw_min=float(r.pose_yaw_left_rad_s.min()),yaw_max=float(r.pose_yaw_left_rad_s.max()),
                mean_confidence=float(r.confidence.mean()),mean_flow_p90=float(r.flow_p90.mean()),sheet=name,
                visual_cause='UNDETERMINED: contact sheet only; no causal attribution')
            episodes.append(record)
            cards.append(f'<article><h2>{number}. {sid} {start/10:.1f}–{end/10:.1f}s ({len(r)} errors)</h2><img src="{name}"><pre>{html.escape(json.dumps(record,indent=2))}</pre></article>')
        cap.release()
        for truth,subset in g[g.partial_valid_steer].groupby('partial_steer_label'):
            for error,part in subset.groupby('error'):
                cache_metrics.append(dict(ID=sid,truth=truth,error=bool(error),rows=len(part),mean_speed_kmh=float(part.speed_mps.mean()*3.6),
                    median_confidence=float(part.confidence.median()),median_flow_p90=float(part.flow_p90.median())))
    assert sum(e['error_frames'] for e in episodes)==72
    f[f.error].to_csv(OUT/'errors.csv',index=False)
    pd.DataFrame(cache_metrics).to_csv(OUT/'condition_summary.csv',index=False)
    report=dict(status='MATERIALS_AND_NUMERIC_ANALYSIS_COMPLETE',errors=72,eligible=428,episodes=episodes,
                confusion_pairs=f[f.error].groupby(['partial_steer_label','prediction']).size().rename('count').reset_index().to_dict('records'),
                comparisons=cache_metrics,source_hashes={'predictions':sha(S/'predictions.csv'),'labels':sha(L/'all_samples.csv')},
                limitations='Selected sensor proxy,2videos in one route; confidence uncalibrated; flow magnitude cannot identify camera shake or moving objects. No label/model changes.',
                output_hashes={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Civic error review</title><style>body{font-family:system-ui;margin:24px}img{max-width:100%}article{margin-bottom:40px}pre{white-space:pre-wrap}</style><h1>June11 Civic:72 errors</h1><p>Sensor proxy diagnostic. Six stills per episode with 1s context. Not human ground truth.</p>'+''.join(cards),encoding='utf-8')
    print(json.dumps(dict(episodes=len(episodes),pairs=report['confusion_pairs'],summary=[{k:v for k,v in e.items() if k not in ['visual_cause']} for e in episodes]),indent=2))


if __name__=='__main__':main()
