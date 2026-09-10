"""Audit immutable Stage3 labels against sensor direction, timestamps, and video samples."""
import json,html,hashlib
from pathlib import Path
import cv2,numpy as np,pandas as pd
D=Path('artifacts/stage3-comma-chunk1-calibrated-v1');BASE=Path('artifacts/stage3-comma-chunk1');OUT=Path('artifacts/stage3-label-audit-20260910')
OUT.mkdir(exist_ok=True)
s=pd.read_csv(D/'signals_and_candidates.csv');labels=pd.read_csv(D/'labels.csv');meta=json.loads((D/'label_version.json').read_text())
angle=s.steering_angle_deg-meta['steer_offset_deg'];expected=np.where(angle>meta['steer_deadzone_deg'],'LEFT',np.where(angle<-meta['steer_deadzone_deg'],'RIGHT','STRAIGHT'))
assert labels[['ID','frame_index']].equals(s[['ID','frame_index']]);assert np.array_equal(labels.steer_label,expected)
s['truth']=labels.steer_label;s['angle_v1']=angle
summary=dict(label_rows=len(s),label_recomputed_exact=True,labels_sha256=hashlib.sha256((D/'labels.csv').read_bytes()).hexdigest(),direction={},timing=[],selected=[])
for split,g in s.groupby('split'):
 moving=g[g.speed_mps>5];strong=moving[moving.pose_yaw_left_rad_s.abs()>.012]
 target=np.where(strong.pose_yaw_left_rad_s>0,'LEFT','RIGHT')
 summary['direction'][split]=dict(strong_pose_rows=len(strong),agreement=float((strong.truth==target).mean()),opposite=float(((strong.truth=='LEFT')&(target=='RIGHT')|(strong.truth=='RIGHT')&(target=='LEFT')).mean()),near_straight_rows=int((moving.pose_yaw_left_rad_s.abs()<.003).sum()),near_straight_label_straight=float((moving.loc[moving.pose_yaw_left_rad_s.abs()<.003,'truth']=='STRAIGHT').mean()))
# Timestamp to frame-index audit for all available segments, not inference from FPS header.
for sid,g in s.groupby('ID'):
 raw=BASE/'segments'/sid/'global_pose/frame_times'
 ft=np.load(raw).reshape(-1);expected_t=ft[::2]
 assert len(expected_t)==len(g)
 error=float(np.max(np.abs(expected_t-g.timestamp.to_numpy())))
 drift=float(np.max(np.abs((expected_t-expected_t[0])-np.arange(len(g))/10)))
 summary['timing'].append(dict(ID=sid,label_timestamp_max_error_s=error,nominal_10hz_max_drift_s=drift))
# Sensor lag diagnostic, centered per video, moving records only; no label tuning.
lagrows=[]
for split in ['train','validation']:
 for lag in range(-10,11):
  xs=[];ys=[]
  for sid,g in s[s.split==split].groupby('ID'):
   x=g.angle_v1.to_numpy();y=g.pose_yaw_left_rad_s.to_numpy()/np.maximum(g.speed_mps.to_numpy(),1)
   mask=(g.speed_mps.to_numpy()>5)&np.isfinite(y)
   if lag>0:a,b,m=x[:-lag],y[lag:],mask[:-lag]&mask[lag:]
   elif lag<0:a,b,m=x[-lag:],y[:lag],mask[-lag:]&mask[:lag]
   else:a,b,m=x,y,mask
   if m.sum()>30:xs.extend(a[m]-a[m].mean());ys.extend(b[m]-b[m].mean())
  lagrows.append(dict(split=split,lag_frames=lag,correlation=float(np.corrcoef(xs,ys)[0,1])))
pd.DataFrame(lagrows).to_csv(OUT/'lag-diagnostic.csv',index=False)
# Error examples plus pose-based strong left/right/straight controls, deterministic.
chosen=[('error_right','COMMA2K19_C1_0134',100),('error_left','COMMA2K19_C1_0134',440),('route_straight','COMMA2K19_C1_0080',210),('opposite','COMMA2K19_C1_0083',460)]
for label,mask in [('strong_left',(s.speed_mps>5)&(s.pose_yaw_left_rad_s>.025)),('strong_right',(s.speed_mps>5)&(s.pose_yaw_left_rad_s<-.025)),('straight',(s.speed_mps>15)&(s.pose_yaw_left_rad_s.abs()<.001)&(s.truth=='STRAIGHT'))]:
 pool=s[mask&(s.frame_index.between(30,550))&(s.split=='train')];r=pool.iloc[len(pool)//2];chosen.append((label,r.ID,int(r.frame_index)))
for i,(kind,sid,center) in enumerate(chosen):
 g=s[s.ID==sid].reset_index(drop=True);start=max(0,center-20);end=min(len(g)-1,center+39)
 cap=cv2.VideoCapture(str(BASE/'converted'/sid/'videos'/f'{sid}.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,start)
 writer=cv2.VideoWriter(str(OUT/f'{i:02d}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,360));tiles=[]
 points=set(np.linspace(start,end,6,dtype=int).tolist())
 for t in range(start,end+1):
  ok,frame=cap.read();assert ok
  frame=cv2.resize(frame,(640,360));r=g.iloc[t]
  cv2.putText(frame,f'{sid} {t/10:.1f}s {r.truth} angle {r.angle_v1:+.1f}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
  cv2.putText(frame,f'pose yaw {r.pose_yaw_left_rad_s:+.3f} rad/s',(8,45),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
  writer.write(frame)
  if t in points:tiles.append(frame)
 cap.release();writer.release();sheet=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]);cv2.imwrite(str(OUT/f'{i:02d}.jpg'),sheet)
 summary['selected'].append(dict(index=i,kind=kind,ID=sid,start=start,end=end,truth=g.iloc[center].truth,angle=float(g.iloc[center].angle_v1),pose_yaw=float(g.iloc[center].pose_yaw_left_rad_s)))
(OUT/'audit.json').write_text(json.dumps(summary,indent=2))
(OUT/'review.html').write_text('<meta charset="utf-8"><title>Stage3 label audit</title><h1>Stage3 label audit</h1>'+''.join(f'<h2>{r["index"]}: {html.escape(r["kind"])} {r["ID"]}</h2><video controls width="640" src="{r["index"]:02d}.mp4"></video><p>10 Hz, {r["start"]/10:.1f}–{r["end"]/10:.1f}s</p>' for r in summary['selected']),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='timing'},indent=2));print('MAX_TIMING',max(r['label_timestamp_max_error_s'] for r in summary['timing']),max(r['nominal_10hz_max_drift_s'] for r in summary['timing']))
print(pd.DataFrame(lagrows).sort_values('correlation',ascending=False).groupby('split').head(2).to_string(index=False))
