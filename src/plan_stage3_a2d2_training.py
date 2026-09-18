"""Build a balanced, route-aware A2D2 training clip plan."""
from __future__ import annotations
import concurrent.futures,json,math,os,time,urllib.request
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1];OUT=P/'artifacts/stage3-a2d2-training-plan-20260917';HZ=30.;GRID=100_000
CONFIG={
 '20190401_121727':dict(bus='20190401121727_bus_signals.json',prefix='20190401121727_camera_frontcenter_',anchor_index=100,max_index=27500,per_class=24),
 '20190401_145936':dict(bus='20190401145936_bus_signals.json',prefix='20190401145936_camera_frontcenter_',anchor_index=80,max_index=22000,per_class=12),
}
BASE='https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar'
CLASSES=('LEFT','STRAIGHT','RIGHT')
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(tmp,path)
def pairs(p,n):
 a=np.asarray(p[n]['values'],np.float64)
 if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or np.any(np.diff(a[:,0])<=0):raise ValueError(n)
 return a[:,0],a[:,1]
def interp(t,s,v):
 o=t[0];return np.interp((t-o)/1e6,(s-o)/1e6,v)
def smooth(x):return np.convolve(np.pad(x,4,mode='edge'),np.ones(9)/9,mode='valid')
def choose(candidates,count,gap,context):
 desired=np.linspace(candidates[0],candidates[-1],count);out=[]
 for x in desired:
  ordered=candidates[np.argsort(np.abs(candidates-x))];match=next((int(i) for i in ordered if all(abs(i-j)>=gap for j in out)),None)
  if match is not None:out.append(match)
 if len(out)!=count:raise ValueError(f'{context}: selected {len(out)}/{count}')
 return sorted(out)
def fetch(row):
 dest=OUT/'metadata'/row['route']/f"{row['camera_index']:09d}.json";dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  try:return int(json.loads(dest.read_text())['cam_tstamp'])-row['timestamp_us']
  except Exception:dest.unlink(missing_ok=True)
 for attempt in range(3):
  try:
   with urllib.request.urlopen(row['metadata_url'],timeout=60) as response:data=response.read()
   dest.write_bytes(data);meta=json.loads(data);return int(meta['cam_tstamp'])-row['timestamp_us']
  except Exception:
   if attempt==2:raise RuntimeError(f"metadata unavailable: {row['route']}/{row['camera_index']} {row['metadata_url']}")
   time.sleep((attempt+1)*2)
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];routes=[]
 for route,cfg in CONFIG.items():
  anchor=json.loads((OUT/'anchors'/f'{route}.json').read_text());anchor_ts=int(anchor['cam_tstamp'])
  payload=json.loads((P/'data_raw/a2d2'/cfg['bus']).read_text())
  yt,y=pairs(payload,'angular_velocity_omega_z');st,mag=pairs(payload,'steering_angle_calculated');sgt,sign=pairs(payload,'steering_angle_calculated_sign');vt,speed=pairs(payload,'vehicle_speed')
  if not np.array_equal(st,sgt):raise ValueError('steering timestamps differ')
  start=max(yt[0],st[0],vt[0],anchor_ts+1_500_000)
  camera_end=anchor_ts+(cfg['max_index']-cfg['anchor_index'])*1_000_000/HZ
  end=min(yt[-1],st[-1],vt[-1],camera_end)
  grid=np.arange(math.ceil(start/GRID)*GRID,end,GRID)
  yg=smooth(interp(grid,yt,y));mg=interp(grid,st,mag);sig=interp(grid,sgt,sign)>=.5;vg=interp(grid,vt,speed);a=np.where(sig,-mg,mg);b=-a;moving=vg>=5
  ca=float(np.corrcoef(a[moving],yg[moving])[0,1]);cb=float(np.corrcoef(b[moving],yg[moving])[0,1]);steer=a if ca>=cb else b;mapping='sign1_negative' if ca>=cb else 'sign1_positive'
  edge=(np.arange(len(grid))>=20)&(np.arange(len(grid))<len(grid)-20);base=moving&edge
  masks={'LEFT':base&(yg>=5)&(steer>=20),'RIGHT':base&(yg<=-5)&(steer<=-20),'STRAIGHT':base&(np.abs(yg)<=.75)&(np.abs(steer)<=5)}
  counts={k:int(v.sum()) for k,v in masks.items()}
  for label,mask in masks.items():
   for i in choose(np.flatnonzero(mask),cfg['per_class'],15,f'{route}/{label}'):
    ts=int(grid[i]);index=cfg['anchor_index']+round((ts-anchor_ts)*HZ/1e6);stem=f"{cfg['prefix']}{index:09d}"
    rows.append(dict(route=route,label=label,timestamp_us=ts,camera_index=index,yaw_deg_s=float(yg[i]),signed_steering_deg=float(steer[i]),speed_kph=float(vg[i]),metadata_url=f'{BASE}/{route}/camera/cam_front_center/{stem}.json',image_url=f'{BASE}/{route}/camera/cam_front_center/{stem}.png'))
  routes.append(dict(route=route,duration_seconds=float((grid[-1]-grid[0])/1e6),steering_sign_mapping=mapping,steering_yaw_correlation=max(ca,cb),candidate_counts=counts))
 save(OUT/'draft.json',dict(status='DRAFT',selection_count=len(rows),selections=rows))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:errors=list(pool.map(fetch,rows))
 for row,error in zip(rows,errors):row['timestamp_error_us']=error
 maximum=max(abs(x) for x in errors)
 if maximum>60_000:raise ValueError(f'max timestamp error {maximum}')
 report=dict(status='COMPLETE_VALIDATED',routes=routes,selection_count=len(rows),class_counts={k:sum(r['label']==k for r in rows) for k in CLASSES},max_timestamp_error_us=maximum,selections=rows,holdout_route='20180810_150607')
 save(OUT/'report.json',report);save(OUT/'status.json',dict(status='COMPLETE_VALIDATED',completed=len(rows),total=len(rows),max_timestamp_error_us=maximum));print(json.dumps({k:v for k,v in report.items() if k!='selections'},indent=2))
if __name__=='__main__':main()
