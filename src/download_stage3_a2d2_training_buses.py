"""Acquire and validate the two A2D2 bus files reserved for training."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np


OUT=Path("artifacts/stage3-a2d2-training-buses-20260917")
RAW=Path("data_raw/a2d2")
BASE="https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar"
FILES={
 "20190401_121727":("20190401121727_bus_signals.json",184_996_122),
 "20190401_145936":("20190401145936_bus_signals.json",149_762_346),
}


def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def acquire(route,name,size,deadline):
 path=RAW/name;url=f"{BASE}/{route}/bus/{name}";path.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(1,4):
  received=path.stat().st_size if path.exists() else 0
  if received==size:break
  if received>size:path.unlink();received=0
  try:
   if time.monotonic()>=deadline:raise TimeoutError('four-hour deadline')
   request=urllib.request.Request(url,headers={'Range':f'bytes={received}-'})
   with urllib.request.urlopen(request,timeout=120) as response:
    if received and response.status!=206:raise RuntimeError('Range request ignored')
    with path.open('ab' if received else 'wb') as f:
     while True:
      block=response.read(1024*1024)
      if not block:break
      f.write(block)
  except Exception:
   if attempt==3:raise
   time.sleep(attempt*5)
 if path.stat().st_size!=size:raise ValueError(f'{route}: size mismatch')
 with path.open('r',encoding='utf-8') as f:payload=json.load(f)
 required=['angular_velocity_omega_z','steering_angle_calculated','steering_angle_calculated_sign','vehicle_speed','acceleration_x']
 missing=[x for x in required if x not in payload]
 if missing:raise ValueError(f'{route}: missing {missing}')
 stats={}
 for key in required:
  a=np.asarray(payload[key]['values'],dtype=np.float64)
  if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or np.any(np.diff(a[:,0])<=0):raise ValueError(f'{route}: invalid {key}')
  stats[key]=dict(samples=len(a),timestamp_min=int(a[0,0]),timestamp_max=int(a[-1,0]),value_min=float(a[:,1].min()),value_max=float(a[:,1].max()))
 return dict(route=route,file=str(path),bytes=size,sha256=sha(path),signals=len(payload),required=stats)


def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'worker.lock'
 try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError as exc:raise RuntimeError('training bus acquisition already running') from exc
 os.write(fd,str(os.getpid()).encode());os.close(fd);status=OUT/'status.json';deadline=time.monotonic()+4*3600
 try:
  save(status,dict(status='DOWNLOADING',completed=0,total=2,pid=os.getpid()))
  reports=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(acquire,route,name,size,deadline) for route,(name,size) in FILES.items()]
   for future in concurrent.futures.as_completed(futures):
    reports.append(future.result());save(status,dict(status='DOWNLOADING',completed=len(reports),total=2,pid=os.getpid()))
  report=dict(status='ACQUIRED_VALIDATED',routes=len(reports),total_bytes=sum(x['bytes'] for x in reports),records=sorted(reports,key=lambda x:x['route']))
  save(OUT/'report.json',report);save(status,dict(status='ACQUIRED_VALIDATED',completed=2,total=2,report=str(OUT/'report.json')))
 except Exception as exc:save(status,dict(status='FAILED',error=repr(exc)));raise
 finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
