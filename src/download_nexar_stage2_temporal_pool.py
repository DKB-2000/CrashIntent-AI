"""Download and validate the frozen 200-video Nexar Stage2 training pool."""
from __future__ import annotations
import csv,json,os,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1];POOL=ROOT/'artifacts/stage2-nexar-temporal-pool-20260917';MANIFEST=POOL/'manifest-sized.csv';OUT=ROOT/'data_raw/nexar-stage2-temporal-train-20260917';STATUS=POOL/'download-status.json'
def write(status,**extra):
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps({'status':status,'updated_unix':time.time(),**extra},indent=2)+'\n',encoding='utf-8');p.replace(STATUS)
def fetch(r):
 target=OUT/r['file_name'];expected=int(r['remote_bytes'])
 if target.exists() and target.stat().st_size==expected:return r['file_name'],expected,'existing'
 part=target.with_suffix('.mp4.part')
 for attempt in range(1,4):
  try:
   offset=part.stat().st_size if part.exists() else 0;headers={'User-Agent':'crashvideo-stage2-pool/1.0'}
   if offset:headers['Range']=f'bytes={offset}-'
   req=urllib.request.Request(r['url'],headers=headers)
   with urllib.request.urlopen(req,timeout=90) as src,part.open('ab' if offset and src.status==206 else 'wb') as dst:
    while True:
     block=src.read(1024*1024)
     if not block:break
     dst.write(block)
   if part.stat().st_size!=expected:raise RuntimeError(f"size {part.stat().st_size} != {expected}")
   part.replace(target);return r['file_name'],expected,'downloaded'
  except Exception:
   if attempt==3:raise
   time.sleep(2**attempt)
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8-sig',newline='')));expected=sum(int(r['remote_bytes']) for r in rows);write('DOWNLOADING',completed=0,total=len(rows),bytes_completed=0,expected_bytes=expected)
 done=0;bytes_done=0
 with ThreadPoolExecutor(max_workers=4) as ex:
  jobs={ex.submit(fetch,r):r for r in rows}
  for f in as_completed(jobs):
   name,size,mode=f.result();done+=1;bytes_done+=size;write('DOWNLOADING',completed=done,total=len(rows),bytes_completed=bytes_done,expected_bytes=expected,last_file=name,last_mode=mode)
 bad=[];decoded=[]
 for i,r in enumerate(rows,1):
  path=OUT/r['file_name'];cap=cv2.VideoCapture(str(path));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));duration=frames/fps if fps>0 else 0;ok=cap.isOpened() and frames>0 and fps>0 and float(r['time_of_event'])<=duration+1.;cap.release()
  if not ok:bad.append({'file':r['file_name'],'frames':frames,'fps':fps,'duration':duration,'event':r['time_of_event']})
  decoded.append({'file_name':r['file_name'],'bytes':path.stat().st_size,'frames':frames,'fps':fps,'duration_seconds':duration,'event_seconds':float(r['time_of_event']),'alert_seconds':float(r['time_of_alert'])})
  if i%20==0:write('VALIDATING',completed=i,total=len(rows),bad=len(bad))
 with (POOL/'decoded-videos.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(decoded[0]));w.writeheader();w.writerows(decoded)
 if bad:write('FAILED_VALIDATION',completed=len(rows),bad=bad[:20]);raise RuntimeError(f'{len(bad)} videos failed decode validation')
 write('COMPLETE_VALIDATED',completed=len(rows),total=len(rows),bytes=bytes_done,decoded=len(decoded),bad=0)
if __name__=='__main__':main()
