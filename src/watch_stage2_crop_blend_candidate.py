from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WATCH=ROOT/'artifacts/stage2-crop-blend-submit-watch-20260917';RESULT=ROOT/'artifacts/stage2-crop-blend-submit-candidate-20260917/status.json';SCRIPT=ROOT/'src/package_stage2_crop_blend_candidate.py';TIMEOUT=1800
def write(status,**extra):
 d={'status':status,'watcher_pid':os.getpid(),'updated_unix':time.time(),'timeout_seconds':TIMEOUT,'result_status_path':str(RESULT.relative_to(ROOT)),**extra};t=WATCH/'status.tmp';t.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');t.replace(WATCH/'status.json')
def main():
 WATCH.mkdir(parents=True,exist_ok=True)
 try:
  with (WATCH/'start.lock').open('x') as f:f.write(str(os.getpid()))
 except FileExistsError as exc:raise RuntimeError('Blend package watcher already started') from exc
 with (WATCH/'run.log').open('w',encoding='utf-8') as log:
  p=subprocess.Popen([sys.executable,str(SCRIPT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));write('RUNNING',child_pid=p.pid)
  try:code=p.wait(timeout=TIMEOUT)
  except subprocess.TimeoutExpired:p.kill();p.wait(timeout=30);write('FAILED_TIMEOUT',child_pid=p.pid,exit_code=p.returncode);return
 try:rs=json.loads(RESULT.read_text(encoding='utf-8')).get('status')
 except Exception as exc:rs=f'UNREADABLE: {exc!r}'
 write('COMPLETE' if code==0 and rs=='ZIP_READY_CPU_VALIDATED_GPU_PENDING' else 'FAILED',child_pid=p.pid,exit_code=code,result_status=rs)
if __name__=='__main__':main()
