from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WATCH=ROOT/'artifacts/stage2-nexar-temporal-features-watch-v3-20260917';RESULT=ROOT/'artifacts/stage2-nexar-temporal-features-20260917/status.json';SCRIPT=ROOT/'src/extract_nexar_stage2_temporal_features.py';TIMEOUT=21600
def write(status,**extra):
 WATCH.mkdir(parents=True,exist_ok=True);p=WATCH/'status.tmp';p.write_text(json.dumps({'status':status,'watcher_pid':os.getpid(),'updated_unix':time.time(),'timeout_seconds':TIMEOUT,**extra},indent=2)+'\n',encoding='utf-8');p.replace(WATCH/'status.json')
def main():
 WATCH.mkdir(parents=True,exist_ok=True)
 try:
  with (WATCH/'start.lock').open('x') as f:f.write(str(os.getpid()))
 except FileExistsError as exc:raise RuntimeError('Nexar feature watcher already started') from exc
 with (WATCH/'run.log').open('w',encoding='utf-8') as log:
  env=os.environ.copy()
  for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
   env[key]='1'
  p=subprocess.Popen([sys.executable,str(SCRIPT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));write('RUNNING',child_pid=p.pid)
  try:code=p.wait(timeout=TIMEOUT)
  except subprocess.TimeoutExpired:p.kill();p.wait(timeout=30);write('FAILED_TIMEOUT',child_pid=p.pid,exit_code=p.returncode);return
 try:rs=json.loads(RESULT.read_text(encoding='utf-8')).get('status')
 except Exception as exc:rs=f'UNREADABLE: {exc!r}'
 write('COMPLETE_VALIDATED' if code==0 and rs=='COMPLETE_VALIDATED' else 'FAILED',child_pid=p.pid,exit_code=code,result_status=rs)
if __name__=='__main__':main()
