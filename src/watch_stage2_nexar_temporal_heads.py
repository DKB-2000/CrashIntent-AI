from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WATCH=ROOT/'artifacts/stage2-nexar-temporal-heads-watch-v2-20260917';RESULT=ROOT/'artifacts/stage2-nexar-temporal-heads-v2-20260917/status.json';SCRIPT=ROOT/'src/train_stage2_nexar_temporal_heads.py';TIMEOUT=7200
def write(status,**extra):
 WATCH.mkdir(parents=True,exist_ok=True);p=WATCH/'status.tmp';p.write_text(json.dumps({'status':status,'watcher_pid':os.getpid(),'updated_unix':time.time(),'timeout_seconds':TIMEOUT,**extra},indent=2)+'\n');p.replace(WATCH/'status.json')
def main():
 WATCH.mkdir(parents=True,exist_ok=True)
 try:(WATCH/'start.lock').open('x').close()
 except FileExistsError as e:raise RuntimeError('already started') from e
 env=os.environ.copy()
 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
 with (WATCH/'run.log').open('w') as log:
  p=subprocess.Popen([sys.executable,str(SCRIPT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));write('RUNNING',child_pid=p.pid)
  try:code=p.wait(timeout=TIMEOUT)
  except subprocess.TimeoutExpired:p.kill();p.wait();write('FAILED_TIMEOUT',child_pid=p.pid);return
 try:result=json.loads(RESULT.read_text()).get('status')
 except Exception as e:result=f'UNREADABLE:{e!r}'
 write('COMPLETE_VALIDATED' if code==0 and result=='COMPLETE_VALIDATED' else 'FAILED',child_pid=p.pid,exit_code=code,result_status=result)
if __name__=='__main__':main()
