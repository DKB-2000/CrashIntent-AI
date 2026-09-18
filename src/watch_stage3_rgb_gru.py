"""Boundedly watch, download, and verify the Stage3 RGB+GRU Kaggle run."""
import datetime,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
P=Path(__file__).resolve().parents[1];OUT=P/'artifacts/kaggle-stage3-rgb-gru-20260918';KERNEL='biadis/crashintent-stage3-rgb-gru';KAGGLE=[sys.executable,'-m','kaggle'];STATUS=OUT/'remote-run.json'
def save(**kw):
 x=json.loads(STATUS.read_text()) if STATUS.exists() else {};x.update(kw,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat());q=STATUS.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,STATUS)
def run(args,timeout=900):
 env=dict(os.environ,PYTHONUTF8='1',PYTHONPATH=str(P/'.venv/Lib/site-packages'))
 p=subprocess.run([str(x) for x in args],cwd=P,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,env=env); 
 if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-4000:])
 return p.stdout+p.stderr
def main():
 start=time.monotonic();errors=0
 while time.monotonic()-start<21600:
  try:s=run(KAGGLE+['kernels','status',KERNEL],120);errors=0
  except Exception as e:
   errors+=1;save(status='RETRYING',error=repr(e))
   if errors>=5:raise
   time.sleep(60);continue
  if 'COMPLETE' in s or 'ERROR' in s:
   if 'ERROR' in s:save(status='ERROR');raise RuntimeError(s)
   result=OUT/'result';run(KAGGLE+['kernels','output',KERNEL,'-p',result,'--quiet'],1800);files=list(result.rglob('stage3-rgb-gru-result.zip'))
   if len(files)!=1:raise RuntimeError(f'expected one result, got {files}')
   with zipfile.ZipFile(files[0]) as z:
    if z.testzip() is not None:raise RuntimeError('CRC failure')
    report=json.loads(z.read('report.json'));status=json.loads(z.read('status.json'));best=z.read('best.pt')
   if report.get('status')!='COMPLETE_VALIDATED' or status.get('status')!='COMPLETE_VALIDATED' or report.get('validation_rows')!=18603:raise RuntimeError('result contract failure')
   validation={'status':'VALIDATED','result_sha256':hashlib.sha256(files[0].read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(best).hexdigest(),'best_joint_mean':report['best_joint_mean'],'epochs_completed':report['epochs_completed']};(OUT/'validation.json').write_text(json.dumps(validation,indent=2));save(status='COMPLETE',monitor_status='VALIDATED',validation=validation);return
  save(status='RUNNING',monitor_status='WAITING');time.sleep(60)
 raise TimeoutError('six-hour watcher deadline')
if __name__=='__main__':
 try:main()
 except Exception as e:save(monitor_status='FAILED',error=repr(e));raise
