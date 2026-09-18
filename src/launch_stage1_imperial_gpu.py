"""Upload once, launch once, monitor, recover and validate Imperial GPU diagnostic."""
import hashlib,json,os,re,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'artifacts/kaggle-stage1-imperial-20260917';KAGGLE=ROOT/'.venv/Scripts/kaggle.exe'
DATASET='biadis/crashintent-stage1-imperial-assets';KERNEL='biadis/crashintent-stage1-imperial-diagnostic'
def write(**kw):
 p=WORK/'remote-run.json';d=json.loads(p.read_text()) if p.exists() else {};d.update(kw,checked_unix=time.time(),pid=os.getpid());q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));os.replace(q,p)
def run(args,timeout):
 p=subprocess.run([str(x) for x in args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'))
 if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-3000:])
 return p.stdout+p.stderr
def main():
 try:
  if (WORK/'launch-attempt.json').exists():raise RuntimeError('Launch already attempted')
  write(status='UPLOADING',monitor_status='ACTIVE')
  out=run([KAGGLE,'datasets','create','-p',WORK/'dataset','--quiet'],2400);(WORK/'upload.log').write_text(out)
  write(status='WAITING_FOR_DATASET',uploaded=True)
  deadline=time.monotonic()+3600
  while time.monotonic()<deadline:
   try:
    if run([KAGGLE,'datasets','status',DATASET],120).strip().lower()=='ready':break
   except Exception as e:write(last_error=repr(e))
   time.sleep(30)
  else:raise TimeoutError('Dataset readiness timeout')
  (WORK/'launch-attempt.json').write_text(json.dumps({'kernel':KERNEL,'time':time.time()}))
  out=run([KAGGLE,'kernels','push','-p',WORK/'kernel'],300);(WORK/'push.log').write_text(out)
  m=re.search(r'Kernel version (\d+) successfully pushed',out)
  if not m:raise RuntimeError('Unknown push response')
  write(status='SUBMITTED',kernel_version=int(m[1]),gpu_started=True,last_error=None)
  deadline=time.monotonic()+14400
  while time.monotonic()<deadline:
   status=run([KAGGLE,'kernels','status',KERNEL],120)
   if 'KernelWorkerStatus.ERROR' in status:raise RuntimeError('Remote kernel error: '+status)
   if 'KernelWorkerStatus.COMPLETE' in status:break
   write(status='RUNNING' if 'RUNNING' in status else 'SUBMITTED',monitor_status='WAITING',remote_response=status.strip())
   time.sleep(60)
  else:raise TimeoutError('Kernel timeout')
  result=WORK/'result';run([KAGGLE,'kernels','output',KERNEL,'-p',result,'--file-pattern','stage1-imperial-result.zip','--quiet'],900)
  files=list(result.rglob('stage1-imperial-result.zip'))
  if len(files)!=1:raise ValueError('No unique result zip')
  with zipfile.ZipFile(files[0]) as z:
   report=json.loads(z.read('report.json'));pred=json.loads(z.read('predictions.json'))
  if report.get('status')!='COMPLETE' or len(pred)!=200 or {r['model'] for r in pred}!={'official_best','screen_mix_25'}:raise ValueError('Result coverage failed')
  if any(not 0<=r['probability']<=1 or r['answer']!=('RERECORDED' if r['probability']>=.5 else 'ORIGINAL') for r in pred):raise ValueError('Prediction validation failed')
  validated=WORK/'validated';validated.mkdir();(validated/'report.json').write_text(json.dumps(report,indent=2));(validated/'predictions.json').write_text(json.dumps(pred,indent=2))
  write(status='COMPLETE',monitor_status='VALIDATED',completed_predictions=200,result_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),last_error=None)
 except Exception as e:write(monitor_status='FAILED',last_error=repr(e));raise
if __name__=='__main__':main()
