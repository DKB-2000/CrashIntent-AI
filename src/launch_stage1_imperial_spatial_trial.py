"""Launch once, monitor, recover and validate the Imperial spatial trial."""
import hashlib,json,os,re,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; WORK=ROOT/'artifacts/kaggle-stage1-imperial-spatial-20260917'; KAGGLE=ROOT/'.venv/Scripts/kaggle.exe'
KERNEL='biadis/crashintent-stage1-imperial-spatial-trial'; DATASET='biadis/crashintent-stage1-imperial-spatial-assets'
def write(**values):
 path=WORK/'remote-run.json'; data=json.loads(path.read_text()) if path.exists() else {}; data.update(values,checked_unix=time.time(),pid=os.getpid()); temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(data,indent=2)); os.replace(temporary,path)
def run(args,timeout):
 process=subprocess.run([str(value) for value in args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'))
 if process.returncode: raise RuntimeError((process.stdout+process.stderr)[-4000:])
 return process.stdout+process.stderr
def main():
 try:
  marker=WORK/'launch-attempt.json'
  if marker.exists(): raise RuntimeError('Launch already attempted')
  write(status='WAITING_FOR_DATASET',monitor_status='ACTIVE')
  deadline=time.monotonic()+3600
  while time.monotonic()<deadline:
   try:
    if run([KAGGLE,'datasets','status',DATASET],120).strip().lower()=='ready': break
   except Exception as error: write(last_error=repr(error))
   time.sleep(30)
  else: raise TimeoutError('Dataset readiness timeout')
  marker.write_text(json.dumps({'kernel':KERNEL,'time':time.time()})); write(status='SUBMITTING',monitor_status='ACTIVE',last_error=None)
  pushed=run([KAGGLE,'kernels','push','-p',WORK/'kernel'],300); (WORK/'push.log').write_text(pushed)
  match=re.search(r'Kernel version (\d+) successfully pushed',pushed)
  if not match: raise RuntimeError('Unknown push response')
  write(status='SUBMITTED',monitor_status='WAITING',kernel_version=int(match.group(1)),gpu_started=True,last_error=None)
  deadline=time.monotonic()+14400
  while time.monotonic()<deadline:
   status=run([KAGGLE,'kernels','status',KERNEL],120)
   if 'KernelWorkerStatus.ERROR' in status:
    error_dir=WORK/'error-output'; error_dir.mkdir(exist_ok=True); run([KAGGLE,'kernels','output',KERNEL,'-p',error_dir,'--quiet'],900)
    raise RuntimeError('Remote kernel error: '+status)
   if 'KernelWorkerStatus.COMPLETE' in status: break
   write(status='RUNNING' if 'RUNNING' in status else 'SUBMITTED',monitor_status='WAITING',remote_response=status.strip()); time.sleep(60)
  else: raise TimeoutError('Kernel timeout')
  result=WORK/'result'; run([KAGGLE,'kernels','output',KERNEL,'-p',result,'--file-pattern','stage1-imperial-spatial-result.zip','--quiet'],900)
  files=list(result.rglob('stage1-imperial-spatial-result.zip'))
  if len(files)!=1: raise ValueError('No unique result ZIP')
  with zipfile.ZipFile(files[0]) as archive:
   report=json.loads(archive.read('report.json')); predictions=json.loads(archive.read('predictions.json'))
   checkpoint=archive.read('spatial.pt')
  if report.get('status')!='COMPLETE' or len(predictions)!=20: raise ValueError('Coverage validation failed')
  keys={row['image'] for row in predictions}
  if len(keys)!=20: raise ValueError('Prediction identity validation failed')
  if any(not 0<=row['probability']<=1 or row['answer']!=int(row['probability']>=.5) for row in predictions): raise ValueError('Prediction value validation failed')
  validated=WORK/'validated'; validated.mkdir(exist_ok=True); (validated/'report.json').write_text(json.dumps(report,indent=2)); (validated/'predictions.json').write_text(json.dumps(predictions,indent=2)); (validated/'spatial.pt').write_bytes(checkpoint)
  write(status='COMPLETE',monitor_status='VALIDATED',completed_predictions=20,result_sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),last_error=None)
 except Exception as error: write(monitor_status='FAILED',last_error=repr(error)); raise
if __name__=='__main__': main()
