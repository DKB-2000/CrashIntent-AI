"""Launch once after upload, then bounded status/download/independent validation."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
from prepare_stage1_robustness import write_json
from advance_stage1_release import tick_lock
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'artifacts/kaggle-stage1-quality-trial-20260911'
KAGGLE=ROOT/'.venv/Scripts/kaggle.exe'
KERNEL='biadis/crashintent-stage1-quality-trial'
DATASET='biadis/crashintent-stage1-quality-trial-assets'


def update(**values):
    state=json.loads((WORK/'remote-run.json').read_text()) if (WORK/'remote-run.json').exists() else {}
    state.update(values,checked_unix=time.time(),pid=os.getpid());write_json(WORK/'remote-run.json',state)


def run(args,timeout=120):
    proc=subprocess.run([str(a) for a in args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    if proc.returncode:raise RuntimeError((proc.stdout+proc.stderr)[-3000:])
    return proc.stdout+proc.stderr


def main(launch):
    with tick_lock(WORK/'watch.lock') as acquired:
        if not acquired:raise RuntimeError('Watcher already running')
        try:
            if launch:
                if (WORK/'launch-attempt.json').exists():raise RuntimeError('Launch already attempted; inspect before retry')
                deadline=time.monotonic()+3600;errors=0
                update(status='WAITING_FOR_DATASET',monitor_status='WAITING')
                while True:
                    if time.monotonic()>deadline:raise TimeoutError('Dataset readiness deadline')
                    try:ready=run([KAGGLE,'datasets','status',DATASET]).strip().lower()=='ready';errors=0
                    except Exception as error:
                        errors+=1;update(last_error=repr(error))
                        if errors>=5:raise
                        ready=False
                    if ready:break
                    time.sleep(30)
                with (WORK/'launch-attempt.json').open('x') as f:json.dump(dict(kernel=KERNEL,created_unix=time.time()),f)
                update(status='LAUNCHING',uploaded=True)
                output=run([KAGGLE,'kernels','push','-p',WORK/'kernel'],180)
                (WORK/'push.log').write_text(output,encoding='utf-8')
                match=re.search(r'Kernel version (\d+) successfully pushed',output)
                if not match:raise RuntimeError('Unknown launch response; do not repeat push')
                update(status='SUBMITTED',kernel_version=int(match[1]),gpu_started=True)
            deadline=time.monotonic()+14400;errors=0
            while time.monotonic()<deadline:
                try:status=run([KAGGLE,'kernels','status',KERNEL]);errors=0
                except Exception as error:
                    errors+=1;update(monitor_status='RETRYING_STATUS',last_error=repr(error))
                    if errors>=5:raise
                    time.sleep(60);continue
                if 'KernelWorkerStatus.COMPLETE' in status or 'KernelWorkerStatus.ERROR' in status:
                    success='KernelWorkerStatus.COMPLETE' in status
                    update(status='COMPLETE' if success else 'ERROR',monitor_status='DOWNLOADING',remote_response=status.strip(),last_error=None)
                    for attempt in range(5):
                        try:
                            run([KAGGLE,'kernels','output',KERNEL,'-p',WORK/'result','--file-pattern','stage1-quality-result.zip','--quiet'],900)
                            files=list((WORK/'result').rglob('stage1-quality-result.zip'))
                            if len(files)!=1:raise RuntimeError('No unique result zip')
                            break
                        except Exception:
                            if attempt==4:raise
                            time.sleep(30)
                    if not success:raise RuntimeError('Remote run failed; output recovered for diagnosis')
                    update(monitor_status='VALIDATING')
                    checked=WORK/'validated'
                    output=run([sys.executable,ROOT/'src/validate_stage1_quality_trial.py','--result',files[0],'--bundle-dir',WORK,'--output-dir',checked],900)
                    (WORK/'validation.log').write_text(output,encoding='utf-8')
                    result=json.loads((checked/'result-validation.json').read_text())
                    if result['status']!='PASS':raise RuntimeError('Independent validation failed')
                    update(status='COMPLETE',monitor_status='VALIDATED',validation=str(checked/'result-validation.json'),decisions=result['decisions'],last_error=None)
                    return
                running='KernelWorkerStatus.RUNNING' in status
                update(status='RUNNING' if running else 'SUBMITTED',monitor_status='WAITING',remote_response=status.strip(),gpu_started=running,last_error=None)
                time.sleep(60)
            raise TimeoutError('Four-hour monitoring deadline')
        except Exception as error:
            update(monitor_status='FAILED',last_error=repr(error));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args();main(args.launch)
