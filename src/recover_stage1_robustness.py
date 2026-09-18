"""Resume verified stopped CPU evaluation, with bounded native-crash recovery."""
import ctypes,json,os,subprocess,sys,datetime,time
from pathlib import Path
from prepare_stage1_robustness import write_json
from advance_stage1_release import tick_lock
ROOT=Path(__file__).resolve().parents[1]
stress=ROOT/'artifacts/stage1-robustness-20260910'
work=stress/'full-model-evaluation'
state_path=work/'watch-status.json'
def main():
    prior=json.loads(state_path.read_text())
    old=json.loads((work/'result/status.json').read_text())
    if prior['status']!='FAILED' or old['status']!='FAILED':
        raise RuntimeError('Expected stopped failed evaluation')
    if not any(reason in old.get('error','') for reason in ('Evaluation budget reached','Native process exited')):
        raise RuntimeError('Only diagnosed timeout/native crash may resume')
    write_json(work/('watch-before-recovery-'+str(time.time_ns())+'.json'),prior)
    command=[sys.executable,str(ROOT/'src/evaluate_stage1_robustness.py'),
             '--dataset-dir',str(stress),'--checkpoint',str(ROOT/'artifacts/kaggle-stage1-full-20260910/validated/best.pt'),
             '--output-dir',str(work/'result'),'--resume','--device','cpu','--threads','1','--max-seconds','14400']
    state=dict(prior,status='EVALUATING',pid=os.getpid(),threads=1,model_validated=True,
               recovery='verified prefix resume; single CPU thread')
    state.pop('error',None)
    write_json(state_path,state)
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for attempt in range(3):
            before=json.loads((work/'result/status.json').read_text())['completed_videos']
            log_path=work/('recovery-evaluation-'+str(time.time_ns())+'.log')
            state.update(attempt=attempt+1,log=str(log_path))
            write_json(state_path,state)
            with log_path.open('x',encoding='utf-8') as log:
                process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                       env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
                                                PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            if process.returncode==0:
                break
            child=json.loads((work/'result/status.json').read_text())
            if process.returncode in (3221225477,-1073741819):
                write_json(work/('native-exit-'+str(time.time_ns())+'.json'),child)
                child.update(status='FAILED',error='Native process exited: '+str(process.returncode))
                write_json(work/'result/status.json',child)
                if child['completed_videos']>before and attempt<2:
                    continue
            process.check_returncode()
        report=json.loads((work/'result/report.json').read_text())
        if report['status']!='PASS' or report['completed_videos']!=357:
            raise RuntimeError('Recovery did not finish full evaluation')
        state.update(status='PASS',completed_videos=357,
                     checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(state_path,state)
        release=ROOT/'artifacts/stage1-auto-release-20260910'
        with tick_lock(release/'tick.lock') as acquired:
            if not acquired:
                raise RuntimeError('Release tick busy; evaluation PASS, gate needs reopening')
            state2=json.loads((release/'status.json').read_text())
            if (state2['status']!='NEEDS_ATTENTION' or state2.get('gpu_started') or state2.get('dataset_uploaded')
                    or 'robustness preparation/evaluation failed' not in state2.get('last_error','')):
                raise RuntimeError('Release gate changed; refusing unrelated reset')
            write_json(release/'status-before-recovery.json',state2)
            state2.update(status='WAITING_FOR_VALIDATION',consecutive_errors=0,recovery='robustness interruption recovered')
            state2.pop('last_error',None)
            write_json(release/'status.json',state2)
    except Exception as error:
        state.update(status='FAILED',error=repr(error),checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(state_path,state)
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':
    main()
