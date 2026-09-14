"""One local child, OS lock, 30-minute limit, zero automatic retries."""
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/stage2-spatial-motion-20260914'


def save(value):
    path = OUT/'watch-status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(value, updated=time.time()), indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'worker.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if (OUT/'protocol.json').exists():
                raise RuntimeError('Existing run found; inspect results instead of launching twice')
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONUTF8='1')
            with (OUT/'run.log').open('w', encoding='utf-8') as log:
                proc = subprocess.Popen([sys.executable, '-u', str(ROOT/'src/compare_stage2_spatial_motion.py')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                save(dict(status='RUNNING', pid=os.getpid(), child_pid=proc.pid, timeout_seconds=1800, automatic_retries=0))
                try:
                    code = proc.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
                    raise TimeoutError('Local experiment exceeded 30 minutes')
                if code:
                    raise RuntimeError(f'Child exit {code}; see run.log')
            assert json.loads((OUT/'validation.json').read_text())['status'] == 'PASS'
            assert json.loads((OUT/'status.json').read_text())['status'] == 'COMPLETE_VALIDATED'
            save(dict(status='COMPLETE_VALIDATED', pid=os.getpid(), automatic_retries=0))
        except Exception as exc:
            save(dict(status='FAILED', error=repr(exc), pid=os.getpid()))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
