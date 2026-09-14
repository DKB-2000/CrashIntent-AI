"""Bounded local conversion watcher; no network, fitting, or model prediction."""
import json
import msvcrt
import os
import subprocess
import sys
import time
from prepare_stage2_final_validation import ROOT, OUT, DATA, save


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'worker.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if DATA.exists() or (OUT/'protocol.json').exists():
                raise RuntimeError('Existing inputs/run found; do not duplicate conversion')
            env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (OUT/'prepare.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, '-u', str(ROOT/'src/prepare_stage2_final_validation.py')],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                save(OUT/'watch-status.json', dict(status='RUNNING', pid=os.getpid(), child_pid=child.pid, timeout_seconds=3600, retries=0))
                try:
                    code = child.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                    raise TimeoutError('Preparation exceeded one hour')
                if code:
                    raise RuntimeError(f'Preparation exit {code}; see prepare.log')
            assert json.loads((OUT/'input-validation.json').read_text())['status'] == 'PASS'
            save(OUT/'watch-status.json', dict(status='COMPLETE', pid=os.getpid(), updated=time.time()))
        except Exception as exc:
            save(OUT/'watch-status.json', dict(status='FAILED', error=repr(exc), pid=os.getpid()))
            save(OUT/'status.json', dict(status='FAILED', error=repr(exc)))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
