"""Launch the frozen comparison once, only after complete human review."""
import argparse
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys

from prepare_stage2_final_validation import ROOT, OUT, DATA, save, sha
from score_stage2_final_validation import read_inputs, read_csv, label_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=DATA/'review-decisions.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT)
    _, times = read_inputs()
    gate = label_gate(read_csv(args.labels), times)
    if gate['status'] != 'READY_TO_SCORE':
        raise RuntimeError('Human review/labels incomplete; inference not started')
    with (OUT/'comparison.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if (OUT/'evaluation-started.json').exists() or args.output.exists():
                raise RuntimeError('An evaluation already started; inspect it before any rerun')
            save(OUT/'evaluation-started.json', dict(labels_sha256=sha(args.labels), output=str(args.output), model_contract_sha256=sha(OUT/'evaluation-contract.json')))
            with (OUT/'comparison.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, '-u', str(ROOT/'src/predict_stage2_final_validation.py'), '--labels', str(args.labels.resolve()), '--output', str(args.output.resolve())],
                                         cwd=ROOT, env=dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'),
                                         stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                save(OUT/'comparison-watch.json', dict(status='RUNNING', pid=os.getpid(), child_pid=child.pid, timeout_seconds=7200, retries=0))
                try:
                    code = child.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                    raise TimeoutError('CPU comparison exceeded two hours')
                if code:
                    raise RuntimeError(f'Comparison exit {code}; see comparison.log')
            assert json.loads((args.output/'comparison.json').read_text())['status'] == 'COMPLETE'
            save(OUT/'comparison-watch.json', dict(status='COMPLETE', pid=os.getpid(), output=str(args.output)))
        except Exception as exc:
            save(OUT/'comparison-watch.json', dict(status='FAILED', error=repr(exc)))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
