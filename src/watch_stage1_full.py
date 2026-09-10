"""Watch an already-approved Stage1 run; download and validate its output only."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/kaggle-stage1-full-20260910'
KERNEL = 'biadis/crashintent-stage1-full-v1'
KAGGLE = ROOT / '.venv/Scripts/kaggle.exe'
STATUS = BUNDLE / 'remote-run.json'


def update(**values):
    record = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    record.update(values, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(STATUS)
    print(json.dumps(values), flush=True)


def run(arguments, timeout=120):
    process = subprocess.run([str(a) for a in arguments], cwd=ROOT, capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=timeout,
                             env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'))
    if process.returncode:
        raise RuntimeError((process.stdout + process.stderr)[-4000:])
    return process.stdout + process.stderr


def main():
    started = time.monotonic()
    errors = 0
    while time.monotonic() - started < 43200:
        try:
            state = run([KAGGLE, 'kernels', 'status', KERNEL])
        except Exception as error:
            errors += 1
            update(monitor_status='RETRYING_STATUS', last_error=repr(error))
            if errors >= 5:
                raise
            time.sleep(45)
            continue
        errors = 0
        if 'KernelWorkerStatus.COMPLETE' in state or 'KernelWorkerStatus.ERROR' in state:
            update(status='COMPLETE' if 'KernelWorkerStatus.COMPLETE' in state else 'ERROR', monitor_status='DOWNLOADING')
            result_dir = BUNDLE / 'result'
            for attempt in range(5):
                run([KAGGLE, 'kernels', 'output', KERNEL, '-p', result_dir,
                     '--file-pattern', 'stage1-training-result.zip', '--quiet'], timeout=900)
                files = list(result_dir.rglob('stage1-training-result.zip'))
                if len(files) == 1:
                    break
                if attempt == 4:
                    raise RuntimeError('No unique result ZIP after kernel completion')
                time.sleep(30)
            update(monitor_status='VALIDATING', result=str(files[0].relative_to(ROOT)))
            checked = BUNDLE / 'validated'
            output = run([sys.executable, ROOT / 'src/validate_stage1_training_result.py',
                          '--result', files[0], '--bundle-dir', BUNDLE, '--output-dir', checked], timeout=600)
            (BUNDLE / 'validation.log').write_text(output, encoding='utf-8')
            validation = json.loads((checked / 'result-validation.json').read_text())
            update(monitor_status='VALIDATED', validation=validation)
            return
        update(status='RUNNING', monitor_status='WAITING')
        time.sleep(45)
    raise TimeoutError('Monitoring deadline reached; remote status must be checked before any retry')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        update(monitor_status='FAILED', last_error=repr(error))
        raise
