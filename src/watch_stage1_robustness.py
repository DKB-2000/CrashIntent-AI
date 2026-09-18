"""Wait for local stress data + independently validated full model, then score once.

Uses local CPU only. Does not start/restart training, upload, submit or select models.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import os

from prepare_stage1_robustness import digest, write_json


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def ready_checkpoint(training):
    validation = read_json(training / 'validated/result-validation.json')
    checkpoint = training / 'validated/best.pt'
    if validation.get('status') != 'PASS' or not checkpoint.exists():
        return None
    if digest(checkpoint) != validation['checkpoint_sha256']:
        raise ValueError('Validated full checkpoint hash mismatch')
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--training-dir', type=Path, default=Path('artifacts/kaggle-stage1-full-20260910'))
    parser.add_argument('--max-wait-hours', type=float, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = args.dataset_dir.resolve()
    training = args.training_dir.resolve()
    if not 0 < args.max_wait_hours <= 24:
        raise ValueError('Invalid wait budget')
    state_dir = dataset / 'full-model-evaluation'
    state_dir.mkdir(parents=True, exist_ok=True)
    status_path = state_dir / 'watch-status.json'
    lock_path = state_dir / 'watch.lock'
    with lock_path.open('x') as lock:
        lock.write(str(os.getpid()))
    started = time.monotonic()
    state = dict(status='WAITING_FOR_DATA_AND_MODEL', pid=os.getpid(), device='cpu',
                 threads=2, uploaded=False, training_changed=False)
    def save(**updates):
        state.update(updates, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(status_path, state)
    try:
        save()
        while time.monotonic() - started < args.max_wait_hours * 3600:
            data_status = read_json(dataset / 'status.json')
            remote_status = read_json(training / 'remote-run.json')
            if data_status.get('status') == 'FAILED':
                raise RuntimeError('Stress generation failed; inspect dataset status.json')
            checkpoint = ready_checkpoint(training)
            if checkpoint is not None and data_status.get('status') == 'DATA_VALIDATED':
                break
            if checkpoint is None and (remote_status.get('status') == 'ERROR' or remote_status.get('monitor_status') == 'FAILED'):
                raise RuntimeError('Full-model recovery needs attention; training is not restarted')
            save(data_status=data_status.get('status'), data_videos=data_status.get('videos', 0),
                 model_validated=checkpoint is not None, training_status=remote_status.get('status'))
            time.sleep(30)
        else:
            raise TimeoutError('Wait budget expired; no training/evaluation restarted')
        attempt = state_dir / 'launch-attempt.json'
        with attempt.open('x', encoding='utf-8') as stream:
            json.dump(dict(checkpoint_sha256=digest(checkpoint), pid=os.getpid()), stream)
        save(status='EVALUATING', checkpoint_sha256=digest(checkpoint))
        command = [sys.executable, str(root / 'src/evaluate_stage1_robustness.py'),
                   '--dataset-dir', str(dataset), '--checkpoint', str(checkpoint),
                   '--output-dir', str(state_dir / 'result'), '--device', 'cpu',
                   '--threads', '2', '--max-seconds', '14400']
        with (state_dir / 'evaluation.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=15000,
                           env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'))
        report = read_json(state_dir / 'result/report.json')
        if report.get('status') != 'PASS' or report.get('scope') != 'ALL_CONDITIONS':
            raise ValueError('Evaluation did not complete all conditions')
        save(status='PASS', report=str(state_dir / 'result/report.json'),
             completed_videos=report['completed_videos'])
    except Exception as error:
        save(status='FAILED', error=repr(error))
        raise
    finally:
        lock_path.unlink()


if __name__ == '__main__':
    main()
