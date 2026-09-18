"""One resumable Windows-scheduled tick: verified Stage1 -> GPU -> registered ZIP."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import daily_submission as release
from prepare_stage1_robustness import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'COMPLETED', 'NEEDS_ATTENTION'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def optional_json(path):
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def checked_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Automation path must remain within project')
    return path


def log_command(command, path, timeout):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    with path.open('w', encoding='utf-8') as stream:
        process = subprocess.run([str(v) for v in command], cwd=ROOT, stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=timeout, env=env)
    output = path.read_text(encoding='utf-8')
    if process.returncode:
        raise RuntimeError(f'Command failed ({process.returncode}); see {path.name}: {output[-1200:]}')
    return output


def validated_inputs(config):
    training = checked_path(config['training_dir'])
    stress = checked_path(config['robustness_dir'])
    validation = optional_json(training / 'validated/result-validation.json')
    report = optional_json(stress / 'full-model-evaluation/result/report.json')
    watch = optional_json(stress / 'full-model-evaluation/watch-status.json')
    data_state = optional_json(stress / 'status.json')
    remote = optional_json(training / 'remote-run.json')
    if watch.get('status') == 'FAILED' or data_state.get('status') == 'FAILED':
        raise ValueError('Stage1 robustness preparation/evaluation failed')
    if remote.get('status') == 'ERROR' or remote.get('monitor_status') == 'FAILED':
        raise ValueError('Stage1 full-model recovery failed')
    if validation.get('status') != 'PASS' or report.get('status') != 'PASS':
        return None
    if report.get('scope') != 'ALL_CONDITIONS' or report.get('completed_videos') != config['expected_stress_videos']:
        raise ValueError('Full robustness evaluation is required; smoke result rejected')
    checkpoint = training / 'validated/best.pt'
    checkpoint_sha = digest(checkpoint)
    if checkpoint_sha != validation['checkpoint_sha256'] or checkpoint_sha != report['checkpoint_sha256']:
        raise ValueError('Model identity differs between training and robustness validation')
    if validation['optimizer_steps'] < 120 or validation['best_macro_f1'] <= validation['baseline_macro_f1']:
        raise ValueError('No validated Stage1 improvement over the same-set baseline')
    import pandas as pd
    predictions = pd.read_csv(training / 'validated/validation_predictions.csv')
    if set(predictions.answer) != {'ORIGINAL', 'RERECORDED'}:
        raise ValueError('Full-model validation still predicts only one class')
    # Recheck every stress input and independently reload saved model outputs.
    from evaluate_stage1_robustness import audit_dataset, verify_predictions, summarize
    table = audit_dataset(stress)
    if len(table) != config['expected_stress_videos'] or report['manifest_sha256'] != digest(stress / 'generation_manifest.csv'):
        raise ValueError('Stress dataset identity mismatch')
    prediction_file = stress / 'full-model-evaluation/result/predictions.jsonl'
    if digest(prediction_file) != report['predictions_sha256']:
        raise ValueError('Stress predictions changed')
    saved = [json.loads(line) for line in prediction_file.read_text().splitlines()]
    verify_predictions(table, saved)
    conditions, _ = summarize(saved)
    if conditions != report['conditions']:
        raise ValueError('Stress condition summary differs from recomputed outputs')
    return dict(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=checkpoint_sha,
                training_validation_sha256=digest(training / 'validated/result-validation.json'),
                robustness_report_sha256=digest(stress / 'full-model-evaluation/result/report.json'),
                baseline_macro_f1=validation['baseline_macro_f1'], model_macro_f1=validation['best_macro_f1'],
                robustness_precision=report['precision'])


def verify_candidate(base, candidate, checkpoint_sha):
    release.inspect_zip(base)
    candidate_sha = release.inspect_zip(candidate)
    with zipfile.ZipFile(base) as before, zipfile.ZipFile(candidate) as after:
        if hashlib.sha256(after.read('model/stage1/best.pt')).hexdigest() != checkpoint_sha:
            raise ValueError('Candidate contains the wrong Stage1 checkpoint')
        for name in before.namelist():
            if name != 'model/stage1/best.pt' and before.read(name) != after.read(name):
                raise ValueError(f'Unexpected non-Stage1 change: {name}')
    return candidate_sha


def require_pinned(config):
    for key in ('base_zip', 'fixtures', 'base_validation', 'notebook'):
        if digest(checked_path(config[key])) != config[key + '_sha256']:
            raise ValueError(f'Pinned input changed: {key}')


def advance(config, state, work, save, run=log_command):
    """Advance at most one phase; cloud mutations have write-ahead markers."""
    kaggle = ROOT / '.venv/Scripts/kaggle.exe'
    kernel = config['kernel_id']
    phase = state['status']
    if phase in TERMINAL:
        return
    if phase == 'WAITING_FOR_VALIDATION':
        inputs = validated_inputs(config)
        if inputs is None:
            save(waiting='Full training validation and all 357 robustness predictions')
            return
        require_pinned(config)
        # Refuse to replace any intervening release from another session.
        release.check_registry(config['registry_sha256'])
        candidate = work / 'candidate/submit.zip'
        if not candidate.exists():
            run([sys.executable, ROOT / 'src/build_improved_submission.py',
                 '--base', checked_path(config['base_zip']), '--stage1', checked_path(inputs['checkpoint']),
                 '--output', candidate], work / 'build.log', 600)
        candidate_sha = verify_candidate(checked_path(config['base_zip']), candidate, inputs['checkpoint_sha256'])
        dataset = work / 'dataset'
        dataset.mkdir(exist_ok=True)
        shutil.copyfile(candidate, dataset / 'candidate.bin')
        shutil.copyfile(checked_path(config['fixtures']), dataset / 'public-fixtures.bin')
        write_json(dataset / 'candidate-assets.json',
                   {'candidate.bin': candidate_sha, 'public-fixtures.bin': config['fixtures_sha256']})
        save(status='CANDIDATE_READY', inputs=inputs, candidate_sha256=candidate_sha)
    elif phase == 'CANDIDATE_READY':
        require_pinned(config)
        if digest(work / 'dataset/candidate.bin') != state['candidate_sha256']:
            raise ValueError('Upload candidate changed')
        # Persist before creating anything remotely. A lost response never repeats create.
        save(status='UPLOAD_STARTED')
        run([kaggle, 'datasets', 'create', '-p', work / 'dataset', '--quiet'], work / 'upload.log', 1200)
        save(status='WAITING_FOR_DATASET', dataset_uploaded=True)
    elif phase == 'UPLOAD_STARTED':
        raise RuntimeError('Upload outcome uncertain after interruption; inspect upload.log and remote dataset before retry')
    elif phase == 'WAITING_FOR_DATASET':
        output = run([kaggle, 'datasets', 'status', config['dataset_id']], work / 'dataset-status.log', 120)
        if output.strip().lower() != 'ready':
            save(dataset_status=output.strip()[-300:])
            return
        save(status='PUSH_STARTED')
        output = run([kaggle, 'kernels', 'push', '-p', work / 'kernel'], work / 'push.log', 180)
        match = re.search(r'Kernel version (\d+) successfully pushed', output)
        if not match:
            raise RuntimeError('Kernel push response did not identify a version; inspect before retry')
        save(status='GPU_RUNNING', kernel_version=int(match[1]), gpu_started=True)
    elif phase == 'PUSH_STARTED':
        raise RuntimeError('GPU launch outcome uncertain after interruption; no duplicate push attempted')
    elif phase == 'GPU_RUNNING':
        versioned = f"{kernel}/{state['kernel_version']}"
        output = run([kaggle, 'kernels', 'status', versioned], work / 'gpu-status.log', 120)
        if 'KernelWorkerStatus.ERROR' in output:
            raise ValueError('GPU integration failed; see remote notebook')
        if 'KernelWorkerStatus.COMPLETE' not in output:
            save(gpu_status=output.strip()[-300:])
            return
        result_dir = work / 'result'
        run([kaggle, 'kernels', 'output', versioned, '-p', result_dir,
             '--file-pattern', 'candidate-integration-result.zip', '--quiet'], work / 'download.log', 900)
        files = list(result_dir.rglob('candidate-integration-result.zip'))
        if len(files) != 1:
            raise RuntimeError('Expected one version-pinned GPU result ZIP')
        run([sys.executable, ROOT / 'src/validate_candidate_gpu_result.py',
             '--result-zip', files[0], '--candidate', work / 'candidate/submit.zip',
             '--fixtures', work / 'dataset/public-fixtures.bin', '--output', work / 'gpu-validation.json'],
            work / 'validation.log', 600)
        gpu = read_json(work / 'gpu-validation.json')
        if gpu['status'] != 'PASS' or gpu['candidate_sha256'] != state['candidate_sha256']:
            raise ValueError('GPU evidence identity mismatch')
        save(status='READY_TO_REGISTER', evidence=str(files[0].relative_to(ROOT)),
             evidence_sha256=digest(files[0]), gpu_validation_sha256=digest(work / 'gpu-validation.json'))
    elif phase == 'READY_TO_REGISTER':
        require_pinned(config)
        candidate = work / 'candidate/submit.zip'
        evidence = checked_path(state['evidence'])
        if (verify_candidate(checked_path(config['base_zip']), candidate, state['inputs']['checkpoint_sha256'])
                != state['candidate_sha256'] or digest(evidence) != state['evidence_sha256']
                or digest(work / 'gpu-validation.json') != state['gpu_validation_sha256']):
            raise ValueError('Candidate/evidence changed before registration')
        # Verify the dependent validation reports have not changed since packaging.
        training = checked_path(config['training_dir'])
        stress = checked_path(config['robustness_dir'])
        if (digest(training / 'validated/result-validation.json') != state['inputs']['training_validation_sha256']
                or digest(stress / 'full-model-evaluation/result/report.json') != state['inputs']['robustness_report_sha256']):
            raise ValueError('Stage1 validation evidence changed')
        notes = ('Stage1 balanced-v2 full model; synthetic grouped validation and 357-video robustness diagnosis complete. '
                 'Stage2/3 preserved from the pinned GPU-validated base candidate. GPU integration PASS; '
                 'real-phone and official score improvement not certified.')
        result = release.register_and_publish(candidate, evidence, notes, config['registry_sha256'])
        if result['sha256'] != state['candidate_sha256']:
            raise ValueError('Published release identity mismatch')
        write_json(work / 'release-report.json', result)
        save(status='COMPLETED', zip=result['zip'], submitted=False)
    else:
        raise ValueError(f'Unknown automation phase: {phase}')


@contextmanager
def tick_lock(path):
    """OS lock is automatically released on process exit/reboot."""
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path('artifacts/stage1-auto-release-20260910'))
    args = parser.parse_args()
    work = checked_path(args.work_dir)
    with tick_lock(work / 'tick.lock') as acquired:
        if not acquired:
            return
        config = read_json(work / 'config.json')
        state = optional_json(work / 'status.json') or {'status': 'WAITING_FOR_VALIDATION'}
        if state['status'] in TERMINAL:
            return
        if state.get('activation_status') == 'BLOCKED_BY_AUTOMATIC_APPROVAL_REVIEW':
            print('Activation blocked by automatic approval review; no work started', flush=True)
            return 1
        def save(**values):
            state.update(values, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(work / 'status.json', state)
        try:
            advance(config, state, work, save)
            save()
        except Exception as error:
            # Read-only status/download failures may retry; cloud mutation uncertainty cannot.
            retryable = state['status'] in {'WAITING_FOR_DATASET', 'GPU_RUNNING'} and isinstance(
                error, (RuntimeError, subprocess.TimeoutExpired))
            failures = state.get('consecutive_errors', 0) + 1
            save(last_error=repr(error), consecutive_errors=failures,
                 status=state['status'] if retryable and failures < 15 else 'NEEDS_ATTENTION')
            print(json.dumps(state), flush=True)
            return 1
        else:
            save(consecutive_errors=0)
        print(json.dumps(state), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
