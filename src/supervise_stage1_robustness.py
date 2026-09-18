"""Local scheduled Stage1 supervisor. No network or competition submission."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from advance_stage1_release import tick_lock, read_json
from prepare_stage1_robustness import write_json

ROOT = Path(__file__).resolve().parents[1]
STRESS = ROOT / 'artifacts/stage1-robustness-20260910'
WORK = STRESS / 'full-model-evaluation'
RESULT = WORK / 'result'
CONTROL = WORK / 'supervisor-status.json'
STALE_SECONDS = 900
MAX_FAILURES = 5


def processes():
    script = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | ForEach-Object { [pscustomobject]@{pid=$_.ProcessId; parent=$_.ParentProcessId; command=$_.CommandLine; created=$_.CreationDate.ToUniversalTime().Ticks.ToString()} } | ConvertTo-Json -Compress"
    raw = subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script], text=True, timeout=30)
    value = json.loads(raw) if raw.strip() else []
    return value if isinstance(value, list) else [value]


def owned_evaluators(snapshot):
    return [p for p in snapshot if str(ROOT / 'src/evaluate_stage1_robustness.py').lower() in (p['command'] or '').lower()
            and str(RESULT).lower() in (p['command'] or '').lower()]


def stop_stalled(snapshot, workers):
    """Only the exact evaluator and its demonstrated recovery ancestry may stop."""
    selected = {p['pid']: p for p in workers}
    by_pid = {p['pid']: p for p in snapshot}
    for worker in workers:
        parent = by_pid.get(worker['parent'])
        while parent and (parent['pid'] in selected or 'src/recover_stage1_robustness.py' in (parent['command'] or '').replace('\\', '/')):
            selected[parent['pid']] = parent
            parent = by_pid.get(parent['parent'])
    # Refuse a tree containing unrelated Python work; do not use tree-wide termination.
    if any(p['parent'] in selected and p['pid'] not in selected for p in snapshot):
        raise RuntimeError('Unexpected child of stalled evaluation; manual inspection required')
    def depth(item):
        seen = set()
        while item['parent'] in selected and item['parent'] not in seen:
            seen.add(item['parent'])
            item = selected[item['parent']]
        return len(seen)
    ordered = sorted(selected.values(), key=lambda p: (p['pid'] in {w['pid'] for w in workers}, -depth(p)))
    for item in ordered:
        # Recheck process creation and command immediately before stopping (PID reuse guard).
        script = f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {int(item['pid'])}'; if ($p) {{ if ($p.CreationDate.ToUniversalTime().Ticks.ToString() -ne '{item['created']}') {{ throw 'PID identity changed' }}; Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }}"
        subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script], check=True, timeout=30)
    if owned_evaluators(processes()):
        raise RuntimeError('Evaluator still present after stop')


def charge_failure(control, completed, reason):
    control['failures'] = control.get('failures', 0) + 1
    same = control.get('last_failure_completed') == completed
    control['same_point_failures'] = control.get('same_point_failures', 0) + 1 if same else 1
    control.update(last_failure_completed=completed, last_error=reason)
    if control['failures'] >= MAX_FAILURES or control['same_point_failures'] >= 2:
        control['status'] = 'NEEDS_ATTENTION'
    write_json(CONTROL, control)
    return control.get('status') != 'NEEDS_ATTENTION'


def mark_interrupted(reason):
    state = read_json(RESULT / 'status.json')
    write_json(WORK / f'interruption-{time.time_ns()}.json', state)
    state.update(status='FAILED', error=reason)
    write_json(RESULT / 'status.json', state)
    return state


def complete(control):
    # A completed report must also pass the release's independent evidence audit.
    from advance_stage1_release import validated_inputs
    release = ROOT / 'artifacts/stage1-auto-release-20260910'
    report = read_json(RESULT / 'report.json')
    if report.get('status') != 'PASS' or report.get('completed_videos') != 357 or report.get('scope') != 'ALL_CONDITIONS':
        raise ValueError('Full evaluation PASS required')
    old_watch = read_json(WORK / 'watch-status.json')
    write_json(WORK / 'watch-status.json', dict(old_watch, status='PASS', completed_videos=357))
    try:
        if validated_inputs(read_json(release / 'config.json')) is None:
            raise ValueError('Release input evidence is incomplete')
    except Exception:
        write_json(WORK / 'watch-status.json', old_watch)
        raise
    with tick_lock(release / 'tick.lock') as acquired:
        if not acquired:
            control['status'] = 'WAITING_FOR_RELEASE_LOCK'
            write_json(CONTROL, control)
            return
        state = read_json(release / 'status.json')
        if (state.get('status') == 'NEEDS_ATTENTION' and not state.get('gpu_started') and not state.get('dataset_uploaded')
                and 'robustness preparation/evaluation failed' in state.get('last_error', '')):
            write_json(release / f'status-before-supervisor-{time.time_ns()}.json', state)
            state.update(status='WAITING_FOR_VALIDATION', consecutive_errors=0, recovery='verified local evaluation recovered')
            state.pop('last_error', None)
            write_json(release / 'status.json', state)
        # Any unrelated gate is preserved. The existing 12-hour release task owns advancement.
    control.update(status='COMPLETED', completed_videos=357, finished_unix=time.time())
    write_json(CONTROL, control)


def stop_child(child):
    # Windows venv Python is a redirector with a real interpreter child.
    # Keep Popen's process handle open while terminating that specific tree.
    if child.poll() is None:
        subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'], check=True, timeout=30, stdout=subprocess.DEVNULL)
        child.wait(timeout=30)


def run_worker(control):
    chunk_size = control.get('max_new_videos', 25)
    if type(chunk_size) is not int or not 1 <= chunk_size <= 25:
        raise ValueError('Chunk size must be an integer from 1 to 25')
    command = [sys.executable, str(ROOT / 'src/evaluate_stage1_robustness.py'), '--dataset-dir', str(STRESS),
               '--checkpoint', str(ROOT / 'artifacts/kaggle-stage1-full-20260910/validated/best.pt'),
               '--output-dir', str(RESULT), '--resume', '--device', 'cpu', '--threads', '1',
               '--max-new-videos', str(chunk_size), '--max-seconds', '1800']
    log_path = WORK / f'chunk-{time.time_ns()}.log'
    with log_path.open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONFAULTHANDLER='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        control.update(status='EVALUATING', worker_pid=child.pid, log=str(log_path), started_unix=time.time())
        write_json(CONTROL, control)
        last_change = time.monotonic()
        stamp = (RESULT / 'status.json').stat().st_mtime_ns
        try:
            while child.poll() is None:
                time.sleep(10)
                current = (RESULT / 'status.json').stat().st_mtime_ns
                if current != stamp:
                    stamp, last_change = current, time.monotonic()
                if time.monotonic() - last_change > STALE_SECONDS:
                    stop_child(child)
                    return 'Worker made no progress for 15 minutes'
            if child.returncode:
                return f'Worker exited: {child.returncode}'
            return None
        finally:
            if child.poll() is None:
                stop_child(child)


def tick():
    control = read_json(CONTROL) if CONTROL.exists() else dict(status='MONITORING', failures=0)
    if control['status'] in ('COMPLETED', 'NEEDS_ATTENTION'):
        return
    snapshot = processes()
    workers = owned_evaluators(snapshot)
    if workers:
        age = time.time() - (RESULT / 'status.json').stat().st_mtime
        control.update(status='MONITORING_EXISTING', observed_pids=[p['pid'] for p in workers], checked_unix=time.time())
        write_json(CONTROL, control)
        if age <= STALE_SECONDS:
            return
        stop_stalled(snapshot, workers)
    elif any('recover_stage1_robustness.py' in (p['command'] or '') for p in snapshot):
        # Legacy recovery may be between workers or opening the release gate.
        return
    state = read_json(RESULT / 'status.json')
    if state['status'] == 'PASS':
        complete(control)
        return
    if state['status'] == 'RUNNING':
        state = mark_interrupted('Supervisor detected stopped/stalled evaluator')
        if not charge_failure(control, state['completed_videos'], state['error']):
            return
    if state['status'] not in ('FAILED', 'PAUSED'):
        raise ValueError('Unexpected evaluation state')
    # Full data, model/code identity, ordered prefix and saved logits are checked by each worker.
    while True:
        error = run_worker(control)
        state = read_json(RESULT / 'status.json')
        if error:
            state = mark_interrupted(error)
            # Python validation errors must not be retried as a native crash.
            recoverable = ('exited: 3221225477' in error or 'exited: -1073741819' in error or 'no progress' in error)
            if not recoverable:
                control.update(status='NEEDS_ATTENTION', last_error=error)
                write_json(CONTROL, control)
                return
            if not charge_failure(control, state['completed_videos'], error):
                return
            continue
        if state['status'] == 'PASS':
            complete(control)
            return
        if state['status'] != 'PAUSED':
            raise ValueError('Worker returned success without PAUSED/PASS')
        control.update(completed_videos=state['completed_videos'], chunks=control.get('chunks', 0) + 1)
        write_json(CONTROL, control)


def main():
    with tick_lock(WORK / 'supervisor.lock') as acquired:
        if not acquired:
            return
        awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            tick()
        except Exception as error:
            control = read_json(CONTROL) if CONTROL.exists() else {}
            control.update(status='NEEDS_ATTENTION', last_error=repr(error), checked_unix=time.time())
            write_json(CONTROL, control)
            raise
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__ == '__main__':
    main()
