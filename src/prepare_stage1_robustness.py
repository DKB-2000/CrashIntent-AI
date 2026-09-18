"""Paired, validation-only Stage1 quality controls and synthetic ablations.

No training data, phone holdout, remote service or model is modified.
ORIGINAL is inherited from the existing CCD proxy label, not newly verified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
import pandas as pd

import prepare_stage1_rerecorded as synth
from prepare_stage1_full_run import inspect_video

SEED = 20260910
GROUPS = {
    'moire': {'moire_amplitude': 0.},
    'band': {'band_amplitude': 0.},
    'flicker': {'flicker_amplitude': 0.},
    'geometry': {'perspective_x': 0., 'perspective_y': 0., 'scale': 1.},
    'reflection': {'reflection_amplitude': 0.},
    'color': {'gamma': 1., 'red_gain': 1., 'blue_gain': 1.},
    'blur': {'blur_sigma': 0.},
    'noise': {'noise_sigma': 0.},
    'jpeg': {'jpeg_quality': None},
    'jitter': {'temporal_jitter_probability': 0.},
}
ORIGINAL_CONDITIONS = ('original_clean', 'original_jpeg', 'original_blur',
                       'original_noise', 'original_quality_mix')
CONDITIONS = ORIGINAL_CONDITIONS + ('rerecorded_full',) + tuple(
    'rerecorded_no_' + key for key in GROUPS) + ('rerecorded_weak_screen',)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def parameters(base, condition):
    """All paired conditions share strengths; removing JPEG really skips coding."""
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition}')
    values = asdict(base)
    if condition.startswith('original_'):
        for neutral in GROUPS.values():
            values.update(neutral)
        for effect in ('jpeg', 'blur', 'noise'):
            if condition in ('original_' + effect, 'original_quality_mix'):
                for key in GROUPS[effect]:
                    values[key] = getattr(base, key)
    elif condition.startswith('rerecorded_no_'):
        values.update(GROUPS[condition.removeprefix('rerecorded_no_')])
    elif condition == 'rerecorded_weak_screen':
        # Only screen/camera-specific cues weaken; nuisance quality stays fixed.
        for effect in ('moire', 'band', 'flicker', 'geometry', 'reflection', 'color', 'jitter'):
            for key, neutral in GROUPS[effect].items():
                values[key] = neutral + .25 * (values[key] - neutral)
    return values


def transform(frame, index, fps, values, fields, rng):
    """Equivalent effect order to training, with exact bypasses for controls."""
    result = frame.copy()
    if (values['perspective_x'], values['perspective_y'], values['scale']) != (0., 0., 1.):
        result = cv2.warpPerspective(result, fields[0], (frame.shape[1], frame.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    result = result.astype(np.float32) / 255.
    time_s = index / fps
    result *= 1. + values['flicker_amplitude'] * math.sin(2 * math.pi * values['flicker_hz'] * time_s)
    phase = int(round(values['moire_phase_speed'] * index))
    result *= 1. + values['moire_amplitude'] * np.roll(fields[1], phase, axis=1)[:, :, None]
    band_center = ((time_s * values['band_speed']) % 1.4 - .2) * frame.shape[0]
    band = np.exp(-.5 * ((np.arange(frame.shape[0], dtype=np.float32) - band_center)
                        / (values['band_width'] * frame.shape[0])) ** 2)
    result *= 1. - values['band_amplitude'] * band[:, None, None]
    result += values['reflection_amplitude'] * fields[2][:, :, None]
    result[:, :, 2] *= values['red_gain']
    result[:, :, 0] *= values['blue_gain']
    result = np.clip(result, 0., 1.) ** (1. / values['gamma'])
    if values['blur_sigma'] > .2:
        result = cv2.GaussianBlur(result, (0, 0), values['blur_sigma'])
    # Consume identical random arrays even when noise is disabled.
    result += rng.normal(0., values['noise_sigma'] / 255., result.shape).astype(np.float32)
    result = np.clip(result * 255., 0, 255).astype(np.uint8)
    if values['jpeg_quality'] is not None:
        ok, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, values['jpeg_quality']])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if result is None:
            raise RuntimeError('JPEG decoding failed')
    return result


def select_sources(table, excluded, per_group):
    required = {'source_id', 'upload_group', 'split', 'variant', 'source_sha256'}
    if table.empty or not required.issubset(table.columns) or table[list(required)].isna().any().any():
        raise ValueError('Incomplete source manifest')
    if set(table.split) != {'train', 'val'} or per_group < 1:
        raise ValueError('Expected original train/val split and positive per-group count')
    for key in ('source_id', 'upload_group'):
        if table.groupby(key).split.nunique().max() != 1:
            raise ValueError(f'{key} leakage')
    if set(table.source_id) & excluded:
        raise ValueError('Phone holdout leakage')
    originals = table[(table.split == 'val') & (table.variant == 0)]
    if originals.source_id.duplicated().any():
        raise ValueError('Duplicate source control')
    selected = []
    for _, part in originals.groupby('upload_group', sort=True):
        records = part.to_dict('records')
        records.sort(key=lambda r: hashlib.sha256(f"{SEED}:robustness:{r['source_id']}".encode()).hexdigest())
        selected.extend(records[:per_group])
    if not selected:
        raise ValueError('No validation sources')
    return selected


def render_source(task):
    row, source_dir, output_dir, code_sha = task
    cv2.setNumThreads(1)
    sid = row['source_id']
    source = Path(source_dir) / (sid + '.mp4')
    out = Path(output_dir)
    source_sha = digest(source)
    if source_sha != row['source_sha256']:
        raise ValueError(f'Original source mismatch: {sid}')
    seed = int.from_bytes(hashlib.sha256(f'{SEED}:{sid}:robustness'.encode()).digest()[:4], 'big')
    base = synth._effect_parameters(random.Random(seed))
    # Match the existing JPEG/blur/noise ranges, with a nontrivial blur dose.
    base = replace(base, blur_sigma=max(.4, base.blur_sigma))
    identity = dict(source_id=sid, source_sha256=source_sha, generator_sha256=code_sha,
                    seed=seed, crf=24, base_effects=asdict(base), upload_group=row['upload_group'])
    record = out / 'records' / (sid + '.json')
    if record.exists():
        saved = json.loads(record.read_text())
        if saved['identity'] != identity:
            raise ValueError(f'Resume identity mismatch: {sid}')
        if [r['condition'] for r in saved['rows']] != list(CONDITIONS):
            raise ValueError('Incomplete condition set')
        for item in saved['rows']:
            inspect_video(out / item['output_path'], item)
        return saved['rows']
    results = []
    for condition in CONDITIONS:
        capture, width, height, fps, expected = synth._open_video(source)
        values = parameters(base, condition)
        effects = synth.Effects(**values)
        fields = (synth._perspective_matrix(width, height, effects),
                  *synth._spatial_fields(width, height, effects))
        relative = f'videos/{sid}_{condition}.mp4'
        destination = out / relative
        rng, jitter_rng = np.random.default_rng(seed), random.Random(seed)
        writer = synth.H264Writer(destination, width, height, fps, 24)
        previous, count = None, 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                # The clean control bypasses all pixel processing.
                if condition != 'original_clean':
                    frame = transform(frame, count, fps, values, fields, rng)
                    if previous is not None and jitter_rng.random() < values['temporal_jitter_probability']:
                        frame = previous.copy()
                writer.write(frame)
                previous = frame
                count += 1
        finally:
            capture.release()
            writer.close()
        if count != expected or count < 1:
            raise ValueError(f'Incomplete render: {relative}')
        item = dict(source_id=sid, upload_group=row['upload_group'], split='val',
                    condition=condition, label='ORIGINAL' if condition in ORIGINAL_CONDITIONS else 'RERECORDED',
                    label_basis='inherited_CCD_proxy' if condition in ORIGINAL_CONDITIONS else 'synthetic_proxy',
                    output_path=relative, source_sha256=source_sha, sha256=digest(destination),
                    seed=seed, crf=24, width=width, height=height, fps=fps, frames=count,
                    effects=json.dumps(values, sort_keys=True))
        inspect_video(destination, item)
        results.append(item)
    write_json(record, dict(identity=identity, rows=results))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, default=Path('artifacts/stage1-full-20260910'))
    parser.add_argument('--source-dir', type=Path, default=Path('data_raw/ccd/videos/Crash-1500'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    out = args.output_dir.resolve()
    if out == args.training_dir.resolve() or out == args.source_dir.resolve():
        raise ValueError('Separate validation output required')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'records').mkdir(exist_ok=True)
    manifest = args.training_dir / 'generation_manifest.csv'
    original_report = json.loads((args.training_dir / 'generation_report.json').read_text())
    if original_report['status'] != 'PASS' or original_report['manifest_sha256'] != digest(manifest):
        raise ValueError('Training manifest is not validated')
    table = pd.read_csv(manifest, dtype={'source_id': str, 'upload_group': str})
    excluded = set(pd.read_csv(args.training_dir / 'excluded_sources.csv', dtype=str).source_id)
    selected = select_sources(table, excluded, args.per_group)
    code_sha = hashlib.sha256(Path(__file__).read_bytes() + Path(synth.__file__).read_bytes()).hexdigest()
    plan = dict(seed=SEED, generator_sha256=code_sha, training_manifest_sha256=digest(manifest),
                per_group=args.per_group, conditions=list(CONDITIONS),
                sources=[{k: r[k] for k in ('source_id', 'upload_group', 'source_sha256')} for r in selected])
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Output directory belongs to a different plan')
    write_json(plan_path, plan)
    pd.DataFrame(selected)[['source_id', 'upload_group', 'split', 'source_sha256']].to_csv(out / 'sources.csv', index=False)
    shutil.copyfile(manifest, out / 'training_manifest.csv')
    shutil.copyfile(args.training_dir / 'excluded_sources.csv', out / 'excluded_sources.csv')
    # Snapshot the code that produced the artifacts, even if future source changes.
    (out / 'code').mkdir(exist_ok=True)
    for file in (Path(__file__), Path(synth.__file__)):
        shutil.copyfile(file, out / 'code' / file.name)
    state = dict(status='GENERATING', expected_sources=len(selected),
                 expected_videos=len(selected) * len(CONDITIONS), completed_sources=0, videos=0,
                 real_phone_validation=False, use='diagnostic_only_not_training', uploaded=False)
    started = time.monotonic()
    def save():
        state.update(seconds=time.monotonic() - started, checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(out / 'status.json', state)
    save()
    rows = []
    try:
        tasks = [(r, str(args.source_dir.resolve()), str(out), code_sha) for r in selected]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(render_source, tasks):
                rows.extend(result)
                state.update(completed_sources=state['completed_sources'] + 1, videos=len(rows))
                save()
                print(json.dumps(state), flush=True)
        pd.DataFrame(rows).to_csv(out / 'generation_manifest.csv', index=False)
        state.update(status='DATA_VALIDATED', manifest_sha256=digest(out / 'generation_manifest.csv'),
                     plan_sha256=digest(plan_path), decoded_frames=sum(r['frames'] for r in rows),
                     bytes=sum((out / r['output_path']).stat().st_size for r in rows),
                     upload_groups=len({r['upload_group'] for r in rows}), source_leakage=False)
        save()
    except Exception as error:
        state.update(status='FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
