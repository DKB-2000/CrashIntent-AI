"""Read-only model diagnosis on external development routes, with resumable CPU inference.

All interventions retain the endpoint image. Perturbed clips are sensitivity probes,
not newly labelled driving events. No training or submission model changes.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_stage3_errors import DATA, RUNS, metrics

OUT = Path('artifacts/stage3-motion-route-20260910')
MODES = ('normal', 'shuffle_history', 'reverse_history', 'repeat_last')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def joined_data():
    manifest = pd.read_csv(DATA/'split_manifest.csv')
    assert manifest.groupby('route').split.nunique().eq(1).all()
    labels = pd.read_csv(DATA/'labels_validation_candidate.csv')
    truth = labels.merge(manifest[['ID', 'route', 'video']], on='ID', validate='many_to_one')
    truth = truth.rename(columns={'frame_index': 'sample_index'}).sort_values(['ID', 'sample_index'])
    for head in ('accel', 'steer'):
        truth[head+'_stable'] = False
        for _, g in truth.groupby('ID'):
            y = g[head+'_label'].to_numpy()
            starts = np.r_[True, y[1:] != y[:-1]]
            age = np.arange(len(g)) - np.maximum.accumulate(np.where(starts, np.arange(len(g)), 0))
            truth.loc[g.index, head+'_stable'] = age >= 15
    signals = pd.read_csv(DATA/'signals_and_candidates.csv')
    signals = signals[['ID', 'sample_index', 'speed_mps', 'steering_angle_deg', 'pose_yaw_left_rad_s']]
    truth = truth.merge(signals, on=['ID', 'sample_index'], validate='one_to_one')
    version = json.loads((DATA/'label_version.json').read_text())
    truth['angle_v1'] = truth.steering_angle_deg - version['steer_offset_deg']
    truth['angle_band'] = pd.cut(truth.angle_v1.abs(), [0, 1, 2, 5, np.inf],
                                labels=['0-1deg', '1-2deg_boundary', '2-5deg', '>5deg'], include_lowest=True).astype(str)
    truth['speed_band'] = pd.cut(truth.speed_mps, [-np.inf, 5, 15, 25, np.inf],
                                labels=['<=5mps', '5-15mps', '15-25mps', '>25mps']).astype(str)
    truth['pose_band'] = pd.cut(truth.pose_yaw_left_rad_s.abs(), [0, .003, .012, np.inf],
                               labels=['near_straight', 'weak_turn', 'strong_turn'], include_lowest=True).astype(str)
    frames, provenance = {}, {}
    for tag, directory in RUNS.items():
        run = Path('artifacts')/directory
        validation = json.loads((run/'result-validation.json').read_text())
        archive = run/'result/stage3-route-trial-result.zip'
        sha = digest(archive)
        assert validation['status'] == 'PASS' and sha == validation['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            pred = pd.read_csv(z.open('trial-run/validation_predictions.csv')).rename(
                columns={'accel_label': 'accel_pred', 'steer_label': 'steer_pred'})
            report = json.loads(z.read('trial-run/report.json'))
        assert digest(DATA/'labels_validation_candidate.csv') == report['data_sha256']['labels_validation_candidate.csv']
        f = truth.merge(pred, on=['ID', 'sample_index'], validate='one_to_one')
        assert len(f) == len(pred) == len(truth) == 18603
        assert abs(metrics(f, 'steer')['macro_f1'] - validation['metrics']['validation']['moving_steer']) < 1e-12
        frames[tag] = f
        provenance[tag] = {'archive': str(archive), 'sha256': sha}
    return frames, provenance


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    frames, provenance = joined_data()
    rows, classes = [], []
    for tag, f in frames.items():
        for dimension in ('all', 'steer_stable', 'angle_band', 'speed_band', 'pose_band'):
            groups = f.groupby('route') if dimension == 'all' else f.groupby(['route', dimension])
            for key, g in groups:
                route, value = (key, 'all') if dimension == 'all' else key
                m = metrics(g, 'steer')
                rows.append(dict(model=tag, route=route, dimension=dimension, value=str(value),
                                 rows=m['rows'], accuracy=m['accuracy'], macro_f1=m['macro_f1']))
                for c in m['classes']:
                    classes.append(dict(model=tag, route=route, dimension=dimension, value=str(value), **c))
    pd.DataFrame(rows).to_csv(out/'route-strata.csv', index=False)
    pd.DataFrame(classes).to_csv(out/'route-strata-classes.csv', index=False)
    # Every video, including counts and class confusion: route failure concentration.
    videos = []
    for sid, g in frames['1000'].groupby('ID'):
        for c in metrics(g, 'steer')['classes']:
            videos.append(dict(ID=sid, route=g.route.iloc[0], **c))
    pd.DataFrame(videos).to_csv(out/'video-classes.csv', index=False)
    # Select before inspecting intervention outputs, equally across route/class/stability.
    # Two endpoints per stratum, prefer different videos, nonoverlapping within a video.
    f = frames['1000']
    pool = f[(f.accel_label != 'STOPPED') & (f.sample_index >= 15)]
    rng = np.random.default_rng(20260910)
    selected, used = [], {}
    for _, group in pool.groupby(['route', 'steer_label', 'steer_stable'], sort=True):
        candidates = group.iloc[rng.permutation(len(group))]
        picked, seen = [], set()
        for distinct in (True, False):
            for idx, row in candidates.iterrows():
                if len(picked) == 2:
                    break
                if distinct and row.ID in seen:
                    continue
                if any(abs(int(row.sample_index)-t) < 16 for t in used.get(row.ID, [])):
                    continue
                picked.append(idx)
                seen.add(row.ID)
                used.setdefault(row.ID, []).append(int(row.sample_index))
        selected.extend(picked)
    sample = pool.loc[selected].sort_values(['ID', 'sample_index']).reset_index(drop=True)
    sample.insert(0, 'probe_id', np.arange(len(sample)))
    sample.to_csv(out/'samples.csv', index=False)
    with zipfile.ZipFile(provenance['1000']['archive']) as z:
        checkpoint = z.read('trial-run/best.pt')
    (out/'diagnostic-model.pt').write_bytes(checkpoint)
    write_json(out/'provenance.json', dict(sources=provenance,
        labels_sha256=digest(DATA/'labels_validation_candidate.csv'),
        signals_sha256=digest(DATA/'signals_and_candidates.csv'),
        manifest_sha256=digest(DATA/'split_manifest.csv'),
        checkpoint_sha256=digest(out/'diagnostic-model.pt'), samples_sha256=digest(out/'samples.csv'),
        seed=20260910, samples=len(sample), modes=MODES,
        selection='Up to 2 per validation route x moving steer class x causal stability; no selection on model correctness',
        scope='Reused external proxy-labelled development routes; no hidden competition data'))
    print('PREPARED', len(sample), 'probes', flush=True)


def time_indices(mode, probe_id):
    if mode == 'normal':
        return np.arange(16)
    if mode == 'repeat_last':
        return np.full(16, 15)
    if mode == 'reverse_history':
        return np.r_[np.arange(14, -1, -1), 15]
    if mode == 'shuffle_history':
        return np.r_[np.random.default_rng(20260910+int(probe_id)).permutation(15), 15]
    raise ValueError(mode)


def run(out, limit, threads):
    import torch
    import stage3_pipeline as p
    torch.set_num_threads(threads)
    p.cv2.setNumThreads(1)
    provenance = json.loads((out/'provenance.json').read_text())
    assert digest(out/'samples.csv') == provenance['samples_sha256']
    assert digest(out/'diagnostic-model.pt') == provenance['checkpoint_sha256']
    samples = pd.read_csv(out/'samples.csv')
    model = p.load_model(out/'diagnostic-model.pt', torch.device('cpu'))
    result = out/'probe-logits.json'
    signature = dict(script_sha256=digest(__file__), pipeline_sha256=digest(p.__file__),
                     samples_sha256=provenance['samples_sha256'], checkpoint_sha256=provenance['checkpoint_sha256'])
    saved = json.loads(result.read_text()) if result.exists() else dict(signature=signature, rows=[],
                       runtime=dict(torch=torch.__version__, device='cpu', precision='float32', threads=threads))
    assert saved['signature'] == signature, 'Inputs/code changed; use a new output directory'
    done = {(r['probe_id'], r['mode']) for r in saved['rows']}
    start = time.monotonic()
    count = 0
    for sid, group in samples.groupby('ID', sort=True):
        if all((int(r.probe_id), mode) in done for r in group.itertuples() for mode in MODES):
            continue
        path = (DATA/group.video.iloc[0]).resolve()
        frames = list(p.video_frames(path))
        video_sha = digest(path)
        for row in group.itertuples():
            indices = p.causal_indices(int(row.sample_index))
            original = [frames[i] for i in indices]
            for mode in MODES:
                if (int(row.probe_id), mode) in done:
                    continue
                order = time_indices(mode, row.probe_id)
                x = p.clip_tensor([original[i] for i in order]).unsqueeze(0)
                before = time.monotonic()
                with torch.inference_mode():
                    a, s = model(x)
                assert torch.isfinite(a).all() and torch.isfinite(s).all()
                saved['rows'].append(dict(probe_id=int(row.probe_id), mode=mode, ID=sid,
                    sample_index=int(row.sample_index), video_sha256=video_sha,
                    input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                    temporal_indices=order.tolist(), accel_logits=a[0].tolist(), steer_logits=s[0].tolist(),
                    seconds=time.monotonic()-before))
                write_json(result, saved)
                count += 1
                print(f'{len(saved["rows"])}/{len(samples)*len(MODES)} {sid} {row.sample_index} {mode} {time.monotonic()-before:.2f}s', flush=True)
                write_json(out/'status.json', dict(status='RUNNING', completed=len(saved['rows']),
                           expected=len(samples)*len(MODES), session_seconds=time.monotonic()-start))
                if limit and count >= limit:
                    return
        del frames
    write_json(out/'status.json', dict(status='INFERENCE_COMPLETE', completed=len(saved['rows']), expected=len(samples)*len(MODES)))


def summarize(out):
    from analyze_stage3_errors import ACCEL, STEER
    samples = pd.read_csv(out/'samples.csv')
    raw = json.loads((out/'probe-logits.json').read_text())
    logits = pd.DataFrame(raw['rows'])
    assert not logits[['probe_id', 'mode']].duplicated().any()
    assert len(logits) == len(samples)*len(MODES)
    output = []
    for mode in MODES:
        f = samples.merge(logits[logits['mode'] == mode], on=['probe_id', 'ID', 'sample_index'], validate='one_to_one')
        ref = samples.merge(logits[logits['mode'] == 'normal'], on=['probe_id', 'ID', 'sample_index'], validate='one_to_one')
        for head, names in [('accel', ACCEL), ('steer', STEER)]:
            z = np.array(f[head+'_logits'].tolist()); rz = np.array(ref[head+'_logits'].tolist())
            def softmax(v):
                e = np.exp(v-v.max(axis=1, keepdims=True))
                return e/e.sum(axis=1, keepdims=True)
            prob, rp = softmax(z), softmax(rz)
            pred = np.array(names)[z.argmax(1)]; reference = np.array(names)[rz.argmax(1)]
            f[head+'_probe_pred'] = pred
            f[head+'_changed'] = pred != reference
            f[head+'_tv'] = np.abs(prob-rp).sum(1)/2
            f[head+'_true_probability_delta'] = prob[np.arange(len(f)), f[head+'_label'].map({n:i for i,n in enumerate(names)})] - rp[np.arange(len(f)), f[head+'_label'].map({n:i for i,n in enumerate(names)})]
            for dimension in ('all', 'route', 'steer_stable', 'steer_label'):
                groups = [('all', f)] if dimension == 'all' else f.groupby(dimension)
                for key, g in groups:
                    output.append(dict(mode=mode, head=head, dimension=dimension, value=str(key), rows=len(g),
                        accuracy=float((g[head+'_probe_pred']==g[head+'_label']).mean()),
                        changed_rate=float(g[head+'_changed'].mean()), mean_probability_tv=float(g[head+'_tv'].mean()),
                        mean_true_probability_delta=float(g[head+'_true_probability_delta'].mean())))
        f.to_csv(out/f'probes-{mode}.csv', index=False)
    table = pd.DataFrame(output)
    table.to_csv(out/'motion-summary.csv', index=False)
    base = pd.read_csv(out/'probes-normal.csv')
    agreement = {h:float((base[h+'_probe_pred']==base[h+'_pred']).mean()) for h in ('accel','steer')}
    write_json(out/'summary.json', dict(status='PASS', samples=len(samples), forward_passes=len(logits),
        baseline_saved_gpu_class_agreement=agreement,
        overall=table[table.dimension=='all'].to_dict('records'),
        limitations=['Stratified 48-or-fewer clips are diagnostic, not prevalence-weighted performance.',
                     'FP32 CPU compared with saved CUDA inference; report normal class agreement.',
                     'Interventions preserve endpoint but can create out-of-distribution clips.',
                     'Sensitivity does not prove physically correct motion reasoning; invariance does not prove no motion features.',
                     'Correlated frames and only four reused development routes; no statistical independence claim.']))
    write_json(out/'status.json', dict(status='COMPLETE', completed=len(logits), expected=len(logits)))
    print(table[table.dimension=='all'].to_string(index=False))
    print('NORMAL_GPU_AGREEMENT', agreement)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output_dir)
    elif args.command == 'run': run(args.output_dir, args.limit, args.threads)
    else: summarize(args.output_dir)
