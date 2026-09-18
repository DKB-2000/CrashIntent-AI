"""Frozen route CV: center crop versus full-frame aspect-preserving letterbox."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'threads;1')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import compare_stage3_representations as base
import compare_stage3_route_cv as reference

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'artifacts/stage3-fullframe-cv-20260914'
REF = PROJECT / 'artifacts/stage3-route-cv-20260911'
SEQ = PROJECT / 'artifacts/stage3-sequence-cv-20260911'
SEEDS = [20260910, 20260911, 20260912]
ARMS = ['baseline', 'full_frame']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def full_frame(bgr):
    """Retain all pixels geometrically; fit RGB into a 96-square zero canvas."""
    h, w = bgr.shape[:2]
    scale = 96 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((96, 96, 3), np.uint8)
    x, y = (96 - nw) // 2, (96 - nh) // 2
    out[y:y+nh, x:x+nw] = resized
    return out


def flow_stats(a, b):
    flow = np.zeros((96, 96, 2), np.float32) if np.array_equal(a, b) else cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
    return np.r_[cv2.resize(flow, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1),
                 np.percentile(np.linalg.norm(flow, axis=2), [10, 50, 90, 99]), flow.mean((0, 1))].astype(np.float32)


def summary(seq):
    assert seq.shape[1:] == (15, 134) and np.isfinite(seq).all()
    return np.concatenate([seq[:, -1], seq[:, -5:].mean(1), seq.mean(1)], 1)


def extract_video(path, ends):
    needed = {i for t in ends for i in range(int(t)-15, int(t)+1)}
    assert min(needed) >= 0
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-10) < .01
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert max(needed) < total
    flows = {arm: np.zeros((total, 134), np.float32) for arm in ARMS}
    previous, previous_index = {}, -2
    try:
        for i in range(total):
            assert cap.grab(), f'Incomplete decode {path} {i}'
            if i not in needed:
                continue
            ok, bgr = cap.retrieve()
            assert ok
            center = cv2.resize(base.p.prepare_frame(bgr).transpose(1, 2, 0), (96, 96), interpolation=cv2.INTER_AREA)
            images = {'baseline': center, 'full_frame': full_frame(bgr)}
            for arm, rgb in images.items():
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if previous_index == i-1:
                    flows[arm][i] = flow_stats(previous[arm], gray)
                previous[arm] = gray
            previous_index = i
    finally:
        cap.release()
    return {arm: np.stack([v[int(t)-14:int(t)+1] for t in ends]) for arm, v in flows.items()}


def check_plan():
    p = json.loads((ROOT / 'plan.json').read_text())
    for path, expected in p['inputs'].items():
        assert sha(PROJECT / path) == expected, path
    return p


def prepare():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / 'plan.json').exists(), 'Existing plan: resume worker instead'
    (ROOT / 'cache').mkdir(exist_ok=True)
    f = pd.read_csv(REF / 'samples.csv')
    assert len(f) == 1000 and f.route.nunique() == 17 and f.ID.nunique() == 156
    validation = json.loads((REF / 'independent-validation.json').read_text())
    assert validation['status'] == 'PASS' and sha(REF / 'results.json') == validation['results_sha256']
    assert sha(REF / 'summary.json') == validation['summary_sha256']
    shutil.copy2(REF / 'samples.csv', ROOT / 'samples.csv')
    inputs = {}
    paths = [Path(__file__), Path(base.__file__), Path(base.p.__file__), Path(reference.__file__),
             REF / 'samples.csv', REF / 'results.json', REF / 'summary.json', REF / 'independent-validation.json',
             PROJECT / 'artifacts/stage3-motion-submit-candidate-20260910/submit.zip']
    for fold in range(5):
        paths.append(REF / f'fold-{fold}-batches.npy')
        for seed in SEEDS:
            paths.extend((REF / f'{fold}-{seed}-baseline' / name) for name in ('model.pt', 'logits.npz', 'report.json'))
    for path in paths:
        inputs[str(path.resolve().relative_to(PROJECT))] = sha(path)
    save('plan.json', dict(status='PREDECLARED', inputs=inputs, seeds=SEEDS, arms=ARMS,
                          folds=5, updates=1200, batch=10, lr=.001, weight_decay=0, threads=2,
                          architecture='Same 1682->128->64 dual-head MLP; image1280 masked zero; motion402 active',
                          intervention='Full original RGB aspect-fit into96x96 zero letterbox vs established resize256 center224 then96; flow/grid/time windows unchanged',
                          confounds='Pixel scale and letterbox boundaries change with field of view. This is a practical input-preprocessing comparison, not an isolated causal estimate of peripheral pixels.',
                          normalization='Fold training rows only; std floor.01; clip10',
                          selection='Final1200 updates/all3 seeds, no hyperparameter search; require all3 OOF steer deltas positive, >=4/5 fold-mean improvements, no class recall mean loss beyond2pp to consider follow-up',
                          baseline='Replay15 saved models; retrain fold0 seed20260910 and require exact weights/logits before training full_frame',
                          scope='Existing balanced1000clips/17routes, same held-out route folds; no old4development routes, no human labels; not official or independent new-domain evidence'))
    save('status.json', dict(status='PREPARED', total=156, completed=0))


def extract():
    check_plan()
    f = pd.read_csv(ROOT / 'samples.csv')
    for n, (sid, group) in enumerate(f.groupby('ID'), 1):
        path = (PROJECT / base.DATA / group.video.iloc[0]).resolve()
        assert path.is_relative_to(PROJECT)
        original = PROJECT / base.ROOT / 'cache' / (sid+'.npz')
        metadata = json.loads(original.with_suffix('.json').read_text())
        assert sha(path) == metadata['video_sha256'] and sha(original) == metadata['features_sha256']
        with np.load(original) as z:
            ends = z['endpoints'].copy()
            expected = z['features'][:, 1280:].copy()
        assert ends.tolist() == sorted(group.endpoint.tolist())
        cache = ROOT / 'cache' / (sid+'.npz')
        info = cache.with_suffix('.json')
        if cache.exists() and info.exists():
            prior = json.loads(info.read_text())
            assert prior['sha256'] == sha(cache) and prior['video_sha256'] == sha(path)
        else:
            seqs = extract_video(path, ends)
            np.testing.assert_array_equal(summary(seqs['baseline']), expected)
            # Independently extracted sequence cache also verifies each of15 history pairs.
            with np.load(SEQ / 'cache' / (sid+'.npz')) as prior:
                np.testing.assert_array_equal(seqs['baseline'], prior['sequences'])
            np.savez_compressed(cache, endpoints=ends, baseline=summary(seqs['baseline']),
                                full_frame=summary(seqs['full_frame']))
            info.write_text(json.dumps(dict(sha256=sha(cache), video_sha256=sha(path),
                                             original_sha256=sha(original))), encoding='utf-8')
        save('status.json', dict(status='EXTRACTING', completed=n, total=156, pid=os.getpid()))
        print('features', n, sid, flush=True)
    save('status.json', dict(status='FEATURES_COMPLETE', completed=156, total=156))


def arrays(f):
    result = {a: np.zeros((len(f), 1682), np.float32) for a in ARMS}
    for sid, group in f.groupby('ID'):
        with np.load(ROOT / 'cache' / (sid+'.npz')) as z:
            mapping = {int(t): i for i, t in enumerate(z['endpoints'])}
            for idx, row in group.iterrows():
                for arm in ARMS:
                    result[arm][idx, 1280:] = z[arm][mapping[int(row.endpoint)]]
    return result


def fit(x, f, fold, seed):
    ti = np.flatnonzero(f.fold.to_numpy() != fold)
    vi = np.flatnonzero(f.fold.to_numpy() == fold)
    mean, std = x[ti].mean(0), x[ti].std(0).clip(.01)
    xt = torch.from_numpy(np.clip((x[ti]-mean)/std, -10, 10))
    ta, ts = torch.tensor(f.iloc[ti].accel.to_numpy()), torch.tensor(f.iloc[ti].steer.to_numpy())
    batches = np.load(REF / f'fold-{fold}-batches.npy')
    assert batches.shape == (1200, 10) and np.all(f.iloc[ti].group.to_numpy()[batches] == np.arange(10))
    torch.manual_seed(seed)
    model = base.Control()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    for ids in batches:
        a, s = model(xt[ids])
        moving = ta[ids] != 3
        loss = F.cross_entropy(a, ta[ids]) + F.cross_entropy(s[moving], ts[ids][moving])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    return dict(model=model.state_dict(), mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                train_indices=ti.tolist(), validation_indices=vi.tolist(), fold=fold, seed=seed, steps=1200)


def replay(ck, x, vi):
    model = base.Control()
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    v = np.clip((x[vi]-ck['mean'].numpy())/ck['std'].numpy(), -10, 10)
    v[:, :1280] = 0
    with torch.inference_mode():
        a, s = model(torch.from_numpy(v))
    return a.numpy(), s.numpy()


def train():
    check_plan()
    f = pd.read_csv(ROOT / 'samples.csv')
    x = arrays(f)
    started = time.monotonic()
    # A single matched baseline retraining gate catches platform/protocol drift.
    control = fit(x['baseline'], f, 0, SEEDS[0])
    saved = torch.load(REF / f'0-{SEEDS[0]}-baseline/model.pt', weights_only=True)
    assert all(torch.equal(v, saved['model'][k]) for k, v in control['model'].items()), 'Baseline retraining drift'
    vi = np.array(control['validation_indices'])
    a, s = replay(control, x['baseline'], vi)
    with np.load(REF / f'0-{SEEDS[0]}-baseline/logits.npz') as z:
        np.testing.assert_array_equal(a, z['accel'])
        np.testing.assert_array_equal(s, z['steer'])
    save('baseline-retraining.json', dict(status='PASS', fold=0, seed=SEEDS[0], weights_exact=True, logits_exact=True))
    runs = []
    for fold in range(5):
        for seed in SEEDS:
            out = ROOT / f'{fold}-{seed}-full_frame'
            out.mkdir(exist_ok=True)
            ck = fit(x['full_frame'], f, fold, seed)
            vi = np.array(ck['validation_indices'])
            a, s = replay(ck, x['full_frame'], vi)
            assert np.isfinite(a).all() and np.isfinite(s).all()
            torch.save(ck, out / 'model.pt')
            np.savez_compressed(out / 'logits.npz', accel=a, steer=s, indices=vi)
            r = dict(fold=fold, seed=seed, arm='full_frame', checkpoint_sha256=sha(out/'model.pt'),
                     metrics=reference.score(f.iloc[vi], a.argmax(1), s.argmax(1)))
            (out / 'report.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
            runs.append(r)
            save('status.json', dict(status='TRAINING', completed=len(runs), total=15, pid=os.getpid()))
            print('trained', len(runs), fold, seed, flush=True)
    save('results.json', dict(runs=runs, seconds=time.monotonic()-started))


def verify():
    plan = check_plan()
    f = pd.read_csv(ROOT / 'samples.csv')
    assert sha(ROOT/'samples.csv') == sha(REF/'samples.csv')
    for sid, g in f.groupby('ID'):
        cache = ROOT / 'cache' / (sid+'.npz')
        meta = json.loads(cache.with_suffix('.json').read_text())
        assert sha(cache) == meta['sha256']
        assert sha((PROJECT/base.DATA/g.video.iloc[0]).resolve()) == meta['video_sha256']
        assert sha(PROJECT/base.ROOT/'cache'/(sid+'.npz')) == meta['original_sha256']
    x = arrays(f)
    runs = json.loads((ROOT/'results.json').read_text())['runs']
    assert len(runs) == 15
    oof = {(seed, arm): np.full((len(f), 2), -1, int) for seed in SEEDS for arm in ARMS}
    fold_results = []
    for fold in range(5):
        ti, vi = np.flatnonzero(f.fold.to_numpy()!=fold), np.flatnonzero(f.fold.to_numpy()==fold)
        assert set(f.iloc[ti].route).isdisjoint(f.iloc[vi].route)
        for seed in SEEDS:
            for arm in ARMS:
                out = (REF if arm == 'baseline' else ROOT) / f'{fold}-{seed}-{arm}'
                ck = torch.load(out/'model.pt', weights_only=True)
                r = json.loads((out/'report.json').read_text())
                assert sha(out/'model.pt') == r['checkpoint_sha256']
                np.testing.assert_array_equal(ck['train_indices'], ti)
                np.testing.assert_array_equal(ck['validation_indices'], vi)
                np.testing.assert_array_equal(ck['mean'].numpy()[1280:], x[arm][ti].mean(0)[1280:])
                np.testing.assert_array_equal(ck['std'].numpy()[1280:], x[arm][ti].std(0).clip(.01)[1280:])
                a, s = replay(ck, x[arm], vi)
                with np.load(out/'logits.npz') as z:
                    np.testing.assert_array_equal(a, z['accel'])
                    np.testing.assert_array_equal(s, z['steer'])
                    np.testing.assert_array_equal(vi, z['indices'])
                metrics = reference.score(f.iloc[vi], a.argmax(1), s.argmax(1))
                assert metrics == r['metrics']
                fold_results.append(dict(fold=fold, seed=seed, arm=arm, **metrics))
                oof[seed, arm][vi] = np.stack([a.argmax(1), s.argmax(1)], 1)
    pooled, routes = [], []
    for (seed, arm), pred in oof.items():
        assert np.all(pred >= 0)
        metrics = reference.score(f, pred[:, 0], pred[:, 1])
        pooled.append(dict(seed=seed, arm=arm, **metrics))
        output = f[['ID','endpoint','route','fold','accel','steer']].copy()
        output['accel_pred'], output['steer_pred'] = pred[:, 0], pred[:, 1]
        output.to_csv(ROOT/f'oof-{seed}-{arm}.csv', index=False)
        # Standard-library CSV independent confusion and F1 recomputation.
        cm = [[0]*3 for _ in range(3)]
        with (ROOT/f'oof-{seed}-{arm}.csv').open() as stream:
            for row in csv.DictReader(stream):
                if int(row['accel']) != 3:
                    cm[int(row['steer'])][int(row['steer_pred'])] += 1
        assert cm == metrics['steer']['confusion']
        f1 = [2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) if sum(cm[i])+sum(row[i] for row in cm) else 0 for i in range(3)]
        assert abs(sum(f1)/3 - metrics['steer']['macro_f1']) < 1e-12
        for route, g in f.groupby('route'):
            m = reference.score(g, pred[g.index, 0], pred[g.index, 1])
            routes.append(dict(seed=seed, arm=arm, route=route, steer_f1=m['steer']['macro_f1'], recall=m['steer']['recall']))
    averages = {arm: dict(steer_f1=float(np.mean([r['steer']['macro_f1'] for r in pooled if r['arm']==arm])),
                         accel_f1=float(np.mean([r['accel']['macro_f1'] for r in pooled if r['arm']==arm])),
                         steer_recall=np.mean([r['steer']['recall'] for r in pooled if r['arm']==arm], axis=0).tolist()) for arm in ARMS}
    delta = [next(r['steer']['macro_f1'] for r in pooled if r['seed']==seed and r['arm']=='full_frame')-
             next(r['steer']['macro_f1'] for r in pooled if r['seed']==seed and r['arm']=='baseline') for seed in SEEDS]
    fold_delta = [float(np.mean([r['steer']['macro_f1'] for r in fold_results if r['fold']==fold and r['arm']=='full_frame'])-
                        np.mean([r['steer']['macro_f1'] for r in fold_results if r['fold']==fold and r['arm']=='baseline'])) for fold in range(5)]
    recall_delta = np.array(averages['full_frame']['steer_recall'])-averages['baseline']['steer_recall']
    followup = all(d>0 for d in delta) and sum(d>0 for d in fold_delta)>=4 and np.all(recall_delta>=-.02)
    save('summary.json', dict(scope=plan['scope'], averages=averages, seed_steer_deltas=delta,
                             fold_mean_steer_deltas=fold_delta, recall_delta=recall_delta.tolist(),
                             decision='FOLLOWUP_CANDIDATE_ONLY' if followup else 'KEEP_BASELINE',
                             pooled_oof=pooled, fold_metrics=fold_results, routes=routes))
    save('independent-validation.json', dict(status='PASS', models=30, new_models=15, baseline_retrained=1,
                                            source_videos=156, samples=1000, max_logits_error=0,
                                            checks='Source hashes, center feature replay, fold/train-only normalization, batch protocol, every checkpoint/logit, independent CSV steer confusion/F1',
                                            summary_sha256=sha(ROOT/'summary.json'), results_sha256=sha(ROOT/'results.json')))
    save('status.json', dict(status='COMPLETE_VALIDATED', models=30, decision='FOLLOWUP_CANDIDATE_ONLY' if followup else 'KEEP_BASELINE'))
    print(json.dumps(averages, indent=2), flush=True)


def watch():
    import msvcrt
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'watch.lock').open('a+b') as lock:
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        assert not (ROOT/'independent-validation.json').exists(), 'Already completed'
        save('watch-status.json', dict(status='RUNNING', pid=os.getpid(), timeout_per_phase=3600, retries=0))
        try:
            for phase in ['extract', 'train', 'verify']:
                with (ROOT/f'{phase}.log').open('w', encoding='utf-8') as stream:
                    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), phase],
                                            cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT)
                    save('watch-status.json', dict(status='RUNNING', pid=os.getpid(), child_pid=proc.pid, phase=phase, timeout_per_phase=3600, retries=0))
                    try:
                        rc = proc.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                           capture_output=True, timeout=30, check=True)
                        else:
                            proc.kill()
                        proc.wait(timeout=30)
                        raise
                    if rc:
                        raise RuntimeError(f'{phase} exit {rc}; see {phase}.log')
            save('watch-status.json', dict(status='COMPLETE_VALIDATED', pid=os.getpid()))
        except Exception as exc:
            save('watch-status.json', dict(status='FAILED', error=repr(exc)))
            save('status.json', dict(status='FAILED', error=repr(exc)))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare','extract','train','verify','watch'])
    args = p.parse_args()
    os.chdir(PROJECT)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    globals()[args.command]()
