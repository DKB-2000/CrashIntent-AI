"""Local development ablation; never consumes Nexar validation candidates."""
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import stage2_pipeline as p
from analyze_stage2_source_cv import details, summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage2-spatial-motion-20260914'
CV = ROOT / 'artifacts/stage2-source-cv-20260910'
CACHE = ROOT / 'artifacts/stage2-controls-20260909/features.pt'
SEEDS = [20260825, 20260826, 20260827]
MODES = ['point', 'motion', 'spatial', 'combined']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def regions(gray, flow):
    """16 regions x (gray mean/std, flow mean x/y/magnitude); full field of view."""
    values = []
    for y in range(4):
        for x in range(4):
            g = gray[y*24:(y+1)*24, x*40:(x+1)*40].astype(np.float32)/255
            f = flow[y*24:(y+1)*24, x*40:(x+1)*40]
            values.extend([g.mean(), g.std(), f[..., 0].mean()/160,
                           f[..., 1].mean()/96, np.linalg.norm(f, axis=2).mean()/160])
    return torch.tensor(values, dtype=torch.float32)


def extract(row):
    path = ROOT / 'data_raw/ccd/videos/Crash-1500' / (row['ID']+'.mp4')
    cap = cv2.VideoCapture(str(path))
    seq, previous = [], None
    started = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if time.monotonic()-started > 60:
                raise TimeoutError('Video extraction exceeded 60 seconds')
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 96), interpolation=cv2.INTER_AREA)
            flow = np.zeros((96, 160, 2), np.float32) if previous is None else cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            seq.append(regions(gray, flow))
            previous = gray
    finally:
        cap.release()
    assert len(seq) == row['frames']
    result = torch.stack(seq)
    assert torch.isfinite(result).all()
    return result, sha(path)


def auxiliary(seq, ci, ei):
    return torch.cat([seq[ci], seq[ei]])


def mode_mask(mode):
    kinds = torch.arange(160) % 5
    return {'point': torch.zeros(160), 'motion': (kinds >= 2).float(),
            'spatial': (kinds < 2).float(), 'combined': torch.ones(160)}[mode]


def head_from(scene):
    head = copy.deepcopy(scene)
    first = nn.Linear(928, 192)
    with torch.no_grad():
        first.weight.zero_()
        first.weight[:, :768].copy_(scene[0].weight)
        first.bias.copy_(scene[0].bias)
    head[0] = first
    return head


def infer(head, rows, inputs, positions):
    head.eval()
    with torch.no_grad():
        logits = head(torch.stack([inputs[r['ID']] for r in rows]))
    pred = [dict(ID=r['ID'], collision_frame=positions[r['ID']][0], entry_frame=positions[r['ID']][1],
                 evasion_space=int(logits[i, :2].argmax()), entry_side=['LEFT', 'RIGHT'][int(logits[i, 2:].argmax())]) for i, r in enumerate(rows)]
    return pred, logits


def run():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    started = time.monotonic()
    old_protocol = json.loads((CV/'protocol.json').read_text())
    assert json.loads((CV/'independent-validation.json').read_text())['status'] == 'PASS'
    assert sha(CACHE) == old_protocol['features_sha256']
    features = torch.load(CACHE, weights_only=True, map_location='cpu')
    rows = []
    splits = []
    protected = {str(CACHE.relative_to(ROOT)): sha(CACHE)}
    for fold in range(5):
        pair = []
        for split in ['train', 'validation']:
            path = CV/f'fold-{fold}/{split}.csv'
            protected[str(path.relative_to(ROOT))] = sha(path)
            pair.append(pd.read_csv(path, dtype={'ID': str, 'source_id': str}).to_dict('records'))
        splits.append(pair)
        rows.extend(pair[1])
        path = CV/f'fold-{fold}/soft_mixed_15.pt'
        protected[str(path.relative_to(ROOT))] = sha(path)
    assert len(rows) == len({r['ID'] for r in rows}) == 66
    assert {r['ID'] for r in rows} == set(features)
    assert len({r['source_id'] for r in rows}) == 23
    for rel in ['data/stage2/labels_manual.csv', 'data_raw/stage2-validation-nexar-20260914/review-contact-queue.csv',
                'artifacts/stage2-controls-20260909/soft_mixed_scene/best.pt']:
        path = ROOT/rel
        if path.exists():
            protected[rel] = sha(path)
    protocol = dict(modes=MODES, seeds=SEEDS, epochs=15, lr=2e-4, batch_size=8,
                    folds=old_protocol['folds'], protected_sha256=protected,
                    script_sha256=sha(Path(__file__)), selection='Fixed settings; no outer-fold epoch selection',
                    input='Full frame resized to 160x96 (aspect slightly distorted); 4x4 regions: gray mean/std and adjacent-frame flow x/y/magnitude; at predicted collision and entry only',
                    normalization='Train-fold auxiliary vectors only; population std clamped at 1e-6',
                    control='Same 928->192->4 head and zero-initialized auxiliary columns in all four modes; identical batches/dropout seeds',
                    scope='Existing 66 CCD development labels only; Nexar never decoded or predicted; no new labels, no deployment')
    save('protocol.json', protocol)
    spatial, video_hashes = {}, {}
    for i, row in enumerate(rows):
        spatial[row['ID']], video_hashes[row['ID']] = extract(row)
        save('status.json', dict(status='EXTRACTING', completed=i+1, total=66))
    torch.save(spatial, OUT/'region-features.pt')
    save('video-hashes.json', video_hashes)
    collected = {f'{mode}_{seed}': [] for mode in MODES for seed in SEEDS}
    all_rows = []
    reference = pd.read_csv(CV/'soft_mixed_15_oof.csv', dtype={'ID': str}).set_index('ID')
    for fold, (train, val) in enumerate(splits):
        assert not {r['source_id'] for r in train} & {r['source_id'] for r in val}
        assert {r['source_id'] for r in val} == set(protocol['folds'][fold])
        model = p.Stage2Temporal()
        model.load_state_dict(torch.load(CV/f'fold-{fold}/soft_mixed_15.pt', weights_only=True)['model'])
        model.eval()
        point, aux, positions = {}, {}, {}
        with torch.no_grad():
            for r in train+val:
                cl, el, h = model.logits(features[r['ID']][None])
                ci, ei = int(cl.argmax()), int(el.argmax())
                positions[r['ID']] = (ci, ei)
                point[r['ID']] = torch.cat([h[0, ci], h[0, ei]])
                aux[r['ID']] = auxiliary(spatial[r['ID']], ci, ei)
        stacked = torch.stack([aux[r['ID']] for r in train])
        mean, std = stacked.mean(0), stacked.std(0, unbiased=False).clamp_min(1e-6)
        inputs = {mode: {r['ID']: torch.cat([point[r['ID']], (aux[r['ID']]-mean)/std*mode_mask(mode)]) for r in train+val} for mode in MODES}
        folder = OUT/f'fold-{fold}'
        folder.mkdir()
        torch.save(dict(mean=mean, std=std, positions=positions, inputs=inputs), folder/'inputs.pt')
        before, _ = infer(head_from(model.scene), val, inputs['point'], positions)
        for q in before:
            assert all(q[k] == reference.loc[q['ID'], 'pred_'+k] for k in p.OUTPUT_COLUMNS[1:])
        all_rows.extend([dict(r, fold=fold) for r in val])
        for seed in SEEDS:
            for mode in MODES:
                head = head_from(model.scene)
                torch.manual_seed(seed)
                rng = random.Random(seed)
                optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4)
                for epoch in range(15):
                    head.train()
                    order = list(train)
                    rng.shuffle(order)
                    for offset in range(0, len(order), 8):
                        batch = order[offset:offset+8]
                        y = head(torch.stack([inputs[mode][r['ID']] for r in batch]))
                        loss = F.cross_entropy(y[:, :2], torch.tensor([r['evasion_space'] for r in batch])) + F.cross_entropy(y[:, 2:], torch.tensor([int(r['entry_side']=='RIGHT') for r in batch]))
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                pred, logits = infer(head, val, inputs[mode], positions)
                name = f'{mode}_{seed}'
                torch.save(dict(model=head.state_dict(), logits=logits), folder/(name+'.pt'))
                collected[name].extend(pred)
        save('status.json', dict(status='TRAINING', folds_completed=fold+1, total_folds=5))
        print('fold', fold+1, 'seconds', round(time.monotonic()-started, 1), flush=True)
    results = {}
    for name, pred in collected.items():
        data = details(all_rows, pred)
        pd.DataFrame(data).to_csv(OUT/(name+'_oof.csv'), index=False)
        results[name] = summary(all_rows, pred)
    save('report.json', dict(status='TRAINED_PENDING_VALIDATION', seconds=time.monotonic()-started, results=results))
    verify()


def verify():
    protocol = json.loads((OUT/'protocol.json').read_text())
    for rel, digest in protocol['protected_sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    assert sha(Path(__file__)) == protocol['script_sha256']
    spatial = torch.load(OUT/'region-features.pt', weights_only=True)
    hashes = json.loads((OUT/'video-hashes.json').read_text())
    features = torch.load(CACHE, weights_only=True)
    report = json.loads((OUT/'report.json').read_text())
    seen, max_error = set(), 0.
    for fold in range(5):
        train = pd.read_csv(CV/f'fold-{fold}/train.csv', dtype={'ID': str, 'source_id': str}).to_dict('records')
        val = pd.read_csv(CV/f'fold-{fold}/validation.csv', dtype={'ID': str, 'source_id': str}).to_dict('records')
        assert not {r['source_id'] for r in train} & {r['source_id'] for r in val}
        assert not seen & {r['ID'] for r in val}
        seen.update(r['ID'] for r in val)
        for r in val:
            regenerated, digest = extract(r)
            assert digest == hashes[r['ID']]
            assert torch.equal(regenerated, spatial[r['ID']])
        model = p.Stage2Temporal()
        model.load_state_dict(torch.load(CV/f'fold-{fold}/soft_mixed_15.pt', weights_only=True)['model'])
        model.eval()
        points, raw, pos = {}, {}, {}
        with torch.no_grad():
            for r in train+val:
                cl, el, h = model.logits(features[r['ID']][None])
                ci, ei = int(cl.argmax()), int(el.argmax())
                pos[r['ID']] = (ci, ei)
                points[r['ID']] = torch.cat([h[0, ci], h[0, ei]])
                raw[r['ID']] = torch.cat([spatial[r['ID']][ci], spatial[r['ID']][ei]])
        a = torch.stack([raw[r['ID']] for r in train]).numpy()
        mean = torch.from_numpy(a.mean(0))
        std = torch.from_numpy(np.maximum(a.std(0), 1e-6))
        stored = torch.load(OUT/f'fold-{fold}/inputs.pt', weights_only=True)
        assert torch.allclose(mean, stored['mean'], atol=1e-6)
        assert torch.allclose(std, stored['std'], atol=1e-6)
        for mode in MODES:
            columns = [i for i in range(160) if mode == 'combined' or (mode == 'motion' and i%5>=2) or (mode == 'spatial' and i%5<2)]
            xs = {}
            for r in val:
                aux = torch.zeros(160)
                normalized = (raw[r['ID']]-stored['mean'])/stored['std']
                aux[columns] = normalized[columns]
                xs[r['ID']] = torch.cat([points[r['ID']], aux])
                assert torch.equal(xs[r['ID']], stored['inputs'][mode][r['ID']])
            for seed in SEEDS:
                name = f'{mode}_{seed}'
                checkpoint = torch.load(OUT/f'fold-{fold}/{name}.pt', weights_only=True)
                head = head_from(model.scene)
                head.load_state_dict(checkpoint['model'], strict=True)
                pred, logits = infer(head, val, xs, pos)
                error = float((logits-checkpoint['logits']).abs().max())
                assert error == 0
                max_error = max(max_error, error)
                frame = pd.read_csv(OUT/(name+'_oof.csv'), dtype={'ID': str}).set_index('ID')
                for q in pred:
                    assert all(q[k] == frame.loc[q['ID'], 'pred_'+k] for k in p.OUTPUT_COLUMNS[1:])
    assert seen == set(features) == set(spatial) == set(hashes)
    for name, expected in report['results'].items():
        d = pd.read_csv(OUT/(name+'_oof.csv'), dtype={'ID': str, 'source_id': str})
        assert d.ID.is_unique and set(d.ID) == seen
        vals = []
        for key, target in [('collision_frame', 't_collision'), ('entry_frame', 't_entry')]:
            v = ((d['pred_'+key]-d[target]).abs()/d.fps <= .3+1e-9).astype(float)
            assert abs(v.mean()-expected[key+'_within_0.3s']) < 1e-12
            vals.append(v)
        for key in ['evasion_space', 'entry_side']:
            v = (d['pred_'+key] == d[key]).astype(float)
            assert abs(v.mean()-expected[key+'_accuracy']) < 1e-12
            vals.append(v)
        scores = sum(vals)/4
        assert abs(scores.mean()-expected['development_mean']) < 1e-12
        assert abs(scores.groupby(d.source_id).mean().mean()-expected['source_equal_mean']) < 1e-12
    averages = {mode: {key: float(np.mean([report['results'][f'{mode}_{seed}'][key] for seed in SEEDS])) for key in ['evasion_space_accuracy', 'entry_side_accuracy', 'development_mean', 'source_equal_mean']} for mode in MODES}
    save('validation.json', dict(status='PASS', videos_redecoded=66, saved_heads_replayed=60, max_logit_error=max_error,
                                 checks=['source exclusion and OOF coverage', 'all original video hashes and region features regenerated', 'train-only normalization independently recalculated', 'all head logits and CSV predictions replayed', 'four metrics and source equal mean independently recalculated', 'protected input files unchanged']))
    save('summary.json', dict(status='COMPLETE_VALIDATED', means_over_seeds=averages, scope='Reused development set; no final generalization claim', deployment_changed=False))
    save('status.json', dict(status='COMPLETE_VALIDATED', models=60, videos=66))
    print(json.dumps(averages), flush=True)


if __name__ == '__main__':
    run()
