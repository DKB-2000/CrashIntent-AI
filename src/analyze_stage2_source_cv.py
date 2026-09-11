"""Stage2 cached-feature source-group CV and error audit; no remote jobs or serving edits."""
import collections
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import stage2_pipeline as p
from train_stage2_controls import metrics, predictions, temporal_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage2-source-cv-20260910'
BASE = ROOT / 'artifacts/stage2-controls-20260909'
SPLIT = ROOT / 'artifacts/stage2-manual-200-20260907'


def write(name, data):
    target = OUT / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def details(rows, pred):
    assert [r['ID'] for r in rows] == [r['ID'] for r in pred]
    output = []
    for r, q in zip(rows, pred):
        d = dict(r, **{f'pred_{k}': v for k, v in q.items() if k != 'ID'})
        for name, target in [('collision_frame', 't_collision'), ('entry_frame', 't_entry')]:
            err = (q[name] - r[target]) / r['fps']
            d[name + '_signed_seconds'] = err
            d[name + '_ok'] = abs(err) <= .3 + 1e-9
        for name in ['evasion_space', 'entry_side']:
            d[name + '_ok'] = r[name] == q[name]
        d['pred_entry_after_collision'] = q['entry_frame'] > q['collision_frame']
        output.append(d)
    return output


def summary(rows, pred):
    result = metrics(rows, pred)
    result['samples'] = len(rows)
    result['sources'] = len({r['source_id'] for r in rows})
    result['by_source'] = {}
    for source in sorted({r['source_id'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['source_id'] == source]
        result['by_source'][source] = dict(n=len(indices), **metrics([rows[i] for i in indices], [pred[i] for i in indices]))
    result['source_equal_mean'] = float(np.mean([m['development_mean'] for m in result['by_source'].values()]))
    result['confusion'] = {}
    for key, labels in [('evasion_space', [0, 1]), ('entry_side', ['LEFT', 'RIGHT'])]:
        result['confusion'][key] = {'labels': labels, 'rows_truth_columns_prediction': [[sum(r[key] == a and q[key] == b for r, q in zip(rows, pred)) for b in labels] for a in labels]}
    return result


def main():
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    known = json.loads((BASE / 'independent-validation.json').read_text())
    inputs = {}
    split = {}
    for name in ['train', 'validation']:
        path = SPLIT / f'labels_{name}.csv'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == known['split_sha256'][name]
        inputs[name] = digest
        split[name] = pd.read_csv(path, dtype={'ID': str, 'source_id': str}).to_dict('records')
    digest = hashlib.sha256((BASE / 'features.pt').read_bytes()).hexdigest()
    assert digest == known['features_sha256']
    features = torch.load(BASE / 'features.pt', map_location='cpu', weights_only=True)
    rows = sorted(split['train'] + split['validation'], key=lambda r: r['ID'])
    assert len({r['ID'] for r in rows}) == len(rows) == len(features)
    for r in rows:
        assert features[r['ID']].shape == (r['frames'], 512)
        assert torch.isfinite(features[r['ID']]).all()
    # Audit the actually selected model only on its original held-out rows.
    model = p.Stage2Temporal()
    model.load_state_dict(torch.load(BASE / 'soft_mixed_scene/best.pt', weights_only=True, map_location='cpu')['model'])
    val = split['validation']
    pred = predictions(model, val, features, 'cpu')
    saved = pd.read_csv(BASE / 'soft_mixed_scene/validation_predictions.csv', dtype={'ID': str}).to_dict('records')
    assert pred == saved
    oracle = []
    model.eval()
    with torch.inference_mode():
        for r, q in zip(val, pred):
            _, _, h = model.logits(features[r['ID']].unsqueeze(0))
            scene = model.scene(torch.cat([h[:, r['t_collision']], h[:, r['t_entry']]], 1))
            oracle.append(dict(q, evasion_space=int(scene[:, :2].argmax(1)), entry_side=['LEFT', 'RIGHT'][int(scene[:, 2:].argmax(1))]))
    pd.DataFrame(details(val, pred)).to_csv(OUT / 'selected_errors.csv', index=False)
    write('selected-audit.json', {'actual': summary(val, pred), 'oracle_event_positions_diagnostic_only': summary(val, oracle)})
    # Size-balanced groups: source membership alone determines folds, no labels.
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['source_id']].append(r)
    ordered = list(sorted(groups))
    random.Random(20260910).shuffle(ordered)
    ordered.sort(key=lambda g: -len(groups[g]))
    folds = [[] for _ in range(5)]
    for g in ordered:
        destination = min(range(5), key=lambda i: (sum(len(groups[s]) for s in folds[i]), i))
        folds[destination].append(g)
    write('protocol.json', {'split_sha256': inputs, 'features_sha256': digest, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'folds': folds, 'seed': p.SEED, 'epochs': [5, 15], 'comparison': ['exact_5', 'exact_15', 'soft_mixed_15'], 'selection': 'Fixed epochs; outer folds never used for checkpoint selection', 'scope': 'Exploratory CV on previously used development labels; not fresh test or official score'})
    combined = collections.defaultdict(list)
    fold_results = []
    for fold, held in enumerate(folds):
        train = [r for r in rows if r['source_id'] not in held]
        val = [r for r in rows if r['source_id'] in held]
        assert not ({r['source_id'] for r in train} & {r['source_id'] for r in val})
        folder = OUT / f'fold-{fold}'
        folder.mkdir()
        pd.DataFrame(train).to_csv(folder / 'train.csv', index=False)
        pd.DataFrame(val).to_csv(folder / 'validation.csv', index=False)
        # Train-only constant baselines, including temporal median in seconds.
        constants = {k: collections.Counter(r[k] for r in train).most_common(1)[0][0] for k in ['evasion_space', 'entry_side']}
        medians = {k: float(np.median([r[t] / r['fps'] for r in train])) for k, t in [('collision_frame','t_collision'),('entry_frame','t_entry')]}
        q = [dict(ID=r['ID'], **constants, **{k: min(r['frames']-1, max(0, round(v*r['fps']))) for k,v in medians.items()}) for r in val]
        combined['train_constant'].extend(details(val, q))
        for mode in ['exact', 'soft_mixed']:
            random.seed(p.SEED); np.random.seed(p.SEED); torch.manual_seed(p.SEED)
            order_rng = random.Random(p.SEED)
            scene_rng = random.Random(p.SEED + 1)
            model = p.Stage2Temporal()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            for epoch in range(1, 16):
                model.train()
                order = list(train); order_rng.shuffle(order)
                losses = []
                for r in order:
                    if time.monotonic() - started > 7200:
                        raise TimeoutError('CPU experiment limit exceeded; partial output is not complete CV')
                    cl, el, h = model.logits(features[r['ID']].unsqueeze(0))
                    ct, et = torch.tensor([r['t_collision']]), torch.tensor([r['t_entry']])
                    mixed = mode == 'soft_mixed' and scene_rng.random() < min(.5, epoch/20)
                    ci, ei = (cl.argmax(1), el.argmax(1)) if mixed else (ct, et)
                    scene = model.scene(torch.cat([h[0, ci], h[0, ei]], 1))
                    sigma = 1. if mode == 'soft_mixed' else 0.
                    loss = temporal_loss(cl, ct, sigma) + temporal_loss(el, et, sigma)
                    loss += F.cross_entropy(scene[:, :2], torch.tensor([r['evasion_space']]))
                    loss += F.cross_entropy(scene[:, 2:], torch.tensor([int(r['entry_side'] == 'RIGHT')]))
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); losses.append(float(loss.detach()))
                write('status.json', {'status': 'RUNNING', 'fold': fold+1, 'mode': mode, 'epoch': epoch, 'seconds': time.monotonic()-started, 'loss': float(np.mean(losses))})
                print(f'fold={fold+1} mode={mode} epoch={epoch} seconds={time.monotonic()-started:.1f}', flush=True)
                if epoch == 15 or (epoch == 5 and mode == 'exact'):
                    name = f'{mode}_{epoch}'
                    q = predictions(model, val, features, 'cpu')
                    checkpoint = folder / f'{name}.pt'
                    torch.save({'model': model.state_dict()}, checkpoint)
                    reload = p.Stage2Temporal(); reload.load_state_dict(torch.load(checkpoint, weights_only=True)['model'])
                    assert predictions(reload, val, features, 'cpu') == q
                    ds = details(val, q)
                    pd.DataFrame(ds).to_csv(folder / f'{name}_errors.csv', index=False)
                    combined[name].extend(ds)
                    fold_results.append(dict(fold=fold, name=name, **summary(val, q)))
    results = {}
    for name, ds in combined.items():
        ds.sort(key=lambda r: r['ID'])
        assert [r['ID'] for r in ds] == [r['ID'] for r in rows]
        q = [dict(ID=r['ID'], **{k: r['pred_'+k] for k in p.OUTPUT_COLUMNS[1:]}) for r in ds]
        results[name] = summary(rows, q)
        pd.DataFrame(ds).to_csv(OUT / f'{name}_oof.csv', index=False)
    write('report.json', {'status': 'COMPLETED', 'seconds': time.monotonic()-started, 'samples': len(rows), 'sources': len(groups), 'results': results, 'fold_results': fold_results})
    write('status.json', {'status': 'COMPLETED', 'seconds': time.monotonic()-started})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            write('status.json', {'status': 'FAILED', 'error': repr(error)})
        raise
