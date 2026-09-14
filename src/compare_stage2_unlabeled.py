"""Paired prediction diagnostics only: no labels, accuracy, or model selection."""
import argparse
import csv
from collections import Counter
from fractions import Fraction
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from prepare_stage2_final_validation import ROOT, OUT as FROZEN, DATA, sha, save

OUT = ROOT/'artifacts/stage2-unlabeled-comparison-20260914'
NAMES = ('incumbent', 'ccd_pretrained')
TARGETS = ('collision_frame', 'entry_frame', 'evasion_space', 'entry_side')


def compare(rows, times):
    indexed = {}
    for name in NAMES:
        indexed[name] = {r['ID']: r for r in rows[name]}
        if len(indexed[name]) != len(rows[name]) or set(indexed[name]) != set(times):
            raise ValueError('Duplicate or missing prediction IDs')
    pairs = []
    for sid, timestamps in sorted(times.items()):
        a, b = (indexed[n][sid] for n in NAMES)
        for r in (a, b):
            for field in TARGETS[:2]:
                if type(r[field]) is not int or not 0 <= r[field] < len(timestamps):
                    raise ValueError('Invalid frame prediction')
            if type(r['evasion_space']) is not int or r['evasion_space'] not in (0, 1) or r['entry_side'] not in ('LEFT', 'RIGHT'):
                raise ValueError('Invalid category prediction')
        p = {'ID': sid, 'all_four_exactly_agree': all(a[k] == b[k] for k in TARGETS)}
        for field in TARGETS[:2]:
            delta = timestamps[b[field]] - timestamps[a[field]]
            p[field+'_delta'] = b[field] - a[field]
            p[field+'_seconds_delta'] = float(delta)
            p[field+'_abs_delta_gt_0_3s'] = abs(delta) > Fraction(3, 10)
        for field in TARGETS[2:]:
            p[field+'_disagrees'] = a[field] != b[field]
        pairs.append(p)
    summary = dict(scope='UNLABELED_PREDICTION_DIAGNOSTIC', videos=len(pairs), accuracy=None,
                   preferred_model=None, ground_truth_used=False,
                   delta_direction='ccd_pretrained minus incumbent',
                   warning='Agreement is not accuracy. Contact eligibility and recording-source independence remain unconfirmed.',
                   all_four_exactly_agree=sum(p['all_four_exactly_agree'] for p in pairs))
    for field in TARGETS[:2]:
        values = [abs(p[field+'_seconds_delta']) for p in pairs]
        summary[field] = dict(mean_absolute_seconds=statistics.mean(values), median_absolute_seconds=statistics.median(values),
                              max_absolute_seconds=max(values), abs_delta_gt_0_3s=sum(p[field+'_abs_delta_gt_0_3s'] for p in pairs))
    for field in TARGETS[2:]:
        summary[field] = dict(disagreements=sum(p[field+'_disagrees'] for p in pairs),
            paired_counts=dict(Counter(str(indexed[NAMES[0]][sid][field])+' -> '+str(indexed[NAMES[1]][sid][field]) for sid in times)))
    summary['model_diagnostics'] = {name: dict(entry_after_collision=sum(r['entry_frame'] > r['collision_frame'] for r in rows[name]),
        evasion_distribution=dict(Counter(str(r['evasion_space']) for r in rows[name])),
        entry_side_distribution=dict(Counter(r['entry_side'] for r in rows[name]))) for name in NAMES}
    return summary, pairs


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights
    from predict_stage2_final_validation import load_models
    from score_stage2_final_validation import read_inputs, read_csv
    torch.set_num_threads(2)
    started = time.monotonic()
    videos, times = read_inputs()
    contract = json.loads((FROZEN/'evaluation-contract.json').read_text())
    protected = {ROOT/rel: digest for rel, digest in contract['code_sha256'].items()}
    protected.update({DATA/'manifest.json': contract['manifest_sha256'], FROZEN/'frozen-models.json': contract['frozen_models_sha256'],
                      FROZEN/'evaluation-contract.json': sha(FROZEN/'evaluation-contract.json'),
                      DATA/'review-decisions.csv': sha(DATA/'review-decisions.csv')})
    for path, digest in protected.items():
        assert sha(path) == digest, str(path)
    save(OUT/'protocol.json', dict(scope='UNLABELED_PREDICTION_DIAGNOSTIC', authorization='User requested model comparison before labeling',
        models=list(NAMES), precision='CPU FP32', videos=len(videos), labels_used=False, accuracy=None, automatic_promotion=False,
        protected_hashes={str(p.relative_to(ROOT)): d for p, d in protected.items()},
        prediction_exposure='Candidate predictions now generated before human labels; do not claim no pre-label prediction exposure.',
        runner_sha256=sha(Path(__file__))))
    models, backbone, frozen = load_models()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    collected = {name: [] for name in NAMES}
    (OUT/'features').mkdir(exist_ok=True)
    (OUT/'records').mkdir(exist_ok=True)
    with torch.inference_mode():
        for index, sid in enumerate(sorted(videos)):
            rows = read_csv(DATA/videos[sid]['frame_map'])
            chunks = []
            for offset in range(0, len(rows), 16):
                batch = []
                for row in rows[offset:offset+16]:
                    path = DATA/row['file']
                    assert sha(path) == row['sha256'], str(path)
                    with Image.open(path) as image:
                        batch.append(transform(image.convert('RGB')))
                chunks.append(backbone(torch.stack(batch)).float())
                save(OUT/'status.json', dict(status='EXTRACTING', completed=index, total=len(videos), current_id=sid,
                    current_frames=min(offset+16, len(rows)), current_total_frames=len(rows), seconds=time.monotonic()-started))
            features = torch.cat(chunks)
            assert features.shape == (videos[sid]['frames'], 512) and torch.isfinite(features).all()
            feature_path = OUT/'features'/(sid+'.pt')
            torch.save(features, feature_path)
            record = dict(feature_sha256=sha(feature_path), frame_map_sha256=videos[sid]['frame_map_sha256'], models={})
            for name, model in models.items():
                ci, ei, scene = model(features[None])
                assert torch.isfinite(scene).all()
                row = dict(ID=sid, collision_frame=int(ci), entry_frame=int(ei), evasion_space=int(scene[0,:2].argmax()),
                           entry_side=['LEFT','RIGHT'][int(scene[0,2:].argmax())])
                collected[name].append(row)
                record['models'][name] = dict(prediction=row, scene_logits=scene.tolist())
            save(OUT/'records'/(sid+'.json'), record)
            print(f'{index+1}/{len(videos)} {sid}: {len(rows)} frames', flush=True)
        save(OUT/'status.json', dict(status='VALIDATING', completed=len(videos), total=len(videos)))
        # Reload saved features and independently reconstruct temporal heads for every video/model.
        for sid in sorted(videos):
            record = json.loads((OUT/'records'/(sid+'.json')).read_text())
            path = OUT/'features'/(sid+'.pt')
            assert sha(path) == record['feature_sha256']
            features = torch.load(path, map_location='cpu', weights_only=True)
            for name, model in models.items():
                h, _ = model.r(features[None])
                cl, el = model.tc(h).squeeze(-1), model.te(h).squeeze(-1)
                assert torch.isfinite(cl).all() and torch.isfinite(el).all()
                ci, ei = int(cl.argmax(1)), int(el.argmax(1))
                scene = model.scene(torch.cat([h[:,ci], h[:,ei]], 1))
                expected = record['models'][name]
                assert ci == expected['prediction']['collision_frame'] and ei == expected['prediction']['entry_frame']
                assert scene.tolist() == expected['scene_logits']
                assert int(scene[0,:2].argmax()) == expected['prediction']['evasion_space']
                assert ['LEFT','RIGHT'][int(scene[0,2:].argmax())] == expected['prediction']['entry_side']
    for path, digest in protected.items():
        assert sha(path) == digest, str(path)
    for entry in frozen.values():
        assert sha(ROOT/entry['file']) == entry['sha256']
    for name, rows in collected.items():
        write_csv(OUT/(name+'-predictions.csv'), rows)
        loaded = read_csv(OUT/(name+'-predictions.csv'))
        for row in loaded:
            for field in TARGETS[:3]:
                row[field] = int(row[field])
        assert loaded == rows
    summary, pairs = compare(collected, times)
    write_csv(OUT/'paired-differences.csv', pairs)
    save(OUT/'comparison.json', summary)
    save(OUT/'validation.json', dict(status='PASS', videos=len(videos), models=2, checks=[
        'Every input JPEG hash verified', 'All saved features hashed and reloaded', 'All temporal logits finite',
        'All predictions and scene logits exactly replayed through component heads', 'Prediction CSV round trip',
        'Labels, frozen models, input manifest and final evaluation code unchanged'],
        output_hashes={p.name: sha(p) for p in [OUT/'comparison.json', OUT/'paired-differences.csv',
            OUT/'incumbent-predictions.csv', OUT/'ccd_pretrained-predictions.csv']}))
    save(OUT/'status.json', dict(status='COMPLETE_VALIDATED', completed=len(videos), total=len(videos), seconds=time.monotonic()-started))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def watch():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'watch.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if (OUT/'protocol.json').exists():
            raise RuntimeError('Existing run found; inspect before restarting')
        try:
            with (OUT/'run.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run'], cwd=ROOT,
                    env=dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'),
                    stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                save(OUT/'watch-status.json', dict(status='RUNNING', pid=os.getpid(), child_pid=child.pid, timeout_seconds=14400, retries=0))
                try:
                    code = child.wait(timeout=14400)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'], capture_output=True, timeout=30)
                    raise TimeoutError('Four-hour CPU diagnostic limit exceeded')
                if code:
                    raise RuntimeError(f'Runner exit {code}; see run.log')
            assert json.loads((OUT/'validation.json').read_text())['status'] == 'PASS'
            assert json.loads((OUT/'status.json').read_text())['status'] == 'COMPLETE_VALIDATED'
            save(OUT/'watch-status.json', dict(status='COMPLETE', pid=os.getpid()))
        except Exception as exc:
            save(OUT/'watch-status.json', dict(status='FAILED', error=repr(exc)))
            save(OUT/'status.json', dict(status='FAILED', error=repr(exc)))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'watch'])
    args = parser.parse_args()
    (watch if args.command == 'watch' else run)()
