"""Apply the user's approved 32..49 UNKNOWN correction and rescore frozen predictions."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from review_stage3_human import dense_rows, save

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/stage3-human-comparison-20260914'
OUT = ROOT / 'artifacts/stage3-human-comparison-20260914-revision1'
LABELS = ROOT / 'data/stage3-human-review'
KEYS = ['ID', 'sample_index']
CLASSES = ['LEFT', 'STRAIGHT', 'RIGHT']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    previous = json.loads((OLD / 'report.json').read_text(encoding='utf-8'))
    annotation_path = LABELS / 'annotations.json'
    state = json.loads(annotation_path.read_text(encoding='utf-8-sig'))
    original_csv = pd.read_csv(OLD / 'random-labels.csv')
    before = copy.deepcopy(state)
    segments = [dict(start=0, end=31, steer='STRAIGHT', motion='MOVING'),
                dict(start=32, end=49, steer='UNKNOWN', motion='MOVING')]
    already_applied = state['videos']['REVIEW_011']['segments'] == segments
    backup = OUT / 'before'
    if not already_applied:
        for path, expected in previous['inputs'].items():
            assert sha(ROOT / path) == expected, path
        backup.mkdir(exist_ok=True)
        for name in ('annotations.json', 'labels_review.csv', 'sources.json'):
            target = backup / name
            assert not target.exists() or sha(target) == sha(LABELS / name)
            shutil.copy2(LABELS / name, target)
        state['videos']['REVIEW_011']['segments'] = segments
        save(state, annotation_path)
    assert (backup / 'annotations.json').is_file(), 'Original backup required'
    original = json.loads((backup / 'annotations.json').read_text(encoding='utf-8-sig'))
    original['videos']['REVIEW_011']['segments'] = segments
    assert original == state
    labels = pd.read_csv(LABELS / 'labels_review.csv')
    expected_csv = original_csv.copy()
    mask = (expected_csv.ID == 'REVIEW_011') & expected_csv.sample_index.between(32, 49)
    assert mask.sum() == 18
    expected_csv.loc[mask, 'steer_label'] = 'UNKNOWN'
    expected_csv.loc[mask, 'valid_steer'] = 0
    pd.testing.assert_frame_equal(labels, expected_csv)
    reconstructed = pd.DataFrame([r for sid, v in state['videos'].items()
                                  for r in dense_rows(sid, v['frames'], v['segments'])])
    pd.testing.assert_frame_equal(labels, reconstructed[labels.columns])
    results = []
    predictions_hashes = {}
    for group, folder, count in [('random', LABELS, 282),
                                 ('right_enriched', ROOT / 'data/stage3-right-review', 50)]:
        truth = pd.read_csv(folder / 'labels_review.csv')
        truth.to_csv(OUT / f'{group}-labels.csv', index=False)
        eligible = truth[(truth.reviewed == 1) & (truth.motion_status == 'MOVING')
                         & truth.steer_label.isin(CLASSES)]
        assert len(eligible) == count
        paired = eligible.copy()
        for model in ('old', 'short'):
            source = OLD / f'{group}-{model}-predictions.csv'
            validation = json.loads((OLD / 'independent-validation.json').read_text())
            assert sha(source) == validation['artifact_sha256'][source.name]
            predictions_hashes[str(source.relative_to(ROOT))] = sha(source)
            dest = OUT / source.name
            shutil.copy2(source, dest)
            score_path = OUT / f'{group}-{model}-score.json'
            subprocess.run([sys.executable, str(ROOT / 'src/score_stage3_human.py'),
                            '--labels', str(folder / 'labels_review.csv'), '--predictions', str(dest),
                            '--output', str(score_path)], check=True, capture_output=True, timeout=30)
            score = json.loads(score_path.read_text())
            pred = pd.read_csv(dest)
            assert set(map(tuple, pred[KEYS].values)) == set(map(tuple, truth[KEYS].values))
            joined = eligible.merge(pred[KEYS + ['steer_label']], on=KEYS, validate='one_to_one',
                                    suffixes=('_truth', '_pred'))
            cm = [[int(((joined.steer_label_truth == a) & (joined.steer_label_pred == b)).sum())
                   for b in CLASSES] for a in CLASSES]
            assert cm == score['confusion']
            f1 = []
            recall = {}
            for i, name in enumerate(CLASSES):
                support = sum(cm[i])
                denominator = support + sum(row[i] for row in cm)
                f1.append(2 * cm[i][i] / denominator if denominator else 0)
                recall[name] = cm[i][i] / support if support else None
            assert abs(sum(f1) / 3 - score['macro_f1']) < 1e-12
            results.append(dict(group=group, model=model, recall=recall, **score))
            paired = paired.merge(pred[KEYS + ['steer_label']].rename(columns={'steer_label': model}),
                                  on=KEYS, validate='one_to_one')
        paired.to_csv(OUT / f'{group}-paired.csv', index=False)
    inputs = {path: sha(ROOT / path) for path in previous['inputs']}
    for path, expected in previous['inputs'].items():
        if path not in [str((LABELS / name).relative_to(ROOT)) for name in ('annotations.json', 'labels_review.csv')]:
            assert inputs[path] == expected
    output = dict(status='COMPLETE_VALIDATED', revision='User approved REVIEW_011 frames 32..49 steer UNKNOWN; motion unchanged',
                  prior_report=str((OLD / 'report.json').relative_to(ROOT)), models=previous['models'],
                  inputs=inputs, prediction_hashes=predictions_hashes, results=results,
                  eligible_frames=332, validation='Exactly 18 steering and validity fields changed; annotation/CSV parity; frozen predictions; independent confusion/F1 PASS',
                  limitations='Fixed three-class F1; separate random/enriched diagnostics, not official or independent generalization; label revised after predictions were examined')
    (OUT / 'report.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
