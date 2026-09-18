"""Independently audit saved human comparison rows and summarize paired errors."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage3-human-comparison-20260914'
CLASSES = ['LEFT', 'STRAIGHT', 'RIGHT']


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row['ID'], int(row['sample_index'])


def main():
    report = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
    for path, digest in report['inputs'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    summaries = {}
    video_rows = []
    expected = {'random': Counter(STRAIGHT=200, LEFT=100),
                'right_enriched': Counter(RIGHT=50)}
    for group in expected:
        labels = rows(OUT / f'{group}-labels.csv')
        truth = {key(r): r for r in labels}
        assert len(truth) == len(labels)
        eligible = {k: r for k, r in truth.items() if r['reviewed'] == '1'
                    and r['motion_status'] == 'MOVING' and r['steer_label'] in CLASSES}
        assert Counter(r['steer_label'] for r in eligible.values()) == expected[group]
        predictions = {}
        for model in ('old', 'short'):
            pred_rows = rows(OUT / f'{group}-{model}-predictions.csv')
            pred = {key(r): r['steer_label'] for r in pred_rows}
            assert len(pred) == len(pred_rows) and set(pred) == set(truth)
            assert set(pred.values()) <= set(CLASSES)
            predictions[model] = pred
            cm = [[0] * 3 for _ in CLASSES]
            for k, row in eligible.items():
                cm[CLASSES.index(row['steer_label'])][CLASSES.index(pred[k])] += 1
            f1 = []
            for i in range(3):
                denominator = sum(cm[i]) + sum(line[i] for line in cm)
                f1.append(2 * cm[i][i] / denominator if denominator else 0)
            saved = next(r for r in report['results'] if r['group'] == group and r['model'] == model)
            assert cm == saved['confusion']
            assert abs(sum(f1) / 3 - saved['macro_f1']) < 1e-12
        paired = rows(OUT / f'{group}-paired.csv')
        assert len(paired) == len(eligible) and {key(r) for r in paired} == set(eligible)
        for row in paired:
            assert row['steer_label'] == eligible[key(row)]['steer_label']
            for model in predictions:
                assert row[model] == predictions[model][key(row)]
        transitions = Counter()
        for k, row in eligible.items():
            old_ok = predictions['old'][k] == row['steer_label']
            short_ok = predictions['short'][k] == row['steer_label']
            transitions[f'old_{int(old_ok)}_short_{int(short_ok)}'] += 1
        for sid in sorted({k[0] for k in eligible}):
            subset = {k: r for k, r in eligible.items() if k[0] == sid}
            record = {'group': group, 'ID': sid, 'frames': len(subset),
                      'truth': dict(Counter(r['steer_label'] for r in subset.values()))}
            for model, pred in predictions.items():
                record[model] = {'correct': sum(pred[k] == r['steer_label'] for k, r in subset.items()),
                                 'predictions': dict(Counter(pred[k] for k in subset))}
            video_rows.append(record)
        summaries[group] = {'total': len(labels), 'eligible': len(eligible),
                            'excluded': len(labels) - len(eligible),
                            'review_motion_steer_counts': dict(Counter(
                                '/'.join(r[c] for c in ('reviewed', 'motion_status', 'steer_label')) for r in labels)),
                            'paired_correctness': dict(transitions)}
    artifact_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(OUT.iterdir()) if p.suffix in ('.csv', '.npz', '.pt')}
    output = {'status': 'PASS', 'checks': 'Original input hashes, exact full prediction keys, eligible supports, independent CSV confusion/F1, paired row parity',
              'groups': summaries, 'per_video': video_rows, 'artifact_sha256': artifact_hashes}
    (OUT / 'independent-validation.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({'groups': summaries, 'per_video': video_rows}, indent=2))


if __name__ == '__main__':
    main()
