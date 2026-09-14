"""Strict local Stage2 scoring using native-frame rational timestamps, never FPS."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
from pathlib import Path

from prepare_stage2_final_validation import ROOT, OUT, DATA, sha, save

TARGETS = ['collision_frame', 'entry_frame', 'evasion_space', 'entry_side']


def integer(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    return int(number)


def unique(rows):
    result = {}
    for row in rows:
        sid = row.get('ID', '')
        if not sid or sid in result:
            raise ValueError('Missing or duplicate ID')
        result[sid] = row
    return result


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_inputs(data=DATA, out=OUT):
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    validation = json.loads((out/'input-validation.json').read_text())
    if manifest['status'] != 'INPUTS_READY_UNLABELED' or validation['status'] != 'PASS' or sha(data/'manifest.json') != validation['manifest_sha256']:
        raise ValueError('Input preparation/manifest verification incomplete')
    videos = unique(manifest['videos'])
    times = {}
    for sid, video in videos.items():
        path = data/video['frame_map']
        if sha(path) != video['frame_map_sha256']:
            raise ValueError('Frame map changed: '+sid)
        rows = read_csv(path)
        if [integer(r['frame_index']) for r in rows] != list(range(video['frames'])):
            raise ValueError('Frame map index mismatch')
        ts = [Fraction((int(r['pts'])-int(r['first_pts']))*int(r['time_base_num']), int(r['time_base_den'])) for r in rows]
        if not ts or ts[0] != 0 or any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError('Invalid presentation timestamp map')
        times[sid] = ts
    return videos, times


def label_gate(rows, times):
    labels = unique(rows)
    if set(labels) != set(times):
        raise ValueError('Every candidate needs exactly one review disposition')
    accepted, excluded, pending = {}, {}, []
    for sid, row in labels.items():
        status = row.get('review_status', '')
        if status == 'PENDING':
            pending.append(sid)
            continue
        if status not in ['ACCEPT', 'EXCLUDE'] or not row.get('reviewer', '').strip():
            raise ValueError('Invalid review status or missing reviewer: '+sid)
        if status == 'EXCLUDE':
            if not row.get('exclusion_reason', '').strip():
                raise ValueError('Exclusion reason required: '+sid)
            excluded[sid] = row['exclusion_reason']
            continue
        truth = dict(row)
        for key in ['collision_frame', 'entry_frame']:
            value = integer(row.get(key))
            if value is None or not 0 <= value < len(times[sid]):
                raise ValueError('Invalid human frame label: '+sid+'/'+key)
            truth[key] = value
        value = integer(row.get('evasion_space'))
        if value not in [0, 1] or row.get('entry_side') not in ['LEFT', 'RIGHT']:
            raise ValueError('Invalid human category label: '+sid)
        truth['evasion_space'] = value
        accepted[sid] = truth
    return dict(status='WAITING_FOR_HUMAN_LABELS' if pending else ('READY_TO_SCORE' if accepted else 'NO_ELIGIBLE_SAMPLES'),
                accepted=accepted, excluded=excluded, pending=pending)


def score_rows(review, predictions, times):
    gate = label_gate(review, times)
    if gate['status'] != 'READY_TO_SCORE':
        raise ValueError('Scoring requires complete human review and eligible labels')
    pred = unique(predictions)
    if set(pred)-set(times):
        raise ValueError('Prediction includes unknown candidate IDs')
    details = []
    for sid, truth in gate['accepted'].items():
        q = pred.get(sid, {})
        row = dict(ID=sid, recording_source=truth.get('recording_source', 'UNKNOWN'), missing_prediction=sid not in pred)
        for key in TARGETS:
            val = q.get(key)
            if key.endswith('_frame'):
                index = integer(val)
                valid = index is not None and 0 <= index < len(times[sid])
                error = abs(times[sid][index]-times[sid][truth[key]]) if valid else None
                ok = valid and error <= Fraction(3, 10)
                row[key+'_error_seconds'] = float(error) if valid else None
            elif key == 'evasion_space':
                index = integer(val)
                valid = index in [0, 1]
                ok = valid and index == truth[key]
            else:
                valid = val in ['LEFT', 'RIGHT']
                ok = valid and val == truth[key]
            row[key+'_valid'] = bool(valid)
            row[key+'_ok'] = bool(ok)
        row['four_item_mean'] = sum(row[k+'_ok'] for k in TARGETS)/4
        details.append(row)
    n = len(details)
    metrics = {key+'_accuracy': sum(r[key+'_ok'] for r in details)/n for key in TARGETS}
    metrics['four_item_mean'] = sum(r['four_item_mean'] for r in details)/n
    groups = {}
    for r in details:
        source = r['recording_source'].strip()
        if source and source != 'UNKNOWN':
            groups.setdefault(source, []).append(r['four_item_mean'])
    return dict(status='SCORED' if n >= 30 else 'SCORED_INSUFFICIENT_ELIGIBLE_COUNT', eligible=n,
                excluded=len(gate['excluded']), missing_prediction_count=sum(r['missing_prediction'] for r in details),
                invalid_by_item={k: sum(not r[k+'_valid'] for r in details) for k in TARGETS},
                metrics=metrics, source_equal_mean=(sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None),
                known_source_videos=sum(len(v) for v in groups.values()), known_source_groups=len(groups),
                scope='Local four-item diagnostic; not official score. Unknown recording sources are not certified independent.', details=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'score'])
    parser.add_argument('--labels', type=Path, default=DATA/'review-decisions.csv')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    videos, times = read_inputs()
    labels = read_csv(args.labels)
    gate = label_gate(labels, times)
    if args.command == 'check':
        result = dict(status=gate['status'], candidates=len(videos), accepted=len(gate['accepted']),
                      excluded=len(gate['excluded']), pending=len(gate['pending']), labels_sha256=sha(args.labels), metrics=None)
        save(OUT/'label-readiness.json', result)
        print(json.dumps(result))
        return
    if not args.predictions or not args.output:
        parser.error('score requires --predictions and --output')
    result = score_rows(labels, read_csv(args.predictions), times)
    result['labels_sha256'] = sha(args.labels)
    result['predictions_sha256'] = sha(args.predictions)
    result['manifest_sha256'] = sha(DATA/'manifest.json')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'details.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['details'][0]))
        writer.writeheader()
        writer.writerows(result.pop('details'))
    save(args.output/'score.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
