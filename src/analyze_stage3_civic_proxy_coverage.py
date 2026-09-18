"""Audit frozen Civic proxy coverage and a calibration-only persistence candidate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage3-civic-proxy-coverage-20260916'
SIGNALS = ROOT / 'artifacts/stage3-civic-calibration-20260914/signals.csv'
PRIOR = ROOT / 'artifacts/stage3-civic-partial-labels-20260914/all_samples.csv'
HELD = ROOT / 'artifacts/stage3-civic-holdout-labels-20260914/evaluation_partial_labels.csv'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prelim(g):
    angle = g.steering_corrected_deg.to_numpy()
    yaw = g.pose_yaw_left_rad_s.to_numpy()
    speed = g.speed_mps.to_numpy()
    domain = g.valid_sensor.to_numpy() & (speed >= 5) & (speed <= 35) & (np.abs(yaw) < .12)
    label = np.full(len(g), 'UNKNOWN', dtype='<U8')
    label[domain & (np.abs(angle) <= .5) & (np.abs(yaw) < .003)] = 'STRAIGHT'
    label[domain & (angle > 1.5) & (yaw > .012)] = 'LEFT'
    label[domain & (angle < -1.5) & (yaw < -.012)] = 'RIGHT'
    return label


def stable(label, radius):
    keep = np.zeros(len(label), dtype=bool)
    for i in range(radius, len(label)-radius):
        keep[i] = label[i] != 'UNKNOWN' and np.all(label[i-radius:i+radius+1] == label[i])
    return keep


def main():
    OUT.mkdir(exist_ok=False)
    signals = pd.read_csv(SIGNALS)
    prior = pd.read_csv(PRIOR)
    held = pd.read_csv(HELD)
    assert set(signals[signals.split == 'calibration'].route).isdisjoint(set(held.route))
    assert len(held) == 5310 and len(signals[signals.split == 'calibration']) == 3602
    plan = dict(status='CALIBRATION_ONLY', candidate='centered7 versus frozen centered11; unchanged thresholds and speed domain',
                gate='At least30 new rows per class; third gyro sensor transfer from other same-route segment: >=90% straight, >=97% turns for each route. No held labels or model predictions used for selection.',
                input_sha256={str(p.relative_to(ROOT)): sha(p) for p in (SIGNALS, PRIOR, HELD)},
                script_sha256=sha(Path(__file__)))
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    added = []
    rows = []
    for sid, g in signals[signals.split == 'calibration'].groupby('ID'):
        g = g.sort_values('sample_index').reset_index(drop=True)
        old = prior[prior.ID == sid].sort_values('sample_index').reset_index(drop=True)
        assert len(g) == len(old) and np.array_equal(g.sample_index, old.sample_index)
        label = prelim(g)
        keep11, keep7 = stable(label, 5), stable(label, 3)
        assert np.array_equal(keep11, old.partial_valid_steer.to_numpy())
        assert np.array_equal(np.where(keep11, label, 'UNKNOWN'), old.partial_steer_label.to_numpy())
        assert np.all(keep11 <= keep7)
        new = g[keep7 & ~keep11].copy()
        new['proxy_label'] = label[keep7 & ~keep11]
        added.append(new)
        rows.append(dict(ID=sid, route=g.route.iloc[0], old11=int(keep11.sum()), candidate7=int(keep7.sum()),
                         added=len(new), added_classes=new.proxy_label.value_counts().to_dict()))
    new = pd.concat(added, ignore_index=True)
    # Independent third-sensor check: bias comes from the other recording segment.
    transfer = []
    calib = signals[(signals.split == 'calibration') & signals.valid_sensor & signals.gyro_valid & (signals.speed_mps >= 5)]
    for route, group in calib.groupby('route'):
        ids = sorted(group.ID.unique())
        assert len(ids) == 2
        for target in ids:
            source = next(x for x in ids if x != target)
            origin = group[group.ID == source]
            bias = float(np.median(origin.gyro_yaw_left_rad_s - origin.pose_yaw_left_rad_s))
            test = new[(new.ID == target) & new.valid_sensor & new.gyro_valid].copy()
            centered = test.gyro_yaw_left_rad_s.to_numpy() - bias
            pred = np.where(centered > .012, 'LEFT', np.where(centered < -.012, 'RIGHT',
                             np.where(np.abs(centered) < .006, 'STRAIGHT', 'UNKNOWN')))
            for label in ('LEFT', 'STRAIGHT', 'RIGHT'):
                mask = test.proxy_label.to_numpy() == label
                transfer.append(dict(route=route, source=source, target=target, proxy_label=label,
                                     added_rows=int(mask.sum()), gyro_agreement=float(np.mean(pred[mask] == label)) if mask.any() else None))
    counts = new.proxy_label.value_counts().to_dict()
    gate = all(counts.get(label, 0) >= 30 for label in ('LEFT', 'STRAIGHT', 'RIGHT'))
    gate &= all(r['added_rows'] >= 10 and r['gyro_agreement'] is not None and
                r['gyro_agreement'] >= (.90 if r['proxy_label'] == 'STRAIGHT' else .97) for r in transfer)
    # Held data is descriptive only. It is not relabeled or used in the gate.
    speed_bins = pd.cut(held.speed_mps, [-1, .5, 2, 5, 35, np.inf], labels=['stopped','0.5-2','2-5','5-35','>35'])
    held_speed = {str(k): int(v) for k, v in speed_bins.value_counts(sort=False).items()}
    new.to_csv(OUT/'calibration-new7-rows.csv', index=False)
    result = dict(status='COMPLETE_VALIDATED', decision='KEEP_FROZEN_11' if not gate else 'CANDIDATE_7_CALIBRATION_PASS',
                  calibration_segments=rows, calibration_added_classes=counts, gyro_transfer=transfer,
                  held_speed_bins=held_speed, held_exclusion_reasons=held.exclusion_reason.value_counts().to_dict(),
                  gate_passed=bool(gate), validation='Frozen11 exact reproduction, disjoint routes, candidate7 superset, source-to-other-segment gyro bias transfer; held data descriptive only.',
                  output_sha256={'calibration-new7-rows.csv': sha(OUT/'calibration-new7-rows.csv')})
    (OUT/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status','decision','calibration_added_classes','held_speed_bins','gate_passed')}, indent=2))


if __name__ == '__main__':
    main()
