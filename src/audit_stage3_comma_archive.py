"""Audit a comma2k19 ZIP without extracting video or Windows-incompatible paths.

Reports provisional label/episode distributions; does not certify label semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from prepare_stage3_comma2k19 import _deduplicate_times, _make_labels, _one_dimensional, _validate_times


def audit(archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    records, failures = [], []
    totals = {key: Counter() for key in ('accel_label', 'steer_label', 'moving_steer')}
    episodes = {key: Counter() for key in ('accel_label', 'steer_label')}
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        segments = sorted(name[:-len('video.hevc')] for name in names if name.endswith('/video.hevc'))
        if not segments:
            raise ValueError('No video segments in archive')
        for prefix in segments:
            try:
                def load(relative):
                    return _one_dimensional(relative, np.load(io.BytesIO(source.read(prefix + relative)), allow_pickle=False))
                times = load('global_pose/frame_times')
                speed_key = 'processed_log/CAN/speed/'
                if prefix + speed_key + 't' not in names:
                    speed_key = 'processed_log/CAN/car_speed/'
                speed_t, speed = load(speed_key + 't'), load(speed_key + 'value')
                speed_t, speed = _deduplicate_times('speed', speed_t, speed)
                steer_t = load('processed_log/CAN/steering_angle/t')
                steer = load('processed_log/CAN/steering_angle/value')
                steer_t, steer = _deduplicate_times('steer', steer_t, steer)
                _validate_times('frames', times, times)
                _validate_times('speed', speed_t, speed)
                _validate_times('steer', steer_t, steer)
                if not np.isclose(np.median(np.diff(times)), .05, atol=.005):
                    raise ValueError('Frame timeline is not approximately 20Hz')
                target = times[::2]
                outside = (target < max(speed_t[0], steer_t[0])) | (target > min(speed_t[-1], steer_t[-1]))
                if outside.mean() > .01:
                    raise ValueError('More than 1% of target samples outside CAN coverage')
                labels = _make_labels(prefix, target, speed_t, speed, steer_t, steer,
                                      smoothing_window=7, stop_speed=.5, accel_threshold=.2,
                                      steer_deadzone=.5, steer_offset=0., positive_steer_label='LEFT')
                record = dict(segment=prefix.rstrip('/'), route=prefix.rstrip('/').rsplit('/', 1)[0],
                              samples=len(labels), extrapolated_samples=int(outside.sum()),
                              duration_seconds=float(times[-1]-times[0]),
                              speed_min=float(speed.min()), speed_max=float(speed.max()),
                              steering_median=float(np.median(steer)))
                for key in ('accel_label', 'steer_label'):
                    counts = Counter(labels[key])
                    totals[key].update(counts)
                    starts = labels[key].ne(labels[key].shift())
                    episodes[key].update(labels.loc[starts, key])
                    record.update({key + '_' + k: v for k, v in counts.items()})
                totals['moving_steer'].update(labels.loc[labels.accel_label != 'STOPPED', 'steer_label'])
                records.append(record)
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
                failures.append(dict(segment=prefix, error=str(exc)))
    pd.DataFrame(records).fillna(0).to_csv(output / 'segments.csv', index=False)
    report = dict(archive=str(archive.resolve()), segments_found=len(segments), accepted=len(records),
                  routes=len({r['route'] for r in records}), failures=failures,
                  distributions=totals, episode_starts_per_segment=episodes,
                  provisional_thresholds=dict(stop_speed=.5, accel_threshold=.2, steer_deadzone=.5,
                                              steer_offset=0., positive_steer_label='LEFT', smoothing_window=7),
                  note='Provisional CAN labels, no video decoding. Episode counts reset at segment boundaries. Calibrate steering per vehicle/route before training.')
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    audit(args.archive, args.output_dir)
