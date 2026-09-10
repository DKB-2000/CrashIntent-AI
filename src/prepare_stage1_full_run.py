"""Incrementally validate completed Stage1 videos, then build a local full bundle.

This process never uploads files or starts a GPU notebook.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import pandas as pd
from prepare_stage1_training import plan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def inspect_video(path, expected):
    require(digest(path) == expected['sha256'], f'Video hash mismatch: {path.name}')
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), f'Cannot open {path}')
    count = 0
    try:
        require(abs(cap.get(cv2.CAP_PROP_FPS) - expected['fps']) < .01, f'FPS mismatch: {path.name}')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            require(frame.shape[:2] == (expected['height'], expected['width']), f'Dimension mismatch: {path.name}')
            count += 1
    finally:
        cap.release()
    require(count == expected['frames'] and count > 0, f'Frame count mismatch: {path.name}')
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-dir', type=Path, required=True)
    ap.add_argument('--state-dir', type=Path, required=True)
    ap.add_argument('--bundle-dir', type=Path, required=True)
    ap.add_argument('--max-hours', type=float, default=8)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = a.dataset_dir.resolve()
    a.state_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    expected, phone, excluded = plan(root / 'data_raw/ccd/videos/Crash-1500', root / 'data_raw/ccd/Crash-1500.txt')
    expected = {r['source_id']: r for r in expected}
    verified, records = {}, {}
    report = dict(status='VALIDATING_GENERATED_SOURCES', expected_sources=len(expected), validated_sources=0,
                  validated_videos=0, decoded_frames=0, uploaded=False, gpu_started=False)
    def save():
        report['checked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = a.state_dir / 'status.tmp'
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(a.state_dir / 'status.json')
    started = time.monotonic()
    save()
    try:
        while time.monotonic() - started < a.max_hours * 3600:
            paths = sorted((data / 'records').glob('*.json'))
            for path in paths:
                sid = path.stem
                require(sid in expected, f'Unexpected source record: {sid}')
                if sid in records:
                    require(digest(path) == records[sid], f'Completed record changed: {sid}')
                    continue
                try:
                    record = json.loads(path.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    continue  # Writer may still be committing its new record.
                for key, value in expected[sid].items():
                    require(record['identity'][key] == value, f'Source identity mismatch: {sid}/{key}')
                require(record['identity']['source_sha256'] == digest(expected[sid]['source_path']), f'Original hash mismatch: {sid}')
                require(record['identity']['variants'] == 2 and len(record['rows']) == 3, 'Expected three derivatives')
                require(sorted(r['variant'] for r in record['rows']) == [0, 1, 2], 'Variant mismatch')
                for row in record['rows']:
                    for key, value in expected[sid].items():
                        require(row[key] == value, f'Derivative identity mismatch: {sid}/{key}')
                    require(row['label'] == ('ORIGINAL' if row['variant'] == 0 else 'RERECORDED'), 'Label mismatch')
                    destination = (data / row['output_path']).resolve()
                    require(destination.is_relative_to(data), 'Output path escape')
                    require(row['output_path'] not in verified, 'Duplicate output path')
                    report['decoded_frames'] += inspect_video(destination, row)
                    verified[row['output_path']] = row
                records[sid] = digest(path)
                report.update(validated_sources=len(records), validated_videos=len(verified))
                save()
                if len(records) % 20 == 0:
                    print(f'Validated {len(records)}/{len(expected)} sources; {len(verified)} videos', flush=True)
            if len(records) == len(expected):
                try:
                    generation = json.loads((data / 'generation_report.json').read_text())
                except json.JSONDecodeError:
                    generation = {}
                if generation.get('sources') == 1110 and generation.get('samples') == 3330:
                    require(generation['status'] == 'PASS', 'Generator did not pass')
                    require(digest(data / 'generation_manifest.csv') == generation['manifest_sha256'], 'Final manifest digest mismatch')
                    table = pd.read_csv(data / 'generation_manifest.csv', dtype={'source_id': str, 'upload_group': str})
                    require(len(table) == 3330 and not table.output_path.duplicated().any(), 'Final sample count mismatch')
                    require(set(table.output_path) == set(verified), 'Final output list mismatch')
                    for row in table.to_dict('records'):
                        prior = verified[row['output_path']]
                        for key in ['source_id', 'upload_group', 'split', 'label', 'sha256', 'variant', 'frames', 'width', 'height', 'fps']:
                            require(row[key] == prior[key], f'Final manifest changed {key}')
                    require(set(pd.read_csv(data / 'phone_holdout.csv', dtype=str).source_id) == set(phone), 'Phone holdout changed')
                    require(set(pd.read_csv(data / 'excluded_sources.csv', dtype=str).source_id) == set(excluded), 'Excluded siblings changed')
                    require(table.groupby('upload_group').split.nunique().max() == 1, 'Upload-group leakage')
                    report.update(status='DATA_VALIDATED', manifest_sha256=generation['manifest_sha256'])
                    save()
                    (a.state_dir / 'data-validation.json').write_text(json.dumps(dict(report, status='PASS'), indent=2))
                    report['status'] = 'BUILDING_LOCAL_BUNDLE'
                    save()
                    with (a.state_dir / 'bundle.log').open('w', encoding='utf-8') as log:
                        subprocess.run([sys.executable, str(root / 'src/prepare_stage1_gpu.py'), '--dataset-dir', str(data),
                                        '--output-dir', str(a.bundle_dir.resolve()), '--mode', 'full'], cwd=root,
                                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
                    bundle = json.loads((a.bundle_dir / 'bundle.json').read_text())
                    require(bundle['expected_sources'] == 1110 and bundle['expected_samples'] == 3330, 'Wrong full bundle')
                    report.update(status='BUNDLE_READY_NOT_UPLOADED', bundle=bundle)
                    save()
                    print(json.dumps(report, indent=2), flush=True)
                    return
            time.sleep(45)
        raise TimeoutError('Local preparation wait budget exceeded')
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
