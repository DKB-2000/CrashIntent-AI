"""Check a converted comma chunk against its accepted source manifest."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audit-csv', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    accepted = pd.read_csv(args.audit_csv).sort_values(['route', 'segment']).to_dict('records')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if len(manifest) != len(accepted):
        raise ValueError('Manifest does not cover all accepted segments')
    columns = ['ID', 'frame_index', 'accel_label', 'steer_label']
    frames = []
    videos = []
    for index, (entry, source) in enumerate(zip(manifest, accepted)):
        sid = f'COMMA2K19_C1_{index:04d}'
        out = root / 'converted' / sid
        report = json.loads((out / 'conversion_report.json').read_text(encoding='utf-8'))
        provenance = json.loads((root / 'segments' / sid / 'source_manifest.json').read_text(encoding='utf-8'))
        if entry['ID'] != sid or entry['source_segment'].rstrip('/') != source['segment'].rstrip('/') or entry['source_route'] != source['route']:
            raise ValueError(f'{sid}: manifest/source mismatch')
        if provenance['segment'].rstrip('/') != source['segment'].rstrip('/'):
            raise ValueError(f'{sid}: extracted source mismatch')
        if report['status'] != 'PASS' or report['id'] != sid or not np.isclose(report['thresholds']['steer_offset'], source['steering_median']):
            raise ValueError(f'{sid}: report/settings mismatch')
        frame = pd.read_csv(out / 'labels.csv')
        if list(frame.columns) != columns or frame.isna().any().any() or not frame['ID'].eq(sid).all():
            raise ValueError(f'{sid}: invalid label schema')
        if not np.array_equal(frame['frame_index'].to_numpy(), np.arange(len(frame))):
            raise ValueError(f'{sid}: noncontiguous frame indices')
        if not set(frame.accel_label) <= {'STOPPED', 'CONSTANT', 'ACCELERATING', 'DECELERATING'} or not set(frame.steer_label) <= {'LEFT', 'RIGHT', 'STRAIGHT'}:
            raise ValueError(f'{sid}: unknown class')
        video = out / 'videos' / f'{sid}.mp4'
        cap = cv2.VideoCapture(str(video))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            first_ok, _ = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(count-1, 0))
            last_ok, _ = cap.read()
        finally:
            cap.release()
        if not first_ok or not last_ok or not np.isclose(fps, 10) or count != len(frame) or count != report['output_frames'] or count != int(source['samples']) or count != entry['samples']:
            raise ValueError(f'{sid}: video/label/count mismatch')
        frames.append(frame)
        videos.append(dict(ID=sid, video=video.relative_to(root).as_posix(), source_route=source['route'], samples=count))
    combined = pd.read_csv(root / 'labels.csv')
    pd.testing.assert_frame_equal(combined, pd.concat(frames, ignore_index=True))
    if combined.duplicated(['ID', 'frame_index']).any():
        raise ValueError('Duplicate frame keys')
    summary = dict(status='PASS', segments=len(manifest), routes=len({x['source_route'] for x in manifest}), samples=len(combined), duration_minutes=len(combined)/600, accel_distribution=combined.accel_label.value_counts().to_dict(), steer_distribution=combined.steer_label.value_counts().to_dict(), moving_steer_distribution=combined.loc[combined.accel_label != 'STOPPED', 'steer_label'].value_counts().to_dict(), checks='Source provenance, settings, CSV schema and indices, combined CSV equality, MP4 metadata and first/last frame decoding; not full video decode or visual label calibration.', label_status='Provisional; per-segment steering median is not calibrated vehicle steering zero.')
    pd.DataFrame(videos).to_csv(root / 'video_manifest.csv', index=False)
    (root / 'validation_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()