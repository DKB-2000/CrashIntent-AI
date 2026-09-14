"""Materialize immutable native-frame JPEG inputs and exact presentation-time maps."""
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from fractions import Fraction
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/stage2-final-validation-20260914'
DATA = ROOT/'data/stage2-validation-nexar-20260914'
RAW = ROOT/'data_raw/stage2-validation-nexar-20260914'
FF = ROOT/'.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def parse_frames(log):
    match = re.search(r'config in time_base:\s*(\d+)/(\d+)', log)
    if not match:
        raise ValueError('Missing decoder time base')
    num, den = map(int, match.groups())
    pairs = [(int(n), int(pts)) for n, pts in re.findall(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:', log)]
    if not pairs or [n for n, _ in pairs] != list(range(len(pairs))):
        raise ValueError('Missing/non-contiguous decoded frame index')
    if num <= 0 or den <= 0 or any(b[1] <= a[1] for a, b in zip(pairs, pairs[1:])):
        raise ValueError('Non-monotonic frame presentation timestamps')
    return num, den, [pts for _, pts in pairs]


def prepare():
    DATA.mkdir(exist_ok=False)
    (DATA/'images').mkdir()
    (DATA/'frame-maps').mkdir()
    rows = list(csv.DictReader((RAW/'review-inventory.csv').open(encoding='utf-8-sig', newline='')))
    assert len(rows) == len({r['ID'] for r in rows}) == 50
    reservation = json.loads((ROOT/'artifacts/stage2-validation-inspection-20260914/reservation.json').read_text(encoding='utf-8-sig'))
    assert set(reservation['IDs']) == {r['ID'] for r in rows}
    assert shutil.disk_usage(DATA).free > 25*1024**3
    protocol = dict(status='PREPARING', candidate_inventory_sha256=sha(RAW/'review-inventory.csv'),
                    ffmpeg_sha256=sha(FF), script_sha256=sha(Path(__file__)),
                    input_policy='All native frames, full original dimensions, JPEG q=2, no crop/fps filter/event trimming; zero-based frame filenames',
                    timestamp_policy='Exact integer decoder PTS and rational time base; zero relative to first decoded frame',
                    model_policy='Frozen incumbent and previously submitted CCD-pretrained comparison; no selection based on these candidates',
                    human_labels_created=0, predictions_created=0, max_jpeg_bytes=20*1024**3)
    save(OUT/'protocol.json', protocol)
    models = {}
    (OUT/'models').mkdir()
    for name, rel in [('incumbent', 'artifacts/stage2-controls-20260909/soft_mixed_scene/best.pt'),
                      ('ccd_pretrained', 'artifacts/stage2-pretrained-submit-candidate-20260911/best.pt')]:
        source = ROOT/rel
        target = OUT/'models'/(name+'.pt')
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        models[name] = dict(file=str(target.relative_to(ROOT)).replace('\\', '/'), sha256=sha(target), original=rel)
    with zipfile.ZipFile(ROOT/'artifacts/stage2-pretrained-submit-candidate-20260911/submit.zip') as z:
        (OUT/'models/resnet.pt').write_bytes(z.read('model/stage2/resnet18-f37072fd.pth'))
        (OUT/'models/reference-inference.py').write_bytes(z.read('inference.py'))
    for name in ['resnet.pt', 'reference-inference.py']:
        path = OUT/'models'/name
        models[name] = dict(file=str(path.relative_to(ROOT)).replace('\\', '/'), sha256=sha(path))
    save(OUT/'frozen-models.json', models)
    videos, total_bytes, total_frames = [], 0, 0
    started = time.monotonic()
    for r in rows:
        source = RAW/r['file']
        assert sha(source).lower() == r['expected_sha256'].lower()
        folder = DATA/'images'/r['ID']
        folder.mkdir()
        logpath = OUT/(r['ID']+'-decode.log')
        with logpath.open('w', encoding='utf-8') as log:
            subprocess.run([str(FF), '-hide_banner', '-nostdin', '-n', '-threads', '1', '-i', str(source),
                            '-map', '0:v:0', '-an', '-vf', 'showinfo', '-fps_mode', 'passthrough',
                            '-q:v', '2', '-threads', '1', '-start_number', '0', str(folder/'frame_%06d.jpg')],
                           stdout=subprocess.DEVNULL, stderr=log, timeout=180, check=True)
        num, den, pts = parse_frames(logpath.read_text(encoding='utf-8'))
        paths = sorted(folder.glob('*.jpg'))
        assert len(paths) == len(pts) <= 10000
        map_path = DATA/'frame-maps'/(r['ID']+'.csv')
        size = 0
        dimensions = set()
        with map_path.open('w', encoding='utf-8', newline='') as f:
            fields = ['frame_index', 'pts', 'time_base_num', 'time_base_den', 'first_pts', 'timestamp_seconds', 'file', 'sha256', 'bytes', 'width', 'height']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, path in enumerate(paths):
                assert path.name == f'frame_{i:06d}.jpg'
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                dimensions.add((width, height))
                amount = path.stat().st_size
                size += amount
                writer.writerow(dict(frame_index=i, pts=pts[i], time_base_num=num, time_base_den=den,
                                     first_pts=pts[0], timestamp_seconds=format(float(Fraction((pts[i]-pts[0])*num, den)), '.12f'),
                                     file=str(path.relative_to(DATA)).replace('\\', '/'), sha256=sha(path), bytes=amount, width=width, height=height))
        assert len(dimensions) == 1
        total_bytes += size
        total_frames += len(pts)
        if total_bytes > protocol['max_jpeg_bytes'] or shutil.disk_usage(DATA).free < 5*1024**3:
            raise RuntimeError('Storage budget reached; partial inputs retained for inspection')
        videos.append(dict(ID=r['ID'], frames=len(pts), frame_map=str(map_path.relative_to(DATA)).replace('\\', '/'),
                           frame_map_sha256=sha(map_path), video_file=str(source.relative_to(ROOT)).replace('\\', '/'),
                           video_sha256=sha(source), jpeg_bytes=size, width=width, height=height, recording_source='UNKNOWN'))
        save(OUT/'status.json', dict(status='PREPARING', completed=len(videos), total=50, frames=total_frames, bytes=total_bytes))
        save(DATA/'manifest.json', dict(status='PREPARING', videos=videos))
        print(r['ID'], len(pts), 'frames', flush=True)
    save(DATA/'manifest.json', dict(status='INPUTS_READY_UNLABELED', videos=videos, total_frames=total_frames, jpeg_bytes=total_bytes))
    with (DATA/'review-decisions.csv').open('x', encoding='utf-8', newline='') as f:
        fields = ['ID', 'review_status', 'exclusion_reason', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side', 'recording_source', 'reviewer', 'note']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(ID=r['ID'], review_status='PENDING', recording_source='UNKNOWN'))
    # The initial template is a blank worksheet, never treated as ground truth.
    save(OUT/'input-validation.json', dict(status='PASS', videos=50, frames=total_frames, jpeg_bytes=total_bytes,
                                          checks=['All source video hashes match acquisition', 'All JPEGs fully decoded and hashed', 'Frame indices and decoder PTS are contiguous/monotonic', 'No frame rate resampling', 'All 50 reserved IDs retained'],
                                          manifest_sha256=sha(DATA/'manifest.json'), blank_template_sha256=sha(DATA/'review-decisions.csv'),
                                          seconds=time.monotonic()-started))
    save(OUT/'status.json', dict(status='INPUTS_READY_UNLABELED', completed=50, total=50, frames=total_frames, bytes=total_bytes, human_labels=0))


if __name__ == '__main__':
    prepare()
