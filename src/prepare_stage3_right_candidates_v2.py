"""Source-disjoint CCD right-turn candidate screening; never creates truth labels."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage3-right-candidates-20260914'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def scan():
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    rows = []
    meta = ROOT / 'data_raw/ccd/Crash-1500.txt'
    for line in meta.read_text().splitlines():
        sid = line.split(',', 1)[0]
        tail = line.split('],', 1)[1].split(',')
        rows.append(dict(source_id=sid, youtube_id=tail[1], timing=tail[2], weather=tail[3], ego=tail[4]))
    by_id = {r['source_id']: r for r in rows}
    exclusions = {}
    input_hashes = {str(meta.relative_to(ROOT)): sha(meta)}
    for folder in ('stage3-human-review', 'stage3-right-review'):
        path = ROOT / 'data' / folder / 'sources.json'
        input_hashes[str(path.relative_to(ROOT))] = sha(path)
        exclusions[folder] = [r['source_id'] for r in json.loads(path.read_text(encoding='utf-8-sig'))['videos']]
    path = ROOT / 'artifacts/stage3-blind-review-20260911/source-manifest.csv'
    input_hashes[str(path.relative_to(ROOT))] = sha(path)
    with path.open() as stream:
        exclusions['previous_blind_pool'] = [r['original_ID'] for r in csv.DictReader(stream)]
    path = ROOT / 'artifacts/stage1-full-20260910/phone_holdout.csv'
    input_hashes[str(path.relative_to(ROOT))] = sha(path)
    with path.open() as stream:
        exclusions['phone_holdout'] = [r['source_id'] for r in csv.DictReader(stream)]
    exclusions['public_ccd_examples'] = [f'{i:06}' for i in range(1, 6)]
    excluded_ids = set(sum(exclusions.values(), []))
    excluded_youtube = {by_id[sid]['youtube_id'] for sid in excluded_ids}
    pool = [r for r in rows if r['youtube_id'] not in excluded_youtube]
    write('plan.json', dict(inputs=input_hashes, excluded_ids_by_reason=exclusions,
                           excluded_youtube_ids=sorted(excluded_youtube), pool_videos=len(pool),
                           pool_sources=len({r['youtube_id'] for r in pool}),
                           method='Rank negative median horizontal Farneback flow (background moving left); frames 0,4,...,28; ROI x10:150/y25:80 of 160x90. Select one video per source for visual screening. No steering model predictions.',
                           scope='Biased candidate mining only, not labels or independent evaluation. CCD source id can denote a compilation, not camera identity.'))
    print('POOL', len(pool), 'videos', len({r['youtube_id'] for r in pool}), 'sources', flush=True)
    ranked = []
    for n, row in enumerate(pool, 1):
        path = ROOT / 'data_raw/ccd/videos/Crash-1500' / (row['source_id'] + '.mp4')
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS) - 10) < .01
        frames = []
        for i in range(50):
            ok, frame = cap.read()
            assert ok
            if i in range(0, 29, 4):
                frames.append(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY))
        ok, _ = cap.read()
        assert not ok
        cap.release()
        shifts = []
        for a, b in zip(frames[:-1], frames[1:]):
            flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
            shifts.append(float(np.median(flow[25:80, 10:150, 0])))
        ranked.append(dict(**row, sha256=sha(path), score=-float(np.median(shifts)),
                           negative_steps=sum(x < 0 for x in shifts), shifts=shifts))
        if n == 1 or n % 25 == 0:
            write('status.json', dict(status='SCANNING', pid=os.getpid(), completed=n, total=len(pool)))
    ranked.sort(key=lambda r: (-r['score'], r['source_id']))
    write('ranking.json', ranked)
    seen, shortlist = set(), []
    for row in ranked:
        if row['youtube_id'] not in seen:
            shortlist.append(row)
            seen.add(row['youtube_id'])
        if len(shortlist) == 36:
            break
    write('shortlist.json', shortlist)
    for batch_start in range(0, len(shortlist), 6):
        lines = []
        for r in shortlist[batch_start:batch_start+6]:
            cap = cv2.VideoCapture(str(ROOT / 'data_raw/ccd/videos/Crash-1500' / (r['source_id']+'.mp4')))
            tiles = []
            for i in [0, 8, 16, 24, 32, 40]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                assert ok
                tile = np.zeros((202, 320, 3), np.uint8)
                tile[:180] = cv2.resize(frame, (320, 180))
                cv2.putText(tile, f'{r["source_id"]} f{i} src{r["youtube_id"]}', (4, 195), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
                tiles.append(tile)
            cap.release()
            lines.append(np.hstack(tiles))
        assert cv2.imwrite(str(OUT / f'screen-{batch_start//6+1}.jpg'), np.vstack(lines))
    for path, expected in input_hashes.items():
        assert sha(ROOT / path) == expected
    write('status.json', dict(status='SCREENING_READY', scanned=len(pool), shortlist=len(shortlist), human_labels=0))
    print('SCREENING_READY', len(shortlist), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.worker:
        scan()
        return
    import msvcrt
    with (OUT / 'watch.lock').open('a+b') as lock:
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        assert not (OUT / 'ranking.json').exists(), 'Already scanned; inspect existing results'
        write('status.json', dict(status='STARTING', supervisor_pid=os.getpid(), timeout_seconds=600, retries=0))
        try:
            with (OUT / 'run.log').open('w', encoding='utf-8') as stream:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            if result.returncode:
                raise RuntimeError(f'worker exit {result.returncode}')
            print((OUT / 'run.log').read_text(encoding='utf-8'))
        except Exception as exc:
            write('status.json', dict(status='FAILED', error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
