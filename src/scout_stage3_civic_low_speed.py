"""Bounded sensor-only scout for disjoint low-speed Civic recording routes."""
import csv
import io
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from download_stage3_civic_pilot import Remote, save, sha

P = Path(__file__).resolve().parents[1]
OUT = P / 'artifacts/stage3-civic-low-speed-scout-20260916'
SOURCE = P / 'artifacts/stage3-civic-holdout-acquisition-20260914/plan.json'
PILOT = P / 'artifacts/stage3-civic-acquisition-20260914/plan.json'
SEED = 20260916


def metadata():
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    assert source['revision'] == '4bff77c7254c654c28d4c2726186b4e825adccee'
    return source


def open_archive():
    s = metadata()
    remote = Remote(s['source'], s['archive']['size'])
    return remote, zipfile.ZipFile(remote)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'plan.json').exists(), 'Existing reservation; do not reselect'
    remote, archive = open_archive()
    try:
        names = set(archive.namelist())
        routes = {}
        for name in names:
            if name.endswith('/video.hevc'):
                seg = name[:-len('/video.hevc')]
                route = seg.rsplit('/', 1)[0]
                prefix = seg+'/processed_log/CAN/'
                speed = 'speed' if prefix+'speed/t' in names else 'car_speed'
                if all(seg+'/'+n in names for n in ('global_pose/frame_times', f'processed_log/CAN/{speed}/t', f'processed_log/CAN/{speed}/value')):
                    routes.setdefault(route, []).append((seg, speed))
        excluded = set(json.loads(PILOT.read_text(encoding='utf-8'))['routes']) | set(metadata()['routes'])
        pool = sorted(r for r in routes if r not in excluded and '99c94dc769b5d96e|' in r)
        assert len(pool) >= 3
        rng = random.Random(SEED)
        chosen = pool
        selected = []
        for route in chosen:
            parts = sorted(routes[route])
            selected.extend(sorted(rng.sample(parts, min(2, len(parts)))))
        plan = dict(status='RESERVED_SENSOR_ONLY', source=metadata()['source'], revision=metadata()['revision'],
                    archive_size=metadata()['archive']['size'], seed=SEED, routes=chosen,
                    segments=[dict(segment=seg, speed_channel=speed) for seg,speed in selected],
                    excluded_routes=sorted(excluded), source_plan_sha256=sha(SOURCE.read_bytes()),
                    pilot_plan_sha256=sha(PILOT.read_bytes()), script_sha256=sha(Path(__file__).read_bytes()),
                    index_transfer_bytes=remote.received,
                    limits='All11disjoint Chunk3 routes/up to22segments,1GiB transfer,30min watcher,3 finite range attempts,OS lock,no retry',
                    scope='Public immutable Chunk_3 sensor scout. No video, model predictions, labels or held-route fitting.')
        save(OUT/'plan.json', plan)
        save(OUT/'status.json', dict(status='PREPARED', completed=0, total=len(selected)))
        print(json.dumps(dict(routes=len(chosen), segments=len(selected), index_bytes=remote.received)), flush=True)
    finally:
        archive.close()


def decode(archive, name):
    return np.asarray(np.load(io.BytesIO(archive.read(name)), allow_pickle=False)).reshape(-1)


def run():
    plan = json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    assert sha(SOURCE.read_bytes()) == plan['source_plan_sha256']
    assert sha(PILOT.read_bytes()) == plan['pilot_plan_sha256']
    assert sha(Path(__file__).read_bytes()) == plan['script_sha256']
    assert not (OUT/'execution.json').exists(), 'Existing execution; no automatic retry'
    save(OUT/'execution.json', dict(status='STARTED', pid=os.getpid()))
    remote, archive = open_archive()
    rows = []
    try:
        assert metadata()['source'] == plan['source']
        for i, item in enumerate(plan['segments']):
            seg = item['segment']; channel = item['speed_channel']
            ft = decode(archive, seg+'/global_pose/frame_times')
            st = decode(archive, seg+f'/processed_log/CAN/{channel}/t')
            sv = decode(archive, seg+f'/processed_log/CAN/{channel}/value')
            assert len(ft) >= 200 and len(st) == len(sv) >= 2
            assert np.isfinite(ft).all() and np.isfinite(st).all() and np.isfinite(sv).all()
            assert (np.diff(ft) > 0).all() and (np.diff(st) >= 0).all()
            t = ft[::2]
            valid = (t >= st[0]) & (t <= st[-1]) & (t >= ft[0]+1) & (t <= ft[-1]-1)
            speed = np.interp(t, st, sv)
            low = valid & (speed >= 2) & (speed < 5)
            runs = []
            edges = np.flatnonzero(np.diff(np.r_[False, low, False]))
            for start,end in edges.reshape(-1,2):
                runs.append(int(end-start))
            rows.append(dict(segment=seg, route=seg.rsplit('/',1)[0], frames_10hz=len(t), sensor_valid=int(valid.sum()),
                             low_2_5=int(low.sum()), low_2_5_longest=max(runs, default=0),
                             low_0_5_2=int((valid & (speed >= .5) & (speed < 2)).sum()),
                             stopped=int((valid & (speed < .5)).sum()), speed_min=float(speed.min()), speed_max=float(speed.max())))
            save(OUT/'status.json', dict(status='SCOUTING', completed=i+1, total=len(plan['segments']), pid=os.getpid()))
            print(f"scouted {i+1}/{len(plan['segments'])} low={rows[-1]['low_2_5']}", flush=True)
        path = OUT/'segment-speed-inventory.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        routes = {}
        for row in rows: routes.setdefault(row['route'], []).append(row)
        candidates = [dict(route=r, segments=len(g), low_2_5=sum(x['low_2_5'] for x in g),
                           longest=max(x['low_2_5_longest'] for x in g)) for r,g in routes.items()]
        candidates.sort(key=lambda x:(-x['low_2_5'],-x['longest'],x['route']))
        result = dict(status='COMPLETE_VALIDATED', selected_routes=len(routes), selected_segments=len(rows),
                      transfer_bytes=remote.received, candidates=candidates,
                      suitable_routes=[x for x in candidates if x['segments'] >= 2 and x['low_2_5'] >= 100 and x['longest'] >= 10],
                      validation='Immutable revision, disjoint routes, all selected speed/frame timestamps finite and ordered; no held labels or model outputs used.',
                      inventory_sha256=sha(path.read_bytes()))
        save(OUT/'report.json', result)
        save(OUT/'status.json', dict(status='COMPLETE_VALIDATED', completed=len(rows), total=len(rows), suitable_routes=len(result['suitable_routes'])))
        print(json.dumps(dict(status=result['status'], suitable_routes=len(result['suitable_routes']), transfer_bytes=remote.received)), flush=True)
    finally:
        archive.close()


def watch():
    import msvcrt
    with (OUT/'watch.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not (OUT/'execution.json').exists(), 'Existing execution; no duplicate or retry'
        try:
            with (OUT/'run.log').open('w',encoding='utf-8') as log:
                child = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=P,stdout=log,stderr=subprocess.STDOUT)
                save(OUT/'watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=child.pid))
                try: rc = child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=True,timeout=30)
                    raise RuntimeError('Scout timed out after 30min')
                if rc: raise RuntimeError(f'Scout exit {rc}; see run.log')
            assert json.loads((OUT/'report.json').read_text(encoding='utf-8'))['status']=='COMPLETE_VALIDATED'
            save(OUT/'watch-status.json',dict(status='COMPLETE'))
        except Exception as exc:
            save(OUT/'status.json',dict(status='FAILED',error=str(exc)))
            save(OUT/'watch-status.json',dict(status='FAILED',error=str(exc)))
            raise


if __name__=='__main__':
    {'prepare':prepare,'run':run,'watch':watch}[sys.argv[1]]()
