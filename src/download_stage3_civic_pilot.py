"""Bounded HTTP-range acquisition from an immutable official comma.ai ZIP."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import time
import zipfile
import zlib

import requests

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'data_raw/comma2k19/civic-pilot-20260914'
STATE = PROJECT/'artifacts/stage3-civic-acquisition-20260914'
API = 'https://huggingface.co/api/datasets/commaai/comma2k19'
SEED = 20260914


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Remote(io.RawIOBase):
    def __init__(self, url, size):
        self.url,self.size,self.pos,self.received=url,size,0,0
        self.session=requests.Session()
        self.started=time.monotonic()

    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos

    def fetch(self, start, length):
        assert 0 <= start < self.size and 0 < length <= 128*1024**2
        assert start+length <= self.size
        assert time.monotonic()-self.started < 3300
        assert self.received+length <= 1024**3, '1GiB transfer budget exceeded'
        for attempt in range(3):
            try:
                # Per-range query prevents stale CDN range caches; revision remains fixed.
                url=self.url+f'?download=true&range_start={start}&range_length={length}'
                with self.session.get(url,headers={'Range':f'bytes={start}-{start+length-1}',
                                                  'Accept-Encoding':'identity'},timeout=(20,60),stream=True) as r:
                    assert r.status_code==206, f'Expected partial content, got {r.status_code}'
                    assert r.headers.get('Content-Range')==f'bytes {start}-{start+length-1}/{self.size}'
                    data=bytearray()
                    for block in r.iter_content(1024*1024):
                        data.extend(block)
                        self.received+=len(block)
                        assert len(data)<=length and self.received<=1024**3
                        assert time.monotonic()-self.started<3300
                    assert len(data)==length
                    return bytes(data)
            except (requests.RequestException,AssertionError) as exc:
                if attempt==2: raise RuntimeError(f'Range {start}/{length} failed: {type(exc).__name__}') from None
                time.sleep(2*(attempt+1))

    def read(self, n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n==0:return b''
        data=self.fetch(self.pos,n); self.pos+=len(data);return data


def prepare():
    STATE.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    assert not (STATE/'plan.json').exists(), 'Plan already exists; do not reselect'
    info=requests.get(API,timeout=30);info.raise_for_status();revision=info.json()['sha']
    listing=requests.get(API+f'/tree/{revision}/raw_data',timeout=30);listing.raise_for_status()
    metadata=next(r for r in listing.json() if r['path']=='raw_data/Chunk_3.zip')
    url=f'https://huggingface.co/datasets/commaai/comma2k19/resolve/{revision}/raw_data/Chunk_3.zip'
    remote=Remote(url,metadata['size'])
    with zipfile.ZipFile(remote) as archive:
        entries={i.filename:i for i in archive.infolist()}
        segments=sorted(n[:-11] for n in entries if n.endswith('/video.hevc'))
        old=json.loads((PROJECT/'artifacts/stage3-comma-chunk1/manifest.json').read_text())
        old_routes={r['source_route'].split('/')[-1] for r in old}
        old_devices={r.split('|')[0] for r in old_routes}
        eligible=[s for s in segments if s.split('/')[-2] not in old_routes and s.split('/')[-2].split('|')[0] not in old_devices]
        routes=sorted({s.rsplit('/',1)[0] for s in eligible})
        assert routes and all(r.split('/')[-1].startswith('99c94dc769b5d96e|') for r in routes)
        rng=random.Random(SEED); chosen_routes=sorted(rng.sample(routes,min(5,len(routes))))
        selected=[]
        for route in chosen_routes:
            pool=sorted(s for s in eligible if s.rsplit('/',1)[0]==route)
            selected.extend(sorted(rng.sample(pool,min(2,len(pool)))))
        files=[]
        for i,segment in enumerate(selected):
            prefix=segment+'/'
            speed='speed' if prefix+'processed_log/CAN/speed/t' in entries else 'car_speed'
            required=['video.hevc','global_pose/frame_times',f'processed_log/CAN/{speed}/t',
                      f'processed_log/CAN/{speed}/value','processed_log/CAN/steering_angle/t',
                      'processed_log/CAN/steering_angle/value']
            optional=['global_pose/frame_positions','global_pose/frame_orientations','global_pose/frame_velocities',
                      'processed_log/IMU/gyro/t','processed_log/IMU/gyro/value']
            assert all(prefix+n in entries for n in required)
            for name in required+[n for n in optional if prefix+n in entries]:
                entry=entries[prefix+name]
                assert entry.compress_type in (0,8) and not entry.flag_bits & 1
                files.append(dict(ID=f'CIVIC_PILOT_{i:03}',segment=segment,member=entry.filename,
                                  relative=name,header_offset=entry.header_offset,compressed_size=entry.compress_size,
                                  size=entry.file_size,crc32=entry.CRC,compression=entry.compress_type))
        assert sum(f['compressed_size'] for f in files)<900*1024**2
        save(STATE/'plan.json',dict(source=url,revision=revision,archive=metadata,seed=SEED,
                                   routes=chosen_routes,segments=selected,files=files,
                                   transfer_bytes_estimate=sum(f['compressed_size'] for f in files),
                                   source_index_transfer_bytes=remote.received,script_sha256=sha(Path(__file__).read_bytes()),
                                   use='Evaluation pilot reservation; no training or predictions. Sensor semantics not yet calibrated.',
                                   limits='1GiB/run,60min worker,3 attempts/range,one worker,OS lock',
                                   scope='New device and recording route IDs; same highway dataset, not geographic independence. Partial ZIP CRC32 and local SHA256; full archive SHA256 not verified.'))
    save(STATE/'status.json',dict(status='PREPARED',completed=0,total=len(selected)))
    print(json.dumps(dict(segments=len(selected),routes=len(chosen_routes),files=len(files),
                          compressed_bytes=sum(f['compressed_size'] for f in files)),indent=2))


def validate_segment(folder):
    import numpy as np
    import cv2
    times=np.load(folder/'global_pose/frame_times',allow_pickle=False).reshape(-1)
    assert len(times)>1 and np.isfinite(times).all() and np.all(np.diff(times)>0)
    sensor={}
    for key in ['speed','car_speed','steering_angle']:
        path=folder/'processed_log/CAN'/key
        if not path.exists():continue
        t=np.load(path/'t',allow_pickle=False).reshape(-1)
        v=np.load(path/'value',allow_pickle=False).reshape(-1)
        assert len(t)==len(v)>1 and np.isfinite(t).all() and np.isfinite(v).all()
        assert np.all(np.diff(t)>=0)
        sensor[key]=dict(samples=len(t),start=float(t[0]),end=float(t[-1]),min=float(v.min()),max=float(v.max()))
    cap=cv2.VideoCapture(str(folder/'video.hevc'));count=0
    assert cap.isOpened()
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame is not None and frame.size
        count+=1
    cap.release();assert count==len(times),f'Decode/timestamp mismatch {count}/{len(times)}'
    lo=max(s['start'] for s in sensor.values());hi=min(s['end'] for s in sensor.values())
    outside=float(np.mean((times[::2]<lo)|(times[::2]>hi)))
    return dict(decoded_frames=count,frame_time_count=len(times),median_frame_interval=float(np.median(np.diff(times))),
                can_outside_fraction=outside,sensors=sensor,sensor_coverage_pass=outside<=.01,
                label_status='UNCALIBRATED_NO_LABELS')


def run():
    plan=json.loads((STATE/'plan.json').read_text(encoding='utf-8'))
    assert sha(Path(__file__).read_bytes())==plan['script_sha256']
    remote=Remote(plan['source'],plan['archive']['size']); records=[];reports=[]
    for i,segment in enumerate(plan['segments']):
        sid=f'CIVIC_PILOT_{i:03}';folder=ROOT/sid
        save(STATE/'status.json',dict(status='DOWNLOADING',pid=os.getpid(),completed=i,total=len(plan['segments']),current=sid,received_bytes=remote.received))
        for f in [f for f in plan['files'] if f['segment']==segment]:
            path=folder/f['relative'];assert path.resolve().is_relative_to(ROOT.resolve())
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                data=path.read_bytes()
            else:
                header=remote.fetch(f['header_offset'],30)
                assert header[:4]==b'PK\x03\x04'
                name_len,extra_len=struct.unpack_from('<HH',header,26)
                payload=remote.fetch(f['header_offset']+30,name_len+extra_len+f['compressed_size'])
                assert payload[:name_len].decode('utf-8')==f['member']
                compressed=payload[name_len+extra_len:]
                data=zlib.decompress(compressed,-15) if f['compression']==8 else compressed
            assert len(data)==f['size'] and zlib.crc32(data)&0xffffffff==f['crc32']
            if not path.exists():
                tmp=path.with_suffix('.part');tmp.write_bytes(data);tmp.replace(path)
            records.append(dict(**f,sha256=sha(data),local=str(path.relative_to(PROJECT))))
        validation=validate_segment(folder)
        report=dict(ID=sid,segment=segment,**validation);reports.append(report)
        save(folder/'source_manifest.json',dict(source=plan['source'],segment=segment,validation=validation))
        save(STATE/'progress.json',dict(files=records,segments=reports))
        print(f'Validated {i+1}/{len(plan["segments"])}: {sid}',flush=True)
    # Re-read all saved payloads and independently compare lengths, CRC and SHA.
    for r in records:
        data=(PROJECT/r['local']).read_bytes()
        assert len(data)==r['size'] and sha(data)==r['sha256'] and zlib.crc32(data)&0xffffffff==r['crc32']
    save(STATE/'validation.json',dict(status='PASS',files=records,segments=reports,received_bytes=remote.received,
                                      checks='All selected member CRC32/size; saved SHA256 reread; full HEVC decode equals frame timestamps; finite sensor arrays and ordered times. No calibrated steering labels.',
                                      scope=plan['scope']))
    save(STATE/'status.json',dict(status='ACQUIRED_VALIDATED_UNCALIBRATED',completed=len(reports),total=len(reports),
                                 files=len(records),received_bytes=remote.received,coverage_pass=sum(r['sensor_coverage_pass'] for r in reports)))


def watch():
    import msvcrt
    with (STATE/'watch.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if (STATE/'validation.json').exists():raise RuntimeError('Already complete')
        try:
            with (STATE/'run.log').open('a',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                save(STATE/'watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=proc.pid,timeout_seconds=3600))
                try:rc=proc.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],check=True,capture_output=True,timeout=30)
                    raise
                if rc:raise RuntimeError(f'Worker exit {rc}; see run.log')
            save(STATE/'watch-status.json',dict(status='COMPLETE'))
        except Exception as exc:
            save(STATE/'watch-status.json',dict(status='FAILED',error=str(exc)))
            save(STATE/'status.json',dict(status='FAILED',error=str(exc)))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','watch']);a=p.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS','1')
    globals()[a.command]()
