"""Train-only matched-quality Stage1 pilot. No fitting or submission changes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import shutil
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import cv2
import numpy as np
import pandas as pd
import prepare_stage1_rerecorded as synth
from prepare_stage1_robustness import GROUPS, digest, transform, write_json
from prepare_stage1_training import plan as original_plan
from prepare_stage1_full_run import inspect_video
from advance_stage1_release import tick_lock

ROOT=Path(__file__).resolve().parents[1]
SEED=20260911
PROFILES=('clean','noise','quality_mix')
LABELS=('ORIGINAL','RERECORDED')


class LimitedH264Writer(synth.H264Writer):
    def __init__(self, destination, width, height, fps, crf):
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([
            ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', f'{fps:g}', '-i', '-', '-an',
            '-filter_threads', '1', '-c:v', 'libx264', '-threads', '1',
            '-preset', 'fast', '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(destination)], stdin=subprocess.PIPE)


def low_memory_transform(frame, index, fps, values, fields, rng):
    """Equivalent effect order to training, with exact bypasses for controls."""
    result = frame.copy()
    if (values['perspective_x'], values['perspective_y'], values['scale']) != (0., 0., 1.):
        result = cv2.warpPerspective(result, fields[0], (frame.shape[1], frame.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    result = result.astype(np.float32) / 255.
    time_s = index / fps
    result *= 1. + values['flicker_amplitude'] * math.sin(2 * math.pi * values['flicker_hz'] * time_s)
    phase = int(round(values['moire_phase_speed'] * index))
    result *= 1. + values['moire_amplitude'] * np.roll(fields[1], phase, axis=1)[:, :, None]
    band_center = ((time_s * values['band_speed']) % 1.4 - .2) * frame.shape[0]
    band = np.exp(-.5 * ((np.arange(frame.shape[0], dtype=np.float32) - band_center)
                        / (values['band_width'] * frame.shape[0])) ** 2)
    result *= 1. - values['band_amplitude'] * band[:, None, None]
    result += values['reflection_amplitude'] * fields[2][:, :, None]
    result[:, :, 2] *= values['red_gain']
    result[:, :, 0] *= values['blue_gain']
    result = np.clip(result, 0., 1.) ** (1. / values['gamma'])
    if values['blur_sigma'] > .2:
        result = cv2.GaussianBlur(result, (0, 0), values['blur_sigma'])
    # Consume identical random arrays even when noise is disabled.
    for first in range(0, result.shape[0], 32):
        strip = result[first:first + 32]
        strip += rng.normal(0., values['noise_sigma'] / 255., strip.shape).astype(np.float32)
    result = np.clip(result * 255., 0, 255).astype(np.uint8)
    if values['jpeg_quality'] is not None:
        ok, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, values['jpeg_quality']])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if result is None:
            raise RuntimeError('JPEG decoding failed')
    return result


def matched_effects(base, profile, label):
    if profile not in PROFILES or label not in LABELS:
        raise ValueError('Unknown profile/label')
    values=asdict(base)
    if label=='ORIGINAL':
        for neutral in GROUPS.values(): values.update(neutral)
    for effect in ('blur','noise','jpeg'):
        values.update(GROUPS[effect])
        if profile=='quality_mix' or (profile=='noise' and effect=='noise'):
            for key in GROUPS[effect]:values[key]=getattr(base,key)
    return values


def choose_sources(table, canonical, excluded):
    if table.groupby('source_id').split.nunique().max()!=1 or table.groupby('upload_group').split.nunique().max()!=1:
        raise ValueError('Split leakage')
    if set(table.source_id)&set(excluded):raise ValueError('Phone leakage')
    expected={r['source_id']:r for r in canonical}
    originals=table[table.variant==0]
    if originals.source_id.duplicated().any() or set(originals.source_id)!=set(expected):raise ValueError('Source coverage changed')
    for r in originals.to_dict('records'):
        if any(r[k]!=expected[r['source_id']][k] for k in ('split','upload_group')):raise ValueError('Canonical split mismatch')
    selected=[]
    for _,group in originals[originals.split=='train'].groupby('upload_group',sort=True):
        rows=group.to_dict('records')
        rows.sort(key=lambda r: hashlib.sha256(f"{SEED}:quality-train:{r['source_id']}".encode()).hexdigest())
        selected.append({k:rows[0][k] for k in ('source_id','upload_group','source_sha256')})
    return selected


def code_digest():
    return hashlib.sha256(b''.join(Path(p).read_bytes() for p in (__file__,synth.__file__,ROOT/'src/prepare_stage1_robustness.py',ROOT/'src/prepare_stage1_training.py',ROOT/'src/prepare_stage1_full_run.py'))).hexdigest()


def make_plan():
    training=ROOT/'artifacts/stage1-full-20260910'
    report=json.loads((training/'generation_report.json').read_text())
    if report['status']!='PASS' or report['manifest_sha256']!=digest(training/'generation_manifest.csv'):raise ValueError('Training manifest changed')
    table=pd.read_csv(training/'generation_manifest.csv',dtype={'source_id':str,'upload_group':str})
    canonical,phone,excluded=original_plan(ROOT/'data_raw/ccd/videos/Crash-1500',ROOT/'data_raw/ccd/Crash-1500.txt')
    selected=choose_sources(table,canonical,excluded)
    if len(selected)!=85:raise ValueError('Expected 85 training groups')
    return dict(seed=SEED,scope='TRAIN_ONLY_PILOT',sources=selected,profiles=list(PROFILES),labels=list(LABELS),
                expected_videos=len(selected)*6,crf=24,training_manifest_sha256=report['manifest_sha256'],
                generator_sha256=code_digest(),phone_sources=len(phone),excluded_sources=len(excluded),
                label_basis='CCD inherited ORIGINAL proxy; synthetic RERECORDED',training_started=False)


def verify_rows(out, row, rows):
    if [(r['profile'],r['label']) for r in rows]!=[(p,l) for p in PROFILES for l in LABELS]:raise ValueError('Pair coverage mismatch')
    seed=int.from_bytes(hashlib.sha256(f"{SEED}:{row['source_id']}:quality-pair".encode()).digest()[:4],'big')
    base=synth._effect_parameters(random.Random(seed))
    for item in rows:
        if item['seed']!=seed or item['crf']!=24 or json.loads(item['effects'])!=matched_effects(base,item['profile'],item['label']):
            raise ValueError('Effect metadata differs from fixed plan')
        if any(item[k]!=row[k] for k in ('source_id','upload_group','source_sha256')) or item['split']!='train':raise ValueError('Provenance mismatch')
        expected=f"videos/{row['source_id']}_{item['profile']}_{item['label']}.mp4"
        if item['output_path']!=expected:raise ValueError('Unexpected output path')
        inspect_video(out/expected,item)
    for profile in PROFILES:
        pair=[r for r in rows if r['profile']==profile]
        a,b=[json.loads(r['effects']) for r in pair]
        if any(a[k]!=b[k] for k in ('noise_sigma','blur_sigma','jpeg_quality')) or pair[0]['seed']!=pair[1]['seed'] or pair[0]['crf']!=pair[1]['crf']:raise ValueError('Quality mismatch')


def render_source(out,row,plan):
    cv2.setNumThreads(1)
    source=ROOT/'data_raw/ccd/videos/Crash-1500'/(row['source_id']+'.mp4')
    if digest(source)!=row['source_sha256'] or code_digest()!=plan['generator_sha256']:raise ValueError('Source/code changed')
    record=out/'records'/(row['source_id']+'.json')
    identity=dict(row,plan_sha256=digest(out/'plan.json'))
    if record.exists():
        saved=json.loads(record.read_text())
        if saved['identity']!=identity:raise ValueError('Resume identity mismatch')
        verify_rows(out,row,saved['rows'])
        return saved['rows']
    results=[]
    seed=int.from_bytes(hashlib.sha256(f"{SEED}:{row['source_id']}:quality-pair".encode()).digest()[:4],'big')
    base=synth._effect_parameters(random.Random(seed))
    for profile in PROFILES:
        for label in LABELS:
            values=matched_effects(base,profile,label)
            cap,w,h,fps,expected=synth._open_video(source)
            effects=synth.Effects(**values)
            fields=(synth._perspective_matrix(w,h,effects),*synth._spatial_fields(w,h,effects))
            relative=f"videos/{row['source_id']}_{profile}_{label}.mp4"
            writer=LimitedH264Writer(out/relative,w,h,fps,plan['crf'])
            rng=np.random.default_rng(seed);jitter=random.Random(seed);count=0;previous=None
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    frame=low_memory_transform(frame,count,fps,values,fields,rng)
                    if previous is not None and jitter.random()<values['temporal_jitter_probability']:frame=previous.copy()
                    writer.write(frame);previous=frame;count+=1
            finally:
                cap.release();writer.close()
            if count!=expected or count<1:raise ValueError('Incomplete render')
            results.append(dict(row,split='train',profile=profile,label=label,output_path=relative,
                                source_path=str(source),seed=seed,crf=plan['crf'],frames=count,width=w,height=h,fps=fps,
                                sha256=digest(out/relative),effects=json.dumps(values,sort_keys=True)))
    verify_rows(out,row,results)
    write_json(record,dict(identity=identity,rows=results))
    return results


def generate(out):
    plan=make_plan()
    if (out/'plan.json').exists():
        if json.loads((out/'plan.json').read_text())!=plan:raise ValueError('Plan changed')
    else:write_json(out/'plan.json',plan)
    state=dict(status='GENERATING',pid=os.getpid(),expected_sources=len(plan['sources']),expected_videos=plan['expected_videos'],
               completed_sources=0,completed_videos=0,started_unix=time.time(),scope=plan['scope'],training_started=False,uploaded=False)
    write_json(out/'status.json',state)
    all_rows=[];deadline=time.monotonic()+6*3600
    try:
        for row in plan['sources']:
            command=[sys.executable,str(Path(__file__).resolve()),'--output-dir',str(out),'--worker-source',row['source_id']]
            with (out/'logs'/(row['source_id']+'.log')).open('a',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                try:
                    code=child.wait(timeout=max(1,min(900,deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    from supervise_stage1_robustness import stop_child
                    stop_child(child)
                    raise TimeoutError('Source generation timeout or six-hour budget reached')
                if code:raise RuntimeError(f"Source {row['source_id']} failed: {code}; see logs")
            saved=json.loads((out/'records'/(row['source_id']+'.json')).read_text())
            if saved['identity']!=dict(row,plan_sha256=digest(out/'plan.json')):raise ValueError('Worker record mismatch')
            verify_rows(out,row,saved['rows'])
            all_rows.extend(saved['rows'])
            state.update(completed_sources=state['completed_sources']+1,completed_videos=len(all_rows),heartbeat_unix=time.time())
            write_json(out/'status.json',state)
            if time.monotonic()>deadline:raise TimeoutError('Six-hour budget reached')
        table=pd.DataFrame(all_rows)
        if len(table)!=plan['expected_videos'] or table.output_path.duplicated().any() or table.label.value_counts().to_dict()!={'ORIGINAL':255,'RERECORDED':255}:raise ValueError('Final coverage mismatch')
        table.to_csv(out/'generation_manifest.csv',index=False)
        table[['output_path','label','split']].rename(columns={'output_path':'path'}).to_csv(out/'labels_train.csv',index=False)
        state.update(status='DATA_VALIDATED',manifest_sha256=digest(out/'generation_manifest.csv'),plan_sha256=digest(out/'plan.json'),
                     decoded_frames=int(table.frames.sum()),finished_unix=time.time(),validation='all frames/hash/provenance/matched quality/coverage PASS')
        write_json(out/'status.json',state)
    except Exception as error:
        state.update(status='FAILED',error=repr(error),finished_unix=time.time());write_json(out/'status.json',state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--worker-source')
    args=parser.parse_args();out=args.output_dir.resolve()
    for folder in ('records','videos','logs'):(out/folder).mkdir(parents=True,exist_ok=True)
    if args.worker_source:
        plan=json.loads((out/'plan.json').read_text());row=next(r for r in plan['sources'] if r['source_id']==args.worker_source)
        render_source(out,row,plan);return
    with tick_lock(out/'generation.lock') as acquired:
        if not acquired:raise RuntimeError('Generation already running')
        import ctypes
        awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:generate(out)
        finally:
            if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
