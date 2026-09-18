"""Train-only weak/ablated screen cues; bounded local generation, no GPU/upload."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import prepare_stage1_quality_balanced as quality
from prepare_stage1_robustness import GROUPS,digest,write_json
from prepare_stage1_full_run import inspect_video
from advance_stage1_release import tick_lock

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/stage1-quality-balanced-20260911-lowmem'
OUT=ROOT/'artifacts/stage1-screen-cue-data-20260914'
CUES=('weak_screen','no_geometry','no_flicker')
PROFILES=quality.PROFILES


def parameters(base,profile,cue):
    values=quality.matched_effects(base,profile,'RERECORDED')
    if cue=='weak_screen':
        for effect in ('moire','band','flicker','geometry','reflection','color','jitter'):
            for key,neutral in GROUPS[effect].items():values[key]=neutral+.25*(values[key]-neutral)
    elif cue in ('no_geometry','no_flicker'):
        values.update(GROUPS[cue.removeprefix('no_')])
    else:raise ValueError('Unknown cue')
    return values


def seed_for(sid):
    return int.from_bytes(hashlib.sha256(f'{quality.SEED}:{sid}:quality-pair'.encode()).digest()[:4],'big')


def code_hash():
    return hashlib.sha256(b''.join(Path(p).read_bytes() for p in (__file__,quality.__file__,quality.synth.__file__,ROOT/'src/prepare_stage1_robustness.py',ROOT/'src/prepare_stage1_full_run.py'))).hexdigest()


def make_plan():
    old_plan=json.loads((OLD/'plan.json').read_text(encoding='utf-8'))
    state=json.loads((OLD/'status.json').read_text(encoding='utf-8'))
    if state['status']!='DATA_VALIDATED' or digest(OLD/'generation_manifest.csv')!=state['manifest_sha256'] or digest(OLD/'plan.json')!=state['plan_sha256']:raise ValueError('Existing quality data not verified')
    if quality.make_plan()!=old_plan:raise ValueError('Canonical training sources changed')
    return dict(scope='TRAIN_ONLY_SCREEN_CUE_AUGMENTATION',sources=old_plan['sources'],seed=quality.SEED,
                profiles=list(PROFILES),cues=list(CUES),crf=24,expected_videos=765,original_references=255,
                existing_plan_sha256=state['plan_sha256'],existing_manifest_sha256=state['manifest_sha256'],
                generator_sha256=code_hash(),training_started=False,uploaded=False)


def verify_rows(out,row,rows):
    if [(r['profile'],r['cue']) for r in rows]!=[(p,c) for p in PROFILES for c in CUES]:raise ValueError('Missing/duplicate cue profile')
    base=quality.synth._effect_parameters(random.Random(seed_for(row['source_id'])))
    for r in rows:
        if any(r[k]!=row[k] for k in ('source_id','upload_group','source_sha256')) or r['split']!='train' or r['label']!='RERECORDED':raise ValueError('Invalid provenance/label')
        expected=f"videos/{row['source_id']}_{r['profile']}_{r['cue']}.mp4"
        if r['output_path']!=expected or r['seed']!=seed_for(row['source_id']) or r['crf']!=24:raise ValueError('Invalid path/seed/codec')
        if json.loads(r['effects'])!=parameters(base,r['profile'],r['cue']):raise ValueError('Effect metadata changed')
        inspect_video(out/r['output_path'],r)


def render(out,row,plan,source_dir):
    cv2.setNumThreads(1)
    source=source_dir/(row['source_id']+'.mp4')
    if digest(source)!=row['source_sha256'] or plan['generator_sha256']!=code_hash():raise ValueError('Source/code changed')
    record=out/'records'/(row['source_id']+'.json')
    identity=dict(row,plan_sha256=digest(out/'plan.json'))
    if record.exists():
        saved=json.loads(record.read_text(encoding='utf-8'))
        if saved['identity']!=identity:raise ValueError('Resume identity mismatch')
        verify_rows(out,row,saved['rows']);return saved['rows']
    seed=seed_for(row['source_id']);base=quality.synth._effect_parameters(random.Random(seed));rows=[]
    for profile in PROFILES:
        for cue in CUES:
            values=parameters(base,profile,cue);effects=quality.synth.Effects(**values)
            cap,w,h,fps,expected=quality.synth._open_video(source)
            fields=(quality.synth._perspective_matrix(w,h,effects),*quality.synth._spatial_fields(w,h,effects))
            relative=f"videos/{row['source_id']}_{profile}_{cue}.mp4"
            writer=quality.LimitedH264Writer(out/relative,w,h,fps,24)
            rng=np.random.default_rng(seed);jitter=random.Random(seed);previous=None;count=0
            try:
                while True:
                    ok,frame=cap.read()
                    if not ok:break
                    frame=quality.low_memory_transform(frame,count,fps,values,fields,rng)
                    if previous is not None and jitter.random()<values['temporal_jitter_probability']:frame=previous.copy()
                    writer.write(frame);previous=frame;count+=1
            finally:cap.release();writer.close()
            if count!=expected or count<1:raise ValueError('Incomplete render')
            rows.append(dict(row,split='train',label='RERECORDED',label_basis='synthetic_proxy',profile=profile,cue=cue,
                             output_path=relative,seed=seed,crf=24,frames=count,fps=fps,width=w,height=h,
                             sha256=digest(out/relative),effects=json.dumps(values,sort_keys=True)))
    verify_rows(out,row,rows);write_json(record,dict(identity=identity,rows=rows));return rows


def generate(out):
    state=dict(status='PREPARING',pid=os.getpid(),completed_videos=0,expected_videos=765,completed_sources=0,expected_sources=85,
               started_unix=time.time(),training_started=False,uploaded=False)
    write_json(out/'status.json',state);deadline=time.monotonic()+21600
    try:
        plan=make_plan()
        if (out/'plan.json').exists():
            if json.loads((out/'plan.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Plan changed')
        else:write_json(out/'plan.json',plan)
        rows=[];originals=[]
        for row in plan['sources']:
            if time.monotonic()>=deadline:raise TimeoutError('Six-hour limit')
            command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--output-dir',str(out),'--worker-source',row['source_id']]
            with (out/'logs'/(row['source_id']+'.log')).open('a',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                state.update(status='GENERATING',worker_pid=child.pid,source_id=row['source_id']);write_json(out/'status.json',state)
                try:code=child.wait(timeout=max(1,min(900,deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    from supervise_stage1_robustness import stop_child
                    stop_child(child);raise TimeoutError('Source 15-minute or total six-hour limit')
                if code:raise RuntimeError(f"Source {row['source_id']} failed: {code}; inspect logs before manual resume")
            saved=json.loads((out/'records'/(row['source_id']+'.json')).read_text(encoding='utf-8'))
            if saved['identity']!=dict(row,plan_sha256=digest(out/'plan.json')):raise ValueError('Worker identity mismatch')
            verify_rows(out,row,saved['rows']);rows.extend(saved['rows'])
            old=json.loads((OLD/'records'/(row['source_id']+'.json')).read_text(encoding='utf-8'))
            if old['identity']!=dict(row,plan_sha256=plan['existing_plan_sha256']):raise ValueError('Original reference identity')
            quality.verify_rows(OLD,row,old['rows'])
            originals.extend(dict(r,data_path=(OLD/r['output_path']).relative_to(ROOT).as_posix()) for r in old['rows'] if r['label']=='ORIGINAL')
            state.update(completed_sources=state['completed_sources']+1,completed_videos=len(rows),heartbeat_unix=time.time())
            write_json(out/'status.json',state)
        table=pd.DataFrame(rows)
        if len(table)!=765 or table.output_path.duplicated().any() or len(originals)!=255:raise ValueError('Final coverage')
        table.to_csv(out/'generation_manifest.csv',index=False)
        combined=[dict(r,data_path=(out/r['output_path']).relative_to(ROOT).as_posix()) for r in rows]+originals
        pd.DataFrame(combined).to_csv(out/'training_manifest.csv',index=False)
        state.update(status='DATA_VALIDATED',finished_unix=time.time(),decoded_frames=int(table.frames.sum()),
                     referenced_originals=255,manifest_sha256=digest(out/'generation_manifest.csv'),training_manifest_sha256=digest(out/'training_manifest.csv'),
                     plan_sha256=digest(out/'plan.json'),validation='all frames/hash/source split/paired quality/coverage PASS')
        write_json(out/'status.json',state)
    except Exception as error:
        state.update(status='FAILED',error=repr(error),finished_unix=time.time());write_json(out/'status.json',state);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--worker-source');a=p.parse_args()
    out=a.output_dir.resolve()
    for name in ('videos','records','logs'):(out/name).mkdir(parents=True,exist_ok=True)
    if a.worker_source:
        plan=json.loads((out/'plan.json').read_text(encoding='utf-8'));row=next(r for r in plan['sources'] if r['source_id']==a.worker_source)
        render(out,row,plan,ROOT/'data_raw/ccd/videos/Crash-1500');return
    with tick_lock(out/'generation.lock') as acquired:
        if not acquired:raise RuntimeError('Generation already running')
        import ctypes
        awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:generate(out)
        finally:
            if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
