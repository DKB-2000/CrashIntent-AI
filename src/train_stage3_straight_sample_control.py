"""Paired, fixed-budget straight-sample experiment; never deploys models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import compare_stage3_representations as base

P=Path(__file__).resolve().parents[1]
REF=P/'artifacts/stage3-representation-20260910'
ROOT=P/'artifacts/stage3-straight-sample-control-20260914'

def save(name,obj):base.save(ROOT/name,obj)

def run():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    plan=json.loads((ROOT/'report.json').read_text())
    for name,h in plan['input_sha256'].items():assert base.sha(P/name)==h
    for name,h in plan['output_sha256'].items():assert base.sha(ROOT/name)==h
    old=pd.read_csv(ROOT/'baseline-samples.csv');new=pd.read_csv(ROOT/'candidate-samples.csv')
    original=pd.read_csv(REF/'train-samples.csv');valid=pd.read_csv(REF/'validation-samples.csv')
    cols=['ID','endpoint','route','accel','steer','group','video']
    pd.testing.assert_frame_equal(old[cols],original[cols])
    pd.testing.assert_frame_equal(old[['route','accel','steer','group','speed_bin']],new[['route','accel','steer','group','speed_bin']])
    assert set(new.route).isdisjoint(valid.route)
    changed=(old.ID!=new.ID)|(old.endpoint!=new.endpoint)
    assert changed.sum()==84 and np.array_equal(changed,new.replaced)
    assert new.loc[changed,'steer'].eq(1).all() and new.loc[changed,'clear_straight'].all()
    base.ROOT=REF
    xold=base.array_for(original);xvalid=base.array_for(valid);xnew=xold.copy()
    cache=ROOT/'new-features';cache.mkdir(exist_ok=False)
    groups=list(new[changed].groupby('ID'))
    provenance=[]
    for i,(sid,g) in enumerate(groups,1):
        path=(base.DATA/g.video.iloc[0]).resolve()
        # Include one old endpoint to verify feature extraction against the frozen cache.
        with np.load(REF/'cache'/(sid+'.npz')) as z:
            anchor=int(z['endpoints'][0]);anchor_features=z['features'][0].copy()
        metadata=json.loads((REF/'cache'/(sid+'.json')).read_text())
        assert base.sha(path)==metadata['video_sha256']
        endpoints=sorted(set(g.endpoint.astype(int))|{anchor})
        features,n=base.extract_video(path,endpoints)
        np.testing.assert_array_equal(features[endpoints.index(anchor)],anchor_features)
        for idx,row in g.iterrows():xnew[idx]=features[endpoints.index(int(row.endpoint))]
        target=cache/(sid+'.npz');np.savez_compressed(target,features=features,endpoints=endpoints)
        provenance.append(dict(ID=sid,video_sha256=metadata['video_sha256'],cache_sha256=base.sha(target),frames=n,anchor_exact=True))
        save('status.json',dict(status='EXTRACTING',completed=i,total=len(groups),pid=os.getpid()))
        print('features',i,len(groups),sid,flush=True)
    save('feature-provenance.json',dict(videos=provenance,baseline_features_sha256=__import__('hashlib').sha256(xold.tobytes()).hexdigest()))
    results={}
    base.MODES=('motion',)
    for arm,frame,values in [('baseline',old,xold),('candidate',new,xnew)]:
        out=ROOT/arm;out.mkdir(exist_ok=False)
        frame.to_csv(out/'train-samples.csv',index=False);valid.to_csv(out/'validation-samples.csv',index=False)
        base.save(out/'plan.json',dict(script_sha256=base.sha(base.__file__)))
        base.ROOT=out
        base.array_for=lambda f: values if len(f)==1000 else xvalid
        save('status.json',dict(status='TRAINING',arm=arm,pid=os.getpid()))
        base.train()
        results[arm]=json.loads((out/'results.json').read_text())['runs']
        if arm=='baseline':
            for seed in base.SEEDS:
                with np.load(out/f'{seed}-motion/validation-logits.npz') as a,np.load(REF/f'{seed}-motion/validation-logits.npz') as b:
                    for key in ['accel','steer']:np.testing.assert_array_equal(a[key],b[key])
            print('All three baseline logits reproduced exactly',flush=True)
    rows=[]
    for seed in base.SEEDS:
        for arm in ['baseline','candidate']:
            with np.load(ROOT/arm/f'{seed}-motion/validation-logits.npz') as z:
                a=z['accel'].argmax(1);s=z['steer'].argmax(1)
            for route in ['ALL']+sorted(valid.route.unique()):
                mask=np.ones(len(valid),dtype=bool) if route=='ALL' else (valid.route==route).to_numpy()
                metric=base.score(valid[mask],a[mask],s[mask])['steer'];cm=np.array(metric['confusion_matrix'])
                rows.append(dict(seed=seed,arm=arm,route=route,macro_f1=metric['macro_f1'],recall=(np.diag(cm)/cm.sum(1)).tolist()))
    save('comparison.json',dict(status='COMPLETE_PENDING_TRANSITION_REVIEW',rows=rows,baseline_exact=True,adopted=False,
         input_plan_sha256=base.sha(ROOT/'report.json'),script_sha256=base.sha(__file__),
         next='Evaluate frozen gate including per-class recall and transition misses/delays before any adoption decision. No Civic tuning.'))
    save('status.json',dict(status='COMPLETE_PENDING_TRANSITION_REVIEW',completed=6,total=6))

def watch():
    import msvcrt
    with (ROOT/'train-watch.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not (ROOT/'comparison.json').exists()
        try:
            with (ROOT/'train.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=P,stdout=log,stderr=subprocess.STDOUT)
                save('train-watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=child.pid))
                try:rc=child.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=True,timeout=30);raise
                if rc:raise RuntimeError(f'Worker exit {rc}; see train.log')
            assert json.loads((ROOT/'comparison.json').read_text())['baseline_exact']
            save('train-watch-status.json',dict(status='COMPLETE'))
        except Exception as e:
            save('status.json',dict(status='FAILED',error=str(e)));save('train-watch-status.json',dict(status='FAILED',error=str(e)));raise

if __name__=='__main__':{'run':run,'watch':watch}[sys.argv[1]]()
