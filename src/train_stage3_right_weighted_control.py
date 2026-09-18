"""Compare a fixed right-turn loss weight against the saved warmstart CE control."""
import os
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import compare_stage3_representations as base
from compare_stage3_causal_smoothing import metric

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/stage3-right-weighted-control-20260916'
CONTROL=P/'artifacts/stage3-prediction-preservation-20260914'
FEATURES=P/'artifacts/stage3-right-recovery-20260914'
PREV=P/'artifacts/stage3-transition-matched-control-20260914'
REF=P/'artifacts/stage3-representation-20260910'

def save(name,obj):base.save(ROOT/name,obj)

def predict(model,x):
    model.eval();aa=[];ss=[]
    with torch.inference_mode():
        for chunk in x.split(256):
            a,s=model(chunk);aa.append(a);ss.append(s)
    a=torch.cat(aa).numpy();s=torch.cat(ss).numpy()
    assert np.isfinite(a).all() and np.isfinite(s).all()
    return a,s

def run():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    plan=json.loads((ROOT/'plan.json').read_text())
    assert plan['arms']==['right_weighted'] and plan['right_weight']==1.25
    for name,h in plan['inputs_sha256'].items():assert base.sha(P/name)==h
    for name,h in plan['outputs_sha256'].items():assert base.sha(ROOT/name)==h
    provenance={str(p.relative_to(P)):base.sha(p) for p in [Path(__file__),Path(base.__file__),ROOT/'plan.json',REF/'validation-samples.csv']}
    save('execution.json',dict(status='STARTED',inputs_sha256=provenance,pid=os.getpid(),threads=2))
    old=pd.read_csv(ROOT/'repeat-samples.csv');new=pd.read_csv(ROOT/'transition-samples.csv');valid=pd.read_csv(REF/'validation-samples.csv')
    schedule=pd.read_csv(ROOT/'schedule.csv').sort_values(['epoch','batch','position'])
    pd.testing.assert_frame_equal(new.iloc[:1000].reset_index(drop=True),old)
    assert len(new)==1090+plan['added_right'] and set(new.route).isdisjoint(valid.route)
    base.ROOT=REF
    for sid in set(old.ID)|set(valid.ID):
        path=REF/'cache'/(sid+'.npz');m=json.loads(path.with_suffix('.json').read_text())
        assert base.sha(path)==m['features_sha256']
    xold=base.array_for(old);xvalid=base.array_for(valid)
    feature_record=json.loads((FEATURES/'feature-provenance.json').read_text())
    assert base.sha(FEATURES/'training-features.npz')==feature_record['training_features_sha256']
    with np.load(FEATURES/'training-features.npz') as z:
        np.testing.assert_array_equal(z['baseline'],xold);xnew=z['transition'].copy()
    assert xnew.shape==(len(new),xold.shape[1]) and np.isfinite(xnew).all()
    np.testing.assert_array_equal(xnew[:1000],xold)
    print('Frozen features and original1000 exact; no video extraction',flush=True)
    ck=torch.load(REF/'20260910-motion/model.pt',weights_only=True,map_location='cpu')
    mean,std,mask=(ck[k] for k in ['mean','std','mask'])
    np.testing.assert_array_equal(mean.numpy(),xold.mean(0));np.testing.assert_array_equal(std.numpy(),xold.std(0).clip(.01))
    assert torch.all(mask[:1280]==0) and torch.all(mask[1280:]==1)
    def norm(x):return torch.from_numpy(np.clip((x-mean.numpy())/std.numpy(),-10,10))*mask
    xv=norm(xvalid);xs={'right_weighted':norm(xnew)}
    frames={'right_weighted':new};completed=0
    for seed in plan['seeds']:
        for arm in plan['arms']:
            out=ROOT/arm/str(seed);out.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed);model=base.Control()
            initial=torch.load(REF/f'{seed}-motion/model.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(initial['model'],strict=True)
            initial_a,initial_s=predict(model,xv)
            with np.load(REF/f'{seed}-motion/validation-logits.npz') as reference:
                np.testing.assert_array_equal(initial_a,reference['accel']);np.testing.assert_array_equal(initial_s,reference['steer'])
            opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=0.)
            frame=frames[arm];ta=torch.tensor(frame.accel.to_numpy());ts=torch.tensor(frame.steer.to_numpy());history=[]
            col='transition_index'
            for epoch,g in schedule.groupby('epoch',sort=True):
                losses=[];penalties=[];model.train()
                for _,batch in g.groupby('batch',sort=True):
                    ids=batch[col].to_numpy();a,s=model(xs[arm][ids]);moving=ta[ids]!=3
                    weights=torch.tensor([1.,1.,plan['right_weight']],dtype=s.dtype)
                    loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving],weight=weights)
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
                assert len(losses)==130
                history.append(dict(epoch=int(epoch),loss=float(np.mean(losses)),updates=len(losses),right_weight=plan['right_weight']))
                save('status.json',dict(status='TRAINING',completed=completed,total=3,seed=seed,arm=arm,epoch=int(epoch),pid=os.getpid()))
            assert sum(r['updates'] for r in history)==plan['updates_per_arm']
            a,s=predict(model,xv);np.savez_compressed(out/'validation-logits.npz',accel=a,steer=s)
            pred=valid[['ID','endpoint']].copy();pred['accel']=a.argmax(1);pred['steer']=s.argmax(1);pred.to_csv(out/'predictions.csv',index=False)
            torch.save(dict(model=model.state_dict(),mean=mean,std=std,mask=mask,seed=seed,arm=arm),out/'model.pt')
            base.save(out/'report.json',dict(status='COMPLETE',history=history,checkpoint_sha256=base.sha(out/'model.pt'),metrics=base.score(valid,a.argmax(1),s.argmax(1))))
            completed+=1;print('trained',completed,seed,arm,metric(valid,s.argmax(1))['f1'],flush=True)
    save('status.json',dict(status='VERIFYING',completed=3,total=3,pid=os.getpid()))
    rows=[];events=[];outputs={}
    for seed in plan['seeds']:
        for arm in ['original','warmstart_ce']+plan['arms']:
            folder=REF/f'{seed}-motion' if arm=='original' else CONTROL/'warmstart_ce'/str(seed) if arm=='warmstart_ce' else ROOT/arm/str(seed)
            report=json.loads((folder/'report.json').read_text());assert base.sha(folder/'model.pt')==report['checkpoint_sha256']
            ck=torch.load(folder/'model.pt',weights_only=True,map_location='cpu')
            for name,expected in [('mean',mean),('std',std),('mask',mask)]:assert torch.equal(ck[name],expected)
            model=base.Control();model.load_state_dict(ck['model'],strict=True);aa,ss=predict(model,xv)
            with np.load(folder/'validation-logits.npz') as z:
                np.testing.assert_array_equal(aa,z['accel']);np.testing.assert_array_equal(ss,z['steer'])
            pred=ss.argmax(1)
            if arm!='original':
                recorded=pd.read_csv(folder/'predictions.csv')
                assert np.array_equal(recorded.ID,valid.ID) and np.array_equal(recorded.endpoint,valid.endpoint)
                np.testing.assert_array_equal(recorded.steer,pred);np.testing.assert_array_equal(recorded.accel,aa.argmax(1))
            for route in ['ALL']+sorted(valid.route.unique()):
                g=valid if route=='ALL' else valid[valid.route==route]
                rows.append(dict(seed=seed,arm=arm,route=route,**metric(g,pred[g.index])))
            for sid,g in valid.groupby('ID'):
                assert np.all(np.diff(g.endpoint)==1)
                truth=g.steer.to_numpy();moving=g.accel.to_numpy()!=3;p=pred[g.index]
                for t in range(10,len(g)-12):
                    before,after=truth[t-1],truth[t]
                    if before==after or (before!=1 and after!=1):continue
                    if not (moving[t-10:t+10].all() and np.all(truth[t-10:t]==before) and np.all(truth[t:t+10]==after)):continue
                    hits=[k for k in range(11) if np.all(truth[t:t+k+3]==after) and np.all(p[t+k:t+k+3]==after)]
                    events.append(dict(seed=seed,arm=arm,ID=sid,endpoint=int(g.endpoint.iloc[t]),kind='turn_start' if before==1 else 'turn_end',detected=bool(hits),delay_s=hits[0]/10 if hits else None))
            for name in ['model.pt','validation-logits.npz']:outputs[str((folder/name).relative_to(P))]=base.sha(folder/name)
    e=pd.DataFrame(events);e.to_csv(ROOT/'transitions.csv',index=False);transition=[]
    for (seed,arm,kind),g in e.groupby(['seed','arm','kind']):
        hit=g[g.detected];transition.append(dict(seed=int(seed),arm=arm,kind=kind,events=len(g),missed=int((~g.detected).sum()),mean_delay_s=float(hit.delay_s.mean()) if len(hit) else None))
    pooled=[r for r in rows if r['route']=='ALL'];gates={}
    for candidate in plan['arms']:
        delta=[next(r['f1'] for r in pooled if r['seed']==seed and r['arm']==candidate)-next(r['f1'] for r in pooled if r['seed']==seed and r['arm']=='original') for seed in plan['seeds']]
        rd=[float(np.mean([r['f1'] for r in rows if r['route']==route and r['arm']==candidate])-np.mean([r['f1'] for r in rows if r['route']==route and r['arm']=='original'])) for route in sorted(valid.route.unique())]
        recall=np.mean([r['recall'] for r in pooled if r['arm']==candidate],0)-np.mean([r['recall'] for r in pooled if r['arm']=='original'],0)
        latency=True
        for r in [r for r in transition if r['arm']==candidate]:
            b=next(b for b in transition if b['seed']==r['seed'] and b['kind']==r['kind'] and b['arm']=='original')
            latency &= r['missed']<=b['missed'] and r['mean_delay_s'] is not None and b['mean_delay_s'] is not None and r['mean_delay_s']<=b['mean_delay_s']+.1
        gates[candidate]=dict(seed_f1_deltas=delta,route_f1_deltas=rd,recall_delta=recall.tolist(),latency_pass=bool(latency),passed=bool(all(v>0 for v in delta) and sum(v>0 for v in rd)>=3 and min(recall)>=-.02 and latency))
    preservation_effect=dict(seed_f1_deltas=[next(r['f1'] for r in pooled if r['seed']==seed and r['arm']=='right_weighted')-next(r['f1'] for r in pooled if r['seed']==seed and r['arm']=='warmstart_ce') for seed in plan['seeds']],
        recall_delta=(np.mean([r['recall'] for r in pooled if r['arm']=='right_weighted'],0)-np.mean([r['recall'] for r in pooled if r['arm']=='warmstart_ce'],0)).tolist())
    for path,h in provenance.items():assert base.sha(P/path)==h
    for name,h in plan['outputs_sha256'].items():assert base.sha(ROOT/name)==h
    result=dict(status='COMPLETE_VALIDATED',decision='FOLLOWUP_ONLY' if any(g['passed'] for g in gates.values()) else 'KEEP_BASELINE',
        averages={a:float(np.mean([r['f1'] for r in pooled if r['arm']==a])) for a in ['original','warmstart_ce']+plan['arms']},gates=gates,preservation_effect=preservation_effect,metrics=rows,transitions=transition,
        validation='Input/manifest/features hashes; all3 initial checkpoints exact; frozen normalization exact; all9 checkpoint/hash/logit replays exact; new3 CSV mapping exact; independent confusion metrics; transitions and frozen gate evaluated.',
        output_sha256=outputs,transitions_sha256=base.sha(ROOT/'transitions.csv'),adopted=False)
    save('final-review.json',result);save('status.json',dict(status='COMPLETE_VALIDATED',completed=3,total=3,decision=result['decision']))
    print(json.dumps(dict(averages=result['averages'],gates=gates),indent=2),flush=True)

def watch():
    import msvcrt
    with (ROOT/'train-watch.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not (ROOT/'execution.json').exists(),'Existing execution must be reviewed; no automatic retry'
        try:
            with (ROOT/'train.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=P,stdout=log,stderr=subprocess.STDOUT)
                save('train-watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=child.pid))
                try:rc=child.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=True,timeout=30);raise
                if rc:raise RuntimeError(f'Worker exit {rc}; see train.log')
            assert json.loads((ROOT/'final-review.json').read_text())['status']=='COMPLETE_VALIDATED'
            save('train-watch-status.json',dict(status='COMPLETE'))
        except Exception as e:
            save('status.json',dict(status='FAILED',error=str(e)));save('train-watch-status.json',dict(status='FAILED',error=str(e)));raise

if __name__=='__main__':{'run':run,'watch':watch}[sys.argv[1]]()
