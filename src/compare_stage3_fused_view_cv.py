"""Preserve center402 features and add full-view402 in previously inactive slots."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import csv
import compare_stage3_fullframe_cv as full
import numpy as np
import pandas as pd
import torch

PROJECT = full.PROJECT
ROOT = PROJECT / 'artifacts/stage3-fused-view-cv-20260914'
SOURCE = full.ROOT
REF = full.REF
sha = full.sha


def save(name, obj):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def fuse(center, context):
    assert center.shape == context.shape and center.ndim == 2 and center.shape[1] == 1682
    assert np.isfinite(center).all() and np.isfinite(context).all()
    result = center.copy()
    result[:, :878] = 0
    result[:, 878:1280] = context[:, 1280:]
    return result


def prepare():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'plan.json').exists()
    validation = json.loads((SOURCE/'independent-validation.json').read_text())
    assert validation['status'] == 'PASS'
    for name in ['summary','results']:
        assert sha(SOURCE/f'{name}.json') == validation[f'{name}_sha256']
    inputs = dict(json.loads((SOURCE/'plan.json').read_text())['inputs'])
    for path in [Path(__file__), SOURCE/'independent-validation.json', SOURCE/'summary.json', SOURCE/'results.json']:
        inputs[str(path.relative_to(PROJECT))] = sha(path)
    for path in (SOURCE/'cache').iterdir():
        if path.suffix in ('.npz','.json'):
            inputs[str(path.relative_to(PROJECT))] = sha(path)
    save('plan.json', dict(inputs=inputs, arms=['baseline','fused'], seeds=full.SEEDS, folds=5,
                          updates=1200, batch=10, lr=.001, weight_decay=0, threads=2,
                          features='Existing center402 kept at1280:1682; full402 placed at878:1280; first878 zero. Same1682->128->64 model and seed;804 active dimensions vs402.',
                          protocol='Same fold samples/batches/final1200 update and train-only normalization as verified fullframe experiment; no human labels or old4development routes',
                          baseline='Reuse15 models with complete checkpoint/logit replay; baseline exact retrain gate already passed in immediately preceding fullframe experiment',
                          selection='Require all3 OOF steer deltas positive, >=4/5 fold-mean improvements, no direction recall mean loss beyond2pp; no tuning',
                          limitations='Repeated balanced1000clip/17route sensor proxy development CV; same allocated parameter count but more active inputs/weights; no official or new-domain claim'))
    check()
    save('status.json', dict(status='PREPARED', completed=0, total=15))


def check():
    plan = json.loads((ROOT/'plan.json').read_text())
    for path, expected in plan['inputs'].items():
        assert sha(PROJECT/path) == expected, path
    return plan


def data():
    f = pd.read_csv(REF/'samples.csv')
    x = full.arrays(f)
    return f, {'baseline': x['baseline'], 'fused': fuse(x['baseline'], x['full_frame'])}


def replay(ck, x, vi, arm):
    model = full.base.Control()
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    v = np.clip((x[vi]-ck['mean'].numpy())/ck['std'].numpy(), -10, 10)
    v[:, :1280 if arm == 'baseline' else 878] = 0
    with torch.inference_mode():
        a, s = model(torch.from_numpy(v))
    assert torch.isfinite(a).all() and torch.isfinite(s).all()
    return a.numpy(), s.numpy()


def run():
    plan = check()
    f, x = data()
    runs = []
    started = time.monotonic()
    for fold in range(5):
        for seed in full.SEEDS:
            out = ROOT/f'{fold}-{seed}-fused'
            out.mkdir(exist_ok=False)
            ck = full.fit(x['fused'], f, fold, seed)
            vi = np.array(ck['validation_indices'])
            a, s = replay(ck, x['fused'], vi, 'fused')
            torch.save(ck, out/'model.pt')
            np.savez_compressed(out/'logits.npz', accel=a, steer=s, indices=vi)
            report = dict(fold=fold, seed=seed, arm='fused', checkpoint_sha256=sha(out/'model.pt'),
                          metrics=full.reference.score(f.iloc[vi], a.argmax(1), s.argmax(1)))
            (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            runs.append(report)
            save('status.json', dict(status='TRAINING', pid=os.getpid(), completed=len(runs), total=15))
            print('trained', len(runs), flush=True)
    save('results.json', dict(runs=runs, seconds=time.monotonic()-started))
    save('status.json', dict(status='VERIFYING', models=30))
    check()
    oof = {(seed, arm):np.full((len(f),2), -1, int) for seed in full.SEEDS for arm in x}
    fold_scores = []
    for fold in range(5):
        ti, vi = np.flatnonzero(f.fold.to_numpy()!=fold), np.flatnonzero(f.fold.to_numpy()==fold)
        assert set(f.iloc[ti].route).isdisjoint(f.iloc[vi].route)
        for seed in full.SEEDS:
            for arm in x:
                out = (REF if arm == 'baseline' else ROOT)/f'{fold}-{seed}-{arm}'
                ck = torch.load(out/'model.pt', weights_only=True)
                report = json.loads((out/'report.json').read_text())
                assert sha(out/'model.pt') == report['checkpoint_sha256']
                np.testing.assert_array_equal(ck['train_indices'], ti)
                np.testing.assert_array_equal(ck['validation_indices'], vi)
                start = 1280 if arm == 'baseline' else 0
                np.testing.assert_array_equal(ck['mean'].numpy()[start:], x[arm][ti].mean(0)[start:])
                np.testing.assert_array_equal(ck['std'].numpy()[start:], x[arm][ti].std(0).clip(.01)[start:])
                a, s = replay(ck, x[arm], vi, arm)
                with np.load(out/'logits.npz') as z:
                    np.testing.assert_array_equal(a, z['accel'])
                    np.testing.assert_array_equal(s, z['steer'])
                    np.testing.assert_array_equal(vi, z['indices'])
                metrics = full.reference.score(f.iloc[vi], a.argmax(1), s.argmax(1))
                assert metrics == report['metrics']
                fold_scores.append(dict(fold=fold, seed=seed, arm=arm, **metrics))
                oof[seed,arm][vi] = np.stack([a.argmax(1),s.argmax(1)],1)
    pooled, routes = [], []
    for (seed,arm), pred in oof.items():
        assert np.all(pred>=0)
        metrics = full.reference.score(f, pred[:,0], pred[:,1])
        pooled.append(dict(seed=seed,arm=arm,**metrics))
        output = f.copy()
        output['accel_pred'],output['steer_pred'] = pred[:,0],pred[:,1]
        path = ROOT/f'oof-{seed}-{arm}.csv'
        output.to_csv(path,index=False)
        cm = [[0]*3 for _ in range(3)]
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if int(row['accel'])!=3:
                    cm[int(row['steer'])][int(row['steer_pred'])]+=1
        assert cm==metrics['steer']['confusion']
        values=[]
        for i in range(3):
            d=sum(cm[i])+sum(row[i] for row in cm)
            values.append(2*cm[i][i]/d if d else 0)
        assert abs(sum(values)/3-metrics['steer']['macro_f1'])<1e-12
        for route,g in f.groupby('route'):
            m=full.reference.score(g,pred[g.index,0],pred[g.index,1])
            routes.append(dict(seed=seed,arm=arm,route=route,steer_f1=m['steer']['macro_f1'],recall=m['steer']['recall']))
    averages={arm:dict(steer_f1=float(np.mean([r['steer']['macro_f1'] for r in pooled if r['arm']==arm])),
                       accel_f1=float(np.mean([r['accel']['macro_f1'] for r in pooled if r['arm']==arm])),
                       recall=np.mean([r['steer']['recall'] for r in pooled if r['arm']==arm],0).tolist()) for arm in x}
    seed_deltas=[next(r['steer']['macro_f1'] for r in pooled if r['arm']=='fused' and r['seed']==s)-
                 next(r['steer']['macro_f1'] for r in pooled if r['arm']=='baseline' and r['seed']==s) for s in full.SEEDS]
    fold_deltas=[float(np.mean([r['steer']['macro_f1'] for r in fold_scores if r['arm']=='fused' and r['fold']==k])-
                       np.mean([r['steer']['macro_f1'] for r in fold_scores if r['arm']=='baseline' and r['fold']==k])) for k in range(5)]
    recall_delta=np.array(averages['fused']['recall'])-averages['baseline']['recall']
    decision='FOLLOWUP_CANDIDATE_ONLY' if all(d>0 for d in seed_deltas) and sum(d>0 for d in fold_deltas)>=4 and np.all(recall_delta>=-.02) else 'KEEP_BASELINE'
    save('summary.json',dict(averages=averages,seed_deltas=seed_deltas,fold_deltas=fold_deltas,recall_delta=recall_delta.tolist(),decision=decision,pooled_oof=pooled,fold_metrics=fold_scores,routes=routes,scope=plan['limitations']))
    save('independent-validation.json',dict(status='PASS',models=30,new_models=15,max_logits_error=0,
                                           checks='Frozen cache/inputs; center preserved; route separation/train normalization; saved batches; all logits and independent CSV steer F1',
                                           summary_sha256=sha(ROOT/'summary.json'),results_sha256=sha(ROOT/'results.json')))
    save('status.json',dict(status='COMPLETE_VALIDATED',models=30,decision=decision))
    print(json.dumps(averages,indent=2),flush=True)


def watch():
    import msvcrt
    with (ROOT/'watch.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not (ROOT/'results.json').exists(), 'Already trained; do not overwrite'
        try:
            with (ROOT/'run.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
                save('watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=proc.pid,timeout_seconds=1800,retries=0))
                try:
                    rc=proc.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],check=True,capture_output=True,timeout=30)
                    proc.wait(timeout=30)
                    raise
                if rc:raise RuntimeError(f'Worker exit {rc}; see run.log')
            save('watch-status.json',dict(status='COMPLETE_VALIDATED'))
        except Exception as exc:
            save('watch-status.json',dict(status='FAILED',error=repr(exc)))
            save('status.json',dict(status='FAILED',error=repr(exc)))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','watch']);args=p.parse_args()
    os.chdir(PROJECT);torch.set_num_threads(2)
    globals()[args.command]()
