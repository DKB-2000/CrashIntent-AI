"""Independent slice scoring for fixed Stage3 motion experiments; sensors are evaluation-only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('artifacts/stage3-motion-independent-20260910')
BASE = Path('artifacts/stage3-representation-20260910')
SIGNALS = Path('artifacts/stage3-comma-chunk1-calibrated-v1/signals_and_candidates.csv')
ZIP = Path('artifacts/stage3-motion-submit-candidate-20260910/submit.zip')
ACCEL = ['ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED']
STEER = ['LEFT', 'STRAIGHT', 'RIGHT']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def slices():
    frame = pd.read_csv(BASE / 'validation-samples.csv')
    signals = pd.read_csv(SIGNALS, usecols=['ID','frame_index','speed_mps','split']).rename(columns={'frame_index':'endpoint'})
    merged = frame.merge(signals, on=['ID','endpoint'], how='left', validate='one_to_one', sort=False)
    assert len(merged) == 18603 and merged.speed_mps.notna().all() and merged['split'].eq('validation').all()
    assert np.isfinite(merged.speed_mps).all()
    moving = merged.accel_label.ne('STOPPED').to_numpy()
    masks = {'all': np.ones(len(frame),bool), 'stopped': ~moving,
        'moving_under_2mps': moving & merged.speed_mps.lt(2).to_numpy(),
        'moving_2_to_5mps': moving & merged.speed_mps.ge(2).to_numpy() & merged.speed_mps.lt(5).to_numpy(),
        'moving_at_least_5mps': moving & merged.speed_mps.ge(5).to_numpy()}
    for task, column in [('accel','accel_label'),('steer','steer_label')]:
        near = np.zeros(len(frame),bool)
        for _, g in frame.groupby('ID', sort=False):
            assert np.array_equal(g.endpoint, np.arange(len(g)))
            labels = g[column].to_numpy()
            changed = labels[1:] != labels[:-1]
            if task == 'steer':
                gm = g.accel_label.ne('STOPPED').to_numpy()
                changed &= gm[1:] & gm[:-1]
            boundaries = np.flatnonzero(changed) + 1
            for b in boundaries:
                near[g.index[max(0,b-5):min(len(g),b+6)]] = True
        masks[task+'_transition_within_5frames'] = near
        masks[task+'_stable_beyond_5frames'] = ~near
    for route in sorted(frame.route.unique()):
        masks['route:'+route] = frame.route.eq(route).to_numpy()
    return frame, masks

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    frame,masks=slices()
    plan=dict(status='FIXED_BEFORE_RESULTS', script_sha256=sha(__file__),
        validation_samples_sha256=sha(BASE/'validation-samples.csv'), signals_sha256=sha(SIGNALS),
        preserved_submission_sha256=sha(ZIP),
        slices={name:int(mask.sum()) for name,mask in masks.items()},
        note='Sensors and future labels used only for retrospective evaluation masks, never inference. Steer excludes true STOPPED. Macro-F1 uses fixed 4/3 labels even if slice support absent. No significance claim from correlated frames.')
    save(ROOT/'plan.json',plan)
    print(json.dumps(plan,indent=2))

def metrics(truth,pred,classes):
    matrix=np.zeros((len(classes),len(classes)),np.int64)
    mapping={label:i for i,label in enumerate(classes)}
    np.add.at(matrix,([mapping[v] for v in truth],[mapping[v] for v in pred]),1)
    support=matrix.sum(1); predicted=matrix.sum(0); diag=matrix.diagonal()
    f1=np.divide(2*diag,support+predicted,out=np.zeros(len(classes),float),where=(support+predicted)>0)
    recall=np.divide(diag,support,out=np.zeros(len(classes),float),where=support>0)
    precision=np.divide(diag,predicted,out=np.zeros(len(classes),float),where=predicted>0)
    return matrix, support, f1, recall, precision

def score(experiments):
    plan=json.loads((ROOT/'plan.json').read_text())
    assert plan['script_sha256']==sha(__file__)
    assert plan['validation_samples_sha256']==sha(BASE/'validation-samples.csv')
    assert plan['signals_sha256']==sha(SIGNALS) and plan['preserved_submission_sha256']==sha(ZIP)
    frame,masks=slices(); rows=[]; classes_rows=[]; evidence=[]
    paths=[BASE]+experiments
    for folder in paths:
        files=sorted(folder.glob('*/validation-predictions.csv'))
        assert files, folder
        if folder==BASE:
            files=[f for f in files if f.parent.name.endswith('-motion')]
            assert len(files)==3
        for file in files:
            seed,mode=file.parent.name.split('-',1)
            pred=pd.read_csv(file)
            assert list(pred.columns)==['ID','sample_index','accel_label','steer_label'], file
            assert len(pred)==len(frame) and not pred[['ID','sample_index']].duplicated().any()
            pd.testing.assert_frame_equal(pred[['ID','sample_index']].rename(columns={'sample_index':'endpoint'}),frame[['ID','endpoint']])
            assert pred.accel_label.isin(ACCEL).all() and pred.steer_label.isin(STEER).all()
            evidence.append(dict(path=str(file),sha256=sha(file),seed=int(seed),mode=mode,experiment=folder.name))
            for name,mask in masks.items():
                for task,labels in [('accel',ACCEL),('steer',STEER)]:
                    active=mask & (frame.accel_label.ne('STOPPED').to_numpy() if task=='steer' else True)
                    n=int(active.sum())
                    if not n: continue
                    matrix,support,f1,recall,precision=metrics(frame.loc[active,task+'_label'],pred.loc[active,task+'_label'],labels)
                    common=dict(experiment=folder.name,seed=int(seed),mode=mode,slice=name,task=task,samples=n)
                    rows.append(dict(**common,macro_f1=float(f1.mean()),accuracy=float(matrix.diagonal().sum()/n)))
                    for i,label in enumerate(labels):
                        classes_rows.append(dict(**common,label=label,support=int(support[i]),predicted=int(matrix[:,i].sum()),f1=float(f1[i]),recall=float(recall[i]),precision=float(precision[i])))
    scores=pd.DataFrame(rows); perclass=pd.DataFrame(classes_rows)
    baseline=scores[scores.experiment.eq(BASE.name)][['seed','slice','task','macro_f1']].rename(columns={'macro_f1':'baseline_f1'})
    scores=scores.merge(baseline,on=['seed','slice','task'],validate='many_to_one')
    scores['delta_f1']=scores.macro_f1-scores.baseline_f1
    scores.to_csv(ROOT/'slice-scores.csv',index=False);perclass.to_csv(ROOT/'slice-classes.csv',index=False)
    mean=scores.groupby(['experiment','mode','slice','task']).agg(seeds=('seed','nunique'),macro_f1=('macro_f1','mean'),min_f1=('macro_f1','min'),max_f1=('macro_f1','max'),delta_f1=('delta_f1','mean')).reset_index()
    mean.to_csv(ROOT/'slice-means.csv',index=False)
    assert mean.seeds.eq(3).all()
    save(ROOT/'validation.json',dict(status='PASS',plan_sha256=sha(ROOT/'plan.json'),inputs=evidence,outputs={f:sha(ROOT/f) for f in ['slice-scores.csv','slice-classes.csv','slice-means.csv']},submission_unchanged=True))
    print(mean[mean['slice'].isin(['all','stopped','moving_under_2mps'])].to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','score']);parser.add_argument('--experiment',type=Path,action='append',default=[]);args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:score(args.experiment)
