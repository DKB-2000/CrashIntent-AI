"""Paired fixed-window dynamics augmentation on the frozen route CV protocol."""
import argparse,hashlib,json,time,shutil
import stage3_motion_dynamics_features as dynamics
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn import functional as F
import stage3_temporal_motion_experiment as source
import compare_stage3_representations as base
import evaluate_stage3_motion_slices as evaluator
ROOT=Path('artifacts/stage3-dynamics-cv-20260911');REFERENCE=Path('artifacts/stage3-route-cv-20260911');SEEDS=[20260910,20260911,20260912];ARMS=['baseline','dynamics']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def score(f,a,s):
 moving=f.accel.to_numpy()!=3;result={}
 for task,truth,pred,labels in [('accel',f.accel.to_numpy(),a,base.p.ACCEL),('steer',f.steer.to_numpy()[moving],s[moving],base.p.STEER)]:
  cm,support,f1,recall,precision=evaluator.metrics(np.array(labels)[truth],np.array(labels)[pred],labels)
  result[task]=dict(macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean()),confusion=cm.tolist(),support=support.tolist(),recall=recall.tolist(),per_class_f1=f1.tolist())
 return result

def prepare():
 ROOT.mkdir(exist_ok=False)
 validation=json.loads((source.ROOT/'independent-validation.json').read_text());assert validation['status']=='PASS' and sha(source.ROOT/'results.json')==validation['results_sha256']
 frame=pd.read_csv(source.ROOT/'train-samples.csv');routes=sorted(frame.route.unique());assert len(routes)==17 and len(frame)==1000
 rng=np.random.default_rng(20260911);rng.shuffle(routes);split={r:k for k,rs in enumerate(np.array_split(routes,5)) for r in rs};frame['fold']=frame.route.map(split);frame.to_csv(ROOT/'samples.csv',index=False)
 assert sha(ROOT/'samples.csv')==sha(REFERENCE/'samples.csv')
 ref=json.loads((REFERENCE/'independent-validation.json').read_text());assert ref['status']=='PASS' and sha(REFERENCE/'results.json')==ref['results_sha256']
 for fold in range(5):assert set(frame[frame.fold!=fold].group)==set(range(10))
 save('plan.json',dict(status='PREDECLARED',seeds=SEEDS,arms=ARMS,folds=5,updates=1200,batch=10,lr=.001,weight_decay=0,threads=2,source_validation_sha256=sha(source.ROOT/'independent-validation.json'),source_samples_sha256=sha(source.ROOT/'train-samples.csv'),samples_sha256=sha(ROOT/'samples.csv'),script_sha256=sha(__file__),feature_script_sha256=sha(dynamics.__file__),baseline_validation_sha256=sha(REFERENCE/'independent-validation.json'),new_features='460dims:current-minus-mean5 and mean5-minus-previous10 over134statistics; grid64 magnitude change/cosine/weighted mean consistency; previous10 recovered from mean15/mean5. Complete15pairs only. Derived summary agreement not per-frame variance.',fold_routes={str(k):[r for r in routes if split[r]==k] for k in range(5)},batch_protocol='Uniform random replacement within each of10 label groups, one/group/batch; same saved batch sequence across arms/seeds within fold. Fold train group counts differ, so original exact balanced_batches is inapplicable.',normalization='Fold training samples only; inactive image mask zero',selection='Fixed final1200updates, all3seeds; no fold/epoch/seed selection',scope='Route crossvalidation of preselected balanced1000 source training clips; not independent new real-world data, not natural frame distribution; old31video validation unused',preserved_zip_sha256=sha('artifacts/stage3-motion-submit-candidate-20260910/submit.zip')))
 print(frame.groupby('fold').size().to_dict(),flush=True)

def train():
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['script_sha256']==sha(__file__)
 f=pd.read_csv(ROOT/'samples.csv');x={arm:get_arrays(f,arm) for arm in ARMS};results=[];started=time.monotonic();mask=torch.ones(1682);mask[:1280]=0
 for fold in range(5):
  ti=np.flatnonzero(f.fold.to_numpy()!=fold);vi=np.flatnonzero(f.fold.to_numpy()==fold);train=f.iloc[ti];valid=f.iloc[vi]
  rng=np.random.default_rng(20260911+fold);groups=[np.flatnonzero(train.group.to_numpy()==g) for g in range(10)];batches=np.stack([np.array([rng.choice(g) for g in groups]) for _ in range(1200)])
  np.save(ROOT/f'fold-{fold}-batches.npy',batches)
  ta=torch.tensor(train.accel.to_numpy());ts=torch.tensor(train.steer.to_numpy())
  for seed in SEEDS:
   for arm in ARMS:
    out=ROOT/f'{fold}-{seed}-{arm}';out.mkdir();mask=torch.from_numpy(dynamics.mask(arm))
    if arm=='baseline' and (fold,seed)!=(0,SEEDS[0]):
     original=REFERENCE/out.name
     for name in ['model.pt','report.json','logits.npz']:shutil.copyfile(original/name,out/name)
     reused=json.loads((out/'report.json').read_text());reused['reused']=True;results.append(reused);continue
    mean=x[arm][ti].mean(0);std=x[arm][ti].std(0).clip(.01);xt=torch.from_numpy(np.clip((x[arm][ti]-mean)/std,-10,10))*mask;xv=torch.from_numpy(np.clip((x[arm][vi]-mean)/std,-10,10))*mask
    torch.manual_seed(seed);model=base.Control();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.);history=[]
    for step,ids in enumerate(batches,1):
     a,s=model(xt[ids]);moving=ta[ids]!=3;loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving]);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
     if step%100==0:history.append(dict(step=step,loss=float(loss.detach())))
    model.eval()
    with torch.inference_mode():a,s=model(xv);a=a.numpy();s=s.numpy()
    np.savez_compressed(out/'logits.npz',accel=a,steer=s,indices=vi)
    checkpoint=dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),mask=mask,train_indices=ti.tolist(),validation_indices=vi.tolist(),fold=fold,seed=seed,arm=arm,steps=1200);torch.save(checkpoint,out/'model.pt')
    if arm=='baseline':
     original=torch.load(REFERENCE/out.name/'model.pt',weights_only=True)
     assert all(torch.equal(v,original['model'][k]) for k,v in checkpoint['model'].items())
     with np.load(REFERENCE/out.name/'logits.npz') as z:np.testing.assert_array_equal(a,z['accel']);np.testing.assert_array_equal(s,z['steer'])
    report=dict(reused=False,fold=fold,seed=seed,arm=arm,train_rows=len(ti),validation_rows=len(vi),metrics=score(valid,a.argmax(1),s.argmax(1)),checkpoint_sha256=sha(out/'model.pt'),history=history)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');results.append(report);save('status.json',dict(status='TRAINING',completed=len(results),total=30,seconds=time.monotonic()-started));print('trained',len(results),fold,seed,arm,report['metrics']['steer']['macro_f1'],flush=True)
 save('results.json',dict(status='TRAINING_COMPLETE',runs=results,seconds=time.monotonic()-started))

def verify():
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['script_sha256']==sha(__file__) and plan['samples_sha256']==sha(ROOT/'samples.csv') and plan['source_samples_sha256']==sha(source.ROOT/'train-samples.csv')
 assert plan['feature_script_sha256']==sha(dynamics.__file__) and plan['baseline_validation_sha256']==sha(REFERENCE/'independent-validation.json')
 f=pd.read_csv(ROOT/'samples.csv');orig=pd.read_csv(source.ROOT/'train-samples.csv');pd.testing.assert_frame_equal(f.drop(columns='fold'),orig)
 assert plan['preserved_zip_sha256']==sha('artifacts/stage3-motion-submit-candidate-20260910/submit.zip')
 for sid in f.ID.unique():
  meta=json.loads((source.ROOT/'cache'/(sid+'.json')).read_text());assert sha(source.ROOT/'cache'/(sid+'.npz'))==meta['sha256']
 x={a:get_arrays(f,a) for a in ARMS};runs=json.loads((ROOT/'results.json').read_text())['runs'];assert len(runs)==30
 oof={(seed,arm):np.full((len(f),2),-1,int) for seed in SEEDS for arm in ARMS};deltas=[]
 for r in runs:
  fold,seed,arm=r['fold'],r['seed'],r['arm'];out=ROOT/f'{fold}-{seed}-{arm}';assert sha(out/'model.pt')==r['checkpoint_sha256'];ck=torch.load(out/'model.pt',weights_only=True);ti=np.array(ck['train_indices']);vi=np.array(ck['validation_indices'])
  assert set(f.iloc[ti].route).isdisjoint(f.iloc[vi].route) and np.array_equal(vi,np.flatnonzero(f.fold.to_numpy()==fold));assert len(ti)+len(vi)==1000
  mean=x[arm][ti].mean(0);std=x[arm][ti].std(0).clip(.01);np.testing.assert_array_equal(ck['mean'],mean);np.testing.assert_array_equal(ck['std'],std);np.testing.assert_array_equal(ck['mask'],dynamics.mask(arm))
  model=base.Control();model.load_state_dict(ck['model'],strict=True);model.eval();v=torch.from_numpy(np.clip((x[arm][vi]-mean)/std,-10,10))*ck['mask']
  with torch.inference_mode():a,s=model(v);a=a.numpy();s=s.numpy()
  with np.load(out/'logits.npz') as z:np.testing.assert_array_equal(a,z['accel']);np.testing.assert_array_equal(s,z['steer']);np.testing.assert_array_equal(vi,z['indices'])
  assert score(f.iloc[vi],a.argmax(1),s.argmax(1))==r['metrics'];oof[seed,arm][vi]=np.stack([a.argmax(1),s.argmax(1)],axis=1)
 for fold in range(5):
  b=np.load(ROOT/f'fold-{fold}-batches.npy');tr=f[f.fold!=fold];np.testing.assert_array_equal(b,np.load(REFERENCE/f'fold-{fold}-batches.npy'));assert b.shape==(1200,10) and np.all(tr.group.to_numpy()[b]==np.arange(10))
 summary=[];routes=[]
 for (seed,arm),pred in oof.items():
  assert np.all(pred>=0);m=score(f,pred[:,0],pred[:,1]);summary.append(dict(seed=seed,arm=arm,**m))
  p=f[['ID','endpoint','route','fold']].copy();p['accel_label']=np.array(base.p.ACCEL)[pred[:,0]];p['steer_label']=np.array(base.p.STEER)[pred[:,1]];p.to_csv(ROOT/f'oof-{seed}-{arm}.csv',index=False)
  for route,g in f.groupby('route'):
   idx=g.index.to_numpy();sm=score(g,pred[idx,0],pred[idx,1]);routes.append(dict(seed=seed,arm=arm,route=route,rows=len(g),steer_f1=sm['steer']['macro_f1'],recall_left=sm['steer']['recall'][0],recall_straight=sm['steer']['recall'][1],recall_right=sm['steer']['recall'][2]))
 pd.DataFrame(routes).to_csv(ROOT/'route-scores.csv',index=False)
 save('summary.json',dict(scope=plan['scope'],pooled_oof=summary,fold_metrics=[dict(fold=r['fold'],seed=r['seed'],arm=r['arm'],**r['metrics']) for r in runs]))
 save('independent-validation.json',dict(status='PASS',models=30,oof_rows_per_seed_arm=1000,max_logit_replay_error=0,results_sha256=sha(ROOT/'results.json'),summary_sha256=sha(ROOT/'summary.json'),submission_unchanged=True))
 save('status.json',dict(status='COMPLETE_VALIDATED',models=30));print(json.dumps(summary,indent=2),flush=True)
def get_arrays(frame,arm):
 x=source.arrays(frame,'baseline')
 return x if arm=='baseline' else dynamics.features(x,frame.endpoint.to_numpy())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','verify']);args=p.parse_args();torch.set_num_threads(2);getattr(__import__(__name__),args.command)()
