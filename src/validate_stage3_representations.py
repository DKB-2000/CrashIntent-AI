"""Independent coverage, cached-feature, checkpoint and metric verification."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
import compare_stage3_representations as c

def metrics(truth,pred,n):
 matrix=np.zeros((n,n),dtype=np.int64)
 for t,p in zip(truth,pred):matrix[int(t),int(p)]+=1
 support=matrix.sum(1);predicted=matrix.sum(0);tp=matrix.diagonal();den=support+predicted
 f1=np.divide(2*tp,den,out=np.zeros(n),where=den!=0)
 recall=np.divide(tp,support,out=np.zeros(n),where=support!=0)
 return dict(macro_f1=float(f1.mean()),rows=int(support.sum()),recall=recall.tolist(),support=support.tolist(),predicted=predicted.tolist(),confusion_matrix=matrix.tolist())

def main():
 torch.set_num_threads(2);c.cv2.setNumThreads(1)
 root=c.ROOT;plan=json.loads((root/'plan.json').read_text());results=json.loads((root/'results.json').read_text())
 assert results['status']=='COMPLETE' and len(results['runs'])==9
 assert {(r['seed'],r['mode']) for r in results['runs']}=={(s,m) for s in c.SEEDS for m in c.MODES}
 assert plan['script_sha256']==c.sha(c.__file__) and plan['pipeline_sha256']==c.sha(c.p.__file__)
 frames={part:pd.read_csv(root/(part+'-samples.csv')) for part in ('train','validation')}
 assert len(frames['train'])==1000 and len(frames['validation'])==18603
 assert c.sha(root/'train-samples.csv')==plan['training_samples_sha256']
 assert c.sha(root/'validation-samples.csv')==plan['validation_samples_sha256']
 assert set(frames['train'].route).isdisjoint(frames['validation'].route)
 with zipfile.ZipFile(c.ARCHIVE) as z:
  report=json.loads(z.read('trial-run/report.json'))
  source=pd.DataFrame(report['samples'])
 pd.testing.assert_frame_equal(frames['train'][source.columns],source,check_dtype=False)
 assert c.sha(c.ARCHIVE)==plan['archive_sha256']
 for part in frames:
  labels=pd.read_csv(c.DATA/('labels_'+part+'_candidate.csv')).rename(columns={'frame_index':'endpoint'})
  assert c.sha(c.DATA/('labels_'+part+'_candidate.csv'))==plan['data_sha256']['labels_'+part+'_candidate.csv']
  merged=frames[part].merge(labels,on=['ID','endpoint'],validate='one_to_one',suffixes=('','_source'))
  assert len(merged)==len(frames[part])
  assert merged.accel.tolist()==merged['accel_label_source' if 'accel_label_source' in merged else 'accel_label'].map(dict(zip(c.p.ACCEL,range(4)))).tolist()
  assert merged.steer.tolist()==merged['steer_label_source' if 'steer_label_source' in merged else 'steer_label'].map(dict(zip(c.p.STEER,range(3)))).tolist()
 # Verify all cached rows, source videos and selected endpoints; materialize once.
 data={};video_count=0
 for part,f in frames.items():
  loaded={}
  for sid,g in f.groupby('ID'):
   info=json.loads((root/'cache'/(sid+'.json')).read_text());path=(c.DATA/g.video.iloc[0]).resolve()
   assert info['video_sha256']==c.sha(path)==plan['original_bundle_hashes']['dataset/videos/'+sid+'.mp4']
   assert info['features_sha256']==c.sha(root/'cache'/(sid+'.npz')) and info['script_sha256']==plan['script_sha256']
   with np.load(root/'cache'/(sid+'.npz')) as z:
    x=z['features'];ends=z['endpoints'];assert ends.tolist()==sorted(g.endpoint.tolist())
    assert x.shape==(len(g),c.IMAGE_DIM+c.MOTION_DIM) and np.isfinite(x).all()
    loaded[sid]={int(t):x[i] for i,t in enumerate(ends)}
   video_count+=1
  data[part]=np.stack([loaded[r.ID][int(r.endpoint)] for r in f.itertuples()])
 print('PASS cached rows and original hashes',video_count,flush=True)
 # Real-video sparse extraction consistency, one train and one validation video.
 for part,f in frames.items():
  sid=f.ID.iloc[0];g=f[f.ID==sid];ends=sorted(g.endpoint.unique());chosen=[ends[0],ends[-1]]
  actual,_=c.extract_video((c.DATA/g.video.iloc[0]).resolve(),chosen)
  indices=[int(np.flatnonzero((f.ID==sid)&(f.endpoint==end))[0]) for end in chosen]
  np.testing.assert_allclose(actual,data[part][indices],rtol=0,atol=0)
 print('PASS real sparse feature replay',flush=True)
 # Independently count all 51 route/steer strata from original train labels.
 train_labels=pd.read_csv(c.DATA/'labels_train_candidate.csv')
 manifest=pd.read_csv(c.DATA/'split_manifest.csv')
 full=train_labels.merge(manifest[['ID','route']],on='ID',validate='many_to_one')
 coverage=pd.read_csv(root/'route-steer-coverage.csv')
 assert len(coverage)==51
 joint=frames['train'][frames['train'].accel!=3].groupby(['route','accel','steer']).size()
 assert len(joint)==153 and int(joint.min())==2 and int(joint.max())==8
 assert frames['train'].groupby('group').size().tolist()==[100]*10
 for row in coverage.itertuples():
  eligible=full[(full.route==row.route)&(full.accel_label!='STOPPED')&(full.steer_label==row.steer)]
  selected=frames['train'][(frames['train'].route==row.route)&(frames['train'].accel!=3)&(frames['train'].steer==c.p.STEER.index(row.steer))]
  assert len(eligible)==row.available_frames
  assert int(((eligible.frame_index>=15)&((eligible.frame_index-15)%16==0)).sum())==row.eligible_clips
  assert len(selected)==row.selected_clips and selected.ID.nunique()==row.selected_videos
 mean=data['train'].mean(0);std=data['train'].std(0).clip(.01)
 normalized={part:torch.from_numpy(np.clip((x-mean)/std,-10,10)) for part,x in data.items()}
 rows=[];route_rows=[];class_rows=[];max_logit_error=0.
 for run in results['runs']:
  mode,seed=run['mode'],run['seed'];directory=root/(str(seed)+'-'+mode)
  assert len(run['history'])==12 and sum(r['updates'] for r in run['history'])==1200
  assert c.sha(directory/'model.pt')==run['checkpoint_sha256']
  ckpt=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
  np.testing.assert_array_equal(ckpt['mean'].numpy(),mean);np.testing.assert_array_equal(ckpt['std'].numpy(),std)
  mask=np.ones(c.IMAGE_DIM+c.MOTION_DIM,dtype=np.float32)
  if mode=='image':mask[c.IMAGE_DIM:]=0
  if mode=='motion':mask[:c.IMAGE_DIM]=0
  np.testing.assert_array_equal(ckpt['mask'].numpy(),mask)
  torch.manual_seed(seed);initial=c.Control()
  excluded=mask==0
  assert torch.equal(initial.net[0].weight[:,excluded],ckpt['model']['net.0.weight'][:,excluded])
  model=c.Control();model.load_state_dict(ckpt['model'],strict=True);model.eval()
  for part,f in frames.items():
   with torch.inference_mode():
    output=[model(x*ckpt['mask']) for x in normalized[part].split(256)]
   logits=[torch.cat([o[k] for o in output]).numpy() for k in range(2)]
   with np.load(directory/(part+'-logits.npz')) as saved:
    for head,z in zip(['accel','steer'],logits):
     error=float(np.max(np.abs(z-saved[head])));max_logit_error=max(max_logit_error,error)
     np.testing.assert_allclose(z,saved[head],rtol=1e-5,atol=1e-6)
   prediction=pd.read_csv(directory/(part+'-predictions.csv'))
   assert prediction.ID.tolist()==f.ID.tolist() and prediction.sample_index.tolist()==f.endpoint.tolist()
   ap,sp=[z.argmax(1) for z in logits]
   assert prediction.accel_label.tolist()==np.array(c.p.ACCEL)[ap].tolist()
   assert prediction.steer_label.tolist()==np.array(c.p.STEER)[sp].tolist()
   moving=f.accel.to_numpy()!=3
   am=metrics(f.accel,ap,4);sm=metrics(f.steer.to_numpy()[moving],sp[moving],3)
   for h,m in [('accel',am),('steer',sm)]:
    assert abs(m['macro_f1']-run['metrics'][part][h]['macro_f1'])<1e-12
    assert m['confusion_matrix']==run['metrics'][part][h]['confusion_matrix']
   rows.append(dict(model=mode,seed=seed,split=part,accel_f1=am['macro_f1'],steer_f1=sm['macro_f1']))
   if part=='validation':
    for route,g in f.assign(pred=sp).groupby('route'):
     g=g[g.accel!=3];m=metrics(g.steer,g.pred,3)
     route_rows.append(dict(model=mode,seed=seed,route=route,steer_f1=m['macro_f1']))
     for i,label in enumerate(c.p.STEER):class_rows.append(dict(model=mode,seed=seed,route=route,label=label,recall=m['recall'][i],support=m['support'][i],predicted=m['predicted'][i]))
  print('PASS checkpoint replay',seed,mode,flush=True)
 # Existing MViT: exact archived predictions and same selected training endpoints.
 for part,f in frames.items():
  pred=pd.read_csv(root/('mvit-'+part+'-predictions.csv'))
  g=f.merge(pred.rename(columns={'sample_index':'endpoint'}),on=['ID','endpoint'],validate='one_to_one',suffixes=('','_prediction'))
  assert len(g)==len(f)==len(pred)
  ap=g['accel_label_prediction' if 'accel_label_prediction' in g else 'accel_label'].map(dict(zip(c.p.ACCEL,range(4)))).to_numpy()
  sp=g['steer_label_prediction' if 'steer_label_prediction' in g else 'steer_label'].map(dict(zip(c.p.STEER,range(3)))).to_numpy()
  moving=g.accel.to_numpy()!=3;am=metrics(g.accel,ap,4);sm=metrics(g.steer.to_numpy()[moving],sp[moving],3)
  reference=report['train_metrics' if part=='train' else 'validation']
  assert abs(am['macro_f1']-reference['accel']['macro_f1'])<1e-12
  assert abs(sm['macro_f1']-reference['moving_steer']['macro_f1'])<1e-12
  rows.append(dict(model='mvit_reference',seed=report['config']['seed'],split=part,accel_f1=am['macro_f1'],steer_f1=sm['macro_f1']))
  if part=='validation':
   for route,v in g.assign(pred=sp).groupby('route'):
    v=v[v.accel!=3];m=metrics(v.steer,v.pred,3);route_rows.append(dict(model='mvit_reference',seed=report['config']['seed'],route=route,steer_f1=m['macro_f1']))
    for i,label in enumerate(c.p.STEER):class_rows.append(dict(model='mvit_reference',seed=report['config']['seed'],route=route,label=label,recall=m['recall'][i],support=m['support'][i],predicted=m['predicted'][i]))
 pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
 pd.DataFrame(route_rows).to_csv(root/'route-comparison.csv',index=False)
 pd.DataFrame(class_rows).to_csv(root/'route-class-comparison.csv',index=False)
 validation=dict(status='PASS',original_videos=video_count,feature_rows=sum(len(f) for f in frames.values()),
  real_feature_replay_rows=4,coverage_strata=51,control_runs=9,max_checkpoint_logit_error=max_logit_error,
  comparison_sha256=c.sha(root/'comparison.csv'),results_sha256=c.sha(root/'results.json'),validator_sha256=c.sha(__file__))
 c.save(root/'independent-validation.json',validation);c.save(root/'status.json',dict(status='COMPLETE',independent_validation='PASS'))
 print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
