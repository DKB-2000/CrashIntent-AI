"""Independent saved-checkpoint, original labels and temporal feature validation."""
import json
import numpy as np,pandas as pd,torch
import stage3_temporal_motion_experiment as e
import compare_stage3_representations as c
from validate_stage3_representations import metrics

def main():
 torch.set_num_threads(2);c.cv2.setNumThreads(1);root=e.ROOT;plan=json.loads((root/'plan.json').read_text());result=json.loads((root/'results.json').read_text())
 optimization=json.loads((root/'execution-optimization.json').read_text());assert optimization['script_sha256']==c.sha('src/extract_stage3_temporal_sparse.py')
 correction=json.loads((root/'training-thread-correction.json').read_text());assert correction['script_sha256']==c.sha('src/run_stage3_temporal_training.py') and correction['original_plan_sha256']==c.sha(root/'plan.json')
 assert correction['numeric_proof_sha256']==c.sha(root/'baseline-thread-repro.json') and correction['training_threads']==2
 source_plan=json.loads((c.ROOT/'plan.json').read_text())
 assert plan['script_sha256']==c.sha(e.__file__) and plan['source_script_sha256']==c.sha(c.__file__)
 assert plan['source_results_sha256']==c.sha(c.ROOT/'results.json') and plan['source_validation_sha256']==c.sha(c.ROOT/'independent-validation.json')
 assert len(result['runs'])==12
 frames={p:pd.read_csv(root/(p+'-samples.csv')) for p in ('train','validation')}
 assert len(frames['train'])==1000 and len(frames['validation'])==18603
 assert set(frames['train'].route).isdisjoint(frames['validation'].route)
 data={p:{a:[] for a in e.ARMS} for p in frames};count=0
 for part,f in frames.items():
  assert c.sha(root/(part+'-samples.csv'))==plan['samples_sha256'][part]==c.sha(c.ROOT/(part+'-samples.csv'))
  assert c.sha(c.DATA/('labels_'+part+'_candidate.csv'))==source_plan['data_sha256']['labels_'+part+'_candidate.csv']
  assert c.sha(root/(part+'-samples.csv'))==source_plan['training_samples_sha256' if part=='train' else 'validation_samples_sha256']
  labels=pd.read_csv(c.DATA/('labels_'+part+'_candidate.csv')).rename(columns={'frame_index':'endpoint'})
  merged=f[['ID','endpoint','accel','steer']].merge(labels,on=['ID','endpoint'],validate='one_to_one');assert len(merged)==len(f)
  assert merged.accel.tolist()==merged.accel_label.map(dict(zip(c.p.ACCEL,range(4)))).tolist()
  assert merged.steer.tolist()==merged.steer_label.map(dict(zip(c.p.STEER,range(3)))).tolist()
  loaded={}
  for sid,g in f.groupby('ID'):
   path=root/'cache'/(sid+'.npz');meta=json.loads((root/'cache'/(sid+'.json')).read_text());assert c.sha(path)==meta['sha256'] and meta['script_sha256']==plan['script_sha256']
   if 'extractor_sha256' in meta:assert meta['extractor_sha256']==optimization['script_sha256']
   original=c.ROOT/'cache'/(sid+'.npz');assert c.sha(original)==meta['source_cache_sha256']==json.loads((c.ROOT/'cache'/(sid+'.json')).read_text())['features_sha256']
   assert c.sha((c.DATA/g.video.iloc[0]).resolve())==meta['video_sha256']==source_plan['original_bundle_hashes']['dataset/videos/'+sid+'.mp4'];count+=1
   with np.load(path) as z,np.load(original) as old:
    assert z['endpoints'].tolist()==sorted(g.endpoint.tolist());np.testing.assert_array_equal(z['baseline'],old['features']);loaded[sid]={a:{int(t):x for t,x in zip(z['endpoints'],z[a])} for a in e.ARMS}
    # Validation has dense independent current-pair data; reconstruct windows with explicit slicing.
    if part=='validation':
     flow=old['features'][:,1280:1414]
     for arm,(w1,w2,_) in e.ARMS.items():
      actual_features=z[arm];actual_endpoints=z['endpoints']
      for j,t in enumerate(actual_endpoints):
       pieces=[flow[t]]
       for w in (w1,w2):pieces.append(np.mean(flow[max(1,int(t)-w+1):int(t)+1],axis=0) if t else np.zeros(134,np.float32))
       np.testing.assert_array_equal(np.concatenate(pieces),actual_features[j,1280:])
  for arm in e.ARMS:data[part][arm]=np.stack([loaded[r.ID][arm][int(r.endpoint)] for r in f.itertuples()])
 print('PASS video hashes, labels, baseline and dense causal windows',count,flush=True)
 # Actual decoded flow on one train+validation video; compare sparse rows across arms.
 real_rows=0
 for part,f in frames.items():
  sid=f.ID.iloc[0];g=f[f.ID==sid];flow=e.dense_train((c.DATA/g.video.iloc[0]).resolve())
  chosen=sorted(g.endpoint.unique())[::max(1,len(g)//3)]
  for t in chosen:
   idx=np.flatnonzero((f.ID==sid)&(f.endpoint==t))[0]
   for arm,(w1,w2,_) in e.ARMS.items():np.testing.assert_array_equal(e.motion_at(flow,int(t),w1,w2),data[part][arm][idx,1280:])
   real_rows+=1
 rows=[];route_rows=[];maximum=0.
 for run in result['runs']:
  arm=run['mode'];seed=run['seed'];out=root/(str(seed)+'-'+arm);assert c.sha(out/'model.pt')==run['checkpoint_sha256'];assert sum(x['updates'] for x in run['history'])==1200
  ck=torch.load(out/'model.pt',weights_only=True);model=c.Control();model.load_state_dict(ck['model'],strict=True);model.eval();mean=data['train'][arm].mean(0);std=data['train'][arm].std(0).clip(.01)
  np.testing.assert_array_equal(ck['mean'],mean);np.testing.assert_array_equal(ck['std'],std)
  mask=np.ones(1682,np.float32)
  if not e.ARMS[arm][2]:mask[:1280]=0
  np.testing.assert_array_equal(ck['mask'],mask)
  for part,f in frames.items():
   x=torch.from_numpy(np.clip((data[part][arm]-mean)/std,-10,10))
   with torch.inference_mode():outputs=[model(v*ck['mask']) for v in x.split(256)]
   logits=[torch.cat([v[k] for v in outputs]).numpy() for k in range(2)]
   with np.load(out/(part+'-logits.npz')) as saved:
    for head,actual in zip(('accel','steer'),logits):np.testing.assert_allclose(saved[head],actual,rtol=1e-5,atol=1e-6);maximum=max(maximum,float(abs(saved[head]-actual).max()))
   pred=pd.read_csv(out/(part+'-predictions.csv'));assert pred.ID.tolist()==f.ID.tolist() and pred.sample_index.tolist()==f.endpoint.tolist()
   ap,sp=[a.argmax(1) for a in logits];assert pred.accel_label.tolist()==np.array(c.p.ACCEL)[ap].tolist();assert pred.steer_label.tolist()==np.array(c.p.STEER)[sp].tolist();moving=f.accel.to_numpy()!=3
   am=metrics(f.accel,ap,4);sm=metrics(f.steer.to_numpy()[moving],sp[moving],3)
   for head,m in [('accel',am),('steer',sm)]:assert abs(m['macro_f1']-run['metrics'][part][head]['macro_f1'])<1e-12
   rows.append(dict(mode=arm,seed=seed,split=part,accel_f1=am['macro_f1'],steer_f1=sm['macro_f1'],steer_balanced_accuracy=float(np.mean(sm['recall']))))
   if part=='validation':
    for route,g in f.assign(pred=sp).query('accel != 3').groupby('route'):
     m=metrics(g.steer,g.pred,3)
     for j,label in enumerate(c.p.STEER):route_rows.append(dict(mode=arm,seed=seed,route=route,label=label,recall=m['recall'][j],support=m['support'][j],steer_f1=m['macro_f1']))
  print('PASS checkpoint',arm,seed,flush=True)
 pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False);pd.DataFrame(route_rows).to_csv(root/'route-class-comparison.csv',index=False)
 c.save(root/'independent-validation.json',dict(status='PASS',runs=12,original_videos=count,feature_rows=19603,real_replay_rows=real_rows,max_logit_error=maximum,results_sha256=c.sha(root/'results.json'),validator_sha256=c.sha(__file__)))
 c.save(root/'status.json',dict(status='COMPLETE',independent_validation='PASS'));print(pd.DataFrame(rows).query('split == "validation"').to_string(index=False))
if __name__=='__main__':main()
