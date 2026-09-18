"""Predeclared causal temporal-window controls; no deployment mutation."""
import os
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','threads;1')
import json,time,shutil,argparse
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from torch import nn
from torch.nn import functional as F
import compare_stage3_representations as c
ROOT=Path('artifacts/stage3-temporal-motion-20260910')
ARMS={'baseline':(5,15,False),'short':(3,5,False),'long':(15,30,False),'long_image':(15,30,True)}

def prepare():
 ROOT.mkdir(exist_ok=True)
 assert not (ROOT/'plan.json').exists()
 original=json.loads((c.ROOT/'independent-validation.json').read_text())
 assert original['status']=='PASS' and c.sha(c.ROOT/'results.json')==original['results_sha256']
 for part in ('train','validation'):shutil.copyfile(c.ROOT/(part+'-samples.csv'),ROOT/(part+'-samples.csv'))
 c.save(ROOT/'plan.json',dict(status='PREDECLARED',arms=ARMS,seeds=c.SEEDS,epochs=12,updates=1200,batch=10,lr=.001,weight_decay=0,threads=1,script_sha256=c.sha(__file__),source_script_sha256=c.sha(c.__file__),source_results_sha256=c.sha(c.ROOT/'results.json'),source_validation_sha256=c.sha(c.ROOT/'independent-validation.json'),samples_sha256={p:c.sha(ROOT/(p+'-samples.csv')) for p in ('train','validation')},normalization='1000 selected train rows only',checkpoint_selection='fixed final epoch12',hypothesis='short improves transition lag; long stabilizes straight/turn; long_image tests appearance added at fixed long windows',temporal='current pair plus means over last W pairs, excludes dummy flow0, no future, reset per video',source_baseline='reused three verified motion checkpoints; exact seed20260910 retrain',evaluation='moving steer macroF1 primary; route class recall, accel and causal transition segments secondary'))
 print('PREDECLARED',flush=True)

def dense_train(path):
 cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert cap.isOpened()
 flows=[];prev=None
 for i in range(n):
  ok,bgr=cap.read();assert ok
  rgb=cv2.resize(c.p.prepare_frame(bgr).transpose(1,2,0),(96,96),interpolation=cv2.INTER_AREA)
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  if prev is None:row=np.zeros(134,np.float32)
  else:
   flow=np.zeros((96,96,2),np.float32) if np.array_equal(prev,gray) else cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0)
   row=np.r_[cv2.resize(flow,(8,8),interpolation=cv2.INTER_AREA).reshape(-1),np.percentile(np.linalg.norm(flow,axis=2),[10,50,90,99]),flow.mean((0,1))].astype(np.float32)
  flows.append(row);prev=gray
 cap.release();return np.asarray(flows)

def motion_at(flow,t,w1,w2):
 def avg(w):return flow[max(1,t-w+1):t+1].mean(0) if t else np.zeros(134,np.float32)
 return np.r_[flow[t],avg(w1),avg(w2)]

def extract():
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['script_sha256']==c.sha(__file__)
 cache=ROOT/'cache';cache.mkdir(exist_ok=True);start=time.monotonic();k=0
 for part in ('train','validation'):
  frame=pd.read_csv(ROOT/(part+'-samples.csv'))
  for sid,g in frame.groupby('ID'):
   target=cache/(sid+'.npz');info=cache/(sid+'.json')
   if target.exists() and info.exists():
    meta=json.loads(info.read_text());assert meta['sha256']==c.sha(target);k+=1;continue
   source=c.ROOT/'cache'/(sid+'.npz');meta=json.loads((c.ROOT/'cache'/(sid+'.json')).read_text())
   assert c.sha(source)==meta['features_sha256']
   path=(c.DATA/g.video.iloc[0]).resolve();assert c.sha(path)==meta['video_sha256']
   with np.load(source) as z:old=z['features'];ends=z['endpoints']
   if part=='validation':
    assert ends.tolist()==list(range(len(ends)));flow=old[:,1280:1414].copy()
   else:flow=dense_train(path)
   baseline=np.stack([motion_at(flow,int(t),5,15) for t in ends])
   np.testing.assert_array_equal(baseline,old[:,1280:])
   values={arm:np.concatenate([old[:,:1280],np.stack([motion_at(flow,int(t),v[0],v[1]) for t in ends])],axis=1) for arm,v in ARMS.items()}
   np.savez_compressed(target,endpoints=ends,**values)
   c.save(info,dict(ID=sid,sha256=c.sha(target),source_cache_sha256=c.sha(source),video_sha256=meta['video_sha256'],script_sha256=c.sha(__file__),part=part,frames=len(flow),baseline_exact=True))
   k+=1;c.save(ROOT/'status.json',dict(status='EXTRACTING',videos_completed=k,total=187,seconds=time.monotonic()-start));print('features',k,187,sid,round(time.monotonic()-start),flush=True)
 c.save(ROOT/'status.json',dict(status='FEATURES_COMPLETE',videos=187,seconds=time.monotonic()-start))

def arrays(frame,arm):
 rows={}
 for sid in frame.ID.unique():
  with np.load(ROOT/'cache'/(sid+'.npz')) as z:rows[sid]={int(t):x for t,x in zip(z['endpoints'],z[arm])}
 return np.stack([rows[r.ID][int(r.endpoint)] for r in frame.itertuples()])

def train():
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['script_sha256']==c.sha(__file__)
 frames={p:pd.read_csv(ROOT/(p+'-samples.csv')) for p in ('train','validation')};ta=torch.tensor(frames['train'].accel.to_numpy());ts=torch.tensor(frames['train'].steer.to_numpy());results=[]
 for arm,(w1,w2,image) in ARMS.items():
  vals={p:arrays(f,arm) for p,f in frames.items()};mean=vals['train'].mean(0);std=vals['train'].std(0).clip(.01)
  normalized={p:torch.from_numpy(np.clip((x-mean)/std,-10,10)) for p,x in vals.items()};mask=torch.ones(1682)
  if not image:mask[:1280]=0
  for seed in c.SEEDS:
   out=ROOT/(str(seed)+'-'+arm);out.mkdir(exist_ok=True)
   if (out/'report.json').exists():results.append(json.loads((out/'report.json').read_text()));continue
   if arm=='baseline' and seed!=c.SEEDS[0]:
    source=c.ROOT/(str(seed)+'-motion')
    for name in ('model.pt','train-logits.npz','validation-logits.npz','train-predictions.csv','validation-predictions.csv'):shutil.copyfile(source/name,out/name)
    report=json.loads((source/'report.json').read_text());report.update(mode=arm,reused=True,source_checkpoint_sha256=c.sha(source/'model.pt'));c.save(out/'report.json',report);results.append(report);continue
   torch.manual_seed(seed);model=c.Control();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.);history=[];start=time.monotonic();x=normalized['train']*mask
   for epoch in range(1,13):
    losses=[];model.train()
    for ids in c.balanced_batches(frames['train'].to_dict('records'),epoch):
     a,s=model(x[ids]);moving=ta[ids]!=3;loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving]);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
    history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
   model.eval();scores={}
   for part in frames:
    with torch.inference_mode():output=[model(chunk*mask) for chunk in normalized[part].split(256)]
    a,s=[torch.cat([r[k] for r in output]).numpy() for k in range(2)]
    np.savez_compressed(out/(part+'-logits.npz'),accel=a,steer=s)
    pred=frames[part][['ID','endpoint']].rename(columns={'endpoint':'sample_index'}).copy();pred['accel_label']=np.array(c.p.ACCEL)[a.argmax(1)];pred['steer_label']=np.array(c.p.STEER)[s.argmax(1)];pred.to_csv(out/(part+'-predictions.csv'),index=False);scores[part]=c.score(frames[part],a.argmax(1),s.argmax(1))
   checkpoint=dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),mask=mask,mode=arm,seed=seed,windows=[w1,w2]);torch.save(checkpoint,out/'model.pt')
   if arm=='baseline':
    original=torch.load(c.ROOT/(str(seed)+'-motion')/'model.pt',weights_only=True)
    for key,value in original['model'].items():torch.testing.assert_close(value,checkpoint['model'][key],rtol=0,atol=0)
    for part in frames:
     with np.load(c.ROOT/(str(seed)+'-motion')/(part+'-logits.npz')) as a,np.load(out/(part+'-logits.npz')) as b:
      for head in ('accel','steer'):np.testing.assert_array_equal(a[head],b[head])
   report=dict(status='COMPLETE',mode=arm,seed=seed,reused=False,history=history,metrics=scores,seconds=time.monotonic()-start,checkpoint_sha256=c.sha(out/'model.pt'),baseline_exact_retrain=arm=='baseline');c.save(out/'report.json',report);results.append(report);print('trained',seed,arm,scores['validation']['steer']['macro_f1'],flush=True)
 c.save(ROOT/'results.json',dict(status='COMPLETE',runs=results));c.save(ROOT/'status.json',dict(status='TRAINING_COMPLETE',runs=12))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','extract','train']);args=parser.parse_args();torch.set_num_threads(1);cv2.setNumThreads(1);globals()[args.command]()
