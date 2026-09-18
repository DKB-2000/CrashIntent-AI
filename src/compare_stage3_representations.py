"""Fixed-sample appearance/motion controls and route coverage audit; local only."""
import os
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','threads;1')
import argparse, hashlib, json, time, zipfile
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import stage3_pipeline as p
from train_stage3_route_trial import balanced_batches

ROOT=Path('artifacts/stage3-representation-20260910')
DATA=Path('artifacts/stage3-comma-chunk1-calibrated-v1')
ARCHIVE=Path('artifacts/kaggle-stage3-expanded-20260910/result/stage3-route-trial-result.zip')
MODES=('image','motion','combined')
IMAGE_DIM=1280
MOTION_DIM=402
SEEDS=(20260910,20260911,20260912)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def inputs():
 with zipfile.ZipFile(ARCHIVE) as z:
  report=json.loads(z.read('trial-run/report.json'))
  predictions={part:pd.read_csv(z.open('trial-run/'+part+'_predictions.csv')) for part in ('train','validation')}
 verified=json.loads(Path('artifacts/kaggle-stage3-expanded-20260910/result-validation.json').read_text())
 assert verified['status']=='PASS' and sha(ARCHIVE)==verified['archive_sha256']
 for n,h in report['data_sha256'].items():
  if n != 'split_manifest.csv':assert sha(DATA/n)==h
 samples=pd.DataFrame(report['samples'])
 manifest=pd.read_csv(DATA/'split_manifest.csv')
 with zipfile.ZipFile('artifacts/kaggle-stage3-training/dataset/stage3-training-v1.zip') as bundle:
  data=bundle.read('dataset/split_manifest.csv')
  assert hashlib.sha256(data).hexdigest()==report['data_sha256']['split_manifest.csv']
  portable=pd.read_csv(bundle.open('dataset/split_manifest.csv'))
  pd.testing.assert_frame_equal(manifest[['ID','route','split']].sort_values('ID').reset_index(drop=True),portable[['ID','route','split']].sort_values('ID').reset_index(drop=True))
 assert manifest.groupby('route').split.nunique().eq(1).all()
 train=pd.read_csv(DATA/'labels_train_candidate.csv').rename(columns={'frame_index':'endpoint'})
 check=samples.merge(train,on=['ID','endpoint'],validate='one_to_one')
 assert len(check)==1000 and not samples[['ID','endpoint']].duplicated().any()
 assert check.accel_label.tolist()==[p.ACCEL[i] for i in check.accel]
 assert check.steer_label.tolist()==[p.STEER[i] for i in check.steer]
 valid=pd.read_csv(DATA/'labels_validation_candidate.csv').rename(columns={'frame_index':'endpoint'})
 valid=valid.merge(manifest[['ID','route','video']],on='ID',validate='many_to_one')
 valid['accel']=valid.accel_label.map(dict(zip(p.ACCEL,range(4))))
 valid['steer']=valid.steer_label.map(dict(zip(p.STEER,range(3))))
 samples=samples.merge(manifest[['ID','video']],on='ID',validate='many_to_one')
 assert set(samples.route).isdisjoint(valid.route)
 return samples,valid,manifest,train,report,predictions

def prepare():
 ROOT.mkdir(exist_ok=True);samples,valid,manifest,train,report,preds=inputs()
 samples.to_csv(ROOT/'train-samples.csv',index=False);valid.to_csv(ROOT/'validation-samples.csv',index=False)
 full=train.merge(manifest[['ID','route']],on='ID',validate='many_to_one')
 eligible=full[(full.endpoint>=15)&((full.endpoint-15)%16==0)]
 rows=[]
 for route in sorted(samples.route.unique()):
  for label in p.STEER:
   a=full[(full.route==route)&(full.accel_label!='STOPPED')&(full.steer_label==label)]
   e=eligible[(eligible.route==route)&(eligible.accel_label!='STOPPED')&(eligible.steer_label==label)]
   s=samples[(samples.route==route)&(samples.accel!=3)&(samples.steer==p.STEER.index(label))]
   rows.append(dict(route=route,steer=label,available_frames=len(a),eligible_clips=len(e),selected_clips=len(s),
    selected_videos=int(s.ID.nunique()),selection_fraction=len(s)/len(e) if len(e) else None))
 pd.DataFrame(rows).to_csv(ROOT/'route-steer-coverage.csv',index=False)
 joint=samples.groupby(['route','accel','steer']).size().rename('clips').reset_index()
 joint.to_csv(ROOT/'route-joint-coverage.csv',index=False)
 coverage=pd.DataFrame(rows)
 config=dict(status='PREPARED',scope='External reused development routes; no competition data',
  archive_sha256=sha(ARCHIVE),data_sha256=report['data_sha256'],script_sha256=sha(__file__),pipeline_sha256=sha(p.__file__),
  training_samples_sha256=sha(ROOT/'train-samples.csv'),validation_samples_sha256=sha(ROOT/'validation-samples.csv'),
  seeds=list(SEEDS),modes=list(MODES),epochs=12,updates=1200,effective_batch=10,learning_rate=.001,weight_decay=0.,
  checkpoint_selection='fixed final, no validation selection',normalization='selected training samples only',
  image_dimensions=IMAGE_DIM,motion_dimensions=MOTION_DIM,
  comparison='Three identically initialized and sized MLPs with masked feature groups; MViT is an architecture-confounded reference',
  coverage=dict(train_routes=int(samples.route.nunique()),train_videos=int(samples.ID.nunique()),
   eligible_nonempty_missing=coverage[(coverage.eligible_clips>0)&(coverage.selected_clips==0)].to_dict('records'),
   min_route_steer=int(coverage.selected_clips.min()),max_route_steer=int(coverage.selected_clips.max())))
 with zipfile.ZipFile('artifacts/kaggle-stage3-training/dataset/stage3-training-v1.zip') as bundle:
  config['original_bundle_hashes']=json.loads(bundle.read('bundle.json'))['file_sha256']
 save(ROOT/'plan.json',config)
 for part,frame in preds.items():frame.to_csv(ROOT/('mvit-'+part+'-predictions.csv'),index=False)
 print(json.dumps(config['coverage'],indent=2));print(coverage.to_string(index=False))

def appearance(rgb):
 small=cv2.resize(rgb,(16,16),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)/255
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
 gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
 mag,angle=cv2.cartToPolar(gx,gy)
 bins=np.floor((angle % np.pi)/(np.pi/8)).astype(int).clip(0,7)
 hist=np.stack([cv2.resize(np.where(bins==b,mag,0),(8,8),interpolation=cv2.INTER_AREA) for b in range(8)],axis=-1)
 return np.concatenate([small,hist.reshape(-1)]).astype(np.float32)

def extract_video(path,endpoints):
 # Same resize/center crop as MViT, reduced to 96 square for fixed feature extraction.
 need={i for end in endpoints for i in range(max(0,int(end)-15),int(end)+1)}
 cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
 assert cap.isOpened() and abs(fps-10)<.01
 images={};flows={};previous=None;previous_i=-2
 for i in range(n):
  ok=cap.grab();assert ok
  if i not in need:continue
  ok,bgr=cap.retrieve();assert ok
  rgb=p.prepare_frame(bgr).transpose(1,2,0)
  rgb=cv2.resize(rgb,(96,96),interpolation=cv2.INTER_AREA)
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  if previous is not None and previous_i==i-1:
   flow=np.zeros((96,96,2),np.float32) if np.array_equal(previous,gray) else cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
   grid=cv2.resize(flow,(8,8),interpolation=cv2.INTER_AREA).reshape(-1)
   mag=np.linalg.norm(flow,axis=2)
   flows[i]=np.r_[grid,np.percentile(mag,[10,50,90,99]),flow.mean((0,1))].astype(np.float32)
  else:flows[i]=np.zeros(134,np.float32)
  if i in endpoints:images[i]=appearance(rgb)
  previous=gray;previous_i=i
 cap.release()
 values=[]
 for end in endpoints:
  history=[flows[i] for i in range(max(1,int(end)-14),int(end)+1)]
  if not history:history=[np.zeros(134,np.float32)]
  motion=np.r_[history[-1],np.mean(history[-5:],axis=0),np.mean(history,axis=0)]
  values.append(np.r_[images[int(end)],motion])
 result=np.asarray(values,dtype=np.float32)
 assert result.shape==(len(endpoints),IMAGE_DIM+MOTION_DIM) and np.isfinite(result).all()
 return result,n

def extract(limit=0):
 from concurrent.futures import ThreadPoolExecutor, as_completed
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['script_sha256']==sha(__file__)
 cache=ROOT/'cache';cache.mkdir(exist_ok=True)
 groups=[]
 for part in ('train','validation'):
  frame=pd.read_csv(ROOT/(part+'-samples.csv'))
  for sid,g in frame.groupby('ID'):groups.append((part,sid,g))
 if limit:groups=groups[:limit]
 def worker(task):
  part,sid,g=task
  target=cache/(sid+'.npz');info=cache/(sid+'.json');path=(DATA/g.video.iloc[0]).resolve()
  endpoints=sorted(g.endpoint.astype(int).tolist());video_hash=sha(path)
  assert video_hash==plan['original_bundle_hashes']['dataset/videos/'+sid+'.mp4']
  if target.exists() and info.exists():
   m=json.loads(info.read_text())
   if m['script_sha256']==sha(__file__):
    assert m['endpoints']==endpoints and m['features_sha256']==sha(target) and m['video_sha256']==video_hash
    return sid,len(endpoints),'cached'
  features,n=extract_video(path,endpoints)
  np.savez_compressed(target,features=features,endpoints=np.array(endpoints,dtype=np.int64))
  save(info,dict(ID=sid,part=part,endpoints=endpoints,video_sha256=video_hash,features_sha256=sha(target),script_sha256=sha(__file__),frames=n))
  return sid,len(endpoints),'extracted'
 start=time.monotonic()
 with ThreadPoolExecutor(max_workers=2) as executor:
  futures=[executor.submit(worker,task) for task in groups]
  for k,future in enumerate(as_completed(futures),1):
   sid,n,state=future.result()
   save(ROOT/'status.json',dict(status='EXTRACTING',videos_completed=k,videos_total=len(groups),last_ID=sid,seconds=time.monotonic()-start))
   print('features',k,'/',len(groups),sid,n,state,round(time.monotonic()-start,1),flush=True)
 save(ROOT/'status.json',dict(status='SMOKE_FEATURES_COMPLETE' if limit else 'FEATURES_COMPLETE',videos_total=len(groups)))

def array_for(frame):
 cache={};mappings={}
 for sid in frame.ID.unique():
  with np.load(ROOT/'cache'/(sid+'.npz')) as z:
   cache[sid]=z['features'];mappings[sid]={int(e):i for i,e in enumerate(z['endpoints'])}
 return np.stack([cache[r.ID][mappings[r.ID][int(r.endpoint)]] for r in frame.itertuples()])

class Control(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(IMAGE_DIM+MOTION_DIM,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU());self.accel=nn.Linear(64,4);self.steer=nn.Linear(64,3)
 def forward(self,x):
  z=self.net(x);return self.accel(z),self.steer(z)

def score(frame,a,s):
 ta=frame.accel.to_numpy();ts=frame.steer.to_numpy();moving=ta!=3
 return dict(accel=p.classification_metrics(ta,a,p.ACCEL),steer=p.classification_metrics(ts[moving],np.array(s)[moving],p.STEER))

def train():
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['script_sha256']==sha(__file__)
 frames={part:pd.read_csv(ROOT/(part+'-samples.csv')) for part in ('train','validation')}
 values={part:array_for(f) for part,f in frames.items()}
 mean=values['train'].mean(0);std=values['train'].std(0).clip(.01)
 normalized={part:torch.from_numpy(np.clip((x-mean)/std,-10,10)) for part,x in values.items()}
 ta=torch.tensor(frames['train'].accel.to_numpy());ts=torch.tensor(frames['train'].steer.to_numpy())
 results=[]
 for seed in SEEDS:
  for mode in MODES:
   out=ROOT/(str(seed)+'-'+mode);out.mkdir(exist_ok=True)
   if (out/'report.json').exists():
    r=json.loads((out/'report.json').read_text());assert r['status']=='COMPLETE';results.append(r);continue
   torch.manual_seed(seed);model=Control();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
   mask=torch.ones(IMAGE_DIM+MOTION_DIM)
   if mode=='image':mask[IMAGE_DIM:]=0
   if mode=='motion':mask[:IMAGE_DIM]=0
   x=normalized['train']*mask;history=[]
   start=time.monotonic()
   for epoch in range(1,13):
    losses=[];model.train()
    # Exact same original 10-group batches in every arm and seed.
    for ids in balanced_batches(frames['train'].to_dict('records'),epoch):
     a,s=model(x[ids]);moving=ta[ids]!=3
     loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving])
     opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
    history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
   model.eval();metrics={}
   for part in ('train','validation'):
    with torch.inference_mode():
     aa=[];ss=[]
     for chunk in normalized[part].split(256):
      a,s=model(chunk*mask);aa.append(a);ss.append(s)
     a=torch.cat(aa).numpy();s=torch.cat(ss).numpy()
    assert np.isfinite(a).all() and np.isfinite(s).all()
    np.savez_compressed(out/(part+'-logits.npz'),accel=a,steer=s)
    pred=frames[part][['ID','endpoint']].rename(columns={'endpoint':'sample_index'}).copy()
    pred['accel_label']=np.array(p.ACCEL)[a.argmax(1)];pred['steer_label']=np.array(p.STEER)[s.argmax(1)]
    pred.to_csv(out/(part+'-predictions.csv'),index=False)
    metrics[part]=score(frames[part],a.argmax(1),s.argmax(1))
   checkpoint=dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),mask=mask,mode=mode,seed=seed)
   torch.save(checkpoint,out/'model.pt')
   report=dict(status='COMPLETE',mode=mode,seed=seed,history=history,metrics=metrics,seconds=time.monotonic()-start,checkpoint_sha256=sha(out/'model.pt'),parameters=sum(v.numel() for v in model.parameters()))
   save(out/'report.json',report);results.append(report)
   print('trained',seed,mode,'train',metrics['train']['steer']['macro_f1'],'validation',metrics['validation']['steer']['macro_f1'],flush=True)
   save(ROOT/'status.json',dict(status='TRAINING_CONTROLS',completed=len(results),expected=9))
 save(ROOT/'results.json',dict(status='COMPLETE',runs=results))
 save(ROOT/'status.json',dict(status='CONTROLS_COMPLETE',completed=9,expected=9))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','extract','train']);parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
 torch.set_num_threads(2);cv2.setNumThreads(1)
 if args.command=='prepare':prepare()
 elif args.command=='extract':extract(args.limit)
 else:train()
