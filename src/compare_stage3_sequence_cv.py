"""Fixed15-flow temporal CNN with route CV and held-out order diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','threads;1')
import argparse,hashlib,json,shutil,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import cv2,numpy as np,pandas as pd,torch
from torch import nn
from torch.nn import functional as F
import compare_stage3_route_cv as cv
import compare_stage3_representations as base
ROOT=Path('artifacts/stage3-sequence-cv-20260911');REF=Path('artifacts/stage3-route-cv-20260911');SEEDS=[20260910,20260911,20260912]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
class SequenceModel(nn.Module):
 def __init__(self):
  super().__init__();self.temporal=nn.Sequential(nn.Conv1d(134,32,3,padding=1),nn.ReLU(),nn.Conv1d(32,32,3,padding=1),nn.ReLU());self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU());self.accel=nn.Linear(64,4);self.steer=nn.Linear(64,3)
 def forward(self,x):
  z=self.temporal(x.transpose(1,2));h=self.head(torch.cat([z[:,:,-1],z.mean(2)],1));return self.accel(h),self.steer(h)
def alter(x,mode):
 if mode=='ordered':return x.copy()
 if mode=='repeat_current':return np.repeat(x[:,-1:,:],15,axis=1)
 assert mode=='shuffle_past';rng=np.random.default_rng(20260911);out=x.copy()
 for i in range(len(x)):out[i,:14]=x[i,rng.permutation(14)]
 return out

def extract_flow(path,ends):
 need={i for end in ends for i in range(int(end)-15,int(end)+1)};assert min(need)>=0
 cap=cv2.VideoCapture(str(path));assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-10)<.01
 n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));flow=np.zeros((n,134),np.float32);previous=None;pi=-2
 try:
  for i in range(n):
   assert cap.grab()
   if i not in need:continue
   ok,bgr=cap.retrieve();assert ok
   rgb=cv2.resize(base.p.prepare_frame(bgr).transpose(1,2,0),(96,96),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
   if previous is not None and pi==i-1:
    f=np.zeros((96,96,2),np.float32) if np.array_equal(previous,gray) else cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
    flow[i]=np.r_[cv2.resize(f,(8,8),interpolation=cv2.INTER_AREA).reshape(-1),np.percentile(np.linalg.norm(f,axis=2),[10,50,90,99]),f.mean((0,1))].astype(np.float32)
   previous=gray;pi=i
 finally:cap.release()
 return np.stack([flow[int(t)-14:int(t)+1] for t in ends])

def prepare():
 ROOT.mkdir();(ROOT/'cache').mkdir();shutil.copyfile(REF/'samples.csv',ROOT/'samples.csv')
 v=json.loads((REF/'independent-validation.json').read_text());assert v['status']=='PASS' and sha(REF/'results.json')==v['results_sha256']
 m=SequenceModel();save('plan.json',dict(status='PREDECLARED',script_sha256=sha(__file__),samples_sha256=sha(ROOT/'samples.csv'),reference_validation_sha256=sha(REF/'independent-validation.json'),reference_results_sha256=sha(REF/'results.json'),seeds=SEEDS,folds=5,updates=1200,batch=10,lr=.001,weight_decay=0.,threads=2,extraction_workers=2,shape=[15,134],architecture='Conv1d134->32k3,ReLU,Conv32->32k3,ReLU,last+mean->Linear64->64,ReLU,heads4/3',parameters=sum(p.numel() for p in m.parameters()),baseline_parameters=224135,comparison='MLP summary vs smaller ordered-flow CNN; architecture and normalization differ, not an isolated proof of sequence benefit. Reuse15 verified baseline models.',normalization='Fold training clips and15past pairs only, per134channel, std floor.01 clip10',diagnostics=['ordered','shuffle_past','repeat_current'],diagnostic_scope='Fixed evaluation-only perturbations; past14 shuffled current pair preserved; not trained controls or validation-based selection',selection='Fixed final1200steps, all3seeds; no hyperparameter search',scope='Existing balanced1000clips/17routes5fold; excludes old4development routes; no human labels or hidden data',preserved_zip_sha256=sha('artifacts/stage3-motion-submit-candidate-20260910/submit.zip')))
 print('PREDECLARED',sum(p.numel() for p in m.parameters()),flush=True)

def extract():
 p=json.loads((ROOT/'plan.json').read_text());assert p['script_sha256']==sha(__file__);frame=pd.read_csv(ROOT/'samples.csv');tasks=list(frame.groupby('ID'));started=time.monotonic()
 def worker(task):
  sid,g=task;file=ROOT/'cache'/(sid+'.npz');info=file.with_suffix('.json');src=base.ROOT/'cache'/(sid+'.npz');sm=json.loads(src.with_suffix('.json').read_text());path=(base.DATA/g.video.iloc[0]).resolve();assert sha(src)==sm['features_sha256'] and sha(path)==sm['video_sha256']
  if file.exists() and info.exists():
   meta=json.loads(info.read_text());assert sha(file)==meta['sha256'] and meta['script_sha256']==sha(__file__);return sid
  with np.load(src) as z:ends=z['endpoints'];old=z['features'][:,1280:]
  assert ends.tolist()==sorted(g.endpoint.tolist());seq=extract_flow(path,ends);summary=np.concatenate([seq[:,-1],seq[:,-5:].mean(1),seq.mean(1)],1);np.testing.assert_array_equal(summary,old)
  np.savez_compressed(file,sequences=seq,endpoints=ends);info.write_text(json.dumps(dict(sha256=sha(file),source_cache_sha256=sha(src),video_sha256=sm['video_sha256'],script_sha256=sha(__file__))),encoding='utf-8');return sid
 with ThreadPoolExecutor(max_workers=2) as ex:
  for k,fu in enumerate(as_completed([ex.submit(worker,t) for t in tasks]),1):
   sid=fu.result();save('status.json',dict(status='EXTRACTING',completed=k,total=len(tasks),seconds=time.monotonic()-started));print('features',k,len(tasks),sid,flush=True)
 save('status.json',dict(status='FEATURES_COMPLETE',videos=len(tasks),seconds=time.monotonic()-started))
def arrays(f):
 data={}
 for sid in f.ID.unique():
  with np.load(ROOT/'cache'/(sid+'.npz')) as z:data[sid]={int(t):a for t,a in zip(z['endpoints'],z['sequences'])}
 return np.stack([data[r.ID][int(r.endpoint)] for r in f.itertuples()])
def train():
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['script_sha256']==sha(__file__);f=pd.read_csv(ROOT/'samples.csv');x=arrays(f);results=[];started=time.monotonic()
 for fold in range(5):
  ti=np.flatnonzero(f.fold.to_numpy()!=fold);vi=np.flatnonzero(f.fold.to_numpy()==fold);mean=x[ti].mean((0,1));std=x[ti].std((0,1)).clip(.01);xt=torch.from_numpy(np.clip((x[ti]-mean)/std,-10,10));batches=np.load(REF/f'fold-{fold}-batches.npy');ta=torch.tensor(f.iloc[ti].accel.to_numpy());ts=torch.tensor(f.iloc[ti].steer.to_numpy())
  for seed in SEEDS:
   out=ROOT/f'{fold}-{seed}';out.mkdir();torch.manual_seed(seed);model=SequenceModel();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
   for ids in batches:
    a,s=model(xt[ids]);moving=ta[ids]!=3;loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving]);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
   model.eval();metrics={}
   for mode in plan['diagnostics']:
    xv=torch.from_numpy(np.clip((alter(x[vi],mode)-mean)/std,-10,10))
    with torch.inference_mode():a,s=model(xv);a=a.numpy();s=s.numpy()
    np.savez_compressed(out/(mode+'.npz'),accel=a,steer=s,indices=vi);metrics[mode]=cv.score(f.iloc[vi],a.argmax(1),s.argmax(1))
   ck=dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),train_indices=ti.tolist(),validation_indices=vi.tolist(),fold=fold,seed=seed,steps=1200);torch.save(ck,out/'model.pt');r=dict(fold=fold,seed=seed,metrics=metrics,checkpoint_sha256=sha(out/'model.pt'));(out/'report.json').write_text(json.dumps(r,indent=2),encoding='utf-8');results.append(r);save('status.json',dict(status='TRAINING',completed=len(results),total=15,seconds=time.monotonic()-started));print('trained',len(results),fold,seed,metrics['ordered']['steer']['macro_f1'],flush=True)
 save('results.json',dict(status='TRAINING_COMPLETE',runs=results,seconds=time.monotonic()-started))
def verify():
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['script_sha256']==sha(__file__) and plan['samples_sha256']==sha(ROOT/'samples.csv')==sha(REF/'samples.csv');assert plan['reference_results_sha256']==sha(REF/'results.json') and plan['reference_validation_sha256']==sha(REF/'independent-validation.json')
 f=pd.read_csv(ROOT/'samples.csv');x=arrays(f)
 for sid,g in f.groupby('ID'):
  file=ROOT/'cache'/(sid+'.npz');meta=json.loads(file.with_suffix('.json').read_text());assert sha(file)==meta['sha256'] and meta['script_sha256']==sha(__file__) and sha(base.ROOT/'cache'/(sid+'.npz'))==meta['source_cache_sha256'] and sha((base.DATA/g.video.iloc[0]).resolve())==meta['video_sha256']
  with np.load(file) as z,np.load(base.ROOT/'cache'/(sid+'.npz')) as old:
   seq=z['sequences'];np.testing.assert_array_equal(np.concatenate([seq[:,-1],seq[:,-5:].mean(1),seq.mean(1)],1),old['features'][:,1280:])
 # Separate full decode for one source checks actual15pair history, including sparse boundaries.
 import stage3_temporal_motion_experiment as e
 g=next(iter(f.groupby('ID')))[1];dense=e.dense_train((base.DATA/g.video.iloc[0]).resolve())
 for idx in g.index:np.testing.assert_array_equal(x[idx],dense[int(f.loc[idx,'endpoint'])-14:int(f.loc[idx,'endpoint'])+1])
 result=json.loads((ROOT/'results.json').read_text());assert len(result['runs'])==15;oof={(s,m):np.full((len(f),2),-1,int) for s in SEEDS for m in plan['diagnostics']}
 for r in result['runs']:
  fold,seed=r['fold'],r['seed'];out=ROOT/f'{fold}-{seed}';assert sha(out/'model.pt')==r['checkpoint_sha256'];ck=torch.load(out/'model.pt',weights_only=True);ti=np.array(ck['train_indices']);vi=np.array(ck['validation_indices']);assert np.array_equal(ti,np.flatnonzero(f.fold.to_numpy()!=fold)) and np.array_equal(vi,np.flatnonzero(f.fold.to_numpy()==fold)) and set(f.iloc[ti].route).isdisjoint(f.iloc[vi].route)
  mean=x[ti].mean((0,1));std=x[ti].std((0,1)).clip(.01);np.testing.assert_array_equal(ck['mean'],mean);np.testing.assert_array_equal(ck['std'],std);model=SequenceModel();model.load_state_dict(ck['model'],strict=True);model.eval()
  for mode in plan['diagnostics']:
   v=torch.from_numpy(np.clip((alter(x[vi],mode)-mean)/std,-10,10))
   with torch.inference_mode():a,s=model(v);a=a.numpy();s=s.numpy()
   with np.load(out/(mode+'.npz')) as z:np.testing.assert_array_equal(a,z['accel']);np.testing.assert_array_equal(s,z['steer']);np.testing.assert_array_equal(vi,z['indices'])
   assert cv.score(f.iloc[vi],a.argmax(1),s.argmax(1))==r['metrics'][mode];oof[seed,mode][vi]=np.stack([a.argmax(1),s.argmax(1)],1)
 summary=[];routes=[]
 for (seed,mode),pred in oof.items():
  assert np.all(pred>=0);summary.append(dict(seed=seed,mode=mode,metrics=cv.score(f,pred[:,0],pred[:,1])));out=f[['ID','endpoint','route','fold']].copy();out['accel_label']=np.array(base.p.ACCEL)[pred[:,0]];out['steer_label']=np.array(base.p.STEER)[pred[:,1]];out.to_csv(ROOT/f'oof-{seed}-{mode}.csv',index=False)
  for route,g in f.groupby('route'):
   m=cv.score(g,pred[g.index,0],pred[g.index,1]);routes.append(dict(seed=seed,mode=mode,route=route,steer_f1=m['steer']['macro_f1'],recall=m['steer']['recall']))
 baseline=[r for r in json.loads((REF/'summary.json').read_text())['pooled_oof'] if r['arm']=='baseline'];save('summary.json',dict(sequence=summary,baseline=baseline,routes=routes))
 assert plan['preserved_zip_sha256']==sha('artifacts/stage3-motion-submit-candidate-20260910/submit.zip')
 save('independent-validation.json',dict(status='PASS',models=15,reused_baseline_models=15,source_videos=int(f.ID.nunique()),sequence_shape=list(x.shape),max_logits_error=0,results_sha256=sha(ROOT/'results.json'),summary_sha256=sha(ROOT/'summary.json'),submission_unchanged=True));save('status.json',dict(status='COMPLETE_VALIDATED'));print('PASS all checks',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','extract','train','verify']);args=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1);globals()[args.command]()
