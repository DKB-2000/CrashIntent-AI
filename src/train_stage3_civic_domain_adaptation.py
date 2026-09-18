"""Steer-head-only Civic proxy adaptation with frozen-feature prediction preservation."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from torch.nn import functional as F
import compare_stage3_representations as base
import stage3_motion_inference as motion
from evaluate_stage3_civic_535_balanced import labels

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/stage3-civic-domain-adaptation-20260917'
RAW=P/'data_raw/comma2k19/civic-535-training-20260917'
ACQ=P/'artifacts/stage3-civic-535-training-acquisition-20260917'
SOURCE=P/'artifacts/stage3-prediction-preservation-20260914/warmstart_ce'
OLD=P/'artifacts/stage3-representation-20260910'
FEATURE=P/'artifacts/stage3-right-recovery-20260914'
HELD=P/'artifacts/stage3-civic-535-balanced-evaluation-20260917'
SEEDS=[20260910,20260911,20260912];CLASSES=['LEFT','STRAIGHT','RIGHT']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,o):
 ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def steer_metric(truth,pred):
 truth=np.asarray(truth);pred=np.asarray(pred);cm=np.array([[int(np.sum((truth==i)&(pred==j))) for j in range(3)] for i in range(3)]);sup=cm.sum(1);den=cm.sum(0)+sup;f=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0);rec=np.divide(np.diag(cm),sup,out=np.zeros(3),where=sup>0);return dict(rows=len(truth),f1=float(f.mean()),recall=rec.tolist(),confusion=cm.tolist())
def prepare():
 ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'plan.json').exists();ap=json.loads((ACQ/'plan.json').read_text());av=ACQ/'validation.json';assert json.loads(av.read_text())['status']=='PASS'
 source={str(s):sha(SOURCE/str(s)/'model.pt') for s in SEEDS}
 plan=dict(status='FROZEN_BEFORE_LABELS',seeds=SEEDS,offset_deg=.011351006928732928,rules='Frozen speed5-35/yaw-angle/centered11 proxy',architecture='Freeze shared net and accel head; update steer head only',updates=300,lr=.00005,batch_civic=48,batch_anchor=48,temperature=2.,loss='Civic balanced CE + original1000 teacher KL*T^2',gate='Every seed: old steer F1 >= baseline-0.002 and heldout Civic F1 improves; ensemble improves both; no class recall loss >2pp.',source_sha256=source,acquisition_validation_sha256=sha(av),script_sha256=sha(__file__),heldout_route=json.loads((ACQ/'plan.json').read_text())['held_out_evaluation_route'])
 save('plan.json',plan);save('status.json',dict(status='PREPARED',completed=0,total=3));print(json.dumps(plan,indent=2))
def extract(plan,ap,av):
 frames=[];features=[]
 for i,seg in enumerate(ap['segments']):
  sid=f'CIVIC_TRAIN_{i:03}';folder=RAW/sid;route=seg.rsplit('/',1)[0];frame=labels(sid,folder,route,plan['offset_deg']);frames.append(frame)
  cache=ROOT/f'{sid}-features.npz'
  if cache.exists():
   with np.load(cache,allow_pickle=False) as z:f=z['features']
  else:
   raw=folder/'video.hevc';source=next(x for x in av['files'] if x['ID']==sid and x['relative']=='video.hevc');assert sha(raw)==source['sha256'];avi=ROOT/f'{sid}.avi';cap=cv2.VideoCapture(str(raw));writer=None;n=k=0
   while True:
    ok,bgr=cap.read()
    if not ok:break
    if n%2==0:
     if writer is None:
      h,w=bgr.shape[:2];writer=cv2.VideoWriter(str(avi),cv2.VideoWriter_fourcc(*'FFV1'),10,(w,h));assert writer.isOpened()
     writer.write(bgr);k+=1
    n+=1
   cap.release();writer.release();assert k==len(frame);f=motion.s3_motion_features(avi);np.savez_compressed(cache,features=f)
  assert len(f)==len(frame);features.append(f);save('status.json',dict(status='EXTRACTING',completed=i+1,total=6,pid=os.getpid()))
 allframe=pd.concat(frames,ignore_index=True);allfeat=np.concatenate(features);eligible=allframe.valid.to_numpy();counts=allframe[allframe.valid].truth.value_counts().to_dict();assert len(allframe)==len(allfeat) and all(counts.get(c,0)>=20 for c in CLASSES)
 allframe.to_csv(ROOT/'training-proxy-labels.csv',index=False);np.savez_compressed(ROOT/'training-features.npz',features=allfeat,eligible=eligible);return allframe,allfeat,eligible,counts
def run():
 torch.set_num_threads(1);cv2.setNumThreads(1);plan=json.loads((ROOT/'plan.json').read_text());assert sha(__file__)==plan['script_sha256'];assert sha(ACQ/'validation.json')==plan['acquisition_validation_sha256'];ap=json.loads((ACQ/'plan.json').read_text());av=json.loads((ACQ/'validation.json').read_text());frame,feat,eligible,counts=extract(plan,ap,av)
 y=np.array([CLASSES.index(v) if v in CLASSES else -1 for v in frame.truth]);civic=np.zeros((len(feat),1682),np.float32);civic[:,1280:]=feat
 repeat=pd.read_csv(P/'artifacts/stage3-prediction-preservation-20260914/repeat-samples.csv');valid=pd.read_csv(OLD/'validation-samples.csv')
 with np.load(FEATURE/'training-features.npz',allow_pickle=False) as z:anchor=z['baseline']
 base.ROOT=OLD;xv=base.array_for(valid);held=pd.read_csv(HELD/'proxy-labels.csv');held_features=[]
 for path in sorted(HELD.glob('*-features.npz')):
  with np.load(path,allow_pickle=False) as z:held_features.append(z['features'])
 hf=np.concatenate(held_features);hx=np.zeros((len(hf),1682),np.float32);hx[:,1280:]=hf;hv=held.valid.to_numpy();hy=np.array([CLASSES.index(v) if v in CLASSES else -1 for v in held.truth]);held_eval=held.copy();held_eval['steer']=hy
 models=[];results=[];rng=np.random.default_rng(20260917);groups=[np.where(eligible&(y==i))[0] for i in range(3)]
 for si,seed in enumerate(SEEDS):
  ck=torch.load(SOURCE/str(seed)/'model.pt',map_location='cpu',weights_only=True);assert sha(SOURCE/str(seed)/'model.pt')==plan['source_sha256'][str(seed)];m=base.Control();m.load_state_dict(ck['model'],strict=True);teacher=base.Control();teacher.load_state_dict(ck['model'],strict=True);teacher.eval()
  for p in m.parameters():p.requires_grad_(False)
  for p in m.steer.parameters():p.requires_grad_(True)
  mean,std,mask=(ck[k] for k in ('mean','std','mask'));norm=lambda x:torch.from_numpy(np.clip((x-mean.numpy())/std.numpy(),-10,10))*mask
  xc,xa,xvalid,xheld=map(norm,(civic,anchor,xv,hx));moving=valid.accel.to_numpy()!=3
  with torch.inference_mode():_,teacher_anchor=teacher(xa);_,bs=teacher(xvalid);_,bh=teacher(xheld)
  opt=torch.optim.AdamW(m.steer.parameters(),lr=plan['lr'],weight_decay=0.)
  for step in range(plan['updates']):
   ids=np.concatenate([rng.choice(g,16,replace=True) for g in groups]);aids=rng.choice(len(anchor),plan['batch_anchor'],replace=True);_,s=m(xc[ids]);_,sa=m(xa[aids]);loss=F.cross_entropy(s,torch.tensor(y[ids]))+F.kl_div(F.log_softmax(sa/2,1),F.softmax(teacher_anchor[aids]/2,1),reduction='batchmean')*4;opt.zero_grad();loss.backward();opt.step()
  with torch.inference_mode():_,vs=m(xvalid);_,hs=m(xheld)
  old_base=steer_metric(valid.steer.to_numpy()[moving],bs.numpy()[moving].argmax(1));old_new=steer_metric(valid.steer.to_numpy()[moving],vs.numpy()[moving].argmax(1));held_base=steer_metric(hy[hv],bh.numpy()[hv].argmax(1));held_new=steer_metric(hy[hv],hs.numpy()[hv].argmax(1))
  out=ROOT/str(seed);out.mkdir(exist_ok=True);torch.save(dict(model=m.state_dict(),mean=mean,std=std,mask=mask,seed=seed,mode='motion',arm='civic_domain_adapt'),out/'model.pt');np.savez_compressed(out/'logits.npz',old=vs.numpy(),held=hs.numpy());results.append(dict(seed=seed,old_base=old_base,old_new=old_new,held_base=held_base,held_new=held_new));models.append((vs.numpy(),hs.numpy()));save('status.json',dict(status='TRAINING',completed=si+1,total=3,pid=os.getpid()))
 old_ens=np.mean([x[0] for x in models],0);held_ens=np.mean([x[1] for x in models],0)
 # Compare adapted ensemble against the already validated source ensemble.
 old_src=np.mean([np.load(SOURCE/str(s)/'validation-logits.npz')['steer'] for s in SEEDS],0);held_src=[]
 for seed in SEEDS:
  ck=torch.load(SOURCE/str(seed)/'model.pt',map_location='cpu',weights_only=True);m=base.Control();m.load_state_dict(ck['model']);m.eval()
  with torch.inference_mode():_,z=m(norm(hx));held_src.append(z.numpy())
 old_source=steer_metric(valid.steer.to_numpy()[moving],old_src[moving].argmax(1));old_adapt=steer_metric(valid.steer.to_numpy()[moving],old_ens[moving].argmax(1));held_source=steer_metric(hy[hv],np.mean(held_src,0)[hv].argmax(1));held_adapt=steer_metric(hy[hv],held_ens[hv].argmax(1))
 per_seed=all(r['old_new']['f1']>=r['old_base']['f1']-.002 and r['held_new']['f1']>r['held_base']['f1'] for r in results);recall=min(np.array(old_adapt['recall'])-np.array(old_source['recall']))>=-.02 and min(np.array(held_adapt['recall'])-np.array(held_source['recall']))>=-.02;passed=per_seed and old_adapt['f1']>old_source['f1'] and held_adapt['f1']>held_source['f1'] and recall
 result=dict(status='COMPLETE_VALIDATED',decision='FOLLOWUP_ONLY' if passed else 'KEEP_ENSEMBLE',training_counts=counts,per_seed=results,ensemble=dict(old_source=old_source,old_adapt=old_adapt,held_source=held_source,held_adapt=held_adapt),gates=dict(per_seed=per_seed,recall=bool(recall),passed=bool(passed)),adopted=False);save('report.json',result);save('status.json',dict(status='COMPLETE_VALIDATED',completed=3,total=3,decision=result['decision']));print(json.dumps(result,indent=2))
def watch():
 import msvcrt
 with (ROOT/'watch.lock').open('a+b') as lock:
  lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  try:
   with (ROOT/'run.log').open('w',encoding='utf-8') as log:
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=P,stdout=log,stderr=subprocess.STDOUT);save('watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=p.pid,timeout_seconds=3600));rc=p.wait(timeout=3600)
    if rc:raise RuntimeError(f'Worker exit {rc}; see run.log')
   save('watch-status.json',dict(status='COMPLETE'))
  except Exception as e:save('status.json',dict(status='FAILED',error=str(e)));save('watch-status.json',dict(status='FAILED',error=str(e)));raise
if __name__=='__main__':{'prepare':prepare,'run':run,'watch':watch}[sys.argv[1]]()
