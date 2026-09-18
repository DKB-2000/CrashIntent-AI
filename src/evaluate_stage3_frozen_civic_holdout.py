"""One-shot frozen 5-35m/s proxy evaluation of four Stage3 model families."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
import stage3_motion_inference as motion
import compare_stage3_representations as base
from prepare_stage3_comma2k19 import _deduplicate_times,_smooth

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/stage3-frozen-civic-holdout-20260916'
ACQ=P/'artifacts/stage3-civic-low-speed-acquisition-20260916'
RAW=P/'data_raw/comma2k19/civic-low-speed-20260916'
CLASSES=['LEFT','STRAIGHT','RIGHT']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,o):
 p=ROOT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def metric(g):
 cm=np.array([[int(((g.truth==a)&(g.pred==b)).sum()) for b in CLASSES] for a in CLASSES]);den=cm.sum(0)+cm.sum(1);sup=cm.sum(1)
 f=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0)
 return dict(rows=len(g),f1=float(f.mean()),recall=[float(cm[i,i]/sup[i]) if sup[i] else None for i in range(3)],confusion=cm.tolist())
def labels(sid,folder,route,offset):
 def load(n):return np.asarray(np.load(folder/n,allow_pickle=False))
 ft=load('global_pose/frame_times').reshape(-1);t=ft[::2];speedkey='speed' if (folder/'processed_log/CAN/speed').exists() else 'car_speed'
 st,sv=_deduplicate_times('speed',load(f'processed_log/CAN/{speedkey}/t').reshape(-1),load(f'processed_log/CAN/{speedkey}/value').reshape(-1));at,av=_deduplicate_times('angle',load('processed_log/CAN/steering_angle/t').reshape(-1),load('processed_log/CAN/steering_angle/value').reshape(-1))
 speed=np.interp(t,st,sv);angle=np.interp(t,at,av)-offset;pos=load('global_pose/frame_positions');vel=load('global_pose/frame_velocities');up=pos/np.linalg.norm(pos,axis=1,keepdims=True);h=vel-(vel*up).sum(1,keepdims=True)*up;h=np.column_stack([_smooth(h[:,j],21) for j in range(3)]);yaw=np.sum(np.cross(h,np.gradient(h,ft,axis=0))*up,1)/np.maximum(np.sum(h*h,axis=1),1.);yaw=np.interp(t,ft,_smooth(yaw,21));valid=(t>=max(st[0],at[0]))&(t<=min(st[-1],at[-1]))&(t>=ft[0]+1)&(t<=ft[-1]-1);domain=valid&(speed>=5)&(speed<=35)&(abs(yaw)<.12)
 pre=np.full(len(t),'UNKNOWN',dtype='<U8');pre[domain&(abs(angle)<=.5)&(abs(yaw)<.003)]='STRAIGHT';pre[domain&(angle>1.5)&(yaw>.012)]='LEFT';pre[domain&(angle< -1.5)&(yaw<-.012)]='RIGHT';keep=np.zeros(len(t),bool)
 for j in range(5,len(t)-5):keep[j]=pre[j]!='UNKNOWN' and np.all(pre[j-5:j+6]==pre[j])
 return pd.DataFrame(dict(ID=sid,sample_index=np.arange(len(t)),route=route,truth=np.where(keep,pre,'UNKNOWN'),valid=keep,speed=speed,angle=angle,yaw=yaw))
def run():
 torch.set_num_threads(1);cv2.setNumThreads(1);plan=json.loads((ROOT/'plan.json').read_text());assert sha(__file__)==plan['script_sha256'];acq=json.loads((ACQ/'validation.json').read_text());ap=json.loads((ACQ/'plan.json').read_text());assert acq['status']=='PASS' and sha(ACQ/'validation.json')==plan['acquisition_validation_sha256']
 evaluation={r for r,s in ap['route_splits'].items() if s=='evaluation'};assert len(evaluation)==2;assert set(plan['evaluation_routes'])==evaluation
 frames=[];videos=[]
 for i,seg in enumerate(ap['segments']):
  route=seg.rsplit('/',1)[0]
  if route not in evaluation:continue
  sid=f'CIVIC_LOW_{i:03}';folder=RAW/sid;frames.append(labels(sid,folder,route,plan['offset_deg']))
  source=next(f for f in acq['files'] if f['ID']==sid and f['relative']=='video.hevc');raw=P/source['local'];assert sha(raw)==source['sha256'];target=ROOT/f'{sid}.avi';cap=cv2.VideoCapture(str(raw));writer=None;n=0;k=0
  while True:
   ok,bgr=cap.read()
   if not ok:break
   if n%2==0:
    if writer is None:
     h,w=bgr.shape[:2];writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'FFV1'),10,(w,h));assert writer.isOpened()
    writer.write(bgr);k+=1
   n+=1
  cap.release();writer.release();assert k==len(frames[-1]);features=motion.s3_motion_features(target);np.savez_compressed(ROOT/f'{sid}-features.npz',features=features);videos.append((sid,features))
 truth=pd.concat(frames,ignore_index=True);truth.to_csv(ROOT/'proxy-labels.csv',index=False);eligible=truth[truth.valid];counts=eligible.truth.value_counts().to_dict();assert len(eligible)>=plan['minimum_rows'] and all(counts.get(c,0)>=plan['minimum_per_class'] for c in CLASSES)
 rows=[];metrics=[]
 for arm,pattern in plan['models'].items():
  for seed in plan['seeds']:
   path=P/pattern.format(seed=seed);assert sha(path)==plan['model_sha256'][arm][str(seed)];ck=torch.load(path,map_location='cpu',weights_only=True);model=base.Control();model.load_state_dict(ck['model'],strict=True);model.eval();mean=ck['mean'].numpy();std=ck['std'].numpy();mask=ck['mask'];assert np.all(mask.numpy()[:1280]==0) and np.all(mask.numpy()[1280:]==1)
   parts=[]
   with torch.inference_mode():
    for sid,feat in videos:
     x=np.zeros((len(feat),1682),np.float32);x[:,1280:]=np.clip((feat-mean[1280:])/std[1280:],-10,10);_,s=model(torch.from_numpy(x));parts.append(pd.DataFrame(dict(ID=sid,sample_index=np.arange(len(feat)),pred=np.array(CLASSES)[s.argmax(1).numpy()])))
   pred=pd.concat(parts,ignore_index=True);joined=truth.merge(pred,on=['ID','sample_index'],validate='one_to_one');joined['arm']=arm;joined['seed']=seed;rows.append(joined);g=joined[joined.valid];metrics.append(dict(arm=arm,seed=seed,route='ALL',**metric(g)))
   for route,q in g.groupby('route'):metrics.append(dict(arm=arm,seed=seed,route=route,**metric(q)))
 allrows=pd.concat(rows,ignore_index=True);allrows.to_csv(ROOT/'predictions.csv',index=False);m=pd.DataFrame(metrics);m.to_json(ROOT/'metrics.json',orient='records',indent=2)
 avg={a:float(m[(m.arm==a)&(m.route=='ALL')].f1.mean()) for a in plan['models']};baseavg=avg['original'];gates={}
 for arm in plan['models']:
  if arm=='original':continue
  pooled=m[(m.arm==arm)&(m.route=='ALL')];orig=m[(m.arm=='original')&(m.route=='ALL')];deltas=[float(pooled[pooled.seed==s].f1.iloc[0]-orig[orig.seed==s].f1.iloc[0]) for s in plan['seeds']];rd=[]
  for route in sorted(evaluation):rd.append(float(m[(m.arm==arm)&(m.route==route)].f1.mean()-m[(m.arm=='original')&(m.route==route)].f1.mean()))
  rec=np.mean(np.array(pooled.recall.tolist()),0)-np.mean(np.array(orig.recall.tolist()),0);gates[arm]=dict(seed_deltas=deltas,route_deltas=rd,recall_delta=rec.tolist(),passed=bool(all(x>0 for x in deltas) and all(x>0 for x in rd) and min(rec)>=-.02))
 result=dict(status='COMPLETE_VALIDATED',decision='FOLLOWUP_ONLY' if any(x['passed'] for x in gates.values()) else 'KEEP_BASELINE',total=len(truth),scored=len(eligible),class_counts=counts,averages=avg,gates=gates,adopted=False,validation='Frozen rules/models before labels; acquisition hashes; even-frame lossless10Hz conversion; shared features; 12 checkpoint hashes; route/class metrics; no threshold tuning.',outputs_sha256={n:sha(ROOT/n) for n in ['proxy-labels.csv','predictions.csv','metrics.json']});save('final-review.json',result);save('status.json',dict(status='COMPLETE_VALIDATED',completed=4,total=4,scored=len(eligible),decision=result['decision']));print(json.dumps(result,indent=2))
def watch():
 import msvcrt
 with (ROOT/'watch.lock').open('a+b') as lock:
  lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);assert not (ROOT/'execution.json').exists();save('execution.json',dict(status='STARTED',pid=os.getpid()))
  try:
   with (ROOT/'run.log').open('w',encoding='utf-8') as log:
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=P,stdout=log,stderr=subprocess.STDOUT);save('watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=p.pid));rc=p.wait(timeout=1800)
    if rc:raise RuntimeError(f'Worker exit {rc}')
   save('watch-status.json',dict(status='COMPLETE'))
  except Exception as e:save('status.json',dict(status='FAILED',error=str(e)));save('watch-status.json',dict(status='FAILED',error=str(e)));raise
if __name__=='__main__':{'run':run,'watch':watch}[sys.argv[1]]()
