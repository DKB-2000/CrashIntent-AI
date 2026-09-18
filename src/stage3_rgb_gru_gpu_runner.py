"""Frozen ImageNet RGB features plus causal GRU Stage3 trial on full comma routes."""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

ACCEL=['ACCELERATING','DECELERATING','CONSTANT','STOPPED']; STEER=['LEFT','STRAIGHT','RIGHT']; SEED=20260918
def save(p,x): p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,indent=2),encoding='utf-8'); os.replace(q,p)
def metric(y,p,n):
 cm=np.array([[int(np.sum((y==i)&(p==j))) for j in range(n)] for i in range(n)]); den=cm.sum(0)+cm.sum(1); f=np.divide(2*np.diag(cm),den,out=np.zeros(n),where=den>0)
 return {'rows':len(y),'macro_f1':float(f.mean()),'per_class_f1':f.tolist(),'confusion':cm.tolist()}
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.gru=nn.GRU(576,192,2,batch_first=True,dropout=.2); self.accel=nn.Linear(192,4); self.steer=nn.Linear(192,3)
 def forward(self,x,h=None): z,h=self.gru(x,h); return self.accel(z),self.steer(z),h
def decode(path,encoder,transform,device,batch=96):
 cap=cv2.VideoCapture(str(path)); frames=[]; spatial=[]; motion=[]; previous=None
 def flush():
  if not frames:return
  tensors=[]
  for frame in frames:
   rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   image=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)
   tensors.append(transform(image))
  x=torch.stack(tensors).to(device)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): spatial.append(encoder(x).float().cpu().numpy())
  frames.clear()
 while True:
  ok,bgr=cap.read()
  if not ok:break
  gray=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)/255
  diff=np.zeros_like(gray) if previous is None else gray-previous; motion.append(cv2.resize(diff,(8,8),interpolation=cv2.INTER_AREA).reshape(-1)); previous=gray
  frames.append(bgr)
  if len(frames)==batch:flush()
 flush();cap.release(); return np.c_[np.concatenate(spatial).astype(np.float32),np.asarray(motion,np.float32)]
def evaluate(model,items,device):
 ya=[];ys=[];pa=[];ps=[]
 model.eval()
 with torch.inference_mode():
  for x,a,s in items:
   al,sl,_=model(torch.from_numpy(x)[None].to(device));ya.extend(a);ys.extend(s);pa.extend(al[0].argmax(1).cpu().numpy());ps.extend(sl[0].argmax(1).cpu().numpy())
 ya=np.asarray(ya);ys=np.asarray(ys);pa=np.asarray(pa);ps=np.asarray(ps);moving=ya!=3
 return {'accel':metric(ya,pa,4),'moving_steer':metric(ys[moving],ps[moving],3),'joint_mean':float((metric(ya,pa,4)['macro_f1']+metric(ys[moving],ps[moving],3)['macro_f1'])/2)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--max-seconds',type=int,default=18000);args=ap.parse_args()
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2);cv2.setNumThreads(1);device=torch.device('cuda');out=args.output;out.mkdir(parents=True,exist_ok=True);started=time.monotonic();save(out/'status.json',{'status':'EXTRACTING'})
 manifest=pd.read_csv(args.dataset/'split_manifest.csv'); labels={p:pd.read_csv(args.dataset/f'labels_{p}_candidate.csv') for p in ('train','validation')}; amap={x:i for i,x in enumerate(ACCEL)};smap={x:i for i,x in enumerate(STEER)}
 weights=ResNet18_Weights.IMAGENET1K_V1; encoder=resnet18(weights=weights);encoder.fc=nn.Identity();encoder.to(device).eval();transform=weights.transforms();cache=out/'cache';cache.mkdir()
 data={p:[] for p in ('train','validation')}
 for number,row in enumerate(manifest.itertuples(index=False),1):
  split='validation' if row.split=='validation' else 'train'; g=labels[split][labels[split].ID==row.ID].sort_values('frame_index'); path=args.dataset/'videos'/f'{row.ID}.mp4'; x=decode(path,encoder,transform,device);assert len(x)==len(g)
  a=g.accel_label.map(amap).to_numpy(np.int64);s=g.steer_label.map(smap).to_numpy(np.int64);np.savez_compressed(cache/f'{row.ID}.npz',x=x,a=a,s=s);data[split].append((x,a,s))
  if number%10==0:save(out/'status.json',{'status':'EXTRACTING','completed':number,'total':len(manifest)})
 del encoder;torch.cuda.empty_cache();model=Net().to(device)
 ca=np.bincount(np.concatenate([x[1] for x in data['train']]),minlength=4);cs=np.bincount(np.concatenate([x[2] for x in data['train']]),minlength=3);wa=torch.tensor(np.sqrt(ca.sum()/np.maximum(ca,1)),device=device,dtype=torch.float32);wa/=wa.mean();ws=torch.tensor(np.sqrt(cs.sum()/np.maximum(cs,1)),device=device,dtype=torch.float32);ws/=ws.mean()
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda');best=-1.;history=[]
 windows=[(x[i:i+64],a[i:i+64],s[i:i+64]) for x,a,s in data['train'] for i in range(0,len(x),64) if len(x[i:i+64])>=16]
 for epoch in range(1,args.epochs+1):
  if time.monotonic()-started>args.max_seconds:break
  random.Random(SEED+epoch).shuffle(windows);model.train();losses=[]
  for x,a,s in windows:
   xt=torch.from_numpy(x)[None].to(device);at=torch.from_numpy(a)[None].to(device);st=torch.from_numpy(s)[None].to(device);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16): al,sl,_=model(xt);loss=F.cross_entropy(al.flatten(0,1),at.flatten(),weight=wa)+F.cross_entropy(sl.flatten(0,1),st.flatten(),weight=ws)
   scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();losses.append(float(loss))
  scores=evaluate(model,data['validation'],device);item={'epoch':epoch,'loss':float(np.mean(losses)),'scores':scores};history.append(item);save(out/'status.json',{'status':'TRAINING','epoch':epoch,'epochs':args.epochs,'best':max(best,scores['joint_mean'])})
  if scores['joint_mean']>best:best=scores['joint_mean'];torch.save({'format':'stage3-rgb-gru-v1','model':model.state_dict(),'accel_classes':ACCEL,'steer_classes':STEER,'epoch':epoch,'scores':scores},out/'best.pt')
  save(out/'history.json',history)
 report={'status':'COMPLETE_VALIDATED' if history else 'FAILED','architecture':'frozen ImageNet ResNet18 RGB + 8x8 frame difference + causal 2-layer GRU','train_videos':len(data['train']),'validation_videos':len(data['validation']),'train_rows':int(sum(len(x[0]) for x in data['train'])),'validation_rows':int(sum(len(x[0]) for x in data['validation'])),'epochs_completed':len(history),'best_joint_mean':best,'history':history,'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'seconds':time.monotonic()-started}
 save(out/'report.json',report);save(out/'status.json',{'status':report['status'],'epochs_completed':len(history),'best_joint_mean':best});print(json.dumps(report,indent=2))
if __name__=='__main__':main()
