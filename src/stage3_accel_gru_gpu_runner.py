"""Train a route-robust accel-only GRU ensemble from cached RGB+motion features."""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.nn import functional as F

CLASSES=['ACCELERATING','DECELERATING','CONSTANT','STOPPED'];SEEDS=(20260918,20260919,20260920)
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def metric(y,p):
 cm=np.array([[int(np.sum((y==i)&(p==j))) for j in range(4)] for i in range(4)]);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(4),where=den>0)
 return {'rows':len(y),'macro_f1':float(f.mean()),'per_class_f1':dict(zip(CLASSES,map(float,f))),'confusion':cm.tolist()}
class Net(nn.Module):
 def __init__(self):super().__init__();self.gru=nn.GRU(576,192,2,batch_first=True,dropout=.2);self.head=nn.Linear(192,4)
 def forward(self,x):z,_=self.gru(x);return self.head(z)
def evaluate(model,items,device):
 ys=[];ls=[];routes=[];model.eval()
 with torch.inference_mode():
  for x,y,r in items:ls.append(model(torch.from_numpy(x)[None].to(device))[0].cpu().numpy());ys.append(y);routes.extend([r]*len(y))
 y=np.concatenate(ys);logits=np.concatenate(ls);route=np.asarray(routes);overall=metric(y,logits.argmax(1));per={r:metric(y[route==r],logits[route==r].argmax(1)) for r in sorted(set(routes))}
 scores=np.array([v['macro_f1'] for v in per.values()]);return logits,y,{'overall':overall,'routes':per,'robust_score':float(scores.mean()-.25*scores.std()),'route_mean':float(scores.mean()),'route_std':float(scores.std())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=12);args=ap.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.set_num_threads(2);device=torch.device('cuda');manifest=pd.read_csv(args.dataset/'split_manifest.csv');items={'train':[],'validation':[]}
 for row in manifest.itertuples(index=False):
  with np.load(args.cache/f'{row.ID}.npz',allow_pickle=False) as z:x=z['x'].astype(np.float32);y=z['a'].astype(np.int64)
  items[row.split].append((x,y,row.route))
 assert sum(len(x) for x,_,_ in items['train'])==93602 and sum(len(x) for x,_,_ in items['validation'])==18603
 counts=np.bincount(np.concatenate([y for _,y,_ in items['train']]),minlength=4);weight=torch.tensor(np.sqrt(counts.sum()/counts),device=device,dtype=torch.float32);weight/=weight.mean();windows=[(x[i:i+64],y[i:i+64]) for x,y,_ in items['train'] for i in range(0,len(x),64) if len(x[i:i+64])>=16]
 members=[];histories=[];validation_logits=[]
 for seed in SEEDS:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=Net().to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda');best=(-1,None,None);history=[]
  for epoch in range(1,args.epochs+1):
   random.Random(seed+epoch).shuffle(windows);model.train();losses=[]
   for x,y in windows:
    xt=torch.from_numpy(x)[None].to(device);yt=torch.from_numpy(y)[None].to(device);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):logits=model(xt);loss=F.cross_entropy(logits.flatten(0,1),yt.flatten(),weight=weight,label_smoothing=.02)
    scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();losses.append(float(loss))
   logits,y,scores=evaluate(model,items['validation'],device);record={'epoch':epoch,'loss':float(np.mean(losses)),'scores':scores};history.append(record)
   if scores['robust_score']>best[0]:best=(scores['robust_score'],{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},record)
   save(out/'status.json',{'status':'TRAINING','seed':seed,'epoch':epoch,'completed_members':len(members),'total_members':len(SEEDS)})
  model.load_state_dict(best[1]);logits,y,scores=evaluate(model,items['validation'],device);members.append(best[1]);validation_logits.append(logits);histories.append({'seed':seed,'selected':best[2],'history':history})
 ensemble=np.mean(validation_logits,axis=0);ensemble_metrics=metric(y,ensemble.argmax(1));checkpoint={'format':'stage3-accel-rgb-gru-ensemble-v1','models':members,'classes':CLASSES,'seeds':list(SEEDS),'input_features':576,'selection':'route_mean_minus_0.25_std'};torch.save(checkpoint,out/'best.pt')
 report={'status':'COMPLETE_VALIDATED','train_rows':93602,'validation_rows':18603,'members':histories,'ensemble':ensemble_metrics,'seconds':time.monotonic()-started,'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()};save(out/'report.json',report);save(out/'history.json',histories);save(out/'status.json',{'status':'COMPLETE_VALIDATED','ensemble_macro_f1':ensemble_metrics['macro_f1']});print(json.dumps(report,indent=2))
if __name__=='__main__':main()
