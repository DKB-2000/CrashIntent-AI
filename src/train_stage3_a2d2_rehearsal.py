"""Frozen three-arm steer-head adaptation with A2D2 and comma2k19 rehearsal."""
from __future__ import annotations
import copy,hashlib,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
import stage3_motion_inference as deploy

P=Path(__file__).resolve().parents[1];OUT=P/'artifacts/stage3-a2d2-rehearsal-20260918'
CKPT=P/'artifacts/stage3-warmstart-ce-submit-candidate-20260917/best.pt'
REF=P/'artifacts/stage3-representation-20260910';TRAIN=P/'artifacts/stage3-a2d2-training-clips-20260917';HOLD=P/'artifacts/stage3-a2d2-clip-pilot-20260917';CIVIC=P/'artifacts/stage3-civic-535-balanced-evaluation-20260917'
ARMS={'mix10':.10,'mix25':.25,'mix50':.50};CLASSES=['LEFT','STRAIGHT','RIGHT']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(tmp,path)
def metric(y,p):
 cm=np.array([[int(np.sum((y==i)&(p==j))) for j in range(3)] for i in range(3)]);sup=cm.sum(1);den=sup+cm.sum(0)
 f=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0);r=np.divide(np.diag(cm),sup,out=np.zeros(3),where=sup>0)
 return dict(rows=len(y),macro_f1=float(f.mean()),accuracy=float(np.mean(y==p)),recall=dict(zip(CLASSES,map(float,r))),confusion=cm.tolist())
def old_arrays(frame):
 cache={};mapping={}
 for sid in frame.ID.unique():
  with np.load(REF/'cache'/f'{sid}.npz',allow_pickle=False) as z:cache[sid]=z['features'][:,1280:];mapping[sid]={int(e):i for i,e in enumerate(z['endpoints'])}
 return np.stack([cache[r.ID][mapping[r.ID][int(r.endpoint)]] for r in frame.itertuples()]).astype(np.float32)
def a2d2_arrays(root):
 frame=pd.read_csv(root/'samples.csv');values=[]
 for row in frame.itertuples(index=False):
  f=deploy.s3_motion_features(root/'videos'/f'{row.ID}.mp4');assert f.shape==(16,402);values.append(f[15])
 y=frame.steer_label.map({x:i for i,x in enumerate(CLASSES)}).to_numpy();return frame,np.asarray(values,np.float32),y
def civic_arrays():
 labels=pd.read_csv(CIVIC/'proxy-labels.csv');xs=[];ys=[]
 for path in sorted(CIVIC.glob('*-features.npz')):
  sid=path.name.removesuffix('-features.npz');g=labels[(labels.ID==sid)&labels.valid].sort_values('sample_index')
  with np.load(path,allow_pickle=False) as z:f=z['features'][g.sample_index.to_numpy()]
  xs.append(f);ys.extend(CLASSES.index(x) for x in g.truth)
 return np.concatenate(xs).astype(np.float32),np.asarray(ys)
def logits(model,x,mean,std):
 z=np.zeros((len(x),1682),np.float32);z[:,1280:]=np.clip((x-mean[1280:])/std[1280:],-10,10)
 with torch.inference_mode():_,s=model(torch.from_numpy(z));return s.numpy()
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);status=OUT/'status.json';save(status,dict(status='EXTRACTING'))
 model,mean,std=deploy.s3_motion_load(CKPT,torch.device('cpu'));train=pd.read_csv(REF/'train-samples.csv');valid=pd.read_csv(REF/'validation-samples.csv')
 train=train[train.accel!=3].reset_index(drop=True);valid=valid[valid.accel!=3].reset_index(drop=True)
 ox=old_arrays(train);oy=train.steer.to_numpy();vx=old_arrays(valid);vy=valid.steer.to_numpy();af,ax,ay=a2d2_arrays(TRAIN);hf,hx,hy=a2d2_arrays(HOLD);cx,cy=civic_arrays()
 baseline=dict(old=metric(vy,logits(model,vx,mean,std).argmax(1)),a2d2=metric(hy,logits(model,hx,mean,std).argmax(1)),civic=metric(cy,logits(model,cx,mean,std).argmax(1)))
 def normalized(x):return torch.from_numpy(np.clip((x-mean[1280:])/std[1280:],-10,10))
 with torch.inference_mode():
  def embed(x):
   full=torch.zeros((len(x),1682));full[:,1280:]=normalized(x);return model.net(full)
  oz,az,vz,hz,cz=map(embed,(ox,ax,vx,hx,cx))
 oz,az,vz,hz,cz=[torch.from_numpy(z.cpu().numpy().copy()) for z in (oz,az,vz,hz,cz)]
 results=[]
 for arm,weight in ARMS.items():
  head=copy.deepcopy(model.steer);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4);oty=torch.tensor(oy);aty=torch.tensor(ay)
  for step in range(300):
   loss=(1-weight)*F.cross_entropy(head(oz),oty)+weight*F.cross_entropy(head(az),aty)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
  with torch.inference_mode():preds=[head(z).argmax(1).numpy() for z in (vz,hz,cz)]
  scores=dict(old=metric(vy,preds[0]),a2d2=metric(hy,preds[1]),civic=metric(cy,preds[2]))
  gates=dict(old=scores['old']['macro_f1']>=baseline['old']['macro_f1']-.005,civic=scores['civic']['macro_f1']>=baseline['civic']['macro_f1']-.01,a2d2_f1=scores['a2d2']['macro_f1']>baseline['a2d2']['macro_f1'],a2d2_straight=scores['a2d2']['recall']['STRAIGHT']>baseline['a2d2']['recall']['STRAIGHT'],a2d2_turns=min(scores['a2d2']['recall']['LEFT'],scores['a2d2']['recall']['RIGHT'])>=.90)
  candidate=copy.deepcopy(torch.load(CKPT,map_location='cpu',weights_only=True));candidate['model']['steer.weight']=head.weight.detach().clone();candidate['model']['steer.bias']=head.bias.detach().clone();path=OUT/f'{arm}.pt';torch.save(candidate,path)
  results.append(dict(arm=arm,a2d2_weight=weight,steps=300,final_loss=float(loss),scores=scores,gates=gates,passed=all(gates.values()),checkpoint=str(path),checkpoint_sha256=sha(path)))
  save(status,dict(status='TRAINING',completed=len(results),total=len(ARMS)))
 passed=[r for r in results if r['passed']];selected=max(passed,key=lambda r:r['scores']['a2d2']['macro_f1'])['arm'] if passed else None
 report=dict(status='COMPLETE_VALIDATED',decision='ADOPT' if selected else 'KEEP_BASELINE',selected=selected,baseline=baseline,arms=results,contracts=dict(train_routes=sorted(af.route.unique()),holdout_routes=sorted(hf.route.unique()),checkpoint_sha256=sha(CKPT),old_train=len(train),a2d2_train=108,a2d2_holdout=36,trainable='steer.weight and steer.bias only'))
 save(OUT/'report.json',report);save(status,dict(status='COMPLETE_VALIDATED',completed=3,total=3,decision=report['decision'],selected=selected));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
