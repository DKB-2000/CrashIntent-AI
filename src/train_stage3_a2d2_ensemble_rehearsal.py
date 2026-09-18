"""Apply fixed A2D2 rehearsal arms to every deployed ensemble member."""
from __future__ import annotations
import copy,json
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
import stage3_motion_inference as deploy
import train_stage3_a2d2_rehearsal as b

OUT=b.P/'artifacts/stage3-a2d2-ensemble-rehearsal-20260918';CKPT=b.P/'artifacts/stage3-warmstart-ce-ensemble-submit-candidate-20260917/best.pt';ARMS={'mix10':.10,'mix50':.50}
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);status=OUT/'status.json';b.save(status,dict(status='EXTRACTING'))
 ck=torch.load(CKPT,map_location='cpu',weights_only=True);mean=ck['mean'].numpy();std=ck['std'].numpy();train=pd.read_csv(b.REF/'train-samples.csv');valid=pd.read_csv(b.REF/'validation-samples.csv');train=train[train.accel!=3].reset_index(drop=True);valid=valid[valid.accel!=3].reset_index(drop=True)
 ox=b.old_arrays(train);oy=train.steer.to_numpy();vx=b.old_arrays(valid);vy=valid.steer.to_numpy();af,ax,ay=b.a2d2_arrays(b.TRAIN);hf,hx,hy=b.a2d2_arrays(b.HOLD);cx,cy=b.civic_arrays()
 def full(x):
  z=np.zeros((len(x),1682),np.float32);z[:,1280:]=np.clip((x-mean[1280:])/std[1280:],-10,10);return torch.from_numpy(z)
 tensors=list(map(full,(ox,ax,vx,hx,cx)));models=[]
 for state in ck['models']:
  m=deploy.S3MotionModel();m.load_state_dict(state);m.eval();models.append(m)
 def ensemble_predictions(ms,index):
  with torch.inference_mode():return torch.stack([m(tensors[index])[1] for m in ms]).mean(0).argmax(1).numpy()
 baseline=dict(old=b.metric(vy,ensemble_predictions(models,2)),a2d2=b.metric(hy,ensemble_predictions(models,3)),civic=b.metric(cy,ensemble_predictions(models,4)))
 results=[]
 for arm,weight in ARMS.items():
  adapted=[]
  for source in models:
   with torch.no_grad():oz=source.net(tensors[0]).detach();az=source.net(tensors[1]).detach()
   head=copy.deepcopy(source.steer);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4);oty=torch.tensor(oy);aty=torch.tensor(ay)
   for _ in range(300):
    loss=(1-weight)*F.cross_entropy(head(oz),oty)+weight*F.cross_entropy(head(az),aty);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
   m=copy.deepcopy(source);m.steer.load_state_dict(head.state_dict());m.eval();adapted.append(m)
  scores=dict(old=b.metric(vy,ensemble_predictions(adapted,2)),a2d2=b.metric(hy,ensemble_predictions(adapted,3)),civic=b.metric(cy,ensemble_predictions(adapted,4)))
  gates=dict(old=scores['old']['macro_f1']>=baseline['old']['macro_f1']-.005,civic=scores['civic']['macro_f1']>=baseline['civic']['macro_f1']-.01,a2d2_f1=scores['a2d2']['macro_f1']>baseline['a2d2']['macro_f1'],a2d2_straight=scores['a2d2']['recall']['STRAIGHT']>baseline['a2d2']['recall']['STRAIGHT'],a2d2_turns=min(scores['a2d2']['recall']['LEFT'],scores['a2d2']['recall']['RIGHT'])>=.90)
  candidate=copy.deepcopy(ck);candidate['models']=[m.state_dict() for m in adapted];path=OUT/f'{arm}.pt';torch.save(candidate,path);results.append(dict(arm=arm,weight=weight,scores=scores,gates=gates,passed=all(gates.values()),checkpoint=str(path),checkpoint_sha256=b.sha(path)))
  b.save(status,dict(status='TRAINING',completed=len(results),total=len(ARMS)))
 passed=[r for r in results if r['passed']];selected=max(passed,key=lambda r:r['scores']['a2d2']['macro_f1'])['arm'] if passed else None
 report=dict(status='COMPLETE_VALIDATED',decision='ADOPT' if selected else 'KEEP_ENSEMBLE',selected=selected,baseline=baseline,arms=results,contract='All three deployed seed members; steer weight/bias only; fixed 300 steps; route-disjoint A2D2 holdout.')
 b.save(OUT/'report.json',report);b.save(status,dict(status='COMPLETE_VALIDATED',completed=2,total=2,decision=report['decision'],selected=selected));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
