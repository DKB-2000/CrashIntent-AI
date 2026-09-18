"""Blend original and A2D2-adapted ensemble logits under fixed gates."""
import json
import numpy as np,pandas as pd,torch
import stage3_motion_inference as d
import train_stage3_a2d2_rehearsal as b
P=b.P;OUT=P/'artifacts/stage3-a2d2-logit-blend-20260918';OLD=P/'artifacts/stage3-warmstart-ce-ensemble-submit-candidate-20260917/best.pt';NEW=P/'artifacts/stage3-a2d2-ensemble-rehearsal-20260918/mix50.pt'
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);old=torch.load(OLD,map_location='cpu',weights_only=True);new=torch.load(NEW,map_location='cpu',weights_only=True);mean=old['mean'].numpy();std=old['std'].numpy()
 train=pd.read_csv(b.REF/'validation-samples.csv');train=train[train.accel!=3].reset_index(drop=True);vx=b.old_arrays(train);vy=train.steer.to_numpy();_,hx,hy=b.a2d2_arrays(b.HOLD);cx,cy=b.civic_arrays();sets={'old':(vx,vy),'a2d2':(hx,hy),'civic':(cx,cy)}
 def models(ck):
  out=[]
  for state in ck['models']:m=d.S3MotionModel();m.load_state_dict(state);m.eval();out.append(m)
  return out
 om,nm=models(old),models(new)
 def logits(ms,x):
  z=np.zeros((len(x),1682),np.float32);z[:,1280:]=np.clip((x-mean[1280:])/std[1280:],-10,10);t=torch.from_numpy(z)
  with torch.inference_mode():return torch.stack([m(t)[1] for m in ms]).mean(0).numpy()
 cache={name:(logits(om,x),logits(nm,x),y) for name,(x,y) in sets.items()};baseline={name:b.metric(y,o.argmax(1)) for name,(o,n,y) in cache.items()};arms=[]
 for alpha in (.25,.50,.75):
  scores={name:b.metric(y,((1-alpha)*o+alpha*n).argmax(1)) for name,(o,n,y) in cache.items()};g=dict(old=scores['old']['macro_f1']>=baseline['old']['macro_f1']-.005,civic=scores['civic']['macro_f1']>=baseline['civic']['macro_f1']-.01,a2d2_f1=scores['a2d2']['macro_f1']>baseline['a2d2']['macro_f1'],a2d2_straight=scores['a2d2']['recall']['STRAIGHT']>baseline['a2d2']['recall']['STRAIGHT'],a2d2_turns=min(scores['a2d2']['recall']['LEFT'],scores['a2d2']['recall']['RIGHT'])>=.90);arms.append(dict(adapted_weight=alpha,scores=scores,gates=g,passed=all(g.values())))
 passed=[x for x in arms if x['passed']];selected=max(passed,key=lambda x:x['scores']['a2d2']['macro_f1'])['adapted_weight'] if passed else None;report=dict(status='COMPLETE_VALIDATED',decision='ADOPT_BLEND' if selected else 'KEEP_ENSEMBLE',selected_adapted_weight=selected,baseline=baseline,arms=arms)
 b.save(OUT/'report.json',report);b.save(OUT/'status.json',dict(status='COMPLETE_VALIDATED',decision=report['decision'],selected=selected));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
