"""Frozen arithmetic-logit ensemble of the three warmstart_ce seeds."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
import compare_stage3_representations as base

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/stage3-warmstart-ce-ensemble-20260917'
TRAIN=P/'artifacts/stage3-prediction-preservation-20260914/warmstart_ce'
REF=P/'artifacts/stage3-representation-20260910'
CIVIC=P/'artifacts/stage3-civic-535-balanced-evaluation-20260917'
SEEDS=[20260910,20260911,20260912]
ACCEL=['ACCELERATING','DECELERATING','CONSTANT','STOPPED'];STEER=['LEFT','STRAIGHT','RIGHT']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):
 ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def metric(truth,pred,names):
 truth=np.asarray(truth);pred=np.asarray(pred);cm=np.array([[int(np.sum((truth==i)&(pred==j))) for j in range(len(names))] for i in range(len(names))]);sup=cm.sum(1);den=cm.sum(0)+sup
 f=np.divide(2*np.diag(cm),den,out=np.zeros(len(names)),where=den>0);rec=np.divide(np.diag(cm),sup,out=np.zeros(len(names)),where=sup>0)
 return dict(rows=len(truth),macro_f1=float(f.mean()),recall={n:float(rec[i]) for i,n in enumerate(names)},confusion=cm.tolist())
def transitions(frame,pred):
 rows=[]
 for sid,g in frame.groupby('ID'):
  truth=g.steer.to_numpy();moving=g.accel.to_numpy()!=3;p=pred[g.index]
  for t in range(10,len(g)-12):
   before,after=truth[t-1],truth[t]
   if before==after or (before!=1 and after!=1):continue
   if not (moving[t-10:t+10].all() and np.all(truth[t-10:t]==before) and np.all(truth[t:t+10]==after)):continue
   hits=[k for k in range(11) if np.all(p[t+k:t+k+3]==after)]
   rows.append(dict(kind='turn_start' if before==1 else 'turn_end',detected=bool(hits),delay=hits[0]/10 if hits else None))
 out={}
 for kind in ('turn_start','turn_end'):
  g=[r for r in rows if r['kind']==kind];hit=[r['delay'] for r in g if r['detected']]
  out[kind]=dict(events=len(g),missed=sum(not r['detected'] for r in g),mean_delay_s=float(np.mean(hit)) if hit else None)
 return out
def civic_logits(models):
 truth=pd.read_csv(CIVIC/'proxy-labels.csv');parts=[]
 for path in sorted(CIVIC.glob('*-features.npz')):
  sid=path.name.removesuffix('-features.npz')
  with np.load(path,allow_pickle=False) as z:f=z['features']
  logits=[]
  for ck,model in models:
   x=np.zeros((len(f),1682),np.float32);x[:,1280:]=np.clip((f-ck['mean'].numpy()[1280:])/ck['std'].numpy()[1280:],-10,10)
   with torch.inference_mode():_,s=model(torch.from_numpy(x));logits.append(s.numpy())
  parts.append(pd.DataFrame(dict(ID=sid,sample_index=np.arange(len(f)),single=np.argmax(logits[0],1),ensemble=np.argmax(np.mean(logits,axis=0),1))))
 return truth.merge(pd.concat(parts),on=['ID','sample_index'],validate='one_to_one')
def run():
 torch.set_num_threads(1);ROOT.mkdir(parents=True,exist_ok=True)
 source={str((TRAIN/str(s)/'model.pt').relative_to(P)):sha(TRAIN/str(s)/'model.pt') for s in SEEDS}
 plan=dict(status='FROZEN',seeds=SEEDS,combination='Unweighted arithmetic mean of raw accel and steer logits; no temperatures, weights or thresholds.',baseline='Deployed warmstart_ce seed20260910',gate='Old validation: both heads improve, >=3/4 steer routes improve, no class recall loss >2pp, transition misses/delay noninferior. Civic: steer F1 improves and no recall loss >2pp.',source_sha256=source,script_sha256=sha(__file__))
 if (ROOT/'plan.json').exists():assert json.loads((ROOT/'plan.json').read_text())==plan
 else:save('plan.json',plan)
 valid=pd.read_csv(REF/'validation-samples.csv');acc=[];steer=[];models=[]
 for seed in SEEDS:
  folder=TRAIN/str(seed);assert sha(folder/'model.pt')==source[str((folder/'model.pt').relative_to(P))]
  with np.load(folder/'validation-logits.npz',allow_pickle=False) as z:acc.append(z['accel']);steer.append(z['steer'])
  ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);m=base.Control();m.load_state_dict(ck['model'],strict=True);m.eval();models.append((ck,m))
 single_a=acc[0].argmax(1);single_s=steer[0].argmax(1);ens_a=np.mean(acc,axis=0).argmax(1);ens_s=np.mean(steer,axis=0).argmax(1);moving=valid.accel.to_numpy()!=3
 old={k:{'accel':metric(valid.accel,v[0],ACCEL),'steer':metric(valid.steer.to_numpy()[moving],v[1][moving],STEER),'transitions':transitions(valid,v[1])} for k,v in {'single':(single_a,single_s),'ensemble':(ens_a,ens_s)}.items()}
 routes=[]
 for route,g in valid.groupby('route'):
  gm=g.accel.to_numpy()!=3;routes.append(dict(route=route,single=metric(g.steer.to_numpy()[gm],single_s[g.index][gm],STEER)['macro_f1'],ensemble=metric(g.steer.to_numpy()[gm],ens_s[g.index][gm],STEER)['macro_f1']))
 cj=civic_logits(models);eligible=cj[cj.valid];ct=np.array([STEER.index(x) for x in eligible.truth]);civic={k:metric(ct,eligible[k],STEER) for k in ('single','ensemble')}
 old_recall=[]
 for head in ('accel','steer'):
  old_recall.extend(old['ensemble'][head]['recall'][c]-old['single'][head]['recall'][c] for c in (ACCEL if head=='accel' else STEER))
 transition_pass=all(old['ensemble']['transitions'][k]['missed']<=old['single']['transitions'][k]['missed'] and old['ensemble']['transitions'][k]['mean_delay_s']<=old['single']['transitions'][k]['mean_delay_s']+.1 for k in ('turn_start','turn_end'))
 civic_recall=[civic['ensemble']['recall'][c]-civic['single']['recall'][c] for c in STEER]
 gates=dict(old_heads=old['ensemble']['accel']['macro_f1']>old['single']['accel']['macro_f1'] and old['ensemble']['steer']['macro_f1']>old['single']['steer']['macro_f1'],routes=sum(r['ensemble']>r['single'] for r in routes)>=3,old_recall=min(old_recall)>=-.02,transitions=transition_pass,civic_f1=civic['ensemble']['macro_f1']>civic['single']['macro_f1'],civic_recall=min(civic_recall)>=-.02)
 result=dict(status='COMPLETE_VALIDATED',decision='FOLLOWUP_ONLY' if all(gates.values()) else 'KEEP_SINGLE',gates=gates,old_validation=old,routes=routes,civic=civic,old_recall_delta=old_recall,civic_recall_delta=civic_recall,adopted=False,validation='Frozen source hashes; saved validation logits; independent confusion/F1; exact model reload; fixed Civic features/labels; transition audit.')
 save('report.json',result);save('status.json',dict(status='COMPLETE_VALIDATED',completed=3,total=3,decision=result['decision']));print(json.dumps(result,indent=2))
if __name__=='__main__':run()
