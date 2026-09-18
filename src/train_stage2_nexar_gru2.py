"""Conservatively adapt only Stage2 GRU layer 2 plus time heads on Nexar-200."""
from __future__ import annotations
import csv,hashlib,json,random,statistics,time
from pathlib import Path
import torch
from torch.nn import functional as F
import stage2_pipeline as pipeline

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'artifacts/stage2-nexar-temporal-features-20260917/features'
MANIFEST=ROOT/'artifacts/stage2-nexar-temporal-pool-20260917/manifest-sized.csv'
START=ROOT/'artifacts/stage2-nexar-temporal-heads-v2-20260917/candidate.pt'
VAL_FEATURES=ROOT/'artifacts/stage2-unlabeled-comparison-20260914/features'
VAL_MANIFEST=ROOT/'data/stage2-validation-nexar-20260914/manifest.json'
VAL_META=ROOT/'data_raw/stage2-validation-nexar-20260914/review-inventory-resolved.csv'
OUT=ROOT/'artifacts/stage2-nexar-gru2-20260918'
SEEDS=(20260918,20260919,20260920);EPOCHS=(1,2,5)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,obj):
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/(name+'.tmp');p.write_text(json.dumps(obj,indent=2)+'\n');p.replace(OUT/name)
def soft_loss(logits,index):
 x=torch.arange(len(logits),device=logits.device,dtype=torch.float32);q=torch.exp(-.5*((x-index)/5.)**2);q/=q.sum();return -(q*F.log_softmax(logits,0)).sum()
def metrics(err):
 a=[abs(x) for x in err];return {'mean_abs_seconds':sum(a)/len(a),'median_abs_seconds':statistics.median(a),'mean_signed_seconds':sum(err)/len(err),**{f'within_{x}s':sum(v<=x+1e-12 for v in a)/len(a) for x in (.3,.5,1,2,5)}}
def score(model,samples):
 ce=[];ee=[];model.eval()
 with torch.inference_mode():
  for s in samples:
   ci,ei,_=model(s['x'][None]);ce.append(float(s['times'][int(ci)]-s['event']));ee.append(float(s['times'][int(ei)]-s['alert']))
 return {'collision_vs_event':metrics(ce),'entry_vs_alert_proxy':metrics(ee),'selection_mae_sum':sum(abs(x) for x in ce+ee)/len(ce)}
def load_data():
 manifest={Path(r['file_name']).stem:r for r in read(MANIFEST)};items=[]
 for pth in sorted(TRAIN.glob('*.pt')):
  z=torch.load(pth,map_location='cpu',weights_only=True);r=manifest[pth.stem];items.append({'id':pth.stem,'x':z['features'].float(),'times':z['source_frame_indices'].float()/float(z['fps']),'event':float(r['time_of_event']),'alert':float(r['time_of_alert']),'ci':int(z['event_index']),'ei':int(z['alert_index'])})
 ranked=sorted(items,key=lambda x:hashlib.sha256(('split-20260917-'+x['id']).encode()).digest());return ranked[40:],ranked[:40]
def load_val():
 meta={r['ID']:r for r in read(VAL_META)};out=[]
 for v in json.loads(VAL_MANIFEST.read_text())['videos']:
  full=torch.load(VAL_FEATURES/(v['ID']+'.pt'),map_location='cpu',weights_only=True).float();fm=read(ROOT/'data/stage2-validation-nexar-20260914'/v['frame_map']);t=torch.tensor([float(r['timestamp_seconds']) for r in fm]);idx=torch.arange(0,len(t),3);m=meta[v['ID']];out.append({'id':v['ID'],'x':full[idx],'times':t[idx],'event':float(m['reference_event_seconds']),'alert':float(m['reference_alert_seconds'])})
 return out
def model_from(state):
 m=pipeline.Stage2Temporal();m.load_state_dict(state)
 for name,q in m.named_parameters():q.requires_grad_(name.startswith('r.') and ('_l1' in name) or name.startswith('tc.') or name.startswith('te.'))
 return m
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();torch.set_num_threads(1);started=time.monotonic();initial=torch.load(START,map_location='cpu',weights_only=True);train,inner=load_data();val=load_val()
 save('protocol.json',{'status':'PREDECLARED','start_sha256':sha(START),'seeds':SEEDS,'epochs':EPOCHS,'split':'same frozen SHA256 160/40 as head-only experiment','trainable':'GRU layer l1 forward/reverse plus tc/te; GRU l0 and scene frozen','loss':'Gaussian soft target sigma 0.5s','lr':{'gru2':1e-4,'heads':5e-4},'weight_decay':1e-4,'gradient_clip':1.0,'selection':'minimum seed-mean internal MAE sum','final':'average selected epoch state across 3 seeds','frozen50_use':'one confirmation comparison against current official candidate; previously exposed','gate':'both MAE improve, MAE sum >=0.25s better, neither within2s lower'})
 runs=[];states={}
 for seed in SEEDS:
  random.seed(seed);torch.manual_seed(seed);m=model_from(initial);gru=[q for n,q in m.named_parameters() if q.requires_grad and n.startswith('r.')];heads=[q for n,q in m.named_parameters() if q.requires_grad and not n.startswith('r.')];opt=torch.optim.AdamW([{'params':gru,'lr':1e-4},{'params':heads,'lr':5e-4}],weight_decay=1e-4);snap={}
  for epoch in range(1,max(EPOCHS)+1):
   m.train();order=list(train);random.shuffle(order)
   for s in order:
    cl,el,_=m.logits(s['x'][None]);loss=soft_loss(cl[0],s['ci'])+soft_loss(el[0],s['ei']);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],1.);opt.step()
   if epoch in EPOCHS:
    q=score(m,inner);runs.append({'seed':seed,'epoch':epoch,**q});snap[epoch]={k:v.detach().clone() for k,v in m.state_dict().items()}
  states[seed]=snap;save('status.json',{'status':'TRAINING','completed_seeds':len(states),'total_seeds':len(SEEDS),'seconds':time.monotonic()-started})
 aggregate=[]
 for e in EPOCHS:
  z=[r for r in runs if r['epoch']==e];aggregate.append({'epoch':e,'seed_mean_selection_mae_sum':sum(r['selection_mae_sum'] for r in z)/len(z),'seed_values':[r['selection_mae_sum'] for r in z]})
 chosen=min(aggregate,key=lambda x:(x['seed_mean_selection_mae_sum'],x['epoch']));trainable_keys={k for k in initial if (k.startswith('r.') and '_l1' in k) or k.startswith('tc.') or k.startswith('te.')};averaged={k:(torch.stack([states[s][chosen['epoch']][k] for s in SEEDS]).mean(0) if k in trainable_keys else initial[k].clone()) for k in initial};candidate=model_from(averaged);base=model_from(initial);b=score(base,val);c=score(candidate,val)
 gates={'collision_mae_improves':c['collision_vs_event']['mean_abs_seconds']<b['collision_vs_event']['mean_abs_seconds'],'entry_mae_improves':c['entry_vs_alert_proxy']['mean_abs_seconds']<b['entry_vs_alert_proxy']['mean_abs_seconds'],'sum_improves_0.25':c['selection_mae_sum']<=b['selection_mae_sum']-.25,'collision_within2_not_lower':c['collision_vs_event']['within_2s']>=b['collision_vs_event']['within_2s'],'entry_within2_not_lower':c['entry_vs_alert_proxy']['within_2s']>=b['entry_vs_alert_proxy']['within_2s']}
 torch.save(averaged,OUT/'candidate.pt');report={'status':'COMPLETE_VALIDATED','decision':'PROMOTE_FOR_INTEGRATION' if all(gates.values()) else 'KEEP_OFFICIAL_91775','chosen':chosen,'internal_runs':runs,'internal_aggregates':aggregate,'frozen50':{'official_91775_time_model':b,'gru2_candidate':c},'gates':gates,'candidate_sha256':sha(OUT/'candidate.pt'),'seconds':time.monotonic()-started,'adopted':False};save('report.json',report);save('status.json',{'status':'COMPLETE_VALIDATED','decision':report['decision'],'seconds':report['seconds']})
if __name__=='__main__':main()
