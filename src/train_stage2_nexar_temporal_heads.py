"""Train only Stage2 time heads on Nexar labels and evaluate once on frozen Nexar-50."""
from __future__ import annotations
import csv, hashlib, json, random, statistics, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import stage2_pipeline as pipeline

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'artifacts/stage2-nexar-temporal-features-20260917'
MANIFEST=ROOT/'artifacts/stage2-nexar-temporal-pool-20260917/manifest-sized.csv'
VAL_FEATURES=ROOT/'artifacts/stage2-unlabeled-comparison-20260914/features'
VAL_MANIFEST=ROOT/'data/stage2-validation-nexar-20260914/manifest.json'
VAL_META=ROOT/'data_raw/stage2-validation-nexar-20260914/review-inventory-resolved.csv'
CKPT=ROOT/'artifacts/stage2-final-validation-20260914/models/incumbent.pt'
OUT=ROOT/'artifacts/stage2-nexar-temporal-heads-v2-20260917'
SEEDS=(20260917,20260918,20260919); EPOCHS=(2,5,10,20); ARMS=('exact','soft_0.5s')

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p:Path):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,obj):
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/(name+'.tmp');p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8');p.replace(OUT/name)
def state():
 z=torch.load(CKPT,map_location='cpu',weights_only=False)
 return z['model'] if isinstance(z,dict) and 'model' in z else z
def target_loss(logits,index,arm):
 if arm=='exact':return F.cross_entropy(logits[None],torch.tensor([index]))
 x=torch.arange(len(logits),dtype=torch.float32);q=torch.exp(-.5*((x-index)/5.0)**2);q/=q.sum()
 return -(q*F.log_softmax(logits,dim=0)).sum()
def metrics(errors):
 a=[abs(x) for x in errors]
 return {'mean_abs_seconds':sum(a)/len(a),'median_abs_seconds':statistics.median(a),'mean_signed_seconds':sum(errors)/len(errors),
         **{f'within_{x}s':sum(v<=x+1e-12 for v in a)/len(a) for x in (.3,.5,1,2,5)}}
def score(heads,samples):
 ce=[];ee=[]
 with torch.no_grad():
  for s in samples:
   ci=int(heads[0](s['hidden']).squeeze(-1).argmax());ei=int(heads[1](s['hidden']).squeeze(-1).argmax())
   ce.append(float(s['times'][ci]-s['event']));ee.append(float(s['times'][ei]-s['alert']))
 return {'collision_vs_event':metrics(ce),'entry_vs_alert_proxy':metrics(ee),'selection_mae_sum':sum(abs(x) for x in ce+ee)/len(ce)}
def load_train(model):
 manifest={Path(r['file_name']).stem:r for r in rows(MANIFEST)};items=[]
 with torch.no_grad():
  for p in sorted((TRAIN/'features').glob('*.pt')):
   z=torch.load(p,map_location='cpu',weights_only=True);r=manifest[p.stem];x=z['features'].float();h,_=model.r(x[None]);h=h[0].contiguous()
   times=z['source_frame_indices'].float()/float(z['fps'])
   items.append({'id':p.stem,'hidden':h,'times':times,'event':float(r['time_of_event']),'alert':float(r['time_of_alert']),'ci':int(z['event_index']),'ei':int(z['alert_index']),'scene':r['scene']})
 if len(items)!=200:raise RuntimeError(f'expected 200 training features, got {len(items)}')
 # Hash split avoids ordering effects and never reads frozen validation labels.
 ranked=sorted(items,key=lambda x:hashlib.sha256(('split-20260917-'+x['id']).encode()).digest())
 return ranked[40:],ranked[:40]
def load_validation(model):
 meta={r['ID']:r for r in rows(VAL_META)};manifest=json.loads(VAL_MANIFEST.read_text(encoding='utf-8'))['videos'];out=[]
 with torch.inference_mode():
  for v in manifest:
   full=torch.load(VAL_FEATURES/(v['ID']+'.pt'),map_location='cpu',weights_only=True).float();fm=rows(ROOT/'data/stage2-validation-nexar-20260914'/v['frame_map']);t=torch.tensor([float(r['timestamp_seconds']) for r in fm])
   if len(full)!=len(t):raise RuntimeError(f'validation length mismatch {v["ID"]}')
   duration=max(float(t[-1]),1e-6);fps=(len(t)-1)/duration;step=max(1,round(fps/10));idx=torch.arange(0,len(t),step);x=full[idx];h,_=model.r(x[None]);m=meta[v['ID']]
   out.append({'id':v['ID'],'hidden':h[0].contiguous(),'times':t[idx],'event':float(m['reference_event_seconds']),'alert':float(m['reference_alert_seconds'])})
 if len(out)!=50:raise RuntimeError('expected frozen 50')
 return out
def heads_from(model):
 a=nn.Linear(384,1);b=nn.Linear(384,1);a.load_state_dict(model.tc.state_dict());b.load_state_dict(model.te.state_dict());return a,b
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();torch.set_num_threads(1);started=time.monotonic();base_state=state();model=pipeline.Stage2Temporal();model.load_state_dict(base_state);model.eval()
 protocol={'status':'PREDECLARED','seeds':SEEDS,'arms':ARMS,'epochs':EPOCHS,'split':'SHA256(split-20260917+ID), first40 internal validation, remaining160 train','selection':'minimum seed-mean internal collision+entry MAE sum; frozen Nexar50 evaluated once after selection','trainable':'tc and te only; GRU and scene frozen','soft_sigma_seconds':.5,'sample_rate_hz':10,'checkpoint_sha256':sha(CKPT),'training_index_sha256':sha(TRAIN/'index.csv'),'validation_manifest_sha256':sha(VAL_MANIFEST),'promotion_gate':'both frozen50 MAEs improve, sum improves >=0.5 sec, neither within2s decreases','limitations':['Nexar event includes collision and near-miss outcomes','Nexar alert is only a proxy for competition entry']};save('protocol.json',protocol)
 train,inner=load_train(model);validation=load_validation(model);save('status.json',{'status':'TRAINING','completed':0,'total':len(SEEDS)*len(ARMS),'seconds':time.monotonic()-started})
 runs=[];saved={}
 for arm in ARMS:
  for seed in SEEDS:
   random.seed(seed);torch.manual_seed(seed);tc,te=heads_from(model);opt=torch.optim.AdamW(list(tc.parameters())+list(te.parameters()),lr=1e-3,weight_decay=1e-4);snap={}
   for epoch in range(1,max(EPOCHS)+1):
    order=list(train);random.shuffle(order)
    for s in order:
     lc=tc(s['hidden']).squeeze(-1);le=te(s['hidden']).squeeze(-1);loss=target_loss(lc,s['ci'],arm)+target_loss(le,s['ei'],arm)
     opt.zero_grad();loss.backward();opt.step()
    if epoch in EPOCHS:
     q=score((tc,te),inner);runs.append({'arm':arm,'seed':seed,'epoch':epoch,**q});snap[epoch]=(tc.state_dict(),te.state_dict())
   saved[(arm,seed)]=snap;save('status.json',{'status':'TRAINING','completed':len(saved),'total':len(SEEDS)*len(ARMS),'seconds':time.monotonic()-started})
 aggregates=[]
 for arm in ARMS:
  for epoch in EPOCHS:
   z=[r for r in runs if r['arm']==arm and r['epoch']==epoch];aggregates.append({'arm':arm,'epoch':epoch,'seed_mean_selection_mae_sum':sum(r['selection_mae_sum'] for r in z)/len(z),'seed_values':[r['selection_mae_sum'] for r in z]})
 chosen=min(aggregates,key=lambda x:(x['seed_mean_selection_mae_sum'],x['arm'],x['epoch']));states=[saved[(chosen['arm'],s)][chosen['epoch']] for s in SEEDS]
 tc,te=heads_from(model)
 for head,pos in ((tc,0),(te,1)):
  avg={k:torch.stack([st[pos][k] for st in states]).mean(0) for k in head.state_dict()};head.load_state_dict(avg)
 baseline=heads_from(model);base_score=score(baseline,validation);candidate_score=score((tc,te),validation)
 gates={'collision_mae_improves':candidate_score['collision_vs_event']['mean_abs_seconds']<base_score['collision_vs_event']['mean_abs_seconds'],'entry_mae_improves':candidate_score['entry_vs_alert_proxy']['mean_abs_seconds']<base_score['entry_vs_alert_proxy']['mean_abs_seconds'],'mae_sum_improves_at_least_0.5':candidate_score['selection_mae_sum']<=base_score['selection_mae_sum']-.5,'collision_within2_not_lower':candidate_score['collision_vs_event']['within_2s']>=base_score['collision_vs_event']['within_2s'],'entry_within2_not_lower':candidate_score['entry_vs_alert_proxy']['within_2s']>=base_score['entry_vs_alert_proxy']['within_2s']}
 candidate=dict(base_state);candidate['tc.weight']=tc.weight.detach();candidate['tc.bias']=tc.bias.detach();candidate['te.weight']=te.weight.detach();candidate['te.bias']=te.bias.detach();torch.save(candidate,OUT/'candidate.pt')
 report={'status':'COMPLETE_VALIDATED','decision':'PROMOTE_FOR_OFFICIAL_INTEGRATION_TEST' if all(gates.values()) else 'KEEP_INCUMBENT','chosen':chosen,'internal_runs':runs,'internal_aggregates':aggregates,'frozen50':{'baseline_10fps':base_score,'candidate_10fps':candidate_score},'gates':gates,'candidate_sha256':sha(OUT/'candidate.pt'),'seconds':time.monotonic()-started,'adopted':False};save('report.json',report);save('status.json',{'status':'COMPLETE_VALIDATED','decision':report['decision'],'seconds':report['seconds']})
if __name__=='__main__':main()
