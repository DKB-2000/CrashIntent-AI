"""Nested source-held-out Stage2 early stopping and encoder freezing study."""
import copy,hashlib,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
import stage2_pipeline as p
from train_stage2_controls import predictions,temporal_loss
from analyze_stage2_source_cv import details,summary
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2-regularization-20260911'
CV=ROOT/'artifacts/stage2-source-cv-20260910'
PRE=ROOT/'artifacts/stage2-ccd-pretrain-20260910'
MODES=['early_stop','low_encoder_lr','frozen_encoder']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
 target=OUT/name;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(target)


def initialize(state,mode):
 model=p.Stage2Temporal();model.load_state_dict(state,strict=True)
 random.seed(p.SEED);np.random.seed(p.SEED);torch.manual_seed(p.SEED)
 if mode=='frozen_encoder':
  for v in model.r.parameters():v.requires_grad_(False)
 groups=[{'params':[v for n,v in model.named_parameters() if not n.startswith('r.')],'lr':2e-4}]
 if mode!='frozen_encoder':groups.append({'params':list(model.r.parameters()),'lr':2e-5 if mode=='low_encoder_lr' else 2e-4})
 return model,torch.optim.AdamW(groups),random.Random(p.SEED),random.Random(p.SEED+1)


def train_epoch(model,opt,rows,features,epoch,mode,order_rng,scene_rng):
 model.train()
 if mode=='frozen_encoder':model.r.eval()
 order=list(rows);order_rng.shuffle(order);losses=[]
 for r in order:
  cl,el,h=model.logits(features[r['ID']][None])
  ct,et=torch.tensor([r['t_collision']]),torch.tensor([r['t_entry']])
  mixed=scene_rng.random()<min(.5,epoch/20)
  ci,ei=(cl.argmax(1),el.argmax(1)) if mixed else (ct,et)
  scene=model.scene(torch.cat([h[0,ci],h[0,ei]],1))
  loss=temporal_loss(cl,ct,1.)+temporal_loss(el,et,1.)+F.cross_entropy(scene[:,:2],torch.tensor([r['evasion_space']]))+F.cross_entropy(scene[:,2:],torch.tensor([int(r['entry_side']=='RIGHT')]))
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
 return float(np.mean(losses))


def main():
 OUT.mkdir(exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
 assert json.loads((PRE/'independent-validation.json').read_text())['status']=='PASS'
 protocol=json.loads((PRE/'protocol.json').read_text())
 path=ROOT/'artifacts/stage2-controls-20260909/features.pt';assert sha(path)==protocol['manual_features_sha256']
 features=torch.load(path,weights_only=True,map_location='cpu')
 state=torch.load(PRE/'collision-pretrain.pt',weights_only=True,map_location='cpu')['model']
 write('protocol.json',{'folds':protocol['folds'],'modes':MODES,'epochs_max':15,'head_lr':2e-4,'encoder_lr':{'early_stop':2e-4,'low_encoder_lr':2e-5,'frozen_encoder':0},'seed':p.SEED,'inner_seed':20260911,'selection':'Highest source-equal 4-item inner score; earliest epoch and listed mode order break ties; refit all outer training rows for selected epoch count','pretrain_sha256':sha(PRE/'collision-pretrain.pt'),'features_sha256':sha(path),'script_sha256':sha(Path(__file__)),'scope':'Nested inner source split inside reused exploratory 5 outer folds; not fresh test','frozen_encoder_policy':'requires_grad False and encoder eval disables its dropout'})
 collected={m:[] for m in MODES+['inner_selected']};all_rows=[];selections=[]
 for fold,held in enumerate(protocol['folds']):
  folder=OUT/f'fold-{fold}';folder.mkdir()
  train=pd.read_csv(CV/f'fold-{fold}/train.csv',dtype={'ID':str,'source_id':str}).to_dict('records')
  val=pd.read_csv(CV/f'fold-{fold}/validation.csv',dtype={'ID':str,'source_id':str}).to_dict('records')
  groups=sorted({r['source_id'] for r in train});random.Random(20260911+fold).shuffle(groups)
  inner_groups=set(groups[:max(2,round(len(groups)*.2))])
  inner_train=[r for r in train if r['source_id'] not in inner_groups];inner_val=[r for r in train if r['source_id'] in inner_groups]
  sets=[{r['source_id'] for r in rows} for rows in [inner_train,inner_val,val]]
  assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
  assert not set(held)&set(pd.read_csv(PRE/'pretrain-labels.csv',dtype={'source_id':str}).source_id)
  for name,rows in [('inner_train',inner_train),('inner_validation',inner_val),('train',train),('validation',val)]:pd.DataFrame(rows).to_csv(folder/f'{name}.csv',index=False)
  decisions=[]
  for mode in MODES:
   model,opt,order_rng,scene_rng=initialize(state,mode);history=[]
   for epoch in range(1,16):
    loss=train_epoch(model,opt,inner_train,features,epoch,mode,order_rng,scene_rng)
    pred=predictions(model,inner_val,features,'cpu');score=summary(inner_val,pred)['source_equal_mean']
    history.append({'epoch':epoch,'loss':loss,'score':score,'predictions':pred})
    write('status.json',{'status':'RUNNING','phase':'INNER_SELECTION','fold':fold+1,'mode':mode,'epoch':epoch,'seconds':time.monotonic()-started})
    print('inner',fold+1,mode,epoch,flush=True)
   best=max(history,key=lambda h:h['score'])
   decision={'mode':mode,'epoch':best['epoch'],'score':best['score']};decisions.append(decision)
   (folder/f'{mode}_inner.json').write_text(json.dumps(history,indent=2))
  # Lock settings before any outer validation inference.
  chosen=max(decisions,key=lambda d:d['score'])
  (folder/'selection.json').write_text(json.dumps({'modes':decisions,'selected':chosen},indent=2))
  selections.append({'fold':fold,**chosen})
  all_rows.extend([dict(r,fold=fold) for r in val])
  for d in decisions:
   mode=d['mode'];model,opt,order_rng,scene_rng=initialize(state,mode)
   for epoch in range(1,d['epoch']+1):
    train_epoch(model,opt,train,features,epoch,mode,order_rng,scene_rng)
    write('status.json',{'status':'RUNNING','phase':'OUTER_REFIT','fold':fold+1,'mode':mode,'epoch':epoch,'target_epochs':d['epoch'],'seconds':time.monotonic()-started})
   if mode=='frozen_encoder':assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items() if k.startswith('r.'))
   path=folder/f'{mode}.pt';torch.save({'model':model.state_dict(),'epoch':d['epoch']},path)
   pred=predictions(model,val,features,'cpu')
   reload=p.Stage2Temporal();reload.load_state_dict(torch.load(path,weights_only=True)['model'],strict=True)
   assert predictions(reload,val,features,'cpu')==pred
   collected[mode].extend(pred)
   if mode==chosen['mode']:collected['inner_selected'].extend(pred)
  write('selections.json',selections)
 assert len(all_rows)==66 and len({r['ID'] for r in all_rows})==66
 results={}
 for mode,pred in collected.items():
  pd.DataFrame(details(all_rows,pred)).to_csv(OUT/f'{mode}_oof.csv',index=False)
  results[mode]=summary(all_rows,pred)
 write('report.json',{'status':'COMPLETED','seconds':time.monotonic()-started,'results':results,'selections':selections,'prior_fixed15_reference':json.loads((PRE/'report.json').read_text())['results']['pretrained']})
 write('status.json',{'status':'COMPLETED','seconds':time.monotonic()-started})


if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():write('status.json',{'status':'FAILED','error':repr(e)})
  raise
