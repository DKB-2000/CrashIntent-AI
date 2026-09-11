"""Train one final Stage2 model and replace only its checkpoint in submission 86692 ZIP."""
import ast,hashlib,io,json,random,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import stage2_pipeline as p
from train_stage2_controls import predictions,temporal_loss
from daily_submission import inspect_zip

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/stage2-ccd-pretrain-20260910'
CV=ROOT/'artifacts/stage2-source-cv-20260910'
OUT=ROOT/'artifacts/stage2-pretrained-submit-candidate-20260911'
BASE=ROOT/'artifacts/stage3-submit-candidate-20260910/submit.zip'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
 target=OUT/name;temp=target.with_suffix('.tmp')
 temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(target)


def main():
 OUT.mkdir(exist_ok=False)
 started=time.monotonic();torch.set_num_threads(2)
 assert json.loads((SOURCE/'independent-validation.json').read_text())['status']=='PASS'
 config=json.loads((SOURCE/'protocol.json').read_text())
 assert sha(BASE)=='cec269d3975db3d55a734505e40a3c03d8c43775e02fc887c2cfeba132964337'
 rows=pd.concat([pd.read_csv(CV/'fold-0/train.csv',dtype={'ID':str,'source_id':str}),pd.read_csv(CV/'fold-0/validation.csv',dtype={'ID':str,'source_id':str})]).sort_values('ID').to_dict('records')
 assert len(rows)==66 and len({r['ID'] for r in rows})==66
 cache=ROOT/'artifacts/stage2-controls-20260909/features.pt';assert sha(cache)==config['manual_features_sha256']
 features=torch.load(cache,weights_only=True,map_location='cpu');assert set(features)=={r['ID'] for r in rows}
 pretrain=SOURCE/'collision-pretrain.pt'
 checkpoint=torch.load(pretrain,weights_only=True,map_location='cpu');assert checkpoint['epochs']==3
 pd.DataFrame(rows).to_csv(OUT/'training-labels.csv',index=False)
 save('provenance.json',{'base_zip_sha256':sha(BASE),'pretrained_checkpoint_sha256':sha(pretrain),'features_sha256':sha(cache),'training_labels_sha256':sha(OUT/'training-labels.csv'),'script_sha256':sha(Path(__file__)),'cv_validation_sha256':sha(SOURCE/'independent-validation.json'),'seed':p.SEED,'epochs':15,'updates':990,'lr':2e-4,'training_rows':66,'sources':len({r['source_id'] for r in rows}),'scope':'Final fit on all accepted manual labels; no independent evaluation of this final fit. Prior 5-fold results are development evidence.'})
 model=p.Stage2Temporal();model.load_state_dict(checkpoint['model'],strict=True)
 random.seed(p.SEED);np.random.seed(p.SEED);torch.manual_seed(p.SEED)
 order_rng=random.Random(p.SEED);scene_rng=random.Random(p.SEED+1)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
 history=[]
 for epoch in range(1,16):
  model.train();order=list(rows);order_rng.shuffle(order);losses=[]
  for r in order:
   cl,el,h=model.logits(features[r['ID']].unsqueeze(0))
   ct,et=torch.tensor([r['t_collision']]),torch.tensor([r['t_entry']])
   mixed=scene_rng.random()<min(.5,epoch/20)
   ci,ei=(cl.argmax(1),el.argmax(1)) if mixed else (ct,et)
   scene=model.scene(torch.cat([h[0,ci],h[0,ei]],1))
   loss=temporal_loss(cl,ct,1.)+temporal_loss(el,et,1.)+F.cross_entropy(scene[:,:2],torch.tensor([r['evasion_space']]))+F.cross_entropy(scene[:,2:],torch.tensor([int(r['entry_side']=='RIGHT')]))
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
  history.append({'epoch':epoch,'loss':float(np.mean(losses))})
  save('status.json',{'status':'TRAINING','epoch':epoch,'epochs':15,'seconds':time.monotonic()-started})
  print(history[-1],flush=True)
 state=model.state_dict();assert all(torch.isfinite(v).all() for v in state.values())
 torch.save({'model':state,'labels':p.OUTPUT_COLUMNS},OUT/'best.pt')
 ref=predictions(model,rows,features,'cpu');pd.DataFrame(ref).to_csv(OUT/'training-predictions.csv',index=False)
 model.eval()
 with zipfile.ZipFile(BASE) as archive:
  tree=ast.parse(archive.read('inference.py').decode('utf-8-sig'))
  classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_Stage2Temporal'];assert len(classes)==1
  namespace={'torch':torch,'nn':nn};exec(compile(ast.Module(body=classes,type_ignores=[]),'<submission stage2>','exec'),namespace)
  packaged=namespace['_Stage2Temporal']().eval()
  packaged.load_state_dict(torch.load(OUT/'best.pt',weights_only=True)['model'],strict=True)
  with torch.inference_mode():
   for r in rows:
    x=features[r['ID']].unsqueeze(0)
    a=model(x);b=packaged(x)
    assert all(torch.equal(u,v) for u,v in zip(a,b))
  backbone=torch.load(io.BytesIO(archive.read('model/stage2/resnet18-f37072fd.pth')),weights_only=True,map_location='cpu')
  original_backbone=torch.load(ROOT/'artifacts/stage2-manual-200-20260907/training/model/stage2/resnet18-f37072fd.pth',weights_only=True,map_location='cpu')
  assert backbone.keys()==original_backbone.keys() and all(torch.equal(backbone[k],original_backbone[k]) for k in backbone)
  with zipfile.ZipFile(OUT/'submit.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as target:
   for info in archive.infolist():
    target.writestr(info.filename,(OUT/'best.pt').read_bytes() if info.filename=='model/stage2/best.pt' else archive.read(info.filename))
  with zipfile.ZipFile(OUT/'submit.zip') as target:
   assert target.testzip() is None
   assert target.namelist()==archive.namelist()
   changes=[name for name in target.namelist() if target.read(name)!=archive.read(name)]
   assert changes==['model/stage2/best.pt']
   assert target.read('model/stage2/best.pt')==(OUT/'best.pt').read_bytes()
 digest=inspect_zip(OUT/'submit.zip')
 save('training-history.json',history)
 save('validation.json',{'status':'STATIC_AND_CPU_PARITY_PASS','sha256':digest,'bytes':(OUT/'submit.zip').stat().st_size,'stage2_checkpoint_sha256':sha(OUT/'best.pt'),'base_zip_sha256':sha(BASE),'changed_entries':changes,'trained_epochs':15,'optimizer_updates':990,'packaged_class_prediction_parity_samples':66,'backbone_tensors_match':True,'zip_crc_pass':True,'gpu_check':'Not run for this new checkpoint','submitted':False,'seconds':time.monotonic()-started})
 save('status.json',{'status':'ZIP_READY','sha256':digest,'seconds':time.monotonic()-started})
 print((OUT/'validation.json').read_text(),flush=True)


if __name__=='__main__':
 try:main()
 except Exception as e:
  if OUT.exists():save('status.json',{'status':'FAILED','error':repr(e)})
  raise
