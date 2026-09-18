"""Package the validated three-seed warmstart_ce logit ensemble."""
import ast,hashlib,importlib.util,json,zipfile
from pathlib import Path
import numpy as np,torch
from daily_submission import inspect_zip

P=Path(__file__).resolve().parents[1]
OUT=P/'artifacts/stage3-warmstart-ce-ensemble-submit-candidate-20260917'
BASE=P/'artifacts/stage3-warmstart-ce-submit-candidate-20260917/submit.zip'
SOURCE=P/'artifacts/stage3-prediction-preservation-20260914/warmstart_ce'
EVAL=P/'artifacts/stage3-civic-535-balanced-evaluation-20260917'
COMPARE=P/'artifacts/stage3-warmstart-ce-ensemble-20260917/report.json'
MODULE=P/'src/stage3_motion_ensemble_inference.py'
SEEDS=[20260910,20260911,20260912]
BASE_SHA='b4b73b2968ecdef39fd923645210ae3e00b8496e06d313be5a587f2a867d8655'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);assert sha(BASE)==BASE_SHA
 review=json.loads(COMPARE.read_text());assert review['status']=='COMPLETE_VALIDATED' and review['decision']=='FOLLOWUP_ONLY' and all(review['gates'].values())
 checkpoints=[torch.load(SOURCE/str(s)/'model.pt',map_location='cpu',weights_only=True) for s in SEEDS]
 for key in ('mean','std','mask'):
  assert all(torch.equal(checkpoints[0][key],c[key]) for c in checkpoints[1:])
 deploy=dict(format='stage3-motion-ensemble-v1',mode='motion',models=[c['model'] for c in checkpoints],mean=checkpoints[0]['mean'],std=checkpoints[0]['std'],mask=checkpoints[0]['mask'],seeds=SEEDS,accel_classes=['ACCELERATING','DECELERATING','CONSTANT','STOPPED'],steer_classes=['LEFT','STRAIGHT','RIGHT'])
 best=OUT/'best.pt';torch.save(deploy,best)
 with zipfile.ZipFile(BASE) as src:
  original=src.read('inference.py').decode('utf-8-sig');tree=ast.parse(original);old=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict_stage3'];assert len(old)>=1
  fn=old[-1];lines=original.splitlines(keepends=True);inference=''.join(lines[:fn.lineno-1]+lines[fn.end_lineno:])+'\n'+MODULE.read_text(encoding='utf-8-sig')
  target=OUT/'submit.zip'
  with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
   for info in src.infolist():dst.writestr(info.filename,inference.encode() if info.filename=='inference.py' else best.read_bytes() if info.filename=='model/stage3/best.pt' else src.read(info.filename))
 with zipfile.ZipFile(BASE) as src,zipfile.ZipFile(target) as dst:
  assert src.namelist()==dst.namelist();changed=[n for n in src.namelist() if src.read(n)!=dst.read(n)];assert changed==['inference.py','model/stage3/best.pt']
 (OUT/'inference.py').write_text(inference,encoding='utf-8');spec=importlib.util.spec_from_file_location('ensemble_candidate',OUT/'inference.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 models,mean,std=module.s3_motion_load(best,torch.device('cpu'));checked=0;maxerr=0.
 for cache in sorted(EVAL.glob('*-features.npz')):
  with np.load(cache,allow_pickle=False) as z:features=z['features']
  actual=module.s3_motion_logits(models,features,mean,std,torch.device('cpu'));expected=[[],[]]
  for ck in checkpoints:
   m=module.S3MotionModel();m.load_state_dict(ck['model'],strict=True);m.eval();x=np.zeros((len(features),1682),np.float32);x[:,1280:]=np.clip((features-mean[1280:])/std[1280:],-10,10)
   with torch.inference_mode():values=m(torch.from_numpy(x))
   for k in range(2):expected[k].append(values[k].numpy())
  for a,parts in zip(actual,expected):
   e=np.mean(parts,axis=0);maxerr=max(maxerr,float(np.max(np.abs(a-e))));np.testing.assert_allclose(a,e,rtol=1e-5,atol=2e-6);np.testing.assert_array_equal(a.argmax(1),e.argmax(1))
  checked+=len(features)
 digest=inspect_zip(target);report=dict(status='STATIC_AND_CPU_PARITY_PASS_GPU_PENDING',candidate_sha256=digest,bytes=target.stat().st_size,base_sha256=sha(BASE),deploy_checkpoint_sha256=sha(best),source_checkpoint_sha256={str(s):sha(SOURCE/str(s)/'model.pt') for s in SEEDS},changed_entries=changed,selection='Fixed equal-logit ensemble of warmstart_ce seeds 20260910/11/12 after all predeclared gates passed',cpu_parity_rows=checked,max_logits_error=maxerr,classification_matches=True,stage1_stage2_requirements_unchanged=True,zip_crc_pass=True,gpu_check='Not run',submitted=False)
 (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
