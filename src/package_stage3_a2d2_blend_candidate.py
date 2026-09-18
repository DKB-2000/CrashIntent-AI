"""Package the validated 50:50 original/A2D2 steer-logit blend."""
import hashlib,importlib.util,json,zipfile
from pathlib import Path
import numpy as np,torch
from daily_submission import inspect_zip
P=Path(__file__).resolve().parents[1];OUT=P/'artifacts/stage3-a2d2-blend-submit-candidate-20260918';BASE=P/'artifacts/stage2-nexar-temporal-submit-candidate-v2-20260917/submit.zip';OLD=P/'artifacts/stage3-warmstart-ce-ensemble-submit-candidate-20260917/best.pt';NEW=P/'artifacts/stage3-a2d2-ensemble-rehearsal-20260918/mix50.pt';REVIEW=P/'artifacts/stage3-a2d2-logit-blend-20260918/report.json';DATA=P/'artifacts/stage3-a2d2-clip-pilot-20260917'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True);review=json.loads(REVIEW.read_text());assert review['decision']=='ADOPT_BLEND' and review['selected_adapted_weight']==.5
 old=torch.load(OLD,map_location='cpu',weights_only=True);new=torch.load(NEW,map_location='cpu',weights_only=True);assert old['format']==new['format']=='stage3-motion-ensemble-v1' and old['seeds']==new['seeds']
 blend=dict(old);blend['models']=[]
 for a,b in zip(old['models'],new['models']):
  assert a.keys()==b.keys();state={}
  for key in a:
   if key.startswith('steer.') :state[key]=(a[key]+b[key])/2
   else:assert torch.equal(a[key],b[key]);state[key]=a[key]
  blend['models'].append(state)
 best=OUT/'best.pt';torch.save(blend,best)
 with zipfile.ZipFile(BASE) as src:
  target=OUT/'submit.zip'
  with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
   for info in src.infolist():dst.writestr(info.filename,best.read_bytes() if info.filename=='model/stage3/best.pt' else src.read(info.filename))
 with zipfile.ZipFile(BASE) as a,zipfile.ZipFile(target) as b:
  assert a.namelist()==b.namelist();changed=[n for n in a.namelist() if a.read(n)!=b.read(n)];assert changed==['model/stage3/best.pt']
 inference=OUT/'inference.py'
 with zipfile.ZipFile(target) as z:inference.write_bytes(z.read('inference.py'))
 spec=importlib.util.spec_from_file_location('candidate',inference);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);models,mean,std=module.s3_motion_load(best,torch.device('cpu'))
 old_models,_,_=module.s3_motion_load(OLD,torch.device('cpu'));new_models,_,_=module.s3_motion_load(NEW,torch.device('cpu'));maxerr=0.;rows=0
 for video in sorted((DATA/'videos').glob('*.mp4')):
  f=module.s3_motion_features(video)[15:16];actual=module.s3_motion_logits(models,f,mean,std,torch.device('cpu'))[1];x=np.zeros((1,1682),np.float32);x[:,1280:]=np.clip((f-mean[1280:])/std[1280:],-10,10);t=torch.from_numpy(x)
  with torch.inference_mode():expect=(torch.stack([m(t)[1] for m in old_models]).mean(0)+torch.stack([m(t)[1] for m in new_models]).mean(0))/2
  error=float(np.max(np.abs(actual-expect.numpy())));maxerr=max(maxerr,error);np.testing.assert_allclose(actual,expect.numpy(),rtol=1e-5,atol=2e-6);rows+=1
 digest=inspect_zip(target);report=dict(status='ZIP_READY_CPU_VALIDATED_GPU_PENDING',candidate_sha256=digest,bytes=target.stat().st_size,base_sha256=sha(BASE),checkpoint_sha256=sha(best),old_checkpoint_sha256=sha(OLD),adapted_checkpoint_sha256=sha(NEW),changed_entries=changed,cpu_parity_rows=rows,max_logits_error=maxerr,review_sha256=sha(REVIEW),submitted=False)
 (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
