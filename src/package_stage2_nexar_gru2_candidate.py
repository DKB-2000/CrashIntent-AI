"""Replace only the deployed 10 Hz Stage2 temporal state in official submission 91775."""
from __future__ import annotations
import csv,hashlib,io,json,statistics,zipfile
from pathlib import Path
import torch
import stage2_pipeline as p
from daily_submission import inspect_zip
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stage2-nexar-temporal-submit-candidate-v2-20260917/submit.zip'
CAND=ROOT/'artifacts/stage2-nexar-gru2-20260918/candidate.pt'
REPORT=ROOT/'artifacts/stage2-nexar-gru2-20260918/report.json'
FEATURES=ROOT/'artifacts/stage2-unlabeled-comparison-20260914/features'
MANIFEST=ROOT/'data/stage2-validation-nexar-20260914/manifest.json';META=ROOT/'data_raw/stage2-validation-nexar-20260914/review-inventory-resolved.csv'
OUT=ROOT/'artifacts/stage2-nexar-gru2-submit-candidate-20260918'
BASE_SHA='7a6c6ff418a5dcfd33b24dab5ef3ec47d00b984ac2361aa4947967444c9f7989';CAND_SHA='b0927a8f2691e5f6c3ee79a85c9ceb3e57705e81005dad7ebbaae37abe7d1af8'
def shab(x):return hashlib.sha256(x).hexdigest()
def sha(path):return shab(Path(path).read_bytes())
def read(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def metrics(err):
 a=[abs(x) for x in err];return {'mean_abs_seconds':sum(a)/len(a),'median_abs_seconds':statistics.median(a),'mean_signed_seconds':sum(err)/len(err),**{f'within_{x}s':sum(v<=x+1e-12 for v in a)/len(a) for x in (.3,.5,1,2,5)}}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();torch.set_num_threads(1)
 if sha(BASE)!=BASE_SHA or sha(CAND)!=CAND_SHA:raise RuntimeError('Frozen input hash mismatch')
 report=json.loads(REPORT.read_text());
 if report['status']!='COMPLETE_VALIDATED' or report['decision']!='PROMOTE_FOR_INTEGRATION' or not all(report['gates'].values()):raise RuntimeError('GRU2 promotion gate failed')
 with zipfile.ZipFile(BASE) as z:
  if z.testzip() is not None:raise RuntimeError('Base CRC failed')
  old_bytes=z.read('model/stage2/best.pt');base_ck=torch.load(io.BytesIO(old_bytes),map_location='cpu',weights_only=True);inference=z.read('inference.py')
 candidate=torch.load(CAND,map_location='cpu',weights_only=True);old=base_ck['temporal_10hz_model']
 if set(candidate)!=set(old):raise RuntimeError('State schema mismatch')
 changed_keys=[k for k in candidate if not torch.equal(candidate[k],old[k])]
 expected=[k for k in old if (k.startswith('r.') and '_l1' in k) or k.startswith('tc.') or k.startswith('te.')]
 if changed_keys!=expected:raise RuntimeError(f'Unexpected state changes: {changed_keys} expected {expected}')
 deploy=dict(base_ck);deploy['temporal_10hz_model']=candidate;deploy['temporal_format']='nexar-soft-heads-10hz-v1';deploy['temporal_adaptation']='gru2-soft-v1';deploy['temporal_source_sha256']=CAND_SHA;deploy['temporal_parent_submission_id']=91775;best=OUT/'best.pt';torch.save(deploy,best)
 target=OUT/'submit.zip'
 with zipfile.ZipFile(BASE) as src,zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
  for info in src.infolist():dst.writestr(info.filename,best.read_bytes() if info.filename=='model/stage2/best.pt' else src.read(info.filename))
 with zipfile.ZipFile(BASE) as src,zipfile.ZipFile(target) as dst:
  if dst.testzip() is not None or src.namelist()!=dst.namelist():raise RuntimeError('Candidate CRC/structure failure')
  changed=[n for n in src.namelist() if src.read(n)!=dst.read(n)]
  if changed!=['model/stage2/best.pt']:raise RuntimeError(f'Unexpected ZIP changes: {changed}')
  if dst.read('inference.py')!=inference:raise RuntimeError('Inference changed')
  preserved={n:shab(dst.read(n)) for n in dst.namelist() if n!='model/stage2/best.pt'}
 # Exact replay of deployed stride-3 candidate on frozen50.
 model=p.Stage2Temporal();model.load_state_dict(candidate);model.eval();meta={r['ID']:r for r in read(META)};ce=[];ee=[]
 with torch.inference_mode():
  for v in json.loads(MANIFEST.read_text())['videos']:
   x=torch.load(FEATURES/(v['ID']+'.pt'),map_location='cpu',weights_only=True).float()[::3];ci,ei,_=model(x[None]);fm=read(ROOT/'data/stage2-validation-nexar-20260914'/v['frame_map']);times=torch.tensor([float(r['timestamp_seconds']) for r in fm]);m=meta[v['ID']];ce.append(float(times[min(int(ci)*3,len(times)-1)]-float(m['reference_event_seconds'])));ee.append(float(times[min(int(ei)*3,len(times)-1)]-float(m['reference_alert_seconds'])))
 replay={'collision_vs_event':metrics(ce),'entry_vs_alert_proxy':metrics(ee),'selection_mae_sum':sum(abs(x) for x in ce+ee)/50};expected_report=report['frozen50']['gru2_candidate']
 for group in ('collision_vs_event','entry_vs_alert_proxy'):
  for k,v in expected_report[group].items():
   if abs(replay[group][k]-v)>2e-5:raise RuntimeError(f'Parity mismatch {group}/{k}')
 if abs(replay['selection_mae_sum']-expected_report['selection_mae_sum'])>2e-5:raise RuntimeError('MAE sum parity mismatch')
 digest=inspect_zip(target);validation={'status':'STATIC_AND_CPU_PARITY_PASS_GPU_PENDING','candidate_sha256':digest,'bytes':target.stat().st_size,'base_sha256':BASE_SHA,'base_submission_id':91775,'changed_entries':changed,'temporal_changed_state_keys':changed_keys,'deploy_checkpoint_sha256':sha(best),'source_candidate_sha256':CAND_SHA,'frozen50_deployed_stride3':replay,'frozen50_report_parity':True,'scene_path':'Byte-preserved inference and unchanged official-best scene_temporal model/crop probes','preserved_entry_sha256':preserved,'zip_crc_pass':True,'gpu_check':'Not run','submitted':False}
 (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
