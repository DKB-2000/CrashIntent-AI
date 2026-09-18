"""Package Nexar-trained 10 Hz time heads while preserving the official-best scene path."""
from __future__ import annotations
import ast,csv,hashlib,io,json,statistics,zipfile
from pathlib import Path
import torch
import stage2_pipeline as p
from daily_submission import inspect_zip

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stage2-crop-blend-stage3-ensemble-submit-candidate-20260917/submit.zip'
CAND=ROOT/'artifacts/stage2-nexar-temporal-heads-v2-20260917/candidate.pt'
REPORT=ROOT/'artifacts/stage2-nexar-temporal-heads-v2-20260917/report.json'
FEATURES=ROOT/'artifacts/stage2-unlabeled-comparison-20260914/features'
MANIFEST=ROOT/'data/stage2-validation-nexar-20260914/manifest.json'
META=ROOT/'data_raw/stage2-validation-nexar-20260914/review-inventory-resolved.csv'
OUT=ROOT/'artifacts/stage2-nexar-temporal-submit-candidate-v2-20260917'
BASE_SHA='290146445dcd56de78e09e7f30dc2681376af4bfb09ae55f312493fcd4243614'
CAND_SHA='8119a6f3648686d40607ed6e2381f71b2e8afc747b7df6633fb44b1d2581b161'

def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha(p):return sha_bytes(Path(p).read_bytes())
def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def metrics(errors):
 a=[abs(x) for x in errors]
 return {'mean_abs_seconds':sum(a)/len(a),'median_abs_seconds':statistics.median(a),'mean_signed_seconds':sum(errors)/len(errors),**{f'within_{x}s':sum(v<=x+1e-12 for v in a)/len(a) for x in (.3,.5,1,2,5)}}

PREDICT='''def predict_stage2(data_dir,model_dir):
    device=_device();model_dir=Path(model_dir);transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(model_dir/'resnet18-f37072fd.pth',map_location='cpu',weights_only=True));backbone.fc=nn.Identity();backbone.to(device).eval()
    checkpoint=torch.load(model_dir/'best.pt',map_location='cpu',weights_only=True)
    if checkpoint.get('crop_format')!='stage2-semantic-crop-blend-v2' or checkpoint.get('temporal_format')!='nexar-soft-heads-10hz-v1' or checkpoint.get('temporal_stride')!=3:raise ValueError('Stage2 integrated checkpoint contract mismatch')
    scene_temporal=_Stage2Temporal();scene_temporal.load_state_dict(checkpoint['model'],strict=True);scene_temporal.to(device).eval()
    time_temporal=_Stage2Temporal();time_temporal.load_state_dict(checkpoint['temporal_10hz_model'],strict=True);time_temporal.to(device).eval()
    probes=[]
    for state in checkpoint['crop_probes']:
        probe=nn.Linear(2816,4);probe.load_state_dict(state,strict=True);probes.append(probe.to(device).eval())
    task_evasion=[];task_side=[]
    for state in checkpoint['task_evasion_probes']:
        probe=nn.Linear(1792,2);probe.load_state_dict(state,strict=True);task_evasion.append(probe.to(device).eval())
    for state in checkpoint['task_side_probes']:
        probe=nn.Linear(1792,2);probe.load_state_dict(state,strict=True);task_side.append(probe.to(device).eval())
    mean=checkpoint['crop_mean'].to(device);std=checkpoint['crop_std'].to(device)
    if mean.shape!=(2816,) or std.shape!=(2816,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std<=0):raise ValueError('Invalid Stage2 crop normalization')
    rows=[];image_root=Path(data_dir)/'images'
    with torch.inference_mode():
        for folder in sorted(p for p in image_root.iterdir() if p.is_dir()):
            paths=sorted((p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}),key=_frame_number)
            if not paths:continue
            loader=DataLoader(_Stage2Frames(paths,transform),batch_size=256,num_workers=6,pin_memory=True);features=[]
            for images in loader:
                with torch.autocast(device_type='cuda',dtype=torch.float16):features.append(backbone(images.to(device,non_blocking=True)).float().cpu())
            sequence=torch.cat(features)[None].to(device)
            old_cl,old_el,h=scene_temporal.logits(sequence);scene_ci,scene_ei=int(old_cl.argmax(1)),int(old_el.argmax(1))
            new_cl,new_el,_=time_temporal.logits(sequence[:,::3]);ci10,ei10=int(new_cl.argmax(1)),int(new_el.argmax(1));ci=min(ci10*3,len(paths)-1);ei=min(ei10*3,len(paths)-1)
            point=torch.cat((h[0,scene_ci],h[0,scene_ei]));crop_c=_stage2_crop_feature(backbone,paths[scene_ci],transform,device);crop_e=_stage2_crop_feature(backbone,paths[scene_ei],transform,device)
            x=torch.clamp((torch.cat((point,crop_c,crop_e))-mean)/std,-10,10)
            shared=torch.stack([q(x) for q in probes]).mean(0)
            evx=torch.cat((x[:768],x[1280:1792],x[2304:2816]));sx=torch.cat((x[:768],x[768:1280],x[1792:2304]))
            task=torch.cat((torch.stack([q(evx) for q in task_evasion]).mean(0),torch.stack([q(sx) for q in task_side]).mean(0)))
            scene=.5*shared+.5*task;nums=[_frame_number(q) for q in paths]
            rows.append({'ID':folder.name,'collision_frame':nums[ci],'entry_frame':nums[ei],'evasion_space':int(scene[:2].argmax()),'entry_side':'RIGHT' if int(scene[2:].argmax()) else 'LEFT'})
    del backbone,scene_temporal,time_temporal,probes,task_evasion,task_side;torch.cuda.empty_cache()
    return pd.DataFrame(rows,columns=['ID','collision_frame','entry_frame','evasion_space','entry_side'])

'''

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();torch.set_num_threads(1)
 if sha(BASE)!=BASE_SHA or sha(CAND)!=CAND_SHA:raise RuntimeError('Frozen input hash mismatch')
 report=json.loads(REPORT.read_text());
 if report['status']!='COMPLETE_VALIDATED' or report['decision']!='PROMOTE_FOR_OFFICIAL_INTEGRATION_TEST' or not all(report['gates'].values()):raise RuntimeError('Candidate report gate failed')
 with zipfile.ZipFile(BASE) as z:
  inference=z.read('inference.py').decode('utf-8-sig');base_ck=torch.load(io.BytesIO(z.read('model/stage2/best.pt')),map_location='cpu',weights_only=True)
 candidate=torch.load(CAND,map_location='cpu',weights_only=True)
 if set(candidate)!=set(base_ck['model']):raise RuntimeError('Temporal state schema mismatch')
 changed_keys=[]
 for k in candidate:
  if not torch.equal(candidate[k],base_ck['model'][k]):changed_keys.append(k)
 if changed_keys!=['tc.weight','tc.bias','te.weight','te.bias']:raise RuntimeError(f'Unexpected temporal changes: {changed_keys}')
 deploy=dict(base_ck);deploy['temporal_format']='nexar-soft-heads-10hz-v1';deploy['temporal_stride']=3;deploy['temporal_10hz_model']=candidate;deploy['temporal_source_sha256']=CAND_SHA;deploy_path=OUT/'best.pt';torch.save(deploy,deploy_path)
 tree=ast.parse(inference);fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict_stage2']
 if len(fn)!=1:raise RuntimeError('predict_stage2 definition mismatch')
 lines=inference.splitlines(keepends=True);node=fn[0];new_inference=''.join(lines[:node.lineno-1])+PREDICT+''.join(lines[node.end_lineno:]);ast.parse(new_inference)
 target=OUT/'submit.zip'
 with zipfile.ZipFile(BASE) as src,zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
  for info in src.infolist():dst.writestr(info.filename,new_inference.encode() if info.filename=='inference.py' else deploy_path.read_bytes() if info.filename=='model/stage2/best.pt' else src.read(info.filename))
 with zipfile.ZipFile(BASE) as src,zipfile.ZipFile(target) as dst:
  if dst.testzip() is not None or src.namelist()!=dst.namelist():raise RuntimeError('ZIP CRC/structure failure')
  changed=[n for n in src.namelist() if src.read(n)!=dst.read(n)]
  if changed!=['model/stage2/best.pt','inference.py']:raise RuntimeError(f'Unexpected ZIP changes {changed}')
  preserved={n:sha_bytes(dst.read(n)) for n in dst.namelist() if n not in changed}
 # Replay the exact deployed stride-3 time path on frozen50 and match the training report.
 model=p.Stage2Temporal();model.load_state_dict(candidate);model.eval();meta={r['ID']:r for r in read_csv(META)};manifest=json.loads(MANIFEST.read_text())['videos'];ce=[];ee=[]
 with torch.inference_mode():
  for v in manifest:
   x=torch.load(FEATURES/(v['ID']+'.pt'),map_location='cpu',weights_only=True).float()[::3];ci,ei,_=model(x[None]);fm=read_csv(ROOT/'data/stage2-validation-nexar-20260914'/v['frame_map']);times=torch.tensor([float(r['timestamp_seconds']) for r in fm]);m=meta[v['ID']];ce.append(float(times[min(int(ci)*3,len(times)-1)]-float(m['reference_event_seconds'])));ee.append(float(times[min(int(ei)*3,len(times)-1)]-float(m['reference_alert_seconds'])))
 replay={'collision_vs_event':metrics(ce),'entry_vs_alert_proxy':metrics(ee),'selection_mae_sum':sum(abs(x) for x in ce+ee)/50}
 expected=report['frozen50']['candidate_10fps']
 for group in ('collision_vs_event','entry_vs_alert_proxy'):
  for k in expected[group]:
   if abs(replay[group][k]-expected[group][k])>2e-5:raise RuntimeError(f'Frozen50 parity mismatch {group}/{k}: {replay[group][k]} vs {expected[group][k]}')
 if abs(replay['selection_mae_sum']-expected['selection_mae_sum'])>2e-5:raise RuntimeError('Frozen50 sum parity mismatch')
 digest=inspect_zip(target);validation={'status':'STATIC_AND_CPU_PARITY_PASS_GPU_PENDING','candidate_sha256':digest,'bytes':target.stat().st_size,'base_sha256':BASE_SHA,'changed_entries':changed,'temporal_changed_state_keys':changed_keys,'deploy_checkpoint_sha256':sha(deploy_path),'source_candidate_sha256':CAND_SHA,'frozen50_deployed_stride3':replay,'frozen50_report_parity':True,'scene_path':'Preserved official-best full-rate temporal model, time indices, crop probes, and crop frame selection','preserved_entry_sha256':preserved,'zip_crc_pass':True,'gpu_check':'Not run','submitted':False}
 (OUT/'inference.py').write_text(new_inference,encoding='utf-8');(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
