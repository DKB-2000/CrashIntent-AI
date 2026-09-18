"""CPU smoke and independent package checks for Stage2 semantic crop ZIP."""
import ast, hashlib, importlib.util, json, shutil, sys, zipfile
from pathlib import Path
import pandas as pd, torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2-semantic-crop-submit-candidate-20260916'
ZIP=OUT/'submit.zip';BASE=ROOT/'artifacts/stage1-auto-release-20260910/candidate/submit.zip'
WORK=OUT/'cpu-smoke'
def sha(x):return hashlib.sha256(x).hexdigest()

def main():
 if WORK.exists():shutil.rmtree(WORK)
 (WORK/'input/images/SMOKE').mkdir(parents=True);(WORK/'model').mkdir(parents=True)
 source=ROOT/'data/stage2-validation-nexar-20260914/images/NEXAR_00000'
 paths=sorted(source.glob('*.jpg'))[:40]
 if len(paths)!=40:raise RuntimeError('Need 40 smoke frames')
 for p in paths:shutil.copy2(p,WORK/'input/images/SMOKE'/p.name)
 with zipfile.ZipFile(ZIP) as z,zipfile.ZipFile(BASE) as base:
  if z.testzip() is not None:raise RuntimeError('CRC failure')
  code=z.read('inference.py').decode('utf-8-sig');ast.parse(code)
  for name in ('best.pt','resnet18-f37072fd.pth'): (WORK/'model'/name).write_bytes(z.read('model/stage2/'+name))
  if z.read('model/stage1/best.pt')!=base.read('model/stage1/best.pt') or z.read('model/stage3/best.pt')!=base.read('model/stage3/best.pt'):raise RuntimeError('Other stage changed')
  if z.read('requirements.txt')!=base.read('requirements.txt'):raise RuntimeError('Requirements changed')
 (WORK/'inference.py').write_text(code,encoding='utf-8')
 spec=importlib.util.spec_from_file_location('candidate_inference',WORK/'inference.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 module._device=lambda:torch.device('cpu')
 original_loader=module.DataLoader
 module.DataLoader=lambda *args,**kwargs: original_loader(*args,**{**kwargs,'num_workers':0,'pin_memory':False})
 result=module.predict_stage2(WORK/'input',WORK/'model')
 expected=['ID','collision_frame','entry_frame','evasion_space','entry_side']
 if list(result.columns)!=expected or len(result)!=1 or result.iloc[0].ID!='SMOKE':raise RuntimeError('Output contract failure')
 row=result.iloc[0]
 if not (0<=int(row.collision_frame)<40 and 0<=int(row.entry_frame)<40 and int(row.evasion_space) in (0,1) and row.entry_side in ('LEFT','RIGHT')):raise RuntimeError('Invalid prediction')
 checkpoint=torch.load(WORK/'model/best.pt',weights_only=True,map_location='cpu')
 if checkpoint.get('crop_format')!='stage2-semantic-crop-ensemble-v1' or len(checkpoint.get('crop_probes',[]))!=5:raise RuntimeError('Checkpoint contract failure')
 report={'status':'STATIC_AND_CPU_SMOKE_PASS_GPU_PENDING','zip_sha256':sha(ZIP.read_bytes()),'zip_bytes':ZIP.stat().st_size,'smoke_frames':40,'prediction':{k:(int(row[k]) if k not in ('ID','entry_side') else row[k]) for k in expected},'other_stages_byte_identical':True,'requirements_byte_identical':True,'zip_crc_pass':True,'gpu_check':'Not run','submitted':False}
 (OUT/'independent-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
