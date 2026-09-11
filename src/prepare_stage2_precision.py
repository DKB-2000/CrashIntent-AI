import ast,hashlib,json,zipfile
from pathlib import Path
import pandas as pd
ROOT=Path.cwd();OUT=ROOT/'artifacts/kaggle-stage2-precision-20260911';ds=OUT/'dataset';kernel=OUT/'kernel'
ds.mkdir(parents=True,exist_ok=False);kernel.mkdir()
files={}
source=ROOT/'artifacts/stage2-pretrained-submit-candidate-20260911'
files['labels.csv']=(source/'training-labels.csv').read_bytes()
files['new.pt']=(source/'best.pt').read_bytes()
files['old.pt']=(ROOT/'artifacts/stage2-controls-20260909/soft_mixed_scene/best.pt').read_bytes()
files['features.pt']=(ROOT/'artifacts/stage2-controls-20260909/features.pt').read_bytes()
files['runner.py']=(ROOT/'src/stage2_precision_runner.py').read_bytes()
with zipfile.ZipFile(ROOT/'artifacts/final-stage2-short-stage3-20260911/submit.zip') as z:
 for name in ['inference.py','requirements.txt']:files[name]=z.read(name)
 files['resnet.pt']=z.read('model/stage2/resnet18-f37072fd.pth')
rows=pd.read_csv(source/'training-labels.csv',dtype={'ID':str})
for sid in rows.ID:files[f'videos/{sid}.mp4']=(ROOT/f'data_raw/ccd/videos/Crash-1500/{sid}.mp4').read_bytes()
manifest={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
files['manifest.json']=json.dumps(manifest,indent=2).encode()
with zipfile.ZipFile(ds/'precision.bin','w',zipfile.ZIP_DEFLATED) as z:
 for name,data in files.items():z.writestr(name,data)
digest=hashlib.sha256((ds/'precision.bin').read_bytes()).hexdigest()
(ds/'precision-assets.json').write_text(json.dumps({'sha256':digest}))
(ds/'dataset-metadata.json').write_text(json.dumps({'id':'biadis/crashintent-stage2-precision-assets','title':'CrashIntent Stage2 Precision Assets','licenses':[{'name':'other'}],'description':'Private existing CCD videos, trained models and inference-only diagnostic. Not for redistribution.'},indent=2))
code='''from pathlib import Path
import hashlib,json,zipfile,tempfile,subprocess,sys,time
asset=next(Path('/kaggle/input').rglob('precision-assets.json')).parent
assert hashlib.sha256((asset/'precision.bin').read_bytes()).hexdigest()==json.loads((asset/'precision-assets.json').read_text())['sha256']
work=Path(tempfile.mkdtemp(dir='/kaggle/working',prefix='precision-'))
with zipfile.ZipFile(asset/'precision.bin') as z:
 for name in z.namelist():assert (work/name).resolve().is_relative_to(work.resolve())
 z.extractall(work)
venv=Path(tempfile.mkdtemp(dir='/tmp',prefix='pinned-'))
subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],check=True)
python=str(venv/'bin/python')
try:
 with (work/'install.log').open('w') as log:
  subprocess.run([sys.executable,'-m','pip','--python',python,'install','--no-cache-dir','-r',str(work/'requirements.txt')],stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
 with (work/'run.log').open('w') as log:
  subprocess.run([python,str(work/'runner.py'),str(work)],stdout=log,stderr=subprocess.STDOUT,timeout=2400,check=True)
finally:
 with zipfile.ZipFile('/kaggle/working/stage2-precision-result.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in ['manifest.json','runner.py','install.log','run.log']:
   if (work/name).exists():z.write(work/name,name)
  if (work/'result').exists():
   for path in (work/'result').iterdir():z.write(path,'result/'+path.name)
'''
ast.parse(code);ast.parse(files['runner.py'].decode('utf-8-sig'))
nb={'cells':[{'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),'outputs':[],'execution_count':None}],'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'nbformat':4,'nbformat_minor':5}
(kernel/'Stage2_Precision.ipynb').write_text(json.dumps(nb,indent=2))
(kernel/'kernel-metadata.json').write_text(json.dumps({'id':'biadis/crashintent-stage2-precision-check','title':'CrashIntent Stage2 Precision Check','code_file':'Stage2_Precision.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['biadis/crashintent-stage2-precision-assets'],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2))
report={'status':'PREPARED_NOT_UPLOADED','bytes':(ds/'precision.bin').stat().st_size,'sha256':digest,'samples':len(rows),'dataset':'biadis/crashintent-stage2-precision-assets','kernel':'biadis/crashintent-stage2-precision-check','max_run_seconds':3000}
(OUT/'bundle.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
