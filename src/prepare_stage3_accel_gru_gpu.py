"""Prepare the cached-feature accel-only Stage3 Kaggle experiment."""
import json,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[1];A=P/'artifacts/kaggle-stage3-accel-gru-20260918';D=A/'dataset';K=A/'kernel';D.mkdir(parents=True,exist_ok=True);K.mkdir(parents=True,exist_ok=True)
shutil.copy2(P/'src/stage3_accel_gru_gpu_runner.py',D/'stage3_accel_gru_gpu_runner.py')
(D/'dataset-metadata.json').write_text(json.dumps({'id':'biadis/crashintent-stage3-accel-gru-assets','title':'CrashIntent Stage3 Accel GRU Assets','licenses':[{'name':'other'}],'description':'Private training code only.'},indent=2))
code="""from pathlib import Path
import subprocess,sys,zipfile,json
inputs=Path('/kaggle/input');work=Path('/kaggle/working/stage3-accel-gru');work.mkdir()
datasets=[p.parent for p in inputs.rglob('split_manifest.csv') if (p.parent/'labels_train_candidate.csv').is_file()]
caches=[p for p in inputs.rglob('cache') if len(list(p.glob('COMMA2K19_C1_*.npz'))) == 187]
runners=list(inputs.rglob('stage3_accel_gru_gpu_runner.py'))
assert len(datasets)==len(caches)==len(runners)==1,(datasets,caches,runners)
subprocess.run([sys.executable,'-u',str(runners[0]),'--cache',str(caches[0]),'--dataset',str(datasets[0]),'--output',str(work/'result'),'--epochs','12'],check=True)
with zipfile.ZipFile('/kaggle/working/stage3-accel-gru-result.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['report.json','history.json','status.json','best.pt']:z.write(work/'result'/name,name)
print(json.loads((work/'result/report.json').read_text())['ensemble']['macro_f1'])
"""
nb={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code.splitlines(True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
(K/'Stage3_Accel_GRU.ipynb').write_text(json.dumps(nb,indent=2));(K/'kernel-metadata.json').write_text(json.dumps({'id':'biadis/crashintent-stage3-accel-gru','title':'CrashIntent Stage3 Accel GRU','code_file':'Stage3_Accel_GRU.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':['biadis/crashintent-stage3-training-v1','biadis/crashintent-stage3-accel-gru-assets'],'kernel_sources':['biadis/crashintent-stage3-rgb-gru'],'competition_sources':[],'model_sources':[]},indent=2))
print(A)
