"""Prepare a private Kaggle RGB+GRU Stage3 experiment and bounded watcher."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parents[1];OUT=P/'artifacts/kaggle-stage3-rgb-gru-20260918';DS=OUT/'dataset';K=OUT/'kernel'
def main():
 DS.mkdir(parents=True,exist_ok=False);K.mkdir();src=P/'src/stage3_rgb_gru_gpu_runner.py';shutil.copy2(src,DS/src.name)
 (DS/'dataset-metadata.json').write_text(json.dumps({'id':'biadis/crashintent-stage3-rgb-gru-assets','title':'CrashIntent Stage3 RGB GRU Assets','licenses':[{'name':'other'}],'description':'Project training code only; no evaluation data.'},indent=2))
 code="""from pathlib import Path
import subprocess,sys,zipfile,json
work=Path('/kaggle/working/stage3-rgb-gru');work.mkdir()
archives=list(Path('/kaggle/input').rglob('stage3-training-v1.zip'))
directories=[p.parent for p in Path('/kaggle/input').rglob('split_manifest.csv') if (p.parent/'labels_train_candidate.csv').is_file() and (p.parent/'videos').is_dir()]
assert len(archives)+len(directories)==1,(archives,directories)
if archives:
 with zipfile.ZipFile(archives[0]) as z:z.extractall(work)
 dataset=work/'dataset'
else:
 dataset=directories[0]
runner=list(Path('/kaggle/input').rglob('stage3_rgb_gru_gpu_runner.py'));assert len(runner)==1,runner
subprocess.run([sys.executable,'-u',str(runner[0]),'--dataset',str(dataset),'--output',str(work/'result'),'--epochs','10','--max-seconds','18000'],check=True)
with zipfile.ZipFile('/kaggle/working/stage3-rgb-gru-result.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['report.json','history.json','status.json','best.pt']:
  z.write(work/'result'/name,name)
print(json.loads((work/'result/report.json').read_text())['best_joint_mean'])
"""
 nb={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[x+'\n' for x in code.splitlines()]}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
 (K/'Stage3_RGB_GRU.ipynb').write_text(json.dumps(nb,indent=2));(K/'kernel-metadata.json').write_text(json.dumps({'id':'biadis/crashintent-stage3-rgb-gru','title':'CrashIntent Stage3 RGB GRU','code_file':'Stage3_RGB_GRU.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':['biadis/crashintent-stage3-training-v1','biadis/crashintent-stage3-rgb-gru-assets'],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2))
 report={'status':'PREPARED_NOT_LAUNCHED','runner_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'kernel':'biadis/crashintent-stage3-rgb-gru','dataset':'biadis/crashintent-stage3-rgb-gru-assets'};(OUT/'prepare.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
