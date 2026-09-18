"""Prepare private assets and Kaggle kernel for the Imperial spatial pilot."""
import ast,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/kaggle-stage1-imperial-spatial-20260917'; DATASET='biadis/crashintent-stage1-imperial-spatial-assets'; KERNEL='biadis/crashintent-stage1-imperial-spatial-trial'
def write(path,value): path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 ds=OUT/'dataset'; kernel=OUT/'kernel'; ds.mkdir(parents=True); kernel.mkdir(); (OUT/'result').mkdir()
 with zipfile.ZipFile(ROOT/'artifacts/stage1-auto-release-20260910/candidate/submit.zip') as z: (ds/'resnet18.pth').write_bytes(z.read('model/stage2/resnet18-f37072fd.pth'))
 write(ds/'dataset-metadata.json',{'id':DATASET,'title':'CrashIntent Imperial Spatial Pilot Assets','licenses':[{'name':'other'}],'description':'Private pretrained ResNet18 weight already shipped in the submission.'})
 runner=(ROOT/'src/stage1_imperial_spatial_trial_runner.py').read_text(encoding='utf-8'); ast.parse(runner)
 split=(ROOT/'artifacts/kaggle-stage1-imperial-motion-20260917/validated/split.json').read_text(encoding='utf-8')
 code="""from pathlib import Path
import shutil,subprocess,sys,tempfile,zipfile
imperial=next(p.parent for p in Path('/kaggle/input').rglob('imperial.bin'))
spatial=next(p.parent for p in Path('/kaggle/input').rglob('resnet18.pth'))
work=Path(tempfile.mkdtemp(dir='/kaggle/working',prefix='imperial-spatial-'))
for source,name in ((imperial/'imperial.bin','imperial.bin'),(spatial/'resnet18.pth','resnet18.pth')): shutil.copyfile(source,work/name)
(work/'split.json').write_text(%r); (work/'runner.py').write_text(%r)
subprocess.run([sys.executable,str(work/'runner.py'),str(work)],check=True)
with zipfile.ZipFile('/kaggle/working/stage1-imperial-spatial-result.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for path in sorted((work/'spatial-result').iterdir()): archive.write(path,path.name)
"""%(split,runner); ast.parse(code)
 notebook={'cells':[{'cell_type':'code','id':'imperial-spatial','metadata':{},'source':code.splitlines(True),'outputs':[],'execution_count':None}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}; write(kernel/'Stage1_Imperial_Spatial.ipynb',notebook)
 write(kernel/'kernel-metadata.json',{'id':KERNEL,'title':'CrashIntent Stage1 Imperial Spatial Trial','code_file':'Stage1_Imperial_Spatial.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':['biadis/crashintent-stage1-imperial-assets',DATASET],'competition_sources':[],'kernel_sources':[],'model_sources':[]})
 write(OUT/'bundle.json',{'status':'PREPARED','dataset':DATASET,'kernel':KERNEL,'resnet_sha256':hashlib.sha256((ds/'resnet18.pth').read_bytes()).hexdigest()}); write(OUT/'remote-run.json',{'status':'PREPARED','monitor_status':'NOT_STARTED'})
if __name__=='__main__': main()
