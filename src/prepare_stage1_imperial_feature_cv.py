"""Prepare a code-only Kaggle kernel for frozen-feature 5-fold CV."""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/kaggle-stage1-imperial-feature-cv-20260918'; KERNEL='biadis/crashintent-stage1-imperial-feature-cv'
def write(path,value): path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 kernel=OUT/'kernel'; kernel.mkdir(parents=True); (OUT/'result').mkdir()
 runner=(ROOT/'src/stage1_imperial_feature_cv_runner.py').read_text(encoding='utf-8'); ast.parse(runner)
 code="""from pathlib import Path
import shutil,subprocess,sys,tempfile,zipfile
imperial=next(p.parent for p in Path('/kaggle/input').rglob('imperial.bin'))
spatial=next(p.parent for p in Path('/kaggle/input').rglob('resnet18.pth'))
work=Path(tempfile.mkdtemp(dir='/kaggle/working',prefix='imperial-feature-cv-'))
shutil.copyfile(imperial/'imperial.bin',work/'imperial.bin'); shutil.copyfile(spatial/'resnet18.pth',work/'resnet18.pth')
(work/'runner.py').write_text(%r)
subprocess.run([sys.executable,str(work/'runner.py'),str(work)],check=True)
with zipfile.ZipFile('/kaggle/working/stage1-imperial-feature-cv-result.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for path in sorted((work/'feature-cv-result').iterdir()): archive.write(path,path.name)
"""%runner; ast.parse(code)
 notebook={'cells':[{'cell_type':'code','id':'feature-cv','metadata':{},'source':code.splitlines(True),'outputs':[],'execution_count':None}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}; write(kernel/'Stage1_Imperial_Feature_CV.ipynb',notebook)
 write(kernel/'kernel-metadata.json',{'id':KERNEL,'title':'CrashIntent Stage1 Imperial Feature CV','code_file':'Stage1_Imperial_Feature_CV.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':['biadis/crashintent-stage1-imperial-assets','biadis/crashintent-stage1-imperial-spatial-assets'],'competition_sources':[],'kernel_sources':[],'model_sources':[]})
 write(OUT/'bundle.json',{'status':'PREPARED','kernel':KERNEL,'folds':5,'images':100}); write(OUT/'remote-run.json',{'status':'PREPARED','monitor_status':'NOT_STARTED'})
if __name__=='__main__': main()
