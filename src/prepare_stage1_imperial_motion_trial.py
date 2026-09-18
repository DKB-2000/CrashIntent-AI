"""Build a code-only private Kaggle kernel for the Imperial motion trial."""
import ast, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/kaggle-stage1-imperial-motion-20260917'
KERNEL='biadis/crashintent-stage1-imperial-motion-trial'; SOURCE='biadis/crashintent-stage1-imperial-assets'
def write(path,value): path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    if OUT.exists(): raise FileExistsError(OUT)
    kernel=OUT/'kernel'; kernel.mkdir(parents=True); (OUT/'result').mkdir()
    runner=(ROOT/'src/stage1_imperial_motion_trial_runner.py').read_text(encoding='utf-8'); ast.parse(runner)
    code="""from pathlib import Path
import hashlib,shutil,subprocess,sys,tempfile,zipfile
asset=next(p.parent for p in Path('/kaggle/input').rglob('imperial.bin'))
work=Path(tempfile.mkdtemp(dir='/kaggle/working',prefix='imperial-motion-'))
for name in ('imperial.bin','official.pt'):
 shutil.copyfile(asset/name,work/name)
runner=%r
(work/'runner.py').write_text(runner)
subprocess.run([sys.executable,str(work/'runner.py'),str(work)],check=True)
with zipfile.ZipFile('/kaggle/working/stage1-imperial-motion-result.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for path in sorted((work/'motion-result').iterdir()): archive.write(path,path.name)
"""%runner
    ast.parse(code)
    notebook={'cells':[{'cell_type':'code','id':'imperial-motion','metadata':{},'source':code.splitlines(True),'outputs':[],'execution_count':None}], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    write(kernel/'Stage1_Imperial_Motion.ipynb',notebook)
    write(kernel/'kernel-metadata.json',{'id':KERNEL,'title':'CrashIntent Stage1 Imperial Motion Trial','code_file':'Stage1_Imperial_Motion.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[SOURCE],'competition_sources':[],'kernel_sources':[],'model_sources':[]})
    write(OUT/'bundle.json',{'status':'PREPARED','kernel':KERNEL,'source_dataset':SOURCE,'runner_sha256':hashlib.sha256(runner.encode()).hexdigest(),'notebook_sha256':sha(kernel/'Stage1_Imperial_Motion.ipynb')})
    write(OUT/'remote-run.json',{'status':'PREPARED','monitor_status':'NOT_STARTED','kernel':KERNEL})
if __name__=='__main__': main()
