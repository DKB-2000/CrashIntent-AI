"""Prepare private VDMoire clip dataset and diagnostic kernel."""
import ast,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/kaggle-stage1-vdmoire-20260918'; DATASET='biadis/crashintent-stage1-vdmoire-clips'; KERNEL='biadis/crashintent-stage1-vdmoire-diagnostic'
def write(path,value): path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 ds=OUT/'dataset'; kernel=OUT/'kernel'; ds.mkdir(parents=True); kernel.mkdir(); (OUT/'result').mkdir()
 with zipfile.ZipFile(ds/'clips.bin','x',zipfile.ZIP_STORED) as archive:
  for path in sorted((ROOT/'artifacts/stage1-vdmoire-pilot-20260918').glob('*.mp4')): archive.write(path,path.name)
 write(ds/'dataset-metadata.json',{'id':DATASET,'title':'CrashIntent VDMoire Diagnostic Clips','licenses':[{'name':'other'}],'description':'Private diagnostic subset recovered from official CVMI-Lab Dropbox; no redistribution.'})
 runner=(ROOT/'src/stage1_vdmoire_gpu_runner.py').read_text(encoding='utf-8'); ast.parse(runner)
 code="""from pathlib import Path
import shutil,subprocess,sys,tempfile,zipfile
clips=next(p.parent for p in Path('/kaggle/input').rglob('clips.bin')); assets=next(p.parent for p in Path('/kaggle/input').rglob('official.pt')); work=Path(tempfile.mkdtemp(dir='/kaggle/working',prefix='vdmoire-'))
for source,name in ((clips/'clips.bin','clips.bin'),(assets/'official.pt','official.pt'),(assets/'screen.pt','screen.pt')): shutil.copyfile(source,work/name)
(work/'runner.py').write_text(%r); subprocess.run([sys.executable,str(work/'runner.py'),str(work)],check=True)
with zipfile.ZipFile('/kaggle/working/stage1-vdmoire-result.zip','w',zipfile.ZIP_DEFLATED) as archive:
 for path in sorted((work/'result').iterdir()): archive.write(path,path.name)
"""%runner; ast.parse(code)
 notebook={'cells':[{'cell_type':'code','id':'vdmoire','metadata':{},'source':code.splitlines(True),'outputs':[],'execution_count':None}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}; write(kernel/'Stage1_VDMoire.ipynb',notebook)
 write(kernel/'kernel-metadata.json',{'id':KERNEL,'title':'CrashIntent Stage1 VDMoire Diagnostic','code_file':'Stage1_VDMoire.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[DATASET,'biadis/crashintent-stage1-imperial-assets'],'competition_sources':[],'kernel_sources':[],'model_sources':[]})
 write(OUT/'bundle.json',{'status':'PREPARED','clips':8,'bytes':(ds/'clips.bin').stat().st_size,'sha256':hashlib.sha256((ds/'clips.bin').read_bytes()).hexdigest()}); write(OUT/'remote-run.json',{'status':'PREPARED','monitor_status':'NOT_STARTED'})
if __name__=='__main__': main()
