"""Prepare a private Kaggle GPU diagnostic bundle for Imperial images."""
import ast,hashlib,io,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/kaggle-stage1-imperial-20260917'
DATASET='biadis/crashintent-stage1-imperial-assets';KERNEL='biadis/crashintent-stage1-imperial-diagnostic'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def main():
 if OUT.exists():raise FileExistsError(OUT)
 ds=OUT/'dataset';kernel=OUT/'kernel';ds.mkdir(parents=True);kernel.mkdir();(OUT/'result').mkdir()
 # Kaggle expands files ending in .zip when mounting a dataset. Preserve the
 # byte-exact source archive under a neutral extension for SHA/CRC validation.
 shutil.copyfile(ROOT/'data_raw/imperial-recapture/SubjectiveTestImages.clean.zip',ds/'imperial.bin')
 shutil.copyfile(ROOT/'artifacts/kaggle-stage1-quarter-mix-trial-20260916/validated/screen_mix_25.pt',ds/'screen.pt')
 with zipfile.ZipFile(ROOT/'artifacts/stage1-auto-release-20260910/candidate/submit.zip') as z:(ds/'official.pt').write_bytes(z.read('model/stage1/best.pt'))
 runner=(ROOT/'src/stage1_imperial_gpu_runner.py').read_text(encoding='utf-8');ast.parse(runner);(ds/'runner.py').write_text(runner,encoding='utf-8')
 manifest={p.name:sha(p) for p in (ds/'imperial.bin',ds/'official.pt',ds/'screen.pt',ds/'runner.py')};write(ds/'manifest.json',manifest)
 write(ds/'dataset-metadata.json',{'id':DATASET,'title':'CrashIntent Stage1 Imperial Diagnostic Assets','licenses':[{'name':'other'}],'description':'Private diagnostic bundle; public Imperial sample and frozen checkpoints; no redistribution.'})
 code="""from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tempfile,zipfile
asset=next(Path('/kaggle/input').rglob('manifest.json')).parent
work=Path(tempfile.mkdtemp(dir='/kaggle/working',prefix='imperial-'))
manifest=json.loads((asset/'manifest.json').read_text(encoding='utf-8-sig'))
for name,value in manifest.items():
 p=asset/name
 with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==value
 shutil.copyfile(p,work/name)
shutil.copyfile(asset/'manifest.json',work/'manifest.json')
subprocess.run([sys.executable,str(work/'runner.py'),str(work)],check=True)
with zipfile.ZipFile('/kaggle/working/stage1-imperial-result.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((work/'result').iterdir()): z.write(p,p.name)
""";ast.parse(code)
 nb={'cells':[{'cell_type':'code','metadata':{},'source':code.splitlines(True),'outputs':[],'execution_count':None}], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5};write(kernel/'Stage1_Imperial.ipynb',nb)
 write(kernel/'kernel-metadata.json',{'id':KERNEL,'title':'CrashIntent Stage1 Imperial Diagnostic','code_file':'Stage1_Imperial.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4','dataset_sources':[DATASET],'competition_sources':[],'kernel_sources':[],'model_sources':[]})
 bundle={'status':'PREPARED','dataset':DATASET,'kernel':KERNEL,'bytes':sum(p.stat().st_size for p in ds.iterdir()),'manifest':manifest,'notebook_sha256':sha(kernel/'Stage1_Imperial.ipynb'),'metadata_sha256':sha(kernel/'kernel-metadata.json')};write(OUT/'bundle.json',bundle);write(OUT/'remote-run.json',{'status':'PREPARED','monitor_status':'NOT_STARTED','dataset':DATASET,'kernel':KERNEL});print(json.dumps(bundle))
if __name__=='__main__':main()
