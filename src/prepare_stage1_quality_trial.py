"""Build the pinned, private Stage1 quality pilot GPU input bundle."""
import ast,json,zipfile
from pathlib import Path
import pandas as pd
import prepare_stage1_quality_balanced as q
from prepare_stage1_robustness import digest,write_json
from evaluate_stage1_robustness import audit_dataset
from stage1_quality_trial_runner import audit_rows,STEPS,LR,SEED
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/kaggle-stage1-quality-trial-20260911'
DATASET='biadis/crashintent-stage1-quality-trial-assets'
KERNEL='biadis/crashintent-stage1-quality-trial'


def main():
    ds=OUT/'dataset';kernel=OUT/'kernel';ds.mkdir(parents=True,exist_ok=False);kernel.mkdir()
    quality=ROOT/'artifacts/stage1-quality-balanced-20260911-lowmem';old=ROOT/'artifacts/stage1-full-20260910';stress=ROOT/'artifacts/stage1-robustness-20260910'
    state=json.loads((quality/'status.json').read_text());plan=json.loads((quality/'plan.json').read_text())
    if state['status']!='DATA_VALIDATED' or state['manifest_sha256']!=digest(quality/'generation_manifest.csv') or state['plan_sha256']!=digest(quality/'plan.json') or plan!=q.make_plan():raise ValueError('Quality data identity changed')
    parts=[]
    for source in plan['sources']:
        record=json.loads((quality/'records'/(source['source_id']+'.json')).read_text())
        if record['identity']!=dict(source,plan_sha256=state['plan_sha256']):raise ValueError('Quality source identity changed')
        q.verify_rows(quality,source,record['rows']);parts.extend(record['rows'])
    generated=pd.read_csv(quality/'generation_manifest.csv',dtype={'source_id':str,'upload_group':str})
    if len(generated)!=510 or set(generated.output_path)!={r['output_path'] for r in parts}:raise ValueError('Quality manifest coverage changed')
    stress_table=audit_dataset(stress)
    existing=pd.read_csv(old/'generation_manifest.csv',dtype={'source_id':str,'upload_group':str})
    if digest(old/'generation_manifest.csv')!=plan['training_manifest_sha256']:raise ValueError('Original training manifest changed')
    selected={r['source_id'] for r in plan['sources']};eval_sources=set(stress_table.source_id)
    rows=[];files={}
    def add(items,base,role,prefix,suite=''):
        for r in items:
            path=base/r['output_path']
            if digest(path)!=r['sha256']:raise ValueError('Video changed')
            name=prefix+'/'+Path(r['output_path']).name
            files[name]=path
            item={k:r[k] for k in ('source_id','upload_group','label','sha256')}
            item.update(path=name,role=role,suite=suite)
            if role=='matched_quality':item['profile']=r['profile']
            else:item['variant']=int(r.get('variant',-1))
            if role=='evaluation':item['condition']=r['condition'] if suite=='stress' else 'normal_'+('original' if int(r['variant'])==0 else 'rerecorded_v'+str(int(r['variant'])))
            rows.append(item)
    add(parts,quality,'matched_quality','quality')
    add(existing[(existing.split=='train')&existing.source_id.isin(selected)].to_dict('records'),old,'existing_recipe','old_train')
    add(stress_table.to_dict('records'),stress,'evaluation','stress','stress')
    add(existing[(existing.split=='val')&existing.source_id.isin(eval_sources)].to_dict('records'),old,'evaluation','normal','normal')
    excluded=pd.read_csv(old/'excluded_sources.csv',dtype=str).source_id.tolist()
    audit_rows(rows,excluded)
    config=dict(scope='Single-seed development pilot; diagnostic sources already examined',seed=SEED,steps=STEPS,lr=LR,effective_batch=8,microbatch=1,rows=rows,excluded=excluded,
                quality_manifest_sha256=state['manifest_sha256'],original_manifest_sha256=plan['training_manifest_sha256'],
                criteria='Both noise and quality_mix original errors decrease; each stress RR condition loses <=1/21; normal Macro-F1 drop <=0.02. Pilot only, no automatic promotion.')
    files['initial.pt']=ROOT/'artifacts/kaggle-stage1-full-20260910/validated/best.pt'
    if digest(files['initial.pt'])!='92b10c992f66eed8d7edcf5c0f266a9017315111bbe4c1bcb8d9e9e9b64267ee':raise ValueError('Initial model changed')
    files['runner.py']=ROOT/'src/stage1_quality_trial_runner.py'
    with zipfile.ZipFile(ROOT/'artifacts/stage1-auto-release-20260910/candidate/submit.zip') as z:
        extras={'serving_inference.py':z.read('inference.py'),'requirements.txt':z.read('requirements.txt'),'plan.json':json.dumps(config,indent=2).encode()}
    import hashlib
    manifest={name:digest(path) for name,path in files.items()}
    manifest.update({name:hashlib.sha256(data).hexdigest() for name,data in extras.items()})
    with zipfile.ZipFile(ds/'quality.bin','x',zipfile.ZIP_STORED) as z:
        for name,path in files.items():z.write(path,name)
        for name,data in extras.items():z.writestr(name,data)
        z.writestr('manifest.json',json.dumps(manifest,indent=2))
    write_json(OUT/'plan.json',config);write_json(OUT/'manifest.json',manifest)
    write_json(ds/'quality-assets.json',dict(sha256=digest(ds/'quality.bin')))
    write_json(ds/'dataset-metadata.json',dict(id=DATASET,title='CrashIntent Stage1 Quality Trial Assets',licenses=[{'name':'other'}],description='Private existing CCD proxy train/validation videos and Stage1 model. Research pilot; not for redistribution.'))
    code='''from pathlib import Path
import hashlib,json,zipfile,tempfile,subprocess,sys
asset=next(Path('/kaggle/input').rglob('quality-assets.json')).parent
with (asset/'quality.bin').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==json.loads((asset/'quality-assets.json').read_text())['sha256']
work=Path(tempfile.mkdtemp(dir='/kaggle/working',prefix='quality-'))
with zipfile.ZipFile(asset/'quality.bin') as z:
 for name in z.namelist(): assert (work/name).resolve().is_relative_to(work.resolve())
 z.extractall(work)
venv=Path(tempfile.mkdtemp(dir='/tmp',prefix='pinned-'))
subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],check=True)
python=str(venv/'bin/python')
try:
 with (work/'install.log').open('w') as log:
  subprocess.run([sys.executable,'-m','pip','--python',python,'install','--no-cache-dir','-r',str(work/'requirements.txt')],stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
 with (work/'run.log').open('w') as log:
  subprocess.run([python,str(work/'runner.py'),str(work)],stdout=log,stderr=subprocess.STDOUT,timeout=10200,check=True)
finally:
 with zipfile.ZipFile('/kaggle/working/stage1-quality-result.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in ['manifest.json','runner.py','install.log','run.log']:
   if (work/name).exists(): z.write(work/name,name)
  if (work/'result').exists():
   for path in (work/'result').iterdir(): z.write(path,'result/'+path.name)
'''
    ast.parse(code);ast.parse(files['runner.py'].read_text(encoding='utf-8-sig'))
    nb=dict(cells=[dict(cell_type='code',metadata={},source=code.splitlines(keepends=True),outputs=[],execution_count=None)],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},nbformat=4,nbformat_minor=5)
    write_json(kernel/'Stage1_Quality.ipynb',nb)
    write_json(kernel/'kernel-metadata.json',dict(id=KERNEL,title='CrashIntent Stage1 Quality Trial',code_file='Stage1_Quality.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=[DATASET],competition_sources=[],kernel_sources=[],model_sources=[]))
    report=dict(status='PREPARED_NOT_UPLOADED',bytes=(ds/'quality.bin').stat().st_size,sha256=digest(ds/'quality.bin'),dataset=DATASET,kernel=KERNEL,training_videos=765,evaluation_videos=420,arms=3,optimizer_updates=256,max_seconds=10800)
    write_json(OUT/'bundle.json',report);print(json.dumps(report))

if __name__=='__main__':main()
