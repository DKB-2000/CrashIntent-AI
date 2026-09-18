"""Prepare a four-arm mixed-data pilot, reusing the existing private GPU dataset."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/kaggle-stage1-quality-trial-20260911'
OUT=ROOT/'artifacts/kaggle-stage1-mixed-trial-20260914'
DATASET='biadis/crashintent-stage1-quality-trial-assets'
KERNEL='biadis/crashintent-stage1-mixed-trial'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    from stage1_quality_trial_runner import audit_rows,ARMS,training_rows,schedule
    baseline=json.loads((OLD/'bundle.json').read_text())
    if digest(OLD/'dataset/quality.bin')!=baseline['sha256']:raise ValueError('Original bundle changed')
    verified=json.loads((OLD/'validated-recovery-20260914/result-validation.json').read_text())
    if verified['status']!='PASS' or verified['bundle_sha256']!=baseline['sha256']:raise ValueError('Original verification missing')
    (OUT/'dataset').mkdir(parents=True,exist_ok=False);(OUT/'kernel').mkdir()
    with zipfile.ZipFile(OLD/'dataset/quality.bin') as source:
        manifest=json.loads(source.read('manifest.json'));plan=json.loads(source.read('plan.json'))
        train,val=audit_rows(plan['rows'],plan['excluded'])
        plan.update(arms=[*ARMS,'mixed_50'],mix_rule='Each update: two original-recipe source pairs and two quality-balanced source pairs; pair parity rotates each step',
                    design='Same 85 sources, seed, source/profile/slot schedule, 128 updates per trained arm; no effect-strength augmentation',
                    prior_result_sha256=verified['result_sha256'])
        runner=(ROOT/'src/stage1_quality_trial_runner.py').read_text(encoding='utf-8-sig')
        ast.parse(runner)
        replacements={'runner.py':runner.encode(),'plan.json':json.dumps(plan,indent=2).encode()}
        for name,data in replacements.items():manifest[name]=hashlib.sha256(data).hexdigest()
        replacements['manifest.json']=json.dumps(manifest,indent=2).encode()
        with zipfile.ZipFile(OUT/'dataset/quality.bin','x',zipfile.ZIP_STORED) as target:
            for name in source.namelist():
                data=replacements.get(name)
                if data is None:
                    data=source.read(name)
                    if hashlib.sha256(data).hexdigest()!=manifest[name]:raise ValueError('Input member changed: '+name)
                target.writestr(name,data)
    batches=schedule(sorted({r['source_id'] for r in train}))
    for step,entries in enumerate(batches):
        selected=training_rows(train,'mixed_50',step,entries)
        if len(selected)!=8 or sum(r['role']=='matched_quality' for r,s in selected)!=4:raise ValueError('Unbalanced recipe')
        if sum(r['label']=='ORIGINAL' for r,s in selected)!=4:raise ValueError('Unbalanced classes')
    write(OUT/'plan.json',plan);write(OUT/'manifest.json',manifest)
    nb=json.loads((OLD/'kernel/Stage1_Quality.ipynb').read_text())
    code=''.join(nb['cells'][0]['source'])
    needle="venv=Path(tempfile.mkdtemp(dir='/tmp',prefix='pinned-'))"
    # Only code/config travel in the notebook. Existing video/model bytes remain in the private dataset.
    overlay={name:data.decode() for name,data in replacements.items()}
    inject=f"assert json.loads((asset/'quality-assets.json').read_text())['sha256']=={baseline['sha256']!r}\n"
    inject+=f"overrides={overlay!r}\nfor name,text in overrides.items(): (work/name).write_text(text,encoding='utf-8')\n"
    if code.count(needle)!=1:raise ValueError('Notebook template changed')
    code=code.replace(needle,inject+needle)
    ast.parse(code);nb['cells'][0]['source']=code.splitlines(keepends=True)
    write(OUT/'kernel/Stage1_Mixed.ipynb',nb)
    meta=json.loads((OLD/'kernel/kernel-metadata.json').read_text())
    meta.update(id=KERNEL,title='CrashIntent Stage1 Mixed Trial',code_file='Stage1_Mixed.ipynb',dataset_sources=[DATASET])
    write(OUT/'kernel/kernel-metadata.json',meta)
    bundle=dict(status='PREPARED_NOT_LAUNCHED',sha256=digest(OUT/'dataset/quality.bin'),bytes=(OUT/'dataset/quality.bin').stat().st_size,
                dataset=DATASET,kernel=KERNEL,source_bundle_sha256=baseline['sha256'],arms=plan['arms'],
                notebook_sha256=digest(OUT/'kernel/Stage1_Mixed.ipynb'),metadata_sha256=digest(OUT/'kernel/kernel-metadata.json'),
                upload_bytes=sum(p.stat().st_size for p in (OUT/'kernel').iterdir()),
                training_videos=765,evaluation_videos=420,prediction_rows=1680,optimizer_updates=384,max_runner_seconds=10200)
    write(OUT/'bundle.json',bundle);write(OUT/'remote-run.json',dict(status='PREPARED',monitor_status='NOT_STARTED',kernel=KERNEL,dataset=DATASET))
    print(json.dumps(bundle))


if __name__=='__main__':main()
