"""Build a screen-cue GPU pilot with a separate incremental private asset bundle."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import random
import zipfile

import prepare_stage1_screen_cues as cues
from stage1_quality_trial_runner import audit_rows,schedule,training_rows

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/kaggle-stage1-quality-trial-20260911'
OUT=ROOT/'artifacts/kaggle-stage1-screen-trial-20260914'
DATASET='biadis/crashintent-stage1-screen-trial-assets'
KERNEL='biadis/crashintent-stage1-screen-trial'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    old_bundle=json.loads((OLD/'bundle.json').read_text())
    if digest(OLD/'dataset/quality.bin')!=old_bundle['sha256']:raise ValueError('Original bundle changed')
    state=json.loads((cues.OUT/'status.json').read_text());cue_plan=json.loads((cues.OUT/'plan.json').read_text())
    if state['status']!='DATA_VALIDATED' or cue_plan!=cues.make_plan():raise ValueError('Unverified cue data/source plan')
    for filename,key in [('generation_manifest.csv','manifest_sha256'),('training_manifest.csv','training_manifest_sha256'),('plan.json','plan_sha256')]:
        if digest(cues.OUT/filename)!=state[key]:raise ValueError('Cue manifest changed')
    with (cues.OUT/'generation_manifest.csv').open(encoding='utf-8',newline='') as f:generated=list(csv.DictReader(f))
    sources={r['source_id']:r for r in cue_plan['sources']}
    additions={};rows=[]
    for r in generated:
        row=sources[r['source_id']];base=cues.quality.synth._effect_parameters(random.Random(cues.seed_for(r['source_id'])))
        if r['split']!='train' or r['label']!='RERECORDED' or any(r[k]!=row[k] for k in ('upload_group','source_sha256')):raise ValueError('Cue provenance changed')
        if json.loads(r['effects'])!=cues.parameters(base,r['profile'],r['cue']) or int(r['seed'])!=cues.seed_for(r['source_id']) or int(r['crf'])!=24:raise ValueError('Cue recipe changed')
        path=cues.OUT/r['output_path'];name='screen/'+path.name
        if digest(path)!=r['sha256']:raise ValueError('Cue video changed')
        additions[name]=path
        rows.append(dict(source_id=r['source_id'],upload_group=r['upload_group'],label=r['label'],path=name,sha256=r['sha256'],role='screen_cue',suite='',profile=r['profile'],cue=r['cue']))
    (OUT/'dataset').mkdir(parents=True,exist_ok=False);(OUT/'upload').mkdir();(OUT/'kernel').mkdir()
    with zipfile.ZipFile(OLD/'dataset/quality.bin') as source:
        manifest=json.loads(source.read('manifest.json'));plan=json.loads(source.read('plan.json'))
        plan['rows'].extend(rows);plan.update(arms=['baseline','mixed_50','screen_mix_50'],cue_manifest_sha256=state['manifest_sha256'],
            design='Same source/profile/slot schedule, 128 updates each. Replace only the two quality-half RERECORDED inputs per update with cyclic weak/no_geometry/no_flicker cues. ORIGINAL and existing-recipe inputs unchanged.')
        train,val=audit_rows(plan['rows'],plan['excluded'])
        runner=(ROOT/'src/stage1_quality_trial_runner.py').read_text(encoding='utf-8-sig');ast.parse(runner)
        replacements={'runner.py':runner.encode(),'plan.json':json.dumps(plan,indent=2).encode()}
        manifest.update({name:digest(path) for name,path in additions.items()})
        manifest.update({name:hashlib.sha256(data).hexdigest() for name,data in replacements.items()})
        replacements['manifest.json']=json.dumps(manifest,indent=2).encode()
        with zipfile.ZipFile(OUT/'dataset/quality.bin','x',zipfile.ZIP_STORED) as full,zipfile.ZipFile(OUT/'upload/screen.bin','x',zipfile.ZIP_STORED) as delta:
            for name in source.namelist():
                data=replacements.get(name)
                if data is None:
                    data=source.read(name)
                    if hashlib.sha256(data).hexdigest()!=manifest[name]:raise ValueError('Old member changed')
                full.writestr(name,data)
            for name,path in additions.items():
                data=path.read_bytes()
                if hashlib.sha256(data).hexdigest()!=manifest[name]:raise ValueError('Cue changed during packing')
                full.writestr(name,data);delta.writestr(name,data)
            for name,data in replacements.items():delta.writestr(name,data)
    counts={};batches=schedule(sorted(sources))
    for step,entries in enumerate(batches):
        old=training_rows(train,'mixed_50',step,entries);new=training_rows(train,'screen_mix_50',step,entries)
        if sum(a!=b for a,b in zip(old,new))!=2:raise ValueError('Uncontrolled sample change')
        for r,slot in new:
            if r['role']=='screen_cue':counts[r['cue']]=counts.get(r['cue'],0)+1
    write(OUT/'plan.json',plan);write(OUT/'manifest.json',manifest)
    delta_sha=digest(OUT/'upload/screen.bin')
    write(OUT/'upload/screen-assets.json',dict(sha256=delta_sha))
    write(OUT/'upload/dataset-metadata.json',dict(id=DATASET,title='CrashIntent Stage1 Screen Trial Assets',licenses=[{'name':'other'}],description='Private CCD train-only synthetic screen-cue videos and pilot code. No redistribution.'))
    nb=json.loads((OLD/'kernel/Stage1_Quality.ipynb').read_text());code=''.join(nb['cells'][0]['source'])
    needle="venv=Path(tempfile.mkdtemp(dir='/tmp',prefix='pinned-'))"
    insert=f"""assert json.loads((asset/'quality-assets.json').read_text())['sha256']=={old_bundle['sha256']!r}
extra=next(Path('/kaggle/input').rglob('screen-assets.json')).parent
with (extra/'screen.bin').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()=={delta_sha!r}
with zipfile.ZipFile(extra/'screen.bin') as z:
 for name in z.namelist(): assert (work/name).resolve().is_relative_to(work.resolve())
 z.extractall(work)
"""
    if code.count(needle)!=1:raise ValueError('Notebook template mismatch')
    code=code.replace(needle,insert+needle);ast.parse(code);nb['cells'][0]['source']=code.splitlines(keepends=True)
    write(OUT/'kernel/Stage1_Screen.ipynb',nb)
    meta=json.loads((OLD/'kernel/kernel-metadata.json').read_text());meta.update(id=KERNEL,title='CrashIntent Stage1 Screen Trial',code_file='Stage1_Screen.ipynb',dataset_sources=[old_bundle['dataset'],DATASET])
    write(OUT/'kernel/kernel-metadata.json',meta)
    report=dict(status='PREPARED_NOT_UPLOADED',dataset=DATASET,kernel=KERNEL,source_dataset=old_bundle['dataset'],
                sha256=digest(OUT/'dataset/quality.bin'),bytes=(OUT/'dataset/quality.bin').stat().st_size,
                upload_sha256=delta_sha,upload_bytes=(OUT/'upload/screen.bin').stat().st_size,
                notebook_sha256=digest(OUT/'kernel/Stage1_Screen.ipynb'),metadata_sha256=digest(OUT/'kernel/kernel-metadata.json'),
                training_videos=len(train),new_videos=len(additions),evaluation_videos=len(val),prediction_rows=1260,optimizer_updates=256,cue_sample_counts=counts)
    write(OUT/'bundle.json',report);write(OUT/'remote-run.json',dict(status='PREPARED',monitor_status='NOT_STARTED',dataset=DATASET,kernel=KERNEL,gpu_started=False))
    print(json.dumps(report))


if __name__=='__main__':main()
