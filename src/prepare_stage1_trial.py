"""CCD synthetic trial with original-upload grouped splits and phone holdout."""
import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd
import prepare_stage1_rerecorded as synth
from select_stage1_holdout import select_holdout

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--sources',type=int,default=80)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    videos=root/'data_raw/ccd/videos/Crash-1500'
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    groups={}
    for line in (root/'data_raw/ccd/Crash-1500.txt').read_text().splitlines():
        match=re.fullmatch(r'(\d+),\[([^\]]+)\],(\d+),([^,]+),.*',line)
        if not match:raise ValueError('Invalid CCD metadata')
        groups[match[1]]=match[4]
    holdout=select_holdout(videos,out/'phone_holdout.csv',30,'*.mp4',20260903,False)
    phone_groups={groups[s] for s in holdout.source_id}
    excluded=sorted(s for s,g in groups.items() if g in phone_groups)
    pd.DataFrame({'source_id':excluded}).to_csv(out/'excluded_sources.csv',index=False)
    config=argparse.Namespace(input_dir=videos,output_dir=out/'dataset',pattern='*.mp4',exclude_file=out/'excluded_sources.csv',variants=1,validation_fraction=.2,seed=20260903,limit=args.sources,crf_min=20,crf_max=28,overwrite=False)
    report=synth.generate(config)
    manifest=pd.read_csv(out/'dataset/generation_manifest.csv',dtype={'source_id':str})
    manifest['upload_group']=manifest.source_id.map(groups)
    unique=sorted(manifest.upload_group.unique(),key=lambda g:hashlib.sha256(('stage1-group:'+g).encode()).hexdigest())
    if len(unique)<5:raise ValueError('Insufficient independent upload groups')
    val=set(unique[:max(1,round(len(unique)*.2))])
    manifest['split']=manifest.upload_group.map(lambda g:'val' if g in val else 'train')
    manifest.to_csv(out/'dataset/generation_manifest.csv',index=False)
    labels=manifest[['output_path','label','split']].rename(columns={'output_path':'path'})
    labels.to_csv(out/'dataset/labels.csv',index=False)
    for split in ['train','val']:labels[labels.split==split].to_csv(out/f'dataset/labels_{split}.csv',index=False)
    manifest[['source_id','split','upload_group']].drop_duplicates().to_csv(out/'dataset/source_splits.csv',index=False)
    assert manifest.groupby('upload_group').split.nunique().max()==1
    assert not set(manifest.upload_group)&phone_groups
    report.update(split_unit='CCD original upload group',phone_sources=30,excluded_with_siblings=len(excluded),upload_groups=len(unique),splits_by_sample=labels.split.value_counts().to_dict(),splits_by_source=manifest.drop_duplicates('source_id').split.value_counts().to_dict(),real_phone_validation_available=False)
    (out/'dataset/generation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
