"""Build a portable full Stage3 training dataset with per-file hashes."""
import argparse,ast,hashlib,io,json,zipfile
from pathlib import Path
import pandas as pd
from stage3_pipeline import load_records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.dataset_dir.resolve();repo=Path(__file__).resolve().parent.parent
    records=load_records(root)
    entries={};manifest=[];counts={}
    for part,rows in records.items():
        labels=pd.read_csv(root/f'labels_{part}_candidate.csv')
        entries[f'dataset/labels_{part}_candidate.csv']=labels.to_csv(index=False).encode()
        for row in rows:
            manifest.append(dict(ID=row['ID'],route=row['route'],split=part,video=f"videos/{row['ID']}.mp4"))
        counts[part]=len(labels)
    entries['dataset/split_manifest.csv']=pd.DataFrame(manifest).to_csv(index=False).encode()
    for file in ('stage3_pipeline.py','test_stage3_pipeline.py'):
        data=(repo/'src'/file).read_bytes();ast.parse(data);entries['src/'+file]=data
    entries['requirements.txt']=(repo/'Baseline'/'requirements.txt').read_bytes()
    entries['dataset/label_version.json']=(root/'label_version.json').read_bytes()
    readme='Stage3 full calibrated v1 dataset. Provisional CAN labels, not official ground truth. Route split retained. Source: https://github.com/commaai/comma2k19 and https://huggingface.co/datasets/commaai/comma2k19 (MIT). Not a submission ZIP.\n'
    entries['README.txt']=readme.encode()
    provenance=dict(kind='STAGE3_FULL_V1',subset_rows=counts,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('split_manifest.csv','labels_train_candidate.csv','labels_validation_candidate.csv')},file_sha256={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()})
    for rows in records.values():
        for row in rows:
            with row['path'].open('rb') as f:
                provenance['file_sha256'][f"dataset/videos/{row['ID']}.mp4"]=hashlib.file_digest(f,'sha256').hexdigest()
    entries['bundle.json']=json.dumps(provenance,indent=2).encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,data in entries.items(): z.writestr(name,data)
        for rows in records.values():
            for row in rows: z.write(row['path'],f"dataset/videos/{row['ID']}.mp4",compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(a.output) as z:
        if z.testzip() is not None: raise ValueError('ZIP CRC check failed')
        ast.parse(z.read('src/stage3_pipeline.py'))
    print(json.dumps(dict(status='PASS',zip=str(a.output.resolve()),bytes=a.output.stat().st_size,subset_rows=counts,submission=False),indent=2))


if __name__=='__main__': main()