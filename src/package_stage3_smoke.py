"""Build a portable two-video GPU smoke bundle. Never creates a submission ZIP."""
import argparse,ast,hashlib,io,json,zipfile
from pathlib import Path
import pandas as pd
from stage3_pipeline import load_records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.dataset_dir.resolve();repo=Path(__file__).resolve().parent.parent
    records=load_records(root);selected={k:v[0] for k,v in records.items()}
    entries={};manifest=[];counts={}
    for part,row in selected.items():
        labels=pd.read_csv(root/f'labels_{part}_candidate.csv')
        labels=labels[labels.ID==row['ID']]
        entries[f'dataset/labels_{part}_candidate.csv']=labels.to_csv(index=False).encode()
        manifest.append(dict(ID=row['ID'],route=row['route'],split=part,video=f"videos/{row['ID']}.mp4"))
        counts[part]=len(labels)
    entries['dataset/split_manifest.csv']=pd.DataFrame(manifest).to_csv(index=False).encode()
    for file in ('stage3_pipeline.py','test_stage3_pipeline.py'):
        data=(repo/'src'/file).read_bytes();ast.parse(data);entries['src/'+file]=data
    entries['requirements.txt']=(repo/'Baseline'/'requirements.txt').read_bytes()
    entries['dataset/label_version.json']=(root/'label_version.json').read_bytes()
    readme='''Stage3 GPU smoke bundle (not a submission and not a training dataset).
Extract to a writable working directory and enable a CUDA GPU.
Use the project's baseline requirements. Restart the notebook kernel after
package changes if NumPy/Pandas ABI errors occur, then run:

python src/test_stage3_pipeline.py
python src/stage3_pipeline.py audit --dataset-dir dataset
python src/stage3_pipeline.py smoke --dataset-dir dataset --output-dir smoke-run --device cuda

Expected: smoke_report.json status PASS, public_cuda_entrypoint_tested true.
Return smoke-run/smoke_report.json and smoke-run/predictions.csv.
The random smoke checkpoint is not a trained submission model.
This bundle contains one full training video and one full validation video.
Source label_version.json describes the full v1 dataset, not this two-video subset.
The full training corpus is not included. No external pretrained weights are used.
'''
    entries['README.txt']=readme.encode()
    provenance=dict(kind='GPU_SMOKE_ONLY',subset_rows=counts,source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('split_manifest.csv','labels_train_candidate.csv','labels_validation_candidate.csv')},file_sha256={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()})
    entries['bundle.json']=json.dumps(provenance,indent=2).encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,data in entries.items(): z.writestr(name,data)
        for row in selected.values(): z.write(row['path'],f"dataset/videos/{row['ID']}.mp4",compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(a.output) as z:
        if z.testzip() is not None: raise ValueError('ZIP CRC check failed')
        ast.parse(z.read('src/stage3_pipeline.py'))
    print(json.dumps(dict(status='PASS',zip=str(a.output.resolve()),bytes=a.output.stat().st_size,subset_rows=counts,submission=False),indent=2))


if __name__=='__main__': main()