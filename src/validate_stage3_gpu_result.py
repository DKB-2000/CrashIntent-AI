"""Validate a returned GPU smoke report against the exact uploaded bundle."""
import argparse
import io
import json
import math
from pathlib import Path
import zipfile
import pandas as pd

REQUIRED = {'environment.json','execution.json','bundle.json','tests.log','audit.log','smoke.log','smoke-run/smoke_report.json','smoke-run/predictions.csv'}


def require(condition, message):
    if not condition: raise ValueError(message)


def validate(result_path, bundle_path):
    with zipfile.ZipFile(bundle_path) as z:
        expected=json.loads(z.read('bundle.json'))
    with zipfile.ZipFile(result_path) as z:
        names=z.namelist()
        require(len(names)==len(set(names)), 'Duplicate ZIP members')
        require(set(names)==REQUIRED, 'Missing or unexpected result files')
        require(sum(i.file_size for i in z.infolist())<=20_000_000,'Result exceeds 20MB limit')
        require(z.testzip() is None,'Invalid result ZIP CRC')
        def read(name): return json.loads(z.read(name))
        provenance=read('bundle.json')
        require(provenance==expected,'Results do not match the expected input bundle')
        report=read('smoke-run/smoke_report.json'); env=read('environment.json'); execution=read('execution.json')
        require(report.get('status')=='PASS' and report.get('device')=='cuda','Not a successful CUDA run')
        require(report.get('public_cuda_entrypoint_tested') is True,'Public CUDA function not tested')
        require(report.get('checkpoint_reload') is True,'Checkpoint reload not verified')
        require(report.get('architecture')=='mvit_v2_s' and report.get('training_steps')==1,'Unexpected smoke architecture or step count')
        require(math.isfinite(float(report.get('loss',float('nan')))),'Invalid loss')
        require(math.isfinite(float(report.get('seconds',float('nan')))) and report['seconds']>0,'Invalid elapsed time')
        require(env.get('cuda_available') is True and bool(env.get('torch_cuda')) and bool(env.get('gpu_names')),'Missing CUDA environment evidence')
        for stage in ('tests','audit','smoke'):
            require(execution.get(stage,{}).get('exit_code')==0,f'{stage} command failed or missing')
            require(bool(z.read(stage+'.log').strip()),f'{stage} log empty')
        frame=pd.read_csv(io.BytesIO(z.read('smoke-run/predictions.csv')))
        require(list(frame.columns)==['ID','sample_index','accel_label','steer_label'],'Prediction columns mismatch')
        require(not frame.isna().any().any(),'Missing predictions')
        require(report.get('prediction_rows')==4 and len(frame)==4,'Expected four predictions')
        require(frame.ID.tolist()==['SMOKE_S3']*4 and frame.sample_index.tolist()==list(range(4)),'Prediction IDs or indices mismatch')
        require(set(frame.accel_label)<={'ACCELERATING','DECELERATING','CONSTANT','STOPPED'},'Invalid acceleration class')
        require(set(frame.steer_label)<={'LEFT','STRAIGHT','RIGHT'},'Invalid steering class')
    return dict(status='PASS',scope='Returned GPU smoke evidence; not full training or performance validation',gpu_names=env['gpu_names'],torch=env.get('torch'),torchvision=env.get('torchvision'),seconds=report['seconds'],prediction_rows=4,bundle_match=True,public_cuda_entrypoint_tested=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args(); summary=validate(a.result,a.bundle)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()