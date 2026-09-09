"""Validate trial predictions against the exact source bundle and recompute metrics."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
import pandas as pd
from stage3_pipeline import ACCEL, STEER, classification_metrics


def validate(result, bundle):
    with zipfile.ZipFile(result) as z, zipfile.ZipFile(bundle) as source:
        if len(z.namelist()) != len(set(z.namelist())) or sum(i.file_size for i in z.infolist()) > 20_000_000:
            raise ValueError('Invalid result ZIP size or duplicate members')
        if z.testzip() is not None: raise ValueError('Invalid CRC')
        def read(name): return json.loads(z.read(name))
        def csv(name): return pd.read_csv(io.BytesIO(z.read(name)))
        if read('bundle.json') != json.loads(source.read('bundle.json')): raise ValueError('Bundle mismatch')
        report=read('trial-report.json')
        if report['status']!='PASS' or not math.isfinite(report['seconds']) or report['seconds']<=0: raise ValueError('Invalid report')
        if report['peak_allocated_bytes']<=0 or report['peak_reserved_bytes']<report['peak_allocated_bytes']: raise ValueError('Missing memory evidence')
        for line in source.read('requirements.txt').decode().splitlines():
            if not line.strip() or line.startswith('#'): continue
            name,expected=line.split('==')
            if report['versions'][name].split('+')[0]!=expected: raise ValueError('Package mismatch')
        fingerprints=read('trial-run/data_fingerprints.json')
        for name,value in fingerprints.items():
            if hashlib.sha256(z.read('trial-dataset/'+name)).hexdigest()!=value: raise ValueError('Training input fingerprint mismatch')
        for part in ('train','validation'):
            full=pd.read_csv(io.BytesIO(source.read(f'dataset/labels_{part}_candidate.csv')))
            ids=report['selected_ids'][part]
            if len(ids)!=2 or len(set(ids))!=2: raise ValueError('Unexpected subset')
            expected=full[full.ID.isin(ids)].reset_index(drop=True)
            actual=csv(f'trial-dataset/labels_{part}_candidate.csv')
            pd.testing.assert_frame_equal(expected,actual)
            if report['full_audit'][part]!={'videos':full.ID.nunique(),'samples':len(full)}: raise ValueError('Full audit mismatch')
        truth=csv('trial-dataset/labels_validation_candidate.csv').rename(columns={'frame_index':'sample_index'})
        pred=csv('trial-run/validation_predictions.csv')
        if pred.isna().any().any() or pred.duplicated(['ID','sample_index']).any() or len(pred)!=len(truth): raise ValueError('Invalid prediction rows')
        joined=truth.merge(pred,on=['ID','sample_index'],suffixes=('_true','_pred'),validate='one_to_one')
        if len(joined)!=len(truth): raise ValueError('Missing predictions')
        metrics=read('trial-run/evaluation.json')
        for column,classes,key in [('accel_label',ACCEL,'accel'),('steer_label',STEER,'moving_steer')]:
            rows=joined if key=='accel' else joined[joined.accel_label_true!='STOPPED']
            indexes={v:i for i,v in enumerate(classes)}
            computed=classification_metrics(rows[column+'_true'].map(indexes),rows[column+'_pred'].map(indexes),classes)
            if computed!=metrics[key]: raise ValueError('Recomputed metrics mismatch')
        history=read('trial-run/history.json')
        if len(history)!=1 or history[0]['steps']<=0 or not math.isfinite(history[0]['mean_loss']): raise ValueError('Invalid training history')
        if history[0]['validation']!=metrics: raise ValueError('Checkpoint reload evaluation differs')
    return dict(status='PASS',report=report,history=history,metrics=metrics,prediction_rows=len(pred),metrics_recomputed=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();summary=validate(args.result,args.bundle)
    with args.output.open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
