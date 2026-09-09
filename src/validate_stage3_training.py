"""Validate full-training results and recompute validation metrics and baselines."""
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
        report=read('training-report.json')
        if report['status']!='PASS' or not math.isfinite(report['seconds']) or report['seconds']<=0: raise ValueError('Invalid report')
        if report['peak_allocated_bytes']<=0 or report['peak_reserved_bytes']<report['peak_allocated_bytes']: raise ValueError('Missing memory evidence')
        for line in source.read('requirements.txt').decode().splitlines():
            if not line.strip() or line.startswith('#'): continue
            name,expected=line.split('==')
            if report['versions'][name].split('+')[0]!=expected: raise ValueError('Package mismatch')
        fingerprints=read('training-run/data_fingerprints.json')
        for name,value in fingerprints.items():
            if hashlib.sha256(source.read('dataset/'+name)).hexdigest()!=value: raise ValueError('Training input fingerprint mismatch')
        for part in ('train','validation'):
            full=pd.read_csv(io.BytesIO(source.read(f'dataset/labels_{part}_candidate.csv')))
            ids=report['selected_ids'][part]
            if len(ids)!=full.ID.nunique() or set(ids)!=set(full.ID): raise ValueError('Missing full dataset IDs')
            if report['full_audit'][part]!={'videos':full.ID.nunique(),'samples':len(full)}: raise ValueError('Full audit mismatch')
        truth=pd.read_csv(io.BytesIO(source.read('dataset/labels_validation_candidate.csv'))).rename(columns={'frame_index':'sample_index'})
        pred=csv('training-run/validation_predictions.csv')
        if list(pred.columns)!=['ID','sample_index','accel_label','steer_label']: raise ValueError('Prediction schema mismatch')
        if not set(pred.accel_label)<=set(ACCEL) or not set(pred.steer_label)<=set(STEER): raise ValueError('Unknown prediction classes')
        if pred.isna().any().any() or pred.duplicated(['ID','sample_index']).any() or len(pred)!=len(truth): raise ValueError('Invalid prediction rows')
        joined=truth.merge(pred,on=['ID','sample_index'],suffixes=('_true','_pred'),validate='one_to_one')
        if len(joined)!=len(truth): raise ValueError('Missing predictions')
        metrics=read('training-run/evaluation.json')
        for column,classes,key in [('accel_label',ACCEL,'accel'),('steer_label',STEER,'moving_steer')]:
            rows=joined if key=='accel' else joined[joined.accel_label_true!='STOPPED']
            indexes={v:i for i,v in enumerate(classes)}
            computed=classification_metrics(rows[column+'_true'].map(indexes),rows[column+'_pred'].map(indexes),classes)
            if computed!=metrics[key]: raise ValueError('Recomputed metrics mismatch')
        history=read('training-run/history.json')
        if not history or any(row['steps']<=0 or not math.isfinite(row['mean_loss']) for row in history): raise ValueError('Invalid training history')
        if max(history,key=lambda row:row['validation']['moving_steer']['macro_f1'])['validation']!=metrics: raise ValueError('Checkpoint reload evaluation differs')
        if metrics['stopped_excluded']!=int((truth.accel_label=='STOPPED').sum()): raise ValueError('STOPPED mask mismatch')
        train=pd.read_csv(io.BytesIO(source.read('dataset/labels_train_candidate.csv')))
        baselines={}; distributions={}; per_video=[]
        for column,classes,key in [('accel_label',ACCEL,'accel'),('steer_label',STEER,'moving_steer')]:
            rows=joined if key=='accel' else joined[joined.accel_label_true!='STOPPED']
            training=train if key=='accel' else train[train.accel_label!='STOPPED']
            majority=training[column].value_counts().idxmax(); indexes={v:i for i,v in enumerate(classes)}
            baselines[key]=dict(label=majority,metrics=classification_metrics(rows[column+'_true'].map(indexes),[indexes[majority]]*len(rows),classes))
            distributions[key]={'truth':rows[column+'_true'].value_counts().to_dict(),'prediction':rows[column+'_pred'].value_counts().to_dict()}
        for sid,rows in joined.groupby('ID'):
            rows=rows[rows.accel_label_true!='STOPPED'];indexes={v:i for i,v in enumerate(STEER)}
            per_video.append(dict(ID=sid,**classification_metrics(rows.steer_label_true.map(indexes),rows.steer_label_pred.map(indexes),STEER)))
    return dict(status='PASS',report=report,history=history,metrics=metrics,prediction_rows=len(pred),metrics_recomputed=True,train_majority_baselines=baselines,distributions=distributions,per_video_moving_steer=per_video)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();summary=validate(args.result,args.bundle)
    with args.output.open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
