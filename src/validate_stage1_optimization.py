"""Independently validate Stage1 optimization logits, inputs and progress."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def require(value,message):
    if not value:raise ValueError(message)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--bundle-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'artifacts/kaggle-stage1-trial-20260910/dataset'
    with zipfile.ZipFile(a.result) as z,zipfile.ZipFile(base/'stage1-training.bin') as original:
        require(z.testzip() is None,'Corrupt result archive')
        report=json.loads(z.read('training/report.json'));runner=json.loads(z.read('runner.json'))
        require(runner['failure'] is None and runner['seconds']<3600,'Runner failed or exceeded budget')
        require(json.loads(z.read('install.json'))['exit_code']==0,'Install failed')
        code=(a.bundle_dir/'dataset/stage1-optimization.py').read_bytes()
        require(z.read('src/diagnose_stage1_optimization.py')==code,'Diagnostic code mismatch')
        require(report['code_sha256']==hashlib.sha256(code).hexdigest(),'Code hash mismatch')
        manifest=original.read('dataset/generation_manifest.csv')
        require(z.read('dataset/generation_manifest.csv')==manifest,'Manifest changed')
        require(report['manifest_sha256']==hashlib.sha256(manifest).hexdigest(),'Manifest hash mismatch')
        require(json.loads(z.read('input-assets.json'))==json.loads((base/'stage1-training-assets.json').read_text()),'Wrong input assets')
        table=pd.read_csv(io.BytesIO(manifest),dtype={'source_id':str,'upload_group':str})
        targets={}
        for part,rows in report['rows'].items():
            expected=table[table.split==part]
            require([r['path'] for r in rows]==expected.output_path.tolist(),'Row identity mismatch')
            require([r['source_id'] for r in rows]==expected.source_id.tolist(),'Source mismatch')
            require([r['upload_group'] for r in rows]==expected.upload_group.tolist(),'Group mismatch')
            targets[part]=torch.tensor([int(v=='RERECORDED') for v in expected.label])
            require([r['target'] for r in rows]==targets[part].tolist(),'Target mismatch')
        require(set(table[table.split=='train'].upload_group).isdisjoint(table[table.split=='val'].upload_group),'Group leakage')
        summary=[];expected_names=['uniform_legacy','uniform_global','balanced_low_lr','balanced_high_lr']
        require([c['name'] for c in report['conditions']]==expected_names[:len(report['conditions'])],'Unexpected conditions')
        initial=None
        for condition in report['conditions']:
            require(condition['samples_seen']==8*condition['updates'],'Exposure mismatch')
            require(sum(condition['class_exposures'])==condition['samples_seen'],'Class exposures mismatch')
            if condition['balanced']:require(condition['class_exposures']==[condition['samples_seen']//2]*2,'Unbalanced diagnostic batch')
            require(condition['history'][0]['step']==0 and condition['history'][-1]['step']==condition['updates'],'Missing endpoints')
            if initial is None:initial=condition['history'][0]['metrics']
            else:require(condition['history'][0]['metrics']==initial,'Initial predictions changed across conditions')
            final={}
            for snapshot in condition['history']:
                for part,measurement in snapshot['metrics'].items():
                    logits=torch.tensor(measurement['logits'],dtype=torch.float64);target=targets[part]
                    require(logits.shape==(len(target),1 if part=='train' else 3,2) and torch.isfinite(logits).all(),'Invalid logits')
                    pred=(logits.softmax(-1).mean(1)[:,1]>=.5).long();accuracy=float((pred==target).double().mean());counts=torch.bincount(pred,minlength=2).tolist();ce=float(F.cross_entropy(logits.mean(1),target))
                    require(abs(accuracy-measurement['accuracy'])<1e-6 and counts==measurement['prediction_counts'],'Prediction metric mismatch')
                    require(abs(ce-measurement['ce'])<1e-5,'Loss mismatch')
                    f1=[]
                    for label in [0,1]:
                        denom=int((target==label).sum()+(pred==label).sum());tp=int(((target==label)&(pred==label)).sum());f1.append(2*tp/denom if denom else 0.)
                    final[part]=dict(accuracy=accuracy,macro_f1=float(np.mean(f1)),prediction_counts=counts,ce=ce)
            if condition['status']=='COMPLETED':require(condition['updates']==120,'False condition completion')
            summary.append(dict(name=condition['name'],updates=condition['updates'],status=condition['status'],**final))
        if report['status']=='COMPLETED':require(len(summary)==4 and all(c['updates']==120 for c in summary),'False full completion')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    result=dict(status='PASS',diagnostic_status=report['status'],conditions=summary,runner_seconds=runner['seconds'],result_sha256=hashlib.sha256(a.result.read_bytes()).hexdigest(),scope='Optimization diagnostic only; no deployment or real-phone performance claim')
    (a.output_dir/'result-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
