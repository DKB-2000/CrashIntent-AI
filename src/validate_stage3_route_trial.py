"""Independently recompute Stage3 route trial metrics and verify source/data identity."""
import argparse,hashlib,json,zipfile,sys
from pathlib import Path
import numpy as np
import pandas as pd
import stage3_pipeline as p

def metrics(truth,pred,names):
    matrix=np.zeros((len(names),len(names)),dtype=np.int64)
    for a,b in zip(truth,pred):matrix[a,b]+=1
    den=matrix.sum(0)+matrix.sum(1)
    f1=np.divide(2*np.diag(matrix),den,out=np.zeros(len(names)),where=den!=0)
    return float(f1.mean())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root;archive=root/'result/stage3-route-trial-result.zip'
    plan=json.loads((root/'sample-plan.json').read_text())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('route-trial-assets.json'))
        assert manifest==json.loads((root/'dataset/route-trial-assets.json').read_text())
        for name,h in manifest['files'].items():assert hashlib.sha256((root/'dataset'/name).read_bytes()).hexdigest()==h
        report=json.loads(z.read('trial-run/report.json'))
        assert report['samples']==plan['samples']
        assert report['planned_validation_ids']==plan['validation_ids']
        assert report['reload_logits_exact']
        if report['config']['initialization']=='checkpoint_weights_optimizer_reset':
            reference=json.loads((root/'dataset/reference-report.json').read_text())
            assert report['optimizer_reset'] is True
            assert report['initial_checkpoint_sha256']==reference['checkpoint_sha256']
            assert report['training_input_sha256']==reference['training_input_sha256']
            assert report['samples']==reference['samples']
            assert report['initial_train_metrics']==reference['train_metrics']

        assert hashlib.sha256(z.read('trial-run/best.pt')).hexdigest()==report['checkpoint_sha256']
        data=Path('artifacts/stage3-comma-chunk1-calibrated-v1')
        with zipfile.ZipFile('artifacts/kaggle-stage3-training/dataset/stage3-training-v1.zip') as bundle:
            for name,h in report['data_sha256'].items():
                content=bundle.read('dataset/'+name)
                assert hashlib.sha256(content).hexdigest()==h
                if name=='split_manifest.csv':
                    import io
                    remote=pd.read_csv(io.BytesIO(content)).sort_values('ID').reset_index(drop=True);local=pd.read_csv(data/name).sort_values('ID').reset_index(drop=True)
                    pd.testing.assert_frame_equal(remote.drop(columns='video'),local.drop(columns='video'))
                    assert remote.video.map(lambda x:Path(x).name).tolist()==local.video.map(lambda x:Path(x).name).tolist()
                else:assert content==(data/name).read_bytes()
        labels=pd.read_csv(data/'labels_validation_candidate.csv').rename(columns={'frame_index':'sample_index'})
        results={}
        for kind in ['train','validation']:
            frame=pd.read_csv(z.open('trial-run/'+kind+'_predictions.csv'))
            assert list(frame.columns)==p.OUTPUT_COLUMNS
            assert not frame.duplicated(['ID','sample_index']).any()
            if kind=='train':
                truth=pd.DataFrame(plan['samples']).rename(columns={'endpoint':'sample_index'})
                truth['accel_label']=truth.accel.map(dict(enumerate(p.ACCEL)))
                truth['steer_label']=truth.steer.map(dict(enumerate(p.STEER)))
            else:truth=labels
            merged=frame.merge(truth[['ID','sample_index','accel_label','steer_label']],on=['ID','sample_index'],suffixes=('_pred','_truth'),validate='one_to_one')
            assert len(merged)==len(frame)
            if kind=='train' or report['validation_complete']:assert len(frame)==len(truth)
            values={}
            ta=merged.accel_label_truth.map(dict(zip(p.ACCEL,range(4)))).to_numpy()
            for name,names,mask in [('accel',p.ACCEL,np.ones(len(frame),dtype=bool)),('moving_steer',p.STEER,ta!=3)]:
                column='accel' if name=='accel' else 'steer'
                mapping=dict(zip(names,range(len(names))))
                a=merged[column+'_label_truth'].map(mapping);b=merged[column+'_label_pred'].map(mapping)
                assert a.notna().all() and b.notna().all()
                value=metrics(a.to_numpy()[mask],b.to_numpy()[mask],names)
                claimed=report['train_metrics' if kind=='train' else 'validation'][name]['macro_f1']
                assert abs(value-claimed)<1e-10,(name,value,claimed)
                values[name]=value
            results[kind]=dict(rows=len(frame),**values)
        if report['status']=='COMPLETED':
            assert report['validation_complete'] and report['optimizer_steps']==len(plan['samples'])//10*report['config']['epochs_requested'] and report['samples_seen']==len(plan['samples'])*report['config']['epochs_requested']
            assert len(report['history'])==report['config']['epochs_requested'] and all(h['epoch_complete'] and h['updates']==len(plan['samples'])//10 for h in report['history'])
        output=dict(status='PASS',experiment_status=report['status'],validation_complete=report['validation_complete'],metrics=results,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (root/'result-validation.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':main()
