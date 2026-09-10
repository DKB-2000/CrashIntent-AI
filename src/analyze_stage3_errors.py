"""Compare verified Stage3 checkpoints by class, route, video and causal label stability."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ACCEL=['ACCELERATING','DECELERATING','CONSTANT','STOPPED']
STEER=['LEFT','STRAIGHT','RIGHT']
ROOT=Path('artifacts/stage3-error-analysis-20260910')
DATA=Path('artifacts/stage3-comma-chunk1-calibrated-v1')
RUNS={'200':'kaggle-stage3-route-trial-20260910','1000':'kaggle-stage3-expanded-20260910'}

def metrics(frame,head):
    names=ACCEL if head=='accel' else STEER
    if head=='steer':frame=frame[frame.accel_label!='STOPPED']
    cm=pd.crosstab(frame[head+'_label'],frame[head+'_pred']).reindex(index=names,columns=names,fill_value=0).to_numpy()
    support=cm.sum(1);predicted=cm.sum(0);tp=np.diag(cm)
    def div(a,b):return np.divide(a,b,out=np.zeros(len(names),dtype=float),where=b!=0)
    precision=div(tp,predicted);recall=div(tp,support);f1=div(2*tp,support+predicted)
    return dict(rows=int(cm.sum()),macro_f1=float(f1.mean()) if cm.sum() else None,accuracy=float(tp.sum()/cm.sum()) if cm.sum() else None,
        classes=[dict(label=c,support=int(support[i]),predicted=int(predicted[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i,c in enumerate(names)],confusion=cm.tolist())

def main():
    ROOT.mkdir(exist_ok=True)
    labels=pd.read_csv(DATA/'labels_validation_candidate.csv').rename(columns={'frame_index':'sample_index'})
    manifest=pd.read_csv(DATA/'split_manifest.csv')
    assert set(manifest.query("split=='train'").route).isdisjoint(set(manifest.query("split=='validation'").route))
    truth=labels.merge(manifest[['ID','route']],on='ID',validate='many_to_one').sort_values(['ID','sample_index'])
    for head in ['accel','steer']:
        # A causal 16-frame window is mixed until 15 frames after the latest truth change.
        truth[head+'_stable']=False
        for _,g in truth.groupby('ID',sort=False):
            y=g[head+'_label'].to_numpy();starts=np.r_[True,y[1:]!=y[:-1]]
            age=np.arange(len(g))-np.maximum.accumulate(np.where(starts,np.arange(len(g)),0))
            truth.loc[g.index,head+'_stable']=age>=15
    summary={};route_rows=[];video_rows=[];class_rows=[];stability=[];merged_runs={};hashes={}
    for tag,directory in RUNS.items():
        run=Path('artifacts')/directory
        verified=json.loads((run/'result-validation.json').read_text());assert verified['status']=='PASS'
        archive=run/'result/stage3-route-trial-result.zip';digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        assert digest==verified['archive_sha256'];hashes[tag]=digest
        with zipfile.ZipFile(archive) as z:
            pred=pd.read_csv(z.open('trial-run/validation_predictions.csv')).rename(columns={'accel_label':'accel_pred','steer_label':'steer_pred'})
            report=json.loads(z.read('trial-run/report.json'))
        assert hashlib.sha256((DATA/'labels_validation_candidate.csv').read_bytes()).hexdigest()==report['data_sha256']['labels_validation_candidate.csv']
        f=truth.merge(pred,on=['ID','sample_index'],validate='one_to_one');assert len(f)==len(truth)==18603
        merged_runs[tag]=f;summary[tag]={}
        for head in ['accel','steer']:
            m=metrics(f,head);summary[tag][head]=m
            key='accel' if head=='accel' else 'moving_steer'
            assert abs(m['macro_f1']-verified['metrics']['validation'][key])<1e-12
            for row in m['classes']:class_rows.append(dict(model=tag,head=head,**row))
            for route,g in f.groupby('route'):
                r=metrics(g,head);route_rows.append(dict(model=tag,head=head,route=route,rows=r['rows'],macro_f1=r['macro_f1'],accuracy=r['accuracy']))
            for sid,g in f.groupby('ID'):
                r=metrics(g,head);video_rows.append(dict(model=tag,head=head,ID=sid,route=g.route.iloc[0],rows=r['rows'],macro_f1=r['macro_f1'],accuracy=r['accuracy']))
            for stable,g in f.groupby(head+'_stable'):
                r=metrics(g,head);stability.append(dict(model=tag,head=head,stable=bool(stable),rows=r['rows'],accuracy=r['accuracy'],macro_f1=r['macro_f1']))
        summary[tag]['epoch_loss']=report['history']
        summary[tag]['train_metrics']=report['train_metrics']
        summary[tag]['training_groups']=pd.Series([r['group'] for r in report['samples']]).value_counts().sort_index().to_dict()
        route_classes=[]
        for route,g in f.groupby('route'):
            for head in ['accel','steer']:
                for row in metrics(g,head)['classes']:route_classes.append(dict(route=route,head=head,**row))
        pd.DataFrame(route_classes).to_csv(ROOT/('route-classes-'+tag+'.csv'),index=False)
    paired={}
    for head in ['accel','steer']:
        a=merged_runs['200'];b=merged_runs['1000'];assert a[['ID','sample_index']].equals(b[['ID','sample_index']])
        mask=a.accel_label!='STOPPED' if head=='steer' else np.ones(len(a),dtype=bool)
        old=(a[head+'_pred']==a[head+'_label'])[mask];new=(b[head+'_pred']==b[head+'_label'])[mask]
        paired[head]=dict(corrected=int((~old&new).sum()),regressed=int((old&~new).sum()),both_wrong=int((~old&~new).sum()),both_correct=int((old&new).sum()))
    for name,rows in [('classes',class_rows),('routes',route_rows),('videos',video_rows),('stability',stability)]:pd.DataFrame(rows).to_csv(ROOT/(name+'.csv'),index=False)
    # Reproducible review queue: longest persistent errors on each validation route/head.
    segments=[]
    for head in ['accel','steer']:
        for sid,g in merged_runs['1000'].groupby('ID'):
            g=g.sort_values('sample_index');wrong=(g[head+'_label']!=g[head+'_pred'])
            if head=='steer':wrong&=g.accel_label!='STOPPED'
            keys=g[head+'_label']+'>'+g[head+'_pred'];boundaries=(keys!=keys.shift())|(wrong!=wrong.shift())
            for _,part in g.assign(wrong=wrong).groupby(boundaries.cumsum()):
                if not part.wrong.iloc[0] or len(part)<10:continue
                segments.append(dict(head=head,ID=sid,route=part.route.iloc[0],start=int(part.sample_index.iloc[0]),end=int(part.sample_index.iloc[-1]),seconds=len(part)/10,truth=part[head+'_label'].iloc[0],prediction=part[head+'_pred'].iloc[0]))
    queue=pd.DataFrame(segments).sort_values('seconds',ascending=False).groupby(['head','route']).head(3)
    queue.to_csv(ROOT/'review-queue.csv',index=False)
    result=dict(status='PASS',scope='Diagnostic comparison on reused external validation routes; correlated frames, no hidden competition data',source_sha256=hashes,metrics=summary,paired=paired)
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2))
    print('CLASSES');print(pd.DataFrame(class_rows).query("model=='1000'").to_string(index=False))
    print('ROUTES');print(pd.DataFrame(route_rows).pivot(index=['head','route'],columns='model',values='macro_f1').to_string())
    print('STABILITY');print(pd.DataFrame(stability).query("model=='1000'").to_string(index=False))
    print('PAIRED',paired)
    print('FINAL LOSSES',[(h['epoch'],h['mean_loss']) for h in summary['1000']['epoch_loss'][-4:]])
if __name__=='__main__':main()
