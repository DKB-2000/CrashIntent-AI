"""Frozen submission model on the fixed Civic consensus subset."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import cv2
import numpy as np
import pandas as pd
import torch
import stage3_motion_inference as model_code

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/stage3-civic-holdout-score-20260914'
LABELS=P/'artifacts/stage3-civic-holdout-labels-20260914'
CLASSES=['LEFT','STRAIGHT','RIGHT']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    tmp=ROOT/(name+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(ROOT/name)


def score(g):
    cm=np.array([[int(((g.partial_steer_label==a)&(g.prediction==b)).sum()) for b in CLASSES] for a in CLASSES])
    den=cm.sum(0)+cm.sum(1);support=cm.sum(1)
    f1=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den!=0)
    return dict(rows=len(g),macro_f1=float(f1.mean()),confusion=cm.tolist(),
                recall={c:float(cm[i,i]/support[i]) if support[i] else None for i,c in enumerate(CLASSES)})


def run():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    labelpath=LABELS/'evaluation_partial_labels.csv'
    labelreport=json.loads((LABELS/'report.json').read_text());assert sha(labelpath)==labelreport['outputs_sha256'][labelpath.name]
    labels=pd.read_csv(labelpath);assert labels.partial_valid_steer.sum()==467 and len(labels)==5310
    folder=P/'artifacts/stage3-motion-submit-candidate-20260910';archive=folder/'submit.zip'
    validation=json.loads((folder/'validation.json').read_text());assert sha(archive)==validation['candidate_sha256']
    with zipfile.ZipFile(archive) as z:
        weights=z.read('model/stage3/best.pt');code=z.read('inference.py').decode('utf-8-sig')
    assert hashlib.sha256(weights).hexdigest()==validation['deploy_checkpoint_sha256']
    deployed={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for n in ast.parse(Path(model_code.__file__).read_text(encoding='utf-8-sig')).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):assert ast.dump(n,include_attributes=False)==deployed[n.name]
    ck=ROOT/'frozen.pt';ck.write_bytes(weights)
    m,mean,std=model_code.s3_motion_load(ck,torch.device('cpu'))
    acquisition=json.loads((P/'artifacts/stage3-civic-holdout-acquisition-20260914/validation.json').read_text())
    save('plan.json',dict(model_zip_sha256=sha(archive),checkpoint_sha256=sha(ck),label_sha256=sha(labelpath),
         script_sha256=sha(Path(__file__)),deployment_ast_match=True,
         sampling='Decode all raw frames, keep even indices;10Hz FFV1 lossless AVI; pixel equality against each kept raw frame',
         scoring='Fixed467 sensor-consensus points; predictions on all5310 points; fixed3class macroF1; excluded91.21%; no tuning'))
    results=[];videos=[]
    for i,(sid,g) in enumerate(labels.groupby('ID')):
        save('status.json',dict(status='SCORING',completed=i,total=9,pid=os.getpid()))
        source=next(f for f in acquisition['files'] if f['ID']==sid and f['relative']=='video.hevc')
        raw=P/source['local'];assert sha(raw)==source['sha256']
        target=ROOT/(sid+'.avi');assert not target.exists()
        cap=cv2.VideoCapture(str(raw));writer=None;count=0;kept=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if count%2==0:
                if writer is None:
                    h,w=bgr.shape[:2];writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'FFV1'),10,(w,h));assert writer.isOpened()
                writer.write(bgr);kept+=1
            count+=1
        cap.release();writer.release();assert kept==len(g)
        # Decode both independently and verify all retained pixels, not just FPS metadata.
        a=cv2.VideoCapture(str(raw));b=cv2.VideoCapture(str(target));n=0
        assert abs(b.get(cv2.CAP_PROP_FPS)-10)<.001
        for j in range(count):
            ok,x=a.read();assert ok
            if j%2==0:
                ok,y=b.read();assert ok;np.testing.assert_array_equal(x,y);n+=1
        assert not b.read()[0] and n==kept;a.release();b.release()
        features=model_code.s3_motion_features(target)
        accel,steer=model_code.s3_motion_logits(m,features,mean,std,torch.device('cpu'))
        cache=ROOT/(sid+'.npz');np.savez_compressed(cache,features=features,accel=accel,steer=steer)
        with np.load(cache) as z:
            aa,ss=model_code.s3_motion_logits(m,z['features'],mean,std,torch.device('cpu'))
            np.testing.assert_array_equal(aa,z['accel']);np.testing.assert_array_equal(ss,z['steer'])
        results.append(pd.DataFrame(dict(ID=sid,sample_index=np.arange(kept),prediction=np.array(CLASSES)[steer.argmax(1)])))
        videos.append(dict(ID=sid,raw_sha256=source['sha256'],converted_sha256=sha(target),frames=kept,pixel_match=True,cache_sha256=sha(cache)))
        print(f'Validated prediction {i+1}/9 {sid}',flush=True)
    pred=pd.concat(results,ignore_index=True)
    joined=labels.merge(pred,on=['ID','sample_index'],validate='one_to_one');assert len(joined)==5310
    joined.to_csv(ROOT/'predictions.csv',index=False);eligible=joined[joined.partial_valid_steer]
    pooled=score(eligible)
    cm=[[0]*3 for _ in CLASSES]
    with (ROOT/'predictions.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            if r['partial_valid_steer']=='True':cm[CLASSES.index(r['partial_steer_label'])][CLASSES.index(r['prediction'])]+=1
    assert cm==pooled['confusion']
    independent=sum(2*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)) if sum(cm[i])+sum(r[i] for r in cm) else 0 for i in range(3))/3
    assert abs(independent-pooled['macro_f1'])<1e-12
    assert sha(labelpath)==labelreport['outputs_sha256'][labelpath.name]
    report=dict(status='COMPLETE_VALIDATED',classes=CLASSES,pooled=pooled,
          routes=[dict(route=r,**score(g)) for r,g in eligible.groupby('route')],
          videos=[dict(ID=s,**score(g)) for s,g in eligible.groupby('ID')],provenance=videos,
          total=5310,scored=467,excluded=4843,excluded_fraction=.9120527306967985,
          prediction_sha256=sha(ROOT/'predictions.csv'),
          validation='ZIP/weights/AST; source hashes; full lossless even-frame pixel parity; saved logits exact replay; independent CSV confusion and F1; labels unchanged PASS',
          scope='Selected stable sensor-consensus subset,9videos5routes, correlated selected frames, same highway different vehicle. Not human truth, full-domain or official performance. CPU only.')
    save('report.json',report);save('status.json',dict(status='COMPLETE_VALIDATED',completed=9,total=9,scored=467))
    print(json.dumps(report,indent=2),flush=True)


def watch():
    import msvcrt
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'watch.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not (ROOT/'report.json').exists()
        try:
            with (ROOT/'run.log').open('w',encoding='utf-8') as log:
                p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=P,stdout=log,stderr=subprocess.STDOUT)
                save('watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=p.pid))
                try:rc=p.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],check=True,capture_output=True,timeout=30);raise
                if rc:raise RuntimeError(f'Worker exit{rc}; see run.log')
            save('watch-status.json',dict(status='COMPLETE'))
        except Exception as e:
            save('status.json',dict(status='FAILED',error=str(e)));save('watch-status.json',dict(status='FAILED',error=str(e)));raise


if __name__=='__main__':
    {'run':run,'watch':watch}[sys.argv[1]]()
