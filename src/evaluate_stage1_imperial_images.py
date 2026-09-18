"""Evaluate frozen Stage1 checkpoints on Imperial's labelled 100-image sample."""
from __future__ import annotations
import hashlib, io, json, os
from pathlib import Path
import time, zipfile

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s

ZIP=Path("data_raw/imperial-recapture/SubjectiveTestImages.clean.zip")
OUT=Path("artifacts/stage1-imperial-evaluation-20260917")
BASE=Path("artifacts/stage1-auto-release-20260910/candidate/submit.zip")
SCREEN=Path("artifacts/kaggle-stage1-quarter-mix-trial-20260916/validated/screen_mix_25.pt")
EXPECTED_SHA="d8ce8a3671bf49235d23190f33122105ffe087a42a1550b08bfc1099bfadf697"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(**kw):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/'status.json'
    d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {};d.update(kw,heartbeat_unix=time.time())
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');os.replace(q,p)

def clip(data):
    bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError('Image decode failed')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];scale=224/min(h,w)
    nh,nw=max(224,round(h*scale)),max(224,round(w*scale));rgb=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
    y,x=(nh-224)//2,(nw-224)//2; frame=rgb[y:y+224,x:x+224]
    value=torch.from_numpy(np.stack([frame]*16)).permute(3,0,1,2).float()/255
    return (value-.45)/.225

def main():
    started=time.time();save(status='RUNNING',pid=os.getpid(),expected_models=2,completed_models=0,expected_images=100)
    try:
        if ZIP.stat().st_size!=70145784 or sha(ZIP)!=EXPECTED_SHA: raise ValueError('ZIP identity mismatch')
        with zipfile.ZipFile(ZIP) as z:
            if z.testzip() is not None: raise ValueError('ZIP CRC failed')
            labels={line.split('\t')[0]:line.split('\t')[1] for line in z.read('labels.txt').decode().splitlines()}
            if len(labels)!=100: raise ValueError('Expected 100 labels')
            # Keep compressed JPEG bytes (~70 MB), not 100 expanded 16-frame
            # float tensors (~1 GB). Decode one image immediately before use.
            cached={name:z.read(name) for name in sorted(labels)}
        with zipfile.ZipFile(BASE) as z: sources=[('official_best',z.read('model/stage1/best.pt'))]
        sources.append(('screen_mix_25',SCREEN.read_bytes()))
        torch.set_num_threads(2);results=[]
        for model_name,payload in sources:
            ck=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
            model=mvit_v2_s(weights=None);model.head[1]=nn.Linear(model.head[1].in_features,2)
            model.load_state_dict(ck['model'],strict=True);model.eval()
            with torch.inference_mode():
                for name,data in cached.items():
                    if time.time()-started>7200: raise TimeoutError('Two-hour evaluation budget reached')
                    tensor=clip(data)
                    probability=float(model(tensor.unsqueeze(0)).softmax(1)[0,1])
                    label='ORIGINAL' if labels[name]=='Original Captured' else 'RERECORDED'
                    answer='RERECORDED' if probability>=.5 else 'ORIGINAL'
                    results.append(dict(model=model_name,image=name,label=label,probability=probability,
                                        answer=answer,correct=answer==label))
                    del tensor
            save(completed_models=len({r['model'] for r in results}),completed_predictions=len(results))
        summaries=[]
        for model_name,_ in sources:
            for label in ('ORIGINAL','RERECORDED'):
                part=[r for r in results if r['model']==model_name and r['label']==label]
                summaries.append(dict(model=model_name,label=label,images=len(part),
                    accuracy=sum(r['correct'] for r in part)/len(part),
                    mean_rerecorded_probability=float(np.mean([r['probability'] for r in part])),
                    median_rerecorded_probability=float(np.median([r['probability'] for r in part]))))
        report=dict(status='COMPLETE_VALIDATED',summaries=summaries,
            limitations=['Static images are repeated for 16 frames; temporal replay cues are not measured.',
                         'This public sample has no explicit SPDX-style license; diagnostic use only.',
                         'No fitting or threshold selection used these images.'])
        (OUT/'predictions.json').write_text(json.dumps(results,indent=2)+'\n')
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        save(status='COMPLETE_VALIDATED',completed_models=2,completed_predictions=200,
             report_sha256=sha(OUT/'report.json'),seconds=time.time()-started)
    except Exception as e:
        save(status='FAILED',error=repr(e),seconds=time.time()-started);raise
if __name__=='__main__':main()
