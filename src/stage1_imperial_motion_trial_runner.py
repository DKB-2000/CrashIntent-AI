"""Train-only Imperial motion augmentation trial for Kaggle T4."""
import hashlib, io, json, math, random, sys, time, zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def macro_f1(rows):
    values=[]
    for label in (0,1):
        tp=sum(r['label']==label and r['answer']==label for r in rows)
        fp=sum(r['label']!=label and r['answer']==label for r in rows)
        fn=sum(r['label']==label and r['answer']!=label for r in rows)
        precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
        values.append(2*precision*recall/(precision+recall) if precision+recall else 0.0)
    return sum(values)/2


def decoded(data):
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Image decode failed')
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def clip(image, seed, moving):
    rng=np.random.default_rng(seed); h,w=image.shape[:2]; scale=256/min(h,w)
    base=cv2.resize(image,(max(256,round(w*scale)),max(256,round(h*scale))),interpolation=cv2.INTER_AREA)
    bh,bw=base.shape[:2]; frames=[]
    phase=float(rng.uniform(0,2*math.pi)); dx=float(rng.uniform(-7,7)); dy=float(rng.uniform(-5,5))
    zoom=float(rng.uniform(-.018,.018)); exposure=float(rng.uniform(.96,1.04)); band=float(rng.uniform(0,.035))
    for index in range(16):
        t=index/15; z=1+(zoom*(2*t-1) if moving else 0); x=dx*(2*t-1) if moving else 0; y=dy*math.sin(t*math.pi) if moving else 0
        matrix=cv2.getRotationMatrix2D((bw/2,bh/2),0,z); matrix[:,2]+=(x,y)
        frame=cv2.warpAffine(base,matrix,(bw,bh),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        cy,cx=bh//2,bw//2; frame=frame[cy-112:cy+112,cx-112:cx+112].astype(np.float32)
        gain=exposure*(1+.025*math.sin(2*math.pi*index/7+phase) if moving else 1)
        if moving and band:
            rows=np.arange(224,dtype=np.float32)[:,None,None]
            frame*=gain*(1+band*np.sin(rows*.17+index*.63+phase))
        else: frame*=gain
        frames.append(np.clip(frame,0,255).astype(np.uint8))
    value=torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float().div_(255)
    return value.sub_(.45).div_(.225)


def load_model(path, device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True); model=mvit_v2_s(weights=None)
    model.head[1]=nn.Linear(model.head[1].in_features,2); model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device)


def predict(model, items, device, moving, views, model_name, split):
    rows=[]; model.eval()
    with torch.inference_mode():
        for name,label,image in items:
            probabilities=[]
            for view in range(views):
                value=clip(image,100000+int(name[3:6])*101+view,moving).unsqueeze(0).to(device)
                with torch.autocast('cuda',dtype=torch.float16): probability=float(model(value).softmax(1)[0,1])
                probabilities.append(probability)
            probability=float(np.mean(probabilities)); answer=int(probability>=.5)
            rows.append(dict(model=model_name,split=split,view='motion' if moving else 'static',image=name,
                             label=label,probability=probability,answer=answer,correct=answer==label))
    return rows


def main(root):
    root=Path(root); out=root/'motion-result'; out.mkdir(exist_ok=False); seed_all(236753)
    with zipfile.ZipFile(root/'imperial.bin') as archive:
        if archive.testzip() is not None: raise ValueError('ZIP CRC failed')
        labels={line.split('\t')[0]:int(line.split('\t')[1]!='Original Captured') for line in archive.read('labels.txt').decode().splitlines()}
        items=[(name,labels[name],decoded(archive.read(name))) for name in sorted(labels)]
    train=[]; holdout=[]
    for label in (0,1):
        group=[item for item in items if item[1]==label]
        ranked=sorted(group,key=lambda item:hashlib.sha256(('split-v1:'+item[0]).encode()).hexdigest())
        holdout.extend(ranked[:10]); train.extend(ranked[10:])
    if len(train)!=80 or len(holdout)!=20: raise ValueError('Split coverage failed')
    device=torch.device('cuda'); torch.set_num_threads(2); cv2.setNumThreads(1); started=time.time()
    baseline=load_model(root/'official.pt',device); rows=[]
    rows+=predict(baseline,holdout,device,False,1,'baseline','holdout')
    rows+=predict(baseline,holdout,device,True,3,'baseline','holdout')
    candidate=load_model(root/'official.pt',device); candidate.train()
    optimizer=torch.optim.AdamW(candidate.parameters(),lr=5e-6,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda'); by_label={label:[item for item in train if item[1]==label] for label in (0,1)}
    losses=[]
    for step in range(64):
        batch=[]; targets=[]
        for label in (0,1):
            item=by_label[label][step%len(by_label[label])]; batch.append(clip(item[2],236753+step*17+label,True)); targets.append(label)
        value=torch.stack(batch).to(device); target=torch.tensor(targets,device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss=nn.functional.cross_entropy(candidate(value),target)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(candidate.parameters(),1.0)
        scaler.step(optimizer); scaler.update(); losses.append(float(loss))
    rows+=predict(candidate,holdout,device,False,1,'motion_finetune','holdout')
    rows+=predict(candidate,holdout,device,True,3,'motion_finetune','holdout')
    summaries=[]
    for model_name in ('baseline','motion_finetune'):
        for view in ('static','motion'):
            selected=[r for r in rows if r['model']==model_name and r['view']==view]
            summaries.append(dict(model=model_name,view=view,images=len(selected),accuracy=sum(r['correct'] for r in selected)/len(selected),macro_f1=macro_f1(selected),
                                  original_accuracy=sum(r['correct'] for r in selected if r['label']==0)/10,
                                  rerecorded_accuracy=sum(r['correct'] for r in selected if r['label']==1)/10))
    torch.save({'model':candidate.state_dict(),'size':224,'frames':16,'trial':'imperial-motion-v1'},out/'motion_finetune.pt')
    report=dict(status='COMPLETE',train_images=80,holdout_images=20,updates=64,lr=5e-6,augmentation='class-symmetric affine motion, exposure oscillation and horizontal banding',summaries=summaries,
                loss_first=float(np.mean(losses[:8])),loss_last=float(np.mean(losses[-8:])),seconds=time.time()-started,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                limitations=['Imperial sample has no explicit SPDX-style license; diagnostic training only.','Image-level split; no paired capture identities are provided.','Candidate is not approved for submission without regression evaluation.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)); (out/'predictions.json').write_text(json.dumps(rows,indent=2))
    (out/'split.json').write_text(json.dumps({'train':[x[0] for x in train],'holdout':[x[0] for x in holdout]},indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main(Path(sys.argv[1]))
