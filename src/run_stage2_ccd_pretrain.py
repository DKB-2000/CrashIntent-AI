"""Step 3: source-disjoint CCD collision pretraining then paired Stage2 CV on CPU."""
import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import stage2_pipeline as p
from train_stage2_controls import predictions,temporal_loss
from analyze_stage2_source_cv import details,summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2-ccd-pretrain-20260910'
CV=ROOT/'artifacts/stage2-source-cv-20260910'
SEED=p.SEED


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    target=OUT/name;tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(target)


def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)


def main():
    OUT.mkdir(exist_ok=False)
    started=time.monotonic()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);p.cv2.setNumThreads(1)
    protocol=json.loads((CV/'protocol.json').read_text())
    assert json.loads((CV/'independent-validation.json').read_text())['status']=='PASS'
    manual=pd.concat([pd.read_csv(CV/'fold-0/train.csv',dtype={'ID':str,'source_id':str}),pd.read_csv(CV/'fold-0/validation.csv',dtype={'ID':str,'source_id':str})]).to_dict('records')
    excluded={r['source_id'] for r in manual}
    labels=ROOT/'data_raw/ccd/Crash-1500.txt'
    records=[]
    for line in labels.read_text().splitlines():
        match=re.fullmatch(r'(\d+),\[([^\]]+)\],(\d+),([^,]+),.*',line)
        assert match
        binary=list(map(int,match[2].split(',')))
        if match[4] not in excluded:
            records.append(dict(ID=match[1],source_id=match[4],t_collision=binary.index(1),frames=len(binary)))
    assert len(records)==1296 and len({r['source_id'] for r in records})==110
    assert not {r['source_id'] for r in records}&excluded
    assert not {r['ID'] for r in records}&{r['ID'] for r in manual}
    pd.DataFrame(records).to_csv(OUT/'pretrain-labels.csv',index=False)
    weights=ROOT/'artifacts/stage2-manual-200-20260907/training/model/stage2/resnet18-f37072fd.pth'
    config={'seed':SEED,'pretrain_epochs':3,'finetune_epochs':15,'lr':2e-4,'threads':2,'pretrain_samples':len(records),'pretrain_sources':110,'excluded_manual_sources':sorted(excluded),'folds':protocol['folds'],'label_sha256':sha(labels),'backbone_sha256':sha(weights),'script_sha256':sha(Path(__file__)),'manual_features_sha256':protocol['features_sha256'],'scope':'Exploratory reused development folds; CCD event-onset proxy, not official score','comparison':'Matched scratch vs collision pretrain, reset AdamW for identical 15-epoch soft/mixed fine-tuning; no outer-fold selection'}
    write('protocol.json',config)
    backbone,transform=p._backbone('none',torch.device('cpu'))
    backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));backbone.fc=nn.Identity();backbone.eval()
    fdir=OUT/'features';fdir.mkdir()
    provenance=[]
    extraction_started=time.monotonic()
    for index,r in enumerate(records):
        path=ROOT/'data_raw/ccd/videos/Crash-1500'/f"{r['ID']}.mp4"
        frames=p._read_video(path)
        assert len(frames)==r['frames'] and 0<=r['t_collision']<len(frames)
        feat=p._features(frames,backbone,transform,torch.device('cpu'),16)
        assert feat.shape==(r['frames'],512) and torch.isfinite(feat).all()
        fp=fdir/f"{r['ID']}.pt";torch.save(feat,fp)
        provenance.append(dict(ID=r['ID'],source_id=r['source_id'],video_sha256=sha(path),feature_sha256=sha(fp)))
        elapsed=time.monotonic()-extraction_started
        write('status.json',{'status':'RUNNING','phase':'FEATURE_EXTRACTION','completed':index+1,'total':len(records),'seconds':time.monotonic()-started,'estimated_feature_seconds_remaining':elapsed/(index+1)*(len(records)-index-1)})
        if (index+1)%10==0:print('features',index+1,'/',len(records),'seconds',round(elapsed,1),flush=True)
    write('feature-provenance.json',provenance)
    del backbone
    features={r['ID']:torch.load(fdir/f"{r['ID']}.pt",weights_only=True,map_location='cpu') for r in records}
    seed();pre=p.Stage2Temporal()
    initial=copy.deepcopy(pre.state_dict())
    optimizer=torch.optim.AdamW(list(pre.r.parameters())+list(pre.tc.parameters()),lr=2e-4)
    rng=random.Random(SEED)
    history=[]
    for epoch in range(1,4):
        pre.train();order=list(records);rng.shuffle(order);losses=[]
        for i,r in enumerate(order):
            cl,_,_=pre.logits(features[r['ID']].unsqueeze(0))
            loss=temporal_loss(cl,torch.tensor([r['t_collision']]),1.)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(pre.r.parameters())+list(pre.tc.parameters()),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            if (i+1)%50==0:write('status.json',{'status':'RUNNING','phase':'COLLISION_PRETRAIN','epoch':epoch,'updates':i+1,'per_epoch':len(order),'seconds':time.monotonic()-started})
        history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
        torch.save({'model':pre.state_dict(),'epochs':epoch},OUT/'collision-pretrain.pt')
        print('pretrain',history[-1],flush=True)
    warm=copy.deepcopy(pre.state_dict())
    assert all(torch.equal(initial[k],warm[k]) for k in initial if k.startswith(('te.','scene.')))
    write('pretrain-history.json',history)
    del features,pre,optimizer
    cache=ROOT/'artifacts/stage2-controls-20260909/features.pt';assert sha(cache)==protocol['features_sha256']
    features=torch.load(cache,weights_only=True,map_location='cpu')
    collected={'scratch':[],'pretrained':[]};all_rows=[]
    for fold,held in enumerate(protocol['folds']):
        folder=OUT/f'fold-{fold}';folder.mkdir()
        train=pd.read_csv(CV/f'fold-{fold}/train.csv',dtype={'ID':str,'source_id':str}).to_dict('records')
        val=pd.read_csv(CV/f'fold-{fold}/validation.csv',dtype={'ID':str,'source_id':str}).to_dict('records')
        assert not {r['source_id'] for r in train}&set(held)
        assert not {r['source_id'] for r in records}&set(held)
        all_rows.extend([dict(r,fold=fold) for r in val])
        for arm,state in [('scratch',initial),('pretrained',warm)]:
            model=p.Stage2Temporal();model.load_state_dict(state,strict=True);seed()
            opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
            order_rng=random.Random(SEED);scene_rng=random.Random(SEED+1)
            for epoch in range(1,16):
                model.train();order=list(train);order_rng.shuffle(order);losses=[]
                for r in order:
                    cl,el,h=model.logits(features[r['ID']].unsqueeze(0))
                    ct,et=torch.tensor([r['t_collision']]),torch.tensor([r['t_entry']])
                    use_pred=scene_rng.random()<min(.5,epoch/20)
                    ci,ei=(cl.argmax(1),el.argmax(1)) if use_pred else (ct,et)
                    scene=model.scene(torch.cat([h[0,ci],h[0,ei]],1))
                    loss=temporal_loss(cl,ct,1.)+temporal_loss(el,et,1.)+F.cross_entropy(scene[:,:2],torch.tensor([r['evasion_space']]))+F.cross_entropy(scene[:,2:],torch.tensor([int(r['entry_side']=='RIGHT')]))
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
                write('status.json',{'status':'RUNNING','phase':'PAIRED_CV','fold':fold+1,'arm':arm,'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.monotonic()-started})
                print('cv',fold+1,arm,epoch,flush=True)
            pred=predictions(model,val,features,'cpu')
            cp=folder/f'{arm}.pt';torch.save({'model':model.state_dict()},cp)
            reload=p.Stage2Temporal();reload.load_state_dict(torch.load(cp,weights_only=True)['model'],strict=True)
            assert predictions(reload,val,features,'cpu')==pred
            collected[arm].extend(pred)
    assert len(all_rows)==66 and len({r['ID'] for r in all_rows})==66
    result={}
    for arm,pred in collected.items():
        frame=pd.DataFrame(details(all_rows,pred));frame.to_csv(OUT/f'{arm}_oof.csv',index=False)
        result[arm]=summary(all_rows,pred)
        result[arm]['fold_means']={str(f):float(frame[frame.fold==f][[k+'_ok' for k in p.OUTPUT_COLUMNS[1:]]].to_numpy().mean()) for f in range(5)}
        assert abs(float(frame[[k+'_ok' for k in p.OUTPUT_COLUMNS[1:]]].to_numpy().mean())-result[arm]['development_mean'])<1e-12
    write('report.json',{'status':'COMPLETED','seconds':time.monotonic()-started,'results':result,'pretrain_updates':3*len(records),'note':'Scratch rerun is the paired comparator; not a promise of reproducing earlier RNG consumption'})
    write('status.json',{'status':'COMPLETED','seconds':time.monotonic()-started})


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():write('status.json',{'status':'FAILED','error':repr(error)})
        raise
