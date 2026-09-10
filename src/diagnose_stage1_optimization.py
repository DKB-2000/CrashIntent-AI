"""Bounded Stage1 collapse controls on fixed training clips; no deployable weights."""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from train_stage1_full import audit
import serving_inference as serving


def weighted_micro_loss(logits, targets, weights, denominator):
    return F.cross_entropy(logits, targets, weight=weights, reduction='sum') / denominator


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--max-seconds',type=int,default=2400);a=ap.parse_args()
    if not 0<a.max_seconds<=2600:raise ValueError('Budget exceeds diagnostic limit')
    start=time.monotonic();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);serving.cv2.setNumThreads(1);torch.manual_seed(20260903);device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    table=audit(a.dataset_dir);rows={p:table[table.split==p].to_dict('records') for p in ['train','val']}
    clips={};targets={}
    for part in rows:
        values=[]
        for r in rows[part]:
            path=a.dataset_dir/r['output_path'];slots=[1] if part=='train' else [0,1,2]
            values.append(torch.stack([serving._decode_stage1_clip(path,224,serving._clip_ids(path,16,s,3)) for s in slots]))
        clips[part]=torch.stack(values);targets[part]=torch.tensor([int(r['label']=='RERECORDED') for r in rows[part]])
    model=mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1);model.head[1]=nn.Linear(model.head[1].in_features,2)
    for module in model.modules():
        if isinstance(module,nn.Dropout):module.p=0.
        if module.__class__.__name__=='StochasticDepth':module.p=0.
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    h=hashlib.sha256()
    for k,v in initial.items():h.update(k.encode());h.update(v.numpy().tobytes())
    report=dict(status='RUNNING',scope='Fixed training-clip optimization diagnostic; held-out synthetic validation only',initial_sha256=h.hexdigest(),manifest_sha256=hashlib.sha256((a.dataset_dir/'generation_manifest.csv').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={p:hashlib.sha256(v.numpy().tobytes()).hexdigest() for p,v in clips.items()},rows={p:[dict(path=r['output_path'],source_id=r['source_id'],upload_group=r['upload_group'],target=int(r['label']=='RERECORDED')) for r in rs] for p,rs in rows.items()},conditions=[])
    model.to(device);weights=torch.tensor([1.5,.75],device=device)
    def measure():
        model.eval();result={}
        with torch.inference_mode():
            for part,values in clips.items():
                logits=[]
                for batch in values.flatten(0,1).split(2):logits.append(model(batch.to(device)).cpu())
                logits=torch.cat(logits).reshape(len(values),values.shape[1],2)
                if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
                probabilities=logits.softmax(-1).mean(1)[:,1];pred=(probabilities>=.5).long()
                result[part]=dict(logits=logits.tolist(),accuracy=float((pred==targets[part]).float().mean()),prediction_counts=torch.bincount(pred,minlength=2).tolist(),ce=float(F.cross_entropy(logits.mean(1),targets[part])))
        return result
    configs=[('uniform_legacy',2e-5,False,True),('uniform_global',2e-5,False,False),('balanced_low_lr',2e-5,True,False),('balanced_high_lr',1e-4,True,False)]
    for name,lr,balanced,legacy in configs:
        if time.monotonic()-start>a.max_seconds-120:break
        model.load_state_dict(initial);torch.manual_seed(20260903);rng=random.Random(20260903)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
        item=dict(name=name,learning_rate=lr,balanced=balanced,legacy_normalization=legacy,updates=0,samples_seen=0,history=[dict(step=0,metrics=measure())],class_exposures=[0,0])
        t=time.monotonic();initial_head=model.head[1].weight.detach().clone();groups=[torch.where(targets['train']==i)[0].tolist() for i in [0,1]]
        for step in range(1,121):
            if time.monotonic()-t>500 or time.monotonic()-start>a.max_seconds-60:break
            ids=([rng.choice(groups[i%2]) for i in range(8)] if balanced else rng.sample(range(len(rows['train'])),8))
            target=targets['train'][ids].to(device);denominator=8 if balanced else weights[target].sum()
            model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.
            for offset in range(0,8,2):
                x=clips['train'][ids[offset:offset+2],0].to(device);y=target[offset:offset+2];logits=model(x)
                if legacy:loss=F.cross_entropy(logits,y,weight=weights)/4
                else:loss=weighted_micro_loss(logits,y,None if balanced else weights,denominator)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();loss_sum+=float(loss.detach())
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            item['updates']=step;item['samples_seen']+=8
            for label in [0,1]:item['class_exposures'][label]+=int((target==label).sum())
            if step in [24,60,120]:
                snapshot=dict(step=step,loss=loss_sum,gradient_norm=float(norm),head_change=float((model.head[1].weight.detach()-initial_head).norm()),metrics=measure());item['history'].append(snapshot)
                print(json.dumps(dict(condition=name,step=step,train_accuracy=snapshot['metrics']['train']['accuracy'],val_accuracy=snapshot['metrics']['val']['accuracy'])),flush=True)
        if item['history'][-1]['step']!=item['updates']:item['history'].append(dict(step=item['updates'],metrics=measure()))
        item.update(status='COMPLETED' if item['updates']==120 else 'TIME_LIMIT',seconds=time.monotonic()-t)
        report['conditions'].append(item);(a.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    report.update(status='COMPLETED' if len(report['conditions'])==4 and all(c['status']=='COMPLETED' for c in report['conditions']) else 'TIME_LIMIT',seconds=time.monotonic()-start,torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name())
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
