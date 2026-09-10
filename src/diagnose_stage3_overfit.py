"""Bounded training-only overfit probe; never a submission model."""
import argparse,json,time,hashlib,random
from pathlib import Path
import numpy as np
import torch
import stage3_pipeline as p


def main():
    a=argparse.ArgumentParser();a.add_argument('--dataset-dir',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--steps',type=int,default=180);args=a.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);p.cv2.setNumThreads(1)
    records=p.load_records(args.dataset_dir)['train']
    chosen=[];used=set()
    for accel in range(4):
        for steer in (range(3) if accel!=3 else [1]):
            options=[]
            for i,row in enumerate(records):
                indexes=np.flatnonzero((row['accel']==accel)&((row['steer']==steer) if accel!=3 else True)&(np.arange(len(row['accel']))>=15))
                if len(indexes):options.append((i,int(indexes[len(indexes)//2])))
            if not options:raise ValueError('Missing training class combination')
            i,end=next((v for v in options if v[0] not in used),options[0]);used.add(i)
            chosen.append(dict(record=i,endpoint=end,ID=records[i]['ID'],accel=accel,steer=int(records[i]['steer'][end])))
    cache={i:list(p.video_frames(records[i]['path'])) for i in used}
    clips=torch.stack([p.clip_tensor([cache[r['record']][t] for t in p.causal_indices(r['endpoint'])]) for r in chosen]);del cache
    target_a=torch.tensor([r['accel'] for r in chosen],device='cuda');target_s=torch.tensor([r['steer'] for r in chosen],device='cuda')
    moving=target_a!=3
    reports=[]
    for mode in ['trained','scratch']:
        random.seed(p.SEED);np.random.seed(p.SEED);torch.manual_seed(p.SEED)
        model=p.load_model(args.checkpoint,torch.device('cuda')) if mode=='trained' else p.Stage3MViT().cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cuda')
        tracked={'backbone':next(model.backbone.parameters()),'accel':model.accel.weight,'steer':model.steer.weight}
        before={k:v.detach().clone() for k,v in tracked.items()}
        history=[];gradients=[];started=time.monotonic();rng=np.random.default_rng(p.SEED)
        def measure(step):
            model.eval();la=[];ls=[]
            with torch.no_grad():
                for start in range(0,len(clips),2):
                    with p.autocast(torch.device('cuda')): aa,ss=model(clips[start:start+2].cuda())
                    la.append(aa.float());ls.append(ss.float())
            aa=torch.cat(la);ss=torch.cat(ls)
            row=dict(step=step,loss=float(p.multitask_loss((aa,ss),target_a,target_s)),accel_accuracy=float((aa.argmax(1)==target_a).float().mean()),steer_accuracy=float((ss.argmax(1)[moving]==target_s[moving]).float().mean()),accel_logits_std_across_inputs=aa.std(0).cpu().tolist(),steer_logits_std_across_inputs=ss.std(0).cpu().tolist(),accel_predictions=aa.argmax(1).cpu().tolist(),steer_predictions=ss.argmax(1).cpu().tolist())
            history.append(row);print(mode,json.dumps(row),flush=True)
            return row
        measure(0)
        for step in range(1,args.steps+1):
            ids=rng.choice(len(clips),size=2,replace=False);idx=torch.tensor(ids,device='cuda')
            model.train();optimizer.zero_grad(set_to_none=True)
            with p.autocast(torch.device('cuda')):loss=p.multitask_loss(model(clips[ids].cuda()),target_a[idx],target_s[idx])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if step in [1,args.steps]:gradients.append(dict(step=step,**{k:float(v.grad.norm()) if v.grad is not None else None for k,v in tracked.items()}))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
            if step%20==0 or step==args.steps:
                row=measure(step)
                if row['accel_accuracy']==1 and row['steer_accuracy']==1:break
        reports.append(dict(initialization=mode,seconds=time.monotonic()-started,history=history,gradients=gradients,weight_changes={k:float((v.detach()-before[k]).norm()) for k,v in tracked.items()}))
        del optimizer,model,tracked,before;torch.cuda.empty_cache()
    result=dict(status='COMPLETED',scope='Training-only overfit diagnostic; not generalization or submission',checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),samples=chosen,runs=reports)
    (args.output_dir/'diagnostic.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
