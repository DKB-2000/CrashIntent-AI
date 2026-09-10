"""Bounded Stage1 transfer-learning trial with source-held-out evaluation."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision.models.video import MViT_V2_S_Weights
import stage1_pipeline as p

def indices(total,slot):
    start=max(0,min(total-16,round((slot+.5)*total/3-8)))
    return np.linspace(start,min(total-1,start+15),16).round().astype(int)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-seconds',type=int,default=2400);parser.add_argument('--epochs',type=int,default=6)
    args=parser.parse_args();started=time.monotonic()
    if not 0<args.max_seconds<=3000:raise ValueError('Budget must be <=3000 seconds')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);p.cv2.setNumThreads(1)
    random.seed(p.SEED);np.random.seed(p.SEED);torch.manual_seed(p.SEED);device=p._device('cuda')
    manifest=pd.read_csv(args.dataset_dir/'generation_manifest.csv',dtype={'source_id':str,'upload_group':str})
    if manifest.groupby('upload_group').split.nunique().max()!=1:raise ValueError('Upload-group leakage')
    rows={s:manifest[manifest.split==s].to_dict('records') for s in ['train','val']}
    if any(set(r['label'] for r in rs)!=set(p.LABEL_TO_INDEX) for rs in rows.values()):raise ValueError('Both classes required')
    # Decode and crop once, keeping compact uint8 clips rather than raw videos.
    cache={}
    for row in manifest.to_dict('records'):
        if time.monotonic()-started>args.max_seconds:raise TimeoutError('Preprocessing exceeded budget')
        frames=p._read_frames(args.dataset_dir/row['output_path'])
        cache[row['output_path']]=torch.stack([(p._clip(frames,indices(len(frames),slot),224)*.225+.45).mul(255).round().clamp(0,255).byte() for slot in range(3)])
    model=p.Stage1MViT(pretrained=True).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    report=dict(status='RUNNING',scope='Synthetic held-out upload groups only; real phone validation unavailable',pretrained_url=MViT_V2_S_Weights.KINETICS400_V1.url,epochs=[],train_samples=len(rows['train']),validation_samples=len(rows['val']),precision='fp16 GradScaler',skipped_steps=0)
    best=-1
    def predict():
        model.eval();pred=[]
        with torch.inference_mode():
            for row in rows['val']:
                clips=(cache[row['output_path']].float()/255-.45)/.225
                with torch.autocast('cuda',dtype=torch.float16):logits=model(clips.to(device))
                probs=logits.float().softmax(1)[:,1]
                if not torch.isfinite(probs).all():raise ValueError('Nonfinite prediction')
                value=float(probs.mean());pred.append(dict(path=row['output_path'],target=row['label'],probability=value,answer=p.INDEX_TO_LABEL[int(value>=.5)]))
        return pred
    for epoch in range(1,args.epochs+1):
        if time.monotonic()-started>args.max_seconds:break
        # First epoch fits only the classification head before fine tuning.
        for name,param in model.net.named_parameters():param.requires_grad_(epoch>1 or name.startswith('head.'))
        model.train()
        if epoch==1:model.net.eval();model.net.head.train()
        order=list(rows['train']);random.shuffle(order);losses=[]
        for start in range(0,len(order),2):
            if time.monotonic()-started>args.max_seconds:break
            batch=order[start:start+2]
            x=torch.stack([cache[r['output_path']][random.randrange(3)] for r in batch]).float().to(device)
            x=(x/255-.45)/.225;y=torch.tensor([p.LABEL_TO_INDEX[r['label']] for r in batch],device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(x).float(),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();report['skipped_steps']+=int(scaler.get_scale()<scale);losses.append(float(loss))
        pred=predict();f1=[]
        for label in p.LABEL_TO_INDEX:
            tp=sum(r['target']==r['answer']==label for r in pred);fp=sum(r['target']!=label and r['answer']==label for r in pred);fn=sum(r['target']==label and r['answer']!=label for r in pred)
            f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        score=float(np.mean(f1));item=dict(epoch=epoch,train_loss=float(np.mean(losses)),macro_f1=score,prediction_counts=pd.Series([r['answer'] for r in pred]).value_counts().to_dict())
        report['epochs'].append(item);print(item,flush=True)
        if score>best:
            best=score;report['best_epoch']=epoch
            torch.save({'model':model.net.state_dict(),'frames':16,'size':224},out/'best.pt')
            pd.DataFrame(pred).to_csv(out/'validation_predictions.csv',index=False)
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    model.net.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)['model'],strict=True)
    expected=pd.read_csv(out/'validation_predictions.csv');actual=predict()
    if [r['answer'] for r in actual]!=expected.answer.tolist() or not np.allclose([r['probability'] for r in actual],expected.probability,atol=1e-6):raise ValueError('Reload mismatch')
    report.update(status='COMPLETED' if len(report['epochs'])==args.epochs else 'TIME_LIMIT',reload_matches=True,seconds=time.monotonic()-started,best_macro_f1=best,checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
