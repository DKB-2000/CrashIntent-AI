"""Offline Stage3 MViTv2-S training, dense causal inference and runtime smoke.

Images alone enter the model. CAN/IMU/pose are never model inputs.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import random
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.video import mvit_v2_s

ACCEL = ['ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED']
STEER = ['LEFT', 'STRAIGHT', 'RIGHT']
OUTPUT_COLUMNS = ['ID', 'sample_index', 'accel_label', 'steer_label']
SEED = 20260825
FORMAT = 'stage3-causal16-v1'


class Stage3MViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        dim = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dim, 4)
        self.steer = nn.Linear(dim, 3)

    def forward(self, x):
        z = self.backbone(x)
        return self.accel(z), self.steer(z)


def device_for(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use CPU only for local tests')
    return torch.device(name)


def prepare_frame(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = 256 / min(h, w)
    rgb = cv2.resize(rgb, (max(256, round(w*scale)), max(256, round(h*scale))), interpolation=cv2.INTER_AREA)
    y, x = (rgb.shape[0]-224)//2, (rgb.shape[1]-224)//2
    return np.ascontiguousarray(rgb[y:y+224, x:x+224].transpose(2, 0, 1))


def causal_indices(endpoint):
    if endpoint < 0: raise ValueError('Negative endpoint')
    return np.maximum(np.arange(endpoint-15, endpoint+1), 0)


def clip_tensor(frames):
    # Input T,C,H,W uint8; shared preprocessing for training and inference.
    return (torch.from_numpy(np.stack(frames)).permute(1, 0, 2, 3).float()/255 - .45)/.225


def video_frames(path, require_10fps=True):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f'Cannot open video: {path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if require_10fps and not np.isclose(fps, 10, atol=.01):
        cap.release(); raise ValueError(f'Expected 10fps, got {fps}: {path}')
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            count += 1
            yield prepare_frame(frame)
        if count == 0 or count != total:
            raise ValueError(f'Incomplete decode: {path}, {count}/{total}')
    finally:
        cap.release()


def load_records(root):
    root = Path(root).resolve()
    manifest = pd.read_csv(root/'split_manifest.csv', dtype={'ID': str, 'route': str})
    if manifest.empty or manifest.ID.duplicated().any() or manifest[['ID','route','split','video']].isna().any().any():
        raise ValueError('Invalid split manifest')
    if set(manifest.split) != {'train','validation'} or manifest.groupby('route').split.nunique().max() != 1:
        raise ValueError('Route leakage or missing split')
    result = {}
    for part in ('train','validation'):
        labels = pd.read_csv(root/f'labels_{part}_candidate.csv', dtype={'ID': str})
        if list(labels.columns) != ['ID','frame_index','accel_label','steer_label'] or labels.isna().any().any():
            raise ValueError('Invalid label schema')
        subset = manifest[manifest.split == part]
        if set(labels.ID) != set(subset.ID): raise ValueError('Manifest/labels IDs differ')
        if not set(labels.accel_label) <= set(ACCEL) or not set(labels.steer_label) <= set(STEER):
            raise ValueError('Unknown class')
        groups = {sid: frame for sid, frame in labels.groupby('ID', sort=False)}
        records = []
        for row in subset.itertuples(index=False):
            frame = groups[row.ID]
            if not np.array_equal(frame.frame_index.to_numpy(), np.arange(len(frame))):
                raise ValueError(f'Noncontiguous indices: {row.ID}')
            path = (root/row.video).resolve()
            cap = cv2.VideoCapture(str(path))
            valid = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frame) and np.isclose(cap.get(cv2.CAP_PROP_FPS),10,atol=.01)
            cap.release()
            if not valid: raise ValueError(f'Video/label metadata mismatch: {path}')
            records.append(dict(ID=row.ID, route=row.route, path=path, accel=frame.accel_label.map(dict(zip(ACCEL,range(4)))).to_numpy(), steer=frame.steer_label.map(dict(zip(STEER,range(3)))).to_numpy()))
        result[part] = records
    paths = [r['path'] for rows in result.values() for r in rows]
    if len(paths) != len(set(paths)): raise ValueError('Same video is referenced by multiple IDs/splits')
    return result


def mixed_clip_groups(records, stride, epoch, video_buffer_size):
    """Shuffle endpoints across bounded groups of videos; visit each endpoint once."""
    if stride < 1 or video_buffer_size < 1: raise ValueError('Counts must be positive')
    rng = np.random.default_rng(SEED+epoch)
    order = rng.permutation(len(records))
    for start in range(0, len(order), video_buffer_size):
        group = [int(i) for i in order[start:start+video_buffer_size]]
        samples = [(i, end) for i in group for end in range(epoch % stride, len(records[i]['accel']), stride)]
        rng.shuffle(samples)
        yield group, samples


def mixed_train_batches(records, batch_size, stride, epoch, video_buffer_size):
    for group, samples in mixed_clip_groups(records, stride, epoch, video_buffer_size):
        frames = {i: list(video_frames(records[i]['path'])) for i in group}
        if any(len(frames[i]) != len(records[i]['accel']) for i in group):
            raise ValueError('Decoded training count mismatch')
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start+batch_size]
            x = torch.stack([clip_tensor([frames[i][t] for t in causal_indices(end)]) for i,end in batch])
            yield x, torch.tensor([int(records[i]['accel'][end]) for i,end in batch]), torch.tensor([int(records[i]['steer'][end]) for i,end in batch])
        del frames


def train_batches(records, batch_size, stride, epoch, video_buffer_size=1):
    if video_buffer_size > 1:
        yield from mixed_train_batches(records, batch_size, stride, epoch, video_buffer_size)
        return
    rng = np.random.default_rng(SEED+epoch)
    for idx in rng.permutation(len(records)):
        row = records[int(idx)]
        frames = list(video_frames(row['path']))  # one resized uint8 video in memory
        if len(frames) != len(row['accel']): raise ValueError('Decoded training count mismatch')
        endpoints = np.arange(epoch % stride, len(frames), stride)
        rng.shuffle(endpoints)
        for start in range(0,len(endpoints),batch_size):
            selected = endpoints[start:start+batch_size]
            x = torch.stack([clip_tensor([frames[i] for i in causal_indices(int(end))]) for end in selected])
            yield x, torch.tensor(row['accel'][selected]), torch.tensor(row['steer'][selected])


def multitask_loss(logits, accel, steer):
    a, s = logits
    accel_loss = F.cross_entropy(a.float(), accel)
    moving = accel != ACCEL.index('STOPPED')
    steer_loss = F.cross_entropy(s[moving].float(), steer[moving]) if moving.any() else s.sum()*0
    return accel_loss + steer_loss


def autocast(device):
    return torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()


def save_checkpoint(model, path, metadata=None):
    content = dict(model=model.state_dict(), format=FORMAT, frames=16, size=224, resize=256, mean=.45, std=.225, accel_classes=ACCEL, steer_classes=STEER, metadata=metadata or {})
    temporary = path.with_suffix('.temporary.pt')
    torch.save(content, temporary)
    temporary.replace(path)


def load_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    expected = dict(format=FORMAT, frames=16, size=224, resize=256, mean=.45, std=.225, accel_classes=ACCEL, steer_classes=STEER)
    if any(ckpt.get(k) != v for k,v in expected.items()):
        raise ValueError('Checkpoint preprocessing/class contract mismatch; legacy weights are initialization only')
    model = Stage3MViT()
    model.load_state_dict(ckpt['model'], strict=True)
    return model.to(device).eval()


def predict_video(model, path, sid, device, batch_size=4):
    history = deque(maxlen=16)
    pending, rows = [], []
    def flush():
        with torch.inference_mode(), autocast(device):
            a, s = model(torch.stack(pending).to(device))
        for ai, si in zip(a.argmax(1).cpu().tolist(),s.argmax(1).cpu().tolist()):
            rows.append(dict(ID=sid, sample_index=len(rows), accel_label=ACCEL[ai], steer_label=STEER[si]))
        pending.clear()
    for frame in video_frames(path, require_10fps=False):
        if not history: history.extend([frame]*15)
        history.append(frame)
        pending.append(clip_tensor(history))
        if len(pending) == batch_size: flush()
    if pending: flush()
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def predict_paths(model, paths, device, batch_size=4):
    if len({p.stem for p in paths}) != len(paths): raise ValueError('Duplicate video IDs')
    frames = [predict_video(model,p,p.stem,device,batch_size) for p in paths]
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)


def predict_stage3(data_dir, model_dir):
    """Competition entrypoint: independent dense predictions, CUDA, offline."""
    device = device_for('cuda')
    checkpoint = Path(model_dir)/'best.pt'  # official caller passes model/stage3
    if not checkpoint.is_file(): checkpoint = Path(model_dir)/'stage3'/'best.pt'
    model = load_model(checkpoint, device)
    paths = sorted(p for p in (Path(data_dir)/'videos').iterdir() if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv','.hevc'})
    return predict_paths(model,paths,device)


def classification_metrics(target, prediction, classes):
    matrix = np.zeros((len(classes),len(classes)),dtype=np.int64)
    for truth, pred in zip(target,prediction): matrix[int(truth),int(pred)] += 1
    denominator = matrix.sum(0)+matrix.sum(1)
    f1 = np.divide(2*np.diag(matrix),denominator,out=np.zeros(len(classes),dtype=float),where=denominator>0)
    return dict(samples=int(matrix.sum()), macro_f1=float(f1.mean()) if matrix.sum() else None, per_class_f1=dict(zip(classes,f1.tolist())), confusion_matrix=matrix.tolist())


def evaluate(model, records, device, batch_size=4, prediction_path=None):
    model.eval()
    all_a, all_s, predicted_a, predicted_s, outputs = [], [], [], [], []
    for row in records:
        pred = predict_video(model,row['path'],row['ID'],device,batch_size)
        if len(pred) != len(row['accel']): raise ValueError('Evaluation count mismatch')
        all_a.extend(row['accel'].tolist()); all_s.extend(row['steer'].tolist())
        predicted_a.extend(pred.accel_label.map(dict(zip(ACCEL,range(4)))).tolist())
        predicted_s.extend(pred.steer_label.map(dict(zip(STEER,range(3)))).tolist())
        outputs.append(pred)
    truth_a, truth_s = np.array(all_a), np.array(all_s)
    moving = truth_a != ACCEL.index('STOPPED')  # ground truth mask, never predicted STOPPED
    result = dict(accel=classification_metrics(all_a,predicted_a,ACCEL), moving_steer=classification_metrics(truth_s[moving],np.array(predicted_s)[moving],STEER), stopped_excluded=int((~moving).sum()), note='Fixed-class macro-F1 on provisional external labels; not a competition score.')
    if prediction_path is not None: pd.concat(outputs,ignore_index=True).to_csv(prediction_path,index=False)
    return result


def train(args):
    device = device_for(args.device)
    records = load_records(args.dataset_dir)
    out = args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    model = Stage3MViT()
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(ckpt['model'],strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best = -1.; history=[]
    source = Path(args.dataset_dir)
    fingerprints = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('split_manifest.csv','labels_train_candidate.csv','labels_validation_candidate.csv')}
    (out/'data_fingerprints.json').write_text(json.dumps(fingerprints,indent=2),encoding='utf-8')
    model_dir=out/'model'/'stage3'; model_dir.mkdir(parents=True)
    (out/'training_config.json').write_text(json.dumps(dict(epochs=args.epochs, epoch_offset=args.epoch_offset, video_buffer_size=args.video_buffer_size, batch_size=args.batch_size, train_stride=args.train_stride, learning_rate=args.learning_rate, optimizer_restarted=True, init_checkpoint_sha256=hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest() if args.init_checkpoint else None),indent=2),encoding='utf-8')
    for epoch in range(args.epoch_offset, args.epoch_offset+args.epochs):
        model.train(); total_loss=0.; steps=0; started=time.monotonic()
        for x,a,s in train_batches(records['train'],args.batch_size,args.train_stride,epoch,args.video_buffer_size):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device): loss=multitask_loss(model(x.to(device)),a.to(device),s.to(device))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer); scaler.update()
            total_loss+=float(loss.detach()); steps+=1
            if steps % 25 == 0: print(f'epoch={epoch+1} step={steps} loss={total_loss/steps:.4f}',flush=True)
        if not steps: raise ValueError('No training samples')
        metrics=evaluate(model,records['validation'],device,args.batch_size)
        score=metrics['moving_steer']['macro_f1']
        if score is None: raise ValueError('No moving validation frames for model selection')
        row=dict(epoch=epoch+1,steps=steps,mean_loss=total_loss/steps,seconds=time.monotonic()-started,validation=metrics)
        history.append(row)
        if score>best:
            best=score; save_checkpoint(model,model_dir/'best.pt',dict(epoch=epoch+1,validation=metrics,data_sha256=fingerprints,seed=SEED,train_stride=args.train_stride,video_buffer_size=args.video_buffer_size))
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    del optimizer, model
    if device.type=='cuda': torch.cuda.empty_cache()
    reloaded=load_model(model_dir/'best.pt',device)
    metrics=evaluate(reloaded,records['validation'],device,args.batch_size,out/'validation_predictions.csv')
    (out/'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')


def smoke(args):
    device=device_for(args.device); records=load_records(args.dataset_dir)
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    model=Stage3MViT().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    x,a,s=next(train_batches(records['train'][:1],1,8,0))
    started=time.monotonic(); model.train()
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    with autocast(device): loss=multitask_loss(model(x.to(device)),a.to(device),s.to(device))
    if not torch.isfinite(loss): raise RuntimeError('Nonfinite smoke loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(),1.)
    scaler.step(optimizer);scaler.update();model.eval()
    with torch.inference_mode(): reference=[v.cpu() for v in model(x.to(device))]
    directory=out/'model'/'stage3'; directory.mkdir(parents=True)
    save_checkpoint(model,directory/'best.pt',dict(smoke_only=True))
    del model,optimizer
    if device.type=='cuda': torch.cuda.empty_cache()
    model=load_model(directory/'best.pt',device)
    with torch.inference_mode(): actual=[v.cpu() for v in model(x.to(device))]
    for before,after in zip(reference,actual): torch.testing.assert_close(before,after)
    # Four-frame real-image input covers initial padding and a partial batch.
    videos=out/'input'/'videos'; videos.mkdir(parents=True)
    path=videos/'SMOKE_S3.mp4'
    cap=cv2.VideoCapture(str(records['train'][0]['path']))
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
    for _ in range(4):
        ok,frame=cap.read()
        if not ok: raise ValueError('Smoke source decode failed')
        writer.write(cv2.resize(frame,(320,240)))
    writer.release();cap.release()
    predictions=predict_paths(model,[path],device,2)
    if list(predictions.columns)!=OUTPUT_COLUMNS or predictions.sample_index.tolist()!=list(range(4)): raise ValueError('Dense prediction contract failed')
    predictions.to_csv(out/'predictions.csv',index=False)
    if device.type=='cuda':
        del model; torch.cuda.empty_cache()
        public=predict_stage3(out/'input',out/'model')
        pd.testing.assert_frame_equal(predictions,public)
    report=dict(status='PASS',device=str(device),architecture='mvit_v2_s',training_steps=1,loss=float(loss.detach()),prediction_rows=4,checkpoint_reload=True,public_cuda_entrypoint_tested=device.type=='cuda',seconds=time.monotonic()-started,note='One real-data optimization step; not trained model performance.')
    (out/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); subs=parser.add_subparsers(dest='command',required=True)
    for command in ('audit','train','smoke','evaluate','predict'):
        p=subs.add_parser(command)
        p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        p.add_argument('--threads',type=int,default=4)
        p.add_argument('--batch-size',type=int,default=2)
        if command!='predict': p.add_argument('--dataset-dir',type=Path,required=True)
        if command in ('train','smoke'): p.add_argument('--output-dir',type=Path,required=True)
        if command=='train':
            p.add_argument('--epochs',type=int,default=3);p.add_argument('--train-stride',type=int,default=8)
            p.add_argument('--video-buffer-size',type=int,default=1)
            p.add_argument('--epoch-offset',type=int,default=0)
            p.add_argument('--learning-rate',type=float,default=1e-4)
            group=p.add_mutually_exclusive_group(required=True)
            group.add_argument('--init-checkpoint',type=Path);group.add_argument('--from-scratch',action='store_true')
        if command in ('evaluate','predict'):
            p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if command=='predict': p.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.batch_size<1 or args.threads<1 or getattr(args,'epochs',1)<1 or getattr(args,'train_stride',1)<1: parser.error('Counts must be positive')
    if getattr(args,'video_buffer_size',1)<1 or getattr(args,'epoch_offset',0)<0: parser.error('Invalid buffer size or epoch offset')
    torch.set_num_threads(args.threads);cv2.setNumThreads(1)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if args.command=='audit':
        records=load_records(args.dataset_dir)
        print(json.dumps({part:dict(videos=len(rows),routes=len({r['route'] for r in rows}),samples=sum(len(r['accel']) for r in rows)) for part,rows in records.items()},indent=2));return
    if args.command=='train': train(args)
    elif args.command=='smoke': smoke(args)
    else:
        device=device_for(args.device);model=load_model(args.checkpoint,device)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.command=='evaluate':
            metrics=evaluate(model,load_records(args.dataset_dir)['validation'],device,args.batch_size,args.output.with_suffix('.predictions.csv'))
            args.output.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        else:
            paths=sorted(p for p in (args.data_dir/'videos').iterdir() if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv','.hevc'})
            predict_paths(model,paths,device,args.batch_size).to_csv(args.output,index=False)


if __name__=='__main__': main()