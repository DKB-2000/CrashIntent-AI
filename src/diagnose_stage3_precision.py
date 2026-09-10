"""Training-only FP32 controls for collapsed Stage3 models; bounded wall time."""
from contextlib import nullcontext
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torchvision.ops import StochasticDepth
import stage3_pipeline as p


def select_samples(records):
    chosen, used = [], set()
    for accel in range(4):
        for steer in (range(3) if accel != 3 else [1]):
            options = []
            for i, row in enumerate(records):
                mask = (row['accel'] == accel) & (np.arange(len(row['accel'])) >= 15)
                if accel != 3:
                    mask &= row['steer'] == steer
                indices = np.flatnonzero(mask)
                if len(indices):
                    options.append((i, int(indices[len(indices)//2])))
            if not options:
                raise ValueError('Missing training class combination')
            i, end = next((v for v in options if v[0] not in used), options[0])
            used.add(i)
            chosen.append(dict(record=i, endpoint=end, ID=records[i]['ID'],
                               accel=accel, steer=int(records[i]['steer'][end])))
    return chosen


def disable_regularization(model):
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, StochasticDepth)):
            module.p = 0.0
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--max-seconds', type=float, default=2400)
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.max_seconds <= 3000:
        parser.error('Positive steps and max-seconds <= 3000 required')
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    p.cv2.setNumThreads(1)
    device = p.device_for('cuda')
    records = p.load_records(args.dataset_dir)['train']
    samples = select_samples(records)
    cache = {r['record']: list(p.video_frames(records[r['record']]['path'])) for r in samples}
    clips = torch.stack([p.clip_tensor([cache[r['record']][i] for i in p.causal_indices(r['endpoint'])]) for r in samples])
    del cache
    ta = torch.tensor([r['accel'] for r in samples], device=device)
    ts = torch.tensor([r['steer'] for r in samples], device=device)
    moving = ta != 3
    result = dict(status='RUNNING', scope='training-only diagnostic, not generalization',
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  samples=samples, runs=[], torch=torch.__version__, gpu=torch.cuda.get_device_name())

    def save():
        result['seconds'] = time.monotonic()-started
        (args.output_dir/'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    for initialization, regularization in [('scratch', True), ('scratch', False), ('trained', False)]:
        if time.monotonic()-started >= args.max_seconds:
            break
        random.seed(p.SEED)
        np.random.seed(p.SEED)
        torch.manual_seed(p.SEED)
        model = p.load_model(args.checkpoint, device) if initialization == 'trained' else p.Stage3MViT().to(device)
        disabled = 0 if regularization else disable_regularization(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01 if regularization else 0.)
        tracked = {'backbone': next(model.backbone.parameters()), 'accel': model.accel.weight, 'steer': model.steer.weight}
        before = {k: v.detach().clone() for k, v in tracked.items()}
        run = dict(initialization=initialization, precision='fp32', regularization=regularization,
                   disabled_modules=disabled, history=[], gradients=[], optimizer_steps=0, nonfinite_steps=0)
        result['runs'].append(run)
        rng = np.random.default_rng(p.SEED)

        def measure(step):
            model.eval()
            with torch.no_grad():
                logits = [model(x.to(device)) for x in clips.split(2)]
                a, s = [torch.cat([v[i] for v in logits]) for i in range(2)]
                if not torch.isfinite(a).all() or not torch.isfinite(s).all():
                    raise ValueError('Nonfinite evaluation logits')
                row = dict(step=step, loss=float(p.multitask_loss((a,s),ta,ts)),
                           accel_accuracy=float((a.argmax(1)==ta).float().mean()),
                           steer_accuracy=float((s.argmax(1)[moving]==ts[moving]).float().mean()),
                           accel_logits_std=a.std(0).cpu().tolist(), steer_logits_std=s.std(0).cpu().tolist(),
                           accel_predictions=a.argmax(1).cpu().tolist(), steer_predictions=s.argmax(1).cpu().tolist())
            run['history'].append(row)
            print(initialization, regularization, json.dumps(row, allow_nan=False), flush=True)
            save()
            return row

        measure(0)
        for step in range(1, args.steps+1):
            if time.monotonic()-started >= args.max_seconds:
                break
            ids = rng.choice(len(clips), size=2, replace=False)
            idx = torch.tensor(ids, device=device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = p.multitask_loss(model(clips[ids].to(device)), ta[idx], ts[idx])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite FP32 loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1 or step % 20 == 0:
                run['gradients'].append(dict(step=step, global_norm_before_clip=float(norm),
                    **{k: float(v.grad.norm()) if v.grad is not None else None for k,v in tracked.items()}))
            optimizer.step()
            run['optimizer_steps'] += 1
            if step % 20 == 0 or step == args.steps:
                row = measure(step)
                if row['accel_accuracy'] == 1. and row['steer_accuracy'] == 1.:
                    break
        if run['history'][-1]['step'] != run['optimizer_steps']:
            measure(run['optimizer_steps'])
        run['weight_changes'] = {k: float((v.detach()-before[k]).norm()) for k,v in tracked.items()}
        save()
        del optimizer, model, tracked, before
        torch.cuda.empty_cache()
    result['status'] = 'COMPLETED' if len(result['runs']) == 3 and time.monotonic()-started < args.max_seconds else 'TIME_LIMIT'
    save()


if __name__ == '__main__':
    main()
