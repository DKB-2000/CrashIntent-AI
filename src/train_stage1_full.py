"""FP32 Stage1 transfer learning with grouped validation and serving preprocessing.

The bundle supplies serving_inference.py extracted from the validated submission.
No public competition examples or real phone holdout are used for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s


LABELS = ['ORIGINAL', 'RERECORDED']
SEED = 20260903


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(predictions):
    matrix = [[sum(r['target'] == target and r['answer'] == answer for r in predictions)
               for answer in LABELS] for target in LABELS]
    f1 = []
    for i in range(2):
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1.append(2 * matrix[i][i] / denominator if denominator else 0.)
    return dict(macro_f1=sum(f1) / 2, per_class_f1=dict(zip(LABELS, f1)),
                confusion_matrix=matrix, labels=LABELS,
                prediction_counts={label: sum(r['answer'] == label for r in predictions) for label in LABELS})


def audit(dataset_dir):
    table = pd.read_csv(dataset_dir / 'generation_manifest.csv', dtype={'source_id': str, 'upload_group': str})
    excluded = set(pd.read_csv(dataset_dir / 'excluded_sources.csv', dtype=str).source_id)
    if table.empty or table.output_path.duplicated().any() or table[['source_id', 'upload_group', 'split', 'label']].isna().any().any():
        raise ValueError('Empty, duplicate or incomplete manifest')
    if set(table.split) != {'train', 'val'} or set(table.label) != set(LABELS):
        raise ValueError('Invalid splits or labels')
    for key in ('source_id', 'upload_group'):
        if table.groupby(key).split.nunique().max() != 1:
            raise ValueError(f'{key} leakage')
    if set(table.source_id) & excluded:
        raise ValueError('Phone holdout leakage')
    if any(set(part.label) != set(LABELS) for _, part in table.groupby('split')):
        raise ValueError('Both classes required in each partition')
    for row in table.to_dict('records'):
        path = (dataset_dir / row['output_path']).resolve()
        if not path.is_relative_to(dataset_dir.resolve()) or digest(path) != row['sha256']:
            raise ValueError(f'Input checksum/path mismatch: {row["output_path"]}')
    report = json.loads((dataset_dir / 'generation_report.json').read_text())
    if report['manifest_sha256'] != digest(dataset_dir / 'generation_manifest.csv'):
        raise ValueError('Manifest checksum mismatch')
    return table


class Clips(Dataset):
    def __init__(self, dataset_dir, rows, training=False):
        self.root, self.rows, self.training = dataset_dir, rows, training

    def __len__(self):
        return len(self.rows) if self.training else len(self.rows) * 3

    def __getitem__(self, index):
        import serving_inference as serving
        cv2.setNumThreads(1)
        row_index, slot = (index, random.randrange(3)) if self.training else divmod(index, 3)
        row = self.rows[row_index]
        path = self.root / row['output_path']
        # Use the actual serving decoder; training fails on invalid inputs.
        clip = serving._decode_stage1_clip(path, 224, serving._clip_ids(path, 16, slot, 3))
        return clip, LABELS.index(row['label']), row_index


def loader(dataset, batch_size, workers, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


class BalancedBatches(Sampler):
    """Cover each class once before cycling; balance every optimizer batch."""
    def __init__(self, labels, batch_size, seed=SEED):
        if batch_size < 2 or batch_size % 2:
            raise ValueError('Effective batch size must be positive and even')
        self.groups = [[i for i, label in enumerate(labels) if label == c] for c in LABELS]
        if any(not group for group in self.groups) or sum(map(len, self.groups)) != len(labels):
            raise ValueError('Both valid classes required')
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.steps = math.ceil(max(map(len, self.groups)) / (batch_size // 2))

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        needed = self.steps * self.batch_size // 2
        streams = []
        for group in self.groups:
            stream = []
            while len(stream) < needed:
                cycle = list(group)
                rng.shuffle(cycle)
                stream.extend(cycle)
            streams.append(stream[:needed])
        half = self.batch_size // 2
        for offset in range(0, needed, half):
            batch = streams[0][offset:offset+half] + streams[1][offset:offset+half]
            rng.shuffle(batch)
            yield batch


def planned_epochs(requested, minimum_updates, steps_per_epoch):
    if min(requested, minimum_updates, steps_per_epoch) < 1:
        raise ValueError('Epochs, update floor and epoch steps must be positive')
    return max(requested, math.ceil(minimum_updates / steps_per_epoch))


def accumulated_step(model, optimizer, clips, targets, microbatch_size, device, weights=None):
    """One update, normalized once across the entire effective batch."""
    if len(targets) == 0 or microbatch_size < 1:
        raise ValueError('Nonempty batch and positive microbatch required')
    targets = targets.to(device)
    denominator = len(targets) if weights is None else weights[targets].sum()
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.
    for start in range(0, len(targets), microbatch_size):
        x = clips[start:start+microbatch_size].to(device, non_blocking=True)
        y = targets[start:start+microbatch_size]
        loss = nn.functional.cross_entropy(model(x).float(), y, weight=weights, reduction='sum') / denominator
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        loss_value += float(loss.detach())
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return loss_value, float(norm)


def evaluate(model, data_loader, rows, device):
    model.eval()
    sums = np.zeros(len(rows), dtype=np.float64)
    counts = np.zeros(len(rows), dtype=np.int64)
    with torch.inference_mode():
        for clips, _, ids in data_loader:
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                probabilities = model(clips.to(device, non_blocking=True)).softmax(1)[:, 1]
            probabilities = probabilities.float().cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError('Nonfinite validation probabilities')
            np.add.at(sums, ids.numpy(), probabilities)
            np.add.at(counts, ids.numpy(), 1)
    if not (counts == 3).all():
        raise ValueError('Expected exactly three slots per validation video')
    return [dict(path=row['output_path'], source_id=row['source_id'], upload_group=row['upload_group'],
                 variant=int(row['variant']), target=row['label'], probability=float(probability),
                 answer=LABELS[int(probability >= .5)]) for row, probability in zip(rows, sums / counts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--min-updates', type=int, default=120)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-seconds', type=int, default=33000)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 34200 or min(args.epochs, args.batch_size, args.accumulation, args.min_updates) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.workers < 0:
        raise ValueError('Invalid budget or training parameters')
    started = time.monotonic()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    table = audit(args.dataset_dir)
    rows = {split: table[table.split == split].to_dict('records') for split in ('train', 'val')}
    sampler = BalancedBatches([r['label'] for r in rows['train']], args.batch_size * args.accumulation)
    training = DataLoader(Clips(args.dataset_dir, rows['train'], True), batch_sampler=sampler,
                          num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    effective_epochs = planned_epochs(args.epochs, args.min_updates, len(sampler))
    validation = loader(Clips(args.dataset_dir, rows['val']), 4, args.workers)
    report = dict(status='RUNNING', scope='Synthetic upload-group validation; real phone validation unavailable',
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  training_policy='balanced-v2', effective_epochs=effective_epochs,
                  planned_optimizer_steps=effective_epochs * len(sampler),
                  epoch_samples=len(sampler) * sampler.batch_size,
                  effective_batch_size=sampler.batch_size, loss_normalization='effective-batch mean; no class weights',
                  regularization='dropout=0, stochastic_depth=0, weight_decay=0; no head warmup',
                  seed=SEED, precision='FP32 training / FP16 serving evaluation', optimizer_steps=0,
                  nonfinite_gradients=0, epochs=[], train_samples=len(rows['train']), validation_samples=len(rows['val']),
                  train_groups=len({r['upload_group'] for r in rows['train']}),
                  validation_groups=len({r['upload_group'] for r in rows['val']}),
                  manifest_sha256=digest(args.dataset_dir / 'generation_manifest.csv'),
                  baseline_sha256=digest(args.baseline), code_sha256=digest(__file__),
                  pretrained_url=MViT_V2_S_Weights.KINETICS400_V1.url,
                  torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name())

    def save_report():
        report['seconds'] = time.monotonic() - started
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    # Compare with the first-submission Stage1 model on precisely the same rows.
    model = mvit_v2_s(weights=None)
    model.head[1] = nn.Linear(model.head[1].in_features, 2)
    baseline = torch.load(args.baseline, map_location='cpu', weights_only=True)
    model.load_state_dict(baseline['model'], strict=True)
    del baseline
    model.to(device)
    baseline_predictions = evaluate(model, validation, rows['val'], device)
    pd.DataFrame(baseline_predictions).to_csv(out / 'baseline_predictions.csv', index=False)
    report['baseline_metrics'] = metrics(baseline_predictions)
    print(json.dumps(dict(baseline=report['baseline_metrics'])), flush=True)
    del model
    torch.cuda.empty_cache()
    model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
    model.head[1] = nn.Linear(model.head[1].in_features, 2)
    model.to(device)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
        if module.__class__.__name__ == 'StochasticDepth':
            module.p = 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    best = -1.
    save_report()
    # Reserve time for a full validation sweep and checkpoint reload checks.
    reserve = max(180., report['seconds'] * 1.5)
    for epoch in range(1, effective_epochs + 1):
        if time.monotonic() - started >= args.max_seconds - reserve:
            break
        sampler.epoch = epoch - 1
        model.train()
        losses, gradients = [], []
        samples_seen = 0
        epoch_start = time.monotonic()
        complete = True
        head_before = model.head[1].weight.detach().clone()
        class_exposures = [0, 0]
        epoch_updates = 0
        for step, (clips, targets, _) in enumerate(training):
            if time.monotonic() - started >= args.max_seconds - reserve:
                complete = False
                break
            counts = torch.bincount(targets, minlength=2).tolist()
            if counts != [sampler.batch_size // 2] * 2:
                raise ValueError('Unbalanced optimizer batch')
            loss, norm = accumulated_step(model, optimizer, clips, targets, args.batch_size, device)
            report['optimizer_steps'] += 1
            epoch_updates += 1
            gradients.append(norm)
            losses.append(loss)
            samples_seen += len(targets)
            class_exposures = [a + b for a, b in zip(class_exposures, counts)]
            if (step + 1) % 25 == 0:
                print(json.dumps(dict(epoch=epoch, update=step + 1, updates=len(training), loss=float(np.mean(losses[-25:])),
                                      optimizer_steps=report['optimizer_steps'], seconds=time.monotonic() - started)), flush=True)
        if not losses:
            break
        val_started = time.monotonic()
        predictions = evaluate(model, validation, rows['val'], device)
        val_seconds = time.monotonic() - val_started
        reserve = max(reserve, val_seconds * 2 + 180)
        item = dict(epoch=epoch, complete=complete, optimizer_steps=epoch_updates, class_exposures=class_exposures, train_loss=float(np.mean(losses)), train_samples_seen=samples_seen,
                    gradient_norm_mean=float(np.mean(gradients)) if gradients else None,
                    head_weight_change=float((model.head[1].weight.detach() - head_before).norm()),
                    seconds=time.monotonic() - epoch_start, validation_seconds=val_seconds, **metrics(predictions))
        report['epochs'].append(item)
        print(json.dumps(item), flush=True)
        pd.DataFrame(predictions).to_csv(out / f'validation_epoch_{epoch}.csv', index=False)
        if item['macro_f1'] > best:
            best = item['macro_f1']
            report['best_epoch'] = epoch
            torch.save(dict(model=model.state_dict(), frames=16, size=224), out / 'best.pt')
            pd.DataFrame(predictions).to_csv(out / 'validation_predictions.csv', index=False)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                        complete_epoch=complete, frames=16, size=224), out / 'last_training.pt')
        save_report()
        if not complete:
            break
    if not (out / 'best.pt').exists():
        raise TimeoutError('Budget ended without a trained checkpoint')
    checkpoint = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in checkpoint['model'].values()):
        raise ValueError('Nonfinite saved weights')
    del checkpoint
    actual = evaluate(model, validation, rows['val'], device)
    expected = pd.read_csv(out / 'validation_predictions.csv')
    if [r['answer'] for r in actual] != expected.answer.tolist() or not np.allclose([r['probability'] for r in actual], expected.probability, atol=1e-6, rtol=0):
        raise ValueError('Reload predictions differ')
    complete = len(report['epochs']) == effective_epochs and all(e['complete'] for e in report['epochs']) and report['optimizer_steps'] >= args.min_updates
    report.update(status='COMPLETED' if complete else 'TIME_LIMIT', reload_matches=True,
                  minimum_updates_met=report['optimizer_steps'] >= args.min_updates,
                  best_metrics=metrics(actual), best_macro_f1=best,
                  completed_epochs=sum(e['complete'] for e in report['epochs']),
                  checkpoint_sha256=digest(out / 'best.pt'), max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    save_report()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
