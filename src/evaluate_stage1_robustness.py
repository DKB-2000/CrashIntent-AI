"""Score Stage1 paired stress conditions using the actual submission decoder.

FP32 CPU is a local diagnostic. CUDA FP16 reproduces serving precision.
No fitting, threshold tuning, checkpoint modification or cross-file inference.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s

from prepare_stage1_robustness import CONDITIONS, ORIGINAL_CONDITIONS, digest, select_sources, write_json


def audit_dataset(root):
    state = json.loads((root / 'status.json').read_text())
    if state['status'] != 'DATA_VALIDATED' or state['manifest_sha256'] != digest(root / 'generation_manifest.csv'):
        raise ValueError('Dataset is incomplete or manifest changed')
    plan = json.loads((root / 'plan.json').read_text())
    if state['plan_sha256'] != digest(root / 'plan.json'):
        raise ValueError('Plan changed')
    if digest(root / 'training_manifest.csv') != plan['training_manifest_sha256']:
        raise ValueError('Original split manifest changed')
    original = pd.read_csv(root / 'training_manifest.csv', dtype={'source_id': str, 'upload_group': str})
    excluded = set(pd.read_csv(root / 'excluded_sources.csv', dtype=str).source_id)
    selected = select_sources(original, excluded, plan['per_group'])
    expected_sources = [{k: r[k] for k in ('source_id', 'upload_group', 'source_sha256')} for r in selected]
    if expected_sources != plan['sources'] or plan['conditions'] != list(CONDITIONS):
        raise ValueError('Source selection or conditions changed')
    table = pd.read_csv(root / 'generation_manifest.csv', dtype={'source_id': str, 'upload_group': str})
    if table.output_path.duplicated().any() or set(table.source_id) != {r['source_id'] for r in selected}:
        raise ValueError('Unexpected or duplicate videos')
    source_map = {r['source_id']: r for r in selected}
    for sid, part in table.groupby('source_id'):
        if len(part) != len(CONDITIONS) or set(part.condition) != set(CONDITIONS):
            raise ValueError('Missing/duplicate paired conditions')
        for row in part.to_dict('records'):
            expected_label = 'ORIGINAL' if row['condition'] in ORIGINAL_CONDITIONS else 'RERECORDED'
            if row['label'] != expected_label or row['split'] != 'val':
                raise ValueError('Invalid proxy label/split')
            for key in ('upload_group', 'source_sha256'):
                if row[key] != source_map[sid][key]:
                    raise ValueError('Source provenance mismatch')
            path = (root / row['output_path']).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != row['sha256']:
                raise ValueError('Video path/hash mismatch')
    return table


def summarize(predictions):
    """Condition-specific rates; no misleading Macro-F1 over 5:12 variants."""
    table = pd.DataFrame(predictions)
    if table.empty or table[['source_id', 'condition']].duplicated().any():
        raise ValueError('Empty/duplicate predictions')
    for _, part in table.groupby('source_id'):
        if len(part) != len(CONDITIONS) or set(part.condition) != set(CONDITIONS):
            raise ValueError('Incomplete paired predictions')
    if not np.isfinite(table.probability).all() or not table.probability.between(0, 1).all():
        raise ValueError('Invalid probability')
    if not (table.answer == np.where(table.probability >= .5, 'RERECORDED', 'ORIGINAL')).all():
        raise ValueError('Prediction threshold mismatch')
    indexed = table.set_index(['source_id', 'condition'])
    summaries, pairs = [], []
    for condition in CONDITIONS:
        part = table[table.condition == condition].sort_values('source_id')
        reference_name = 'original_clean' if condition in ORIGINAL_CONDITIONS else 'rerecorded_full'
        reference = indexed.loc[[(sid, reference_name) for sid in part.source_id]].reset_index(drop=True)
        probabilities = part.probability.to_numpy()
        reference_probabilities = reference.probability.to_numpy()
        positive = probabilities >= .5
        reference_positive = reference_probabilities >= .5
        item = dict(condition=condition, label=part.target.iloc[0], videos=len(part),
                    upload_groups=part.upload_group.nunique(), reference=reference_name,
                    mean_probability=float(probabilities.mean()),
                    positive_rate=float(positive.mean()),
                    mean_probability_delta=float((probabilities - reference_probabilities).mean()),
                    positive_rate_delta=float(positive.mean() - reference_positive.mean()),
                    original_to_rerecorded=int((positive & ~reference_positive).sum()),
                    rerecorded_to_original=int((~positive & reference_positive).sum()))
        item['false_positive_rate' if condition in ORIGINAL_CONDITIONS else 'recall'] = item['positive_rate']
        summaries.append(item)
        for row, base_p in zip(part.to_dict('records'), reference_probabilities):
            pairs.append(dict(source_id=row['source_id'], upload_group=row['upload_group'],
                              condition=condition, reference=reference_name,
                              probability=row['probability'], reference_probability=float(base_p),
                              probability_delta=float(row['probability'] - base_p),
                              target=row['target'], answer=row['answer']))
    return summaries, pairs


def verify_predictions(table, predictions):
    expected = table.set_index('output_path')
    if len(predictions) != len(table) or {r['output_path'] for r in predictions} != set(expected.index):
        raise ValueError('Prediction coverage mismatch')
    for row in predictions:
        original = expected.loc[row['output_path']]
        for key in ('source_id', 'upload_group', 'condition', 'sha256'):
            if row[key] != original[key]:
                raise ValueError('Prediction provenance mismatch')
        if row['answer'] != ('RERECORDED' if row['probability'] >= .5 else 'ORIGINAL'):
            raise ValueError('Prediction threshold mismatch')
        if row['target'] != original['label']:
            raise ValueError('Prediction label mismatch')
        logits = np.asarray(row['slot_logits'], dtype=np.float64)
        probabilities = np.asarray(row['slot_probabilities'], dtype=np.float64)
        if logits.shape != (3, 2) or probabilities.shape != (3,) or not np.isfinite(logits).all():
            raise ValueError('Invalid slot outputs')
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        recomputed = exp[:, 1] / exp.sum(axis=1)
        if not np.allclose(recomputed, probabilities, atol=5e-4, rtol=0):
            raise ValueError('Slot softmax mismatch')
        if not np.isclose(probabilities.mean(), row['probability'], atol=1e-12, rtol=0):
            raise ValueError('Slot aggregation mismatch')


def load_resume(out, table, identity):
    previous = json.loads((out / 'status.json').read_text())
    if previous.get('status') not in ('FAILED', 'PAUSED'):
        raise ValueError('Resume requires a stopped FAILED or PAUSED evaluation')
    for key in ('scope', 'expected_videos', 'checkpoint_sha256', 'serving_sha256',
                'manifest_sha256', 'precision', 'threshold', 'torch_version'):
        if previous.get(key) != identity.get(key):
            raise ValueError(f'Resume identity mismatch: {key}')
    saved = [json.loads(line) for line in (out / 'predictions.jsonl').read_text().splitlines()]
    paths = [r['output_path'] for r in saved]
    if paths != table.output_path.tolist()[:len(saved)]:
        raise ValueError('Resume outputs must be an exact ordered prefix')
    verify_predictions(table.iloc[:len(saved)], saved)
    return previous, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Omit to evaluate baseline Stage1 from serving ZIP')
    parser.add_argument('--serving-zip', type=Path, default=Path('artifacts/first-submit-candidate-v3/submit.zip'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Verify and reuse a failed run prefix')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--source-limit', type=int, help='Smoke only: output explicitly marked SUBSET')
    parser.add_argument('--max-seconds', type=int, default=14400)
    parser.add_argument('--max-new-videos', type=int, default=0, help='Pause after this many new videos; 0 means all')
    args = parser.parse_args()
    if args.max_new_videos < 0 or args.threads < 1 or args.max_seconds < 1 or (args.source_limit is not None and args.source_limit < 1):
        raise ValueError('Positive limits required')
    from advance_stage1_release import tick_lock
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tick_lock(args.output_dir.parent / (args.output_dir.name + '.evaluation.lock')) as acquired:
        if not acquired:
            raise RuntimeError('Another evaluator owns this output')
        root = args.dataset_dir.resolve()
        table = audit_dataset(root)
        if args.source_limit is not None:
            selected = sorted(table.source_id.unique())[:args.source_limit]
            table = table[table.source_id.isin(selected)].copy()
        out = args.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=args.resume)
        started = time.monotonic()
        torch.set_num_threads(args.threads)
        import cv2
        cv2.setNumThreads(1)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        device = torch.device(args.device)
        with zipfile.ZipFile(args.serving_zip) as archive:
            serving_code = archive.read('inference.py')
            checkpoint_bytes = args.checkpoint.read_bytes() if args.checkpoint else archive.read('model/stage1/best.pt')
        checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
        if args.resume and (out / 'serving_inference.py').read_bytes() != serving_code:
            raise ValueError('Resume serving code changed')
        (out / 'serving_inference.py').write_bytes(serving_code)
        spec = importlib.util.spec_from_file_location('stage1_robustness_serving', out / 'serving_inference.py')
        serving = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = serving
        spec.loader.exec_module(serving)
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
        if checkpoint.get('size') != 224 or checkpoint.get('frames') != 16:
            raise ValueError('Unexpected model input metadata')
        model = mvit_v2_s(weights=None)
        model.head[1] = nn.Linear(model.head[1].in_features, 2)
        model.load_state_dict(checkpoint['model'], strict=True)
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError('Nonfinite checkpoint')
        model.to(device).eval()
        # load_state_dict copied weights; serialized bytes and source tensors are no longer needed.
        del checkpoint_bytes, checkpoint
        import gc
        gc.collect()
        state = dict(status='RUNNING', scope='SUBSET_SMOKE' if args.source_limit else 'ALL_CONDITIONS',
                     expected_videos=len(table), completed_videos=0, checkpoint_sha256=checkpoint_sha,
                     serving_sha256=hashlib.sha256(serving_code).hexdigest(),
                     manifest_sha256=digest(root / 'generation_manifest.csv'),
                     evaluator_sha256=digest(__file__),
                     precision='CUDA_FP16' if device.type == 'cuda' else 'CPU_FP32_DIAGNOSTIC',
                     threshold=.5, torch_version=torch.__version__, real_phone_validation=False)
        predictions = []
        if args.resume:
            previous, predictions = load_resume(out, table, state)
            history = out / ('resume-' + str(time.time_ns()) + '.json')
            write_json(history, previous)
            state.update(completed_videos=len(predictions), resumed_videos=len(predictions),
                         previous_evaluator_sha256=previous['evaluator_sha256'])
        chunk_start = len(predictions)
        state.update(pid=os.getpid(), chunk_start=chunk_start, max_new_videos=args.max_new_videos)
        write_json(out / 'status.json', state)
        try:
            with torch.inference_mode(), (out / 'predictions.jsonl').open('a' if args.resume else 'x', encoding='utf-8') as stream:
                for row in table.iloc[len(predictions):].to_dict('records'):
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError('Evaluation budget reached')
                    path = root / row['output_path']
                    logits, probabilities = [], []
                    for slot in range(3):
                        clip = serving._decode_stage1_clip(path, 224, serving._clip_ids(path, 16, slot, 3))
                        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                            values = model(clip.unsqueeze(0).to(device))
                            probability = values.softmax(1)[0, 1].float().item()
                        logits.append(values[0].float().cpu().tolist())
                        probabilities.append(probability)
                    probability = float(np.mean(probabilities))
                    prediction = {k: row[k] for k in ('source_id', 'upload_group', 'condition', 'output_path', 'sha256')}
                    prediction.update(target=row['label'], probability=probability,
                                      answer='RERECORDED' if probability >= .5 else 'ORIGINAL',
                                      slot_logits=logits, slot_probabilities=probabilities)
                    stream.write(json.dumps(prediction, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    predictions.append(prediction)
                    state.update(completed_videos=len(predictions), seconds=time.monotonic() - started, heartbeat_unix=time.time())
                    write_json(out / 'status.json', state)
                    if len(predictions) % 17 == 0:
                        print(json.dumps(state), flush=True)
                    if args.max_new_videos and len(predictions) - chunk_start >= args.max_new_videos:
                        break
            # Independently reload saved outputs and recompute coverage, softmax and means.
            saved = [json.loads(line) for line in (out / 'predictions.jsonl').read_text().splitlines()]
            verify_predictions(table.iloc[:len(saved)], saved)
            if len(saved) < len(table):
                state.update(status='PAUSED', pause_reason='verified chunk complete', seconds=time.monotonic() - started)
                write_json(out / 'status.json', state)
                print(json.dumps(state), flush=True)
                return
            summaries, pairs = summarize(saved)
            pd.DataFrame(summaries).to_csv(out / 'condition_summary.csv', index=False)
            pd.DataFrame(pairs).to_csv(out / 'paired_changes.csv', index=False)
            state.update(status='PASS', seconds=time.monotonic() - started,
                         predictions_sha256=digest(out / 'predictions.jsonl'),
                         prediction_verification='coverage/provenance/slot-softmax/mean/threshold PASS',
                         conditions=summaries)
            write_json(out / 'report.json', state)
            write_json(out / 'status.json', state)
            print(json.dumps(state), flush=True)
        except Exception as error:
            state.update(status='FAILED', error=repr(error), seconds=time.monotonic() - started)
            write_json(out / 'status.json', state)
            raise

if __name__ == '__main__':
    main()
