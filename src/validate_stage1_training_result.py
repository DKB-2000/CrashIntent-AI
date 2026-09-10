"""Independently check a downloaded Stage1 training result against its input bundle."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

from stage1_pipeline import Stage1MViT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score(frame):
    values = []
    for label in ('ORIGINAL', 'RERECORDED'):
        actual, predicted = frame.target == label, frame.answer == label
        tp = int((actual & predicted).sum())
        denominator = int(actual.sum() + predicted.sum())
        values.append(2 * tp / denominator if denominator else 0.)
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    local_assets = json.loads((args.bundle_dir / 'dataset/stage1-training-assets.json').read_text())
    with zipfile.ZipFile(args.result) as result, zipfile.ZipFile(args.bundle_dir / 'dataset/stage1-training.bin') as inputs:
        require(len(result.namelist()) == len(set(result.namelist())), 'Duplicate result archive members')
        runner = json.loads(result.read('runner.json'))
        report = json.loads(result.read('training/report.json'))
        install = json.loads(result.read('install.json'))
        config = json.loads(inputs.read('run_config.json'))
        require(json.loads(result.read('input-assets.json')) == local_assets, 'Wrong input bundle')
        require(runner['failure'] is None and install['exit_code'] == 0, 'Runner or installation failed')
        require(runner['config'] == config, 'Run config mismatch')
        require(runner['seconds'] < (3600 if config['mode'] == 'trial' else 36000), 'GPU budget exceeded')
        require(report['status'] in ('COMPLETED', 'TIME_LIMIT'), 'Training did not produce a valid result')
        require(report['reload_matches'] and report['optimizer_steps'] > 0, 'No optimizer updates/reload verification')
        require(report['precision'] == 'FP32 training / FP16 serving evaluation', 'Precision mismatch')
        require(report['nonfinite_gradients'] == 0, 'Nonfinite gradients')
        require(report['baseline_sha256'] == hashlib.sha256(inputs.read('baseline.pt')).hexdigest(), 'Baseline mismatch')
        for name in ['src/train_stage1_full.py', 'src/serving_inference.py', 'dataset/generation_manifest.csv']:
            require(result.read(name) == inputs.read(name), f'Provenance mismatch: {name}')
        require(report['code_sha256'] == hashlib.sha256(result.read('src/train_stage1_full.py')).hexdigest(), 'Code digest mismatch')
        manifest = pd.read_csv(io.BytesIO(inputs.read('dataset/generation_manifest.csv')), dtype={'source_id': str, 'upload_group': str})
        require(len(manifest) == config['expected_samples'] and manifest.source_id.nunique() == config['expected_sources'], 'Input counts mismatch')
        require(report['manifest_sha256'] == config['dataset_manifest_sha256'], 'Manifest digest mismatch')
        require(manifest.groupby('upload_group').split.nunique().max() == 1, 'Upload-group leakage')
        excluded = set(pd.read_csv(io.BytesIO(inputs.read('dataset/excluded_sources.csv')), dtype=str).source_id)
        require(not set(manifest.source_id) & excluded, 'Phone holdout leakage')
        validation = manifest[manifest.split == 'val'].set_index('output_path')
        require(report['train_samples'] == int((manifest.split == 'train').sum()), 'Training sample count mismatch')
        balanced = report.get('training_policy') == 'balanced-v2'
        expected_epochs = config['epochs']
        epoch_samples = report['train_samples']
        if balanced:
            require(config.get('training_policy') == 'balanced-v2', 'Training policy mismatch')
            for key in ('learning_rate', 'min_updates', 'batch_size', 'accumulation'):
                require(report['config'][key] == config[key], f'Config mismatch: {key}')
            effective_batch = config['batch_size'] * config['accumulation']
            require(effective_batch > 0 and effective_batch % 2 == 0, 'Invalid effective batch')
            steps = math.ceil(manifest[manifest.split == 'train'].label.value_counts().max() / (effective_batch // 2))
            epoch_samples = steps * effective_batch
            expected_epochs = max(config['epochs'], math.ceil(config['min_updates'] / steps))
            require(report['effective_epochs'] == expected_epochs and report['epoch_samples'] == epoch_samples, 'Update plan mismatch')
            require(report['planned_optimizer_steps'] == expected_epochs * steps, 'Planned update mismatch')
            require(report['optimizer_steps'] == sum(e['optimizer_steps'] for e in report['epochs']), 'Update total mismatch')
            require(report['minimum_updates_met'] == (report['optimizer_steps'] >= config['min_updates']), 'Update floor mismatch')
            for epoch in report['epochs']:
                count = epoch['optimizer_steps'] * effective_batch
                require(epoch['train_samples_seen'] == count and epoch['class_exposures'] == [count // 2] * 2, 'Balanced exposure mismatch')
        scores = {}
        prediction_files = ['baseline_predictions.csv', 'validation_predictions.csv'] + [f'validation_epoch_{e["epoch"]}.csv' for e in report['epochs']]
        for name in prediction_files:
            frame = pd.read_csv(io.BytesIO(result.read('training/' + name)), dtype={'source_id': str, 'upload_group': str})
            require(not frame.path.duplicated().any() and set(frame.path) == set(validation.index), f'Validation rows mismatch: {name}')
            aligned = validation.loc[frame.path]
            require(frame.target.tolist() == aligned.label.tolist(), 'Targets differ from input manifest')
            require(frame.source_id.tolist() == aligned.source_id.tolist() and frame.upload_group.tolist() == aligned.upload_group.tolist(), 'Prediction provenance mismatch')
            require(np.isfinite(frame.probability).all() and frame.probability.between(0, 1).all(), 'Invalid probabilities')
            require(frame.answer.tolist() == np.where(frame.probability >= .5, 'RERECORDED', 'ORIGINAL').tolist(), 'Threshold mismatch')
            scores[name] = score(frame)
        require(abs(scores['baseline_predictions.csv'] - report['baseline_metrics']['macro_f1']) < 1e-12, 'Baseline score mismatch')
        require(abs(scores['validation_predictions.csv'] - report['best_macro_f1']) < 1e-12, 'Best score mismatch')
        best_predictions = result.read('training/validation_predictions.csv')
        selected_predictions = result.read(f'training/validation_epoch_{report["best_epoch"]}.csv')
        require(best_predictions == selected_predictions, 'Best predictions differ from selected epoch')
        for epoch in report['epochs']:
            require(abs(scores[f'validation_epoch_{epoch["epoch"]}.csv'] - epoch['macro_f1']) < 1e-12, 'Epoch score mismatch')
            if epoch['complete']:
                require(epoch['train_samples_seen'] == epoch_samples, 'Incomplete epoch marked complete')
        require(abs(max(e['macro_f1'] for e in report['epochs']) - report['best_macro_f1']) < 1e-12, 'Best checkpoint selection mismatch')
        if report['status'] == 'COMPLETED':
            require(len(report['epochs']) == expected_epochs and all(e['complete'] for e in report['epochs']) and (not balanced or report['minimum_updates_met']), 'False completion')
        checkpoint_bytes = result.read('training/best.pt')
        require(hashlib.sha256(checkpoint_bytes).hexdigest() == report['checkpoint_sha256'], 'Checkpoint digest mismatch')
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
        require(checkpoint['size'] == 224 and checkpoint['frames'] == 16, 'Serving metadata mismatch')
        model = Stage1MViT().net
        model.load_state_dict(checkpoint['model'], strict=True)
        require(all(torch.isfinite(t).all() for t in model.state_dict().values()), 'Nonfinite checkpoint')
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / 'best.pt').write_bytes(checkpoint_bytes)
        for name in ['report.json', 'baseline_predictions.csv', 'validation_predictions.csv']:
            (args.output_dir / name).write_bytes(result.read('training/' + name))
    summary = dict(status='PASS', training_status=report['status'], completed_epochs=report['completed_epochs'],
                   optimizer_steps=report['optimizer_steps'], sources=config['expected_sources'],
                   train_samples=report['train_samples'], validation_samples=len(validation),
                   baseline_macro_f1=scores['baseline_predictions.csv'], best_macro_f1=scores['validation_predictions.csv'],
                   improvement=scores['validation_predictions.csv'] - scores['baseline_predictions.csv'],
                   result_sha256=hashlib.sha256(args.result.read_bytes()).hexdigest(),
                   checkpoint_sha256=report['checkpoint_sha256'], real_phone_validation_available=False)
    (args.output_dir / 'result-validation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
