"""Build a private, checksummed Stage1 GPU bundle without publishing it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['trial', 'full'], required=True)
    parser.add_argument('--baseline-zip', type=Path, default=Path('artifacts/first-submit-candidate-v3/submit.zip'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir
    dataset = out / 'dataset'
    kernel = out / 'kernel'
    table = pd.read_csv(args.dataset_dir / 'generation_manifest.csv', dtype=str)
    report = json.loads((args.dataset_dir / 'generation_report.json').read_text())
    if report['status'] != 'PASS' or report['samples'] != len(table):
        raise ValueError('Dataset not complete')
    if args.mode == 'full' and (report['sources'] != 1110 or len(table) != 3330
                               or report['splits_by_source'] != {'train': 880, 'val': 230}):
        raise ValueError('Full training requires all 1110 sources / 3330 samples, not the preview manifest')
    if hashlib.sha256((args.dataset_dir / 'generation_manifest.csv').read_bytes()).hexdigest() != report['manifest_sha256']:
        raise ValueError('Manifest checksum mismatch')
    dataset.mkdir(parents=True, exist_ok=False)
    kernel.mkdir(parents=True, exist_ok=False)
    slug = f'crashintent-stage1-{args.mode}-v1'
    config = dict(mode=args.mode, epochs=6, training_policy='balanced-v2', learning_rate=1e-4, min_updates=120, batch_size=2, accumulation=4, max_seconds=2600 if args.mode == 'trial' else 33000,
                  expected_sources=report['sources'], expected_samples=len(table),
                  dataset_manifest_sha256=report['manifest_sha256'])
    with zipfile.ZipFile(dataset / 'stage1-training.bin', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in ['stage1_pipeline.py', 'train_stage1_full.py', 'prepare_stage1_training.py',
                     'prepare_stage1_rerecorded.py', 'select_stage1_holdout.py']:
            archive.write(root / 'src' / name, 'src/' + name)
        for name in ['generation_manifest.csv', 'generation_report.json', 'source_splits.csv',
                     'phone_holdout.csv', 'excluded_sources.csv', 'labels.csv', 'labels_train.csv', 'labels_val.csv']:
            archive.write(args.dataset_dir / name, 'dataset/' + name)
        for row in table.to_dict('records'):
            path = (args.dataset_dir / row['output_path']).resolve()
            if not path.is_relative_to(args.dataset_dir.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Dataset hash/path mismatch')
            archive.write(path, 'dataset/' + row['output_path'])
        with zipfile.ZipFile(args.baseline_zip) as baseline:
            archive.writestr('baseline.pt', baseline.read('model/stage1/best.pt'))
            archive.writestr('src/serving_inference.py', baseline.read('inference.py'))
            archive.writestr('requirements.txt', baseline.read('requirements.txt'))
        archive.writestr('run_config.json', json.dumps(config, indent=2))
    sha = hashlib.sha256((dataset / 'stage1-training.bin').read_bytes()).hexdigest()
    (dataset / 'stage1-training-assets.json').write_text(json.dumps({'stage1-training.bin': sha}, indent=2))
    (dataset / 'dataset-metadata.json').write_text(json.dumps(dict(id=f'biadis/{slug}-assets',
        title=f'CrashIntent Stage1 {args.mode.title()} V1 Assets', licenses=[dict(name='other')],
        description='Private CCD-derived synthetic training pairs and project code. Original-upload grouped splits; phone holdout excluded. Not for redistribution.'), indent=2))
    shutil.copyfile(root / 'notebooks/Stage1_Full_Training.ipynb', kernel / 'Stage1_Full_Training.ipynb')
    (kernel / 'kernel-metadata.json').write_text(json.dumps(dict(id=f'biadis/{slug}',
        title=f'CrashIntent Stage1 {args.mode.title()} V1', code_file='Stage1_Full_Training.ipynb',
        language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=True,
        machine_shape='NvidiaTeslaT4', dataset_sources=[f'biadis/{slug}-assets'],
        competition_sources=[], kernel_sources=[], model_sources=[]), indent=2))
    result = dict(status='PREPARED_NOT_UPLOADED', **config, sha256=sha,
                  bytes=(dataset / 'stage1-training.bin').stat().st_size, kernel=f'biadis/{slug}')
    (out / 'bundle.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
