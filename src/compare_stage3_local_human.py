"""Compare frozen submission models on the September 13 human review, on CPU."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import cv2
import numpy as np
import pandas as pd
import torch
import stage3_motion_inference as old
import stage3_short_motion_inference as short
from review_stage3_human import add_segment, dense_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage3-human-comparison-20260914'
CLASSES = ['LEFT', 'STRAIGHT', 'RIGHT']
KEYS = ['ID', 'sample_index']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'status.json', {'status': 'RUNNING', 'pid': os.getpid()})
    provenance = {}
    models = {}
    for name, module, folder in (
        ('old', old, 'stage3-motion-submit-candidate-20260910'),
        ('short', short, 'final-stage2-short-stage3-20260911'),
    ):
        source = ROOT / 'artifacts' / folder
        validation = json.loads((source / 'validation.json').read_text())
        archive = source / 'submit.zip'
        assert sha(archive) == validation['candidate_sha256']
        with zipfile.ZipFile(archive) as z:
            weights = z.read('model/stage3/best.pt')
            code = z.read('inference.py').decode('utf-8-sig')
        assert hashlib.sha256(weights).hexdigest() == validation['deploy_checkpoint_sha256']
        # Compare every deployment function/class against the actual submitted source.
        tree = ast.parse(code)
        deployed = {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        local = ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig'))
        for node in local.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert deployed[node.name] == ast.dump(node, include_attributes=False), node.name
        checkpoint = OUT / (name + '.pt')
        checkpoint.write_bytes(weights)
        model, mean, std = module.s3_motion_load(checkpoint, torch.device('cpu'))
        models[name] = (module, model, mean, std)
        provenance[name] = {'zip': str(archive.relative_to(ROOT)), 'zip_sha256': sha(archive),
                            'checkpoint_sha256': sha(checkpoint), 'deployment_ast_match': True}

    results = []
    input_hashes = {}
    source_rows = []
    for group, folder in [('random', 'stage3-human-review'), ('right_enriched', 'stage3-right-review')]:
        directory = ROOT / 'data' / folder
        for filename in ('annotations.json', 'labels_review.csv', 'sources.json'):
            path = directory / filename
            input_hashes[str(path.relative_to(ROOT))] = sha(path)
        annotations = json.loads((directory / 'annotations.json').read_text(encoding='utf-8-sig'))['videos']
        sources = json.loads((directory / 'sources.json').read_text(encoding='utf-8-sig'))['videos']
        assert set(annotations) == {s['ID'] for s in sources}
        labels = pd.read_csv(directory / 'labels_review.csv')
        assert not labels.duplicated(KEYS).any()
        reconstructed = []
        predictions = {name: [] for name in models}
        for source in sources:
            sid = source['ID']
            annotation = annotations[sid]
            path = ROOT / 'data_raw/ccd/videos/Crash-1500' / (source['source_id'] + '.mp4')
            assert sha(path) == source['sha256'] == annotation['sha256'], sid
            cap = cv2.VideoCapture(str(path))
            assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS) - 10) < .01
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            assert count == source['frames'] == annotation['frames']
            segments = []
            for segment in annotation['segments']:
                segments = add_segment(segments, segment['start'], segment['end'],
                                       segment['steer'], segment['motion'], count)
            reconstructed.extend(dense_rows(sid, count, segments))
            source_rows.append(dict(group=group, ID=sid, path=str(path.relative_to(ROOT)), sha256=sha(path)))
            for name, (module, model, mean, std) in models.items():
                features = module.s3_motion_features(path)
                assert len(features) == count
                accel, steer = module.s3_motion_logits(model, features, mean, std, torch.device('cpu'))
                np.savez_compressed(OUT / f'{group}-{sid}-{name}.npz', features=features, accel=accel, steer=steer)
                pred = pd.DataFrame({'ID': sid, 'sample_index': np.arange(count),
                                     'accel_label': np.array(module._S3M_ACCEL)[accel.argmax(1)],
                                     'steer_label': np.array(CLASSES)[steer.argmax(1)]})
                predictions[name].append(pred)
            print(f'{group}: {sid} decoded and predicted', flush=True)
        pd.testing.assert_frame_equal(labels.sort_values(KEYS).reset_index(drop=True),
                                      pd.DataFrame(reconstructed)[labels.columns].sort_values(KEYS).reset_index(drop=True))
        eligible = labels[(labels.reviewed == 1) & (labels.motion_status == 'MOVING') & labels.steer_label.isin(CLASSES)]
        labels.to_csv(OUT / f'{group}-labels.csv', index=False)
        paired = eligible.copy()
        for name in models:
            pred = pd.concat(predictions[name], ignore_index=True)
            predpath = OUT / f'{group}-{name}-predictions.csv'
            pred.to_csv(predpath, index=False)
            assert not pred.duplicated(KEYS).any() and len(pred) == len(labels)
            scorepath = OUT / f'{group}-{name}-score.json'
            subprocess.run([sys.executable, str(ROOT / 'src/score_stage3_human.py'),
                            '--labels', str(directory / 'labels_review.csv'), '--predictions', str(predpath),
                            '--output', str(scorepath)], check=True, capture_output=True, text=True, timeout=60)
            score = json.loads(scorepath.read_text())
            joined = eligible.merge(pred[KEYS + ['steer_label']], on=KEYS, validate='one_to_one', suffixes=('_truth', '_pred'))
            cm = [[sum((joined.steer_label_truth == truth) & (joined.steer_label_pred == prediction))
                   for prediction in CLASSES] for truth in CLASSES]
            cm = np.array(cm, dtype=int)
            assert cm.tolist() == score['confusion']
            den = cm.sum(0) + cm.sum(1)
            independent_f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(3), where=den != 0)
            assert abs(independent_f1.mean() - score['macro_f1']) < 1e-12
            support = cm.sum(1)
            recalls = {c: (float(cm[i, i] / support[i]) if support[i] else None) for i, c in enumerate(CLASSES)}
            results.append(dict(group=group, model=name, eligible_videos=int(eligible.ID.nunique()),
                                recall=recalls, **score))
            paired = paired.merge(pred[KEYS + ['steer_label']].rename(columns={'steer_label': name}), on=KEYS, validate='one_to_one')
        paired.to_csv(OUT / f'{group}-paired.csv', index=False)
    for path, digest in input_hashes.items():
        assert sha(ROOT / path) == digest, 'Human input changed during scoring'
    report = dict(status='COMPLETE_VALIDATED', scope='CPU human development diagnostic; not independent generalization or official score',
                  macro_f1_policy='Fixed LEFT/STRAIGHT/RIGHT; zero division 0. Missing truth classes limit interpretation; do not pool groups.',
                  models=provenance, inputs=input_hashes, sources=source_rows, results=results,
                  validation='17 source hashes/FPS/full decodes, annotation-to-CSV parity, deployment function AST, checkpoint hashes, independent confusion/F1, unchanged human inputs PASS',
                  versions={'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__})
    write(OUT / 'report.json', report)
    write(OUT / 'status.json', {'status': 'COMPLETE_VALIDATED', 'predictions': 1700, 'eligible_frames_per_model': 350})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / 'status.json', {'status': 'FAILED', 'error': repr(exc)})
        raise
