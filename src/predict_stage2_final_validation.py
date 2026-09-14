"""Frozen paired CPU inference; human review gate runs before any candidate inference."""
import argparse
import ast
import csv
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

from prepare_stage2_final_validation import ROOT, OUT, DATA, FF, sha, save
from score_stage2_final_validation import read_inputs, read_csv, label_gate, score_rows, TARGETS


def load_models():
    frozen = json.loads((OUT/'frozen-models.json').read_text())
    for entry in frozen.values():
        assert sha(ROOT/entry['file']) == entry['sha256']
    source = (ROOT/frozen['reference-inference.py']['file']).read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_Stage2Temporal']
    assert len(nodes) == 1
    ns = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen reference Stage2>', 'exec'), ns)
    models = {}
    for name in ['incumbent', 'ccd_pretrained']:
        model = ns['_Stage2Temporal']()
        model.load_state_dict(torch.load(ROOT/frozen[name]['file'], weights_only=True, map_location='cpu')['model'], strict=True)
        models[name] = model.eval()
    backbone = resnet18(weights=None)
    backbone.load_state_dict(torch.load(ROOT/frozen['resnet.pt']['file'], weights_only=True, map_location='cpu'), strict=True)
    backbone.fc = nn.Identity()
    return models, backbone.eval(), frozen


def smoke():
    """Only existing development features; no candidate video or human labels."""
    import stage2_pipeline as pipeline
    torch.set_num_threads(2)
    models, backbone, frozen = load_models()
    features = torch.load(ROOT/'artifacts/stage2-controls-20260909/features.pt', weights_only=True)
    with torch.inference_mode():
        for name, model in models.items():
            reference = pipeline.Stage2Temporal().eval()
            reference.load_state_dict(torch.load(ROOT/frozen[name]['file'], weights_only=True)['model'])
            for sid in sorted(features)[:3]:
                actual, expected = model(features[sid][None]), reference(features[sid][None])
                assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    folder = OUT/'public-smoke-images'
    folder.mkdir(exist_ok=True)
    subprocess.run([str(FF), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-threads', '1',
                    '-i', str(ROOT/'Baseline/data/stage2/videos/000001.mp4'), '-frames:v', '3',
                    '-q:v', '2', '-threads', '1', '-start_number', '0', str(folder/'frame_%06d.jpg')], check=True, timeout=30)
    images = sorted(folder.glob('*.jpg'))
    assert len(images) == 3
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    tensors = []
    for path in images:
        with Image.open(path) as image:
            tensors.append(transform(image.convert('RGB')))
    with torch.inference_mode():
        feats = backbone(torch.stack(tensors))
        assert feats.shape == (len(images), 512) and torch.isfinite(feats).all()
        for model in models.values():
            ci, ei, scene = model(feats[None])
            assert 0 <= int(ci) < len(images) and 0 <= int(ei) < len(images) and torch.isfinite(scene).all()
    save(OUT/'model-smoke.json', dict(status='PASS', models=2, development_videos=3, public_jpeg_frames=len(images), candidate_predictions=0, checks=['Frozen hashes and strict loads', 'Actual archived inference class vs local class exact outputs', 'Public JPEG to ResNet to both temporal model outputs']))
    print('PASS: 2 frozen models x 3 existing development inputs, exact output parity')


def predict(destination, labels_path):
    torch.set_num_threads(2)
    videos, times = read_inputs()
    labels = read_csv(labels_path)
    gate = label_gate(labels, times)
    if gate['status'] != 'READY_TO_SCORE':
        raise ValueError('Complete human dispositions and labels before inference')
    contract = json.loads((OUT/'evaluation-contract.json').read_text())
    assert sha(OUT/'frozen-models.json') == contract['frozen_models_sha256']
    for rel, digest in contract['code_sha256'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Evaluation code changed after freeze: '+rel)
    assert sha(DATA/'manifest.json') == contract['manifest_sha256']
    initial_labels_hash = sha(labels_path)
    models, backbone, frozen = load_models()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    destination.mkdir(parents=True, exist_ok=False)
    save(destination/'frozen-labels.json', dict(sha256=initial_labels_hash, rows=labels))
    collected = {name: [] for name in models}
    started = time.monotonic()
    with torch.inference_mode():
        for sid in sorted(gate['accepted']):
            rows = read_csv(DATA/videos[sid]['frame_map'])
            chunks = []
            for offset in range(0, len(rows), 16):
                batch = []
                for row in rows[offset:offset+16]:
                    path = DATA/row['file']
                    if sha(path) != row['sha256']:
                        raise ValueError('JPEG changed: '+str(path))
                    with Image.open(path) as image:
                        batch.append(transform(image.convert('RGB')))
                chunks.append(backbone(torch.stack(batch)).float())
            features = torch.cat(chunks)
            assert features.shape == (videos[sid]['frames'], 512) and torch.isfinite(features).all()
            record = {}
            for name, model in models.items():
                ci, ei, scene = model(features[None])
                assert torch.isfinite(scene).all()
                row = dict(ID=sid, collision_frame=int(ci), entry_frame=int(ei), evasion_space=int(scene[0, :2].argmax()), entry_side=['LEFT', 'RIGHT'][int(scene[0, 2:].argmax())])
                collected[name].append(row)
                record[name] = dict(prediction=row, scene_logits=scene.tolist())
            save(destination/(sid+'.json'), record)
            save(destination/'status.json', dict(status='PREDICTING', completed=len(collected['incumbent']), total=len(gate['accepted'])))
    assert sha(labels_path) == initial_labels_hash
    for entry in frozen.values():
        assert sha(ROOT/entry['file']) == entry['sha256']
    scores = {}
    for name, rows in collected.items():
        path = destination/(name+'-predictions.csv')
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ID']+TARGETS)
            writer.writeheader()
            writer.writerows(rows)
        score = score_rows(labels, read_csv(path), times)
        scores[name] = score
    save(destination/'comparison.json', dict(status='COMPLETE', models=frozen, labels_sha256=initial_labels_hash,
                                            manifest_sha256=sha(DATA/'manifest.json'), precision='CPU FP32',
                                            seconds=time.monotonic()-started, scores=scores,
                                            scope='Paired local evaluation; no model fitting, no automatic model promotion'))
    save(destination/'status.json', dict(status='COMPLETE', evaluated=len(gate['accepted']), seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--labels', type=Path, default=DATA/'review-decisions.csv')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.output:
        predict(args.output, args.labels)
    else:
        parser.error('--output is required for gated prediction')
