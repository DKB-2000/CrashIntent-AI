"""Resume full-resolution CCD synthesis with upload-group and phone isolation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re

import cv2
import pandas as pd

import prepare_stage1_rerecorded as synth
from select_stage1_holdout import _rank


SEED = 20260903


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(input_dir, metadata):
    groups = {}
    for line in Path(metadata).read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'(\d+),\[([^\]]+)\],(\d+),([^,]+),.*', line)
        if not match:
            raise ValueError('Invalid CCD metadata')
        groups[match[1]] = match[4]
    videos = {p.stem: p for p in Path(input_dir).glob('*.mp4')}
    if set(videos) != set(groups) or len(videos) != 1500:
        raise ValueError('Expected all 1500 CCD sources and matching metadata')
    phone = sorted(videos, key=lambda s: (_rank(s, SEED), s))[:30]
    phone_groups = {groups[s] for s in phone}
    excluded = sorted(s for s in groups if groups[s] in phone_groups)
    usable = sorted(set(groups) - set(excluded))
    ordered_groups = sorted({groups[s] for s in usable}, key=lambda g: hashlib.sha256(('stage1-group:' + g).encode()).hexdigest())
    val_groups = set(ordered_groups[:max(1, round(len(ordered_groups) * .2))])
    rows = [dict(source_id=s, source_path=str(videos[s].resolve()), upload_group=groups[s],
                 split='val' if groups[s] in val_groups else 'train') for s in usable]
    return rows, phone, excluded


def render_source(task):
    row, output_dir, variants, code_sha = task
    cv2.setNumThreads(1)
    out = Path(output_dir)
    source = Path(row['source_path'])
    sid = row['source_id']
    record = out / 'records' / f'{sid}.json'
    identity = dict(source_sha256=sha256(source), generator_sha256=code_sha,
                    seed=SEED, variants=variants, **row)
    if record.exists():
        saved = json.loads(record.read_text())
        if saved['identity'] != identity:
            raise ValueError(f'Resume identity mismatch: {sid}')
        if all((out / r['output_path']).is_file() and sha256(out / r['output_path']) == r['sha256'] for r in saved['rows']):
            return saved['rows']
        raise ValueError(f'Resume output corruption: {sid}')
    result = []
    for variant in range(variants + 1):
        seed = int.from_bytes(hashlib.sha256(f'{SEED}:{sid}:{variant}'.encode()).digest()[:4], 'big')
        rng = random.Random(seed)
        effects = synth._effect_parameters(rng) if variant else None
        relative = f'rerecorded/{sid}_v{variant:02d}.mp4' if variant else f'original/{sid}.mp4'
        destination = out / relative
        # A source record is committed only after every video finishes successfully.
        info = synth._render(source, destination, rng.randint(20, 28), effects, seed)
        result.append(dict(**row, output_path=relative, label='RERECORDED' if variant else 'ORIGINAL',
                           variant=variant, effects=json.dumps(asdict(effects), sort_keys=True) if effects else None,
                           sha256=sha256(destination), source_sha256=identity['source_sha256'], **info))
    record.write_text(json.dumps(dict(identity=identity, rows=result), indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data_raw/ccd/videos/Crash-1500'))
    parser.add_argument('--metadata', type=Path, default=Path('data_raw/ccd/Crash-1500.txt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--variants', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.variants < 1 or args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError('variants, workers and limit must be positive')
    rows, phone, excluded = plan(args.input_dir, args.metadata)
    out = args.output_dir.resolve()
    (out / 'records').mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'source_id': phone}).to_csv(out / 'phone_holdout.csv', index=False)
    pd.DataFrame({'source_id': excluded}).to_csv(out / 'excluded_sources.csv', index=False)
    if args.limit:
        # Interleave sources from both fixed partitions for a representative preview.
        partitions = [[r for r in rows if r['split'] == split] for split in ('train', 'val')]
        selected = []
        for i in range(max(map(len, partitions))):
            for partition in partitions:
                if i < len(partition):
                    selected.append(partition[i])
        rows = selected[:args.limit]
    pd.DataFrame(rows).to_csv(out / 'source_plan.csv', index=False)
    code_sha = hashlib.sha256((Path(__file__).read_bytes() + Path(synth.__file__).read_bytes())).hexdigest()
    tasks = [(r, str(out), args.variants, code_sha) for r in rows]
    manifest = []
    print(json.dumps(dict(sources=len(rows), samples=len(rows)*(args.variants+1), phone_sources=len(phone), excluded_with_siblings=len(excluded))), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(render_source, tasks), 1):
            manifest.extend(result)
            print(f'Completed {i}/{len(rows)} sources', flush=True)
    table = pd.DataFrame(manifest)
    if table.groupby('upload_group').split.nunique().max() != 1 or set(table.source_id) & set(excluded):
        raise ValueError('Split leakage')
    table.to_csv(out / 'generation_manifest.csv', index=False)
    table[['source_id', 'split', 'upload_group']].drop_duplicates().to_csv(out / 'source_splits.csv', index=False)
    labels = table[['output_path', 'label', 'split']].rename(columns={'output_path': 'path'})
    labels.to_csv(out / 'labels.csv', index=False)
    for split in ('train', 'val'):
        labels[labels.split == split].to_csv(out / f'labels_{split}.csv', index=False)
    report = dict(status='PASS', sources=len(rows), samples=len(table), variants_per_source=args.variants,
                  phone_sources=30, excluded_with_siblings=len(excluded), source_leakage=False,
                  split_unit='CCD original upload group', generator_sha256=code_sha,
                  manifest_sha256=sha256(out / 'generation_manifest.csv'),
                  splits_by_source=table.drop_duplicates('source_id').split.value_counts().to_dict(),
                  splits_by_sample=table.split.value_counts().to_dict(),
                  labels=table.label.value_counts().to_dict(), real_phone_validation_available=False)
    (out / 'generation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
