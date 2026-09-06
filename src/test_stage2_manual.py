"""Audit a manual-label snapshot, train a small prototype, and evaluate held-out sources.

This is a development experiment, not a competition submission or a label-quality certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import re

import cv2
import pandas as pd
import torch
from torch import nn

from project_config import configured_path
import stage2_pipeline as pipeline


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def macro_f1(truth, predicted):
    scores = []
    for label in (0, 1):
        tp = sum(a == b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--exclude-id", action="append", default=[],
                        help="Video ID held for manual review; repeat for multiple IDs")
    args = parser.parse_args()
    excluded_ids = {f"{int(vid):06d}" for vid in args.exclude_id}
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(configured_path("artifacts.dir") / "torch-cache/hub"))
    raw = configured_path("stage2.manual.labels").read_bytes()
    (out / "labels_manual_snapshot.csv").write_bytes(raw)
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if len({r["ID"] for r in rows}) != len(rows):
        raise ValueError("Duplicate manual IDs")
    official = {}
    for line in configured_path("ccd.label.file").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"(\d+),\[([^\]]+)\],(\d+),([^,]+),.*", line)
        if not m:
            raise ValueError(f"Malformed CCD label: {line[:30]}")
        official[m[1]] = (list(map(int, m[2].split(","))).index(1), m[4])
    accepted, review = [], []
    for r in rows:
        vid = r["ID"]
        path = configured_path("ccd.video.dir") / f"{vid}.mp4"
        frames = pipeline._read_video(path)
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        c, e, space = (int(r[k]) for k in ("collision_frame", "entry_frame", "evasion_space"))
        if not (0 <= c < len(frames) and 0 <= e < len(frames) and fps > 0):
            raise ValueError(f"Invalid frames or FPS: {vid}")
        if space not in (0, 1) or r["entry_side"] not in ("LEFT", "RIGHT"):
            raise ValueError(f"Invalid category: {vid}")
        if c != official[vid][0]:
            raise ValueError(f"CCD collision mismatch: {vid}")
        reason = "entry_after_CCD_collision_reference" if e > c else None
        if vid in excluded_ids:
            reason = "excluded_for_manual_review"
        record = dict(ID=vid, path=str(path), t_collision=c, t_entry=e,
                      evasion_space=space, entry_side=r["entry_side"],
                      source_id=official[vid][1], fps=fps, frames=len(frames))
        if reason:
            review.append(dict(record, reason=reason))
        else:
            accepted.append(record)
    groups = sorted({r["source_id"] for r in accepted})
    if len(groups) < 2:
        raise ValueError("Need at least two independent sources")
    random.Random(pipeline.SEED).shuffle(groups)
    validation_groups = set(groups[:max(1, round(len(groups) * .2))])
    train = [r for r in accepted if r["source_id"] not in validation_groups]
    validation = [r for r in accepted if r["source_id"] in validation_groups]
    assert not ({r["source_id"] for r in train} & {r["source_id"] for r in validation})
    for name, records in (("train", train), ("validation", validation), ("review", review)):
        pd.DataFrame(records).to_csv(out / f"labels_{name}.csv", index=False)
    audit = dict(snapshot_sha256=hashlib.sha256(raw).hexdigest(), saved=len(rows),
                 accepted=len(accepted), train=len(train), validation=len(validation),
                 review=review, source_overlap=False,
                 distributions={name: {key: dict(Counter(r[key] for r in records))
                                       for key in ("evasion_space", "entry_side")}
                                for name, records in (("train", train), ("validation", validation))},
                 note="Manual semantics not certified. CCD collision is an event-onset proxy; review before final training.")
    write_json(out / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if args.audit_only:
        return
    config = argparse.Namespace(output_dir=out / "training", device=args.device,
                                data_dir=Path.cwd(), limit=None, backbone_weights="imagenet",
                                feature_batch_size=16, epochs=args.epochs,
                                learning_rate=2e-4, command="train")
    print("Training with ImageNet features...", flush=True)
    pipeline.run_pipeline(config, out / "labels_train.csv")
    device = pipeline._device(args.device)
    model_dir = out / "training/model/stage2"
    backbone, transform = pipeline._backbone("none", device)
    backbone.load_state_dict(torch.load(model_dir / "resnet18-f37072fd.pth", weights_only=True))
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    model = pipeline.Stage2Temporal().to(device)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device, weights_only=True)["model"])
    model.eval()
    results = []
    for r in validation:
        features = pipeline._features(pipeline._read_video(Path(r["path"])), backbone, transform, device, 16)
        with torch.inference_mode():
            collision, entry, scene = model(features.unsqueeze(0).to(device))
        c, e = int(collision.item()), int(entry.item())
        space, side = int(scene[:, :2].argmax(1).item()), int(scene[:, 2:].argmax(1).item())
        results.append(dict(ID=r["ID"], collision_frame=c, entry_frame=e,
                            evasion_space=space, entry_side=("LEFT", "RIGHT")[side],
                            collision_error_seconds=abs(c-r["t_collision"])/r["fps"],
                            entry_error_seconds=abs(e-r["t_entry"])/r["fps"]))
    frame = pd.DataFrame(results)
    frame.to_csv(out / "validation_details.csv", index=False)
    frame[pipeline.OUTPUT_COLUMNS].to_csv(out / "validation_predictions.csv", index=False)
    n = len(validation)
    metrics = {key+"_accuracy": sum(a[key] == b[key] for a, b in zip(validation, results))/n
               for key in ("entry_side", "evasion_space")}
    for key in ("collision", "entry"):
        errors = [r[key+"_error_seconds"] for r in results]
        metrics[key+"_within_0.3s"] = sum(x <= .3 + 1e-9 for x in errors)/n
        metrics[key+"_mae_seconds"] = sum(errors)/n
    metrics["evasion_macro_f1"] = macro_f1([r["evasion_space"] for r in validation], [r["evasion_space"] for r in results])
    metrics["side_macro_f1"] = macro_f1([int(r["entry_side"] == "RIGHT") for r in validation], [int(r["entry_side"] == "RIGHT") for r in results])
    majority = {}
    for key in ("entry_side", "evasion_space"):
        constant = Counter(r[key] for r in train).most_common(1)[0][0]
        majority[key] = dict(prediction=constant, accuracy=sum(r[key] == constant for r in validation)/n)
    report = dict(status="PASS", device=str(device), epochs=args.epochs,
                  train_samples=len(train), validation_samples=n, metrics=metrics,
                  train_majority_baseline=majority,
                  warning="Small source-held-out development check. Not official Stage2/overall score; label uncertainty remains.")
    write_json(out / "test_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
