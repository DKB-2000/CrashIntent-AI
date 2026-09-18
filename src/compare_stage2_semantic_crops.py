"""Frozen ResNet18 crop-feature probe for Stage2 scene labels on source-CV folds.

Only the scene probe is fitted. Event positions, source splits, backbone and
manual labels are frozen. This is exploratory development, not deployment.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

import stage2_pipeline as p
from analyze_stage2_source_cv import details, summary

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "artifacts/stage2-source-cv-20260910"
OUT = ROOT / os.environ.get("STAGE2_SEMANTIC_CROPS_OUT", "artifacts/stage2-semantic-crops-20260916")
ZIP = ROOT / "artifacts/stage2-pretrained-submit-candidate-20260911/submit.zip"
# Same SHA256 in Baseline, the pretrained candidate, and the best Stage1 ZIP.
BACKBONE_HASH = "b55eb2f9f3f559e2101e507d9acc49a5e0768f9d710eeb6a0912080e57595860"
EPOCHS = 80
SEED = int(os.environ.get("STAGE2_SEMANTIC_CROPS_SEED", "20260916"))
EVENT_POLICY = os.environ.get("STAGE2_EVENT_POLICY", "argmax")
CROP_PROJECTION = os.environ.get("STAGE2_CROP_PROJECTION", "none")
FEATURE_POLICY = os.environ.get("STAGE2_CROP_FEATURE_POLICY", "shared")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_backbone():
    with zipfile.ZipFile(ZIP) as z:
        data = z.read("model/stage2/resnet18-f37072fd.pth")
    if sha(data) != BACKBONE_HASH:
        raise RuntimeError("Frozen ResNet18 weight hash mismatch")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(io.BytesIO(data), weights_only=True, map_location="cpu"), strict=True)
    model.fc = nn.Identity()
    model.eval()
    return model, ResNet18_Weights.IMAGENET1K_V1.transforms()


def frame_at(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {video} frame {index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crops(frame: np.ndarray):
    h, w = frame.shape[:2]
    # Keep the road-facing lower area and separate lateral/central evidence.
    y = int(h * .25)
    return (frame[y:, :int(w * .48)], frame[y:, int(w * .52):], frame[y:, int(w * .25):int(w * .75)])


def semantic(frame: np.ndarray, backbone, transform) -> torch.Tensor:
    batch = torch.stack([transform(Image.fromarray(x)) for x in crops(frame)])
    with torch.inference_mode():
        result = backbone(batch)
    if result.shape != (3, 512):
        raise RuntimeError("Unexpected crop feature shape")
    return torch.cat((result[0] - result[1], result[2]), dim=0)


def normalize(train: torch.Tensor, val: torch.Tensor):
    mean = train.mean(0)
    std = train.std(0, unbiased=False).clamp_min(.1)
    return (train - mean) / std, (val - mean) / std


def project_semantic(train_point: torch.Tensor, val_point: torch.Tensor,
                     train_crop: torch.Tensor, val_crop: torch.Tensor):
    """Standardize on the training fold and optionally compress crop features."""
    train_point, val_point = normalize(train_point, val_point)
    train_crop, val_crop = normalize(train_crop, val_crop)
    if CROP_PROJECTION == "none":
        return torch.cat((train_point, train_crop), 1), torch.cat((val_point, val_crop), 1)
    if CROP_PROJECTION != "pca16":
        raise ValueError(f"Unknown crop projection: {CROP_PROJECTION}")
    # Exact SVD is deterministic.  Only training-fold rows determine the basis.
    _, _, vh = torch.linalg.svd(train_crop, full_matrices=False)
    basis = vh[:16].T.contiguous()
    return (torch.cat((train_point, train_crop @ basis), 1),
            torch.cat((val_point, val_crop @ basis), 1))


class TaskSpecificProbe(nn.Module):
    def __init__(self, point_dim: int):
        super().__init__()
        self.point_dim = point_dim
        self.evasion = nn.Linear(point_dim + 1024, 2)
        self.side = nn.Linear(point_dim + 1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.point_dim
        point = x[:, :p]
        # Crop layout per event is [left-right difference, center].
        evasion_x = torch.cat((point, x[:, p+512:p+1024], x[:, p+1536:p+2048]), 1)
        side_x = torch.cat((point, x[:, p:p+512], x[:, p+1024:p+1536]), 1)
        return torch.cat((self.evasion(evasion_x), self.side(side_x)), 1)


def fit_probe(x: torch.Tensor, evasion: torch.Tensor, side: torch.Tensor, point_dim: int,
              feature_policy: str | None = None):
    torch.manual_seed(SEED)
    policy = FEATURE_POLICY if feature_policy is None else feature_policy
    if policy == "shared" or x.shape[1] == point_dim:
        model = nn.Linear(x.shape[1], 4)
    elif policy == "task_specific":
        if CROP_PROJECTION != "none" or x.shape[1] != point_dim + 2048:
            raise RuntimeError("task_specific requires unprojected crop features")
        model = TaskSpecificProbe(point_dim)
    else:
        raise ValueError(f"Unknown crop feature policy: {policy}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.1)
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits[:, :2], evasion) + nn.functional.cross_entropy(logits[:, 2:], side)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def event_indices(collision: torch.Tensor, entry: torch.Tensor) -> tuple[int, int]:
    if EVENT_POLICY == "argmax":
        return int(collision.argmax()), int(entry.argmax())
    if EVENT_POLICY != "ordered_joint":
        raise ValueError(f"Unknown event policy: {EVENT_POLICY}")
    best_entry_score, best_entry_index = torch.cummax(entry, dim=0)
    collision_index = int((collision + best_entry_score).argmax())
    return collision_index, int(best_entry_index[collision_index])


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(2)
    protocol = json.loads((CV / "protocol.json").read_text(encoding="utf-8"))
    valid = json.loads((CV / "independent-validation.json").read_text(encoding="utf-8"))
    if valid["status"] != "PASS":
        raise RuntimeError("Original CV is not validated")
    backbone, transform = load_backbone()
    cache_path = ROOT / "artifacts/stage2-controls-20260909/features.pt"
    if sha(cache_path.read_bytes()) != protocol["features_sha256"]:
        raise RuntimeError("Original feature cache hash mismatch")
    features = torch.load(cache_path, weights_only=True, map_location="cpu")
    expected = pd.read_csv(CV / "soft_mixed_15_oof.csv", dtype={"ID": str}).set_index("ID")
    OUT.mkdir()
    report = {"status": "RUNNING", "epochs": EPOCHS, "seed": SEED, "event_policy": EVENT_POLICY,
              "crop_projection": CROP_PROJECTION, "crop_feature_policy": FEATURE_POLICY,
              "scope": "Reused 66-video, 23-source development folds. No new labels or deployment.",
              "conditions": {}, "input_sha256": {"zip": sha(ZIP.read_bytes()), "feature_cache": protocol["features_sha256"]}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        all_truth, outputs = [], {"point": [], "semantic_crop": [], "old_scene": []}
        if FEATURE_POLICY == "task_specific":
            outputs["semantic_shared"] = []
        feature_cache = {}
        for fold in range(5):
            folder = CV / f"fold-{fold}"
            train = pd.read_csv(folder / "train.csv", dtype={"ID": str, "source_id": str})
            val = pd.read_csv(folder / "validation.csv", dtype={"ID": str, "source_id": str})
            if set(train.source_id) & set(val.source_id):
                raise RuntimeError("Source leakage")
            checkpoint = folder / "soft_mixed_15.pt"
            report["input_sha256"][f"fold-{fold}"] = sha(checkpoint.read_bytes())
            temporal = p.Stage2Temporal()
            temporal.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"], strict=True)
            temporal.eval()
            vectors = {}
            for row in pd.concat((train, val)).to_dict("records"):
                id_ = row["ID"]
                with torch.inference_mode():
                    cl, el, hidden = temporal.logits(features[id_].unsqueeze(0))
                    ci, ei = event_indices(cl[0], el[0])
                    point = torch.cat((hidden[0, ci], hidden[0, ei]))
                    old = temporal.scene(point.unsqueeze(0))[0]
                if id_ in expected.index:
                    ex = expected.loc[id_]
                    if all((ci == ex.pred_collision_frame, ei == ex.pred_entry_frame,
                            int(old[:2].argmax()) == ex.pred_evasion_space,
                            ("LEFT", "RIGHT")[int(old[2:].argmax())] == ex.pred_entry_side)):
                        pass
                    elif id_ in set(val.ID) and EVENT_POLICY == "argmax":
                        raise RuntimeError(f"Original OOF prediction differs: {id_}")
                crop_vec = []
                for index in (ci, ei):
                    key = (id_, index)
                    if key not in feature_cache:
                        feature_cache[key] = semantic(frame_at(Path(row["path"]), index), backbone, transform)
                    crop_vec.append(feature_cache[key])
                vectors[id_] = (point, torch.cat((*crop_vec,)), ci, ei, old)
            train_rows, val_rows = train.to_dict("records"), val.to_dict("records")
            train_point = torch.stack([vectors[r["ID"]][0] for r in train_rows])
            val_point = torch.stack([vectors[r["ID"]][0] for r in val_rows])
            train_crop = torch.stack([vectors[r["ID"]][1] for r in train_rows])
            val_crop = torch.stack([vectors[r["ID"]][1] for r in val_rows])
            train_sem, val_sem = project_semantic(train_point, val_point, train_crop, val_crop)
            train_point, val_point = normalize(train_point, val_point)
            evasion = torch.tensor([int(r["evasion_space"]) for r in train_rows])
            side = torch.tensor([0 if r["entry_side"] == "LEFT" else 1 for r in train_rows])
            conditions = [("point", train_point, val_point, "shared"),
                          ("semantic_crop", train_sem, val_sem, FEATURE_POLICY)]
            if FEATURE_POLICY == "task_specific":
                conditions.append(("semantic_shared", train_sem, val_sem, "shared"))
            for condition, tx, vx, policy in conditions:
                probe = fit_probe(tx, evasion, side, train_point.shape[1], policy)
                with torch.inference_mode():
                    pred = probe(vx)
                for row, logits in zip(val_rows, pred):
                    id_ = row["ID"]
                    _, _, ci, ei, _ = vectors[id_]
                    outputs[condition].append({"ID": id_, "collision_frame": ci, "entry_frame": ei,
                                               "evasion_space": int(logits[:2].argmax()),
                                               "entry_side": ("LEFT", "RIGHT")[int(logits[2:].argmax())],
                                               "evasion_logit_0": float(logits[0]), "evasion_logit_1": float(logits[1]),
                                               "side_logit_0": float(logits[2]), "side_logit_1": float(logits[3])})
            for row in val_rows:
                id_ = row["ID"]
                _, _, ci, ei, old = vectors[id_]
                outputs["old_scene"].append({"ID": id_, "collision_frame": ci, "entry_frame": ei,
                                             "evasion_space": int(old[:2].argmax()),
                                             "entry_side": ("LEFT", "RIGHT")[int(old[2:].argmax())]})
                all_truth.append(row)
            report["fold_completed"] = fold + 1
            (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if len(all_truth) != 66 or len({r["ID"] for r in all_truth}) != 66:
            raise RuntimeError("Incomplete OOF")
        for name, pred in outputs.items():
            frame = pd.DataFrame(details(all_truth, pred))
            frame.to_csv(OUT / f"{name}_oof.csv", index=False)
            report["conditions"][name] = summary(all_truth, pred)
        original = pd.DataFrame(details(all_truth, outputs["old_scene"])).set_index("ID")
        if EVENT_POLICY == "argmax" and any(original.loc[id_, "pred_" + k] != expected.loc[id_, "pred_" + k]
               for id_ in expected.index for k in p.OUTPUT_COLUMNS[1:]):
            raise RuntimeError("Old scene predictions did not reproduce")
        report["status"] = "COMPLETE_VALIDATED"
        report["output_sha256"] = {f.name: sha(f.read_bytes()) for f in OUT.glob("*_oof.csv")}
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = repr(exc)
        raise
    finally:
        (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
