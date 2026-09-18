"""Evaluate frozen Stage 3 motion models on the sensor-labelled A2D2 pilot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


P = Path(__file__).resolve().parents[1]
DATA = P / "artifacts/stage3-a2d2-clip-pilot-20260917"
OUT = P / "artifacts/stage3-a2d2-zero-shot-20260917"
SINGLE = P / "artifacts/stage3-warmstart-ce-submit-candidate-20260917/best.pt"
ENSEMBLE = P / "artifacts/stage3-warmstart-ce-ensemble-submit-candidate-20260917/best.pt"
SINGLE_MODULE = P / "src/stage3_motion_inference.py"
ENSEMBLE_MODULE = P / "src/stage3_motion_ensemble_inference.py"
CLASSES = ["LEFT", "STRAIGHT", "RIGHT"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def metric(truth: np.ndarray, pred: np.ndarray) -> dict:
    matrix = np.asarray(
        [[int(np.sum((truth == i) & (pred == j))) for j in range(3)] for i in range(3)]
    )
    support = matrix.sum(1)
    denominator = support + matrix.sum(0)
    f1 = np.divide(2 * np.diag(matrix), denominator, out=np.zeros(3), where=denominator > 0)
    recall = np.divide(np.diag(matrix), support, out=np.zeros(3), where=support > 0)
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(np.mean(truth == pred)),
        "per_class_f1": dict(zip(CLASSES, map(float, f1))),
        "recall": dict(zip(CLASSES, map(float, recall))),
        "confusion": matrix.tolist(),
    }


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "status.json"
    save(status, {"status": "EVALUATING", "completed": 0, "total": 36})
    single_module = load_module("a2d2_single", SINGLE_MODULE)
    ensemble_module = load_module("a2d2_ensemble", ENSEMBLE_MODULE)
    device = torch.device("cpu")
    single_model, single_mean, single_std = single_module.s3_motion_load(SINGLE, device)
    ensemble_models, ensemble_mean, ensemble_std = ensemble_module.s3_motion_load(ENSEMBLE, device)
    np.testing.assert_allclose(single_mean, ensemble_mean)
    np.testing.assert_allclose(single_std, ensemble_std)

    samples = pd.read_csv(DATA / "samples.csv")
    if len(samples) != 36 or samples.groupby("steer_label").size().to_dict() != {
        "LEFT": 12, "RIGHT": 12, "STRAIGHT": 12
    }:
        raise ValueError("unexpected A2D2 sample contract")
    rows=[]
    for number, row in enumerate(samples.itertuples(index=False), 1):
        path = DATA / "videos" / f"{row.ID}.mp4"
        features = single_module.s3_motion_features(path)
        if features.shape != (16, 402):
            raise ValueError(f"unexpected features for {row.ID}: {features.shape}")
        feature = features[15:16]
        _, single_logits = single_module.s3_motion_logits(
            single_model, feature, single_mean, single_std, device
        )
        _, ensemble_logits = ensemble_module.s3_motion_logits(
            ensemble_models, feature, ensemble_mean, ensemble_std, device
        )
        rows.append({
            "ID":row.ID, "truth":row.steer_label,
            "single_prediction":CLASSES[int(single_logits.argmax(1)[0])],
            "ensemble_prediction":CLASSES[int(ensemble_logits.argmax(1)[0])],
            **{f"single_logit_{name}":float(single_logits[0,i]) for i,name in enumerate(CLASSES)},
            **{f"ensemble_logit_{name}":float(ensemble_logits[0,i]) for i,name in enumerate(CLASSES)},
        })
        save(status, {"status":"EVALUATING", "completed":number, "total":len(samples)})
    predictions=pd.DataFrame(rows)
    predictions.to_csv(OUT/"predictions.csv",index=False)
    truth=predictions.truth.map({x:i for i,x in enumerate(CLASSES)}).to_numpy()
    single=predictions.single_prediction.map({x:i for i,x in enumerate(CLASSES)}).to_numpy()
    ensemble=predictions.ensemble_prediction.map({x:i for i,x in enumerate(CLASSES)}).to_numpy()
    report={
        "status":"COMPLETE_VALIDATED", "samples":len(samples),
        "data_report_sha256":sha(DATA/"report.json"),
        "single_checkpoint_sha256":sha(SINGLE), "ensemble_checkpoint_sha256":sha(ENSEMBLE),
        "single":metric(truth,single), "ensemble":metric(truth,ensemble),
        "agreement":float(np.mean(single==ensemble)),
        "scope":"Frozen zero-shot endpoint-15 predictions on 36 sensor-labelled A2D2 causal clips; no fitting or threshold selection.",
    }
    save(OUT/"report.json",report)
    save(status,{"status":"COMPLETE_VALIDATED", "completed":len(samples), "total":len(samples), "report":str(OUT/"report.json")})
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
