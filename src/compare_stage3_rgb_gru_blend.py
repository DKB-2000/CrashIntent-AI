"""Same-frame incumbent/RGB-GRU comparison with route-held-out blend selection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import stage3_motion_ensemble_inference as incumbent_deploy

P = Path(__file__).resolve().parents[1]
OUT = P / "artifacts/stage3-rgb-gru-blend-20260918"
REF = P / "artifacts/stage3-representation-20260910"
RGB = P / "artifacts/kaggle-stage3-rgb-gru-20260918/result/stage3-rgb-gru/result"
INCUMBENT = P / "artifacts/stage3-a2d2-blend-submit-candidate-20260918/best.pt"
ACCEL = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER = ["LEFT", "STRAIGHT", "RIGHT"]
ALPHAS = np.linspace(0, 1, 21)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0)


class RGBGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(576, 192, 2, batch_first=True, dropout=0.2)
        self.accel = nn.Linear(192, 4)
        self.steer = nn.Linear(192, 3)

    def forward(self, x):
        z, _ = self.gru(x)
        return self.accel(z), self.steer(z)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def metric(y, pred, classes):
    n = len(classes)
    cm = np.array([[int(np.sum((y == i) & (pred == j))) for j in range(n)] for i in range(n)])
    support, predicted = cm.sum(1), cm.sum(0)
    f1 = np.divide(2 * np.diag(cm), support + predicted, out=np.zeros(n), where=(support + predicted) > 0)
    recall = np.divide(np.diag(cm), support, out=np.zeros(n), where=support > 0)
    return {
        "rows": int(len(y)), "macro_f1": float(f1.mean()),
        "per_class_f1": dict(zip(classes, map(float, f1))),
        "per_class_recall": dict(zip(classes, map(float, recall))), "confusion": cm.tolist(),
    }


def centered(logits, temperature):
    logits = logits - logits.mean(1, keepdims=True)
    return logits / temperature


def choose(old, new, y, mask, classes):
    best = None
    for old_t in TEMPERATURES:
        a = centered(old[mask], old_t)
        for new_t in TEMPERATURES:
            b = centered(new[mask], new_t)
            for alpha in ALPHAS:
                score = metric(y[mask], ((1-alpha)*a + alpha*b).argmax(1), classes)["macro_f1"]
                row = (score, -abs(float(alpha)-0.5), -new_t, -old_t, float(alpha), old_t, new_t)
                if best is None or row > best:
                    best = row
    return {"alpha": best[4], "old_temperature": best[5], "new_temperature": best[6], "tune_macro_f1": best[0]}


def predict_blend(old, new, params):
    return ((1-params["alpha"])*centered(old, params["old_temperature"]) +
            params["alpha"]*centered(new, params["new_temperature"])).argmax(1)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "status.json", {"status": "RUNNING"})
    frame = pd.read_csv(REF / "validation-samples.csv").sort_values(["ID", "endpoint"]).reset_index(drop=True)
    incumbent_checkpoint = torch.load(INCUMBENT, map_location="cpu", weights_only=True)
    mean, std = incumbent_checkpoint["mean"].numpy(), incumbent_checkpoint["std"].numpy()
    old_models = []
    for state in incumbent_checkpoint["models"]:
        model = incumbent_deploy.S3MotionModel(); model.load_state_dict(state); model.eval(); old_models.append(model)
    rgb_checkpoint = torch.load(RGB / "best.pt", map_location="cpu", weights_only=True)
    rgb_model = RGBGRU(); rgb_model.load_state_dict(rgb_checkpoint["model"]); rgb_model.eval()
    old_accel, old_steer, new_accel, new_steer = [], [], [], []
    with torch.inference_mode():
        for sid, group in frame.groupby("ID", sort=False):
            with np.load(REF / "cache" / f"{sid}.npz", allow_pickle=False) as z:
                endpoints = z["endpoints"].astype(int); old_x = z["features"][:, 1280:].astype(np.float32)
            assert np.array_equal(endpoints, group.endpoint.to_numpy(dtype=int))
            oa, os_ = incumbent_deploy.s3_motion_logits(old_models, old_x, mean, std, torch.device("cpu"))
            with np.load(RGB / "cache" / f"{sid}.npz", allow_pickle=False) as z:
                new_x, a, s = z["x"].astype(np.float32), z["a"], z["s"]
            assert len(new_x) == len(group) and np.array_equal(a, group.accel.to_numpy(dtype=int))
            assert np.array_equal(s, group.steer.to_numpy(dtype=int))
            na, ns = rgb_model(torch.from_numpy(new_x)[None]); old_accel.append(oa); old_steer.append(os_)
            new_accel.append(na[0].numpy()); new_steer.append(ns[0].numpy())
    old_accel, old_steer, new_accel, new_steer = map(np.concatenate, (old_accel, old_steer, new_accel, new_steer))
    ya, ys = frame.accel.to_numpy(dtype=int), frame.steer.to_numpy(dtype=int)
    moving = ya != 3
    baseline = {
        "incumbent": {"accel": metric(ya, old_accel.argmax(1), ACCEL), "moving_steer": metric(ys[moving], old_steer[moving].argmax(1), STEER)},
        "rgb_gru": {"accel": metric(ya, new_accel.argmax(1), ACCEL), "moving_steer": metric(ys[moving], new_steer[moving].argmax(1), STEER)},
    }
    for value in baseline.values():
        value["joint_mean"] = (value["accel"]["macro_f1"] + value["moving_steer"]["macro_f1"]) / 2
    routes = sorted(frame.route.unique())
    folds = []
    oof_accel = np.full(len(frame), -1); oof_steer = np.full(len(frame), -1)
    for route in routes:
        audit = frame.route.to_numpy() == route
        train = ~audit
        accel_params = choose(old_accel, new_accel, ya, train, ACCEL)
        steer_params = choose(old_steer, new_steer, ys, train & moving, STEER)
        oof_accel[audit] = predict_blend(old_accel[audit], new_accel[audit], accel_params)
        oof_steer[audit] = predict_blend(old_steer[audit], new_steer[audit], steer_params)
        folds.append({"held_out_route": route, "rows": int(audit.sum()), "accel_params": accel_params,
                      "steer_params": steer_params,
                      "audit": {"accel": metric(ya[audit], oof_accel[audit], ACCEL),
                                "moving_steer": metric(ys[audit & moving], oof_steer[audit & moving], STEER)}})
    oof = {"accel": metric(ya, oof_accel, ACCEL), "moving_steer": metric(ys[moving], oof_steer[moving], STEER)}
    oof["joint_mean"] = (oof["accel"]["macro_f1"] + oof["moving_steer"]["macro_f1"]) / 2
    final_params = {"accel": choose(old_accel, new_accel, ya, np.ones(len(frame), bool), ACCEL),
                    "steer": choose(old_steer, new_steer, ys, moving, STEER)}
    gates = {
        "joint_gain_ge_0.02": oof["joint_mean"] >= baseline["incumbent"]["joint_mean"] + 0.02,
        "accel_not_worse": oof["accel"]["macro_f1"] >= baseline["incumbent"]["accel"]["macro_f1"],
        "steer_not_worse": oof["moving_steer"]["macro_f1"] >= baseline["incumbent"]["moving_steer"]["macro_f1"],
        "left_f1_not_worse": oof["moving_steer"]["per_class_f1"]["LEFT"] >= baseline["incumbent"]["moving_steer"]["per_class_f1"]["LEFT"],
        "right_f1_not_worse": oof["moving_steer"]["per_class_f1"]["RIGHT"] >= baseline["incumbent"]["moving_steer"]["per_class_f1"]["RIGHT"],
    }
    report = {"status": "COMPLETE_VALIDATED", "decision": "ADVANCE_TO_EXTERNAL_GATES" if all(gates.values()) else "DO_NOT_PACKAGE",
              "scope": "Same 31 route-separated comma validation videos; blend hyperparameters selected out-of-route per fold; correlated frames and reused proxy labels",
              "rows": len(frame), "videos": int(frame.ID.nunique()), "routes": routes, "baseline": baseline,
              "route_held_out_blend": oof, "folds": folds, "final_full_validation_parameters": final_params,
              "gates": gates, "sources": {"incumbent_sha256": sha(INCUMBENT), "rgb_gru_sha256": sha(RGB / "best.pt")}}
    save(OUT / "report.json", report)
    save(OUT / "status.json", {"status": "COMPLETE_VALIDATED", "decision": report["decision"], "gates": gates})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
