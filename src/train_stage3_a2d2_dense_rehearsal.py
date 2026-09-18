"""Rehearse dense, sensor-aligned A2D2 adaptation using existing causal clips."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import stage3_motion_inference as deploy
import train_stage3_a2d2_rehearsal as base


P = Path(__file__).resolve().parents[1]
OUT = P / "artifacts/stage3-a2d2-dense-rehearsal-20260918"
CKPT = P / "artifacts/stage3-warmstart-ce-ensemble-submit-candidate-20260917/best.pt"
TRAIN = P / "artifacts/stage3-a2d2-training-clips-20260917"
TRAIN_PLAN = P / "artifacts/stage3-a2d2-training-plan-20260917/report.json"
HOLD = P / "artifacts/stage3-a2d2-clip-pilot-20260917"
HOLD_PLAN = P / "artifacts/stage3-a2d2-pilot-plan-20260917/report.json"
BUSES = {
    "20190401_121727": P / "data_raw/a2d2/20190401121727_bus_signals.json",
    "20190401_145936": P / "data_raw/a2d2/20190401145936_bus_signals.json",
    "20180810_150607": P / "data_raw/a2d2/20180810_150607_bus_signals.json",
}
CLASSES = ["LEFT", "STRAIGHT", "RIGHT"]
ARMS = {"dense10": .10, "dense25": .25, "dense50": .50}


def pairs(payload, name):
    array = np.asarray(payload[name]["values"], np.float64)
    assert array.ndim == 2 and array.shape[1] == 2 and np.all(np.diff(array[:, 0]) > 0)
    return array[:, 0], array[:, 1]


def route_signals(route):
    payload = json.loads(BUSES[route].read_text(encoding="utf-8"))
    yt, yaw = pairs(payload, "angular_velocity_omega_z")
    st, magnitude = pairs(payload, "steering_angle_calculated")
    sgt, sign = pairs(payload, "steering_angle_calculated_sign")
    vt, speed = pairs(payload, "vehicle_speed")
    assert np.array_equal(st, sgt)
    # A2D2 sign bit is opposite the desired left-positive convention on all audited routes.
    signed = np.where(sign >= .5, -magnitude, magnitude)
    return yt, yaw, st, signed, vt, speed


def label_at(signals, timestamp):
    yt, yaw, st, steer, vt, speed = signals
    y = float(np.interp(timestamp, yt, yaw))
    a = float(np.interp(timestamp, st, steer))
    v = float(np.interp(timestamp, vt, speed))
    if v < 5:
        return None
    if y >= 5 and a >= 20:
        return 0
    if abs(y) <= .75 and abs(a) <= 5:
        return 1
    if y <= -5 and a <= -20:
        return 2
    return None


def dense_arrays(root, plan_path, route_default=None):
    samples = pd.read_csv(root / "samples.csv")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))["selections"]
    assert len(samples) == len(plan)
    signal_cache = {}
    xs, ys, rows = [], [], []
    for sample, selected in zip(samples.itertuples(index=False), plan):
        route = selected.get("route", route_default)
        signal_cache.setdefault(route, route_signals(route))
        features = deploy.s3_motion_features(root / "videos" / f"{sample.ID}.mp4")
        assert features.shape == (16, 402)
        center_ts = int(selected["timestamp_us"])
        for endpoint in range(8, 16):
            timestamp = center_ts - (15 - endpoint) * 100_000
            label = label_at(signal_cache[route], timestamp)
            if label is None:
                continue
            xs.append(features[endpoint])
            ys.append(label)
            rows.append((sample.ID, route, endpoint, timestamp, CLASSES[label]))
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64), rows


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "status.json"
    base.save(status, {"status": "EXTRACTING"})
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    mean, std = ckpt["mean"].numpy(), ckpt["std"].numpy()
    old_train = pd.read_csv(base.REF / "train-samples.csv")
    old_valid = pd.read_csv(base.REF / "validation-samples.csv")
    old_train = old_train[old_train.accel != 3].reset_index(drop=True)
    old_valid = old_valid[old_valid.accel != 3].reset_index(drop=True)
    ox, oy = base.old_arrays(old_train), old_train.steer.to_numpy()
    vx, vy = base.old_arrays(old_valid), old_valid.steer.to_numpy()
    ax, ay, train_rows = dense_arrays(TRAIN, TRAIN_PLAN)
    hx, hy, hold_rows = dense_arrays(HOLD, HOLD_PLAN, "20180810_150607")
    _, endpoint_hx, endpoint_hy = base.a2d2_arrays(HOLD)
    cx, cy = base.civic_arrays()

    def full(x):
        z = np.zeros((len(x), 1682), np.float32)
        z[:, 1280:] = np.clip((x - mean[1280:]) / std[1280:], -10, 10)
        return torch.from_numpy(z)

    tensors = list(map(full, (ox, ax, vx, hx, endpoint_hx, cx)))
    models = []
    for state in ckpt["models"]:
        model = deploy.S3MotionModel()
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    def predictions(group, index):
        with torch.inference_mode():
            return torch.stack([m(tensors[index])[1] for m in group]).mean(0).argmax(1).numpy()

    baseline = {
        "old": base.metric(vy, predictions(models, 2)),
        "a2d2_dense": base.metric(hy, predictions(models, 3)),
        "a2d2_endpoint": base.metric(endpoint_hy, predictions(models, 4)),
        "civic": base.metric(cy, predictions(models, 5)),
    }
    results = []
    for arm, weight in ARMS.items():
        adapted = []
        for source in models:
            with torch.no_grad():
                old_embedding = source.net(tensors[0]).detach()
                a2d2_embedding = source.net(tensors[1]).detach()
            head = copy.deepcopy(source.steer)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            old_y = torch.tensor(oy)
            a2d2_y = torch.tensor(ay)
            for _ in range(300):
                loss = ((1-weight) * F.cross_entropy(head(old_embedding), old_y)
                        + weight * F.cross_entropy(head(a2d2_embedding), a2d2_y))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                optimizer.step()
            model = copy.deepcopy(source)
            model.steer.load_state_dict(head.state_dict())
            model.eval()
            adapted.append(model)
        scores = {
            "old": base.metric(vy, predictions(adapted, 2)),
            "a2d2_dense": base.metric(hy, predictions(adapted, 3)),
            "a2d2_endpoint": base.metric(endpoint_hy, predictions(adapted, 4)),
            "civic": base.metric(cy, predictions(adapted, 5)),
        }
        gates = {
            "old": scores["old"]["macro_f1"] >= baseline["old"]["macro_f1"] - .005,
            "civic": scores["civic"]["macro_f1"] >= baseline["civic"]["macro_f1"] - .01,
            "dense_f1": scores["a2d2_dense"]["macro_f1"] > baseline["a2d2_dense"]["macro_f1"],
            "endpoint_f1": scores["a2d2_endpoint"]["macro_f1"] > baseline["a2d2_endpoint"]["macro_f1"],
        }
        candidate = copy.deepcopy(ckpt)
        candidate["models"] = [m.state_dict() for m in adapted]
        path = OUT / f"{arm}.pt"
        torch.save(candidate, path)
        results.append({"arm": arm, "weight": weight, "scores": scores, "gates": gates,
                        "passed": all(gates.values()), "checkpoint": str(path),
                        "checkpoint_sha256": base.sha(path)})
        base.save(status, {"status": "TRAINING", "completed": len(results), "total": len(ARMS)})
    passed = [r for r in results if r["passed"]]
    selected = max(passed, key=lambda r: r["scores"]["a2d2_dense"]["macro_f1"])["arm"] if passed else None
    report = {
        "status": "COMPLETE_VALIDATED", "decision": "ADOPT_FOR_BLEND_SEARCH" if selected else "KEEP_CURRENT",
        "selected": selected, "baseline": baseline, "arms": results,
        "contracts": {"train_samples": len(ax), "holdout_samples": len(hx),
                      "train_class_counts": dict(zip(CLASSES, np.bincount(ay, minlength=3).tolist())),
                      "holdout_class_counts": dict(zip(CLASSES, np.bincount(hy, minlength=3).tolist())),
                      "endpoints": "8..15 only", "train_routes": sorted({r[1] for r in train_rows}),
                      "holdout_routes": sorted({r[1] for r in hold_rows}),
                      "trainable": "steer.weight and steer.bias only"},
    }
    base.save(OUT / "report.json", report)
    base.save(status, {"status": "COMPLETE_VALIDATED", "completed": len(ARMS), "total": len(ARMS),
                       "decision": report["decision"], "selected": selected})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
