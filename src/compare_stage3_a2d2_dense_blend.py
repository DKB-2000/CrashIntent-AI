"""Compare low-weight dense A2D2 blends against the submitted Stage3 checkpoint."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import stage3_motion_inference as deploy
import train_stage3_a2d2_rehearsal as base
import train_stage3_a2d2_dense_rehearsal as dense

P = Path(__file__).resolve().parents[1]
OUT = P / "artifacts/stage3-a2d2-dense-blend-20260918"
CURRENT = P / "artifacts/stage3-a2d2-blend-submit-candidate-20260918/best.pt"
DENSE = P / "artifacts/stage3-a2d2-dense-rehearsal-20260918"
ALPHAS = (.10, .20, .30, .40, .50)


def models(checkpoint):
    result = []
    for state in checkpoint["models"]:
        model = deploy.S3MotionModel(); model.load_state_dict(state); model.eval(); result.append(model)
    return result


def main():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    current_ck = torch.load(CURRENT, map_location="cpu", weights_only=True)
    mean, std = current_ck["mean"].numpy(), current_ck["std"].numpy()
    valid = pd.read_csv(base.REF / "validation-samples.csv")
    valid = valid[valid.accel != 3].reset_index(drop=True)
    vx, vy = base.old_arrays(valid), valid.steer.to_numpy()
    hx, hy, _ = dense.dense_arrays(dense.HOLD, dense.HOLD_PLAN, "20180810_150607")
    _, ex, ey = base.a2d2_arrays(base.HOLD)
    cx, cy = base.civic_arrays()
    sets = {"old": (vx, vy), "a2d2_dense": (hx, hy), "a2d2_endpoint": (ex, ey), "civic": (cx, cy)}

    def logits(group, x):
        full = np.zeros((len(x), 1682), np.float32)
        full[:, 1280:] = np.clip((x-mean[1280:])/std[1280:], -10, 10)
        tensor = torch.from_numpy(full)
        with torch.inference_mode():
            return torch.stack([m(tensor)[1] for m in group]).mean(0).numpy()

    incumbent = models(current_ck)
    old_logits = {name: logits(incumbent, x) for name, (x, _) in sets.items()}
    baseline = {name: base.metric(y, old_logits[name].argmax(1)) for name, (_, y) in sets.items()}
    arms = []
    for candidate_name in ("dense10", "dense25", "dense50"):
        candidate_ck = torch.load(DENSE / f"{candidate_name}.pt", map_location="cpu", weights_only=True)
        candidate = models(candidate_ck)
        new_logits = {name: logits(candidate, x) for name, (x, _) in sets.items()}
        for alpha in ALPHAS:
            scores = {name: base.metric(y, ((1-alpha)*old_logits[name] + alpha*new_logits[name]).argmax(1))
                      for name, (_, y) in sets.items()}
            gates = {
                "old": scores["old"]["macro_f1"] >= baseline["old"]["macro_f1"] - .003,
                "civic": scores["civic"]["macro_f1"] >= baseline["civic"]["macro_f1"] - .005,
                "dense": scores["a2d2_dense"]["macro_f1"] > baseline["a2d2_dense"]["macro_f1"],
                "endpoint": scores["a2d2_endpoint"]["macro_f1"] >= baseline["a2d2_endpoint"]["macro_f1"],
            }
            arms.append({"candidate": candidate_name, "adapted_weight": alpha, "scores": scores,
                         "gates": gates, "passed": all(gates.values())})
    passed = [x for x in arms if x["passed"]]
    selected = max(passed, key=lambda x: (x["scores"]["a2d2_dense"]["macro_f1"],
                                          x["scores"]["a2d2_endpoint"]["macro_f1"])) if passed else None
    report = {"status": "COMPLETE_VALIDATED", "decision": "ADOPT_BLEND" if selected else "KEEP_CURRENT",
              "baseline": baseline, "selected": selected, "arms": arms,
              "current_checkpoint_sha256": base.sha(CURRENT)}
    base.save(OUT / "report.json", report)
    base.save(OUT / "status.json", {"status": "COMPLETE_VALIDATED", "decision": report["decision"],
                                     "selected": None if selected is None else {k:selected[k] for k in ("candidate","adapted_weight")}})
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
