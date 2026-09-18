"""Bounded CPU diagnostic on the automatically labelled DLC-2021 partial pilot."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s

from evaluate_stage1_robustness import digest


DATA = Path("artifacts/stage1-dlc-partial-pilot-20260917")
OUT = Path("artifacts/stage1-dlc-pilot-evaluation-20260917")
BASE_ZIP = Path("artifacts/stage1-auto-release-20260910/candidate/submit.zip")
SCREEN25 = Path("artifacts/kaggle-stage1-quarter-mix-trial-20260916/validated/screen_mix_25.pt")


def save(value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "status.json.tmp"
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, OUT / "status.json")


def checkpoint_sources():
    with zipfile.ZipFile(BASE_ZIP) as z:
        yield "official_best", z.read("model/stage1/best.pt")
        serving = z.read("inference.py")
    yield "screen_mix_25", SCREEN25.read_bytes()
    return serving


def main() -> None:
    started = time.time()
    state = {"status": "RUNNING", "pid": os.getpid(), "started_unix": started,
             "dataset_status_sha256": digest(DATA / "status.json"), "completed_models": 0,
             "expected_models": 2, "scope": "SINGLE_TEMPLATE_INITIAL_DIAGNOSTIC"}
    save(state)
    try:
        dataset_state = json.loads((DATA / "status.json").read_text(encoding="utf-8"))
        if dataset_state["status"] != "DATA_VALIDATED":
            raise ValueError("Dataset is not validated")
        with (DATA / "manifest.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        with zipfile.ZipFile(BASE_ZIP) as z:
            code = z.read("inference.py")
        serving_path = OUT / "serving_inference.py"
        serving_path.write_bytes(code)
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("dlc_pilot_serving", serving_path)
        serving = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = serving
        spec.loader.exec_module(serving)
        torch.set_num_threads(2)
        clips = []
        for row in rows:
            path = Path(row["video_path"])
            clips.append([serving._decode_stage1_clip(path, 224, serving._clip_ids(path, 16, slot, 3))
                          for slot in range(3)])
        results = []
        sources = []
        with zipfile.ZipFile(BASE_ZIP) as z:
            sources.append(("official_best", z.read("model/stage1/best.pt")))
        sources.append(("screen_mix_25", SCREEN25.read_bytes()))
        for model_name, payload in sources:
            checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
            model = mvit_v2_s(weights=None)
            model.head[1] = nn.Linear(model.head[1].in_features, 2)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            with torch.inference_mode():
                for row, slots in zip(rows, clips):
                    probs = [float(model(x.unsqueeze(0)).softmax(1)[0, 1]) for x in slots]
                    probability = float(np.mean(probs))
                    results.append({"model": model_name, "source_id": row["source_id"], "label": row["label"],
                                    "probability": probability,
                                    "answer": "RERECORDED" if probability >= .5 else "ORIGINAL",
                                    "correct": (probability >= .5) == (row["label"] == "RERECORDED"),
                                    "slot_probabilities": probs})
            state.update(completed_models=state["completed_models"] + 1, heartbeat_unix=time.time())
            save(state)
        summaries = []
        for name, _ in sources:
            part = [r for r in results if r["model"] == name]
            for label in ("ORIGINAL", "RERECORDED"):
                group = [r for r in part if r["label"] == label]
                summaries.append({"model": name, "label": label, "clips": len(group),
                                  "accuracy": sum(r["correct"] for r in group) / len(group),
                                  "mean_rerecorded_probability": float(np.mean([r["probability"] for r in group]))})
        (OUT / "predictions.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        (OUT / "report.json").write_text(json.dumps({"summaries": summaries,
            "limitations": dataset_state.get("limitations", []) + ["No fitting or threshold selection used this pilot."]}, indent=2) + "\n", encoding="utf-8")
        state.update(status="COMPLETE_VALIDATED", completed_models=2, completed_predictions=len(results),
                     report_sha256=digest(OUT / "report.json"), seconds=time.time() - started)
        save(state)
    except Exception as exc:
        state.update(status="FAILED", error=repr(exc), seconds=time.time() - started)
        save(state)
        raise


if __name__ == "__main__":
    main()
