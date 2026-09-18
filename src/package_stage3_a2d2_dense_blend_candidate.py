"""Package the preselected dense-A2D2 steer-logit blend as a submit candidate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from daily_submission import inspect_zip


P = Path(__file__).resolve().parents[1]
OUT = P / "artifacts/stage3-a2d2-dense-blend-submit-candidate-20260918"
BASE = P / "artifacts/stage2-nexar-gru2-submit-candidate-20260918/submit.zip"
CURRENT = P / "artifacts/stage3-a2d2-blend-submit-candidate-20260918/best.pt"
DENSE_DIR = P / "artifacts/stage3-a2d2-dense-rehearsal-20260918"
REVIEW = P / "artifacts/stage3-a2d2-dense-blend-20260918/report.json"
PARITY_DATA = P / "artifacts/stage3-a2d2-clip-pilot-20260917"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    selected = review["selected"]
    assert review["status"] == "COMPLETE_VALIDATED"
    assert review["decision"] == "ADOPT_BLEND"
    assert selected["passed"] is True
    candidate_name = selected["candidate"]
    alpha = float(selected["adapted_weight"])
    assert candidate_name == "dense25" and alpha == 0.3

    current = torch.load(CURRENT, map_location="cpu", weights_only=True)
    adapted_path = DENSE_DIR / f"{candidate_name}.pt"
    adapted = torch.load(adapted_path, map_location="cpu", weights_only=True)
    assert current["format"] == adapted["format"] == "stage3-motion-ensemble-v1"
    assert current["seeds"] == adapted["seeds"]

    blend = dict(current)
    blend["models"] = []
    for old_state, new_state in zip(current["models"], adapted["models"]):
        assert old_state.keys() == new_state.keys()
        state = {}
        for key in old_state:
            if key.startswith("steer."):
                state[key] = (1.0 - alpha) * old_state[key] + alpha * new_state[key]
            else:
                assert torch.equal(old_state[key], new_state[key])
                state[key] = old_state[key]
        blend["models"].append(state)

    best = OUT / "best.pt"
    torch.save(blend, best)
    target = OUT / "submit.zip"
    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(
        target, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as destination:
        for info in source.infolist():
            payload = best.read_bytes() if info.filename == "model/stage3/best.pt" else source.read(info.filename)
            destination.writestr(info.filename, payload)

    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(target) as candidate_zip:
        assert source.namelist() == candidate_zip.namelist()
        changed = [
            name for name in source.namelist() if source.read(name) != candidate_zip.read(name)
        ]
        assert changed == ["model/stage3/best.pt"]
        inference = OUT / "inference.py"
        inference.write_bytes(candidate_zip.read("inference.py"))

    spec = importlib.util.spec_from_file_location("dense_candidate", inference)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    device = torch.device("cpu")
    actual_models, mean, std = module.s3_motion_load(best, device)
    current_models, _, _ = module.s3_motion_load(CURRENT, device)
    adapted_models, _, _ = module.s3_motion_load(adapted_path, device)
    max_error = 0.0
    rows = 0
    for video in sorted((PARITY_DATA / "videos").glob("*.mp4")):
        features = module.s3_motion_features(video)[15:16]
        actual = module.s3_motion_logits(actual_models, features, mean, std, device)[1]
        full = np.zeros((1, 1682), np.float32)
        full[:, 1280:] = np.clip((features - mean[1280:]) / std[1280:], -10, 10)
        tensor = torch.from_numpy(full)
        with torch.inference_mode():
            current_logits = torch.stack([model(tensor)[1] for model in current_models]).mean(0)
            adapted_logits = torch.stack([model(tensor)[1] for model in adapted_models]).mean(0)
            expected = (1.0 - alpha) * current_logits + alpha * adapted_logits
        error = float(np.max(np.abs(actual - expected.numpy())))
        max_error = max(max_error, error)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=2e-6)
        rows += 1

    digest = inspect_zip(target)
    report = {
        "status": "ZIP_READY_CPU_VALIDATED_GPU_PENDING",
        "candidate_sha256": digest,
        "bytes": target.stat().st_size,
        "base_sha256": sha(BASE),
        "checkpoint_sha256": sha(best),
        "current_checkpoint_sha256": sha(CURRENT),
        "adapted_checkpoint_sha256": sha(adapted_path),
        "selected_candidate": candidate_name,
        "adapted_weight": alpha,
        "changed_entries": changed,
        "cpu_parity_rows": rows,
        "max_logits_error": max_error,
        "review_sha256": sha(REVIEW),
        "submitted": False,
    }
    (OUT / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
