"""Run fixed semantic-crop probe over predefined seeds and validate aggregation."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage2-semantic-crop-seeds-20260916"
SCRIPT = ROOT / "src/compare_stage2_semantic_crops.py"
SEEDS = (20260825, 20260826, 20260827, 20260916, 20260917)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True)
    report = {
        "status": "RUNNING", "seeds": list(SEEDS), "completed": 0,
        "scope": "Same frozen 5 source folds and 66 reused development videos; seed stability only",
        "results": {},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        for seed in SEEDS:
            result_rel = f"artifacts/stage2-semantic-crop-seeds-20260916/seed-{seed}"
            env = os.environ.copy()
            env["STAGE2_SEMANTIC_CROPS_OUT"] = result_rel
            env["STAGE2_SEMANTIC_CROPS_SEED"] = str(seed)
            log_path = OUT / f"seed-{seed}.log"
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            folder = ROOT / result_rel
            child = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            if child.get("status") != "COMPLETE_VALIDATED" or child.get("seed") != seed:
                raise RuntimeError(f"Invalid seed result: {seed}")
            report["results"][str(seed)] = {
                name: {key: child["conditions"][name][key] for key in
                       ("evasion_space_accuracy", "entry_side_accuracy", "development_mean", "source_equal_mean")}
                for name in ("old_scene", "point", "semantic_crop")
            }
            report["results"][str(seed)]["output_sha256"] = child["output_sha256"]
            report["completed"] += 1
            (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        semantic = [report["results"][str(s)]["semantic_crop"] for s in SEEDS]
        point = [report["results"][str(s)]["point"] for s in SEEDS]
        old = report["results"][str(SEEDS[0])]["old_scene"]
        report["aggregate"] = {
            key: {
                "old_scene": old[key],
                "point_mean": sum(x[key] for x in point) / len(point),
                "semantic_crop_mean": sum(x[key] for x in semantic) / len(semantic),
                "semantic_crop_min": min(x[key] for x in semantic),
                "semantic_crop_max": max(x[key] for x in semantic),
                "semantic_better_than_old_all_seeds": all(x[key] > old[key] for x in semantic),
            }
            for key in ("evasion_space_accuracy", "entry_side_accuracy", "development_mean", "source_equal_mean")
        }
        # Frozen original predictions must be identical across all seed runs.
        old_hashes = [report["results"][str(s)]["output_sha256"]["old_scene_oof.csv"] for s in SEEDS]
        if len(set(old_hashes)) != 1:
            raise RuntimeError("Original scene output changed across seed runs")
        report["status"] = "COMPLETE_VALIDATED"
        report["report_sha256_before_final_write"] = sha(OUT / "report.json")
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = repr(exc)
        raise
    finally:
        (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
