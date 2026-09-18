"""Measure whether the existing Stage2 scene ensemble is safe for pseudo-labeling.

This uses only source-separated OOF predictions for manually labeled CCD videos.
Seed agreement is treated as a selection signal, never as ground truth.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/stage2-semantic-crop-seeds-20260916"
OUTPUT = ROOT / "artifacts/stage2-pseudolabel-precision-20260918"
TASKS = {
    "evasion_space": ("evasion_space", "pred_evasion_space"),
    "entry_side": ("entry_side", "pred_entry_side"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_lower(correct: int, total: int, z: float = 1.96) -> float:
    if not total:
        return 0.0
    p = correct / total
    den = 1 + z * z / total
    centre = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - radius) / den


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = sorted(INPUT.glob("seed-*/semantic_crop_oof.csv"))
    if len(files) != 5:
        raise RuntimeError(f"Expected five seed outputs, found {len(files)}")

    frames = []
    hashes = {}
    for path in files:
        frame = pd.read_csv(path, dtype={"ID": str, "source_id": str})
        if len(frame) != 66 or frame.ID.nunique() != 66:
            raise RuntimeError(f"Incomplete OOF file: {path}")
        frames.append(frame.set_index("ID"))
        hashes[str(path.relative_to(ROOT))] = sha256(path)

    ids = list(frames[0].index)
    if any(list(f.index) != ids for f in frames[1:]):
        raise RuntimeError("OOF ID order differs between seeds")
    truth_columns = ["evasion_space", "entry_side", "source_id"]
    if any(not frames[0][truth_columns].equals(f[truth_columns]) for f in frames[1:]):
        raise RuntimeError("Truth/source columns differ between seeds")

    selected_rows = []
    report = {
        "status": "RUNNING",
        "scope": "Five-seed source-separated OOF predictions on 66 manually labeled CCD videos",
        "limitations": [
            "Seeds share the same architecture, data and folds, so agreement is correlated evidence.",
            "The sample is small and source distribution may differ from evaluation data.",
            "No evaluation data or Nexar frozen validation labels are used.",
        ],
        "input_sha256": hashes,
        "tasks": {},
    }

    for task, (truth_col, pred_col) in TASKS.items():
        observations = []
        for sample_id in ids:
            predictions = [str(f.at[sample_id, pred_col]) for f in frames]
            counts = Counter(predictions)
            predicted, votes = counts.most_common(1)[0]
            truth = str(frames[0].at[sample_id, truth_col])
            observations.append({
                "ID": sample_id,
                "source_id": str(frames[0].at[sample_id, "source_id"]),
                "task": task,
                "truth": truth,
                "prediction": predicted,
                "votes": votes,
                "correct": predicted == truth,
            })
        selected_rows.extend(observations)
        thresholds = {}
        for minimum_votes in range(3, 6):
            chosen = [r for r in observations if r["votes"] >= minimum_votes]
            correct = sum(r["correct"] for r in chosen)
            per_class = {}
            for label in sorted({r["prediction"] for r in chosen}):
                rows = [r for r in chosen if r["prediction"] == label]
                hits = sum(r["correct"] for r in rows)
                per_class[label] = {
                    "selected": len(rows),
                    "precision": hits / len(rows),
                    "wilson95_lower": wilson_lower(hits, len(rows)),
                }
            thresholds[str(minimum_votes)] = {
                "selected": len(chosen),
                "coverage": len(chosen) / len(observations),
                "correct": correct,
                "precision": correct / len(chosen) if chosen else None,
                "wilson95_lower": wilson_lower(correct, len(chosen)),
                "unique_sources": len({r["source_id"] for r in chosen}),
                "prediction_distribution": dict(Counter(r["prediction"] for r in chosen)),
                "per_class": per_class,
            }
        report["tasks"][task] = {"thresholds": thresholds}

    # A pseudo-label stream is permitted only if both classes of a task have
    # reasonable observed precision and a non-trivial lower confidence bound.
    gates = {}
    for task, task_report in report["tasks"].items():
        strict = task_report["thresholds"]["5"]
        class_stats = list(strict["per_class"].values())
        gates[task] = {
            "pass": bool(
                strict["selected"] >= 20
                and strict["precision"] >= 0.75
                and strict["wilson95_lower"] >= 0.60
                and len(class_stats) == 2
                and all(x["selected"] >= 5 and x["precision"] >= 0.70 for x in class_stats)
            ),
            "policy": "5/5 seed agreement; n>=20; precision>=.75; Wilson lower>=.60; both classes n>=5 and precision>=.70",
        }
    report["promotion_gates"] = gates
    report["decision"] = (
        "SAFE_TO_GENERATE_CANDIDATE_PSEUDOLABELS"
        if any(v["pass"] for v in gates.values())
        else "DO_NOT_GENERATE_PSEUDOLABELS_FROM_THIS_ENSEMBLE"
    )
    report["status"] = "COMPLETE_VALIDATED"

    OUTPUT.mkdir(parents=True)
    pd.DataFrame(selected_rows).to_csv(OUTPUT / "oof_consensus.csv", index=False)
    report["output_sha256"] = {"oof_consensus.csv": sha256(OUTPUT / "oof_consensus.csv")}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
