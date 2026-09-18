"""Audit fixed, label-free combinations of validated Stage1 pilot predictions."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/kaggle-stage1-screen-trial-20260914/validated"
OUT = ROOT / "artifacts/stage1-fixed-ensemble-20260916"
ARMS = ("baseline", "mixed_50", "screen_mix_50")
KEYS = ("path", "source_id", "upload_group", "condition", "suite", "sha256", "target")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(arm):
    path = PILOT / f"{arm}-predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 420 or len({r["path"] for r in rows}) != 420:
        raise ValueError(f"Invalid prediction coverage: {arm}")
    for row in rows:
        if row["target"] not in ("ORIGINAL", "RERECORDED"):
            raise ValueError("Invalid target")
        if len(row["slot_probabilities"]) != 3:
            raise ValueError("Invalid slots")
        if abs(row["probability"] - sum(row["slot_probabilities"]) / 3) > 1e-7:
            raise ValueError("Probability is not slot mean")
        if row["answer"] != ("RERECORDED" if row["probability"] >= 0.5 else "ORIGINAL"):
            raise ValueError("Saved decision mismatch")
    return path, rows


def macro_f1(rows):
    scores = []
    for label in ("ORIGINAL", "RERECORDED"):
        tp = sum(r["target"] == label and r["answer"] == label for r in rows)
        fp = sum(r["target"] != label and r["answer"] == label for r in rows)
        fn = sum(r["target"] == label and r["answer"] != label for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / 2


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    normal = [r for r in rows if r["suite"] != "stress"]
    return {
        "normal_macro_f1": macro_f1(normal),
        "conditions": {name: {"videos": len(group), "errors": sum(r["answer"] != r["target"] for r in group)}
                       for name, group in sorted(groups.items())},
    }


def main():
    validation = json.loads((PILOT / "result-validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS" or validation["prediction_rows"] != 1260:
        raise ValueError("Pilot validation is not complete")
    loaded = {arm: load(arm) for arm in ARMS}
    reference = loaded["baseline"][1]
    for arm in ARMS[1:]:
        if [tuple(r[k] for k in KEYS) for r in loaded[arm][1]] != [tuple(r[k] for k in KEYS) for r in reference]:
            raise ValueError("Input/target/order mismatch")
    variants = {arm: rows for arm, (_, rows) in loaded.items()}
    for arm in ARMS[1:]:
        rows = []
        for left, right in zip(reference, loaded[arm][1]):
            probability = (left["probability"] + right["probability"]) / 2
            rows.append({**{k: left[k] for k in KEYS}, "probability": probability,
                         "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL"})
        variants[f"baseline_plus_{arm}_half"] = rows
    summaries = {name: summarize(rows) for name, rows in variants.items()}
    baseline = summaries["baseline"]["conditions"]
    for name, summary in summaries.items():
        conditions = summary["conditions"]
        original_improved = all(conditions[c]["errors"] < baseline[c]["errors"]
                                for c in ("original_noise", "original_quality_mix"))
        rerecording_preserved = all(conditions[c]["errors"] <= baseline[c]["errors"] + 1
                                     for c in conditions if c.startswith("rerecorded_"))
        summary["gate"] = {
            "original_noise_and_mix_improved": original_improved,
            "rerecorded_each_condition_preserved": rerecording_preserved,
            "normal_f1_preserved": summary["normal_macro_f1"] >= summaries["baseline"]["normal_macro_f1"] - 0.02,
        }
        summary["gate"]["pass"] = all(summary["gate"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"status": "COMPLETE_VALIDATED", "scope": "Fixed 50/50 probability average on reused synthetic development videos; no independent generalization claim",
              "threshold": 0.5, "input_sha256": {arm: sha(path) for arm, (path, _) in loaded.items()},
              "validation_sha256": sha(PILOT / "result-validation.json"), "summaries": summaries}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: {"gate": value["gate"], "normal_macro_f1": value["normal_macro_f1"],
                             "noise_errors": value["conditions"]["original_noise"]["errors"],
                             "quality_mix_errors": value["conditions"]["original_quality_mix"]["errors"],
                             "weak_screen_errors": value["conditions"]["rerecorded_weak_screen"]["errors"]}
                      for name, value in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
