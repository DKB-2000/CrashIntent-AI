"""Exploratory ordered joint event decoding on frozen Stage2 source-CV models.

The original argmax output is reproduced before any comparison. This script
does not train models or modify the deployed inference path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

import stage2_pipeline as p
from analyze_stage2_source_cv import details, summary

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "artifacts/stage2-source-cv-20260910"
OUT = ROOT / "artifacts/stage2-ordered-joint-20260916"
NAMES = ("exact_5", "exact_15", "soft_mixed_15")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_pair(collision: torch.Tensor, entry: torch.Tensor) -> tuple[int, int]:
    """Maximize independent event logits jointly subject to entry <= collision."""
    if collision.ndim != 1 or entry.shape != collision.shape or not len(collision):
        raise ValueError("Expected equal nonempty one-dimensional event logits")
    # For each collision position, find the best entry up to that position.
    best_entry_score, best_entry_index = torch.cummax(entry, dim=0)
    pair_score = collision + best_entry_score
    c = int(pair_score.argmax())
    return c, int(best_entry_index[c])


def predict(model: p.Stage2Temporal, features: torch.Tensor, ordered: bool) -> dict:
    collision, entry, hidden = model.logits(features.unsqueeze(0))
    cl, el = collision[0], entry[0]
    ci, ei = ordered_pair(cl, el) if ordered else (int(cl.argmax()), int(el.argmax()))
    scene = model.scene(torch.cat((hidden[:, ci], hidden[:, ei]), dim=1))
    return {
        "collision_frame": ci,
        "entry_frame": ei,
        "evasion_space": int(scene[:, :2].argmax(dim=1)),
        "entry_side": ("LEFT", "RIGHT")[int(scene[:, 2:].argmax(dim=1))],
    }


def smoke() -> None:
    for c, e, expected in (
        ([3., 0., 0.], [0., 2., 0.], (0, 0)),
        ([0., 3., 0.], [2., 0., 0.], (1, 0)),
        ([0., 0., 3.], [0., 0., 2.], (2, 2)),
    ):
        assert ordered_pair(torch.tensor(c), torch.tensor(e)) == expected
    assert ordered_pair(torch.zeros(3), torch.zeros(3)) == (0, 0)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite previous run: {OUT}")
    smoke()
    torch.set_num_threads(2)
    validation = json.loads((CV / "independent-validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS":
        raise RuntimeError("Original source-CV validation is not PASS")
    protocol = json.loads((CV / "protocol.json").read_text(encoding="utf-8"))
    cache = ROOT / "artifacts/stage2-controls-20260909/features.pt"
    if sha(cache) != protocol["features_sha256"]:
        raise RuntimeError("Frozen feature cache hash changed")
    features = torch.load(cache, weights_only=True, map_location="cpu")
    report = {
        "status": "RUNNING",
        "method": "Maximum collision_logit[c] + entry_logit[e], with e <= c; scene head recomputed at chosen positions",
        "scope": "Exploratory OOF comparison on reused 66-video development folds, not independent or official validation",
        "input_sha256": {"features": sha(cache)},
        "conditions": {},
    }
    OUT.mkdir()
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        with torch.inference_mode():
            for name in NAMES:
                expected = pd.read_csv(CV / f"{name}_oof.csv", dtype={"ID": str, "source_id": str}).set_index("ID")
                truth, base, joint = [], [], []
                for fold in range(5):
                    folder = CV / f"fold-{fold}"
                    train = pd.read_csv(folder / "train.csv", dtype={"ID": str, "source_id": str})
                    val = pd.read_csv(folder / "validation.csv", dtype={"ID": str, "source_id": str})
                    if set(train.source_id) & set(val.source_id):
                        raise RuntimeError("Training/validation source overlap")
                    checkpoint = folder / f"{name}.pt"
                    report["input_sha256"][f"fold-{fold}/{name}"] = sha(checkpoint)
                    model = p.Stage2Temporal()
                    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"], strict=True)
                    model.eval()
                    for row in val.to_dict("records"):
                        id_ = row["ID"]
                        old = predict(model, features[id_], ordered=False)
                        if any(old[k] != expected.loc[id_, "pred_" + k] for k in p.OUTPUT_COLUMNS[1:]):
                            raise RuntimeError(f"Original predictions do not reproduce: {name}/{id_}")
                        new = predict(model, features[id_], ordered=True)
                        truth.append(row)
                        base.append({"ID": id_, **old})
                        joint.append({"ID": id_, **new})
                if len(truth) != 66 or len({r["ID"] for r in truth}) != 66:
                    raise RuntimeError("Incomplete OOF coverage")
                old_rows = pd.DataFrame(details(truth, base))
                new_rows = pd.DataFrame(details(truth, joint))
                old_rows.to_csv(OUT / f"{name}_argmax.csv", index=False)
                new_rows.to_csv(OUT / f"{name}_ordered_joint.csv", index=False)
                old_score, new_score = summary(truth, base), summary(truth, joint)
                report["conditions"][name] = {
                    "argmax": old_score, "ordered_joint": new_score,
                    "changed_collision": sum(a["collision_frame"] != b["collision_frame"] for a, b in zip(base, joint)),
                    "changed_entry": sum(a["entry_frame"] != b["entry_frame"] for a, b in zip(base, joint)),
                    "changed_evasion": sum(a["evasion_space"] != b["evasion_space"] for a, b in zip(base, joint)),
                    "changed_side": sum(a["entry_side"] != b["entry_side"] for a, b in zip(base, joint)),
                    "ordered_violations": int(new_rows.pred_entry_after_collision.sum()),
                }
                if report["conditions"][name]["ordered_violations"]:
                    raise RuntimeError("Ordered decoder emitted entry after collision")
        report["status"] = "COMPLETE_VALIDATED"
        report["output_sha256"] = {f.name: sha(f) for f in OUT.glob("*.csv")}
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = repr(exc)
        raise
    finally:
        (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
