"""Select MM-AU videos worth acquiring for Stage2 scene supervision.

The output is an acquisition priority list, not pseudo ground truth. Selection
uses only official metadata and object boxes around t_ai/t_co.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data_raw/mm-au-official-metadata-20260917/video_metadata.csv"
LABELS = ROOT / "data_raw/mm-au-detection-labels-20260917/labels.tar.gz"
OUT = ROOT / "artifacts/stage2-mmau-scene-selection-20260918"
TARGET = 2500
PATTERN = re.compile(r"^labels/(train|val|test)/(\d+)_(\d+)_(\d+)\.txt$")

SIDE_TERMS = {
    "cross": 3, "turn": 2, "chang": 2, "overtak": 2, "merge": 3,
    "reverse": 2, "left": 3, "right": 3, "intersection": 2,
    "pedestrian": 1, "motorcycle": 1, "motorbike": 1,
}
EVASION_TERMS = {
    "blocked": 3, "blind spot": 3, "narrow": 3, "obstacle": 3,
    "curb": 2, "out of control": 3, "brak": 1, "stop": 1,
    "roadside": 2, "give way": 1, "avoid": 2,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def term_score(text: str, terms: dict[str, int]) -> int:
    text = text.lower()
    return sum(weight for term, weight in terms.items() if term in text)


def parse_boxes(raw: str) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, width, height = map(float, parts[1:])
        except ValueError:
            continue
        if 0 <= cls <= 6 and all(0 <= x <= 1 for x in (cx, cy, width, height)):
            boxes.append((cls, cx, cy, width, height))
    return boxes


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    with META.open(encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))

    candidates = {}
    rejected = Counter()
    for row in raw_rows:
        try:
            if row["whether an accident occurred (1/0)"].strip() != "1":
                rejected["negative"] += 1
                continue
            tai = int(row["abnormal start frame"])
            tco = int(row["accident frame"])
            tae = int(row["abnormal end frame"])
            total = int(row["total frames"])
            if not (0 <= tai <= tco <= tae < total):
                rejected["invalid_temporal"] += 1
                continue
            key = f"{int(row['type'])}_{int(row['video'])}"
            text = " ".join((row.get("texts", ""), row.get("causes", ""), row.get("measures", "")))
            side_score = term_score(text, SIDE_TERMS)
            evasion_score = term_score(text, EVASION_TERMS)
            candidates[key] = {
                "hashcode": row["hashcode"], "video_key": key,
                "video_name": row["VideoName"], "type": int(row["type"]),
                "video_id": int(row["video"]), "t_ai": tai, "t_co": tco,
                "t_ae": tae, "total_frames": total,
                "text_side_score": side_score, "text_evasion_score": evasion_score,
                "texts": row.get("texts", "").strip(),
                "causes": row.get("causes", "").strip(),
                "label_split": "", "frames_near_tai": 0, "frames_near_tco": 0,
                "boxes_near_tai": 0, "boxes_near_tco": 0,
                "vehicle_left_near_tai": 0, "vehicle_right_near_tai": 0,
            }
        except (ValueError, KeyError):
            rejected["parse_error"] += 1

    seen_frames = defaultdict(lambda: {"ai": set(), "co": set()})
    with tarfile.open(LABELS, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = PATTERN.match(member.name)
            if not match:
                continue
            split, typ, video, frame_text = match.groups()
            key = f"{int(typ)}_{int(video)}"
            row = candidates.get(key)
            if row is None:
                continue
            frame = int(frame_text)
            near_ai = abs(frame - row["t_ai"]) <= 5
            near_co = abs(frame - row["t_co"]) <= 5
            if not (near_ai or near_co):
                continue
            boxes = parse_boxes(archive.extractfile(member).read().decode("utf-8", "replace"))
            row["label_split"] = split
            if near_ai:
                seen_frames[key]["ai"].add(frame)
                row["boxes_near_tai"] += len(boxes)
                for cls, cx, _cy, _w, _h in boxes:
                    if cls in (0, 1, 2, 4, 5, 6):
                        if cx <= 0.4:
                            row["vehicle_left_near_tai"] += 1
                        elif cx >= 0.6:
                            row["vehicle_right_near_tai"] += 1
            if near_co:
                seen_frames[key]["co"].add(frame)
                row["boxes_near_tco"] += len(boxes)

    ranked = []
    for key, row in candidates.items():
        row["frames_near_tai"] = len(seen_frames[key]["ai"])
        row["frames_near_tco"] = len(seen_frames[key]["co"])
        left, right = row["vehicle_left_near_tai"], row["vehicle_right_near_tai"]
        total_lr = left + right
        row["lateral_asymmetry"] = round(abs(left - right) / total_lr, 6) if total_lr else 0.0
        coverage = min(row["frames_near_tai"], 3) + min(row["frames_near_tco"], 3)
        spatial = min(math.log1p(row["boxes_near_tai"] + row["boxes_near_tco"]), 5)
        row["selection_score"] = round(
            2.0 * row["text_side_score"] + 1.5 * row["text_evasion_score"]
            + coverage + spatial + 2.0 * row["lateral_asymmetry"], 6
        )
        row["target_reason"] = "+".join(x for x, yes in (
            ("SIDE_TEXT", row["text_side_score"] > 0),
            ("EVASION_TEXT", row["text_evasion_score"] > 0),
            ("LATERAL_BOX", row["lateral_asymmetry"] >= 0.4),
            ("TEMPORAL_BOX", row["frames_near_tai"] and row["frames_near_tco"]),
        ) if yes)
        if row["label_split"] and row["target_reason"]:
            ranked.append(row)

    ranked.sort(key=lambda r: (-r["selection_score"], r["hashcode"]))
    # Cap dominant accident types first, then fill remaining slots by score.
    selected, deferred = [], []
    type_counts = Counter()
    for row in ranked:
        if type_counts[row["type"]] < 150:
            selected.append(row)
            type_counts[row["type"]] += 1
        else:
            deferred.append(row)
        if len(selected) == TARGET:
            break
    if len(selected) < TARGET:
        chosen = {r["hashcode"] for r in selected}
        for row in deferred + ranked:
            if row["hashcode"] in chosen:
                continue
            selected.append(row)
            chosen.add(row["hashcode"])
            if len(selected) == TARGET:
                break
    if len(selected) != TARGET or len({r["hashcode"] for r in selected}) != TARGET:
        raise RuntimeError("Could not form unique target selection")

    OUT.mkdir(parents=True)
    fields = list(selected[0])
    with (OUT / "selected_2500.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(selected)
    report = {
        "status": "COMPLETE_VALIDATED",
        "purpose": "Acquisition priority only; no pseudo labels are asserted",
        "metadata_rows": len(raw_rows), "valid_positive_temporal": len(candidates),
        "rankable_with_detection_and_scene_signal": len(ranked), "selected": len(selected),
        "rejected": dict(rejected),
        "selected_type_counts": dict(sorted(Counter(str(r["type"]) for r in selected).items(), key=lambda x: int(x[0]))),
        "selected_split_counts": dict(Counter(r["label_split"] for r in selected)),
        "reason_counts": dict(Counter(reason for r in selected for reason in r["target_reason"].split("+") if reason)),
        "score": {"min": min(r["selection_score"] for r in selected), "max": max(r["selection_score"] for r in selected),
                  "mean": sum(r["selection_score"] for r in selected) / len(selected)},
        "inputs": {"metadata_sha256": sha256(META), "detection_labels_sha256": sha256(LABELS)},
        "output_sha256": {"selected_2500.csv": sha256(OUT / "selected_2500.csv")},
        "limitations": [
            "Selection terms and box asymmetry prioritize downloads; they are not competition labels.",
            "MM-AU boxes have no track IDs or collision-actor identity.",
            "Video archives are large type-group shards, so selective retrieval depends on shard layout.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
