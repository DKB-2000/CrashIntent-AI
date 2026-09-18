"""Audit official DAD object-track annotations for Stage2 scene supervision."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data_raw/dad/annotation.zip"
OUT = ROOT / "artifacts/stage2-dad-annotation-audit-20260918"
FRAME_WIDTH = 1280.0  # Official dataset is 720p (1280x720).


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = []
    malformed = []
    class_counts = Counter()
    involved_class_counts = Counter()
    all_lines = involved_lines = 0

    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC failure")
        names = sorted(x for x in archive.namelist() if x.endswith(".txt"))
        if len(names) != 620:
            raise RuntimeError(f"Expected 620 annotations, found {len(names)}")
        for name in names:
            tracks = defaultdict(list)
            for line_number, raw in enumerate(archive.read(name).decode("utf-8").splitlines(), 1):
                parts = raw.split()
                if len(parts) != 8:
                    malformed.append({"file": name, "line": line_number, "reason": "field_count"})
                    continue
                try:
                    frame, track = int(parts[0]), int(parts[1])
                    category = parts[2]
                    x1, y1, x2, y2 = map(float, parts[3:7])
                    involved = int(parts[7])
                    if frame < 1 or track < 0 or involved not in (0, 1) or not (x1 < x2 and y1 < y2):
                        raise ValueError("invalid value")
                except Exception as exc:
                    malformed.append({"file": name, "line": line_number, "reason": repr(exc)})
                    continue
                all_lines += 1
                class_counts[category] += 1
                if involved:
                    involved_lines += 1
                    involved_class_counts[category] += 1
                    tracks[track].append((frame, (x1 + x2) / 2, category, x1, y1, x2, y2))

            summaries = []
            for track_id, observations in tracks.items():
                observations.sort()
                first, last = observations[0], observations[-1]
                summaries.append({
                    "track_id": track_id,
                    "category": Counter(x[2] for x in observations).most_common(1)[0][0],
                    "first_frame": first[0],
                    "last_frame": last[0],
                    "observations": len(observations),
                    "first_center_x": first[1],
                    "last_center_x": last[1],
                    "delta_x": last[1] - first[1],
                })
            # Prefer the longest observed accident-involved track. Short tracks
            # and central first observations abstain rather than invent a side.
            primary = max(summaries, key=lambda x: (x["observations"], -x["first_frame"]), default=None)
            side = ""
            confidence_rule = "ABSTAIN"
            if primary and primary["observations"] >= 10:
                normalized = primary["first_center_x"] / FRAME_WIDTH
                if normalized <= 0.35:
                    side, confidence_rule = "LEFT", "FIRST_CENTER_LE_0.35"
                elif normalized >= 0.65:
                    side, confidence_rule = "RIGHT", "FIRST_CENTER_GE_0.65"
            rows.append({
                "video_id": Path(name).stem,
                "involved_tracks": len(summaries),
                "primary_track_id": "" if primary is None else primary["track_id"],
                "primary_category": "" if primary is None else primary["category"],
                "primary_first_frame": "" if primary is None else primary["first_frame"],
                "primary_last_frame": "" if primary is None else primary["last_frame"],
                "primary_observations": 0 if primary is None else primary["observations"],
                "primary_first_center_x": "" if primary is None else round(primary["first_center_x"], 4),
                "primary_last_center_x": "" if primary is None else round(primary["last_center_x"], 4),
                "primary_delta_x": "" if primary is None else round(primary["delta_x"], 4),
                "entry_side_candidate": side,
                "selection_rule": confidence_rule,
            })

    if malformed:
        raise RuntimeError(f"Malformed rows: {len(malformed)}")
    OUT.mkdir(parents=True)
    with (OUT / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    candidates = [r for r in rows if r["entry_side_candidate"]]
    report = {
        "status": "COMPLETE_VALIDATED",
        "source": "Official DAD annotation.zip linked by the authors' GitHub repository",
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "annotation_files": len(rows),
        "malformed_rows": 0,
        "box_rows": all_lines,
        "accident_involved_box_rows": involved_lines,
        "object_class_counts": dict(sorted(class_counts.items())),
        "involved_object_class_counts": dict(sorted(involved_class_counts.items())),
        "videos_with_involved_tracks": sum(int(r["involved_tracks"] > 0) for r in rows),
        "entry_side_candidates": len(candidates),
        "entry_side_distribution": dict(Counter(r["entry_side_candidate"] for r in candidates)),
        "candidate_policy": "Longest accident-involved track, >=10 observations; first center <=35% LEFT or >=65% RIGHT of official 1280px frame",
        "limitations": [
            "Candidate entry_side is a geometric weak label and is not accepted as competition ground truth.",
            "DAD video access requires a signed academic request agreement; videos are not locally available.",
            "The annotations mark accident-involved objects, not an exact collision frame or evasion-space label.",
        ],
        "stage2_mapping": {
            "entry_side": "geometric candidate available; validate on labeled in-domain videos before training",
            "entry_frame": "first involved-track observation is only a visibility proxy",
            "collision_frame": "accident occurs in final ten frames per official dataset description; no exact frame here",
            "evasion_space": "not annotated",
        },
        "output_sha256": {"inventory.csv": sha256(OUT / "inventory.csv")},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
