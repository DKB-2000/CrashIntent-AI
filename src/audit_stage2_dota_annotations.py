"""Audit local official DoTA annotations for Stage2 supervision value."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_raw/dota/annotations/annotations"
OUT = ROOT / "artifacts/stage2-dota-annotation-audit-20260918"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    files = sorted(SOURCE.glob("*.json"))
    if len(files) != 4677:
        raise RuntimeError(f"Expected 4677 annotations, found {len(files)}")

    accident_names = Counter()
    ego = Counter()
    split = Counter()
    object_keys = Counter()
    object_frames = object_count = 0
    anomaly_lengths = []
    declared_frame_count_mismatches = []
    rows = []
    malformed = []
    for path in files:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            required = {"video_name", "num_frames", "ego_involve", "anomaly_start",
                        "anomaly_end", "accident_id", "accident_name", "labels"}
            missing = sorted(required - item.keys())
            if missing:
                raise ValueError(f"missing {missing}")
            labels = item["labels"]
            if len(labels) != int(item["num_frames"]):
                declared_frame_count_mismatches.append({
                    "file": path.name,
                    "declared": int(item["num_frames"]),
                    "label_rows": len(labels),
                })
            frame_ids = [int(x["frame_id"]) for x in labels]
            if frame_ids != list(range(len(labels))):
                raise ValueError("non-contiguous frame IDs")
            start, end = int(item["anomaly_start"]), int(item["anomaly_end"])
            # Official DoTA intervals are half-open [start, end); end may equal
            # the clip length. Per-frame rows are authoritative for one known
            # num_frames mismatch in the released archive.
            if not (0 <= start < end <= len(labels)):
                raise ValueError("invalid anomaly interval")
            objects = [o for label in labels for o in label.get("objects", [])]
            frames_with_objects = sum(bool(label.get("objects")) for label in labels)
            object_frames += frames_with_objects
            object_count += len(objects)
            for obj in objects:
                object_keys.update(obj.keys())
            name = str(item["accident_name"])
            accident_names[name] += 1
            ego[str(bool(item["ego_involve"]))] += 1
            subset = "train" if "train" in str(path).lower() else "annotation_pool"
            split[subset] += 1
            anomaly_lengths.append(end - start + 1)
            rows.append({
                "video_name": item["video_name"],
                "accident_id": item["accident_id"],
                "accident_name": name,
                "ego_involve": bool(item["ego_involve"]),
                "night": bool(item.get("night", False)),
                "num_frames": len(labels),
                "anomaly_start": start,
                "anomaly_end": end,
                "anomaly_length": end - start + 1,
                "frames_with_spatial_objects": frames_with_objects,
                "spatial_object_count": len(objects),
            })
        except Exception as exc:
            malformed.append({"file": path.name, "error": repr(exc)})

    OUT.mkdir(parents=True)
    columns = list(rows[0])
    with (OUT / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    directional_tokens = ("left", "right", "lateral", "turn", "cross", "oncoming")
    directional = {k: v for k, v in accident_names.items()
                   if any(token in k.lower() for token in directional_tokens)}
    report = {
        "status": "COMPLETE_VALIDATED" if not malformed else "FAILED",
        "source": "Official DoTA annotation archive already present locally",
        "annotation_files": len(files),
        "valid_annotations": len(rows),
        "malformed": malformed,
        "declared_frame_count_mismatches": declared_frame_count_mismatches,
        "ego_involve": dict(ego),
        "accident_class_counts": dict(sorted(accident_names.items())),
        "directional_candidate_class_counts": dict(sorted(directional.items())),
        "directional_candidate_videos": sum(directional.values()),
        "spatial": {
            "frames_with_objects": object_frames,
            "object_annotations": object_count,
            "object_schema_key_frequency": dict(object_keys),
            "videos_with_objects": sum(int(r["spatial_object_count"] > 0) for r in rows),
        },
        "anomaly_length_frames": {
            "min": min(anomaly_lengths),
            "max": max(anomaly_lengths),
            "mean": sum(anomaly_lengths) / len(anomaly_lengths),
        },
        "stage2_mapping": {
            "entry_frame_proxy": "anomaly_start",
            "collision_frame": "not explicitly separate; anomaly interval/category only",
            "entry_side": "candidate from accident class plus spatial box trajectory; requires validation",
            "evasion_space": "not directly annotated; must remain weak/derived",
        },
        "input": {
            "archive": str((ROOT / "data_raw/dota/DoTA_annotations.zip").relative_to(ROOT)),
            "archive_sha256": sha256(ROOT / "data_raw/dota/DoTA_annotations.zip"),
        },
        "output_sha256": {"inventory.csv": sha256(OUT / "inventory.csv")},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if malformed:
        raise RuntimeError(f"Malformed annotations: {len(malformed)}")


if __name__ == "__main__":
    main()
