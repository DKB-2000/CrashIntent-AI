"""Recover a provenance-labelled Stage1 pilot from a partially downloaded DLC-2021 TAR.

The archive is uncompressed, so every complete member before the interrupted byte is
usable.  This script deliberately keeps only ``or`` (original) and ``re`` (screen
recapture) clips and never infers labels from pixels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

import cv2


SOURCE = Path("data_raw/dlc-2021/clips.tar")
OUT = Path("artifacts/stage1-dlc-partial-pilot-20260917")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    frames_root = OUT / "frames"
    videos_root = OUT / "videos"
    frames_root.mkdir(exist_ok=True)
    videos_root.mkdir(exist_ok=True)

    recovered: dict[str, list[Path]] = {}
    stopped = None
    with tarfile.open(SOURCE, "r:") as archive:
        while True:
            try:
                member = archive.next()
            except tarfile.ReadError as exc:
                stopped = str(exc)
                break
            if member is None:
                break
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) != 5 or parts[:3] != ("clips", "images", "alb_id")
                    or not parts[4].lower().endswith(".jpg")):
                continue
            clip = parts[3]
            kind = clip.split(".", 1)[1][:2] if "." in clip else ""
            if kind not in {"or", "re"}:
                continue
            src = archive.extractfile(member)
            if src is None:
                raise RuntimeError(f"Cannot read {member.name}")
            target = frames_root / clip / parts[4]
            target.parent.mkdir(exist_ok=True)
            try:
                payload = src.read()
            except tarfile.ReadError as exc:
                stopped = f"partial member: {member.name}: {exc}"
                break
            target.write_bytes(payload)
            if target.stat().st_size != member.size:
                target.unlink(missing_ok=True)
                stopped = f"partial member: {member.name}"
                break
            recovered.setdefault(clip, []).append(target)

    rows = []
    for clip, paths in sorted(recovered.items()):
        paths.sort(key=lambda p: p.name)
        if len(paths) < 16:
            continue
        first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"Unreadable image: {paths[0]}")
        height, width = first.shape[:2]
        video = videos_root / f"{clip}.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {video}")
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                writer.release()
                raise ValueError(f"Invalid frame: {path}")
            writer.write(image)
        writer.release()
        capture = cv2.VideoCapture(str(video))
        decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if decoded != len(paths):
            raise ValueError(f"Frame count mismatch for {video}: {decoded} != {len(paths)}")
        kind = clip.split(".", 1)[1][:2]
        rows.append({
            "source_id": f"dlc2021/alb_id/{clip}",
            "clip": clip,
            "label": "ORIGINAL" if kind == "or" else "RERECORDED",
            "label_source": "DLC-2021 filename type code",
            "frames": len(paths),
            "video_path": video.as_posix(),
            "video_sha256": sha256(video),
        })
    if not rows or {r["label"] for r in rows} != {"ORIGINAL", "RERECORDED"}:
        raise ValueError("Pilot must contain both provenance classes")
    manifest = OUT / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    state = {
        "status": "DATA_VALIDATED",
        "source_tar": SOURCE.as_posix(),
        "source_tar_partial_bytes": SOURCE.stat().st_size,
        "source_tar_partial_sha256": sha256(SOURCE),
        "archive_stop": stopped,
        "clips": len(rows),
        "class_counts": {label: sum(r["label"] == label for r in rows) for label in ("ORIGINAL", "RERECORDED")},
        "frames": sum(int(r["frames"]) for r in rows),
        "manifest_sha256": sha256(manifest),
        "scope": "INITIAL_PILOT_SINGLE_DOCUMENT_TEMPLATE",
        "limitations": ["Only the complete prefix of an interrupted TAR is used.",
                        "Clips share one document type/template and are correlated."],
    }
    write_json(OUT / "status.json", state)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
