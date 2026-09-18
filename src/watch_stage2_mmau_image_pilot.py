"""Resumable acquisition and validation of one official MM-AU image shard."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_raw/mm-au-detection-images-pilot-20260918"
ART = ROOT / "artifacts/stage2-mmau-detection-images-pilot-20260918"
TARGET = OUT / "train_part_aw.tar.gz"
PART = OUT / "train_part_aw.tar.gz.part"
EXPECTED_SIZE = 1_972_510_663
EXPECTED_SHA256 = "7f4e4390da3b8d8f71a6b5d8be246dac56c5ac32ee6c586fa748b9d84c48f1a1"
REMOTE_PATH = "MMAU_Det_paper/images/train_chunks/train_part_aw"
URL = "https://huggingface.co/datasets/JeffreyChou/MM-AU/resolve/main/" + REMOTE_PATH + "?download=true"
SELECTION = ROOT / "artifacts/stage2-mmau-scene-selection-20260918/selected_2500.csv"
DEADLINE_SECONDS = 3 * 60 * 60
FRAME_PATTERN = re.compile(r"(?:^|/)(\d+)_(\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.I)


def save(status: str, **extra) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_unix": time.time(), "remote_path": REMOTE_PATH,
               "target": str(TARGET.relative_to(ROOT)), "expected_bytes": EXPECTED_SIZE, **extra}
    temp = ART / "status.json.tmp"
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            temp.replace(ART / "status.json")
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(deadline: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size == EXPECTED_SIZE:
        return
    if TARGET.exists():
        TARGET.replace(PART)
    for attempt in range(1, 4):
        if time.monotonic() >= deadline:
            raise TimeoutError("three-hour pilot deadline")
        current = PART.stat().st_size if PART.exists() else 0
        if current > EXPECTED_SIZE:
            PART.unlink()
            current = 0
        headers = {"User-Agent": "crashvideo-mmau-pilot/1.0"}
        if current:
            headers["Range"] = f"bytes={current}-"
        save("DOWNLOADING", attempt=attempt, bytes=current)
        try:
            request = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if current and response.status != 206:
                    PART.unlink(missing_ok=True)
                    raise RuntimeError(f"resume rejected with HTTP {response.status}")
                with PART.open("ab" if current else "wb") as output:
                    last_status = time.monotonic()
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("three-hour pilot deadline")
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        if time.monotonic() - last_status >= 30:
                            output.flush()
                            save("DOWNLOADING", attempt=attempt, bytes=output.tell())
                            last_status = time.monotonic()
            if PART.stat().st_size != EXPECTED_SIZE:
                raise RuntimeError(f"size {PART.stat().st_size} != {EXPECTED_SIZE}")
            PART.replace(TARGET)
            return
        except Exception as exc:
            save("RETRYING" if attempt < 3 else "FAILED", attempt=attempt,
                 bytes=PART.stat().st_size if PART.exists() else 0, error=repr(exc))
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def validate() -> dict:
    save("VERIFYING", bytes=TARGET.stat().st_size)
    actual_sha = digest(TARGET)
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(f"SHA256 mismatch: {actual_sha}")
    with SELECTION.open(encoding="utf-8", newline="") as stream:
        selected = {row["video_key"] for row in csv.DictReader(stream)}
    members = files = image_files = unsafe = 0
    top = Counter()
    samples = []
    candidate_videos = set()
    candidate_images = 0
    with tarfile.open(TARGET, "r:gz") as archive:
        for member in archive:
            members += 1
            parts = Path(member.name).parts
            if Path(member.name).is_absolute() or ".." in parts:
                unsafe += 1
            if not member.isfile():
                continue
            files += 1
            if len(samples) < 30:
                samples.append(member.name)
            if parts:
                top[parts[0]] += 1
            match = FRAME_PATTERN.search(member.name)
            if match:
                image_files += 1
                key = f"{int(match.group(1))}_{int(match.group(2))}"
                if key in selected:
                    candidate_videos.add(key)
                    candidate_images += 1
    if unsafe or not image_files:
        raise RuntimeError(f"unsafe={unsafe}, image_files={image_files}")
    return {"archive_sha256": actual_sha, "members": members, "files": files,
            "image_files": image_files, "top_level_file_counts": dict(top),
            "sample_members": samples, "selected_candidate_videos_present": len(candidate_videos),
            "selected_candidate_images_present": candidate_images,
            "selected_candidate_video_keys_first_100": sorted(candidate_videos)[:100]}


def main() -> None:
    started = time.monotonic()
    save("STARTING", bytes=PART.stat().st_size if PART.exists() else 0)
    try:
        download(started + DEADLINE_SECONDS)
        result = validate()
        result.update({"status": "ACQUIRED_VALIDATED", "elapsed_seconds": time.monotonic() - started,
                       "archive_bytes": TARGET.stat().st_size})
        ART.mkdir(parents=True, exist_ok=True)
        (ART / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        save("ACQUIRED_VALIDATED", bytes=TARGET.stat().st_size,
             selected_candidate_videos_present=result["selected_candidate_videos_present"],
             report=str((ART / "report.json").relative_to(ROOT)))
    except Exception as exc:
        save("FAILED", bytes=PART.stat().st_size if PART.exists() else 0, error=repr(exc))
        raise


if __name__ == "__main__":
    main()
