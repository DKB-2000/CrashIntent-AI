"""Download and validate the 36 center frames in the A2D2 Stage 3 pilot."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2


PLAN = Path("artifacts/stage3-a2d2-pilot-plan-20260917/report.json")
OUT = Path("artifacts/stage3-a2d2-pilot-acquisition-20260917")
RAW = Path("data_raw/a2d2/pilot-20180810_150607")


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, deadline: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("one-hour pilot acquisition deadline")
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt * 3)


def main() -> None:
    lock = OUT / "worker.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("A2D2 pilot acquisition already running") from exc
    os.write(fd, str(os.getpid()).encode("ascii"))
    os.close(fd)
    status = OUT / "status.json"
    deadline = time.monotonic() + 3600
    try:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        if plan.get("status") != "COMPLETE_VALIDATED":
            raise ValueError("pilot plan is not COMPLETE_VALIDATED")
        rows = plan["selections"]
        reports = []
        for number, row in enumerate(rows, 1):
            name = row["image_url"].rsplit("/", 1)[-1]
            destination = RAW / name
            save(status, {"status":"DOWNLOADING", "pid":os.getpid(), "completed":number-1, "total":len(rows), "current":name})
            if not destination.exists():
                download(row["image_url"], destination, deadline)
            image = cv2.imread(str(destination))
            if image is None or image.ndim != 3 or image.shape[0] < 720 or image.shape[1] < 1000:
                raise ValueError(f"invalid image: {destination}")
            metadata = Path("artifacts/stage3-a2d2-pilot-plan-20260917/metadata") / f"{row['camera_index']:09d}.json"
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            if meta["image_png"] != name:
                raise ValueError(f"metadata filename mismatch: {name}")
            reports.append({
                "label":row["label"], "camera_index":row["camera_index"],
                "timestamp_error_us":row["timestamp_error_us"], "file":str(destination),
                "bytes":destination.stat().st_size, "sha256":sha256(destination),
                "width":int(image.shape[1]), "height":int(image.shape[0]),
            })
        report={
            "status":"ACQUIRED_VALIDATED", "images":len(reports),
            "class_counts":dict(Counter(x["label"] for x in reports)),
            "total_bytes":sum(x["bytes"] for x in reports),
            "max_timestamp_error_us":max(abs(x["timestamp_error_us"]) for x in reports),
            "records":reports,
        }
        save(OUT/"report.json", report)
        save(status, {"status":"ACQUIRED_VALIDATED", "completed":len(reports), "total":len(reports), "report":str(OUT/"report.json")})
    except Exception as exc:
        save(status, {"status":"FAILED", "error":repr(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
