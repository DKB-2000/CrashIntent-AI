"""Resume one A2D2 bus file and audit its Stage 3 label signals.

The worker is deliberately bounded and writes machine-readable progress so a
long public-dataset download does not need an interactive polling loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np


URL = (
    "https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json"
)
EXPECTED_BYTES = 105_582_443
DEFAULT_RAW = Path("data_raw/a2d2/20180810_150607_bus_signals.json")
DEFAULT_OUT = Path("artifacts/stage3-a2d2-scout-20260917")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def state(path: Path, status: str, **extra: object) -> None:
    atomic_json(path, {"status": status, "updated_unix": time.time(), **extra})


def download(raw: Path, status_path: Path, deadline: float) -> None:
    raw.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        received = raw.stat().st_size if raw.exists() else 0
        if received == EXPECTED_BYTES:
            return
        if received > EXPECTED_BYTES:
            # An interrupted external downloader may have appended to the same
            # partial file. Such a stream cannot be repaired safely.
            discarded = received
            raw.unlink()
            received = 0
            state(
                status_path,
                "RESTARTING_CORRUPT_PARTIAL",
                pid=os.getpid(),
                discarded_bytes=discarded,
                expected_bytes=EXPECTED_BYTES,
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("two-hour acquisition deadline")
        request = urllib.request.Request(URL, headers={"Range": f"bytes={received}-"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                code = getattr(response, "status", response.getcode())
                if received and code != 206:
                    raise RuntimeError(f"server ignored Range request at {received}: HTTP {code}")
                mode = "ab" if received else "wb"
                with raw.open(mode) as handle:
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("two-hour acquisition deadline")
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        received += len(block)
                        state(
                            status_path,
                            "DOWNLOADING",
                            pid=os.getpid(),
                            attempt=attempt,
                            received_bytes=received,
                            expected_bytes=EXPECTED_BYTES,
                        )
            if raw.stat().st_size == EXPECTED_BYTES:
                return
        except Exception as exc:
            state(
                status_path,
                "RETRYING",
                pid=os.getpid(),
                attempt=attempt,
                received_bytes=raw.stat().st_size if raw.exists() else 0,
                expected_bytes=EXPECTED_BYTES,
                error=repr(exc),
            )
            if attempt == 3:
                raise
            time.sleep(attempt * 5)
    raise RuntimeError("download retries exhausted")


def numeric_pairs(value: object) -> np.ndarray | None:
    if not isinstance(value, dict) or not isinstance(value.get("values"), list):
        return None
    rows = value["values"]
    if not rows:
        return None
    try:
        array = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    return array[:, :2]


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def audit(raw: Path) -> dict:
    with raw.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("A2D2 bus JSON root is not an object")
    signals = []
    candidates = []
    words = ("steer", "yaw", "angular", "speed", "velocity", "acceleration")
    for name, value in payload.items():
        pairs = numeric_pairs(value)
        item = {
            "name": name,
            "unit": value.get("unit") if isinstance(value, dict) else None,
            "samples": int(len(pairs)) if pairs is not None else 0,
        }
        if pairs is not None:
            timestamps, values = pairs[:, 0], pairs[:, 1]
            if not np.isfinite(pairs).all():
                raise ValueError(f"non-finite values in {name}")
            item.update(
                {
                    "timestamp_min": int(timestamps.min()),
                    "timestamp_max": int(timestamps.max()),
                    "value_quantiles": quantiles(values),
                }
            )
        signals.append(item)
        if any(word in name.lower() for word in words):
            candidates.append(item)
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    return {
        "status": "COMPLETE_VALIDATED",
        "source_url": URL,
        "source_bytes": raw.stat().st_size,
        "source_sha256": sha,
        "signal_count": len(signals),
        "candidate_signals": candidates,
        "all_signals": signals,
        "conclusion": (
            "Usable for Stage 3 sensor-derived labels if steering and speed/yaw "
            "candidates cover the front-camera timestamp range."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    status_path = args.output / "status.json"
    lock_path = args.output / "worker.lock"
    deadline = time.monotonic() + 2 * 60 * 60
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another A2D2 scout owns {lock_path}") from exc
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    try:
        state(
            status_path,
            "STARTING",
            pid=os.getpid(),
            received_bytes=args.raw.stat().st_size if args.raw.exists() else 0,
            expected_bytes=EXPECTED_BYTES,
        )
        download(args.raw, status_path, deadline)
        if args.raw.stat().st_size != EXPECTED_BYTES:
            raise ValueError(f"unexpected size: {args.raw.stat().st_size} != {EXPECTED_BYTES}")
        state(status_path, "ANALYZING", pid=os.getpid(), expected_bytes=EXPECTED_BYTES)
        report = audit(args.raw)
        atomic_json(args.output / "report.json", report)
        state(
            status_path,
            "COMPLETE_VALIDATED",
            pid=os.getpid(),
            received_bytes=EXPECTED_BYTES,
            expected_bytes=EXPECTED_BYTES,
            report=str(args.output / "report.json"),
        )
    except Exception as exc:
        state(
            status_path,
            "FAILED",
            pid=os.getpid(),
            received_bytes=args.raw.stat().st_size if args.raw.exists() else 0,
            expected_bytes=EXPECTED_BYTES,
            error=repr(exc),
        )
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
