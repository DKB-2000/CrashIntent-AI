"""Select balanced A2D2 steering windows and verify camera timestamp mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np


DEFAULT_BUS = Path("data_raw/a2d2/20180810_150607_bus_signals.json")
DEFAULT_OUT = Path("artifacts/stage3-a2d2-pilot-plan-20260917")
BASE = (
    "https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
    "camera_lidar/20180810_150607/camera/cam_front_center"
)
NAME_PREFIX = "20180810150607_camera_frontcenter_"
ANCHOR_INDEX = 60
ANCHOR_TIMESTAMP_US = 1533906414722153
CAMERA_HZ = 30.0
SENSOR_GRID_PERIOD_US = 100_000


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def pairs(payload: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(payload[name]["values"], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.isfinite(rows).all():
        raise ValueError(f"invalid signal {name}")
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError(f"non-increasing timestamps in {name}")
    return rows[:, 0], rows[:, 1]


def interp(target: np.ndarray, source_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    origin = target[0]
    return np.interp((target-origin)/1e6, (source_t-origin)/1e6, values)


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(np.pad(values, width//2, mode="edge"), np.ones(width)/width, mode="valid")


def choose(candidates: np.ndarray, count: int, min_gap_frames: int) -> list[int]:
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} candidates for requested {count}")
    # Cover the whole drive rather than taking a burst from one intersection.
    desired = np.linspace(candidates[0], candidates[-1], count)
    chosen: list[int] = []
    for point in desired:
        order = candidates[np.argsort(np.abs(candidates - point))]
        match = next((int(i) for i in order if all(abs(i-j) >= min_gap_frames for j in chosen)), None)
        if match is None:
            continue
        chosen.append(match)
    if len(chosen) != count:
        raise ValueError(f"could select only {len(chosen)}/{count} separated candidates")
    return sorted(chosen)


def camera_index(timestamp_us: int) -> int:
    return ANCHOR_INDEX + int(round((timestamp_us - ANCHOR_TIMESTAMP_US) * CAMERA_HZ / 1_000_000))


def fetch_json(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                content = response.read()
            destination.write_bytes(content)
            return json.loads(content)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus", type=Path, default=DEFAULT_BUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--download-metadata", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / "status.json"
    write_json(status, {"status": "ANALYZING", "pid": os.getpid()})
    try:
        payload = json.loads(args.bus.read_text(encoding="utf-8"))
        yaw_t, yaw = pairs(payload, "angular_velocity_omega_z")
        steer_t, steer_mag = pairs(payload, "steering_angle_calculated")
        sign_t, steer_sign = pairs(payload, "steering_angle_calculated_sign")
        speed_t, speed_kph = pairs(payload, "vehicle_speed")
        if not np.array_equal(steer_t, sign_t):
            raise ValueError("steering magnitude/sign timestamps differ")

        start = max(yaw_t[0], steer_t[0], speed_t[0], ANCHOR_TIMESTAMP_US)
        end = min(yaw_t[-1], steer_t[-1], speed_t[-1])
        grid = np.arange(
            math.ceil(start / SENSOR_GRID_PERIOD_US) * SENSOR_GRID_PERIOD_US,
            end,
            SENSOR_GRID_PERIOD_US,
        )
        yaw_g = smooth(interp(grid, yaw_t, yaw), 9)
        mag_g = interp(grid, steer_t, steer_mag)
        sign_g = interp(grid, sign_t, steer_sign) >= 0.5
        speed_g = interp(grid, speed_t, speed_kph)

        signed_a = np.where(sign_g, -mag_g, mag_g)
        signed_b = -signed_a
        moving = speed_g >= 5.0
        corr_a = float(np.corrcoef(signed_a[moving], yaw_g[moving])[0, 1])
        corr_b = float(np.corrcoef(signed_b[moving], yaw_g[moving])[0, 1])
        signed_steer = signed_a if corr_a >= corr_b else signed_b
        steer_mapping = "sign1_negative" if corr_a >= corr_b else "sign1_positive"

        edge = np.arange(len(grid)) >= 20
        edge &= np.arange(len(grid)) < len(grid)-20
        base = moving & edge
        masks = {
            "LEFT": base & (yaw_g >= 5.0) & (signed_steer >= 20.0),
            "RIGHT": base & (yaw_g <= -5.0) & (signed_steer <= -20.0),
            "STRAIGHT": base & (np.abs(yaw_g) <= 0.75) & (np.abs(signed_steer) <= 5.0),
        }
        selections = []
        for label, mask in masks.items():
            selected = choose(np.flatnonzero(mask), args.per_class, min_gap_frames=30)
            for index in selected:
                ts = int(grid[index])
                frame = camera_index(ts)
                stem = f"{NAME_PREFIX}{frame:09d}"
                selections.append({
                    "label": label,
                    "timestamp_us": ts,
                    "camera_index": frame,
                    "yaw_deg_s": float(yaw_g[index]),
                    "signed_steering_deg": float(signed_steer[index]),
                    "speed_kph": float(speed_g[index]),
                    "metadata_url": f"{BASE}/{stem}.json",
                    "image_url": f"{BASE}/{stem}.png",
                })

        errors=[]
        if args.download_metadata:
            for row in selections:
                dest=args.output/"metadata"/f"{row['camera_index']:09d}.json"
                meta=fetch_json(row["metadata_url"], dest)
                observed=int(meta["cam_tstamp"])
                error=observed-row["timestamp_us"]
                row["camera_timestamp_us"]=observed
                row["timestamp_error_us"]=error
                errors.append(error)
        maximum_error=max((abs(x) for x in errors), default=None)
        report={
            "status": "COMPLETE_VALIDATED" if maximum_error is not None and maximum_error <= 60_000 else "PLANNED",
            "source_sha256": hashlib.sha256(args.bus.read_bytes()).hexdigest(),
            "grid_samples": int(len(grid)),
            "duration_seconds": float((grid[-1]-grid[0])/1e6),
            "steering_sign_mapping": steer_mapping,
            "steering_yaw_correlation": max(corr_a, corr_b),
            "candidate_counts": {k:int(v.sum()) for k,v in masks.items()},
            "selected_counts": {k:sum(r["label"]==k for r in selections) for k in masks},
            "max_camera_timestamp_error_us": maximum_error,
            "selections": selections,
        }
        if args.download_metadata and report["status"] != "COMPLETE_VALIDATED":
            raise ValueError(f"camera timestamp mapping error: {maximum_error} us")
        write_json(args.output/"report.json", report)
        write_json(status, {"status":report["status"], "selected":len(selections), "max_camera_timestamp_error_us":maximum_error})
        print(json.dumps({k:v for k,v in report.items() if k != "selections"}, indent=2))
    except Exception as exc:
        write_json(status, {"status":"FAILED", "error":repr(exc)})
        raise


if __name__ == "__main__":
    main()
