"""Convert ZOD Sequences into the Dacon Stage 3 training format.

ZOD camera frames carry UTC timestamps in their filenames.  The vehicle HDF5
files cover a longer source drive, so this converter crops/interpolates the
vehicle signals to every camera timestamp instead of aligning by array index.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


TRAIN_COLUMNS = ["ID", "frame_index", "accel_label", "steer_label"]
DEBUG_COLUMNS = [
    "ID",
    "source_sequence_id",
    "frame_index",
    "sample_index",
    "timestamp_ns",
    "image_path",
    "speed_mps",
    "long_accel_mps2",
    "steering_angle_deg",
    "steering_corrected_deg",
    "acceleration_pedal_ratio",
    "brake_pressed",
    "accel_label",
    "steer_label",
]
_TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)$")


def _timestamp_ns(path: Path) -> int:
    match = _TIMESTAMP_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"cannot parse UTC timestamp from image name: {path.name}")
    return int(np.datetime64(match.group(1), "ns").astype(np.int64))


def _one_dimensional(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).reshape(-1)
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least two values")
    return result


def _validate_signal(name: str, timestamps: np.ndarray, values: np.ndarray) -> None:
    if len(timestamps) != len(values):
        raise ValueError(f"{name}: timestamp/value length mismatch")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name}: timestamps are not strictly increasing")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: values contain NaN or infinity")


def _interpolate(target_ns: np.ndarray, source_ns: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Subtracting a shared origin avoids loss of precision from epoch nanoseconds.
    origin = int(target_ns[0])
    target_seconds = (target_ns - origin).astype(np.float64) / 1e9
    source_seconds = (source_ns - origin).astype(np.float64) / 1e9
    return np.interp(target_seconds, source_seconds, values.astype(np.float64))


def _nearest(target_ns: np.ndarray, source_ns: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(source_ns, target_ns, side="left")
    right = np.clip(positions, 0, len(source_ns) - 1)
    left = np.clip(positions - 1, 0, len(source_ns) - 1)
    choose_left = np.abs(target_ns - source_ns[left]) <= np.abs(source_ns[right] - target_ns)
    indices = np.where(choose_left, left, right)
    return values[indices]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd number")
    if window == 1:
        return values.copy()
    padding = window // 2
    padded = np.pad(values, padding, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _read_signal(h5: h5py.File, path: str, dtype: type = np.float64) -> np.ndarray:
    if path not in h5:
        raise KeyError(f"missing ZOD vehicle signal: {path}")
    return _one_dimensional(path, h5[path][:]).astype(dtype)


def _labels_for_sequence(
    sequence_dir: Path,
    segment_id: str,
    *,
    smoothing_window: int,
    stop_speed: float,
    accel_threshold: float,
    steer_deadzone: float,
    steer_offset: float,
    positive_steer_label: str,
) -> tuple[pd.DataFrame, list[Path], dict]:
    image_dir = sequence_dir / "camera_front_blur"
    images = sorted(image_dir.glob("*.jpg"), key=_timestamp_ns)
    if len(images) < 2:
        raise ValueError(f"expected at least two camera images in {image_dir}")
    target_ns = np.asarray([_timestamp_ns(path) for path in images], dtype=np.int64)
    if np.any(np.diff(target_ns) <= 0):
        raise ValueError(f"camera timestamps are not strictly increasing: {sequence_dir.name}")

    with h5py.File(sequence_dir / "vehicle_data.hdf5", "r") as h5:
        control_ns = _read_signal(
            h5, "ego_vehicle_controls/timestamp/nanoseconds/value", np.int64
        )
        vehicle_ns = _read_signal(
            h5, "ego_vehicle_data/timestamp/nanoseconds/value", np.int64
        )
        steering = np.rad2deg(
            _read_signal(
                h5,
                "ego_vehicle_controls/steering_wheel_angle/angle/radians/value",
            )
        )
        pedal = _read_signal(
            h5, "ego_vehicle_controls/acceleration_pedal/ratio/unitless/value"
        )
        brake = _read_signal(
            h5,
            "ego_vehicle_controls/brake_pedal_pressed/is_brake_pedal_pressed/unitless/value",
            np.uint8,
        )
        speed = _read_signal(
            h5, "ego_vehicle_data/lon_vel_data/velocity/meters_per_second/value"
        )
        acceleration = _read_signal(
            h5,
            "ego_vehicle_data/lon_acc_data/acceleration/meters_per_second2/value",
        )

    for name, timestamps, values in (
        ("steering", control_ns, steering),
        ("acceleration pedal", control_ns, pedal),
        ("brake", control_ns, brake),
        ("speed", vehicle_ns, speed),
        ("longitudinal acceleration", vehicle_ns, acceleration),
    ):
        _validate_signal(name, timestamps, values)
        if target_ns[0] < timestamps[0] or target_ns[-1] > timestamps[-1]:
            raise ValueError(f"{name}: sensor timestamps do not cover all camera frames")

    aligned_speed = _interpolate(target_ns, vehicle_ns, speed)
    aligned_acceleration = _smooth(
        _interpolate(target_ns, vehicle_ns, acceleration), smoothing_window
    )
    aligned_steering = _interpolate(target_ns, control_ns, steering)
    aligned_pedal = _interpolate(target_ns, control_ns, pedal)
    aligned_brake = _nearest(target_ns, control_ns, brake).astype(np.uint8)
    corrected_steering = aligned_steering - steer_offset

    accel_labels = np.where(
        aligned_speed < stop_speed,
        "STOPPED",
        np.where(
            aligned_acceleration > accel_threshold,
            "ACCELERATING",
            np.where(
                aligned_acceleration < -accel_threshold,
                "DECELERATING",
                "CONSTANT",
            ),
        ),
    )
    negative_steer_label = "RIGHT" if positive_steer_label == "LEFT" else "LEFT"
    steer_labels = np.where(
        corrected_steering > steer_deadzone,
        positive_steer_label,
        np.where(
            corrected_steering < -steer_deadzone,
            negative_steer_label,
            "STRAIGHT",
        ),
    )
    indices = np.arange(len(images), dtype=np.int64)
    labels = pd.DataFrame(
        {
            "ID": segment_id,
            "source_sequence_id": sequence_dir.name,
            "frame_index": indices,
            "sample_index": indices,
            "timestamp_ns": target_ns,
            "image_path": [str(path) for path in images],
            "speed_mps": aligned_speed,
            "long_accel_mps2": aligned_acceleration,
            "steering_angle_deg": aligned_steering,
            "steering_corrected_deg": corrected_steering,
            "acceleration_pedal_ratio": aligned_pedal,
            "brake_pressed": aligned_brake,
            "accel_label": accel_labels,
            "steer_label": steer_labels,
        },
        columns=DEBUG_COLUMNS,
    )
    duration = float((target_ns[-1] - target_ns[0]) / 1e9)
    stats = {
        "source_sequence_id": sequence_dir.name,
        "id": segment_id,
        "frames": len(images),
        "duration_seconds": duration,
        "estimated_camera_hz": float((len(images) - 1) / duration),
        "matched_control_samples": int(
            np.count_nonzero((control_ns >= target_ns[0]) & (control_ns <= target_ns[-1]))
        ),
        "matched_vehicle_samples": int(
            np.count_nonzero((vehicle_ns >= target_ns[0]) & (vehicle_ns <= target_ns[-1]))
        ),
        "speed_mps": {"min": float(aligned_speed.min()), "max": float(aligned_speed.max())},
        "long_accel_mps2": {
            "min": float(aligned_acceleration.min()),
            "max": float(aligned_acceleration.max()),
        },
        "steering_angle_deg": {
            "min": float(aligned_steering.min()),
            "max": float(aligned_steering.max()),
        },
        "accel_distribution": dict(Counter(labels["accel_label"])),
        "steer_distribution": dict(Counter(labels["steer_label"])),
    }
    return labels, images, stats


def _write_video(images: list[Path], destination: Path, fps: float, width: int) -> None:
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"cannot read image: {images[0]}")
    source_height, source_width = first.shape[:2]
    output_width = source_width if width <= 0 else width
    output_height = round(source_height * output_width / source_width)
    if output_height % 2:
        output_height += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.temporary.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {temporary}")
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"cannot read image: {image_path}")
            if frame.shape[1] != output_width or frame.shape[0] != output_height:
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        temporary.replace(destination)
        return
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary), "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(destination),
        ],
        check=True,
    )
    temporary.unlink()


def _write_overlay(video_path: Path, labels: pd.DataFrame, destination: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for overlay: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temporary = destination.with_name(f"{destination.stem}.temporary.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    written = 0
    try:
        while written < len(labels):
            ok, frame = capture.read()
            if not ok:
                break
            row = labels.iloc[written]
            dark = frame.copy()
            cv2.rectangle(dark, (15, 15), (800, 220), (0, 0, 0), -1)
            frame = cv2.addWeighted(dark, 0.55, frame, 0.45, 0)
            lines = [
                f"sample: {written:04d}  time: {written / fps:.1f}s",
                f"speed: {row['speed_mps'] * 3.6:.1f} km/h",
                f"accel: {row['long_accel_mps2']:+.2f}  {row['accel_label']}",
                f"steer: {row['steering_angle_deg']:+.2f} deg  {row['steer_label']}",
                f"pedal: {row['acceleration_pedal_ratio']:.3f}  brake: {int(row['brake_pressed'])}",
            ]
            for line_index, text in enumerate(lines):
                cv2.putText(
                    frame, text, (35, 48 + line_index * 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                )
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written != len(labels):
        raise RuntimeError(f"overlay frame mismatch: wrote {written}, labels {len(labels)}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        temporary.replace(destination)
    else:
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary), "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "21",
                "-pix_fmt", "yuv420p", str(destination),
            ],
            check=True,
        )
        temporary.unlink()


def convert(args: argparse.Namespace) -> dict:
    dataset = args.dataset_dir.resolve()
    sequence_root = dataset / "sequences"
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"ZOD sequences directory not found: {sequence_root}")
    available = sorted(path.name for path in sequence_root.iterdir() if path.is_dir())
    requested = args.sequence_ids or available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"unknown sequence IDs: {', '.join(missing)}")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    all_labels: list[pd.DataFrame] = []
    sequence_reports = []
    for sequence_id in requested:
        segment_id = f"{args.id_prefix}{sequence_id}"
        labels, images, stats = _labels_for_sequence(
            sequence_root / sequence_id,
            segment_id,
            smoothing_window=args.smoothing_window,
            stop_speed=args.stop_speed,
            accel_threshold=args.accel_threshold,
            steer_deadzone=args.steer_deadzone,
            steer_offset=args.steer_offset,
            positive_steer_label=args.positive_steer_label,
        )
        video_path = output / "videos" / f"{segment_id}.mp4"
        _write_video(images, video_path, args.target_hz, args.output_width)
        capture = cv2.VideoCapture(str(video_path))
        actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if actual_frames != len(labels):
            raise RuntimeError(
                f"{segment_id}: video has {actual_frames} frames but labels have {len(labels)} rows"
            )
        overlay_path = None
        if args.overlay:
            overlay_path = output / f"{segment_id}_overlay.mp4"
            _write_overlay(video_path, labels, overlay_path)
        stats.update(
            {
                "video": str(video_path),
                "overlay": str(overlay_path) if overlay_path else None,
            }
        )
        sequence_reports.append(stats)
        all_labels.append(labels)

    labels = pd.concat(all_labels, ignore_index=True)
    train_path = output / "labels.csv"
    debug_path = output / "labels_debug.csv"
    labels[TRAIN_COLUMNS].to_csv(train_path, index=False)
    labels.to_csv(debug_path, index=False)
    report = {
        "status": "PASS",
        "source": str(dataset),
        "output": str(output),
        "sequences": len(sequence_reports),
        "samples": len(labels),
        "target_hz": args.target_hz,
        "thresholds": {
            "stop_speed": args.stop_speed,
            "accel_threshold": args.accel_threshold,
            "steer_deadzone": args.steer_deadzone,
            "steer_offset": args.steer_offset,
            "positive_steer_label": args.positive_steer_label,
            "smoothing_window": args.smoothing_window,
        },
        "accel_distribution": dict(Counter(labels["accel_label"])),
        "steer_distribution": dict(Counter(labels["steer_label"])),
        "labels": str(train_path),
        "debug_labels": str(debug_path),
        "sequence_reports": sequence_reports,
    }
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-ids", nargs="*")
    parser.add_argument("--id-prefix", default="ZOD_")
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--stop-speed", type=float, default=0.5)
    parser.add_argument("--accel-threshold", type=float, default=0.2)
    parser.add_argument("--steer-deadzone", type=float, default=3.0)
    parser.add_argument("--steer-offset", type=float, default=0.0)
    parser.add_argument("--positive-steer-label", choices=("LEFT", "RIGHT"), default="LEFT")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    convert(_parser().parse_args())


if __name__ == "__main__":
    main()
