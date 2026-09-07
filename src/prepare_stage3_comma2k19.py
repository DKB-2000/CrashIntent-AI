"""Convert comma2k19 segments into the Dacon Stage 3 training format.

The source video is raw HEVC. Its container reports 25 fps, but comma2k19's
``global_pose/frame_times`` and decoded frame count establish a 20 Hz timeline.
This converter therefore treats frame_times as the source of truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TRAIN_COLUMNS = ["ID", "frame_index", "accel_label", "steer_label"]
DEBUG_COLUMNS = [
    "ID",
    "frame_index",
    "sample_index",
    "timestamp",
    "speed_mps",
    "long_accel_mps2",
    "steering_angle_deg",
    "steering_corrected_deg",
    "accel_label",
    "steer_label",
]


def _load(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path))


def _one_dimensional(name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).reshape(-1).astype(np.float64)
    if len(result) < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain at least two finite values")
    return result


def _validate_times(name: str, timestamps: np.ndarray, values: np.ndarray) -> None:
    if len(timestamps) != len(values):
        raise ValueError(f"{name}: timestamp/value length mismatch")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name}: timestamps are not strictly increasing")


def _deduplicate_times(name: str, timestamps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first value for repeated sensor timestamps."""
    timestamps = np.asarray(timestamps)
    values = np.asarray(values)
    if len(timestamps) != len(values):
        raise ValueError(f"{name}: timestamp/value length mismatch")
    keep = np.r_[True, np.diff(timestamps) > 0]
    return timestamps[keep], values[keep]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install ffmpeg on PATH or imageio-ffmpeg") from exc


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open converted video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def _transcode_video(
    source: Path, destination: Path, source_hz: float, target_hz: float
) -> None:
    ffmpeg = _ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{source_hz:g}",
            "-i",
            str(source),
            "-vf",
            f"fps={target_hz:g}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    )


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd number")
    if window == 1:
        return values.copy()
    padding = window // 2
    padded = np.pad(values, padding, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _make_labels(
    segment_id: str,
    target_times: np.ndarray,
    speed_t: np.ndarray,
    speed_values: np.ndarray,
    steer_t: np.ndarray,
    steer_values: np.ndarray,
    *,
    smoothing_window: int,
    stop_speed: float,
    accel_threshold: float,
    steer_deadzone: float,
    steer_offset: float,
    positive_steer_label: str,
) -> pd.DataFrame:
    speed = np.interp(target_times, speed_t, speed_values)
    steer = np.interp(target_times, steer_t, steer_values)
    speed_smooth = _smooth(speed, smoothing_window)
    acceleration = np.gradient(speed_smooth, target_times)
    corrected_steer = steer - steer_offset

    accel_labels = np.where(
        speed < stop_speed,
        "STOPPED",
        np.where(
            acceleration > accel_threshold,
            "ACCELERATING",
            np.where(acceleration < -accel_threshold, "DECELERATING", "CONSTANT"),
        ),
    )
    negative_steer_label = "RIGHT" if positive_steer_label == "LEFT" else "LEFT"
    steer_labels = np.where(
        corrected_steer > steer_deadzone,
        positive_steer_label,
        np.where(corrected_steer < -steer_deadzone, negative_steer_label, "STRAIGHT"),
    )
    indices = np.arange(len(target_times), dtype=np.int64)
    return pd.DataFrame(
        {
            "ID": segment_id,
            "frame_index": indices,
            "sample_index": indices,
            "timestamp": target_times,
            "speed_mps": speed,
            "long_accel_mps2": acceleration,
            "steering_angle_deg": steer,
            "steering_corrected_deg": corrected_steer,
            "accel_label": accel_labels,
            "steer_label": steer_labels,
        },
        columns=DEBUG_COLUMNS,
    )


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
    colors = {
        "LEFT": (0, 255, 255),
        "RIGHT": (255, 180, 0),
        "STRAIGHT": (0, 255, 0),
        "ACCELERATING": (0, 255, 0),
        "DECELERATING": (0, 0, 255),
        "CONSTANT": (255, 255, 255),
        "STOPPED": (160, 160, 160),
    }
    written = 0
    while written < len(labels):
        ok, frame = capture.read()
        if not ok:
            break
        row = labels.iloc[written]
        accel_label = str(row["accel_label"])
        steer_label = str(row["steer_label"])
        dark = frame.copy()
        cv2.rectangle(dark, (15, 15), (720, 210), (0, 0, 0), -1)
        frame = cv2.addWeighted(dark, 0.55, frame, 0.45, 0)
        lines = [
            (f"sample: {written:04d}  time: {written / fps:.1f}s", (255, 255, 255)),
            (f"speed: {row['speed_mps'] * 3.6:.1f} km/h", (255, 255, 255)),
            (f"accel: {row['long_accel_mps2']:+.2f}  {accel_label}", colors[accel_label]),
            (f"steer: {row['steering_angle_deg']:+.2f} deg  {steer_label}", colors[steer_label]),
        ]
        for line_index, (text, color) in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (35, 50 + line_index * 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        writer.write(frame)
        written += 1
    capture.release()
    writer.release()
    if written != len(labels):
        raise RuntimeError(f"overlay frame mismatch: wrote {written}, labels {len(labels)}")
    ffmpeg = _ffmpeg()
    _run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    )
    temporary.unlink()


def convert(args: argparse.Namespace) -> dict:
    source = args.segment_dir.resolve()
    output = args.output_dir.resolve()
    segment_id = args.id or f"COMMA2K19_{source.name}"
    ratio = args.source_hz / args.target_hz
    stride = round(ratio)
    if stride < 1 or not np.isclose(ratio, stride):
        raise ValueError("source-hz must be an integer multiple of target-hz")

    frame_times = _one_dimensional("frame_times", _load(source / "global_pose/frame_times"))
    speed_t = _one_dimensional("speed_t", _load(source / "processed_log/CAN/speed/t"))
    speed = _one_dimensional("speed", _load(source / "processed_log/CAN/speed/value"))
    speed_t, speed = _deduplicate_times("speed", speed_t, speed)
    steer_t = _one_dimensional(
        "steer_t", _load(source / "processed_log/CAN/steering_angle/t")
    )
    steer = _one_dimensional(
        "steer", _load(source / "processed_log/CAN/steering_angle/value")
    )
    steer_t, steer = _deduplicate_times("steering", steer_t, steer)
    _validate_times("speed", speed_t, speed)
    _validate_times("steering", steer_t, steer)
    if np.any(np.diff(frame_times) <= 0):
        raise ValueError("frame_times are not strictly increasing")
    overlap_start = max(frame_times[0], speed_t[0], steer_t[0])
    overlap_end = min(frame_times[-1], speed_t[-1], steer_t[-1])
    if overlap_end <= overlap_start:
        raise ValueError("video and CAN timestamps do not overlap")

    target_times = frame_times[::stride]
    labels = _make_labels(
        segment_id,
        target_times,
        speed_t,
        speed,
        steer_t,
        steer,
        smoothing_window=args.smoothing_window,
        stop_speed=args.stop_speed,
        accel_threshold=args.accel_threshold,
        steer_deadzone=args.steer_deadzone,
        steer_offset=args.steer_offset,
        positive_steer_label=args.positive_steer_label,
    )
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "videos" / f"{segment_id}.mp4"
    _transcode_video(source / "video.hevc", video_path, args.source_hz, args.target_hz)
    actual_frames = _video_frame_count(video_path)
    if actual_frames != len(labels):
        raise RuntimeError(
            f"converted video has {actual_frames} frames but labels have {len(labels)} rows"
        )

    train_path = output / "labels.csv"
    debug_path = output / "labels_debug.csv"
    labels[TRAIN_COLUMNS].to_csv(train_path, index=False)
    labels.to_csv(debug_path, index=False)
    overlay_path = None
    if args.overlay:
        overlay_path = output / f"{segment_id}_overlay.mp4"
        _write_overlay(video_path, labels, overlay_path)

    report = {
        "status": "PASS",
        "source": str(source),
        "id": segment_id,
        "source_frames": int(len(frame_times)),
        "output_frames": int(actual_frames),
        "source_duration_seconds": float(frame_times[-1] - frame_times[0]),
        "source_hz": args.source_hz,
        "target_hz": args.target_hz,
        "timestamp_edge_offsets_seconds": {
            "speed_start": float(speed_t[0] - frame_times[0]),
            "speed_end": float(speed_t[-1] - frame_times[-1]),
            "steer_start": float(steer_t[0] - frame_times[0]),
            "steer_end": float(steer_t[-1] - frame_times[-1]),
        },
        "speed_mps": {"min": float(speed.min()), "max": float(speed.max())},
        "steering_angle_deg": {"min": float(steer.min()), "max": float(steer.max())},
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
        "video": str(video_path),
        "labels": str(train_path),
        "debug_labels": str(debug_path),
        "overlay": str(overlay_path) if overlay_path else None,
    }
    report_path = output / "conversion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id")
    parser.add_argument("--source-hz", type=float, default=20.0)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--stop-speed", type=float, default=0.5)
    parser.add_argument("--accel-threshold", type=float, default=0.2)
    parser.add_argument("--steer-deadzone", type=float, default=0.5)
    parser.add_argument("--steer-offset", type=float, default=0.0)
    parser.add_argument("--positive-steer-label", choices=("LEFT", "RIGHT"), default="LEFT")
    parser.add_argument("--overlay", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    convert(args)


if __name__ == "__main__":
    main()
