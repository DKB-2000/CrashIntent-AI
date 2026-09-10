"""Build leakage-safe Stage 1 ORIGINAL/RERECORDED training pairs from CCD.

All derivatives of one source video stay in the same split. Both classes pass
through the same raw-frame-to-H.264 encoder, preventing container/codec choice
from becoming a trivial class shortcut. Synthetic parameters are recorded in a
manifest so every sample can be reproduced and audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


LABEL_COLUMNS = ["path", "label"]


def _load_excluded_source_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"exclude file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, dtype=str)
        if "source_id" not in table.columns:
            raise ValueError(f"exclude CSV must contain a source_id column: {path}")
        values = table["source_id"].dropna().astype(str)
    else:
        values = pd.Series(
            [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        )
    source_ids = {value.strip() for value in values if value.strip()}
    if not source_ids:
        raise ValueError(f"exclude file contains no source IDs: {path}")
    return source_ids


@dataclass(frozen=True)
class Effects:
    perspective_x: float
    perspective_y: float
    scale: float
    blur_sigma: float
    gamma: float
    red_gain: float
    blue_gain: float
    flicker_amplitude: float
    flicker_hz: float
    band_amplitude: float
    band_width: float
    band_speed: float
    moire_amplitude: float
    moire_period_x: float
    moire_period_y: float
    moire_phase_speed: float
    reflection_amplitude: float
    reflection_x: float
    reflection_y: float
    noise_sigma: float
    jpeg_quality: int
    temporal_jitter_probability: float


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _split(source_id: str, seed: int, validation_fraction: float) -> str:
    return "val" if _stable_fraction(source_id, seed) < validation_fraction else "train"


def _effect_parameters(rng: random.Random) -> Effects:
    return Effects(
        perspective_x=rng.uniform(-0.035, 0.035),
        perspective_y=rng.uniform(-0.025, 0.025),
        scale=rng.uniform(0.94, 1.02),
        blur_sigma=rng.uniform(0.15, 1.25),
        gamma=rng.uniform(0.82, 1.20),
        red_gain=rng.uniform(0.92, 1.10),
        blue_gain=rng.uniform(0.90, 1.12),
        flicker_amplitude=rng.uniform(0.005, 0.055),
        flicker_hz=rng.uniform(3.0, 13.0),
        band_amplitude=rng.uniform(0.015, 0.10),
        band_width=rng.uniform(0.10, 0.32),
        band_speed=rng.uniform(0.25, 1.4),
        moire_amplitude=rng.uniform(0.008, 0.055),
        moire_period_x=rng.uniform(2.8, 8.0),
        moire_period_y=rng.uniform(3.0, 10.0),
        moire_phase_speed=rng.uniform(-2.5, 2.5),
        reflection_amplitude=rng.uniform(0.0, 0.09),
        reflection_x=rng.uniform(0.15, 0.85),
        reflection_y=rng.uniform(0.05, 0.65),
        noise_sigma=rng.uniform(0.5, 4.0),
        jpeg_quality=rng.randint(68, 96),
        temporal_jitter_probability=rng.uniform(0.0, 0.045),
    )


def _open_video(path: Path) -> tuple[cv2.VideoCapture, int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"invalid video metadata: {path}")
    return capture, width, height, fps, frames


class H264Writer:
    def __init__(self, destination: Path, width: int, height: int, fps: float, crf: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError as exc:
                raise RuntimeError("Install ffmpeg on PATH or imageio-ffmpeg") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(destination),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is closed")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def _perspective_matrix(width: int, height: int, effects: Effects) -> np.ndarray:
    inset_x = abs(effects.perspective_x) * width
    inset_y = abs(effects.perspective_y) * height
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    if effects.perspective_x >= 0:
        left_top, left_bottom, right_top, right_bottom = inset_y, inset_y, 0.0, 0.0
    else:
        left_top, left_bottom, right_top, right_bottom = 0.0, 0.0, inset_y, inset_y
    target = np.float32(
        [
            [inset_x, left_top],
            [width - 1 - inset_x, right_top],
            [width - 1, height - 1 - right_bottom],
            [0, height - 1 - left_bottom],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    if not math.isclose(effects.scale, 1.0):
        scale_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, effects.scale)
        scale_h = np.vstack([scale_matrix, [0, 0, 1]]).astype(np.float32)
        matrix = scale_h @ matrix
    return matrix


def _spatial_fields(width: int, height: int, effects: Effects) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    moire = np.sin(2 * np.pi * xx / effects.moire_period_x)
    moire *= np.sin(2 * np.pi * yy / effects.moire_period_y)
    reflection_x = effects.reflection_x * width
    reflection_y = effects.reflection_y * height
    radius = max(width, height) * 0.55
    reflection = np.exp(-((xx - reflection_x) ** 2 + (yy - reflection_y) ** 2) / (2 * radius**2))
    return moire.astype(np.float32), reflection.astype(np.float32)


def _synthetic_frame(
    frame: np.ndarray,
    index: int,
    fps: float,
    effects: Effects,
    perspective: np.ndarray,
    base_moire: np.ndarray,
    reflection: np.ndarray,
    noise_rng: np.random.Generator,
) -> np.ndarray:
    height, width = frame.shape[:2]
    result = cv2.warpPerspective(
        frame, perspective, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    ).astype(np.float32) / 255.0
    time_s = index / fps
    flicker = 1.0 + effects.flicker_amplitude * math.sin(2 * math.pi * effects.flicker_hz * time_s)
    result *= flicker

    phase = effects.moire_phase_speed * index
    moving_moire = np.roll(base_moire, int(round(phase)), axis=1)
    result *= 1.0 + effects.moire_amplitude * moving_moire[:, :, None]

    band_center = ((time_s * effects.band_speed) % 1.4 - 0.2) * height
    rows = np.arange(height, dtype=np.float32)
    band = np.exp(-0.5 * ((rows - band_center) / (effects.band_width * height)) ** 2)
    result *= 1.0 - effects.band_amplitude * band[:, None, None]
    result += effects.reflection_amplitude * reflection[:, :, None]
    result[:, :, 2] *= effects.red_gain
    result[:, :, 0] *= effects.blue_gain
    result = np.clip(result, 0.0, 1.0) ** (1.0 / effects.gamma)
    if effects.blur_sigma > 0.2:
        result = cv2.GaussianBlur(result, (0, 0), effects.blur_sigma)
    result += noise_rng.normal(0.0, effects.noise_sigma / 255.0, result.shape).astype(np.float32)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, effects.jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG artifact simulation failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG artifact simulation decode failed")
    return decoded


def _render(
    source: Path,
    destination: Path,
    crf: int,
    effects: Effects | None,
    seed: int,
) -> dict:
    capture, width, height, fps, expected_frames = _open_video(source)
    writer = H264Writer(destination, width, height, fps, crf)
    perspective = _perspective_matrix(width, height, effects) if effects else None
    fields = _spatial_fields(width, height, effects) if effects else None
    noise_rng = np.random.default_rng(seed)
    jitter_rng = random.Random(seed)
    previous = None
    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if effects is not None:
                frame = _synthetic_frame(
                    frame,
                    written,
                    fps,
                    effects,
                    perspective,
                    fields[0],
                    fields[1],
                    noise_rng,
                )
                if previous is not None and jitter_rng.random() < effects.temporal_jitter_probability:
                    frame = previous.copy()
            writer.write(frame)
            previous = frame
            written += 1
    finally:
        capture.release()
        writer.close()
    if written == 0 or (expected_frames > 0 and written != expected_frames):
        raise RuntimeError(f"frame mismatch for {source}: expected {expected_frames}, wrote {written}")
    return {"width": width, "height": height, "fps": fps, "frames": written, "crf": crf}


def generate(args: argparse.Namespace) -> dict:
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between 0 and 1")
    if args.variants < 1:
        raise ValueError("variants must be at least 1")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    videos = sorted(path for path in input_dir.glob(args.pattern) if path.is_file())
    source_ids = [path.stem for path in videos]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source IDs detected; video filename stems must be unique")
    excluded_source_ids: set[str] = set()
    exclusion_file = None
    if args.exclude_file is not None:
        exclusion_file = args.exclude_file.resolve()
        requested_exclusions = _load_excluded_source_ids(exclusion_file)
        missing = sorted(requested_exclusions.difference(source_ids))
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"exclude source IDs not found in input ({len(missing)}): {preview}")
        excluded_source_ids = requested_exclusions
        videos = [path for path in videos if path.stem not in excluded_source_ids]
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise ValueError(f"no videos matched {args.pattern!r} in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    manifest: list[dict] = []
    for source_index, source in enumerate(videos):
        source_id = source.stem
        split = _split(source_id, args.seed, args.validation_fraction)
        original_rel = Path("original") / f"{source_id}.mp4"
        original_path = output_dir / original_rel
        if original_path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace: {original_path}")
        crf_rng = random.Random(args.seed + source_index * 10007)
        original_crf = crf_rng.randint(args.crf_min, args.crf_max)
        metadata = _render(source, original_path, original_crf, None, args.seed + source_index)
        rows.append({"path": original_rel.as_posix(), "label": "ORIGINAL", "split": split})
        manifest.append(
            {
                "source_id": source_id,
                "source_path": str(source),
                "output_path": original_rel.as_posix(),
                "label": "ORIGINAL",
                "split": split,
                "variant": 0,
                "effects": None,
                **metadata,
            }
        )
        for variant in range(1, args.variants + 1):
            variant_seed = args.seed + source_index * 10007 + variant * 997
            variant_rng = random.Random(variant_seed)
            effects = _effect_parameters(variant_rng)
            rerecorded_rel = Path("rerecorded") / f"{source_id}_v{variant:02d}.mp4"
            rerecorded_path = output_dir / rerecorded_rel
            if rerecorded_path.exists() and not args.overwrite:
                raise FileExistsError(f"output exists; pass --overwrite to replace: {rerecorded_path}")
            # CRF ranges overlap exactly between classes.
            metadata = _render(
                source,
                rerecorded_path,
                variant_rng.randint(args.crf_min, args.crf_max),
                effects,
                variant_seed,
            )
            rows.append({"path": rerecorded_rel.as_posix(), "label": "RERECORDED", "split": split})
            manifest.append(
                {
                    "source_id": source_id,
                    "source_path": str(source),
                    "output_path": rerecorded_rel.as_posix(),
                    "label": "RERECORDED",
                    "split": split,
                    "variant": variant,
                    "effects": json.dumps(asdict(effects), sort_keys=True),
                    **metadata,
                }
            )

    labels = pd.DataFrame(rows)
    labels[LABEL_COLUMNS].to_csv(output_dir / "labels.csv", index=False)
    for split in ("train", "val"):
        labels.loc[labels["split"] == split, LABEL_COLUMNS].to_csv(
            output_dir / f"labels_{split}.csv", index=False
        )
    pd.DataFrame(manifest).to_csv(output_dir / "generation_manifest.csv", index=False)
    source_splits = pd.DataFrame(
        sorted({(row["source_id"], row["split"]) for row in manifest}),
        columns=["source_id", "split"],
    )
    source_splits.to_csv(output_dir / "source_splits.csv", index=False)
    leakage = source_splits.groupby("source_id")["split"].nunique().max()
    if int(leakage) != 1:
        raise RuntimeError("source leakage detected across train/validation")

    report = {
        "status": "PASS",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "exclusion_file": str(exclusion_file) if exclusion_file else None,
        "excluded_sources": len(excluded_source_ids),
        "sources": len(videos),
        "variants_per_source": args.variants,
        "samples": len(labels),
        "labels": labels["label"].value_counts().to_dict(),
        "splits_by_sample": labels["split"].value_counts().to_dict(),
        "splits_by_source": source_splits["split"].value_counts().to_dict(),
        "source_leakage": False,
        "seed": args.seed,
    }
    (output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument(
        "--exclude-file",
        type=Path,
        help="CSV with a source_id column, or a text file with one source ID per line",
    )
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--crf-min", type=int, default=20)
    parser.add_argument("--crf-max", type=int, default=28)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 0 <= args.crf_min <= args.crf_max <= 51:
        raise ValueError("CRF range must satisfy 0 <= min <= max <= 51")
    generate(args)


if __name__ == "__main__":
    main()
