"""Train and smoke-test the official-compatible Stage 1 MViTv2-S model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s


SEED = 20260903
LABEL_TO_INDEX = {"ORIGINAL": 0, "RERECORDED": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
OUTPUT_COLUMNS = ["ID", "answer"]


class Stage1MViT(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
        self.net = mvit_v2_s(weights=weights)
        self.net.head[1] = nn.Linear(self.net.head[1].in_features, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def _read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"cannot decode video: {path}")
    return frames


def _crop_tensor(rgb: np.ndarray, size: int) -> torch.Tensor:
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    resized_h = max(size, round(height * scale))
    resized_w = max(size, round(width * scale))
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    y = (resized_h - size) // 2
    x = (resized_w - size) // 2
    cropped = resized[y : y + size, x : x + size].copy()
    return torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0


def _clip(frames: list[np.ndarray], indices: np.ndarray, size: int) -> torch.Tensor:
    value = torch.stack([_crop_tensor(frames[int(index)], size) for index in indices], dim=1)
    mean = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
    std = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
    return (value - mean) / std


def _uniform_indices(total: int, count: int) -> np.ndarray:
    return np.linspace(0, total - 1, count).round().astype(np.int64)


def _slot_indices(total: int, count: int, slots: int) -> list[np.ndarray]:
    boundaries = np.linspace(0, total, slots + 1).round().astype(np.int64)
    result = []
    for index in range(slots):
        start = min(boundaries[index], total - 1)
        end = max(start, min(boundaries[index + 1] - 1, total - 1))
        result.append(np.linspace(start, end, count).round().astype(np.int64))
    return result


def _load_labels(data_dir: Path, labels_path: Path, limit: int | None) -> list[dict]:
    frame = pd.read_csv(labels_path)
    missing = sorted(set(["path", "label"]).difference(frame.columns))
    if missing:
        raise ValueError(f"missing label columns: {missing}")
    if limit is not None:
        # Preserve both classes instead of taking a potentially one-class prefix.
        selected = []
        for label in LABEL_TO_INDEX:
            selected.append(frame.loc[frame["label"] == label].head(limit))
        frame = pd.concat(selected, ignore_index=True)
    rows = []
    for row in frame.to_dict("records"):
        label = str(row["label"]).upper()
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"invalid Stage1 label: {label}")
        video_path = data_dir / str(row["path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        rows.append({"path": video_path, "relative_path": str(row["path"]), "label": label})
    if not rows:
        raise ValueError("labels file contains no usable rows")
    return rows


def _forward(model: nn.Module, clip: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(clip.unsqueeze(0).to(device))
    return model(clip.unsqueeze(0).to(device))


def train_and_smoke(args: argparse.Namespace) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = _device(args.device)
    data_dir = args.data_dir.resolve()
    labels_path = (args.labels or data_dir / "labels.csv").resolve()
    rows = _load_labels(data_dir, labels_path, args.limit_per_class)
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "model" / "stage1"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = Stage1MViT(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    last_loss = None
    for _ in range(args.epochs):
        model.train()
        random.shuffle(rows)
        for row in rows:
            frames = _read_frames(row["path"])
            clip = _clip(frames, _uniform_indices(len(frames), args.frames), args.size)
            target = torch.tensor([LABEL_TO_INDEX[row["label"]]], device=device)
            logits = _forward(model, clip, device)
            loss = F.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss for {row['relative_path']}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    checkpoint_path = model_dir / "best.pt"
    torch.save(
        {"model": model.net.state_dict(), "size": args.size, "frames": args.frames},
        checkpoint_path,
    )

    reloaded = Stage1MViT(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.net.load_state_dict(checkpoint["model"])
    reloaded.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for row in sorted(rows, key=lambda item: item["relative_path"]):
            frames = _read_frames(row["path"])
            probabilities = []
            for indices in _slot_indices(len(frames), args.frames, args.slots):
                logits = _forward(
                    reloaded, _clip(frames, indices, args.size), device
                )
                probabilities.append(float(logits.softmax(1)[0, 1].cpu()))
            rerecorded_probability = float(np.mean(probabilities))
            predictions.append(
                {
                    "ID": Path(row["relative_path"]).stem,
                    "answer": INDEX_TO_LABEL[int(rerecorded_probability >= args.threshold)],
                    "rerecorded_probability": rerecorded_probability,
                    "target": row["label"],
                }
            )

    prediction_frame = pd.DataFrame(predictions)
    contract_path = output_dir / "stage1_predictions.csv"
    prediction_frame[OUTPUT_COLUMNS].to_csv(contract_path, index=False)
    debug_path = output_dir / "stage1_predictions_debug.csv"
    prediction_frame.to_csv(debug_path, index=False)
    if list(pd.read_csv(contract_path).columns) != OUTPUT_COLUMNS:
        raise RuntimeError("Stage1 prediction contract validation failed")
    report = {
        "status": "PASS",
        "device": str(device),
        "samples": len(rows),
        "epochs": args.epochs,
        "last_loss": last_loss,
        "checkpoint": str(checkpoint_path),
        "predictions": str(contract_path),
        "prediction_columns": OUTPUT_COLUMNS,
        "warning": "Smoke accuracy on synthetic training samples is not a generalization metric.",
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-per-class", type=int, default=1)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pretrained", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.epochs < 1 or args.frames < 1 or args.slots < 1:
        raise ValueError("epochs, frames, and slots must be positive")
    train_and_smoke(args)


if __name__ == "__main__":
    main()
