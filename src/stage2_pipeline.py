"""Stage 2 four-target training and end-to-end smoke test.

The ``smoke`` command deliberately creates proxy labels in its output directory.
They prove that the pipeline is wired correctly; they are never suitable for a
real model or copied into ``data/stage2/labels_manual.csv``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


SEED = 20260825
OUTPUT_COLUMNS = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]


class Stage2Temporal(nn.Module):
    """Architecture kept state-dict compatible with the official inference code."""

    def __init__(self) -> None:
        super().__init__()
        self.r = nn.GRU(512, 192, 2, batch_first=True, bidirectional=True, dropout=0.15)
        self.tc = nn.Linear(384, 1)
        self.te = nn.Linear(384, 1)
        self.scene = nn.Sequential(
            nn.Linear(768, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 4)
        )

    def logits(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, _ = self.r(sequence)
        return self.tc(hidden).squeeze(-1), self.te(hidden).squeeze(-1), hidden

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        collision, entry, hidden = self.logits(sequence)
        collision_index = collision.argmax(1)
        entry_index = entry.argmax(1)
        batch = torch.arange(len(hidden), device=hidden.device)
        scene_input = torch.cat(
            [hidden[batch, collision_index], hidden[batch, entry_index]], dim=1
        )
        return collision_index, entry_index, self.scene(scene_input)

    def training_outputs(
        self, sequence: torch.Tensor, collision_target: torch.Tensor, entry_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use teacher-forced event positions so both scene heads receive gradients."""
        collision, entry, hidden = self.logits(sequence)
        batch = torch.arange(len(hidden), device=hidden.device)
        scene_input = torch.cat(
            [hidden[batch, collision_target], hidden[batch, entry_target]], dim=1
        )
        return collision, entry, self.scene(scene_input)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def _read_video(path: Path) -> list:
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


def _backbone(weights_name: str, device: torch.device) -> tuple[nn.Module, object]:
    weights = ResNet18_Weights.IMAGENET1K_V1 if weights_name == "imagenet" else None
    model = resnet18(weights=weights)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return model, transform


def _features(
    frames: list, backbone: nn.Module, transform: object, device: torch.device, batch_size: int
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            images = torch.stack(
                [transform(Image.fromarray(frame)) for frame in frames[start : start + batch_size]]
            ).to(device)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    chunk = backbone(images)
            else:
                chunk = backbone(images)
            chunks.append(chunk.float().cpu())
    return torch.cat(chunks)


def _normalized_rows(csv_path: Path, data_dir: Path, limit: int | None) -> list[dict]:
    frame = pd.read_csv(csv_path, dtype={"ID": str})
    required = {"ID", "path", "t_collision", "t_entry", "evasion_space", "entry_side"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing label columns: {missing}")
    if limit is not None:
        frame = frame.head(limit)
    rows = []
    for row in frame.to_dict("records"):
        video_path = data_dir / str(row["path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        side = str(row["entry_side"]).upper()
        if side in {"0", "LEFT"}:
            side_index = 0
        elif side in {"1", "RIGHT"}:
            side_index = 1
        else:
            raise ValueError(f"invalid entry_side for {row['ID']}: {row['entry_side']}")
        evasion = int(row["evasion_space"])
        if evasion not in {0, 1}:
            raise ValueError(f"invalid evasion_space for {row['ID']}: {evasion}")
        rows.append(
            {
                "ID": str(row["ID"]),
                "video_path": video_path,
                "collision": int(row["t_collision"]),
                "entry": int(row["t_entry"]),
                "evasion": evasion,
                "side": side_index,
            }
        )
    if not rows:
        raise ValueError("labels file contains no rows")
    return rows


def make_smoke_labels(source_csv: Path, output_csv: Path, limit: int | None) -> Path:
    source = pd.read_csv(source_csv)
    if limit is not None:
        source = source.head(limit)
    required = {"ID", "path", "t_collision"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"source labels missing columns: {missing}")
    smoke = source[["ID", "path", "t_collision"]].copy()
    smoke["t_entry"] = (smoke["t_collision"].astype(int) - 8).clip(lower=0)
    smoke["evasion_space"] = [index % 2 for index in range(len(smoke))]
    smoke["entry_side"] = ["LEFT" if index % 2 == 0 else "RIGHT" for index in range(len(smoke))]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    smoke.to_csv(output_csv, index=False)
    return output_csv


def run_pipeline(args: argparse.Namespace, labels_csv: Path) -> dict:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = _device(args.device)
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "model" / "stage2"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = _normalized_rows(labels_csv, args.data_dir.resolve(), args.limit)

    backbone, transform = _backbone(args.backbone_weights, device)
    # Official inference loads this state dict before replacing ``fc`` with
    # Identity, so preserve the original 1000-class layer in the saved file.
    torch.save(backbone.state_dict(), model_dir / "resnet18-f37072fd.pth")
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    samples = []
    for row in rows:
        features = _features(
            _read_video(row["video_path"]), backbone, transform, device, args.feature_batch_size
        )
        frame_count = len(features)
        collision, entry = row["collision"], row["entry"]
        if not (0 <= collision < frame_count and 0 <= entry < frame_count):
            raise ValueError(f"event frame out of range for {row['ID']}: {collision}, {entry}")
        samples.append((row, features, collision, entry))

    temporal = Stage2Temporal().to(device)
    optimizer = torch.optim.AdamW(temporal.parameters(), lr=args.learning_rate)
    last_loss = None
    for _ in range(args.epochs):
        temporal.train()
        random.shuffle(samples)
        for row, features, collision, entry in samples:
            sequence = features.unsqueeze(0).to(device)
            collision_target = torch.tensor([collision], device=device)
            entry_target = torch.tensor([entry], device=device)
            collision_logits, entry_logits, scene_logits = temporal.training_outputs(
                sequence, collision_target, entry_target
            )
            loss = F.cross_entropy(collision_logits, collision_target)
            loss = loss + F.cross_entropy(entry_logits, entry_target)
            loss = loss + F.cross_entropy(
                scene_logits[:, :2], torch.tensor([row["evasion"]], device=device)
            )
            loss = loss + F.cross_entropy(
                scene_logits[:, 2:], torch.tensor([row["side"]], device=device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss for {row['ID']}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    checkpoint_path = model_dir / "best.pt"
    torch.save({"model": temporal.state_dict(), "labels": OUTPUT_COLUMNS}, checkpoint_path)

    reloaded = Stage2Temporal()
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
    reloaded.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for row, features, _, _ in samples:
            collision_index, entry_index, scene = reloaded(features.unsqueeze(0).to(device))
            predictions.append(
                {
                    "ID": row["ID"],
                    "collision_frame": int(collision_index.item()),
                    "entry_frame": int(entry_index.item()),
                    "evasion_space": int(scene[:, :2].argmax(1).item()),
                    "entry_side": "RIGHT" if int(scene[:, 2:].argmax(1).item()) else "LEFT",
                }
            )
    prediction_frame = pd.DataFrame(predictions, columns=OUTPUT_COLUMNS).sort_values("ID")
    if list(prediction_frame.columns) != OUTPUT_COLUMNS or len(prediction_frame) != len(samples):
        raise RuntimeError("prediction contract validation failed")
    predictions_path = output_dir / "stage2_predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)
    report = {
        "status": "PASS",
        "device": str(device),
        "samples": len(samples),
        "epochs": args.epochs,
        "last_loss": last_loss,
        "labels": str(labels_csv.resolve()),
        "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_path),
        "warning": "Proxy smoke labels are not valid training annotations." if args.command == "smoke" else None,
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "train"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--data-dir", type=Path, default=Path("Baseline/data/stage2"))
        sub.add_argument("--output-dir", type=Path, default=Path("artifacts/stage2-smoke"))
        sub.add_argument("--labels", type=Path)
        sub.add_argument("--limit", type=int)
        sub.add_argument("--epochs", type=int, default=1)
        sub.add_argument("--learning-rate", type=float, default=2e-4)
        sub.add_argument("--feature-batch-size", type=int, default=64)
        sub.add_argument("--backbone-weights", choices=("none", "imagenet"), default="none")
        sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.command == "smoke":
        source = args.labels or args.data_dir / "labels.csv"
        labels = make_smoke_labels(source, args.output_dir / "labels_proxy_smoke.csv", args.limit)
        # The generated CSV is already limited; avoid applying the limit twice.
        args.limit = None
    else:
        if args.labels is None:
            raise ValueError("train requires --labels")
        labels = args.labels
    run_pipeline(args, labels)


if __name__ == "__main__":
    main()
