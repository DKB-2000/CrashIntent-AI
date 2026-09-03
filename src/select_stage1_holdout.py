"""Select a deterministic source-level holdout for real phone rerecording."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _rank(source_id: str, seed: int) -> str:
    return hashlib.sha256(f"stage1-phone-holdout:{seed}:{source_id}".encode("utf-8")).hexdigest()


def select_holdout(
    input_dir: Path,
    output_file: Path,
    count: int,
    pattern: str,
    seed: int,
    overwrite: bool,
) -> pd.DataFrame:
    input_dir = input_dir.resolve()
    output_file = output_file.resolve()
    videos = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    source_ids = [path.stem for path in videos]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source IDs detected; video filename stems must be unique")
    if count < 1 or count >= len(videos):
        raise ValueError(f"count must satisfy 1 <= count < source videos ({len(videos)})")
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output_file}")

    selected = sorted(videos, key=lambda path: (_rank(path.stem, seed), path.stem))[:count]
    rows = [
        {
            "source_id": path.stem,
            "source_path": str(path),
            "selection_rank": index + 1,
            "seed": seed,
        }
        for index, path in enumerate(selected)
    ]
    result = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f"PASS: selected {len(result)} of {len(videos)} sources -> {output_file}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    select_holdout(
        args.input_dir,
        args.output_file,
        args.count,
        args.pattern,
        args.seed,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
