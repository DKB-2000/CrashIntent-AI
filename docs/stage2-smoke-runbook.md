# Stage2 4-output smoke test

This test skips manual labeling and checks the complete Stage2 development path:
video decoding, ResNet18 feature extraction, four supervised losses, backward pass,
checkpoint save/reload, and the required prediction columns.

The generated `labels_proxy_smoke.csv` contains deterministic **fake labels**. It is
kept below `artifacts/stage2-smoke/` and must never be used for actual training or
copied to `data/stage2/labels_manual.csv`.

## Kaggle GPU run

From the project root, after installing `Baseline/requirements.txt`:

```bash
python src/stage2_pipeline.py smoke \
  --data-dir Baseline/data/stage2 \
  --output-dir artifacts/stage2-smoke \
  --limit 2 \
  --epochs 1 \
  --device cuda
```

A successful run prints `"status": "PASS"` and creates:

- `artifacts/stage2-smoke/model/stage2/best.pt`
- `artifacts/stage2-smoke/model/stage2/resnet18-f37072fd.pth`
- `artifacts/stage2-smoke/stage2_predictions.csv`
- `artifacts/stage2-smoke/smoke_report.json`

Use `--backbone-weights imagenet` only for a real training run. The smoke default is
`none` so it does not depend on a download or an already-populated Torch cache.

## Real labels later

When manual labels exist, use the same path without proxy-label generation:

```bash
python src/stage2_pipeline.py train \
  --data-dir data/stage2 \
  --labels data/stage2/labels_training.csv \
  --output-dir artifacts/stage2-train \
  --backbone-weights imagenet \
  --device cuda
```

The training CSV contract is:

```text
ID,path,t_collision,t_entry,evasion_space,entry_side
```

`entry_side` accepts `LEFT`/`RIGHT` (or `0`/`1`), and `evasion_space` accepts
only `0`/`1`.
