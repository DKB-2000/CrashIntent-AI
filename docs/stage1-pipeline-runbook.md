# Stage1 MViTv2-S training and inference smoke test

This smoke test trains the official-compatible architecture on one ORIGINAL and
one RERECORDED generated sample, saves and reloads `model/stage1/best.pt`, runs
three-slot inference, and checks the required `ID,answer` output contract.

Upload `src/stage1_pipeline.py` through the tools Dataset, then run on Kaggle GPU:

```bash
python stage1_pipeline.py \
  --data-dir /kaggle/working/stage1-smoke \
  --output-dir /kaggle/working/stage1-model-smoke \
  --limit-per-class 1 \
  --epochs 1 \
  --device cuda
```

In a notebook code cell, prefix the command with `!` or use `%%bash`. The default
does not download pretrained weights. Success prints `"status": "PASS"` and creates:

```text
stage1-model-smoke/
├── model/stage1/best.pt
├── stage1_predictions.csv
├── stage1_predictions_debug.csv
└── smoke_report.json
```

This only validates execution and checkpoint compatibility. Accuracy on the two
training samples is not a useful estimate of generalization.
