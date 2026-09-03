# Stage1 synthetic rerecording data smoke test

The generator creates one normalized ORIGINAL and configurable synthetic
RERECORDED variants per CCD source. It assigns the source to train/validation
before generation, so all derivatives remain in one split.

Both labels use the same raw-frame-to-H.264 encoding path and overlapping CRF
range. A manifest records every synthetic parameter.

## Kaggle smoke test

Upload `src/prepare_stage1_rerecorded.py` through the existing tools Dataset.
For the smoke test, the five official Stage1 ORIGINAL examples may be used before
uploading the much larger CCD collection. Then run two sources first:

```bash
python prepare_stage1_rerecorded.py \
  --input-dir /path/to/Baseline/data/stage1/original \
  --output-dir /kaggle/working/stage1-smoke \
  --limit 2 \
  --variants 2
```

In a notebook cell, prefix the command with `!` or use `%%bash`. Success prints
`"status": "PASS"` and `"source_leakage": false`.

Outputs:

```text
stage1-smoke/
├── original/*.mp4
├── rerecorded/*_v01.mp4, *_v02.mp4
├── labels.csv
├── labels_train.csv
├── labels_val.csv
├── source_splits.csv
├── generation_manifest.csv
└── generation_report.json
```

With only two sources, a deterministic hash split may put both in the same split;
an empty train or validation CSV is valid for this smoke test. Use enough source
videos for real training.

Synthetic videos are training candidates, not proof of real rerecording
generalization. Final validation must use the separately recorded phone/display
set, which must not share source scenes with training.

## Full CCD build with a phone holdout

Select the 30 source scenes before generating any derivatives. Selection is
deterministic for the seed, and the CSV is the permanent audit record for the
real phone/display validation set.

```bash
python select_stage1_holdout.py \
  --input-dir /kaggle/input/ccd-crash-1500/videos \
  --output-file /kaggle/working/stage1_phone_holdout_sources.csv \
  --count 30 \
  --seed 20260903
```

Generate only from the remaining 1,470 sources:

```bash
python prepare_stage1_rerecorded.py \
  --input-dir /kaggle/input/ccd-crash-1500/videos \
  --output-dir /kaggle/working/stage1-full-v1 \
  --exclude-file /kaggle/working/stage1_phone_holdout_sources.csv \
  --variants 2 \
  --validation-fraction 0.2 \
  --seed 20260903
```

Expected report values are `excluded_sources=30`, `sources=1470`,
`samples=4410`, and `source_leakage=false`. Preserve the holdout CSV with the
generated Dataset. Never use the 30 source scenes or their phone recordings for
training or synthetic validation.

Run a small `--limit 20` build first, inspect random outputs, and then start the
full build without `--limit`. `ffmpeg` with `libx264` must be available on PATH;
it is available in the normal Kaggle image.
