# comma2k19 Stage3 conversion smoke test

The converter uses comma2k19 `frame_times` as the authoritative 20Hz video timeline,
interpolates CAN speed and steering to 10Hz, derives the four acceleration and three
steering classes, and writes a 10fps MP4 plus the official baseline training CSV.

## Kaggle command

Clone the official sample and make this project's `src/prepare_stage3_comma2k19.py`
available in the notebook. Then run:

```bash
python src/prepare_stage3_comma2k19.py \
  --segment-dir '/kaggle/working/comma2k19/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40' \
  --output-dir /kaggle/working/stage3-comma2k19-smoke \
  --id COMMA2K19_EXAMPLE_40 \
  --steer-offset -0.2 \
  --overlay
```

In a Kaggle notebook code cell, prefix the command with `!` or use `%%bash`.

Success prints `"status": "PASS"` with 1,200 source frames and 600 output frames.
Expected files:

```text
stage3-comma2k19-smoke/
├── videos/COMMA2K19_EXAMPLE_40.mp4
├── labels.csv
├── labels_debug.csv
├── conversion_report.json
└── COMMA2K19_EXAMPLE_40_overlay.mp4
```

`labels.csv` has the exact baseline training columns:

```text
ID,frame_index,accel_label,steer_label
```

The current thresholds are feasibility defaults, not final competition labels. Before
bulk conversion, thresholds and per-vehicle steering zero offsets must be calibrated
on more than one route. In particular, this highway sample contains no STOPPED frames.
