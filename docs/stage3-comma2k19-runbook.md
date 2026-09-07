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


## Local ZIP audit and preview (2026-09-07)

Download raw_data/Chunk_1.zip from https://huggingface.co/datasets/commaai/comma2k19 . Expected size: 8,731,252,405 bytes.

Run: `.venv/Scripts/python.exe src/audit_stage3_comma_archive.py --archive data_raw/comma2k19/Chunk_1.http.zip --output-dir artifacts/stage3-comma-chunk1-audit`

The audit reads sensor NPY arrays directly from ZIP without extracting Windows-incompatible paths. Outputs are audit.json and segments.csv. It checks time order, approximate 20Hz frame timing, and at most 1% CAN edge extrapolation. Both CAN/speed and CAN/car_speed are supported. Counts are provisional, with steering offset 0 degrees. Episode starts reset at segment boundaries. No video decoding or label-quality certification is performed.

Use src/extract_stage3_comma_preview.py with --archive, --segment (exact segment prefix from segments.csv), and a new --output-dir to extract only the video and required sensors. The source route remains in source_manifest.json for leakage-safe splitting. Convert the extracted folder using prepare_stage3_comma2k19.py. The converter supports imageio-ffmpeg when no system ffmpeg is available.

Synthetic tests: `.venv/Scripts/python.exe src/test_audit_stage3_comma_archive.py` (2 tests passed). Official example ZIP audit also passed (600 samples; acceleration counts 158/285/157). Calibrate steering on multiple routes before bulk training.

## Chunk conversion resume and validation (2026-09-07)

The accepted audit contains 187 of 188 segments across 21 routes. One segment
failed CAN coverage checks. The stopped batch had 88 label CSVs, but only 86
completion reports. Resume uses `conversion_report.json` as the completion marker
and reuses extracted sources only after checking their source manifest. Each new
conversion writes `conversion.log`, preserving errors for diagnosis.

```powershell
.\.venv\Scripts\python.exe src/prepare_stage3_comma_chunk.py --archive data_raw/comma2k19/Chunk_1.http.zip --audit-csv artifacts/stage3-comma-chunk1-audit-dedup/segments.csv --output-dir artifacts/stage3-comma-chunk1 --workers 2
.\.venv\Scripts\python.exe src/validate_stage3_comma_chunk.py --audit-csv artifacts/stage3-comma-chunk1-audit-dedup/segments.csv --output-dir artifacts/stage3-comma-chunk1
```

Run validation after the conversion command completes. It requires full coverage
of the accepted audit and checks source provenance, conversion settings, CSV
schema, contiguous frame indices, video frame count and 10fps, first/last frame
decoding, and exact equality of combined and individual CSVs. It writes
`validation_report.json` and `video_manifest.csv` with relative video paths.
Videos remain under `converted/<ID>/videos/<ID>.mp4`; the combined labels alone
do not create the flat `videos/` directory expected by the baseline loader.
Use the video manifest when preparing training inputs. Split by source route.

These labels retain the previous batch's provisional settings: each segment's
steering median is subtracted. This is not a calibrated vehicle zero offset and
can remove genuine sustained steering. Review straight/curved/stopped episodes
and calibrate before training. The sensor audit uses offset zero, so its steering
counts are not expected to equal the converted labels. Validation does not certify
physical label accuracy or fully decode every frame.

### Chunk_1 변환·검증 완료 (2026-09-07)

- 원본 ZIP 크기: 8,731,252,405 bytes. 센서 검사 188개 중 187개 통과, 21개 route.
- 제외 구간: `2018-07-29--12-02-42/31` (CAN 범위 밖 표본 비율 1% 초과).
- 중단된 변환을 재개해 187개 MP4, 112,205개 10Hz 라벨(약 187.01분)을 생성했다.
- 가감속 분포: CONSTANT 62,160 / ACCELERATING 21,684 / DECELERATING 20,533 / STOPPED 7,828.
- 조향 분포: RIGHT 39,480 / LEFT 37,987 / STRAIGHT 34,738.
- `artifacts/stage3-comma-chunk1/validation_report.json`: PASS. 원본 대응, 변환 설정,
  CSV 컬럼·연속 프레임 번호, 영상 10fps·프레임 수, 첫/마지막 프레임 디코딩, 통합 CSV 일치 확인.
- `labels.csv`, `manifest.json`, `video_manifest.csv` 생성 완료. 영상은 `converted/<ID>/videos/`에 있다.
- 조향은 기존 작업의 구간별 중앙값 보정을 유지한 임시 라벨이다. 물리적 영점 보정이나 라벨 정확도를
  검증했다는 뜻은 아니며, 본 학습 전에 직진·곡선·정지 표본 검수와 차량/route별 보정이 필요하다.
- 본 학습과 제출 ZIP 생성은 아직 진행하지 않았다. 실행·재개 명령은 `docs/stage3-comma2k19-runbook.md` 참고.

### Stage3 조향 보정 실험 v1 (2026-09-07)

Chunk_1의 학습 17 route에서 IMU·pose를 비교해 영점 −0.2455도와 직진 범위 ±1.5도의
실험용 v1을 생성했다. 검증 4 route는 보정 선택에서 제외했으며 156/31영상, 총 112,205행이다.
원본 라벨은 보존했고 v1은 `artifacts/stage3-comma-chunk1-calibrated-v1/`에 있다.
검수 영상 14개와 전체 1,120프레임 디코딩, 분할·라벨 검사 및 코드 테스트 4개가 통과했다.
공식 정답 기준이나 차량·지역 일반화를 확정한 결과는 아니다. 근거·한계·실행법은
`docs/stage3-calibration-v1.md` 참고. 다음은 v1 기반 Stage3 모델 파이프라인/GPU 스모크 준비다.
