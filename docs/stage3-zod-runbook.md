# Stage 3 ZOD Sequences 변환 런북

ZOD는 comma2k19의 고속도로 편향을 보완하기 위한 도시·저속·정지·회전 데이터 후보다. 원본 데이터는 Git에 넣지 않는다.

## Kaggle 세션 초기화 후 설치

메일의 비공개 Dropbox URL은 Kaggle Secret `ZOD_DOWNLOAD_URL`에 저장하며 노트북이나 Git에 직접 쓰지 않는다.

```python
%pip install -q "zod[cli]" "dataclass-wizard==0.30.1"
```

ZOD 0.8.0은 `dataclass-wizard` 1.x와 호환되지 않아 0.30.1로 고정한다.

## Sequences mini 변환

세션 초기화 후 저장소를 clone했다면 실제 위치에 맞게 `PROJECT_DIR`을 지정한다.

```python
PROJECT_DIR = "/kaggle/working/CrashIntent-AI"

import subprocess

subprocess.run([
    "python", f"{PROJECT_DIR}/src/prepare_stage3_zod.py",
    "--dataset-dir", "/kaggle/working/zod-sequences-mini",
    "--output-dir", "/kaggle/working/stage3-zod-mini",
    "--overlay",
    "--overwrite",
], check=True)
```

기본 임계값은 정지 0.5m/s, 종가속도 ±0.2m/s², 조향 deadzone ±3°다. 조향각 양수는 임시로 `LEFT`에 매핑한다.
오버레이에서 좌우가 반대로 보이면 `--positive-steer-label RIGHT`를 붙여 다시 실행한다.

## 기대 출력

```text
stage3-zod-mini/
├── videos/ZOD_000000.mp4
├── videos/ZOD_000002.mp4
├── ZOD_000000_overlay.mp4
├── ZOD_000002_overlay.mp4
├── labels.csv
├── labels_debug.csv
└── conversion_report.json
```

`labels.csv`는 베이스라인 계약인 `ID,frame_index,accel_label,steer_label`을 사용한다. 가속페달 ratio는 sequence별
스케일 차이가 있어 라벨 판정에는 쓰지 않고 디버그 CSV에만 보존한다. mini 두 sequence에는 저속·정지 표본이 없으므로
스키마와 변환기 검증에만 사용하고, full 확대 시 센서 통계로 도시·저속·정지·좌우 회전 sequence를 먼저 선별한다.
