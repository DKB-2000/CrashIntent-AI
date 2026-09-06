# clone 후 로컬 준비

2026-09-06 확인한 로컬 작업 루트: `C:\Crash_AI\CrashIntent-AI`.
`project.properties`는 이동 가능한 기본값을 유지하고, 이 PC의 루트는 git에서 제외되는
`project.local.properties`에 설정했다. 다른 위치로 이동하면 이 파일의 `project.root`를
수정하거나 로컬 설정을 제거해 기본값(`.`)을 사용한다.

## 별도로 복원할 파일

아래 경로는 모두 저장소 루트 기준이다. 이전 PC의 확보 기록과 새 clone의 파일 존재 여부는 별개다.

| 우선순위 | 경로 | 준비 방법 |
|---|---|---|
| Stage1·2 필수 | `data_raw/ccd/Crash-1500.txt` | 이전 PC에서 CCD 공식 라벨 복사 |
| Stage1·2 필수 | `data_raw/ccd/videos/Crash-1500/000001.mp4` ~ `001500.mp4` | 이전 PC의 영상 1,500개 복사(기록상 약 825MB) |
| 기존 작업이 있으면 필수 | `data/stage2/labels_manual.csv` | 수작업 라벨 복원. 처음 시작하면 도구가 자동 생성 |
| 기존 작업이 있으면 필수 | Stage1 holdout/split CSV 및 실제 휴대전화 재촬영 영상 | 기존 분할과 검증 데이터를 함께 복원해 학습·검증 분할 유지 |
| 기준선 비교 시 | `Baseline/submit.zip` | 이전 PC 또는 Kaggle 출력에서 복사. 없으면 학습·추론 노트북으로 재생성 |
| 학습 재개 시 | 기존 `.pt`/`.pth` 체크포인트와 `artifacts/` | 이전 PC/Kaggle에서 기존 상대경로를 유지해 복원 |
| Stage3 작업 시 | `data_raw/` 아래 comma2k19 예제 | `video.hevc`, `frame_times`, processed log를 기존 폴더 구조 그대로 복원 |
| 선택 | CCD `README.md`, `_validation_report.json` | 출처와 이전 검증 기록 보존용 |

CCD를 다시 받아야 한다면 `DATA_SOURCES.md` 1장의 확보 방법을 참고한다.
DoTA는 현재 보류 상태라 복원할 필요가 없다. `Baseline/`의 노트북, requirements,
공개 예제는 clone에 이미 포함되어 있어 `Baseline.zip`을 별도로 추가할 필요가 없다.
원본 데이터, 모델, 가상환경은 Git에 추가하지 않는다.

## Python 환경

2026-09-06 Python 3.11.9를 설치하고 `.venv`를 새로 생성했다.
NumPy 1.26.4와 OpenCV 4.10.0.84 설치 및 `pip check`를 통과했다.
CCD 1,500개와 라벨의 ID 일치를 확인했고, 000001/000250/000750/001500의
각 50프레임(1280×720) 디코딩과 오버레이 생성을 검증했다.
기존 `.venv`는 복사하지 말고 새로 만든다. 환경 재생성 명령은 다음과 같다.

```powershell
Set-Location 'C:\Crash_AI\CrashIntent-AI'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4 opencv-python==4.10.0.84
.\.venv\Scripts\python.exe src/label_stage2.py
```

전체 모델 학습 환경은 `docs/phase0-kaggle-runbook.md`를 따른다.
2026-09-06 NVIDIA RTX 4060 Ti 8GB, 드라이버 591.86을 확인했다.
PyTorch 2.8.0+cu126 / torchvision 0.23.0+cu126으로 CUDA 연산과 Stage2 실제 학습·검증을 통과했다.
Windows의 기본 PyPI 설치본은 CPU용이었으므로 GPU 환경 재생성 시 다음 명령을 추가한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r Baseline/requirements.txt
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

버전별 공식 설치 안내: https://pytorch.org/get-started/previous-versions/
