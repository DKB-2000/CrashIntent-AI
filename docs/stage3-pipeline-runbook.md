# Stage3 학습·추론 파이프라인

구현: `src/stage3_pipeline.py`. 입력은 영상만이며 CAN·IMU·pose는 모델에 전달하지 않는다.
공식 MViTv2-S의 backbone/accel/steer 구조와 클래스 순서를 유지한다.

## 현재 검증 상태 (2026-09-07)

- v1 데이터 로더: 학습 156영상/17 route/93,602행, 검증 31영상/4 route/18,603행 확인.
- 행동 테스트 6개 PASS: 과거 프레임·시작 패딩, 모든 시점 출력과 잔여 배치,
  파일별 독립성, STOPPED 손실·정답 기반 평가 마스크, Macro-F1, route 누수와 잘못된 fps 거부.
- 실제 MViTv2-S CPU 스모크 PASS: 실제 영상으로 학습 1스텝, 저장·재로딩 출력 일치,
  짧은 영상 4시점 추론. 최종 실행 약 12초. `artifacts/stage3-pipeline-cpu-smoke-final/smoke_report.json`.
- 정식 train 명령도 별도 4프레임×2영상 데이터에서 학습·검증·best 저장·재로딩 평가를 완료했다.
  `artifacts/stage3-pipeline-integration/` 참고. 이 가중치와 점수는 실행 검증용이다.
- 로컬 torch 2.8.0+cpu / torchvision 0.23.0+cpu. CUDA 없음.
- **GPU 스모크, 전체 v1 본 학습, 비공개 평가 규모의 추론 60분 제한 검증은 미완료다.**

## 데이터와 시간축

`split_manifest.csv`의 video 경로는 manifest 폴더 기준 상대경로다.
분할 라벨은 `labels_train_candidate.csv`, `labels_validation_candidate.csv`를 읽는다.
ID·연속 프레임 번호·클래스·영상 프레임 수·10fps·route 누수·중복 영상 경로를 검사한다.

현재 프레임 t의 입력은 max(0,t−15)..t의 16프레임이다. 시작 부분은 첫 프레임으로 패딩한다.
짧은 변을 256으로 리사이즈 → 224 중심 crop → RGB → mean 0.45/std 0.225 정규화한다.
학습과 추론에 같은 전처리를 사용한다. 공식 학습 예제의 중심 시점 주변 샘플링 대신 과거만 사용하는
방식으로 통일했으므로 체크포인트에 `stage3-causal16-v1` 전처리 계약을 저장한다.
이 코드의 추론 로더는 전처리·클래스 정보가 없는 예전 체크포인트를 거부한다.

학습은 비디오 하나를 리사이즈된 uint8로 보관하고 그 안의 시점을 섞어 배치로 학습한다.
기본 train-stride=8, epoch별 시작 offset을 바꾼다. 학습량 절감을 위한 표본 추출이며
**검증·추론은 stride=1로 모든 시점을 출력**한다. 추론은 최근 16프레임과 작은 배치만 보관한다.
정지 정답 표본은 조향 손실에서 제외하지만 가감속 손실에는 포함한다.
CUDA 학습은 autocast/GradScaler와 gradient clipping을 사용한다.

## CLI

```powershell
# 실제 입력 감사 (GPU 불필요)
.\.venv\Scripts\python.exe src/stage3_pipeline.py audit --dataset-dir artifacts/stage3-comma-chunk1-calibrated-v1

# CPU 실행 연결 검사: 새 output-dir을 사용
.\.venv\Scripts\python.exe src/stage3_pipeline.py smoke --dataset-dir artifacts/stage3-comma-chunk1-calibrated-v1 --output-dir artifacts/stage3-cpu-new-smoke --device cpu
```

GPU 학습 환경에서:

```bash
python src/stage3_pipeline.py train --dataset-dir DATASET --output-dir RUN --device cuda --epochs 3 --batch-size 2 --train-stride 8 --from-scratch
python src/stage3_pipeline.py evaluate --dataset-dir DATASET --checkpoint RUN/model/stage3/best.pt --output RUN/evaluation-extra.json --device cuda
python src/stage3_pipeline.py predict --data-dir STAGE3_INPUT --checkpoint RUN/model/stage3/best.pt --output predictions.csv --device cuda
```

`--from-scratch`는 무작위 초기화임을 명시적으로 선택하는 옵션이다. 기존 공식 Stage3 호환 가중치를
초기화에 쓸 경우 대신 `--init-checkpoint LOCAL_STAGE3_BEST_PT`를 사용한다. 새 가중치를 다운로드하지 않는다.
기존 베이스라인 스모크 가중치가 성능용 사전학습 모델이라는 뜻은 아니다.
학습 출력 폴더가 이미 있으면 덮어쓰지 않는다. 현재 optimizer resume 기능은 없다.

학습 결과: `model/stage3/best.pt`, `history.json`, `data_fingerprints.json`,
`evaluation.json`, `validation_predictions.csv`. 최적 모델은 검증 주행 중 조향 Macro-F1으로 선택한다.
Macro-F1은 고정된 전체 클래스 집합으로 계산하고 0분모 클래스 F1은 0으로 둔다.
정지 제외는 예측 STOPPED가 아니라 **정답 STOPPED** 기준이다. 외부 자동 라벨의 개발 지표이며
대회 점수를 재현한 것으로 주장하지 않는다. 전체 v1 학습 전 실행시간과 실제 검증 오류를 확인한다.

## GPU 스모크 번들

`artifacts/stage3-gpu-smoke.zip` (66,556,814 bytes, 약 63.5 MiB).
영상 2개(학습/검증 각 1개), 각 600라벨, 독립 실행 코드·테스트·baseline requirements·README를 포함한다.
CRC와 내부 Python 구문 검증 PASS. 전체 187개 영상이나 성능용 모델은 포함하지 않는다.

Kaggle의 쓰기 가능한 작업 폴더에 ZIP을 풀고 GPU를 활성화한 뒤 번들의 README 명령을 실행한다.
기존 Phase 0처럼 패키지 교체 후 NumPy/Pandas ABI 충돌이 나면 커널을 재시작한다.

```bash
python src/test_stage3_pipeline.py
python src/stage3_pipeline.py audit --dataset-dir dataset
python src/stage3_pipeline.py smoke --dataset-dir dataset --output-dir smoke-run --device cuda
```

확인할 결과는 `smoke-run/smoke_report.json`의 status=PASS 및
public_cuda_entrypoint_tested=true, `smoke-run/predictions.csv`의 연속 4시점 출력이다.
이 과정은 GPU 혼합정밀도 학습·추론과 공개 함수까지 검증한다. Kaggle T4 실행과 반환 ZIP 검증에서 GPU PASS를 확인했다(아래 실행 기록 참고).

번들 재생성(새 ZIP 경로):

```powershell
.\.venv\Scripts\python.exe src/package_stage3_smoke.py --dataset-dir artifacts/stage3-comma-chunk1-calibrated-v1 --output artifacts/stage3-gpu-smoke-new.zip
```

## 평가 서버 계약과 제출 통합

`predict_stage3(data_dir, model_dir)`는 CUDA 전용이며
`data_dir/videos/*`와 `model_dir/stage3/best.pt`를 읽어 다음 DataFrame을 반환한다.

```text
ID,sample_index,accel_label,steer_label
```

모든 시점에 두 라벨을 출력하며 파일 간 예측을 참조하지 않는다. 모델 생성은 weights=None이고
가중치는 로컬 체크포인트만 읽으므로 추론 중 다운로드하지 않는다.

현재 파일은 Stage3 개발 모듈이다. 최종 제출 시 Stage1/2와 함께 inference.py로 통합하고 기존 ZIP
검증을 다시 수행해야 한다. **이번 GPU 번들은 submit.zip이 아니며 제출하면 안 된다.**
## Kaggle 실행 노트북 및 결과 회수 (2026-09-07)

`notebooks/Stage3_GPU_Smoke.ipynb`를 추가했다. Kaggle CLI 2.2.4에서 사용자 로그인과
`biadis` 계정 접근을 확인했고, 아래 비공개 Dataset과 T4 GPU 노트북을 생성했다.
CLI로 실행 및 결과 회수가 가능하다. 로그인 토큰은 저장소나 실행 번들에 포함하지 않는다.

- Dataset: https://www.kaggle.com/datasets/biadis/crashintent-stage3-gpu-smoke
- Notebook: https://www.kaggle.com/code/biadis/crashintent-stage3-gpu-check
- 업로드 메타데이터: `artifacts/kaggle-stage3-smoke/{dataset,kernel}/`
- Windows CLI의 업로드 경로에는 역슬래시를 사용한다. Dataset과 Notebook 제목은 서로 달라야 한다.
- 실행 설정: 비공개, `NvidiaTeslaT4`, 인터넷 비활성화, 제한 시간 900초.

```powershell
.\.venv\Scripts\kaggle.exe kernels status biadis/crashintent-stage3-gpu-check
.\.venv\Scripts\kaggle.exe kernels output biadis/crashintent-stage3-gpu-check -p .\artifacts\kaggle-stage3-smoke\result --file-pattern 'stage3-gpu-result[.]zip$'
```

웹에서 수동 실행할 경우:

1. 기존 `artifacts/stage3-gpu-smoke.zip`을 비공개 Kaggle Dataset으로 올린다.
2. `notebooks/Stage3_GPU_Smoke.ipynb`를 Kaggle Notebook으로 가져오고 위 Dataset을 연결한다.
3. GPU를 활성화하고 셀을 순서대로 실행한다. ZIP 그대로이거나 자동 압축 해제된 입력을 모두 지원한다.
4. 마지막 셀의 `stage3-gpu-result.zip`을 내려받아 로컬 프로젝트 `artifacts/`에 둔다.

노트북은 새 작업 폴더를 만들어 파일 해시 검증, 환경 기록, 코드 테스트, 데이터 감사,
CUDA 학습·재로딩·공개 함수 추론을 순서대로 실행한다. 패키지를 자동 변경하지 않으며 필요한 경우
설치 명령과 커널 재시작 절차를 노트북에 안내했다. 결과 ZIP에는 환경·로그·report·predictions만
넣으며 모델이나 원본 영상을 다시 담지 않는다.

로컬 반환 결과 검사:

```powershell
.\.venv\Scripts\python.exe src/validate_stage3_gpu_result.py --result artifacts/stage3-gpu-result.zip --bundle artifacts/stage3-gpu-smoke.zip --output artifacts/stage3-gpu-result-validation.json
```

이 검사는 반환된 증거의 일관성을 확인한다. CUDA 실행·공개 함수 테스트·체크포인트 재로딩,
입력 번들 일치, 세 명령 성공, 연속 4개 예측과 유효 클래스를 모두 요구한다.
원격 실행의 암호학적 증명이나 전체 데이터 성능·시간 제한 인증은 아니다.

검증 기록: 노트북 코드 셀 4개 구문 검사와 실제 ZIP 준비 셀 실행 PASS(파일 해시 8개 확인).
반환 결과 검사기의 합성 성공/실패 테스트 7개 PASS. 합성 테스트는 실제 GPU 결과가 아니다.
실제 GPU 결과 ZIP을 회수하고 로컬 검사도 PASS했다. 참고: https://www.kaggle.com/docs/notebooks

### 실제 원격 실행 결과 (2026-09-09 기록)

- Notebook version 1: `biadis/crashintent-stage3-gpu-check`, Kaggle 상태 COMPLETE.
- Tesla T4 2개가 노출된 환경에서 단일 CUDA 장치로 실행했다. 분산 학습 검증은 아니다.
- Python 3.12.13, torch 2.10.0+cu128, torchvision 0.25.0+cu128, CUDA 12.8.
- NumPy 2.0.2, Pandas 2.3.3, OpenCV 4.13.0. Kaggle 기본 패키지를 사용했다.
- 테스트·데이터 감사·CUDA 스모크 명령 모두 exit code 0.
- 실제 데이터 학습 1스텝, loss 3.1902566, 체크포인트 재로딩 true, 공개 CUDA 함수 테스트 true.
- 예측 4행, 번들 메타데이터 일치, ZIP CRC 및 결과 검사 PASS.
- 보고서 내부 측정 4.13초, smoke 프로세스 전체 15.32초. 전체 데이터 처리 시간 추정값은 아니다.
- 반환 ZIP: `artifacts/kaggle-stage3-smoke/result/stage3-gpu-_wkpmaa9/stage3-gpu-result.zip`
- 로컬 검증 보고서: `artifacts/kaggle-stage3-smoke/result-validation.json`

```powershell
.\.venv\Scripts\python.exe src/validate_stage3_gpu_result.py --result artifacts/kaggle-stage3-smoke/result/stage3-gpu-_wkpmaa9/stage3-gpu-result.zip --bundle artifacts/stage3-gpu-smoke.zip --output artifacts/kaggle-stage3-smoke/result-validation-recheck.json
```

다운로드 CLI는 파일 저장 후 콘솔 CP949 인코딩 오류로 종료 코드 1을 반환했지만,
저장된 결과 ZIP은 CRC 및 모든 내용 검사를 통과했다. 재실행 시 `PYTHONUTF8=1`을 설정할 수 있다.

이 PASS는 실행 연결 검증이다. 전체 187개 영상 학습·성능 평가·제출 ZIP 통합은 아직 하지 않았다.
제출 requirements의 torch 2.8.0 / torchvision 0.23.0과 실제 GPU 환경 버전이 다르므로,
이후 아래 제출 requirements 검증에서 해당 버전 조합도 PASS했다. 최종 통합 제출 환경 검증은 남아 있다.


## 제출 requirements GPU 호환성 검증 (2026-09-09)

`notebooks/Stage3_GPU_Compatibility.ipynb`는 기존 스모크 번들을 검증한 뒤 `/tmp`에 독립
가상환경을 만들고 번들의 requirements 전체를 설치한다. 새 Python 프로세스에서 정확한
패키지 버전을 검사하고 테스트·감사·CUDA 스모크를 실행하므로 수동 커널 재시작이 필요 없다.
설치에만 패키지 다운로드가 필요하며 모델 가중치는 다운로드하지 않는다.

- 비공개 Notebook: https://www.kaggle.com/code/biadis/crashintent-stage3-pinned-check
- 설정: T4, 인터넷 활성화(의존성 설치), 제한 시간 1,800초.
- 최초 실행은 Kaggle `ensurepip` 실패로 종료됐다. `venv --without-pip` 및 기존 pip의
  `--python` 옵션으로 독립 환경에 설치하도록 수정했다.
- 결과: 기존 검사기용 `stage3-gpu-result.zip`과 별도 `stage3-compatibility-evidence.zip`.
  후자는 설치 시간·로그·requirements 해시·실제 설치 버전·실행 스크립트와 로그를 포함한다.
- 이 검증은 제출 패키지 버전 호환성 확인이다. Kaggle T4/Python 환경과 평가 서버 전체 환경은
  같지 않으며, 인터넷이 비활성화된 실행이나 제출 설치 시간 제한 충족을 자동 보증하지 않는다.
- 공식 PyTorch 버전 조합 참고: https://pytorch.org/get-started/previous-versions/

후속 전체 데이터는 영상만 4,914,740,942 bytes(약 4.58 GiB)이며, 학습 156개와 검증 31개다.

### 제출 버전 실행 결과

Notebook version 2가 COMPLETE이며 반환 ZIP CRC·기존 GPU 결과 검사·requirements 원문 및 SHA256
일치·설치 패키지 6개 버전 검사 모두 PASS했다.

| 패키지 | 설치 버전 |
|---|---|
| torch | 2.8.0 (런타임 2.8.0+cu128) |
| torchvision | 0.23.0 (런타임 0.23.0+cu128) |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| opencv-python | 4.10.0.84 |
| tqdm | 4.66.5 |

Python 3.12.13 / CUDA 12.8 / Tesla T4에서 실제 데이터 학습 1스텝, 체크포인트 재로딩,
공개 CUDA 함수 4행 추론까지 PASS했다. 패키지 설치는 166.72초, 스모크 보고서 내부 측정은
3.89초였다. 이 설치 시간은 이번 Kaggle 측정이며 평가 서버의 설치 시간 보장은 아니다.

- 노트북: https://www.kaggle.com/code/biadis/crashintent-stage3-pinned-check
- 결과 폴더: `artifacts/kaggle-stage3-compatibility/result/stage3-gpu-wob66o6h/`
- 통합 로컬 검증: `artifacts/kaggle-stage3-compatibility/result-validation.json`
- 후속 작업: 전체 v1 Dataset 업로드 후 짧은 시험 학습으로 메모리·처리 시간·검증 지표를 확인한다.


## 전체 v1 업로드와 시험 학습 (2026-09-09)

`src/package_stage3_training.py`로 기존 route 분할을 유지한 전체 학습 번들을 구성했다.
187개 영상, train 93,602행 / validation 18,603행, ZIP 4,915,193,617 bytes다.
영상 및 코드·라벨별 SHA256을 기록하고 ZIP CRC 검사도 통과했다.

- 비공개 Dataset: https://www.kaggle.com/datasets/biadis/crashintent-stage3-training-v1
- Notebook: https://www.kaggle.com/code/biadis/crashintent-stage3-training-trial
- 로컬 Notebook: `notebooks/Stage3_Training_Trial.ipynb`
- 번들: `artifacts/kaggle-stage3-training/dataset/stage3-training-v1.zip`
- 설정: 제출 requirements, T4, 학습 2영상/검증 2영상, 1 epoch, batch 2, stride 32,
  무작위 초기화. 전체 데이터 본 학습이 아닌 실행 시간·메모리·평가 흐름 확인이다.
- 시작 시 전체 번들 해시와 187영상 메타데이터·라벨을 감사한다.
- 결과: `stage3-trial-result.zip`에 선택 ID, 최고 GPU 메모리, 실행 시간, history,
  evaluation, 검증 예측과 라벨, 입력 해시, 실행 로그를 담는다.
- 체크포인트는 Kaggle 출력의 `trial-run/model/stage3/best.pt`에 보존된다.
  시험용 무작위 초기화 모델이며 제출용 성능 모델로 취급하지 않는다.


### 시험 학습 실제 결과

Notebook version 1 COMPLETE, `src/validate_stage3_trial.py` 로컬 검증 PASS.
전체 파일 해시와 156/31영상·93,602/18,603라벨 감사가 통과했다.

- 시험 ID: train `COMMA2K19_C1_0000`, `0001`; validation `0022`, `0023`.
- 1 epoch / 20 optimizer steps / mean loss 3.2936543.
- 학습+첫 검증 97.38초, 모델 재로딩과 재검증 포함 176.85초(패키지 설치·전체 감사 제외).
- 최고 GPU 할당 2,928,471,040 bytes(2.73 GiB), 예약 3,296,722,944 bytes(3.07 GiB).
- 검증 예측 1,200행, GT STOPPED 159행 제외 후 조향 평가 1,041행.
- 가감속 Macro-F1 0.1068152, 주행 중 조향 Macro-F1 0.0945315.
- 모든 예측이 가감속 CONSTANT·조향 RIGHT로 쏠렸다. 두 학습 영상의 무작위 초기화 20스텝
  결과이며 성능용 모델로 쓰지 않는다. 전체 데이터의 성능이나 대회 점수로 해석하지 않는다.
- 원본 번들·선택 라벨·학습 입력 해시 일치, 예측으로 지표 재계산 일치,
  체크포인트 재로딩 전후 평가 일치 PASS.
- 결과 ZIP: `artifacts/kaggle-stage3-training/result/stage3-gpu-_dmv4lem/stage3-trial-result.zip`
- 보고서: `artifacts/kaggle-stage3-training/result-validation.json`

```powershell
.\.venv\Scripts\python.exe src/validate_stage3_trial.py --result artifacts/kaggle-stage3-training/result/stage3-gpu-_dmv4lem/stage3-trial-result.zip --bundle artifacts/kaggle-stage3-training/dataset/stage3-training-v1.zip --output artifacts/kaggle-stage3-training/recheck.json
```

다음은 기존 route 분할 전체로 본 학습한다. batch2는 이번 T4 시험에서 동작했으며,
기본 stride8로 학습 표본을 늘린다. 첫 전체 epoch 결과에서 클래스 쏠림·검증 오류와 실행 시간을
확인한다. 이번 시험은 제한된 두 route의 일부 영상이므로 전체 처리 시간을 단순 비례로 확정하지 않는다.


## 전체 데이터 첫 본 학습 실행 (2026-09-09)

- Notebook: https://www.kaggle.com/code/biadis/crashintent-stage3-full-training (version 1)
- 로컬: `notebooks/Stage3_Full_Training.ipynb`
- 기존 전체 v1 비공개 Dataset을 사용한다. 156영상 학습 / 31영상 검증.
- 1 epoch, batch2, stride8, learning rate1e-4, 무작위 초기화, seed20260825.
- 제출 requirements의 독립 환경, T4, 최대14,400초. 25스텝마다 학습 로그를 출력한다.
- 첫 전체 에폭 후 검증 점수·클래스 쏠림을 평가하기 위한 초기 본 학습이다.
- 실행 완료 후 파일·체크포인트·전체 예측 검증 PASS. 단일 클래스 쏠림이 확인돼 성능 개선이 필요하다. 상세 `stage3-full-training-v1-results.md` 참고.
- 결과 ZIP은 모델을 제외한 로그·환경 버전·입력해시·history·evaluation·검증예측을 담는다.
  체크포인트는 별도 `training-run/model/stage3/best.pt`로 보존한다.

```powershell
.\.venv\Scripts\kaggle.exe kernels status biadis/crashintent-stage3-full-training
$env:PYTHONUTF8='1'
.\.venv\Scripts\kaggle.exe kernels output biadis/crashintent-stage3-full-training -p .\artifacts\kaggle-stage3-full-training\result --file-pattern '(stage3-training-result[.]zip|training-run/model/stage3/best[.]pt)$'
```

재개 시 먼저 상태를 확인한다. 완료되면 입력 번들·18,603행 검증예측·GT STOPPED 제외 지표를
검사하고 체크포인트를 회수한다. 실패 시 로그를 확인한다. 동일 노트북 push는 학습을 새로 실행하므로
상태 확인을 위해 push하지 않는다. optimizer resume은 현재 구현돼 있지 않다.


## 영상 혼합 배치 추가 학습 (2026-09-09)

사용자 요청으로 기존 첫 에폭 가중치에서 추가3에폭(누적 epoch2~4)을 실행한다.
새 데이터나 라벨 변경은 없다. 156/31 route 분할, batch2, stride8, lr1e-4를 유지한다.
`--video-buffer-size 8`은 무작위 영상8개를 디코딩해 표본을 영상 간 섞고,
다음 영상 그룹으로 진행한다. 전체156영상의 모든 클립을 한 번에 섞는 방식은 아니다.
각 에폭에서 선택 표본을 중복·누락 없이 사용하며, 16프레임 클립은 한 영상 안에서만 만든다.
`--epoch-offset 1`로 순서 seed와 stride 위상을 누적 에폭에 맞춘다.

- Notebook: https://www.kaggle.com/code/biadis/crashintent-stage3-mixed-finetuning (v1)
- 로컬: `notebooks/Stage3_Mixed_Finetuning.ipynb`
- 추가 비공개 코드·가중치 Dataset: `biadis/crashintent-stage3-mixed-assets`
- 초기 체크포인트 SHA256: `5d11e953aa24656119408bd210c9001ad99cf05793a208e9f4b34ded9e6e1c01`
- AdamW/GradScaler 상태는 기존 체크포인트에 없어 새로 생성한다. 가중치 추가 학습이며
  optimizer까지 정확히 복원하는 resume은 아니다.
- T4, 제출 requirements, 최대8시간. 매 에폭 전체 검증을 수행하고 주행 중 조향 F1이
  가장 높은 추가 학습 모델을 저장한다. 원래 첫 에폭 모델은 별도로 보존한다.
- 테스트8개 PASS(기존6개+혼합 표본 커버리지/재현성/클립·라벨 대응2개).
- 실제 영상8개를 읽은 배치 shape `(2,3,16,224,224)` 확인 PASS.
- 추가 각 에폭: 11,700표본 / 5,850배치. 서로 다른 영상으로 구성된 배치는
  epoch2 5,123개(87.57%), epoch3 5,145개(87.95%), epoch4 5,068개(86.63%).
- 사전검사: `artifacts/kaggle-stage3-mixed-training/local-preflight.json`.
- 원본 데이터 번들 해시를 검사한 뒤 별도 `finetune.json`의 코드·초기 가중치 해시를 검사해
  새 코드를 적용한다. 결과 ZIP에 해당 manifest와 `training_config.json`도 기록한다.

```powershell
.\.venv\Scripts\kaggle.exe kernels status biadis/crashintent-stage3-mixed-finetuning
$env:PYTHONUTF8='1'
.\.venv\Scripts\kaggle.exe kernels output biadis/crashintent-stage3-mixed-finetuning -p .\artifacts\kaggle-stage3-mixed-training\result --file-pattern '(stage3-training-result[.]zip|training-run/model/stage3/best[.]pt)$'
```

완료 후 `src/validate_stage3_training.py`로 지표를 재계산하고 첫 에폭과 같은 검증 세트에서
클래스별 F1·예측 분포를 비교한다. `finetune.json` 및 초기 가중치 해시도 로컬 assets와 대조한다.
추가 학습과 배치 혼합을 함께 바꾼 실행이므로 개선되더라도 혼합만의 인과 효과로 주장하지 않는다.
