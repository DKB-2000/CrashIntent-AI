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
이 과정은 GPU 혼합정밀도 학습·추론과 공개 함수까지 검증한다. GPU PASS는 아직 받지 않았다.

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