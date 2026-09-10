# Stage 1 본 학습 진행 기록

2026-09-10 사용자 요청으로 우선순위 3번 Stage 1 본 학습에 착수했다.
**현재: 로컬 전체 합성 생성 중, GPU 사전 학습 6에폭 및 결과 회수·독립 검증 완료.**
GPU 사전 학습 실행을 확인했다. 전체 본 학습 완료 모델은 아직 확보하지 않았다.

## 데이터

CCD 원본 1,500개 실재 확인. 기존 `artifacts/stage1-trial-20260909/dataset/`는 152개 MP4만
있고 완료 manifest/report가 없다. 기존 `artifacts/kaggle-stage1-trial/` 메타데이터는
Stage 3 실행과 첫 제출 후보 Dataset을 가리키므로 재사용하지 않는다.

고정 seed 20260903으로 기존 선택기와 동일한 휴대전화 검증용 30원본을 선택한다.
같은 유튜브 업로드의 형제 클립까지 총 390개를 제외한다. 새 외부 데이터는 사용하지 않는다.

| 분할 | 원본 | 유튜브 출처 | 생성 영상 |
|---|---:|---:|---:|
| 학습 | 880 | 85 | 2,640 |
| 합성 검증 | 230 | 21 | 690 |
| 합계 | 1,110 | 106 | 3,330 |

원본당 ORIGINAL 1개와 RERECORDED 2개. 기존 720p 합성기를 사용하며 두 클래스의
H.264/CRF 범위가 같다. 원본별 고정 seed, 효과, 원본·출력·코드 SHA256을 기록한다.
완료한 원본은 기록과 해시가 일치하면 재시작 시 재사용한다.
실제 휴대전화 재촬영본은 아직 없어 합성 검증 성능을 실제 일반화나 대회 점수로 해석하지 않는다.

출력: `artifacts/stage1-full-20260910/`.
진행 로그: `artifacts/stage1-full-generation-20260910.log`.

```powershell
./.venv/Scripts/python.exe -u src/prepare_stage1_training.py --output-dir artifacts/stage1-full-20260910 --workers 4
```

기존 프로세스가 종료되었는지 확인한 뒤에만 재개한다. 동일 폴더에 중복 실행하지 않는다.
전체 완료 기준은 report의 sources=1110, samples=3330과 manifest 해시다.
현재 report/manifest는 첫 20개 preview 결과일 수 있으므로 완료 로그와 개수를 함께 확인한다.

## 학습 및 확인

`src/train_stage1_full.py`: Kinetics-400 MViTv2-S, 2클래스 head, 기존 제출 체크포인트 호환.
실제 제출 ZIP에서 추출한 serving decoder를 직접 사용한다.
FP32 학습, microbatch 2 × accumulation 4, 역빈도 클래스 가중치.
첫 에폭은 head만 lr 1e-3, 이후 backbone 2e-5/head 1e-4부터 cosine 감쇠, 총 6에폭.
검증은 제출과 같은 FP16 세 구간 확률 평균/0.5 임계값이다.
첫 제출 Stage 1 모델을 동일 검증 영상으로 평가해 대조 점수를 남긴다.
optimizer step·gradient norm·head 변화·에폭 완주·표본 수를 기록하며 최적 모델 저장 후
strict 재로딩과 확률/예측 일치를 확인한다. TIME_LIMIT과 COMPLETED를 구분한다.

`notebooks/Stage1_Full_Training.ipynb`: 제출 requirements 독립 설치, 입력 해시 확인,
실패 시에도 로그·존재하는 모델을 `stage1-training-result.zip`에 보존.
무료 T4 사전 실행 전체 1시간 이내, 본 학습 전체 10시간 이내다.

완료한 확인:

- 첫 20원본(학습 10/검증 10) → 60영상 생성, 전부 50프레임/총 3,000프레임 디코딩 PASS.
- 4원본의 원본형·합성 2종 접촉 시트 확인: `artifacts/stage1-preview-20260910.jpg`.
- 기존 pipeline과 실제 제출 decoder의 세 구간 tensor 완전 일치.
- `test_stage1_training.py` 3테스트 PASS: Macro-F1 쏠림 처리, phone 형제/출처 분리,
  manifest 누수·holdout 포함·해시 손상 거절.
- Python 문법과 노트북 code cell AST 검사 PASS.

## GPU 번들 및 승인

검토 가능한 사전 실행 번들: `artifacts/kaggle-stage1-trial-20260910/`.

- `dataset/stage1-training.bin`: 189,751,816 bytes.
- SHA256: `08707d7d4f1e8876983e4b16dd848db1d8027a2712a2cb16a442595931ffe2f5`.
- CCD 합성 60영상, 분할/해시 manifest, 학습 코드, 기존 Stage 1 모델, 제출 추론 코드.
- 예정 비공개 Dataset: `biadis/crashintent-stage1-trial-v1-assets`.
- 예정 비공개 Notebook: `biadis/crashintent-stage1-trial-v1`.

업로드는 자동 승인 검토에서 **현재 사용자 메시지에 전송 자료와 Kaggle 목적지에 대한
구체적 승인이 없다**는 이유로 실행 전 거절됐다. 인수인계의 기존 승인 기록만으로는 통과하지 못했다.
사용자에게 위 자료의 비공개 업로드·무료 T4 실행 승인을 요청했다. 승인 전 전송을 우회하지 않는다.
로컬 전체 합성은 별도로 진행한다.

승인 후 Dataset 생성 및 사용 가능 상태 확인, GPU 노트북 실행:

```powershell
./.venv/Scripts/kaggle.exe datasets create -p artifacts/kaggle-stage1-trial-20260910/dataset
./.venv/Scripts/kaggle.exe kernels push -p artifacts/kaggle-stage1-trial-20260910/kernel
./.venv/Scripts/kaggle.exe kernels status biadis/crashintent-stage1-trial-v1
```

결과 ZIP 회수 후 입력 번들·코드·manifest·원본 모델·예측 출처를 대조하고 Macro-F1을
독립 재계산한다. 완주 여부, 예산, 최적 모델 해시와 strict 로딩을 확인한다.

```powershell
./.venv/Scripts/python.exe src/validate_stage1_training_result.py --result artifacts/kaggle-stage1-trial-20260910/stage1-training-result.zip --bundle-dir artifacts/kaggle-stage1-trial-20260910 --output-dir artifacts/kaggle-stage1-trial-20260910/validated
```

전체 합성 완료와 GPU 사전 실행 확인 후 본 학습 번들을 준비한다:

```powershell
./.venv/Scripts/python.exe src/prepare_stage1_gpu.py --dataset-dir artifacts/stage1-full-20260910 --output-dir artifacts/kaggle-stage1-full-20260910 --mode full
```

실제 번들 크기·해시·전송 승인 범위를 확인한 뒤 비공개 본 학습을 실행한다.
대회 제출과 일일 배포 후보 교체는 하지 않았다.

## 사용자 승인 후 재개 (2026-09-10)

사용자가 이 대화에서 약 190MB 사전 번들의 비공개 Kaggle 업로드와 무료 T4 실행을
명시적으로 승인했다. 해당 범위는 재확인 없이 진행한다. 상대 경로의 Kaggle CLI 임시
파일 오류는 절대 경로 사용으로 해결하여 업로드를 시작했다.

비공개 Dataset 상태 ready 및 Notebook version 1 push 성공, RUNNING 확인.
실행: https://www.kaggle.com/code/biadis/crashintent-stage1-trial-v1 . 중복 실행하지 않는다.
전체 완료 전 미리보기 manifest로 본 학습 번들을 만들지 못하는 회귀 검사까지 4테스트 PASS.

## 자동 결과 회수

`src/watch_stage1_trial.py`를 숨김 프로세스로 실행했다. 이미 실행된 위 Notebook의 상태만
조회하고, 종료 후 결과 ZIP 다운로드 및 `validate_stage1_training_result.py` 검증을 수행한다.
새 업로드·새 GPU 실행은 하지 않는다. 모니터링 한도는 70분이며 로컬 PC가 켜져 있어야 한다.

- 진행 상태/최종 검증 요약: `artifacts/kaggle-stage1-trial-20260910/remote-run.json`.
- 로그: 같은 폴더 `monitor.log`, `monitor-error.log`.
- 회수 ZIP: 같은 폴더 `result/stage1-training-result.zip`.
- 검증 완료 모델·지표: 같은 폴더 `validated/`.
- `monitor_status=VALIDATED`와 `validated/result-validation.json`을 확인한 뒤 결과를 보고한다.
- 실패 시 last_error와 원격 상태를 먼저 확인한다. 노트북 또는 watcher를 중복 실행하지 않는다.

이 승인 범위의 비공개 전송·무료 사전 실행은 다시 확인하지 않는다.
전체 합성 생성과 그 이후 본 학습은 아직 진행/준비 단계다.

## 사전 결과 검증 완료 및 예상 시간 (2026-09-10 10:04 KST)

Notebook version 1 COMPLETE. 자동 회수는 Kaggle 하위 프로세스의 cp949 출력 오류로
한 번 중단됐으나 PYTHONUTF8/PYTHONIOENCODING을 명시해 복구했다.
`validated/result-validation.json` PASS, 6에폭/24 optimizer updates 완료,
strict 체크포인트 로딩·원본 manifest 및 저장 예측의 Macro-F1 재계산 PASS.
Tesla T4 / torch 2.8.0+cu128, 학습·평가 프로세스 164.79초, 설치 포함 runner 418.71초.

동일 합성 검증 30영상에서 baseline F1=0.25(전부 ORIGINAL), 새 모델 F1=0.4(전부 RERECORDED).
클래스 쏠림이 유지되므로 점수 차이를 유효한 판별 성능 개선으로 해석하지 않는다.
최적 epoch 1, checkpoint SHA256 1209dc4386ea05fa11fc9f0fee267066c42cb9af611179961f45c2d96e55a76b.

10:03에 전체 원본 1,110개 중 140개 완료. 최근 19개 완료 간격 211.24초로 추산하면
전체 합성 완료는 13시 전후(±30분)다. 사전 실행의 역전파 에폭은 약 8초/30학습 영상,
검증 약14초/30영상이다. 학습2640/검증690영상으로 환산하면 6에폭 및 추가 검증은
약2시간, 업로드·설치·결과 회수를 포함해 약2~3시간을 예상한다.
13시경 준비 완료 후 전송·GPU 대기 없이 이어서 실행할 경우 첫 본 학습 결과는
16~17시 예상이다. 이는 조건부 추정이며 성능 개선 완료 시각이나 자동 실행 보장이 아니다.

## 본 학습 코드 반영 완료: balanced-v2 (2026-09-10)

사용자가 다음 단계 중 학습 코드 반영을 먼저 요청했다. GPU 본 학습은 이번 작업에서 시작하지 않았다.
앞 절의 head warmup/역빈도 가중치/cosine 설정은 과거 사전 실행 기록이다.
새 본 학습 기본 후보는 아래와 같으며 실행 번들의 run_config에도 명시한다.

- FP32, Kinetics-400 MViTv2-S, 전체 모델을 첫 업데이트부터 학습.
- 학습률1e-4 고정, dropout/stochastic depth/weight decay=0. 진단 성공 조건에서 출발한다.
- 유효 배치8(원본4/재녹화4), microbatch2. 클래스별 순열을 모두 소비한 뒤 재순환한다.
  소수 원본형 클래스는 반복 사용하며 매 에폭 모든 학습 영상이 최소 한 번 포함된다.
- 균형 배치에 추가 클래스 가중치를 적용하지 않는다. 손실은 microbatch별 CE 합을
  누적 배치 전체 표본 수로 나눈다. 가중치 사용 시에도 전체 가중치 합으로 나누는 공통 함수를 검증했다.
- 기본6에폭, 최소120 optimizer updates. 작은 데이터이면 예정 에폭을 자동 늘린다.
  시간 제한은 그대로 우선하며 미달 시 TIME_LIMIT 및 minimum_updates_met=false로 기록한다.
- 전체 학습 원본형880/재녹화1760영상이면 에폭당3520표본(각1760), 440업데이트,
  6에폭2640업데이트다. 원본 영상 수2640과 반복 포함 처리 표본 수를 구별해 기록한다.
- 진단과 다른 점: 고정 중앙 클립 대신 매번3구간 중 하나를 선택하고, 클래스 내 무작위 복원 추출
  대신 전 영상 포함을 보장하는 순환 순열을 사용한다. 따라서 진단 성능을 그대로 재현한다고 보장하지 않는다.
- 제출 전처리·체크포인트 구조·FP16 세 구간 검증은 유지한다. 최적 검증 모델 선택 및 재로딩도 유지한다.

변경 파일: src/train_stage1_full.py, src/prepare_stage1_gpu.py,
notebooks/Stage1_Full_Training.ipynb, src/validate_stage1_training_result.py,
src/test_stage1_training.py. 과거 업로드된 번들과 별도 Stage3 세션 코드는 변경하지 않았다.

7테스트 PASS: 균형/전체 표본 포함/재현성, 잘못된 배치 거절, 가중·비가중 손실의
부분 microbatch 포함 실제 SGD 업데이트 동등성, 최소 업데이트 계획,
기존 출처 누수·손상 검사와 미완성 데이터 번들 거절. Python·노트북 문법 PASS.
수정된 학습 루프 자체의 전체 GPU 본 학습은 아직 미실행이다.

## 전체 생성 후속 검증·번들 자동 준비 착수 (2026-09-10 11:16 KST)

사용자가 코드 반영에 이어 다음 작업을 요청했다. 당시 전체 생성은543/1110원본으로
아직 진행 중이었다. 생성기와 중복 실행하지 않고, 완료된 원본별 기록만 읽는
src/prepare_stage1_full_run.py를 숨김 프로세스로 실행했다.

이 도구는 원본·파생본 SHA256, 실제 영상 전 프레임 디코딩·FPS·해상도·프레임 수,
원본 출처·라벨·variant 및 최종 manifest/phone holdout/제외 목록을 대조한다.
전체1110원본/3330영상의 검사와 최종 manifest 작성이 끝나야 데이터 PASS로 기록한다.
남아 있는20원본 preview report를 전체 완료로 해석하지 않는다.

- 상태: artifacts/stage1-full-preparation-20260910/status.json
- 로그: 같은 폴더 preparation.log / preparation-error.log
- 전체 검사 완료 보고서: 같은 폴더 data-validation.json
- 전체 검사 후 로컬 번들 자동 생성: artifacts/kaggle-stage1-full-20260910/
- 최종 준비 상태: BUNDLE_READY_NOT_UPLOADED. 번들의 실제 크기·해시는 status.json/bundle.json에 기록.
- 이 프로세스는 외부 업로드나 GPU 학습을 실행하지 않는다. 준비 완료 후 실제 업로드·실행 단계가 남는다.
- 재실행 전 기존 프로세스 종료 여부를 확인한다. 같은 출력으로 중복 실행하지 않는다.

실제 작은AVI4프레임 테스트 PASS: 올바른 영상 검사와 잘못된 프레임수/FPS/크기/해시
거부 확인(src/test_stage1_full_preparation.py). 11:16 첫15원본/45영상/2250프레임 검사 성공,
오류 로그 없음. 전체 검사 완료를 뜻하지 않는다. 로컬 PC가 켜져 있어야 진행된다.


## 전체 합성·검증·학습 번들 완료 (2026-09-10 14:36 KST)

원본1110개에서 영상3330개 생성 완료. 학습2640/검증690영상,
원본형1110/합성재녹화2220개다. 전체166500프레임 디코딩과 해시·FPS·해상도·
출처분리·phone holdout 및 최종manifest 대조 PASS.
상태: artifacts/stage1-full-preparation-20260910/status.json,
BUNDLE_READY_NOT_UPLOADED. 이전 생성 중 기록보다 이 완료 기록을 우선한다.

본 학습 번들: artifacts/kaggle-stage1-full-20260910/dataset/stage1-training.bin
크기4411579071bytes(약4.41GB), SHA256
069fad70203ea83c036205841257b36ef1e1cefc8a7f60705518e82193138758.
manifest SHA256 b0ad9a5280616ea904d8f7243136da069e9add86a44dbb8be3a8c62707dab34d.
설정balanced-v2, 6에폭, lr1e-4, microbatch2/accumulation4.
예정 비공개Notebook biadis/crashintent-stage1-full-v1.
아직 이 전체 번들의 업로드와 GPU 본 학습은 실행하지 않았다.

## 전체 번들 사용자 승인·업로드 및 자동 실행 대기 (2026-09-10 14:55 KST)

사용자가 전체 4.41GB 번들을 비공개 biadis/crashintent-stage1-full-v1-assets에 업로드하고
무료 T4 본학습 최대10시간 실행하는 범위를 명시적으로 승인했다. 동일 범위 재승인 불필요.
Dataset create 업로드 진행 중이며 아직 GPU RUNNING 확인 전이다.
src/launch_stage1_full_when_ready.py 숨김 PID32984가 Dataset ready를 기다린 뒤 kernel push를
한 번만 실행하고 src/watch_stage1_full.py로 결과 회수·독립 검증을 이어간다.
launch-attempt.json 배타 생성으로 중복 push를 방지하며 실패 시 자동 재실행하지 않는다.
모니터는 원격 실행 후 최대12시간, 로컬 PC가 켜져 있어야 업로드·자동 시작·회수가 진행된다.
상태 artifacts/kaggle-stage1-full-20260910/remote-run.json, upload.log, monitor.log,
monitor-error.log, push.log 확인. 다른 세션은 상태를 먼저 확인하고 중복 실행하지 않는다.
잠정 완료 예상17~19시 KST는 이전 GPU 처리시간 환산이며 전체 첫 에폭 실측 전 추정이다.
