# Stage2 동일 GPU 입력 정밀도 진단 준비 (2026-09-11)

> 현재 상태 (2026-09-11): GPU 실행·회수·독립 검증 완료. 아래 전송 차단·실행 중 내용은 과거 기록이며 마지막 완료 절을 우선한다.

사용자가 하락 분석 후1번 구·신 동일GPU 입력 비교 진행을 요청했다.
준비는 완료했으나 업로드는 자동 승인 검토에 의해 실행 전 차단됐다.
사용자가 GPU 비교를 승인했지만 구체적 payload와 새 destination 업로드에 대한
명시적 승인이 없다는 거절 사유다. 우회하지 않으며 구체적 승인 대기다.

## 검토 가능한 입력

artifacts/kaggle-stage2-precision-20260911/dataset/precision.bin
93245490bytes(93.25MB), SHA256
fe0d0cee5a3f6b79c63f28d2108458581fa620b8feacfb18af05f08004c85826.
기존CCD66원본영상, 수동라벨, CPU특징, 구·신Stage2 가중치, 동일ResNet,
추론코드 및진단runner·requirements. 새외부자료는 없다.
목적지 비공개 biadis/crashintent-stage2-precision-assets.
노트북 비공개 biadis/crashintent-stage2-precision-check,무료T4.
설치600초/추론2400초 이내, 새학습·공식제출 없음.

## 고정 비교

같은66영상,실제패키지 Stage2 클래스에 구·신 모델을 strict로딩한다.
CPU FP32 캐시+CPU temporal, 같은캐시+GPU temporal,
원본디코드+GPU FP32/FP16 특징,
JPEG재인코드+GPU FP32/FP16 특징의6조건을 비교한다.
시간모델은 제출과 같이FP32, 두 모델은 조건별 동일한 특징 tensor를 사용한다.
입력·모델·코드 해시 검사,모든특징·시간logits·scene logits·예측CSV를 회수하도록 구성했다.
JPEG는 원본을 기본OpenCV JPEG로 재인코드한 대조이며 공식이미지 재현은 아니다.
전체66개는 최종학습에 사용된 자료로 정확도 일반화 검증으로 해석하지 않는다.

src/stage2_precision_runner.py, src/prepare_stage2_precision.py.
노트북 artifacts/kaggle-stage2-precision-20260911/kernel/Stage2_Precision.ipynb.
Python/노트북 AST검사PASS, 실제GPU동작은 아직 미검증.
번들 상태 PREPARED_NOT_UPLOADED. 중복검사 시 같은dataset/kernel 없음 확인.
다음은 구체적 승인 후 Dataset create→ready→kernel push→결과 회수 및 독립재검산.

## 명시 승인 후 업로드·GPU 실행 시작 (2026-09-11)

사용자가93.25MB 자료·비공개 precision-assets 목적지·무료T4 최대50분 실행을 승인했다.
Dataset create 성공 및ready 확인, precision-check version1 push 성공, RUNNING 직접 확인.
이 승인 범위 재확인 불필요. 앞의 전송 차단은 해소됐다. 중복 업로드/실행하지 않는다.

src/watch_stage2_precision.py 숨김PID37760이 상태를45초마다 조회하며 완료 시
stage2-precision-result.zip 회수 후 src/validate_stage2_precision.py를 실행한다.
상태 artifacts/kaggle-stage2-precision-20260911/remote-run.json 및 monitor.log 확인.
검증은 입력manifest/runner 일치,66×6×2=792예측 포함,실제모델클래스 strict로딩,
회수특징에서CPU logits재계산(rtol/atol1e-3),저장logits와범주·시간예측 일치,
동일입력 구신 모델의 조건별 예측변경 수와 특징 차이를 산출한다.
현재 GPU 결과는 아직 없으며 validated/result-validation.json PASS가 있어야 완료다.
로컬 PC가 켜져 있어야 자동 회수·검증이 계속된다. 원격 작업 자체는 Kaggle에서 실행된다.

## 완료: T4 실행·독립 검증 PASS (2026-09-11 09:58 KST)

precision-check v1 COMPLETE, 자동회수 및 validated/result-validation.json PASS.
66영상×6조건×구신2모델=792예측, 원래 모델로 logits 재계산 최대오차2.67e-5.
GPU모델·입력manifest·저장특징·예측동등성 확인.

- 같은CPU캐시→GPU temporal: 구신 모두 예측변화0.
- CPU캐시→원본GPU FP32 특징: 구신 모두 예측변화0.
- 원본GPU FP32→FP16: 구모델0변화,신모델 충돌1건만1프레임(0.1초) 변경.
- JPEG GPU FP32→FP16: 구신 모두 예측변화0.
- 원본GPU FP16→JPEG GPU FP16: 구모델9/66영상변화(충돌5/진입5/회피1/방향1),
  신모델12/66영상변화(충돌6/진입6/회피0/방향0).
- 신모델 최대진입변경25프레임(2.5초),ID000005. 구모델 최대충돌변경1.5초,ID000040.

이 표본에서 FP16 정밀도 차이를 공식점수 하락의 주요 원인으로 볼 근거는 약하다.
JPEG 전처리 영향은 더 크지만 예측변화가 곧 성능하락은 아니다:
신모델 진입±0.3초 정답65/66→66/66,충돌66/66 그대로.
구모델 충돌61/66→59/66,진입57/66그대로. 전체자료가신모델학습에포함되어 있어
이는 전처리민감도 진단이며 모델간 일반화 비교나 공식하락원인 확정에 쓸 수 없다.
JPEG재생성도 공식이미지와 동일하다고 보장하지 않는다.

다음 우선은 학습하지 않은 별도출처의 검증자료 및 해당자료의 실제이미지입력 대조다.
이번 실행에서 새학습·모델교체·ZIP변경·공식제출은 하지 않았다.
