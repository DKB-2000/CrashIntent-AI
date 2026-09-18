# Stage2 사전학습 + Stage3 짧은 시간창 최종 제출 후보 준비

2026-09-11 사용자 요청: 다른 세션의 개선 ZIP을 포함해 Stage3 변경을 테스트할 최종 ZIP 생성 준비.

상태: CPU_PASS_GPU_PENDING. 결합 ZIP 생성 및 CPU 검증 완료, GPU 통합검사 자료 준비 완료. 외부 업로드·GPU 실행·대회 제출은 이번 준비 단계에서 수행하지 않았다.

## 파일과 구성

`artifacts/final-stage2-short-stage3-20260911/submit.zip`

177,371,036 bytes. SHA256 `f94995a9b6303fde41b65153372ad9e48d220e5034d86c9e1c05cfea04fe45fb`.

- Stage1: Stage2 세션이 생성한 기준 ZIP의 기존 모델 유지. 별도로 복구·검증 중인 Stage1 본학습 모델은 포함하지 않았다.
- Stage2: 다른 세션에서 완성한 사전학습 개선 후보 ZIP의 가중치 그대로. 해당 ZIP SHA256 `46db5bde8d42ee22b1582ab3869ea249721571e31aeb0367fa94e7978e697003`, `artifacts/stage2-pretrained-submit-candidate-20260911/validation.json` STATIC_AND_CPU_PARITY_PASS 확인.
- Stage3: 시간창 독립 실험의 short, 첫 고정 seed20260910. 현재 흐름+최근3/5쌍 평균. checkpoint SHA256 `40acb057fdc29b2f2b1d8164c2d68d92c092fbd7b12fe9f61ff2bf4b10047fe0`.

Stage3는 정지·저속 직진 개선과 우회전 손실이 공존하는 테스트 후보다. 공식 .4187467089보다 좋아졌다는 근거는 아직 없다. Stage2와 Stage3를 함께 교체하므로 후속 평가에서는 각 Stage 점수를 따로 비교한다.

## 완료 검증

- 기준 Stage2 ZIP의 Stage1/2 모델·백본·requirements 바이트 보존. 기존 함수/클래스19개의 AST 보존, predict_stage3 교체 및 전용 helper 추가.
- ZIP 필수6파일·규격 검사 PASS.
- 실제 결합 inference 모듈에서 검증18,603행의 원본 short 예측 클래스 전부 일치. 로짓 최대차이1.1444091796875e-05이며 rtol1e-5/atol1e-6 검사 PASS.
- 실제2영상 각각600시점의 short 움직임 특징이 실험 캐시와 정확히 일치.
- 공개 OPEN_001의1,200행 CPU 기준예측을 생성하여 GPU 대조용 fixture 안에만 보관. 제출 ZIP에는 포함하지 않는다.
- 상세 `artifacts/final-stage2-short-stage3-20260911/validation.json`.

## 후속 실행 준비

재현 결합기 `src/package_final_short_candidate.py`, 추론 모듈 `src/stage3_short_motion_inference.py`. 이미 검증된 후보가 있으면 덮어쓰기를 거절한다.
GPU 검사 데이터·노트북은 `artifacts/kaggle-final-short-candidate-20260911/`. 데이터셋 메타데이터는 기존 비공개 `biadis/crashintent-stage2-candidate-assets`, 커널 메타데이터는 `biadis/crashintent-final-short-check`를 가리킨다. 준비 파일만 존재하며 아직 원격 실행하지 않았다.

다음 단계는 이 정확한 ZIP으로 세 단계 T4 통합검사 및 Stage3 CPU/GPU 예측 일치 확인이다. 결과는 `src/validate_candidate_gpu_result.py`와 expected-stage3.csv 대조로 검증한다. 검사 전 기존 후보의 GPU PASS를 이 ZIP의 결과로 재사용하지 않는다.

기존 제출 .4187467089 ZIP, 다른 세션의 원본 Stage2 ZIP, selected.json, Stage1 예약은 변경하지 않았다.
