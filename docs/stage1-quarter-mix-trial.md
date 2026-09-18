# Stage1 25% 혼합 대조 (2026-09-16)

사람 라벨 없이 생성 절차가 정답을 확정한 기존 학습 영상을 재사용한다. 9월15일 `screen_mix_50` 공식 Stage1 점수가 기존 최고보다 낮았으므로, 원본 화질과 약한 화면단서의 균형을 바꿔 같은 GPU에서 네 조건을 비교한다. 개발 진단 자료는 반복 사용됐고 독립 실제 재촬영 검증은 없다. 학습 출처85와 검증 출처21·phone 보류 출처는 분리한 채 유지한다.

조건은 baseline / mixed_50 / mixed_25 / screen_mix_25다. 세 학습 조건은 동일 최고 체크포인트·seed20260911·128업데이트·lr1e-5·유효배치8·FP32학습에서 출발한다. `mixed_25`는 매배치4출처쌍 중 기존합성3쌍, 화질균형1쌍이다. `screen_mix_25`는 같은 배치의 화질균형 재녹화1입력만 기존 검증된 weak_screen/no_geometry/no_flicker 영상으로 교체한다. 원본4/재녹화4, 출처·profile·slot·업데이트 수를 유지한다. 고정420개 개발 영상은 CUDA FP16, 3slot 평균, 임계값0.5로 평가한다. 매조건 원본 noise/quality_mix 오류 각각 baseline보다 감소, 모든 stress 재녹화 오류증가≤1/21, normal63 Macro-F1 감소≤0.02를 사전 gate로 사용한다. 통과해도 자동 승격·공식 일반화 확정은 하지 않는다.

기존 비공개 `biadis/crashintent-stage1-quality-trial-assets`와 `biadis/crashintent-stage1-screen-trial-assets`의 영상·초기모델을 재사용한다. 새 비공개 `biadis/crashintent-stage1-quarter-mix-assets`에는 runner/plan/manifest와 노트북 실행에 필요한 코드·설정만 넣었다. 코드 delta 874053bytes SHA256 `40cf383c01ca306b77f813300a62c5b1dcedafaacc9157de017a094cc84658e2`. 무료T4 목적지는 `biadis/crashintent-stage1-quarter-mix-trial`. 로컬 전체 검증 번들 SHA256 `7a41e06ef8de0ada6853e3f6c57b069d8c08ef39ac3439d96f6ffc5b820a0357`이며 1530학습영상·420평가영상·384학습업데이트·1680예측 계획이다. 배치/쌍/화면단서 교체 3테스트와 Python 문법검사 PASS.

사용자의 후속 진행 지시에 따라 구체적 전송자료·목적지·해시를 `upload-approval.json`에 기록하고 숨김 감시기 PID27284를 시작했다. 최초 확인은 `remote-run.json` UPLOADING/ACTIVE, 프로세스 존재다. 업로드 완료나 GPU 시작은 아직 확인되지 않았다. 중복 실행 금지. 감시기는 단일 업로드·push 마커, 원격 상태/결과 회수 유한 재시도, 독립 결과 검사와 실패 로그를 남긴다. 완료 조건은 remote-run COMPLETE/VALIDATED 및 validated/result-validation.json PASS다. 상태는 `./scripts/Get-BackgroundWorkStatus.ps1`에서 한 번 조회한다. Codex가 반복 폴링하지 않는다. 기존 공식 최고 ZIP·selected·12시간 예약은 유지한다.

## 완료 결과

Kaggle T4 실행, 결과 회수와 독립 검증이 완료됐다. `remote-run.json`은 COMPLETE/VALIDATED, `validated/result-validation.json`은 PASS다. 4조건 1,680예측과 384학습 업데이트, 실제 배치 일정, 입력·코드·모델 해시, 저장 로짓과 조건별 지표 재계산을 검증했다.

| 조건 | 원본 noise 오류 /21 | 원본 quality_mix 오류 /21 | weak_screen 재녹화 오류 /21 | 나머지 재녹화 조건 최대 오류 | normal63 Macro-F1 | 사전 gate |
|---|---:|---:|---:|---:|---:|---|
| baseline | 19 | 18 | 0 | 0 | 1.0 | 기준 |
| mixed_50 | 0 | 9 | 3 | 1 | 1.0 | 실패 |
| mixed_25 | 6 | 12 | 2 | 0 | 1.0 | 실패 |
| screen_mix_25 | 7 | 16 | 0 | 0 | 1.0 | 통과 |

`screen_mix_25`는 원본 noise 19→7, quality_mix 18→16으로 둘 다 개선하면서 모든 재녹화 조건 오류0과 normal Macro-F1 1.0을 유지해 사전 gate를 통과했다. 체크포인트 SHA256은 `49118651e9617c48216f5eff3baf3c5bb5f515a471a628f576d327e08db8e534`다.

그러나 quality_mix 개선은 2/21에 그치고, 반복 사용한 단일 합성 개발셋·단일 seed 결과다. 직전 `screen_mix_50`은 로컬 개발셋에서 유망하지 않았고 공식 점수도 기존 최고보다 하락했다. 이번 결과는 후속 제출 후보로 검토할 근거지만 공식 개선 확정은 아니다. 자동 승격하지 않았으며 기존 모델·ZIP·selected·예약은 유지한다.

## 제출 후보 준비

검증된 `screen_mix_25.pt`를 사용자 보고 최고 제출의 로컬 기준 ZIP인 `artifacts/stage1-auto-release-20260910/candidate/submit.zip`에 적용해 `artifacts/stage1-quarter-submit-candidate-20260916/submit.zip`을 만들었다. Stage1 체크포인트만 교체했고 Stage2·Stage3·추론 코드·requirements 등 나머지5개 ZIP 멤버는 기준과 바이트 단위로 같다.

- 후보 ZIP: 177351283bytes
- 후보 SHA256: `3dec7b2caf918dd448e3c47aacc4a1db5215845bce5bdab15498498a1c212154`
- Stage1 체크포인트 SHA256: `49118651e9617c48216f5eff3baf3c5bb5f515a471a628f576d327e08db8e534`
- 검증: `local-verification.json`의 `STATIC_AND_MODEL_IDENTITY_PASS_GPU_PENDING`

모델 구조·전처리 메타·유한 가중치·ZIP 구조와 체크포인트 바이트 동일성은 PASS다. 이 정확한 후보 ZIP의 별도 GPU 통합검사는 아직 수행하지 않았고 대회 제출도 하지 않았다. 기존 최고 ZIP과 selected는 바꾸지 않았다.

## 공식 제출 결과

사용자 보고 기준 2026-09-16 13:36:56 평가 완료. Stage1 0.5868856347 / Stage2 0.2096390642 / Stage3 0.4187467089, 소요11분30초, 가중합0.36873143618이다. 제출 ZIP SHA256은 로컬 후보 `3dec7b2c...`와 연결된다.

직전 screen_mix_50의 Stage1 0.5597201095보다 +0.0271655252 회복했지만 기존 최고0.6386345895보다 -0.0517489548 낮다. 합성 개발 gate 통과가 공식 개선으로 이어지지 않았다. `screen_mix_25`는 미채택하며 기존 최고 모델·ZIP·selected를 유지한다. 별도 GPU 통합검사 미실행 기록보다 이번 실제 평가 완료 보고가 최신이다.
