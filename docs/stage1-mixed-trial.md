# Stage1 기존 합성·화질 균형 혼합 대조학습

2026-09-14 사용자 진행 요청. 화질 보강의 원본 오탐 개선을 유지하면서 재녹화 탐지 손실을
줄이는지 확인한다. 기존 실패분석의 7원본·검증21출처는 학습에 넣지 않는다.

## 고정 실험

4조건: 현재 모델 baseline / existing_recipe / matched_quality / mixed_50.
각 학습 조건은 동일 공식최고 Stage1 체크포인트 `92b10c99...`, seed20260911,
85학습출처·128업데이트·lr1e-5·FP32 AdamW·규제비활성으로 시작한다.
유효배치8(원본4/재녹화4), microbatch1, 원본·profile·slot 일정은 이전 실험과 같다.
대조군을 같은 GPU 실행에서 다시 학습·평가해 비교한다. 총384업데이트/1680영상예측이다.

mixed_50은 매 업데이트 4출처쌍 중 2쌍을 기존합성, 2쌍을 화질균형에서 가져온다.
쌍마다 ORIGINAL/RERECORDED 및 slot을 함께 유지하며 recipe 위치를 step마다 교대한다.
한 업데이트의 recipe별 비율과 클래스별 비율이 정확히 50%다.
효과강도 변경이나 새 증강은 추가하지 않아 혼합의 효과를 우선 분리한다.
모든 학습 step에 실제 선택한 영상 경로·slot을 기록하고 결과 검사기에서 재계산한다.

평가는 기존420영상(357조건진단+normal63), 고정0.5 임계값·3slot·CUDA FP16이다.
판정 기준은 이전과 동일: noise와quality_mix 원본 오류 각각 감소,
각 stress 재녹화 조건 오류 증가 <=1/21, normal Macro-F1 감소 <=0.02.
최종128업데이트 모델만 평가하고 검증점수로 에폭/임계값을 선택하지 않는다.
단일seed·반복 사용한 개발출처이므로 통과해도 독립 일반화 개선이나 자동 승격이 아니다.

## 실행 자료와 검증

로컬 `artifacts/kaggle-stage1-mixed-trial-20260914/`.
기존 비공개 `biadis/crashintent-stage1-quality-trial-assets`의 1.55GB 자료를 재사용한다.
새 비공개 무료T4 노트북 `biadis/crashintent-stage1-mixed-trial`에 전송하는 것은
코드·설정·manifest를 포함한 notebook/metadata 합계601754bytes다. 영상/모델 재업로드는 없다.
원래 bundle SHA256 `d31d0d1e...`를 확인하고 notebook이 runner/plan/manifest만 덮어쓴다.
원격 덮어쓰기 내용과 로컬 검증용 quality.bin의 해당3파일 바이트일치 PASS.
로컬 검증번들 SHA256 `20f969472a88e1f46a88e6f96b1599405b62e64aa2ace786f18738fd8aca3931`.
기존 입력 member 해시·출처분리 재검사 및 배치/이전표본일정/손상거절6테스트 PASS.

코드: prepare_stage1_mixed_trial.py, watch_stage1_mixed_trial.py,
stage1_quality_trial_runner.py 및 validate_stage1_quality_trial.py.
기존3조건 실행과 예전결과 검증을 지원한다. 원격결과 ZIP과 검증은 새 출력폴더로 분리한다.

## 감시 및 완료 조건

watch_stage1_mixed_trial.py --launch는 OS잠금과 launch-attempt.json으로 단일push를 보장한다.
push 응답이 불확실하거나 실패하면 자동으로 재push하지 않는다.
기존 감시기를 연결해 상태조회오류5회, 다운로드5회, 결과대기4시간, 검사900초로 제한한다.
노트북 설치600초/runner10200초 상한. 이전3조건 약36분36초를 기준으로 이번4조건은
약50~60분 예상이나 GPU 대기·설치 시간에 따라 달라진다.
완료는 remote-run.json의 COMPLETE/VALIDATED와 validated/result-validation.json PASS로 확인한다.
현재모델·ZIP·selected·기존12시간예약은 변경하지 않는다. 대회 제출은 하지 않는다.

## 준비 완료·새 목적지 전송 승인 대기

기존 실험 전체 결과의 수정 검사기 회귀 검증도 PASS했다.
숨김 감시기 시작 명령은 자동 승인 검토에서 실행 전에 거절됐다.
사유: 새 Kaggle 노트북 목적지와 코드·설정 payload에 대한 사용자 명시 승인 증거 부족.
따라서 노트북 전송·GPU 시작·감시기 시작은 모두 미실행이다.
remote-run.json BLOCKED_APPROVAL, approval-required.json에 정확한 자료/목적지를 보존한다.
동일 시도를 반복하지 않는다. 사용자가 명시적으로 승인하면 upload-approval.json에
approved=true, kernel, notebook_sha256, metadata_sha256 일치를 기록하고 launch한다.
코드/설정 합계601754bytes, 목적지는 비공개 biadis/crashintent-stage1-mixed-trial이며
기존 비공개 quality-trial-assets의 영상·모델을 재사용하는 무료T4 대조학습이다.

## 명시 승인 후 실행 연결 (2026-09-14)

사용자가 위 전송자료·목적지·무료T4 확인에 "실행해"로 명시 승인했다.
upload-approval.json에 notebook/metadata 해시와 승인범위를 기록하고 두 파일 해시를 재확인했다.
숨김 Python 감시기 PID28368을 시작했다. 이전 승인 차단은 해소됐으며
remote-run.json 및 push.log에서 실제 제출·GPU 시작 여부를 확인한다.
감시기가 단일push 후 완료·실패/결과회수·독립검증을 처리한다. 중복 실행하지 않는다.

최초 감시기는 승인파일의 한국어를 cp949로 읽다가 전송 전에 종료됐다.
UTF-8 명시 읽기로 수정하고 launch-attempt.json 부재를 확인한 뒤
`python -X utf8`로 재개했다(PID20076). 초기 오류로그는
monitor-error-encoding-20260914.log에 보존했으며 중복push는 없었다.

09:35 확인: kernel version1 push 성공, 원격 `KernelWorkerStatus.QUEUED`.
watcher PID22812(venv 부모20076), 로컬 SUBMITTED/WAITING 및 오류없음 확인.
아직 실제 GPU RUNNING/실험 완료 상태가 아니다. 이후 감시는 기존 프로세스에 맡긴다.

## 완료 결과 (2026-09-14 10:38)

Kaggle v1 COMPLETE, 결과 회수 및 독립검증 PASS. 4조건1680예측/384업데이트,
실제 학습 표본·slot 일정, 코드/입력/모델 해시와 저장 로짓 기반 지표 검증을 통과했다.
remote-run.json COMPLETE/VALIDATED, 최종 증거는 같은 실험폴더의
validated/result-validation.json이다. 앞선 대기·실행 중 기록보다 이 완료 결과를 우선한다.

| 항목 | baseline | existing_recipe | matched_quality | mixed_50 |
|---|---:|---:|---:|---:|
| 노이즈 원본 오류 /21 | 19 | 19 | 0 | 0 |
| 복합화질 원본 오류 /21 | 18 | 20 | 2 | 9 |
| 화면 단서 약화 재녹화 오류 /21 | 0 | 0 | 6 | 3 |
| normal63 Macro-F1 | 1.000000 | 1.000000 | 0.982348 | 1.000000 |

혼합은 화질균형 단독 대비 약한 재녹화 오류6→3과 normal F1을 회복했지만,
복합화질 원본 오류2→9로 일부 오탐 개선을 잃었다. baseline 대비 원본 두조건 개선과
normal F1 유지 기준은 통과하나 재녹화 각조건 오류증가 <=1/21 기준은 실패한다.
mixed_50도 promising_pilot=false이며 현재 제출모델로 승격하지 않는다.
단일seed·기존개발셋 결과이고 실제 재촬영/공식점수 개선 증거가 아니다.
기존모델·ZIP·selected·예약은 변경하지 않았다. 추가 GPU 실행은 시작하지 않았다.
