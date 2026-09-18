# Stage1 화면 단서 보강 GPU 대조학습

2026-09-14 사용자 진행 요청. 검증된765개 학습전용 보강 영상을 이용한다.
현재 단계와 완료 여부는 `artifacts/kaggle-stage1-screen-trial-20260914/remote-run.json`을 확인한다.

## 사전 고정 비교

3조건: baseline / mixed_50 / screen_mix_50.
baseline은 현 공식최고 Stage1 모델, 두 학습 조건은 그 동일 초기모델로부터 각각 시작한다.
기존 85학습출처·seed20260911·128업데이트·lr1e-5·FP32 AdamW·유효배치8·microbatch1 유지.
동일 source/profile/slot 추출, 원본4/재녹화4의 균형배치, optimizer 재초기화.
총256업데이트와3모델×420영상=1260예측을 계획한다. 마지막 모델만 평가한다.

mixed_50은 매 업데이트 기존합성2출처쌍+화질균형2출처쌍이다.
screen_mix_50은 동일8영상 중 화질균형쪽 RERECORDED2개만 보강영상으로 교체한다.
ORIGINAL4개와 기존합성 RERECORDED2개는 경로·슬롯까지 동일하다.
보강cue는 weak_screen/no_geometry/no_flicker를 `(step+pair//2)%3`으로 사전 순환한다.
노이즈·블러·JPEG profile과 원본source·slot을 유지한다. 새 학습량이나 원본 비율 변화는 없다.
총256개의 보강 입력이 사용되며,765개 pool 전체를 반드시 한 번씩 학습하는 설계는 아니다.

평가는 기존 357조건진단+normal63개 개발셋이다. 고정0.5 임계값·3slot 평균·CUDA FP16.
기준: baseline 대비 noise/quality_mix 원본 오류 각각 감소,
각 stress 재녹화 조건 오류증가<=1/21,normal F1 감소<=0.02.
기존 mixed_50과 비교한 조건별 변화도 함께 보고한다. 검사PASS는 성능조건PASS와 구분한다.
반복 분석한 동일개발출처·단일seed이므로 공식일반화로 확정하거나 자동승격하지 않는다.

## 자료와 실행 구조

기존 private quality-trial-assets의 영상·초기모델을 재사용한다.
새765영상과 runner/plan/manifest를 담은 증분 screen.bin을
비공개 `biadis/crashintent-stage1-screen-trial-assets`로 전송할 계획이다.
정확한 전송용량·SHA256은 bundle.json의 upload_bytes/upload_sha256을 따른다.
무료T4 비공개 notebook: `biadis/crashintent-stage1-screen-trial`.
노트북은 원래자료·증분자료 해시를 확인하고 합친다. 로컬 quality.bin은 그 최종 내용의 검증용이다.
새데이터의 전수검증 상태,manifest·plan·각영상해시·effect recipe·출처분할을 준비 때 재확인한다.

실행코드: prepare_stage1_screen_trial.py / launch_stage1_screen_trial.py.
공용 stage1_quality_trial_runner.py와 validate_stage1_quality_trial.py는 기존실험과 새3조건을 지원한다.
매업데이트 실제영상경로와slot을 결과에 저장하고 독립검사기가 일정을 재계산한다.

단일upload/push 시도마커와 OS잠금으로 중복을 막는다. 불확실한 전송/실행은 재시도하지 않는다.
업로드최대1800초,ready대기1시간,원격결과대기4시간,조회오류5회/다운로드5회/검사900초 제한.
노트북설치600초/runner10200초 제한. 완료/실패/로그·결과회수·검증은 기존유한감시기에 연결한다.
최종 증거는 validated/result-validation.json PASS와 remote-run COMPLETE/VALIDATED다.
모델·ZIP·selected·기존예약을 변경하지 않으며 대회제출은 하지 않는다.

## 준비 검증 (2026-09-14)

표본교체2개/배치·클래스균형·이전일정동등성·출처/조건손상거절 포함8테스트 PASS.
cue사용횟수는weak85/geometry86/flicker85로 사전고정됐다.
증분 screen.bin 956113828bytes(약956MB),SHA256
`21ff2f9b6462491eb56e41985879076a824563ce8d9e693e3bf8b2ed1adefa9a`.
로컬통합번들2501157827bytes,SHA256
`98efada2c74865ad14693293a07784332876b579c27fdfd9073bc2d93037907b`.
증분768파일(영상765+code/config3)의 로컬통합번들 내용일치 및 notebook/launcher 구문 PASS.
학습pool1530영상/평가420영상. 과거자료는기존private dataset에서재사용하고 새증분만전송한다.

직전mixed-trial 전체결과도 새검사기로 재검증PASS해 기존4조건 동작을 확인했다.
자동승인검토가 새목적지/956MB보강영상·코드 payload에 대한 사용자구체적명시승인 부족으로
숨김감시기시작 명령을 실행전에 거절했다. upload-attempt/launch-attempt 모두없음,
외부전송/GPU/감시기 미실행. remote-run BLOCKED_APPROVAL 및 approval-required.json 기록.
동일시도 재실행금지. 구체적사용자승인후 upload-approval.json에 approved=true 및
dataset/kernel/upload_sha256/notebook_sha256/metadata_sha256를 bundle과 일치하게 기록한 뒤
launch_stage1_screen_trial.py를 실행한다. 준비된자료는보존,기존모델/ZIP/예약변경없음.

## 구체적 승인 후 실행 연결 (2026-09-14 17:40)

사용자가 약956MB 보강영상·코드/새private dataset/무료T4 확인에 "진행해"로 명시 승인했다.
upload-approval.json에 dataset/kernel/자료·notebook·metadata 해시를 기록했다.
기존 upload/launch-attempt 부재 확인 후 UTF-8 숨김 launcher PID26660을 시작했다.
launcher가 해시 재검사→단일 업로드→dataset ready→단일push→회수·독립검증을 수행한다.
이전 승인 차단은 해소됐다. 실제 단계는 remote-run.json을 확인하며 중복 실행하지 않는다.
현재모델/ZIP/예약은 변경하지 않았다. 시작 기록은 GPU 완료나 성능개선 결과가 아니다.
