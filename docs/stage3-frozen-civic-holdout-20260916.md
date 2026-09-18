# Stage3 미사용 Civic 2경로 고정 비교 (2026-09-16)

저속 보정용으로 예약했지만 보정 gate 실패로 신호를 읽지 않았던 평가2경로를 기존 5~35m/s 센서 합의 평가에 한 번 사용한다. 경로는 `2018-05-02--16-01-39`, `2018-05-03--08-13-12`이며 기존 Civic 보정·평가 및 이번 보정3경로와 분리돼 있다.

정답을 생성하기 전에 `artifacts/stage3-frozen-civic-holdout-20260916/plan.json`을 고정했다. 기존 Civic 영점+0.011351°, 속도5~35m/s, 자세yaw범위, 직진/좌/우 각도·yaw 합의, centered11 지속 규칙을 그대로 쓴다. 최소 전체90시점·각 클래스20시점이 없으면 모델 비교 결과를 만들지 않는다. 비교 모델은 원본, 일반CE추가학습, 기존예측KL보존, 우회전손실1.25배의 각3시드 총12개이며 모든 체크포인트 SHA256을 계획에 고정했다.

채택 gate는 후보가 원본보다3시드 모두, 평가2경로 모두 F1이 높고 어떤 클래스 재현율도2%p 넘게 하락하지 않는 것이다. 통과해도 `FOLLOWUP_ONLY`이며 자동 모델 교체나 ZIP 생성은 하지 않는다. 센서 대리 정답이므로 공식 정답·사람 정답과 동일하다고 주장하지 않는다.

`src/evaluate_stage3_frozen_civic_holdout.py watch`를 30분 제한·재시도0·OS잠금의 숨김 감시기로 시작했다. watcher와 자식 추론 프로세스 및 실행 마커를 확인했다. 원본20Hz HEVC의 짝수 프레임을 무손실10Hz AVI로 만들고 공통402차원 흐름특징을 한 번 추출해12모델에 동일 적용한다. 아직 완료 결과는 없다. 상태는 `./scripts/Get-BackgroundWorkStatus.ps1`의 `Stage3FrozenCivicHoldout`을 한 번 조회한다. 완료 조건은 출력 폴더의 `watch-status.json` `COMPLETE`, `final-review.json`과 `status.json` `COMPLETE_VALIDATED`이다.
