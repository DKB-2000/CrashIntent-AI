# Stage1 로컬 검사 자동 복구 (2026-09-11)

Windows 작업 `CrashIntent-Stage1LocalEvaluation-20260911` 등록 및 실행 완료.
이 작업은 로그인된 사용자 권한으로 2분마다 검사 상태를 확인한다. 정상 평가 중에는 중복 실행하지 않는다.
별도 `CrashIntent-Stage1AutoRelease-20260910`의 **12시간 주기와 다음 실행 시각은 유지**한다.
로컬 감시기는 업로드나 대회 제출을 실행하지 않는다.

- 새 평가 프로세스는 `--max-new-videos 25 --threads 1`로 실행한다. 저장 완료된 영상은 fsync 후 상태에 기록한다.
- 25개 후 저장 결과의 출처·슬롯 출력·확률·판정을 검증하고 PAUSED로 종료한다. 다음 프로세스는 데이터/모델/서빙 코드 해시와 정확한 순서의 저장 prefix를 검증한 뒤 이어간다.
- 전체 357개를 검증하고 조건별 결과를 재계산한 경우에만 PASS 보고서를 쓴다. 부분 검사는 제출 승격 근거가 아니다.
- 감시기와 평가 출력 각각 OS 파일 잠금을 사용한다. Windows venv 실행기 및 실제 Python 자식 프로세스를 함께 관리한다.
- 프로세스 종료 또는 15분 동안 상태 갱신이 없으면 해당 평가만 종료·복구한다. 기존 평가를 멈출 때 PID 생성 시각과 작업 경로, 자식 프로세스를 확인한다.
- 자동 복구는 총 5회 실패 또는 같은 완료 개수에서 2회 실패 시 NEEDS_ATTENTION으로 중단한다. Python 검증 오류·손상된 JSONL은 반복 재시도하지 않는다. 파일 손상을 임의로 잘라내지 않는다.
- 실행 중 자동 유휴 절전을 방지하고 종료 시 해제한다. 수동 절전·로그아웃·전원 종료 중에는 실행되지 않는다. 다시 로그인된 실행 기회에 저장분에서 재개한다.
- 전체 PASS 후 제출 자동화의 독립 검증을 다시 수행하고, 검사 실패 때문에 닫힌 gate만 복구한다. 다른 실패나 다른 세션의 등록 결과는 덮어쓰지 않는다. 이후 ZIP 단계는 기존 12시간 예약이 담당한다.

## 확인 결과

기존 Stage1 테스트 32개 PASS 후 Windows 실제 venv 자식 종료·관계 없는 자식 종료 차단 테스트를 추가하여 감시기 테스트 10개 PASS.
실제 소형 영상 추론에서 native access violation 발생 기록을 보존한 뒤 저장 1개를 검증해 재개했다.
새 PID에서 1→3개 PAUSED, 다시 다른 PID에서 3→5개 PAUSED를 확인했다. 각 묶음은 신규 2개로 제한했고 부분 PASS 보고서는 생성되지 않았다.
이는 자동화 동작 검증이며 17개 소형 세트 전체 성능 검사 완료를 뜻하지 않는다.
본검사는 기존 복구기 종료 후 새 Windows 예약이 265개 저장분부터 자동 인계했다.
Windows 자식 프로세스 종료 보강을 적용하기 위한 한 차례 계획 재시작에서도 저장분을 보존했다.

## 확인 경로

- `src/supervise_stage1_robustness.py`, `src/evaluate_stage1_robustness.py`
- `scripts/Register-Stage1LocalEvaluationTask.ps1`, `scripts/Supervise-Stage1Robustness.ps1`
- `artifacts/stage1-robustness-20260910/full-model-evaluation/supervisor-status.json`
- 같은 폴더 `result/status.json`, `chunk-*.log`, `supervisor-logs`, `supervisor-task.xml`
- `artifacts/stage1-supervisor-smoke-20260911b/status.json`, `resume-*.json`

NEEDS_ATTENTION이면 supervisor-status의 오류와 해당 로그를 확인한다. 원인 해결·저장분 검증 없이 상태나 실패 횟수를 초기화하지 않는다.
등록 작업은 Interactive 권한이므로 로그아웃 상태의 무인 실행은 보장하지 않는다. 화면 잠금 상태는 로그인 상태다.

## 349/357 native 충돌 점검 후 복구 (2026-09-11)

사용자 오류 점검·복구 요청으로 recovery-audit-20260911.json 생성(PASS).
모든 입력 해시/학습 분리 및 저장349개 ordered prefix·슬롯logit·softmax·평균·판정,
모델/서빙코드 일치 재검증. 남은8영상 각50프레임 총400프레임 정상 디코딩.
실패 지점은341→345→346→347→349로 달라졌고 마지막 traceback은
Torchvision MViT `_add_rel_pos`의 텐서 연산, Windows native 0xc0000005다.
점검 시 가용물리메모리394320KB(약385MiB)/총약15.7GiB, 가용가상메모리약3GiB.
메모리 압박은 관측했지만 충돌의 근본 원인으로 확정하지 않는다. WHEA 이벤트 조회는 결과 없음.

모델 load_state_dict 이후 필요 없는 직렬화 바이트/원본 checkpoint 텐서를 해제해
추론 시 메모리 보유량을 줄였다. 모델 파라미터·입력·정밀도·임계값은 그대로다.
이번 복구 supervisor 설정 max_new_videos=1: 매 영상 새 프로세스.
기존5회 실패 control은 supervisor-before-single-video-*.json에 보존하고,
검증된349개 SHA256을 재확인한 뒤 사용자 요청에 따른 복구 회차를 시작했다.
총5회/동일지점2회 제한 유지, 과거실패는 prior_recovery_failures=5로 명시.
11개 감시기 테스트 PASS(1개 단위 명령·잘못된 설정 차단 포함).
Windows 기존 로컬 예약을 즉시 시작. ZIP12시간 예약 불변.
이 절은 복구 시작 기록이며 완료 판정은 result/report.json 및 supervisor-status.json을 따른다.
