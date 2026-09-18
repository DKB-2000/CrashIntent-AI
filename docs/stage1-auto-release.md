# Stage1 본학습 모델 자동 제출 ZIP 연결

2026-09-10 사용자 요청으로 모델 회수·강건성 검증 이후의 자동 패키징·GPU 통합검사·
일일 후보 등록을 구현했다. **사용자 명시적 승인 후 16:30 KST 예약 등록·첫 실행 성공(LastTaskResult=0)을 확인했다.**
기존 일일 08:30 작업과 Stage1 생성/학습/강건성 검증 프로세스는 별도로 계속 동작한다.

## 준비된 작업

- Windows 작업명: CrashIntent-Stage1AutoRelease-20260910
- src/advance_stage1_release.py: 1회 실행마다 한 단계 진행, 상태는 JSON으로 보존.
- scripts/Advance-Stage1Release.ps1: 숨김 실행·로그.
- scripts/Register-Stage1ReleaseTask.ps1: 등록 12시간 뒤부터 12시간 간격으로 계속 반복, 로그인·즉시 실행 없음,
  중복 실행 무시, 1회 최대45분, 놓친 실행 재개·절전 해제 요청.
- artifacts/stage1-auto-release-20260910/config.json: 경로·모델 기준·해시 고정.
- status.json, logs/, upload.log, push.log, gpu-validation.json, release-report.json.
- 대회 홈페이지 업로드는 포함하지 않는다.

## 자동 연결 순서

1. 기존 본학습 validated/result-validation.json PASS 및 최소120업데이트,
   동일 합성검증에서 baseline보다 높은 Macro-F1, 양 클래스 출력 확인.
2. 357영상 강건성 평가 ALL_CONDITIONS/PASS, 동일 체크포인트 해시 확인.
   모든 입력 해시와 저장 로짓·확률·범주·조건별 지표를 다시 대조한다.
3. GPU 검증된 Stage2 개선 후보(c0e8f2e3fc5847650e2e4a63f3e2b48e4058ad34edfbfe24a9a05333a4cf7047)
   안의 model/stage1/best.pt만 본학습 모델로 교체한다. 다른5파일은 바이트 동일 검사.
   이 기준은 최근 Stage3 하락 후보를 자동 승격하지 않기 위한 고정 선택이다.
4. 아래 전용 비공개 Kaggle Dataset으로 자료 업로드, ready 확인 후 무료 T4 노트북 1회 실행.
5. 고정 requirements 설치(최대600초), 추론 중 인터넷 Python socket 차단,
   세 Stage 공개 예제 통합검사(최대2400초).
6. 정확한 노트북 버전의 결과를 회수하고 기존 validate_candidate_gpu_result.py로 재검증.
7. selected.json 등록과 날짜별 submit.zip 생성을 잠금 안에서 수행.
   기존 일일08:30 예약은 이후 이 등록 후보를 사용한다.

최종 ZIP: artifacts/stage1-auto-release-20260910/candidate/submit.zip 및
artifacts/daily-submissions/YYYY-MM-DD/submit.zip.
같은 날짜에 다른 ZIP이 있으면 SHA256 하위 폴더를 사용한다.
최종 정확한 경로는 release-report.json 또는 artifacts/daily-submissions/latest.json.

## 동시 작업·실패 처리

- 등록 대기 중 다른 세션이 selected.json을 바꾸면 덮어쓰지 않고 NEEDS_ATTENTION으로 중단.
- daily_submission.py의 register/publish에 OS 파일 잠금을 추가했으며 기존 CLI는 유지.
- 업로드·GPU push 직전에 상태를 먼저 저장한다. 실행 결과가 불확실한 중단 후 동일 원격 변경을
  무조건 재시도하지 않는다. UPLOAD_STARTED/PUSH_STARTED는 상태 조사 후 재개가 필요하다.
- 읽기 전용 원격 상태 조회/결과 회수 실패는 최대15회 후 NEEDS_ATTENTION으로 기록한다.
- COMPLETED/NEEDS_ATTENTION이면 후속 예약 호출은 상태를 덮어쓰지 않고 종료한다.
- 이 후속 예약은 선행 본학습/영상생성/CPU평가 프로세스 자체의 재부팅 복구를 수행하지 않는다.
  기존 프로세스가 계속 동작하도록 PC와 로그인 상태를 유지해야 한다.
- 강건성 지표는 진단 결과이며 임의 오탐률 임계값으로 성능을 인증하지 않는다.
  CPU FP32 진단과 CUDA FP16 공개예제 통합 PASS는 실제 재촬영·공식 점수 개선 보증이 아니다.

## 전송 자료와 자동 승인 차단

준비된 전송:
- 새 Stage1 가중치를 넣은 제출 후보(세 모델, inference.py, requirements.txt).
- 기존 공개 예제 public-fixtures.bin: 40,946,139bytes.
- 위 파일의 해시 manifest. 예상 합계 약344.6MB(새 체크포인트 압축률에 따라 변동).
- 노트북 코드 및 실행 메타데이터.

예정 목적지:
- 비공개 Dataset: biadis/crashintent-stage1-release-20260910-assets
- 비공개 Notebook: biadis/crashintent-stage1-release-20260910
- 무료 Tesla T4. 추가 학습이 아닌 통합 추론 검사, 설치·추론 합산 최대50분.

자동 승인 검토가 예약 등록/시작 명령을 거절했다:
“사용자가 해당 민감 payload와 Kaggle 목적지를 명시적으로 승인하지 않았다.”
등록 명령 전체가 실행 전 거절됐으므로 작업은 아직 없다. 새 원격 업로드/GPU 실행도 없다.
위 자료·목적지·예약 범위에 대한 사용자 승인을 받은 뒤 아래를 실행한다.

```powershell
./scripts/Register-Stage1ReleaseTask.ps1
Get-ScheduledTaskInfo -TaskName CrashIntent-Stage1AutoRelease-20260910
```

등록 직후 상태가 WAITING_FOR_VALIDATION이고 LastTaskResult=0인지 확인한다.
최초 실행 전후 selected.json이 바뀌지 않았고 GPU가 조기에 시작되지 않았는지도 확인한다.

## 로컬 검사

6테스트 PASS: Stage1만 교체, 미완료 검증의 원격 실행 금지, 업로드/push 불확실 상태의
중복 실행 금지, 다른 세션 후보 보존, 등록/복사 중단 후 재개, 스모크 성능자료 승격 거절.
별도 실제 체크포인트 패키징 스모크는 artifacts/stage1-auto-release-smoke-20260910/에
보관하며 제출/일일 후보 등록 대상으로 사용하지 않는다.

실제 체크포인트 패키징 스모크 완료:
Stage1 사전학습 모델 strict 로딩·유한값 검사·ZIP 구조/CRC/해시 검사와 나머지5파일 바이트 동일 PASS.
별도 스모크 SHA256 8f6c5aed90e71c18d8aa27465c3aa66b5acdb937b6109f14aa42f62f7b62fa78.
artifacts/stage1-auto-release-smoke-20260910/validation.json.
예약 조회 Registered=false 확인. 자동 승인 차단 상태에서는 직접 tick 호출도 작업을 시작하지 않는다.


## 사용자 승인 후 등록 완료 (2026-09-10 16:30 KST)

사용자가 앞서 제시한 약345MB 전송 자료·비공개 Kaggle 목적지·무료T4·자동ZIP등록 범위를
명시적으로 승인했다. 이전 승인 차단은 해소됐으며 위 같은 범위 재승인 불필요.
Windows 작업 CrashIntent-Stage1AutoRelease-20260910 실제 등록 완료:
Enabled/Ready, 첫 실행16:30:03, LastTaskResult=0.
첫 예약16:31:03부터 2분 간격으로24시간 반복하며 로그인 트리거도 등록돼 있다.
현재 WAITING_FOR_VALIDATION, 기존 본학습 검증과357영상 강건성 평가를 기다린다.
기존 selected.json 해시 불변 확인. 아직 새 모델 전송·GPU 통합검사는 시작하지 않았다.
최종 ZIP은 검사 통과 후 즉시 생성·일일 후보 등록되며 대회 사이트 제출은 하지 않는다.


## 2026-09-11 검증 시간초과 복구 및 기준 후보 갱신

357영상 검증이78영상에서 시간제한으로 중단돼 예약은 NEEDS_ATTENTION에 멈췄다.
로컬 본검사 재개중이며 PASS일 때만 기존 실패 gate를 다시 연다. 중복 GPU 실행 없음.
그동안 Stage3 움직임 모델의 공식점수 .4187467089가 보고돼, 이전 Stage3로 회귀하지 않도록
아직 미업로드인 config의 기준 ZIP을 검증된 stage3-motion-submit-candidate-20260910으로 갱신했다.
SHA256 7f98769355146458abdd5d1a897de1cdc77b7f1706142c814ba6ff5bd90c3020.
실제 GPU증거 PASS·ZIP해시 확인, 기존설정은 config-before-20260911-recovery.json에 보존.
새 후보는 이 ZIP의 Stage1만 교체한다. 전송목적지·승인범위·selected.json은 변경하지 않았다.

## 예약 주기 변경 완료 (2026-09-11)

사용자 요청으로 실제 Windows 작업 CrashIntent-Stage1AutoRelease-20260910을
2분에서 12시간 간격으로 변경하고 로그인 트리거를 제거했다.
다음 실행은 2026-09-11 21:02:33 KST, 이후 12시간 반복(만료 없음).
실제 XML에서 PT12H, 트리거 1개, 실행 대상 불변을 확인했고 등록 스크립트 구문 검사도 통과했다.
등록 스크립트는 등록 12시간 뒤부터 반복하며 즉시 실행하지 않는다. 기존 08:30 예약은 유지한다.
위 2분 간격 기록보다 이 변경을 우선한다. 한 번에 한 단계만 진행하므로 완료까지 여러 주기가 필요할 수 있다.
