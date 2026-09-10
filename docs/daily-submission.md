# 매일 제출 ZIP 준비

2026-09-09 사용자 요청에 따라 Windows 작업 `CrashIntent-DailySubmission-0830`을 등록했다.
매일 한국시간 08:30 실행하며 최초 정기 실행은 2026-09-10 08:30이다.
시험 실행의 `LastTaskResult`는 0이었다. 홈페이지 업로드는 수행하지 않는다.

- 실행 파일: `scripts/Prepare-DailySubmission.ps1`
- 배포 도구: `src/daily_submission.py`
- 검증된 후보 선택: `artifacts/submission-release/selected.json`
- 결과: `artifacts/daily-submissions/YYYY-MM-DD/submit.zip`, `report.json`, `README.txt`
- 최신 파일 위치: `artifacts/daily-submissions/latest.json`
- 실행 로그: `artifacts/daily-submissions/logs/`
- 예약 XML 사본: `artifacts/submission-release/scheduled-task.xml`

PC가 켜져 있고 KIMBEEN 사용자가 Windows에 로그인되어 있어야 한다. 절전 해제를 요청하며,
놓친 예약은 사용 가능해지면 실행한다. 전원이 꺼진 PC에서 정시 실행을 보장하지는 않는다.
배터리에서도 실행하며 중복 실행은 무시하고 실행시간은 15분으로 제한한다.

GPU 공개 예제 통합 검사 결과와 ZIP 해시가 일치하는 후보만 등록한다. 매일 CRC, 필수 6파일,
함수 인자, 크기 및 해시를 재검사한 후 원자적으로 사본을 완성한다. 변경된 후보가 같은 날
다시 등록되면 SHA256 앞 12자 하위 폴더를 사용해 이전 ZIP을 보존한다. 모델 개선이 없으면
직전 검증본을 다시 제공한다. 공개 예제 실행 통과는 모델 정확도 또는 비공개 전체 입력의
60분 추론 제한을 보증하지 않는다. 진단용 과적합 가중치는 배포 후보로 등록하지 않는다.

수동 준비:

```powershell
.\scripts\Prepare-DailySubmission.ps1
Get-ScheduledTaskInfo -TaskName CrashIntent-DailySubmission-0830
```

새 통합 후보가 GPU 검증을 마치면 다음 명령으로 교체한다.

```powershell
.\.venv\Scripts\python.exe src/daily_submission.py register --candidate artifacts/후보/submit.zip --evidence artifacts/검증/candidate-integration-result.zip --notes '구성과 성능 한계'
```

실제 대회 제출 및 유료 자원 사용은 사용자에게 별도로 확인한다.
