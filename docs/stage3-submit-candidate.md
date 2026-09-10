# Stage3 확대 모델 제출 후보

2026-09-10 사용자 요청: 새 모델을 ZIP 후보에 넣고 GPU 통합검사.

후보: `artifacts/stage3-submit-candidate-20260910/submit.zip` (303597984 bytes).
SHA256: `cec269d3975db3d55a734505e40a3c03d8c43775e02fc887c2cfeba132964337`.

기존 Stage2 개선 후보에서 `model/stage3/best.pt`만 1000클립 확대 학습 마지막 체크포인트로 교체했다. 나머지 ZIP 파일은 바이트 동일 확인. 원본 결과 ZIP과 체크포인트 SHA256 일치 및 정적 ZIP 검사 PASS.

새 Stage3 외부 검증 F1: 가감속0.461828 / 조향0.409353. 공식 제출 점수는 아직 없다.

GPU 검사 자료: `artifacts/kaggle-stage3-candidate-20260910/`.
기존 비공개 Dataset `biadis/crashintent-stage2-candidate-assets`에 새 입력 버전 업로드. 예정 Notebook `biadis/crashintent-stage3-candidate-check`.
검사 범위: 후보 해시·고정 requirements 설치(600초 제한)·인터넷 Python소켓 차단 후 Stage1/2/3 공개 예제 추론(2400초 제한)·반환 CSV 계약.

현재 정적 검사 완료, GPU 검사 준비 및 업로드 중. 실행 완료 후 `src/watch_stage3_candidate.py`가 결과 회수 및 `src/validate_candidate_gpu_result.py` 재검증 예정.
대회 실제 제출 및 일일 selected.json 변경은 하지 않았다.


## GPU 통합 및 독립 재검증 완료

2026-09-10 `biadis/crashintent-stage3-candidate-check` version1 COMPLETE. 반환 ZIP 다운로드와 로컬 독립 재검증 PASS.

- 정확한 후보 SHA256 일치: cec269d3975db3d55a734505e40a3c03d8c43775e02fc887c2cfeba132964337.
- 제출 requirements6개 버전 일치. 설치145.50초.
- Stage1 5행/3.97초, Stage2 5행/5.26초, Stage3 1200행/82.41초.
- 공개 예제 ID·컬럼·범주·프레임 범위·연속 시점 및 입력 해시 재검증 PASS.
- 추론 중 Python 인터넷소켓 차단. 공개 예제 실행 검사이며 비공개 전체 점수와 실행시간은 별도.
- 결과 보고서: `artifacts/kaggle-stage3-candidate-20260910/result-validation.json`.
- 반환 ZIP SHA256: 572ef3f22c2469860d3d9e4277f9d206ec935872b2643b2e119fa1608f8255b0.

앞 절의 GPU 대기 상태는 이 완료 결과로 대체한다. 제출 후보 ZIP 준비 완료이며 대회 실제 제출은 하지 않았다.
