# Stage3 움직임 모델 제출 후보 (2026-09-10)

상태: CPU·Tesla T4 GPU 통합검증 및 결과 로컬 독립검증 PASS. 대회 제출 안 함.

## 제출 파일

`artifacts/stage3-motion-submit-candidate-20260910/submit.zip` (177,368,840 bytes)

SHA256: `7f98769355146458abdd5d1a897de1cdc77b7f1706142c814ba6ff5bd90c3020`

기준 ZIP은 `artifacts/stage2-submit-candidate-20260910/submit.zip`.
Stage1·2 가중치와 requirements.txt는 바이트 단위 동일하다. 기존 함수·클래스 19개의 AST를 보존하고 predict_stage3만 교체했다. 제출 규격 6개 파일 검사 PASS.

Stage3는 고정 첫 seed 20260910의 움직임 MLP이며 검증 최고 seed를 고르지 않았다. 검증 조향 F1 .681430은 외부 센서 대리 평가로 공식 점수 개선을 보장하지 않는다.

## 검증

- 검증 18,603행의 원본 체크포인트 대비 예측 클래스 전부 일치. 로짓 허용 오차 검사 PASS.
- 실제 영상 2개 각 600행의 움직임 특징은 기존 캐시와 정확히 일치.
- 공개 OPEN_001 1,200행의 CPU 기준 예측 생성 완료.
- 자세한 기록: `artifacts/stage3-motion-submit-candidate-20260910/validation.json`.
- 구현: `src/stage3_motion_inference.py`, 패키징: `src/package_stage3_motion_candidate.py`.

## GPU 후속

준비 디렉터리: `artifacts/kaggle-stage3-motion-candidate-20260910/`.
비공개 데이터셋 `biadis/crashintent-stage2-candidate-assets`에 candidate.bin 177,368,840 bytes와 public-fixtures.bin 40,950,128 bytes 및 해시/메타데이터를 올리고, 비공개 커널 `biadis/crashintent-stage3-motion-check`에서 무료 T4로 세 단계 계약 및 CPU/GPU Stage3 예측 일치를 검사할 계획이다. 의존성 설치 후 추론 중 네트워크는 차단한다. 테스트 영상과 기준 CSV는 제출 ZIP에 포함하지 않는다.

자동 승인 검토가 해당 목적지로 payload를 전송하는 명시적 사용자 승인이 없다는 이유로 datasets version 실행을 거절했다. 업로드·GPU 커널 실행은 안 됐다. 사용자 승인 후 업로드하고 커널을 실행할 것. 기존 Stage1 예약 작업과 selected.json은 변경하지 않았다.


## 승인 후 GPU 검사 완료

사용자가 업로드·무료 T4 실행을 명시 승인했고, biadis/crashintent-stage3-motion-check v1 COMPLETE 및 로컬 독립검증 PASS를 확인했다. 이전 승인 대기 기록보다 이 완료 상태를 우선한다. 다운로드 DNS 오류는 같은 v1 결과 다운로드 재시도로 해결했으며 GPU를 재실행하지 않았다.

- 설치 165.50초, Stage1 5행/3.91초, Stage2 5행/5.33초, Stage3 1,200행/14.25초.
- Stage3 CPU 기준 CSV와 GPU 출력 1,200행 모두 일치.
- 정확한 ZIP·fixture 해시, 고정 의존성, 출력 계약, 추론 중 Python 인터넷 socket 차단 검증 PASS.
- 최종 증거: artifacts/kaggle-stage3-motion-candidate-20260910/result-validation.json.
- 제출 ZIP SHA256과 파일 크기는 위 값 그대로다. 실제 대회 제출 및 selected.json 변경 없음.

공개 예제 실행 검증이며 비공개 평가 전체 실행 시간이나 공식 점수 개선의 보장은 아니다.
