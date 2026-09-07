# Stage3 comma2k19 조향 보정 v1 (2026-09-07)

## 결과

- 원본 187개 영상과 기존 라벨을 보존하고 `artifacts/stage3-comma-chunk1-calibrated-v1/`에 실험용 라벨을 생성했다.
- 조향 영점 −0.2454875도, 직진 범위 ±1.5도, 양수 LEFT. 공식 정답 기준을 확정한 값은 아니다.
- seed 20260825, route 단위 학습 17개/156영상/93,602행, 검증 4개/31영상/18,603행. route 중복 없음.
- 가감속은 기존 속도·미분 기반 라벨을 유지한다. 본 학습과 제출은 아직 실행하지 않았다.

## 추정 근거와 한계

공식 좌표계·단위: https://github.com/commaai/comma2k19#log-format
IMU는 forward/right/down, gyro rad/s, CAN 조향각 degree다. Pose 위치·속도는 ECEF다.
이 좌표계에서 gyro의 down 축 부호를 반전해 좌회전 양수 신호를 만들었다.
별도 비교 신호로 수평 ECEF 속도와 시간 미분의 외적을 local up에 투영해 회전속도를 구했다.
이는 프로젝트의 파생 계산이며 대회 공식 라벨 생성식이 아니다.

학습 route의 유효한 이동 표본 7,902개(1초 간격)에서 조향각을 회전속도/속도와 회전속도×속도에
강건 회귀했다. IMU 추정 절편은 −0.2455도, pose 추정은 −0.2596도다.
학습 route bootstrap 100회 절편 구간은 약 [−0.4746, −0.0080]도다. 이 구간은 route 민감도이며
물리적 보정 정확도를 보증하지 않는다. 검증 route는 영점 추정과 임계값 선택에 쓰지 않았다.

학습 데이터의 진단만으로, 근직진 pose 표본의 STRAIGHT 비율 ≥90%와 뚜렷한 회전 방향 일치율 ≥97%를
동시에 만족하는 가장 작은 시험 임계값을 선택했다. 이 기준은 실험용 공학적 절충안이다.

| 직진 범위 | 근직진 pose 표본의 STRAIGHT 비율 | 뚜렷한 회전의 방향 일치율 |
|---|---:|---:|
| ±0.5도 | 58.24% | 99.60% |
| ±1.0도 | 85.28% | 99.06% |
| **±1.5도 (v1)** | **93.94%** | **97.87%** |
| ±2.0도 | 97.42% | 95.23% |

이 수치는 센서 proxy 일치율이며 정확도·Macro-F1·대회 점수가 아니다. 작은 조향이나 차선 변경을
넓은 직진 범위가 지울 수 있으므로 실제 모델 오류 분석에서 다시 검토한다. 같은 RAV4·고속도로 구간의
여러 route이므로 이 분할만으로 차량·지역 일반화를 검증한 것은 아니다.

## 검수 및 검증

- 최초 비교 자료: `artifacts/stage3-comma-chunk1-calibration/review.html`.
  7유형×2개, 총 14클립의 각 4프레임 요약(56프레임)을 어시스턴트가 확인했다.
  직진·좌우 곡선·정지·가감속·보정 불일치 표본이다. 연속 영상의 사람 검수와 구분한다.
- v1 비교 자료: `artifacts/stage3-comma-chunk1-calibrated-v1/review.html`.
  동일 설정으로 14개×8초 영상을 재생성했고 직진 및 새 불일치 표본 3개를 추가 확인했다.
  거의 직진인 표본에서 불필요한 방향 라벨이 줄었지만, 미세 경계 사례는 여전히 존재한다.
- 재생 가능성: v1 14클립의 전체 1,120프레임 디코딩 PASS.
- 후보 CSV 컬럼·연속 인덱스·split·참조 영상·보정 파라미터와 라벨 일치 검증 PASS.
- 강건 회귀의 알려진 영점 복원, 식별 불가능한 데이터 거부, 임계값 선택 조건,
  부호·경계값 테스트 4개 PASS.

## 파일 사용

- `label_version.json`: 보정값, 선택 근거, 분할별 라벨 분포, 한계.
- `labels.csv`: 전체 v1, 공식 baseline 학습 컬럼 `ID,frame_index,accel_label,steer_label`.
- `labels_train_candidate.csv`, `labels_validation_candidate.csv`: 분할별 v1.
- `split_manifest.csv`: ID, route, split, 원본 변환 MP4 상대경로. 상대경로 기준은 manifest 폴더다.
- `signals_and_candidates.csv`: 원래 라벨과 보정 라벨, IMU·pose 진단 신호.
- `visual_review.csv`: 사람이 연속 영상 판정을 기록할 빈 항목을 포함한다.
- `bundle_validation.json`: 검증 결과.

영상은 복제하지 않았으며 baseline의 평탄한 videos 폴더를 생성하지 않았다. 다음 모델 파이프라인은
manifest의 영상 경로를 사용하거나 학습 환경에서 입력 구조를 준비해야 한다.
학습 입력에는 영상만 사용하고 CAN·IMU·pose는 외부 학습 라벨 구성과 진단에만 사용한다.

## 재현

새 출력 폴더로 실행한다. 생성기는 기존 결과 덮어쓰기를 거부한다.

```powershell
.\.venv\Scripts\python.exe src/review_stage3_comma_calibration.py --archive data_raw/comma2k19/Chunk_1.http.zip --dataset-dir artifacts/stage3-comma-chunk1 --output-dir artifacts/stage3-comma-chunk1-calibration
.\.venv\Scripts\python.exe src/finalize_stage3_comma_calibration.py --calibration-dir artifacts/stage3-comma-chunk1-calibration --dataset-dir artifacts/stage3-comma-chunk1 --output-dir artifacts/stage3-comma-chunk1-calibrated-v1
.\.venv\Scripts\python.exe src/test_stage3_comma_calibration.py
.\.venv\Scripts\python.exe src/validate_stage3_comma_calibration.py --review-dir artifacts/stage3-comma-chunk1-calibrated-v1
```

다음 작업은 v1과 route 분할을 사용하는 Stage3 학습·추론 파이프라인 및 GPU 스모크 준비다.