# Stage2 개선 후보 검증 (2026-09-10)

수동 라벨 기반 대조 실험의 저장 모델과 예측값을 독립 재검증했다.
학습 51영상 / 검증 15영상이며 검증 출처는 5개다. 출처·영상 ID 누수 없음과
원본 분할 CSV 해시 일치를 확인했다. 동일 개발 집합에서 후보를 선택했으므로
일반화 성능과 공식 Stage2 점수를 뜻하지 않는다.

| 후보 | 충돌 ±0.3초 | 진입 ±0.3초 | 회피 공간 정답률 | 방향 정답률 | 개발 평균 |
|---|---|---|---|---|---|
| baseline_5epoch (수동 라벨 대조군) | 0.4667 | 0.2667 | 0.4667 | 0.5333 | 0.4333 |
| longer_exact | 0.6000 | 0.4667 | 0.4000 | 0.8000 | 0.5667 |
| soft_mixed_scene (선택) | 0.6667 | 0.4000 | 0.4667 | 0.8667 | 0.6000 |

개발 평균은 위 네 항목의 단순 평균이며 공식 종합점수 산식과 구분한다.
선택 후보는 진입 시점 목표 분포를 완만하게 하고, scene 학습 때 일부 예측 시점을 사용한다.
최초 제출의 미학습 진입·회피·방향 헤드를 수동 라벨로 학습한 가중치로 교체하는 후보다.
비교표의 baseline_5epoch는 최초 제출 모델이 아니라 같은 실험 안의 수동 라벨 대조군이다.

## 검증 근거

- src/validate_stage2_controls.py: 예측 CSV의 ID·범위·범주와 지표 재계산, 저장 가중치 strict 로딩,
  모든 검증 캐시에서 예측 재현 및 실제 제출 ZIP에서 추출한 모델 클래스 출력 일치.
- ResNet 파일 해시는 직렬화 차이로 다르지만 모든 가중치 텐서는 정확히 일치한다.
- artifacts/stage2-controls-20260909/independent-validation.json: PASS.
- artifacts/stage2-controls-20260909/feature-spot-check.json: 검증 영상 000039의
  50프레임 특징을 원본에서 재추출했으며 캐시와 최대 절대 차이 0.0.
  나머지 영상은 기존 캐시를 사용했으며 전체 원본 재추출 검사는 아니다.

## 실행 기록 한계

과거 report.json은 COMPLETED이지만 관측 에폭은 5 / 25 / 17이고,
기록된 경과 시간은 20,513.062초다. 마지막 대조군의 25에폭 완료나 설정 시간 제한 준수를
인증할 수 없다. 원본 보고서는 보존한다. 저장된 선택 가중치와 지표 자체의 재검증은 통과했다.

## 제출 후보와 다음 단계

- ZIP: artifacts/stage2-submit-candidate-20260910/submit.zip
- SHA256: c0e8f2e3fc5847650e2e4a63f3e2b48e4058ad34edfbfe24a9a05333a4cf7047
- Stage2만 soft_mixed_scene으로 교체했고 Stage1·3 및 추론 코드는 기존 v3다.
- 정적 ZIP 검사 PASS. GPU 통합 검사와 실제 대회 제출은 미실행.
- 검사 노트북: notebooks/Stage2_Candidate_GPU_Check.ipynb
- 로컬 GPU 검사 입력: artifacts/kaggle-stage2-candidate-20260910/dataset
  (총 344,624,633 bytes, 후보 ZIP과 공개 예제 및 메타데이터).
- 설치 제한 600초, 추론 subprocess 제한 2400초. 공개 예제 3 Stage의 출력 계약과
  후보 해시를 확인하는 검사이며 비공개 전체 성능을 인증하지 않는다.
- 예정 비공개 Dataset: biadis/crashintent-stage2-candidate-assets.
- 예정 비공개 Notebook: biadis/crashintent-stage2-candidate-check.
- 외부 업로드는 자동 승인 검토에서 거절돼 실행하지 않았다. 이 후보와 공개 예제를
  해당 비공개 Kaggle Dataset에 전송하는 사용자 승인이 필요하다.
- 승인 후 업로드 → GPU 통합 검사 → 결과 회수·해시 대조를 수행한다.
  그 전에는 일일 배포의 selected.json을 바꾸지 않는다.

## Stage1 원격 상태 정정

로컬 Stage1 trial 메타데이터가 가리킨 biadis/crashintent-stage3-training-trial은
서버에서 과거 Stage3 결과를 제공했다. COMPLETE만으로 Stage1 학습 완료라 판단하면 안 된다.
잘못 시작한 다운로드는 중단했고 Stage1 학습 성공 결과는 아직 확보하지 않았다.

## 비공개 GPU 작업 승인 (2026-09-10)

사용자가 "해당 비공개 Kaggle Dataset에 업로드하고 GPU 검사 진행해. 앞으로는 일일이 묻지 않아도 돼"라고 명시적으로 승인했다.
준비한 후보·공개 예제의 biadis/crashintent-stage2-candidate-assets 비공개 업로드와 GPU 검사를 진행한다.
이후 같은 프로젝트 범위의 비공개 Kaggle 업로드·GPU 검사도 다시 묻지 않고 진행한다.
이 승인은 대회 실제 제출이나 유료 자원 사용에 대한 별도 승인으로 확대 해석하지 않는다.
앞 절의 업로드 승인 대기 상태는 이 승인 기록으로 해소됐다.

## GPU 통합 검사 완료 (2026-09-10)

사용자 명시적 승인 후 새 비공개 Dataset 생성과 T4 GPU 검사를 완료했다.
Dataset: https://www.kaggle.com/datasets/biadis/crashintent-stage2-candidate-assets
Notebook: https://www.kaggle.com/code/biadis/crashintent-stage2-candidate-check (version 1, COMPLETE).

- 결과: PASS. 후보 ZIP SHA256 c0e8f2e3fc5847650e2e4a63f3e2b48e4058ad34edfbfe24a9a05333a4cf7047 일치.
- 제출 requirements 6개 버전 일치. 설치 186.60초.
- Stage1 5행 / 4.20초, Stage2 5행 / 5.41초, Stage3 1200행 / 84.18초.
- 로컬에서 결과·예제 ZIP CRC, 입력 해시, ID·출력 컬럼·범주·프레임 범위 및 연속 인덱스 재검증 PASS.
- Python 인터넷 소켓 차단 상태 추론. 공개 예제 검사이며 비공개 성능·전체 실행시간 인증은 아니다.
- 로그에 Kaggle sitecustomize의 wrapt 누락 경고가 있었으나 subprocess와 검사 결과는 정상 완료했다.
- 결과 ZIP: artifacts/kaggle-stage2-candidate-20260910/result/candidate-check-sw2ikddl/candidate-integration-result.zip
- 독립 검증: artifacts/kaggle-stage2-candidate-20260910/result-validation.json
- 검증기: src/validate_candidate_gpu_result.py. 기존 PASS 결과의 재검증과
  다른 후보 해시를 가진 결과 거부 검사도 통과했다.
- 이번 작업은 GPU 추론 검사이며 신규 학습이나 실제 대회 제출은 수행하지 않았다.
- 앞 절의 GPU 미실행·승인 대기 상태는 이 완료 기록으로 대체한다.
- 일일 배포 selected.json은 이번 작업에서 변경하지 않았다.

## 병렬 GPU 학습 운영

Stage1·2·3은 별도 노트북·입력 manifest·출력 경로로 나누어 독립 실행할 수 있다.
2026-09-10 CLI quota 조회 시 GPU 6.16h 사용 / 23.84h 잔여 / 총 30h였다.
계정 동시 실행 개수 한도는 아직 직접 확인하지 않았다. 3개 동시 실행 가능으로 단정하지 않는다.
Stage3은 FP32 과적합 진단도 실패했으므로 기존 모델 장시간 재학습보다 짧은 원인 진단을 우선한다.
