# Stage2 화면 영역 특징 시험 제출 후보 (2026-09-16)

사용자 요청으로 기존 공식 최고 조합 ZIP에서 Stage2만 화면 영역 특징 후보로 교체했다.
사용자가 실제 데이콘 시험 제출을 완료했고 공식 개선 결과를 보고했다.

## 구성

- 기준 ZIP: `artifacts/stage1-auto-release-20260910/candidate/submit.zip`
  - SHA256 `2c0090adb06f272c0833cfb151ed479ed9c78273830f724e0439fef220626197`
  - 사용자 보고 최고 제출87406에 대응하는 조합: Stage1 `.6386345895`,
    Stage2 `.2096390642`, Stage3 `.4187467089`.
- Stage1·Stage3 가중치 및 `requirements.txt`: 기준 ZIP과 바이트 동일.
- Stage2 시점 모델: 기준 ZIP의 공식 최고 가중치를 그대로 보존.
- Stage2 상황 판단: 충돌·진입 예측 프레임의 하단 좌·우·중앙 crop을 같은
  frozen ResNet18에 통과시킨 특징과 GRU hidden 두 시점을 결합한 선형 probe 5개 평균.
- probe 전체학습: 기존 수동66영상·23출처, seed
  `20260825/26/27/20260916/17`, 각80epoch. 최종66개 학습 적합도는 회피·방향
  모두100%로 독립 성능이 아니며 과적합 위험을 명시한다.

## 결과 파일과 검증

- ZIP: `artifacts/stage2-semantic-crop-submit-candidate-20260916/submit.zip`
- 크기: `177585722` bytes
- SHA256: `401eade6940a64c80eecbffd7365562143b957ffa2792ef218b3de89fb475f9b`
- 변경 ZIP 항목은 `model/stage2/best.pt`, `inference.py` 두 개뿐이다.
- ZIP 구조·CRC·필수 함수 정적 검사 PASS.
- 40개 실제 JPEG를 사용한 패키지 `predict_stage2` CPU 스모크 PASS.
- 출력 컬럼·프레임 범위·범주 검사 PASS.
- GPU 통합 검사는 미실행. CPU 스모크는 검증기에서만 DataLoader worker0과
  CPU device로 실행했으며 제출 코드의 CUDA/DataLoader 설정은 유지했다.
- 생성 당시 상태는 `STATIC_AND_CPU_SMOKE_PASS_GPU_PENDING`, `submitted:false`였으나,
  이후 사용자 제출·평가 완료 보고가 최신이다.

## 공식 시험 제출 결과

- 제출명: `2026_09_16_002 edit`
- 표시 시각: 2026-09-16 16:47:25
- Stage1 / Stage2 / Stage3: `0.6386345895 / 0.21955508 / 0.4187467089`
- 평가 시간: 12분12초
- Stage2 기존 최고 대비 `+0.0099160158`(상대 약4.73%)
- 가중합: `0.38304763346`, 이전 최고 대비 `+0.00396640632`
- 판정: **공식 개선 확인, 새 최고 조합으로 채택**

재현 코드는 `src/package_stage2_semantic_crop_candidate.py`, 독립 검증은
`src/validate_stage2_semantic_crop_candidate.py`. 상세 파일은 같은 후보 폴더의
`validation.json`, `independent-validation.json`을 사용한다.

## 시험 제출 해석

5fold 개발에서 crop 조건은 5seed 모두 기존보다 방향·개발평균·출처동일가중이
높았지만 같은66개를 반복 사용했다. 공식 Stage2 점수가 기존 `.2096390642`보다
높아야 개선 근거가 된다. 낮거나 같으면 기존 모델을 유지한다. Stage2 외 점수가
달라지면 동일 구성이라는 전제부터 재점검한다. 서버 업로드 ZIP 해시가 제공되지
않으면 제출과 로컬 파일 연결은 사용자 확인에 의존한다.
