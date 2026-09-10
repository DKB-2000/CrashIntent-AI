# 첫 제출 후보

후보 파일: `artifacts/first-submit-candidate-v3/submit.zip` (303,672,145 bytes).
SHA256: `69dd0a448b4e0a3f17e24692d9ca01c10dd23d5e09c3462440e773ae957d7581`.
최초 1회 제출은 사용자 보고 기준 평가 완료다(2026-09-10 결과 반영). 표시 시각 2026-09-09 17:51:28, 점수는 Stage1 0.4045598914 / Stage2 0.159801139 / Stage3 0.1656369753, 평가 소요 시간은 13분 54초다. 사용자가 Stage1·2·3 순서를 확인했으며 로컬 계산 종합점수는 0.211087224다. `submission-history.md` 참고. GPU 통합 검증 결과는 아래 최신 기록을 따른다.

## 구성과 한계

- Stage1: 기존 공식 베이스라인 스모크 가중치·추론 코드.
- Stage2: 기존 공식 베이스라인 가중치·추론 코드. 진입·회피 공간·진입 방향 헤드는
  별도 정답으로 학습되지 않은 상태다.
- Stage3: 혼합 추가 학습의 최적epoch3 가중치, 로컬 causal16 추론 코드.
  외부 검증에서 CONSTANT/STRAIGHT만 출력하며 성능은 최빈 클래스 기준과 같다.
- requirements는 공식 베이스라인의6개 고정 버전이다.

이 후보는 실행 점수를 확인할 수 있는 첫 통합물이다. 경쟁력 있는 성능을 확보했다는 뜻은 아니다.
진단에서 잠시 학습한10개 표본용 가중치는 포함하지 않았다.

## 발견·수정한 제출 오류

1. 공식 공개 Stage3 OPEN_001.mp4는1200프레임이며 OpenCV FPS 헤더는 약479.7793이다.
   Stage3 의미상 시간축은 sample_index 단위로 처리하므로 추론에서 FPS 헤더10 여부를 강제하면
   공개 예제를 거부한다. 추론은 모든 프레임을 순서대로 처리하고0부터 연속 인덱스를 출력한다.
   학습 데이터의10fps 검사는 유지했다.
2. 공식 추론 노트북은 각 함수에 MODEL_DIR/stage1, stage2, stage3를 전달한다.
   Stage3 공개 함수는 model_dir/best.pt를 우선 사용하도록 수정했다.
   기존 내부 스모크의 model root 호출도 호환된다.

회귀 테스트10개 PASS. ZIP CRC·필수6파일·AST·공개 함수 시그니처·실제 import 및
기존 Phase0 정적 검사 PASS. 공개 예제 GPU 검사는 Stage1 영상5개, Stage2 영상5개에서
전체 프레임 추출, Stage3 영상1개1200프레임을 사용한다. 이는 전체 비공개 입력의60분 제한
인증이나 성능 검증이 아니다. 추론 프로세스의 Python socket 연결을 막아 실행한다.

## 재현

```powershell
.\.venv\Scripts\python.exe src/build_first_submission.py --baseline Baseline/submit.zip --stage3-checkpoint artifacts/kaggle-stage3-mixed-training/result/stage3-gpu-ks8b27_v/training-run/model/stage3/best.pt --output artifacts/first-submit-candidate-new/submit.zip
```

- GPU 검사 Notebook: https://www.kaggle.com/code/biadis/crashintent-first-candidate-check
- 로컬 Notebook: `notebooks/First_Candidate_GPU_Check.ipynb`
- 비공개 검증 입력: `biadis/crashintent-first-candidate-assets` version3.
- 원래 후보v1/v2는 중간 산출물이다. 실제 사용할 파일은v3이며 GPU 결과와 해시 일치를 확인한다.
- 다음 성능 진단: `stage3-overfit-diagnostic.md` 참고.


## GPU 통합 결과: PASS

Notebook version3 COMPLETE. 최종 후보v3 SHA256과 반환 manifest가 일치하고 결과 ZIP CRC,
Stage1 5행, Stage2 5행, Stage3 1200행 검사 PASS다. 원격 실행에서 출력 컬럼·ID 집합·중복·누락·
범주 및 Stage2 프레임 범위·Stage3 연속 인덱스를 확인했다.

Tesla T4 / Python3.12.13 / 제출 requirements6개 버전 환경이다. 패키지 설치128.09초,
Stage1 3.85초 / Stage2 5.48초 / Stage3 80.89초였다. 공개 예제 소량 실행이며 비공개 전체
데이터의60분 제한 충족을 보증하지 않는다. 추론 중 Python AF_INET/AF_INET6 연결은 차단했고,
DataLoader가 사용하는 로컬 AF_UNIX 통신은 허용했다. 완전한 OS 네트워크 격리 검사는 아니다.

초기 통합 검사에서 모델 폴더 전달 오류를 수정했다. 다음 실행은 검사 코드의 과도한 socket
차단이 로컬 DataLoader 통신을 막아 실패했고, 인터넷 소켓만 차단하도록 검사 코드를 고쳐 통과했다.
최종 후보 파일 자체는v3이며 마지막 검사에서는 후보 ZIP 변경 없이 같은 파일을 검증했다.

- 통합 보고서: `artifacts/kaggle-first-candidate/result-validation.json`
- 원격 결과: `artifacts/kaggle-first-candidate/result/candidate-check-rmggep5t/candidate-integration-result.zip`
- 공개 프레임의 standalone/integrated Stage3 CPU 로짓 정확 일치:
  `artifacts/first-submit-candidate-v3/local-equivalence.json`
- 최종 상태: 공개 예제 GPU 실행을 통과한 첫 제출 후보. 사용자 보고 기준 최초 1회 제출 평가 완료, 점수 3개와 소요 시간 기록 완료. 사용자가 Stage1·2·3 순서를 확인했고 로컬 계산 종합점수는 0.211087224다.

## 통합 GPU 검증 완료 (2026-09-09)

v3의 반환 `candidate-integration-result.zip`에서 PASS, 설치 시간, 3개 Stage 실행,
오프라인 socket 차단 기록과 ZIP SHA256 일치를 확인했다. 검증된 SHA256은 위 v3 값과 같다.
근거: `artifacts/kaggle-first-candidate/current-result/candidate-check-rmggep5t/candidate-integration-result.zip`.
일일 배포 대상으로 등록했으며 `docs/daily-submission.md`에 운영 방법을 기록했다.
