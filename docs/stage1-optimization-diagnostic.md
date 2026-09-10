# Stage 1 작은 표본 쏠림 진단 (2026-09-10)

사용자가 Stage3은 별도 세션에서 진행하며 여기서는 Stage1 작은 표본 쏠림 진단을 요청했다.
기존 사전 학습은 6에폭이지만 optimizer updates는 24번에 불과했다.

## 로컬 확인

기존 train_stage1_full.py는 각 microbatch의 weighted mean CE를 따로 계산해 누적한다.
이는 누적 배치 전체의 가중치 합으로 나눈 CE와 일반적으로 같지 않다.
원본/재녹화 비율이 다른 microbatch에서는 의도한 클래스 보정 효과가 달라질 수 있다.
혼합 8표본 기울기 대조에서 수정식과 전체 배치의 최대 차이는 0,
기존 방식과의 최대 차이는 0.0319168195였다.
`artifacts/kaggle-stage1-optimization-20260910/normalization-test.json` PASS.
이 사실만으로 Stage1 쏠림의 원인이 확정되는 것은 아니다.

## GPU 대조 설계

기존 비공개 Stage1 trial Dataset을 그대로 재사용한다. 학습30영상의 고정 중앙 클립,
합성 검증30영상의 고정3구간. 같은 Kinetics-400 초기화·FP32, 규제 비활성,
4조건 각각 유효 배치8/microbatch2, 최대120 optimizer updates, 조건별500초 제한이다.

| 조건 | 표본 선택 | 손실 정규화 | 학습률 |
|---|---|---|---|
| uniform_legacy | 기존 1:2 집합에서 무작위 | microbatch별 weighted mean | 2e-5 |
| uniform_global | 위와 같은 표본 순서 | 누적 배치 가중치 합 | 2e-5 |
| balanced_low_lr | 매 배치 원본4/재녹화4 | unweighted mean | 2e-5 |
| balanced_high_lr | 위와 같은 균형 표본 순서 | unweighted mean | 1e-4 |

0/24/60/120 업데이트에서 학습·검증 로짓, 정답률, 클래스 분포와 손실을 저장한다.
입력 텐서·manifest·코드·초기 가중치 해시를 기록한다. 전체 실행은 무료 T4 1시간 이내.
첫 두 조건은 손실 정규화 차이 대조, 가운데 두 조건은 표본 균형+손실 보정 방식 대조,
마지막 두 조건은 학습률 대조다. 기존 사전 학습 전체 설정을 재현한 실험은 아니며
규제·클립·head warmup 차이를 구분해야 한다. 진단 가중치는 제출용으로 저장하지 않는다.

## 준비와 차단 상태

- 코드: src/diagnose_stage1_optimization.py
- 노트북: notebooks/Stage1_Optimization_Diagnostic.ipynb (code cell AST PASS)
- 준비 폴더: artifacts/kaggle-stage1-optimization-20260910/
- 새 비공개 Dataset 예정: biadis/crashintent-stage1-optimization-assets
- 새 비공개 Notebook 예정: biadis/crashintent-stage1-optimization-check
- Dataset 전송 자료: 진단 Python 코드·해시·메타데이터 총7,052 bytes. 새 영상은 전송하지 않는다.
- 기존 데이터 연결: biadis/crashintent-stage1-trial-v1-assets (이미 승인·업로드된 자료).

자동 승인 검토가 새 코드와 목적지 전송을 거절했다. 기존190MB 사전 번들 승인에
새 진단 코드·목적지는 포함되지 않는다는 사유다. 아직 업로드·GPU 실행하지 않았다.
사용자의 구체적 승인을 받은 뒤 준비된 Dataset create 및 kernel push를 실행한다.
Stage3 관련 파일이나 원격 작업은 변경하지 않았다. 전체 Stage1 합성 생성은 계속 진행 중이다.

## 사용자 승인 후 GPU 실행 (2026-09-10)

사용자가 바로 다음 메시지에서 진행을 명시적으로 승인했다. 기존 전송 차단은 해소됐다.
비공개 Dataset 업로드 완료 및 ready 확인, Notebook version 1 push 후 RUNNING 확인.
실행: https://www.kaggle.com/code/biadis/crashintent-stage1-optimization-check .
같은 범위의 전송·실행은 다시 승인 요청하지 않는다. 중복 실행하지 않는다.

결과 자동 회수·검증: `src/watch_stage1_optimization.py` 숨김 프로세스.
상태: `artifacts/kaggle-stage1-optimization-20260910/remote-run.json`.
로그: 같은 폴더 monitor.log / monitor-error.log.
예정 결과: result/stage1-optimization-result.zip 및 validated/result-validation.json.
독립 검사기 src/validate_stage1_optimization.py는 코드·manifest·입력 출처를 대조하고
저장 로짓에서 정확도·분포·CE·Macro-F1을 재계산하며 초기 예측 동일성과 120업데이트
완주, 균형 배치 노출 수, 실행 예산을 검사한다. 코드 문법 검사를 통과했다.
실제 결과 회수·검증 완료 여부는 위 상태 파일을 확인한다. 현재 결과는 아직 미확보다.
전체 합성 생성은 별도로 계속 진행한다.

## GPU 진단 완료 및 독립 검증 PASS (2026-09-10)

Notebook version1 COMPLETE, runner1504.29초(약25분), 4조건 모두120업데이트 완료.
자동 회수·독립 검증 PASS. 모든 조건에서 고정 학습30클립 정답률100%, Macro-F1=1.0.

| 조건 | 학습 F1 | 합성 검증 F1 | 검증 정답률 |
|---|---:|---:|---:|
| uniform_legacy | 1.0000 | 0.7778 | 0.8333 |
| uniform_global | 1.0000 | 0.5833 | 0.7333 |
| balanced_low_lr | 1.0000 | 0.7205 | 0.8000 |
| balanced_high_lr | 1.0000 | 0.8295 | 0.8667 |

균형 배치+lr1e-4가 이번 검증30영상에서 가장 높았다. 다만 검증 출처는2개뿐이고,
같은 검증 표본으로 조건을 비교했으므로 일반화 최적 설정이나 공식 점수로 확정하지 않는다.
기존 손실 누적 방식도 학습 표본 암기에 성공했다. 따라서 손실 정규화 차이만을 기존
쏠림 원인으로 단정할 수 없다. 기존 사전 학습과는 업데이트 수·규제·클립 고정·head warmup도
달라져 각 요인의 단독 인과 효과를 분리한 결과는 아니다.
다음은 전체 합성 완료 후 균형 배치와 충분한 업데이트를 본 학습 대조의 출발점으로 검토하고,
전체 독립 검증230원본에서 원본 클래스 재현율·Macro-F1을 확인하는 것이다.
진단 가중치는 저장하지 않았으며 본 학습 또는 제출 모델이 완성된 것은 아니다.
결과: artifacts/kaggle-stage1-optimization-20260910/validated/result-validation.json.
