# Stage3 최적화 진단 준비 (2026-09-10)

사용자 요청에 따라 Stage2 점수 대기는 별도로 두고 Stage3 쏠림 진단을 우선한다.
기존 FP32 실험의 동일 학습 10클립을 사용한다. 평가 데이터·새 외부 데이터는 사용하지 않는다.

| 조건 | 초기화 | 유효 배치 | 학습률 | 최대 optimizer steps |
|---|---|---|---|---|
| scratch_b2_lr1e4 | 무작위 | 2 | 1e-4 | 120 |
| scratch_b10_lr1e4 | 무작위 | 10 | 1e-4 | 120 |
| scratch_b10_lr3e5 | 무작위 | 10 | 3e-5 | 120 |
| trained_b10_lr3e5 | 기존 쏠림 모델 | 10 | 3e-5 | 120 |

모든 조건은 FP32, dropout/stochastic depth/weight decay 비활성이다.
전체 배치는 microbatch 2로 누적하되 가감속은 10개, 조향은 실제 moving 표본 수로 나눠
완전한 전체 배치 손실과 같은 기울기를 만든다. 불균등 정지 표본과 전부 정지한 배치의
기울기·손실 동등성 테스트 2개 PASS. 동일 step과 동일 표본 노출 수는 다르므로
samples_seen도 저장해 비교하며 큰 배치의 개선을 원인 확정으로 단정하지 않는다.

각 조건 최대 600초, 전체 진단은 노트북 잔여 예산 내 최대 약 2440초다.
설치 timeout 600초, 학습 subprocess timeout 최대 2500초 및 전체 시작 시각 기반 잔여 예산을 적용한다.
모든 표본 정답 시 조기 종료한다. 진단 가중치는 제출용으로 저장하지 않는다.
학습 데이터는 /tmp에 풀고 반환 보고서 ZIP만 /kaggle/working에 저장한다.

- 코드: src/diagnose_stage3_optimization.py
- 기울기 검사: src/test_stage3_optimization.py
- 노트북: notebooks/Stage3_Optimization_Diagnostic.ipynb
- 로컬 준비: artifacts/kaggle-stage3-optimization-20260910/
- 예정 비공개 Dataset: biadis/crashintent-stage3-optimization-assets
- 예정 비공개 Notebook: biadis/crashintent-stage3-optimization-check

자동 승인 검토가 새 비공개 Dataset에 프로젝트 소스 코드를 보내는 작업을 거절했다.
기존 사용자 포괄 승인만으로는 새 목적지와 파일에 대한 구체적 승인이 없다는 이유다.
이후 사용자가 해당 새 비공개 Dataset 업로드와 GPU 실험을 명시적으로 승인했다. 업로드·실행을 재개한다.

로컬 원본 입력 검증 PASS: 동일 학습 10클립, 서로 다른 입력, 유한한 텐서 및 causal16 형태 확인. input SHA256 94bbd0d63cab8d1d4d26e4339da4b3b729e8beccdf772099990242cc74a4ed8f. artifacts/kaggle-stage3-optimization-20260910/input-validation.json 참고.

## GPU 실험 완료·독립 검증 PASS (2026-09-10)

사용자 승인 후 비공개 Dataset 생성과 Notebook version 1 실행을 완료했다.
Notebook: https://www.kaggle.com/code/biadis/crashintent-stage3-optimization-check
상태 COMPLETE. 전체 runner 1310.35초(약 21분 50초), 진단 1091.77초, 설치 147.71초.
T4 / torch 2.8.0+cu128, 각 조건 120 optimizer steps. 시간 제한 중단 없이 네 조건 완료.

| 조건 | 본 표본 수 | 가감속 정답률 | 주행 중 조향 정답률 | 최종 손실 |
|---|---|---|---|---|
| scratch_b2_lr1e4 | 240 | 80% | 33.3% | 2.064134 |
| scratch_b10_lr1e4 | 1200 | 100% | 100% | 0.086278 |
| scratch_b10_lr3e5 | 1200 | 100% | 100% | 0.014658 |
| trained_b10_lr3e5 | 1200 | 40% | 33.3% | 2.247634 |

두 scratch 배치10 조건 모두 과적합 성공(MEMORIZED). 기존 쏠림 모델은 동일 배치·학습률
3e-5·표본 노출 수에서도 과적합 실패했다. 새 초기화와 큰 유효 배치 조건에서 입력과
두 정답을 학습할 수 있음을 실증했다. 학습 불가능한 구조나 모든 입력이 같은 버그라는
설명은 지지되지 않는다. 이전 전체 학습의 쏠림 원인을 하나로 확정한 것은 아니다.

동일 표본 노출 수 200에서 비교하면 scratch 배치2(100스텝)는 손실 3.743919 /
가감속 30% / 조향 33.3%, scratch 배치10 lr1e-4(20스텝)는 손실 1.743921 /
가감속 60% / 조향 44.4%였다. 최종 성능 차이 전체를 배치 크기의 단독 효과로
해석하지 않는다. 큰 배치와 작은 배치는 업데이트 횟수·표본 순서·노출 구성도 다르다.

### 독립 검증

- 결과 ZIP CRC PASS.
- 입력 텐서 SHA256이 로컬 원본 10클립과 정확히 일치:
  94bbd0d63cab8d1d4d26e4339da4b3b729e8beccdf772099990242cc74a4ed8f.
- 표본 목록·학습 라벨 SHA256·초기 체크포인트 SHA256·실험 코드 manifest 대조 PASS.
- 저장 로짓에서 argmax·정답률·CrossEntropy 합을 독립 재계산해 모든 측정 시점과 일치.
- 본 표본 수, 마지막 측정 시점, MEMORIZED·COMPLETED 상태의 조건 검사 PASS.
- 원본 결과: artifacts/kaggle-stage3-optimization-20260910/result/stage3-optimization-result.zip
- 검증 보고서: artifacts/kaggle-stage3-optimization-20260910/result-validation.json
- 검증기: src/validate_stage3_optimization.py

### 다음 성능 작업

1. 기존 쏠림 체크포인트를 이어 학습하는 접근은 우선순위를 낮춘다.
2. 새 초기화·FP32·유효 배치10 이상을 기본으로, 학습률 3e-5를 다음 짧은 대조의 출발점으로 사용한다.
   이번 손실이 더 낮았다는 이유만으로 일반화에 가장 좋은 학습률이라고 확정하지 않는다.
3. 여러 학습 route의 표본을 섞은 작은 실제 학습 집합과 독립 검증 route로
   조향 Macro-F1·예측 분포를 확인한다. 학습/검증 양쪽에서 단일 클래스 쏠림 해소 여부를 본다.
4. 기존 comma2k19 주행 중 조향 최빈 클래스 기준 F1 0.221897보다 개선되는지 같은 검증 분할에서 비교한다.
5. 작은 표본 과적합 성공을 공식 Stage3 점수 개선으로 간주하지 않는다.
   진단 가중치는 제출용으로 저장하지 않았고, 제출 ZIP·일일 배포 후보는 변경하지 않았다.
