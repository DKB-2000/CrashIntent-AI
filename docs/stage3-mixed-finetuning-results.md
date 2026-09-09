# Stage3 혼합 배치 추가 학습 결과 (2026-09-09)

추가3에폭(누적2~4) 실행과 결과 검증을 완료했다. 표본 혼합과 추가 학습 후에도 검증 예측의
단일 클래스 쏠림이 남았다. 성능용 모델 확보로 처리하지 않는다.

## 설정·검증

기존 첫 에폭 가중치에서 AdamW를 새로 시작했다. 영상8개 그룹 안에서 표본을 섞었으며,
156개 학습/31개 검증, 라벨v1, lr1e-4, batch2, stride8은 유지했다.
각 에폭5,850스텝, 합계17,550스텝. T4에서 재로딩 검증 포함13,324.88초(3시간42분5초),
최고 GPU 할당2.73GiB였다. 이 시간은 패키지 설치와 데이터 준비를 제외한 runner 측정이다.

ZIP CRC, 원본 번들·입력해시·18,603개 예측의 ID/인덱스/범주/누락/중복을 검사했다.
원본 라벨로 F1과 confusion matrix를 재계산해 원격 결과와 일치했다. GT STOPPED1,257개를
제외한17,346개로 조향을 평가했다. finetune manifest와 초기 체크포인트 및 코드의 SHA256,
실행 설정과 에폭2/3/4의 스텝 수를 확인했다. 최적 모델 strict CPU 로딩, 유한 가중치,
검증 지표·입력해시·stride·buffer 메타데이터 검사 PASS다.

## 성능

| 누적 에폭 | 평균 학습 손실 | 가감속 Macro-F1 | 주행 중 조향 Macro-F1 |
|---|---:|---:|---:|
| 1 (이전) | — | 0.069475 | 0.221897 |
| 2 | 2.160289 | 0.180613 | 0.140887 |
| 3 | 2.142766 | 0.180613 | 0.221897 |
| 4 | 2.171014 | 0.180613 | 0.221897 |

조향 F1 기준 최적은 epoch3이다. epoch4는 동점이며 strict improvement 저장 규칙에 따라
epoch3 가중치가 유지됐다. 최적 모델의 가감속 예측18,603개가 모두 CONSTANT,
주행 중 조향17,346개가 모두 STRAIGHT다. 가감속 CONSTANT F1은0.722453이며 다른3개는0,
조향 STRAIGHT F1은0.665692이며 LEFT/RIGHT는0이다.

가감속 F1은 이전보다 높지만 학습 최빈 클래스 CONSTANT 고정 출력과 같다.
조향 F1은 이전과 같고 STRAIGHT 고정 출력과 같다. 따라서 감속 쏠림이 일정속도 쏠림으로
바뀌었을 뿐, 검증 집합에서 구분 능력이 확보됐다는 증거는 없다.

이번 실행은 배치 혼합·추가 에폭·optimizer 재시작을 함께 적용했으므로 혼합만의 인과 효과를
분리할 수 없다. 데이터는 센서 기반 임시 라벨이며 이 결과는 대회 성능이 아니다.

## 다음 작업

새 대규모 학습에 앞서 클래스가 섞인 작은 고정 학습 표본의 과적합 검사를 한다.
입력별 로짓 차이, 두 헤드와 백본의 gradient/가중치 변화, train/eval 모드 차이와
학습 표본 자체의 정확도를 확인한다. 이를 통해 학습 흐름 문제와 일반화 문제를 구분한다.
현재 결과만으로 데이터 추가나 에폭 증가가 문제를 해결한다고 가정하지 않는다.

## 파일

- Notebook: https://www.kaggle.com/code/biadis/crashintent-stage3-mixed-finetuning
- 결과 검사: `artifacts/kaggle-stage3-mixed-training/result-validation.json`
- ZIP: `artifacts/kaggle-stage3-mixed-training/result/stage3-gpu-ks8b27_v/stage3-training-result.zip`
- 모델: `artifacts/kaggle-stage3-mixed-training/result/stage3-gpu-ks8b27_v/training-run/model/stage3/best.pt`
- 모델크기:137,100,468 bytes
- 모델 SHA256: `fb8cdf83fccf0fad6fddaad4f05ead8e7e83fc51d67d59934320432fedd10607`
