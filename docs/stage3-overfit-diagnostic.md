# Stage3 소량 학습 진단

학습 split에서 10개 고정 클립을 선택했다. 가감속3개×조향3개 조합9개와 STOPPED1개다.
초기 체크포인트 해시와 각 표본 ID·프레임·라벨을 원본 학습 CSV로 대조했고 일치했다.
모델에 들어가는 클립은 기존 causal16 전처리를 사용한다. 이는 일반화 성능 검사가 아니다.

| 초기화 | 180스텝 후 가감속 정답률 | 주행 중 조향 정답률 | 손실 변화 |
|---|---:|---:|---|
| 기존 추가학습 모델 | 40% | 33.3% | 2.6024 → 2.3635 |
| 무작위 초기화 | 40% | 55.6% | 2.8760 → 2.0832 |

기존 모델의 초기 입력별 로짓 표준편차는 대부분0, 최대 약0.000039였다.
서로 다른 영상에서 거의 같은 출력을 내는 현상을 고정 표본에서도 확인했다.
180스텝 후에는 입력별 로짓 차이가 생겼지만 표본을 충분히 외우지는 못했다.
무작위 초기화 모델도 180스텝 내 과적합을 달성하지 못했다. 이 제한된 실행만으로
학습 불가능이나 코드 오류의 원인을 확정하지 않는다.

두 실행 모두 추적한 백본 첫 파라미터와 가감속·조향 헤드 가중치가 변했다.
첫 스텝 헤드 gradient norm에 Infinity가 기록됐으며 마지막 스텝은 유한했다.
초기 mixed-precision overflow 여부를 분리하려면 FP32 대조와 scaler/skip 횟수 기록이 필요하다.
현재 JSON에는 이 비유한 측정값이 Python JSON의 Infinity로 보존돼 있다.

다음 진단은 같은10개 표본에서 FP32 및 dropout/stochastic-depth를 끈 대조 실행으로
수치 안정성과 규제 영향을 구분하는 것이다. 작은 표본 학습 성공 전에 장시간 재학습을 반복하지 않는다.
진단 모델 가중치는 제출 후보에 사용하지 않았다.

- Notebook: https://www.kaggle.com/code/biadis/crashintent-stage3-overfit-diagnostic
- 검증된 원본 기록: `artifacts/kaggle-stage3-diagnostic/validated-diagnostic.json`
- 코드: `src/diagnose_stage3_overfit.py`

## FP32 대조 결과 검증 (2026-09-10)

기존 반환 ZIP을 학습 라벨·체크포인트 SHA256·예측 정답률과 독립 대조했다.
CRC PASS, 10개 표본은 모두 학습 split에 속하고 원본 라벨과 일치한다.
torch 2.8.0+cu128 / Tesla T4에서 3조건 각 240 optimizer steps, 진단 실행 503.36초다.

| 초기화 | 규제 | 최종 가감속 정답률 | 주행 중 조향 정답률 |
|---|---|---|---|
| 무작위 | 사용 | 10% | 44.4% |
| 무작위 | dropout·stochastic depth·weight decay 비활성 | 70% | 33.3% |
| 기존 모델 | 동일 규제 비활성 | 40% | 33.3% |

모든 조건에서 추적 가중치가 변했고 비유한 step은 0으로 기록됐다.
FP32와 규제 비활성만으로 과적합 문제가 해소되지 않았다. 혼합정밀도 오류가
유일한 원인이라는 설명은 이 결과로 뒷받침되지 않는다.
정지 표본을 제외하면 조향 학습이 특히 부족하다. 이 결과는 일반화 평가가 아니다.

- 원본: artifacts/kaggle-stage3-precision/result/stage3-gpu-z5wilo6z/stage3-diagnostic-result.zip
- 독립 검증: artifacts/kaggle-stage3-precision/independent-validation.json
- 재검증: artifacts/kaggle-stage3-precision/validate_local.py
- 다음: 작은 고정 표본에서 학습률·유효 배치 크기 대조와 영상 움직임 특징 대조를 검증한다.
  원인이 확인되기 전에 기존 MViT 전체 장시간 재학습을 반복하지 않는다.

## 후속 배치·학습률 진단에서 과적합 성공 (2026-09-10)

같은10개 학습 클립에서 FP32·규제 비활성·새 초기화·유효 배치10을 적용한 두 조건
(lr1e-4 / 3e-5)이 120 optimizer steps에 가감속·주행 중 조향 정답률100%를 달성했다.
기존 쏠림 모델은 배치10·lr3e-5에서도40%/33.3%에 머물렀다. 원본 입력 해시와
모든 측정 로짓·손실·정답률의 로컬 재검증PASS. 상세 stage3-optimization-diagnostic.md 참고.
다음은 새 초기화·큰 유효 배치 조건의 독립 route 검증이다. 공식 성능 개선은 아직 미검증이다.
