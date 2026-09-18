# Stage3 공간별 움직임 독립 실험

2026-09-10 사용자 요청 1번. 공식 제출86809 움직임 모델을 보존한 로컬 대조 실험이다. 제출ZIP, selected.json, 외부 업로드는 변경하지 않는다.

## 사전 고정 설계

- 기존 균형1000 학습클립, 별도4경로31영상18603 검증시점, 3seed(20260910/11/12), 동일1682→128→64 dual-head MLP.
- 동일 초기화·균형배치10·12에폭1200업데이트·AdamW lr .001·weight_decay0·FP32 CPU2스레드. 고정 마지막 체크포인트만 평가한다.
- 학습 표본으로만 정규화한다. 표준편차 하한 .01, 정규화 후 [-10,10] clip. 정지 정답은 조향손실/조향F1에서 제외한다.
- `baseline`: 기존402차원 움직임특징 그대로. 기존 이미지1280차원은 mask0. 원래3개 seed 모델과 가중치·정규화·전로짓 정확 일치를 확인한다.
- `common_residual`: 현재/최근5평균/최근15평균의 각8×8 xy흐름에서 공간 median xy를 빼고, median6차원을 원래 비활성 이미지입력 위치에 추가한다. 기존6개 흐름통계는 유지한다(활성408차원).
- `regional_augmented`: 기존402차원은 유지하고, 각 시간블록의 median xy 및 상/하/좌/우/중앙4×4 영역 residual의 mean xy·크기평균·크기표준편차를 추가한다(66추가, 활성468차원).

정규화 이후 활성차원수가 다르지만 전체 모델 파라미터 수와 초기화는 같다. 실험의 개입은 입력 특징만이다. 각 조건의 계획·입력 캐시·원본기준모델·표본·코드 해시는 `artifacts/stage3-spatial-motion-20260910/plan.json`에 사전 기록했다.

## 범위와 한계

기존 영상에서 추출한8×8 평균 flow grid를 재사용한다. dense flow를 복원하지 않았으며, spatial median은 공통 평행이동의 대리 특징이다. 실제 카메라 흔들림/자차 이동/주변차량 움직임의 물리적 분리를 보장하지 않는다. 시간평균 grid의 median과 프레임별 median의 시간평균도 다르다. 속도 센서는 모델입력에 쓰지 않는다.

검증은 이미 반복 사용한 comma2k19 개발 경로의 센서 대리 라벨이다. 공식 점수나 신규 출처·사람 검증의 개선으로 해석하지 않는다. 가감속F1은 진단이며 공식Stage3 핵심 지표는 주행중 조향F1이다.

## 재현

```powershell
.\.venv\Scripts\python.exe src/stage3_spatial_motion_experiment.py prepare
.\.venv\Scripts\python.exe src/stage3_spatial_motion_experiment.py train
.\.venv\Scripts\python.exe src/stage3_spatial_motion_experiment.py verify
```

기존 plan/완료 run 덮어쓰기는 거절한다. 결과 CSV 형식은 `ID,sample_index,accel_label,steer_label`이다.

## 완료 결과

9개 학습 및 독립 검증 PASS. 아래 값은3seed별F1의 평균이며 단일 결합 예측의F1이 아니다.

| 조건 | 가감속 Macro-F1 | 주행중 조향 Macro-F1 | 조향 seed 범위 |
|---|---:|---:|---:|
| 기존 motion 재현 | .517241 | .684695 | .681430~.690056 |
| 공통이동·residual 분리 | .512456 | .680354 | .664715~.692860 |
| 원본+지역요약 | .490498 | .679482 | .668044~.689691 |

새 두 조건 모두 평균 개선이 없어 채택하지 않는다. 공통분리의 seed20260911 조향 .692860만 선택하는 것은 사전 고정3seed비교와 맞지 않으므로 승격 근거로 삼지 않는다.

### 가감속 클래스별 평균F1 / 재현율

| 클래스 | 기존 F1 / recall | 공통분리 F1 / recall | 지역요약 F1 / recall |
|---|---:|---:|---:|
| ACCELERATING | .332099 / .419718 | .341898 / .439592 | .319272 / .448483 |
| DECELERATING | .331688 / .384744 | .331233 / .439263 | .316803 / .360315 |
| CONSTANT | .609471 / .517205 | .563950 / .451458 | .540145 / .432953 |
| STOPPED | .795704 / .869796 | .812742 / .840626 | .785773 / .880668 |

공통분리는 STOPPED precision .736006→.786955와F1을 올리지만 recall은 낮춘다. 가속·감속 recall 향상과 함께 CONSTANT recall이 크게 하락해 전체가감속F1은 손해다. 지역요약도 정지 recall만 소폭 개선하고 precision과 전체F1이 하락한다.

### 경로별 평균F1: 가감속 / 주행중 조향

| 경로 날짜 | 기존 | 공통분리 | 지역요약 |
|---|---:|---:|---:|
| 07-29 | .490316 / .672043 | .496477 / .664358 | .488787 / .666449 |
| 08-02 | .501291 / .614326 | .506442 / .607005 | .490731 / .613629 |
| 08-14 | .432898 / .755995 | .412673 / .755170 | .387862 / .747095 |
| 08-16 | .314473 / .562152 | .291583 / .557256 | .294915 / .554983 |

조향은 두 변형 모두4개경로 평균이 기준선보다 낮다. 세부 confusion matrix·각 클래스 support/precision/recall/F1은 각run `report.json`과 `class-summary.csv`, `route-summary.csv`에 있다. 일부 경로에 정지정답이 없어도 가감속4클래스 고정 macro로 산출한다.

공통 독립 속도구간 평가(`artifacts/stage3-motion-independent-20260910/`)에서는 정답주행중 speed<2m/s의 공통분리 가감속F1 .268714→.286010, 지역요약 조향F1 .386112→.421842의 국소 개선이 있었다. 속도는 원본 센서값을 사후 평가에만 사용했으며 모델입력에 포함하지 않았다. 전체저하를 뒤집는 근거는 아니다.

### 검증 증거

- `independent-validation.json`: 입력391파일 해시 일치;9체크포인트 재로딩으로 각19603개(train1000+validation18603) 로짓·CSV예측 정확 재현; 독립 기존채점함수로 혼동행렬/전체F1/경로별 혼동행렬 일치.
- 기준선3개 seed의 모든 가중치·mean/std/mask·학습/검증 로짓이 원본 motion실험과 정확 일치한다. 최대 replay 로짓오차0.
- `normalization-and-feature-validation.json`:9개 mean/std가 학습1000개에서만 계산됐는지, mask가사전설계와일치하는지 검증. 공통 평행이동 추가시 residual/지역요약불변, median변화 정확, 좌상단patch가 겹치는 영역만 바꾸는지 PASS.
- `feature-invariants.json`:상수 grid, 원본motion보존, 별도 loop의 median/residual 계산 PASS.
- 추가 독립 검사 재현: `.\.venv\Scripts\python.exe artifacts/stage3-spatial-motion-20260910/verify_normalization.py`.

현재0.4187467089 공식 모델과 제출파일은 유지한다. 이번 결과는 coarse grid 기반의 두 공간특징 설계에 대한 부정 결과이며, 모든 공간적 운동분리 방법의 개선 가능성을 배제하지 않는다.
