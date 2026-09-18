# Stage3 시간 길이·이미지 결합 독립 실험

사용자 요청 2번: 현재 움직임 모델의 직진/회전 균형을 시간 길이와 이미지 결합으로 개선할 수 있는지 확인한다. 제출86809의 공식 Stage3 점수0.4187467089를 얻은 모델은 보존한다. 이번 실험의 평가는 기존 comma2k19 센서 대리 라벨31영상이며 공식 평가와 다르다.

## 사전 고정 조건

artifacts/stage3-temporal-motion-20260910/plan.json을 특징 추출/신규 모델 결과 이전 저장했다.

| 조건 | 현재 flow | 첫 평균 | 둘째 평균 | 이미지 |
|---|---|---|---|---|
| baseline | 1쌍 | 최근5쌍 | 최근15쌍 | 없음 |
| short | 1쌍 | 최근3쌍 | 최근5쌍 | 없음 |
| long | 1쌍 | 최근15쌍 | 최근30쌍 | 없음 |
| long_image | 1쌍 | 최근15쌍 | 최근30쌍 | 현재프레임 RGB/HOG1280차원 |

10Hz에서 평균은 약0.3/0.5/1.5/3초 움직임에 해당한다. 각 시점은 현재·과거만 사용하고 첫프레임 dummy flow는 평균에서 제외한다. 각 영상 시작에서 독립적으로 초기화한다. 기존MLP1682→128→64 및 가감속4/조향3 head, inactive 이미지1280차원은0마스크. 정규화는 고정학습1000표본만 사용한다. 같은3seed20260910/11/12, balanced batch10, AdamW lr.001 wd0,12에폭1200업데이트, final checkpoint만 평가한다.

baseline의20260910은 원본과 가중치·전체로짓 exact 재현을 요구한다. 나머지 baseline2seed는 기존 독립검증 완료 결과를 재사용해 보고서 reused=true로 구별한다. 신규학습10개+재사용2개=총12조건이다. 기존 combined(1/5/15+image)는 참고용이며 신규long_image와 동일 조건으로 취급하지 않는다.

## 실행과 검증

```
.venv/Scripts/python.exe src/stage3_temporal_motion_experiment.py prepare
.venv/Scripts/python.exe src/extract_stage3_temporal_sparse.py
.venv/Scripts/python.exe src/run_stage3_temporal_training.py
.venv/Scripts/python.exe src/validate_stage3_temporal_motion.py
```

초기 전프레임 학습영상 디코딩은 느려 동일한31프레임 인과적 필요구간만 특징 추출하는 별도 추출기로 변경했다. 검증31영상은 기존 dense current-flow134 캐시를 재사용한다. execution-optimization.json에 코드해시와 모델조건불변을 기록하고 첫실제학습영상 dense/sparse 전window exact 검증 후 실행했다. 기존 완료캐시는 보존한다.

원계획의 CPU1thread 학습을 사전 baseline 재현으로 시험했을 때 원본2threads 가중치와 최대7.376e-7 차이가 확인됐다. 신규arm 결과 이전 전체학습을 원본2threads로 복원하며 training-thread-correction.json에 수치 증거와 원plan해시를 기록한다. OpenCV추출은 worker당1thread다.

독립검증은 원표본/라벨/영상/캐시 해시, train-only mean/std, dense validation 모든window 재구성, 실제 train/validation영상 재추출,12체크포인트 로짓·CSV·macroF1 재계산을 수행한다. 인과성·초기경계 및 정지flow 단위검사2개 PASS. 결과는 아직 실행 중이며 아래 완료 절이 생기기 전 개선으로 해석하지 않는다.

## 완료 결과 (2026-09-10)

사전 고정4조건×3seed, 학습1000행/검증31영상18603행(주행17346행)을 완료했다. 신규10개학습의 순수 train+evaluation 합198.08초이며 baseline2개는 기존결과 재사용이다. 최적화추출511.21초는 기존완료10영상 재사용 이후 실행시간이며 초기dense추출·재현진단을 포함한 총벽시계시간은 아니다.

| 조건 | 조향F1 3seed평균 | 범위 | 가감속F1평균 |
|---|---:|---:|---:|
| baseline 1/5/15 | 0.684695 | 0.681430–0.690056 | 0.517241 |
| short 1/3/5 | 0.687982 | 0.684549–0.690690 | 0.516314 |
| long 1/15/30 | 0.685540 | 0.682172–0.690991 | 0.486119 |
| long_image 1/15/30+이미지 | 0.667402 | 0.659503–0.674082 | 0.483918 |

short는 동일seed의baseline보다 세번 모두조금높지만 평균+0.003287로 작다. long은평균+0.000844에그치고 가감속은-0.031122다. long_image는같은long보다조향평균-0.018138로, 긴창에서이미지결합의이득은확인되지않았다. 기존combined(1/5/15+image)의평균0.684117은참고용이며본신규조건과구분한다. 이검증에서short를후속작은후보로남길수있으나 공식점수개선이나현제출모델교체를정당화할정도는아니다.

baseline20260910의2threads 재훈련은 원본가중치 전체·train/validation 로짓 전체exact PASS다. 모든영상의원baseline402차원도exact 일치했다. 독립결과검증 보고서는 artifacts/stage3-temporal-motion-20260910/independent-validation.json을 확인한다. 경로별 recall은 route-class-comparison.csv, 전체지표는 comparison.csv이며 전환·속도구간은 통합실험 보고서의 동일조건비교를 따른다.

구간별 독립평가(artifacts/stage3-motion-independent-20260910/slice-means.csv)에서 short의 조향전환±5프레임 F1은0.492569→0.502053, 안정구간은0.740898→0.742906이었다. 전환구간의개선도+0.009484로크지않다. 주행중2m/s미만305행은0.386112→0.517077, 그중직진124행재현율은25.0%→66.94%로개선됐다. 속도·전환라벨은분석용이며모델입력이나추론분기조건으로사용하지않았다.

short의전체직진재현율은57.13%→60.21%로개선됐으나우회전은81.39%→76.91%로하락했다. 경로07-29/08-02/08-14/08-16의조향F1변화는각각+0.022187/+0.017062/-0.011860/+0.000133다. 모든경로·클래스가일관되게개선된것은아니다. long_image는08-16에서+0.036670이지만나머지3경로하락으로평균은악화했다.

따라서본실험결론은 short를후속독립검증후보로보존하고, long/long_image는현재채택하지않는것이다. 제출ZIP·selected.json·외부GPU/업로드는변경하거나실행하지않았다.

최종독립검증 **PASS**:187원영상·19603특징행·실제재디코딩8행·12체크포인트 전체재추론 최대로짓오차0.0. 결과SHA256은588d26d20bbf966c9e05a31540aba1aa6bd935c6b36889b23d4b3e880c67285d다. 앞선실행중표시보다이완료기록을우선한다. 조향3클래스평균재현율(balanced accuracy)은baseline약0.708695→short약0.704769로소폭하락해, short의macroF1증가가모든클래스균형개선을뜻하지않는다는점도확인했다.
