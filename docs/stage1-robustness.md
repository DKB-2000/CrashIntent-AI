# Stage1 화질 오탐·합성 효과 의존성 검사

2026-09-10 사용자가 추가 라벨링 없는 개선 중 1·2번을 요청했다.
기존 balanced-v2 본학습과 제출 가중치를 바꾸지 않고 별도 진단을 실행한다.

## 목적과 해석 범위

1. 화면 재촬영 효과 없이 JPEG 압축·블러·노이즈만 추가한 CCD 원본형 영상에서
   RERECORDED 오탐이 늘어나는지 확인한다.
2. 합성 재녹화에서 효과를 하나씩 제거하거나 화면 단서를 약하게 만들었을 때
   RERECORDED 재현율 및 확률이 얼마나 변하는지 확인한다.

합성 조건의 정답은 생성 과정에 따른 proxy다. CCD 원본형의 ORIGINAL도 기존 라벨을
상속한 것으로 실제 촬영 이력을 새로 확인한 정답은 아니다. 기존 본학습에서 모델 선택에
사용하는 검증 출처의 부분집합이므로 독립 최종 테스트라고 부르지 않는다.
결과를 공식 점수나 실제 휴대전화 재촬영 성능으로 해석하지 않는다.

## 고정 설계

- 기존 전체 학습 manifest의 val 출처 21개에서 각각 원본 1개를 해시 순서로 선택.
- 21원본 × 17조건 = 357영상, 원본과 같은 1280×720 / 10fps / 50프레임.
- train 출처와 겹치지 않으며 phone holdout 및 그 형제 출처도 제외.
- 조건마다 같은 원본, 효과 기본 강도, 난수 seed, H.264 CRF 24를 사용한다.
- JPEG/노이즈 강도는 기존 합성 범위. 블러는 효과가 꺼지는 구간을 피하도록 최소 sigma 0.4.
- 전체 효과 조건은 기존 합성기와 픽셀 단위 동등성을 시험했다.
- JPEG 제거는 quality=100 대체가 아니라 JPEG encode/decode 완전 생략.
- 검증셋 자체로 재학습하거나 임계값을 조정하지 않는다.

| 종류 | 조건 |
|---|---|
| 원본형 5종 | 기본, JPEG만, 블러만, 노이즈만, JPEG+블러+노이즈 |
| 합성 재녹화 기본 | 기존 효과 전체 |
| 개별 제거 10종 | 무아레, 밝기 띠, 깜박임, 원근/스케일, 반사, 감마/색이득, 블러, 노이즈, JPEG, 시간 지터 |
| 화면 단서 약화 | 무아레·띠·깜박임·원근/스케일·반사·색변화·지터를 중립값 대비 25%로 약화. JPEG/블러/노이즈는 유지 |

화면 단서를 전부 없애고도 재녹화 정답으로 강제하는 조건은 두지 않는다.
화질만 변경한 대조군은 ORIGINAL로 둔다. 개별 제거·약화 재녹화는 남은 합성 단서를
검사하는 진단용이며, 인간이 반드시 판별할 수 있는 난이도라는 보장은 없다.

## 실행과 상태

코드:

- src/prepare_stage1_robustness.py: 생성·재개, 해시/전 프레임 검사, 출처 분리.
- src/evaluate_stage1_robustness.py: 실제 제출 decoder와 16프레임×3구간, 확률 평균, 고정 임계값 0.5.
- src/watch_stage1_robustness.py: 데이터 완료와 기존 본학습 모델의 독립 검증 완료를 기다려 한 번 자동 평가.
- src/test_stage1_robustness.py / src/test_stage1_robustness_scoring.py.

실제 데이터:
artifacts/stage1-robustness-20260910/

- plan.json, sources.csv: 사전 고정 표본·조건 및 원래 분할 manifest 해시.
- records/<source_id>.json: 해당 원본의 17영상이 모두 검사를 통과한 뒤에만 기록.
- status.json: GENERATING → DATA_VALIDATED 또는 FAILED.
- generation_manifest.csv: 전체 생성 후 확정. 효과값, 라벨 근거, 원본/출력 해시 포함.
- code/: 실제 생성 코드 사본.
- full-model-evaluation/watch-status.json: 대기/평가/완료 상태.
- full-model-evaluation/result/report.json: 본학습 모델 평가가 끝난 뒤 생성.

현재 357영상 생성 작업과 자동 평가 대기 프로세스를 시작했다. 전체 완료 여부는 상태
파일을 우선한다. 실행 중인 작업을 중복 시작하지 않는다. 로컬 PC가 켜져 있어야 한다.

평가 모델은 artifacts/kaggle-stage1-full-20260910/validated/best.pt이며
같은 디렉터리 result-validation.json의 PASS와 SHA256 일치를 확인한 뒤에만 사용한다.
CPU 2스레드, 최대 4시간으로 한 번 평가하며 기존 학습/모델 선택/제출을 변경하지 않는다.
추가 외부 업로드나 GPU 작업은 없다. CPU FP32 결과는 로컬 진단이며 CUDA FP16 제출과
완전히 같은 수치라고 간주하지 않는다. 첫 실제 모델 1구간 CPU 측정은 약 9.47초였으나
초기 비용과 병행 작업 영향을 포함하므로 전체 소요시간의 확정값은 아니다.

필요한 경우 수동 실행(실행 중인 프로세스/출력 상태 확인 후):

```powershell
.venv/Scripts/python.exe src/prepare_stage1_robustness.py --output-dir artifacts/stage1-robustness-20260910 --workers 2
.venv/Scripts/python.exe src/evaluate_stage1_robustness.py --dataset-dir artifacts/stage1-robustness-20260910 --checkpoint artifacts/kaggle-stage1-full-20260910/validated/best.pt --output-dir artifacts/stage1-robustness-20260910/manual-evaluation --device cpu
```

생성은 완료 출처의 identity/해시/전 프레임을 재검사해 재개한다. 평가기는 기존 결과 폴더를
덮어쓰지 않는다. 자동 감시기의 launch-attempt.json은 중복 평가를 막으며 실패 뒤 자동
재실행하지 않는다.

## 결과를 읽는 기준

- condition_summary.csv: 원본 조건별 오탐률, 재녹화 조건별 재현율, 평균 재녹화 확률,
  기준 조건 대비 변화 및 판정이 뒤집힌 원본 수.
- paired_changes.csv: 같은 원본별 확률 변화. 원본 조건의 기준은 original_clean,
  재녹화 조건의 기준은 rerecorded_full.
- predictions.jsonl: 세 구간의 로짓·확률과 평균, 출처 및 영상 해시.
- 저장 파일을 다시 읽어 정답/출처/누락, softmax, 평균, 임계값을 재계산한다.

17조건을 합친 Macro-F1은 원본형 5종/재녹화 12종의 임의 비율에 좌우되므로 주요 지표로
사용하지 않는다. 21출처라 한 건은 약 4.76%p이며 작은 차이를 확정적 개선/악화로 단정하지 않는다.

1. 기본 원본은 맞히지만 JPEG/블러/노이즈 원본에서 재녹화 오탐이 생기면 화질 단서 의존 후보.
2. 전체 효과는 맞히지만 특정 효과 제거에서 놓치면 해당 효과 의존 후보.
3. JPEG 제거만으로 재현율이 크게 떨어지고 JPEG 원본 오탐도 늘면 압축 단서 의존을
   우선 조사한다. 두 대조군의 결과를 함께 해석한다.
4. 취약 조건이 확인되면 기존 train 출처에서 별도로 대응 학습 데이터를 만든다.
   이 진단 영상과 phone holdout을 학습에 섞지 않는다.
5. 이번 작업은 취약점 진단이며 모델 성능 개선/승격 완료를 의미하지 않는다.

## 검증

6개 단위/통합 테스트 PASS:
기존 합성과 전체 효과의 픽셀 동등성, JPEG 완전 우회, 품질 대조군 화면 단서 중립화,
출처 선택 재현성과 누수 거절, 실제 17영상 생성·재개·손상 거절,
조건별 오탐/재현율 변화·저장 로짓 손상/누락 거절.

별도 artifacts/stage1-robustness-smoke-20260910/는 64×48 인공영상 소프트웨어 시험자료다.
실제 제출 모델을 이용한 생성→추론→저장→재검산 스모크를 실행하며 SUBSET_SMOKE로 표시한다.
이 자료의 점수는 성능 근거로 사용하지 않는다. 스모크 완료 여부는 evaluation/status.json을 확인한다.

### 실제 제출 모델 연결 스모크 완료

17조건/51구간을 실제 baseline Stage1 모델과 제출 decoder로 추론하고,
저장한 로짓의 softmax·3구간 평균·임계값·출처/누락 검사를 재계산해 PASS했다.
CPU FP32, 273.77초. 결과는 artifacts/stage1-robustness-smoke-20260910/evaluation/report.json.
인공영상 소프트웨어 시험이며 실제 검증 성능은 아니다.
실제 21출처/357영상 생성과 본학습 모델 자동 평가 대기는 계속 진행 중이다.

## 2026-09-11 본검사 시간초과 복구

본학습6에폭/2640업데이트와690영상 독립검증PASS(합성Macro-F1 .9983713, baseline .2968900).
357영상 CPU본검사는78영상에서 Evaluation budget reached로 중단됐다.
68영상 시점1229초→종료21323초의 큰 시간 간격이 있으나 절전 등 원인은 확정하지 않았다.
저장된78영상의 모델/전처리/데이터/버전·출처·순서·로짓을 확인해 재사용하는 --resume 추가.
재개·손상거절2테스트+기존채점2테스트PASS. 원래실패 상태를 resume-*.json에 보존.
recover_stage1_robustness.py 로컬숨김실행,80/357까지 증가 확인 및 오류로그없음.
평가중 SetThreadExecutionState로 자동유휴절전을 일시 방지하고 종료시해제한다.
완료후에만 기존ZIP예약의 해당실패 gate를 WAITING_FOR_VALIDATION으로 되돌린다.
recovery-evaluation.log 및 result/status.json 참고. 전체본검사·새ZIP은 아직미완료.

2026-09-11 09:39 확인: 본검사가205/357에서09:27 네이티브종료.
Windows Application1000: c10.dll,0xc0000005. 근본원인 미확정.
오래된RUNNING표시를 실제프로세스부재·watcher반환코드3221225477과 대조해FAILED로정정,
원상태 native-crash-20260911-0927.json 보존.205저장예측 재검증PASS.
CPU1스레드로재개,프로세스실패시결과상태도FAILED로기록하며 진전있을때만 최대3회 제한재시도.
재시도/동일지점정지/성공후ZIPgate복구2통합테스트PASS. 새로그 recovery-evaluation-*.log.
