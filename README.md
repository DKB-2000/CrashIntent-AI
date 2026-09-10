# crashvideo-project — 블랙박스 영상 기반 지능형 고의사고 분석 (Dacon 236753)

이 폴더는 어느 경로에 clone해도 동작하는 독립 Git 프로젝트다. 기본 경로는 `project.properties`, 머신별 외부 데이터
경로는 git에서 제외되는 `project.local.properties`로 관리한다. 저장소 게시 절차와 제외 파일은
`docs/git-portability.md`를 참고한다.

새 PC에서 clone한 뒤 별도로 복원할 파일과 로컬 환경 준비 순서는
`docs/local-setup.md`를 참고한다.

실제 코드(데이터 파이프라인·학습·추론·제출 패키징)가 사는 Codex 작업 루트다. 대회 규칙·데이터 규격·평가 산식·
베이스라인 코드 분석 정본은 `docs/dacon-236753-대회안내.md`를 참고한다. 이 문서는 **실행 로드맵**만 다룬다.
세 Stage의 확정 데이터 구성과 보강 조건은 `docs/data-strategy-status.md`에서 한눈에 확인할 수 있다.

Codex는 작업 시작 시 `AGENTS.md` → `CODEX_HANDOFF.md`의 안내를 따르고, 데이터·모델 관련 도메인 지식은 각각
`docs/domain-knowledge-data.md`, `docs/domain-knowledge-model.md`를 참고한다. 두 문서의 YAML과 에이전트 호출 표현은
Claude Code 형식의 흔적이며, Codex에서는 일반 참고 문서로 취급한다.

## 대회 목표 (2026-09-10 사용자 확정)

**최종 1차 평가 상위 15위 이내에 진입하여 2차 평가에 진출한다.**
이후 모델 개선과 제출 결과는 이 목표에 맞춰 추적한다.
사용자 제공 순위표의 15위는 장재홍이며 Stage1 0.84684 / Stage2 0.4073 / Stage3 0.72274,
표시 점수 가중합은 약 0.621384다. 우리 첫 제출의 가중합은 0.211087224다.
이는 사용자 제공 순위표의 비교용 스냅샷이며 최종 진출선은 아니다.
상세 비교 기준과 제출 결과는 `docs/submission-history.md` 참고.

## 일정 기준점

- 오늘: 2026-09-02
- 팀 병합 마감: 09-23 (약 3주)
- 1차(리더보드) 마감: 09-29 (약 4주)
- 2차 평가 자료 제출: 10-05
- 최종 결과 발표: 10-16

## 로드맵

### Phase 0 — 파이프라인 리스크 제거 (최우선)
제출 메커니즘 자체가 막히면 모델이 아무리 좋아도 소용없다. 모델 고도화보다 먼저 끝낸다.

Kaggle 실행 절차와 실행 전후 검증 방법은 `docs/phase0-kaggle-runbook.md`를 따른다.

**상태: 완료(2026-09-02).** Kaggle Tesla T4에서 학습·추론·공개 예제 스모크 테스트·`submit.zip` 생성을 완료했고,
로컬 ZIP 구조 검증도 통과했다. `Baseline/submit.zip`을 이후 변경의 회귀 검증 기준선으로 사용한다.

1. 데이콘 데이터 탭에서 `baseline.zip` 다운로드 → 이 폴더에 배치
2. 로컬 GPU 환경 확인 (없으면 클라우드: Colab Pro / RunPod / 기타 대여 GPU 검토 — CUDA 필수, 베이스라인 코드는 CUDA 없으면 즉시 실패하도록 작성돼 있음)
3. 학습 노트북(`14151`) → 추론 노트북(`14152`) 순서로 그대로 실행해 `submit.zip`이 정상 생성되는지 확인 (제출은 하지 않음, 로컬 스모크 테스트만)
4. 통과하면 이 결과를 기준선(baseline)으로 커밋 — 이후 모든 변경은 "이 스모크 테스트를 계속 통과하는가"로 회귀 검증

→ 관련 도메인 지식: `docs/domain-knowledge-model.md`

### Phase 1 — 데이터 확보 전략 (가장 큰 리스크, Phase 0과 병렬 착수)

데이콘은 3개 Stage 모두 **학습 데이터를 제공하지 않는다**(코드 실행 확인용 공개 예제만 제공). 참가자가 직접 구성해야 한다.

- **Stage 2 최우선 단서**: 베이스라인 학습 코드 주석에 "공개 CCD 5건"이라는 표현이 있음 — 사고 예측 연구에서 널리 쓰이는 공개 데이터셋 **Car Crash Dataset(CCD)**를 가리킬 가능성이 높음. 라이선스·이용조건·실제 라벨 스키마(충돌 시점 등)를 조사해 최우선으로 확보 시도.
- **Stage 1(재녹화 판별)**: 데이터 구축 방침 확정(2026-09-03). CCD를 source ID 기준으로 먼저 분할한 뒤 무아레·주사율 간섭·밝기 띠·원근·초점·이중 압축 등을 무작위 합성한다. 학습에 쓰지 않은 원본 30개를 휴대전화로 각 3조건 재촬영한 약 90개는 실제 재촬영 검증 세트로 보존한다.
- **Stage 3(거동 분석)**: 기본 데이터는 `comma2k19`, 도시·저속·정지·회전 보강은 ZOD로 확정했다. comma2k19 1분 예제의 영상↔CAN 동기화 및 10Hz 범주 변환은 PASS했고, ZOD는 접근 신청 후 승인 대기 중이다. nuScenes와 A2D2는 라이선스 제약으로 보류한다.
- 대회 규칙상 **법적 제한 없는 외부 데이터·사전학습 모델 사용은 허용**되지만, 2차 진출 시 출처 명시가 의무다. 데이터를 확보하는 즉시 출처·라이선스·이용조건을 기록해 둔다.

→ 관련 도메인 지식: `docs/domain-knowledge-data.md`

#### Stage2 수동 라벨링 도구 (CCD 1,500건 확보 완료 → entry_frame/evasion_space/entry_side 라벨링)

CCD는 `collision_frame`(충돌시점)만 공식 라벨이 있고 나머지 3개 항목은 어떤 공개 데이터셋에도 없다는 것을
확인했다(`DATA_SOURCES.md` 1·2번 섹션 — DoTA 등 대안도 조사했으나 전부 부적합/실물 검증 불가로 종결). 그래서
사람이 직접 보고 라벨링하는 로컬 GUI 도구를 만들었다: `src/label_stage2.py`.

```bash
pip install opencv-python
# 저장소 루트에서 실행
python src/label_stage2.py
```

- 조작: `a`/`d`(또는 ←/→) 프레임 이동, `스페이스` entry_frame 기록, `1`/`2` entry_side(LEFT/RIGHT),
  `y`/`n` evasion_space(있음/없음), `s` 이 영상 건너뛰기(defer), `q`/`ESC` 저장 후 종료.
- `collision_frame`은 CCD 공식 라벨(`Crash-1500.txt`)에서 그대로 읽어와 화면에 참고용으로만 표시 — 새로 계산하지 않음.
- 영상 1건이 끝날 때마다 즉시 `data/stage2/labels_manual.csv`(컬럼: `ID,collision_frame,entry_frame,evasion_space,entry_side`,
  대회 제출 CSV와 동일한 컬럼명)에 append. 이미 라벨링된 ID는 재실행 시 자동으로 건너뛴다(resume 가능).
- 자세한 조작키·동작은 스크립트 상단 docstring 참고.

### Phase 2 — Stage별 모델 고도화

베이스라인(EPOCHS=1, 공개 예제 5건)은 **평가서버 호환성 확인용 스모크 테스트일 뿐 실제 성능용이 아니다.** 구조(함수 시그니처·제출 포맷)는 유지하고 학습 로직을 실제 데이터 기준으로 교체한다.

가중치가 큰 순서(Stage2·3 = 0.4, Stage1 = 0.2)로 투자 우선순위를 둔다.

- Stage 1: 확보한 재녹화 페어로 MViTv2-S 파인튜닝, 클립(slots) 수 확대·앙상블 검토
- Stage 2: 베이스라인은 `collision_frame`만 학습됨(확인됨) — `entry_frame`/`evasion_space`/`entry_side`를 실제로 학습시키는 것이 최대 개선 포인트. 백본 파인튜닝 여부, GRU→Transformer 교체 검토
- Stage 3: 프레임 단위 정밀 라벨 확보가 성능의 핵심, 클래스 불균형(STOPPED 등) 처리

→ 관련 도메인 지식: `docs/domain-knowledge-model.md` (데이터 요구사항은 `docs/domain-knowledge-data.md` 참고)

### Phase 3 — 로컬 평가 하네스

일일 제출 3회 제한 때문에 실제 리더보드 제출 전에 로컬에서 점수를 예측할 수 있어야 한다. 대회 공식 산식을 그대로 재현하는 스크립트를 만든다.

- Stage1·3: Macro-F1
- Stage2: 프레임 번호를 초 단위로 변환 후 정답과 비교, 오답 처리 조건(결측·범위초과 등) 포함
- 종합점수 = Stage1×0.2 + Stage2×0.4 + Stage3×0.4

→ 관련 도메인 지식: `docs/domain-knowledge-model.md`

### Phase 4 — 제출 컴플라이언스 상시 점검

모델을 바꿀 때마다 다음을 재확인한다: 추론 60분/설치 10분 이내, zip 10GB/압축해제 32GB 이내, 완전 오프라인 동작, `predict_stage1/2/3` 함수 시그니처 및 반환 컬럼 불변, ast 파싱 + zip 내부 재검증 통과.

### Phase 5 — 반복 및 2차 준비

1차 마감(09-29)까지 리더보드 피드백을 반복하고, 상위 15팀 진출 시 2차 자료(모델 개발 보고서·학습데이터 구성 보고서·학습 코드·팀원 정보, 이메일 제출)를 준비한다. Phase 1에서 기록해 둔 데이터 출처 로그를 여기서 그대로 재사용한다.

## 지금 당장 할 일

2026-09-10 사용자 요청으로 **Stage 1 본 학습을 우선 진행**한다. 전체 합성 생성과
첫 60영상 검증, FP32 학습 코드 및 비공개 GPU 사전 번들을 준비했다. 업로드는 자동 승인
검토에서 처음 차단됐으나 사용자 명시적 승인 후 비공개 Kaggle GPU 사전 학습 version 1을 실행했다.
자동 결과 회수·검증을 연결했으며 전체 본 학습 완료 결과는 아직 없다.
현재 상태와 재개 명령은 `docs/stage1-full-training.md`를 따른다.


2026-09-06 로컬 업데이트: 영상 ID 200번까지 확인을 진행했고 수동 라벨 74건이 저장됐다.
추가 평가는 실행하지 않았으며, 앞서 44건으로 수행한 최초 개발 테스트 결과와 결론은
`docs/stage2-manual-test-100.md`에 보존한다. 다음 목표는 적합한 직접 충돌 표본 200~300건이다.

### 현재 진행 현황 (2026-09-03)

- [x] Phase 0 공식 베이스라인 학습·추론·ZIP 생성 검증
- [x] Stage1 데이터 전략 확정: CCD 합성 재녹화본 + 휴대전화 실제 재촬영 검증 약 90개
- [x] Stage2 CCD 1,500개 확보 및 수동 라벨링 GUI 완성
- [x] Stage2 4-output 학습·추론 스모크 코드 작성 및 정적 계약 검사
- [ ] Stage2 4-output Kaggle 런타임 스모크 PASS 확인
- [x] Stage3 `comma2k19` 1분 예제 확보 및 영상/CAN/IMU 동기화 실측
- [x] Stage3 20Hz 영상→10Hz 변환 및 임시 범주 라벨 600개 생성·육안 확인
- [x] Stage3 라벨 변환 코드 프로젝트에 정식 구현 및 정적 계약 검사
- [x] Stage3 변환기 Kaggle 런타임 스모크 PASS 확인(1,200→600프레임, 수동 검증 분포와 일치)
- [x] Stage3 데이터 전략 검증 완료: comma2k19 사용, 대용량 확대·보강은 실제 학습 시 필요에 따라 진행
- [x] Stage1 합성 데이터 생성기 구현 및 정적 계약 검사
- [x] Stage1 합성 생성기 Kaggle 런타임 스모크 PASS(2 sources, 6 samples, source leakage 없음)
- [x] Stage1 합성본 육안 품질 검증(사용자 확인)
- [x] Stage1 MViTv2-S 학습·재로딩·추론 스모크 코드 작성 및 정적 계약 검사
- [x] Stage1 모델 파이프라인 Kaggle 런타임 PASS(2 samples, CUDA, checkpoint 재로딩·출력 계약 정상)
- [x] Stage1 휴대전화 검증용 source holdout 선택기 및 생성기 제외 옵션 구현
- [x] Stage1 외부 보강 후보 Nexar 샘플 다운로드·재생·합성 검증 및 라이선스 확인
- [x] Stage1·2·3 데이터 전략 확정 및 `docs/data-strategy-status.md` 문서화
- [x] Stage3 보강 데이터 ZOD 접근 신청(승인 대기)
- [ ] Stage2 CCD 200~300개 수동 라벨링 (ID 200까지 진행, 저장 74건)

### 다음 작업

1. 고정 seed로 CCD 30개 holdout CSV를 생성한 뒤, 이를 제외한 1,470개에서 Stage1 ORIGINAL/RERECORDED 학습 세트를 생성한다.
2. 휴대전화 실제 재촬영 검증 세트(원본 30개×3조건, 약 90개)를 준비한다.
3. 실제 재촬영 검증 세트를 제외한 합성 데이터로 Stage1 본 학습을 수행한다.

### Stage3 모델 코드 구현 완료 (2026-09-07)

`src/stage3_pipeline.py`에 audit/train/evaluate/predict/smoke와 CUDA 전용
`predict_stage3(data_dir, model_dir)`를 구현했다. 과거 16프레임, 모든 10Hz 시점 출력,
STOPPED 조향 손실 제외 및 route 검증을 적용한다. 테스트 6개와 실제 MViTv2-S CPU 스모크,
짧은 데이터의 정식 학습·검증·체크포인트 재로딩 연결이 통과했다.
GPU 스모크용 ZIP은 `artifacts/stage3-gpu-smoke.zip`이며 실제 GPU PASS와 본 학습은 아직 미완료다.
다음은 Kaggle GPU 스모크 확인이다. 실행법과 한계는 `docs/stage3-pipeline-runbook.md` 참고.

### 2026-09-09 자동 일일 ZIP 준비

매일 한국시간 08:30 Windows 예약 실행을 등록하고 즉시 시험 실행을 통과했다.
현재 GPU 검증된 후보로 날짜별 ZIP을 준비하며 실제 대회 제출은 하지 않는다.
파일 위치·실행 조건·후보 갱신법은 `docs/daily-submission.md` 참고.
Stage1·2·3 개선은 실험별 GPU 1시간, 유망 후보 본 학습 최대 10시간 범위로 위임받았다.

### 2026-09-10 Stage2 개선 후보 GPU 검증 완료

수동 라벨 대조 실험 soft_mixed_scene 모델을 반영한 별도 후보
artifacts/stage2-submit-candidate-20260910/submit.zip의 T4 통합 검사와 반환 ZIP 로컬 검증이 PASS했다.
동일 개발 검증 평균은 0.6000이며 공식 점수는 아니다. 실제 대회 제출은 아직 하지 않았다.
상세 docs/stage2-improved-candidate.md 참고. 사용자 승인으로 같은 범위의 비공개 Kaggle
업로드·GPU 검사는 재확인 없이 진행한다.

### 2026-09-10 Stage3 과적합 진단 성공

동일10클립에서 새 초기화·FP32·유효 배치10은 두 학습률 모두 가감속/조향100%를 달성했고
GPU 결과의 독립 로컬 재검증도PASS했다. 기존 쏠림 가중치는 동일 조건에서 회복하지 못했다.
다음은 새 초기화·큰 유효 배치의 독립 route 검증이다. 아직 공식 점수 개선을 확인한 것은 아니다.
docs/stage3-optimization-diagnostic.md 참고.

2026-09-10 Stage1 사전 GPU 6에폭 및 결과 회수·독립 검증 PASS. 합성 검증 F1은0.4이나
전부 RERECORDED 예측으로 쏠림이 남아 있다. 전체 합성은 진행 중이며 전체 본 학습은 아직 미실행.
상세 `docs/stage1-full-training.md` 참고.


Stage3 직접 검증셋 구축: [도구 실행 및 판정 안내](docs/stage3-human-validation-guide.md). 구간별 수동 라벨 저장·재개와 사람 기준 조향채점을 지원합니다.
