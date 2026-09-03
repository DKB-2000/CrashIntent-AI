# crashvideo-project — 블랙박스 영상 기반 지능형 고의사고 분석 (Dacon 236753)

이 폴더는 어느 경로에 clone해도 동작하는 독립 Git 프로젝트다. 기본 경로는 `project.properties`, 머신별 외부 데이터
경로는 git에서 제외되는 `project.local.properties`로 관리한다. 저장소 게시 절차와 제외 파일은
`docs/git-portability.md`를 참고한다.

실제 코드(데이터 파이프라인·학습·추론·제출 패키징)가 사는 Codex 작업 루트다. 대회 규칙·데이터 규격·평가 산식·
베이스라인 코드 분석 정본은 `docs/dacon-236753-대회안내.md`를 참고한다. 이 문서는 **실행 로드맵**만 다룬다.

Codex는 작업 시작 시 `AGENTS.md` → `CODEX_HANDOFF.md`의 안내를 따르고, 데이터·모델 관련 도메인 지식은 각각
`docs/domain-knowledge-data.md`, `docs/domain-knowledge-model.md`를 참고한다. 두 문서의 YAML과 에이전트 호출 표현은
Claude Code 형식의 흔적이며, Codex에서는 일반 참고 문서로 취급한다.

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
- **Stage 3(거동 분석)**: 1차 후보는 `comma2k19`로 선정했다. 전체 약 100GB를 받기 전에 공식 저장소의 1분 예제 구간으로 영상↔CAN 동기화 및 10Hz 범주 라벨 변환을 검증한다. 보강 후보는 nuScenes CAN bus와 A2D2이며 상세 기록은 `DATA_SOURCES.md`를 참고한다.
- 대회 규칙상 **법적 제한 없는 외부 데이터·사전학습 모델 사용은 허용**되지만, 2차 진출 시 출처 명시가 의무다. 데이터를 확보하는 즉시 출처·라이선스·이용조건을 기록해 둔다.

→ 관련 도메인 지식: `docs/domain-knowledge-data.md`

#### Stage2 수동 라벨링 도구 (CCD 1,500건 확보 완료 → entry_frame/evasion_space/entry_side 라벨링)

CCD는 `collision_frame`(충돌시점)만 공식 라벨이 있고 나머지 3개 항목은 어떤 공개 데이터셋에도 없다는 것을
확인했다(`DATA_SOURCES.md` 1·2번 섹션 — DoTA 등 대안도 조사했으나 전부 부적합/실물 검증 불가로 종결). 그래서
사람이 직접 보고 라벨링하는 로컬 GUI 도구를 만들었다: `src/label_stage2.py`.

```bash
pip install opencv-python
cd crashvideo-project
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
- [ ] Stage2 CCD 200~300개 수동 라벨링

### 다음 작업

1. 고정 seed로 CCD 30개 holdout CSV를 생성한 뒤, 이를 제외한 1,470개에서 Stage1 ORIGINAL/RERECORDED 학습 세트를 생성한다.
2. 휴대전화 실제 재촬영 검증 세트(원본 30개×3조건, 약 90개)를 준비한다.
3. 실제 재촬영 검증 세트를 제외한 합성 데이터로 Stage1 본 학습을 수행한다.
