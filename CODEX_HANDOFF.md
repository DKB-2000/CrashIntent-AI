# CODEX 인계 가이드 — Dacon 236753 crashvideo-project

이 문서는 Claude Code 세션의 토큰 예산 소진으로 작업을 Codex(또는 다른 에이전트)가 이어받기 위한 인계 문서입니다.
이 문서 하나만 읽어도 프로젝트 전체 맥락과 "지금 어디서부터 이어가야 하는지"를 알 수 있게 구성했습니다.

## 최신 재개 지점 (2026-09-07)

마지막 작업인 **Stage3 comma2k19 Chunk_1 변환과 전수 검증을 완료**했다.
187개 영상 / 21개 route / 112,205개 10Hz 임시 라벨이며 검증은 PASS다.
상세 결과와 재현 명령은 `docs/stage3-comma2k19-runbook.md`,
`artifacts/stage3-comma-chunk1/validation_report.json`을 참고한다.
추가로 조향 보정 실험 v1과 route 분할을 완료했다. 영점 −0.2455도·직진 ±1.5도, 학습 156개/검증 31개다. 상세는 `docs/stage3-calibration-v1.md`와 `artifacts/stage3-comma-chunk1-calibrated-v1/`을 참고한다. Stage3 모델 파이프라인도 구현 완료했다. `docs/stage3-pipeline-runbook.md` 참고. 테스트 6개와 실제 MViTv2-S CPU 학습·저장·재로딩·추론 스모크가 통과했다. GPU 번들은 `artifacts/stage3-gpu-smoke.zip`이며 다음은 Kaggle GPU 스모크 실제 PASS 확인이다.
초기 구간별 중앙값 라벨은 보존했고 v1은 별도 생성했다. v1도 센서 proxy와 표본 검수로 정한 실험용이며 공식 정답은 아니다. 본 학습은 미진행이다.
Stage2는 74건 개발 평가도 수행됐으며 `docs/stage2-manual-test-200.md`에 결과가 있다.
아래 과거 기록의 Stage3 확대 보류·Stage2 추가 평가 미실행 상태보다 이 기록을 우선한다.
## 0. 프로젝트 한 줄 요약

데이콘 대회 236753 "블랙박스 영상 기반 지능형 고의사고 분석 모델"(https://dacon.io/competitions/official/236753) 참가 프로젝트.
실제 코드는 이 저장소 루트 안에서 작업한다. 현재 로컬 경로는 `C:\Crash_AI\CrashIntent-AI`다.

> 이 폴더는 독립 Git 저장소이며 **이 폴더 자체를 작업 루트로 사용**한다.
> 필요한 문서는 모두 `docs/`에 있다. **상위 폴더로 올라갈 필요가 없다.**
> 아래 완료 기록과 구조는 이전 PC에서의 작업 이력이다. 새 clone에 외부 데이터나 모델이
> 존재한다는 의미는 아니다. 로컬 복원 목록은 `docs/local-setup.md`를 참고한다.

## 1. 먼저 읽어야 할 문서 (이 순서로)

1. **`docs/dacon-236753-대회안내.md`** — 대회 배경·규칙·제출조건·Stage별 데이터 규격(정확한 폴더구조·CSV컬럼)·평가 산식·
   공식 베이스라인 코드 전체 분석(정확한 모델 아키텍처: Stage1 MViTv2-S / Stage2 ResNet18+BiGRU / Stage3 MViTv2-S
   듀얼헤드)이 원문 기준으로 정리돼 있음. **가장 먼저, 전체를 읽을 것.** 이 프로젝트의 정본 문서.
2. **`README.md`**(이 폴더) — 실행 로드맵(Phase 0~5), "지금 당장 할 일"
3. **`DATA_SOURCES.md`** — Stage2용 외부 데이터셋(CCD, DoTA) 조사·다운로드·검증 결과와 최종 판단
4. **`docs/domain-knowledge-data.md`**, **`docs/domain-knowledge-model.md`** — 원래 Claude Code 전용
   서브에이전트(`crashvideo-data-expert`/`crashvideo-model-expert`) 정의 파일의 사본. Codex는 Claude Code의 Agent
   도구 자체를 쓸 수 없으니 "에이전트로 호출"하는 개념은 없고, 그냥 이 프로젝트의 도메인 지식 문서(Stage별 데이터
   규격표, 베이스라인 아키텍처 표, 평가서버 제약 요약, 작업 규칙)로 읽으면 된다. 파일 앞부분의 YAML은 원래 형식의
   흔적일 뿐 무시해도 됨.

## 2. 폴더 구조 (실측, 2026-09-02 기준)

```
crashvideo-project/                          ← Codex 작업 루트로 이 폴더를 그대로 사용 (cd 후 codex 실행)
├── AGENTS.md                                ← Codex가 시작 시 자동으로 읽는 관례 파일 (CODEX_HANDOFF.md로 안내)
├── CODEX_HANDOFF.md                         ← 이 문서
├── README.md                                ← 로드맵(Phase 0~5)
├── DATA_SOURCES.md                          ← CCD/DoTA 조사 결과·라이선스 판단
├── docs/
│   ├── dacon-236753-대회안내.md              ← 대회 정본 문서 사본 (§1 참고)
│   ├── domain-knowledge-data.md             ← 데이터 소싱·라벨링 도메인 지식 사본
│   └── domain-knowledge-model.md            ← 모델·추론·패키징 도메인 지식 사본
├── Baseline/                                ← 데이콘 공식 baseline.zip 압축해제본 (사용자가 직접 다운로드해 넣음)
│   ├── [Baseline_Train]_3Stage_학습.ipynb
│   ├── [Baseline_Inference]_3Stage_추론및ZIP생성.ipynb
│   ├── requirements.txt
│   └── data/stage{1,2,3}/...                ← 데이콘이 배포한 소규모 공개 예제(각 5건)
├── src/
│   └── label_stage2.py                      ← Stage2 수동 라벨링 GUI 도구 (완성, §5 참고)
├── data/stage2/labels_manual.csv            ← 라벨링 도구 실행 시 자동 생성됨 (아직 미생성 = 라벨링 미시작)
└── data_raw/                                ← 외부 원본 데이터, git 미추적(.gitignore 처리됨, 용량 큼)
    ├── ccd/videos/Crash-1500/*.mp4          ← CCD 1,500건 확보·검증 완료
    ├── ccd/Crash-1500.txt                   ← CCD 공식 라벨(binlabels 등)
    ├── ccd/_validation_report.json          ← CCD 검증 결과(collision_frame 계산값 포함)
    └── dota/                                ← DoTA 조사용 다운로드분 (라벨만 + 검증용 클립 일부)
```

> 현재 이 폴더 자체가 독립 Git 저장소다. 경로 기본값은 `project.properties`,
> PC별 경로는 git에서 제외되는 `project.local.properties`로 관리한다.

## 3. 완료된 것

- [x] 대회 규칙·데이터 규격·평가 산식·공식 베이스라인 코드(정확한 아키텍처) 전부 원문 확인·문서화
- [x] 로드맵 설계 (Phase 0~5)
- [x] Claude Code 서브에이전트 2개 등록 (`crashvideo-data-expert`, `crashvideo-model-expert`)
- [x] 로컬 컴퓨트 확인: **Intel 내장그래픽뿐, CUDA GPU 없음** → 클라우드 필수. 전략: 무료(Kaggle 등) 우선 → 예산제 유료 전환.
      대략적 예산 시나리오(데이터량 미확정이라 근사치): 최소 $15~25 / 적정 $30~60 / 넉넉 $80~250
- [x] Stage2용 **CCD(Car Crash Dataset) 조사·다운로드·검증 완료** — 사고 영상 1,500건. `collision_frame`은 CCD 공식
      라벨(`binlabels`)로 이미 확보됨. 나머지 3개 항목(entry_frame/evasion_space/entry_side)은 CCD에 없음이 확인됨.
- [x] **DoTA 데이터셋 조사** — entry_frame 대리 라벨로는 **부적합 확정**(라벨 스키마 구조상 데이터 자체가 없음, 논문
      원문·실측 전수조사로 확인). entry_side/evasion_space는 좌표상 신호는 있어 보였으나, 참조하는 원본 유튜브 영상이
      대부분 삭제되어(150개 이상 시도 전부 실패) **실물 이미지 검증이 불가능** → 이 트랙은 **보류**로 종결.
- [x] **Stage2 수동 라벨링 GUI 도구 제작·실사용 검증 완료** (`src/label_stage2.py`) — OpenCV 기반, 헤드리스
      스모크테스트(영상 1,500건 로드·CSV append/resume 로직) 통과. 2026-09-02 로컬에 Python 3.11과 프로젝트 전용
      `.venv` 및 OpenCV를 구성하고, 사용자가 실제 GUI 영상 표시와 키 조작이 정상임을 확인함.
- [x] **Phase 0 Baseline 전체 파이프라인 검증 완료(2026-09-02)** — Kaggle Tesla T4에서 공식 학습 노트북 → 추론
      노트북 → 공개 예제 Stage 1/2/3 스모크 테스트 → `submit.zip` 생성을 끝까지 실행했다. 로컬
      `scripts/Test-Phase0Baseline.ps1 -SubmitZip Baseline/submit.zip` 검증도 통과했으며 필수 모델 파일과 ZIP 구조가 정상이다.
      Kaggle Python 3.12 환경에서는 requirements 설치 직후 NumPy/Pandas ABI 충돌이 발생하므로 설치 후 세션 재시작이
      필요했고, Kaggle Dataset 파일명 제한 때문에 대괄호 없는 노트북 사본을 사용했다.
- [x] **Stage1 데이터 구축 전략 확정(2026-09-03)** — CCD 원본에 재녹화 특성을 합성하고, 학습에 쓰지 않은 원본
      30개를 휴대전화로 각 3조건 재촬영한 약 90개를 실제 재녹화 검증 세트로 사용한다.
- [x] **Stage3 comma2k19 1분 예제 타당성 검증(2026-09-03)** — 20Hz 전방 영상 1,200프레임과 약 89Hz
      speed/steering CAN, 약 104Hz IMU가 약 60초 timestamp 범위로 동기화됨을 확인했다. 10Hz 600개 임시 범주 라벨과
      오버레이를 생성해 사용자가 육안 확인했다. 상세 실측값은 `DATA_SOURCES.md` 6장 참고.
- [x] **Stage3 comma2k19 변환기 구현·런타임 검증 완료(2026-09-03)** —
      `src/prepare_stage3_comma2k19.py`가 10Hz MP4, 베이스라인 학습 CSV, 디버그 CSV, 분포 JSON, 오버레이를 자동 생성한다.
      Kaggle 공식 1분 예제에서 PASS했고 1,200→600프레임 및 기존 수동 검증 분포와 정확히 일치했다.
- [x] **Stage1 합성·모델 스모크 파이프라인 검증 완료(2026-09-03)** — 재녹화 합성기는 공식 원본 2개에서
      ORIGINAL 2개+RERECORDED 4개를 생성했고 사용자 육안 검증 및 source leakage 검사를 통과했다. MViTv2-S 모델은
      Kaggle CUDA에서 2 samples/1 epoch 학습, 체크포인트 저장·재로딩, 3-slot 추론, `ID,answer` 계약까지 PASS했다.

## 4. 진행 중 / 미완료 — Codex가 이어받을 것 (우선순위 순)

> Stage별 확정 데이터 구성과 보강 조건의 최신 요약은 `docs/data-strategy-status.md`를 우선 참고한다.

1. **[현재] Stage1 실제 데이터 구축** — 합성 생성기와 MViTv2-S 모델 파이프라인은 Kaggle 런타임 PASS했다.
   CCD 전체를 source ID 기준으로 분할해 ORIGINAL/RERECORDED 학습 세트를 생성하고, 별도 원본 30개는 휴대전화
   3조건 재촬영 검증용으로 제외한다.
3. **Stage2 CCD 라벨링** — 목표는 우선 200~300건. 2026-09-06 기준 영상 ID 200까지 확인했고
   수동 라벨 74건이 저장됐다. 다음 재개 지점은 ID 201이며, 200~300건을 확보한 뒤 평가를 재개한다.
4. **Stage2 4-output 런타임 검증·학습** — `src/stage2_pipeline.py`와 정적 검사는 작성 완료. Kaggle 런타임 PASS 확인 후
   실제 라벨이 채워지면 entry_frame/evasion_space/entry_side까지 학습한다.
5. **Phase 3 로컬 평가 하네스** — 대회 공식 산식(Stage1·3 Macro-F1, Stage2 프레임→초 변환 비교, 종합점수
   0.2/0.4/0.4 가중)을 재현하는 채점 스크립트. 미착수. 일일 제출 3회 제한 때문에 실제 제출 전 필수.
6. **Phase 4 제출 컴플라이언스 점검** — Phase 0 기준선 ZIP을 바탕으로 모델 변경 때마다 병행한다.

> Stage3 데이터 확대(약 10GB 청크)와 nuScenes/A2D2 보강은 2026-09-03 사용자 결정으로 현재 범위에서 보류했다.
> comma2k19 1분 예제의 데이터·변환·런타임 검증은 완료 상태로 간주한다.

## 5. 반드시 지켜야 할 제약 (위반 시 실격 또는 파이프라인 붕괴)

- `predict_stage1/2/3(data_dir, model_dir)` 함수 시그니처와 반환 DataFrame 컬럼명은 **절대 변경 금지** (평가 서버 계약)
- `submit.zip` 구조 고정: `inference.py`, `requirements.txt`, `model/stage1/best.pt`,
  `model/stage2/{best.pt,resnet18-f37072fd.pth}`, `model/stage3/best.pt`
- 평가 서버는 **완전 오프라인·CUDA GPU 전제**, 추론 60분/설치 10분/zip 10GB/압축해제 32GB 이내
- 비공개 평가 데이터로 추가 학습·튜닝·Pseudo-Labeling 금지, 파일(샘플) 간 예측값 교차 참조 금지
- 일일 제출 3회 제한 — 로컬 검증(Phase 3) 없이 실제 제출 낭비 금지
- `crashvideo-project/data_raw/`는 git 추적 안 함(대용량) — 새로 받는 대용량 데이터도 이 규칙 유지, `.gitignore` 확인
- 새 외부 데이터·사전학습 모델 사용 시 출처·라이선스를 `DATA_SOURCES.md`에 기록(2차 평가 제출 시 출처 명시 의무)

## 6. 일정 (이 문서 작성 시점 기준)

오늘 2026-09-02 / 팀 병합 마감 09-23 / 1차(리더보드) 마감 09-29 / 2차 평가 자료 제출 10-05 / 최종 결과 발표 10-16

## 7. Codex 실행 방법

```bash
cd "C:\Crash_AI\CrashIntent-AI"
codex
```

이 폴더를 작업 루트로 잡고 실행할 것. Codex CLI는 관례상 `AGENTS.md`를 자동으로 읽으므로, 이 폴더의 `AGENTS.md`가
이 문서(`CODEX_HANDOFF.md`)로 안내해준다 — 별도로 컨텍스트를 붙여넣을 필요 없이 시작하면 됨.

**상위 폴더로 올라가지 말 것.** 현재 clone한 저장소 안에서만 작업한다.
필요한 문서는 전부 `docs/`에 사본으로 들어있다.

## 8. 문서 갱신 규칙

작업하면서 새로 알게 된 대회 관련 사실·데이터 조사 결과는 이 문서에 새로 쓰지 말고, 해당 정본 문서
(`docs/dacon-236753-대회안내.md`, `DATA_SOURCES.md`, `README.md`)를 갱신하는 방식으로 남길 것. 단, 이 문서들은
**상위 폴더 원본과는 이제 별개 사본**이라는 점을 기억할 것 — 나중에 Claude Code 세션으로 돌아가면 두 사본(원본
`../dacon-236753-대회안내.md`, `../.claude/agents/*.md` vs 이 폴더의 `docs/*`) 중 어느 쪽이 최신인지 비교해서 반드시
동기화해야 한다. 이 `CODEX_HANDOFF.md`는 방향이 크게 바뀌었을 때만 "완료된 것/미완료" 섹션을 갱신하면 된다.
