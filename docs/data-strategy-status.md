# Stage 1·2·3 데이터 전략 및 진행 현황

기준일: 2026-09-03

이 문서는 세 Stage의 확정 데이터 구성, 보강 후보, 도입 조건과 현재 상태를 한곳에서 추적한다.
데이터별 상세 조사 근거와 라이선스는 `DATA_SOURCES.md`를 정본으로 사용한다.

## 요약

| Stage | 기본 데이터 | 보강 데이터 | 현재 데이터 상태 | 본 학습 |
|---|---|---|---|---|
| Stage 1 재녹화 판별 | CCD | Nexar | 전략·샘플 검증 완료 | 미진행 |
| Stage 2 사고 분석 | CCD 수동 라벨 | Nexar 선별, 필요 시 DADA-2000 | ID 200까지 확인, 74건 저장 | 미진행 |
| Stage 3 차량 거동 | comma2k19 | ZOD | comma 변환 검증 완료, ZOD 승인 대기 | 미진행 |

## Stage 1 — CCD 프로토타입 후 Nexar 추가학습

### 확정 구성

- CCD Crash-1500 원본 1,500개 중 고정 seed로 30개 source를 실제 휴대전화 검증용으로 격리한다.
- 나머지 1,470개에서 ORIGINAL 1개와 합성 RERECORDED 2개씩, 총 4,410개 후보를 생성한다.
- holdout 30개는 휴대전화로 각 3조건을 촬영해 약 90개 실제 RERECORDED 검증 세트로 사용한다.
- CCD 최초 프로토타입과 실제 holdout 평가 후 Nexar train 1,500개를 추가학습 후보로 사용한다.
- Nexar positive/negative는 사고 여부 구분일 뿐 Stage 1 라벨이 아니다. 양쪽 모두 ORIGINAL source이며 각 source에서
  ORIGINAL과 합성 RERECORDED를 함께 생성한다.

### 검증 상태

- 합성 생성기, source-level split, holdout 선택기와 제외 옵션 구현 완료.
- Kaggle에서 MViTv2-S 학습·체크포인트 재로딩·추론 계약 스모크 PASS.
- Nexar positive 2개와 negative 2개를 다운로드·재생하고 ORIGINAL 4개와 합성본 4개 생성 성공.
- Nexar Open Data License 전문 확인 및 사용 결정 기록 완료.

### 다음 단계

1. 실행 환경이 준비되면 CCD holdout 30개를 확정한다.
2. CCD 20개 preview 후 1,470개 전체를 생성한다.
3. 휴대전화 실제 재촬영 검증 세트를 만들고 최초 프로토타입을 평가한다.
4. Nexar 50개 확대 검수 후 필요 시 전체 또는 층화 subset으로 혼합 재학습한다.

## Stage 2 — CCD exact labels 중심, Nexar 선별 보강

### 확정 구성

- CCD 사고 영상 1,500개와 공식 collision 구간 라벨을 기본으로 사용한다.
- `entry_frame`, `evasion_space`, `entry_side`는 공개 데이터에 대회 정의 그대로 존재하지 않아 직접 라벨링한다.
- 최초 프로토타입은 CCD 200~300개 수동 4항목 라벨로 만든다.
- 성능 모델은 active learning으로 CCD 수동 라벨을 500~700개, 수상권 도전 시 800~1,200개까지 확대한다.
- Nexar positive 750개는 실제 물리적 접촉과 근접사고를 먼저 분리한다. 실제 접촉이 있고 차선 진입·좌우 방향·회피
  공간을 판정할 수 있는 영상만 4항목 수동 라벨 후보로 사용한다(목표 100~300개).
- 외부 partial label은 collision head에만 사용하고, 라벨이 없는 head는 masked multi-task loss로 제외한다.

### 보강 우선순위

1. Nexar: 접근성과 라이선스 검증 완료. 대회 정의에 맞는 실제 충돌 영상만 선별한다.
2. DADA-2000: CCD+Nexar로 일반화가 부족할 때 50개로 영상·라이선스를 검증한 뒤 200~300개를 검토한다.
3. DAD/A3D: 배포 안정성과 라이선스가 불명확해 후순위다.
4. DoTA/MM-AU: 원본 확보 또는 출처·라이선스 재현성 문제로 현재 사용하지 않는다.

### 현재 상태와 다음 단계

- CCD 확보·검증, OpenCV 수동 라벨링 GUI, Stage 2 4-output 코드와 정적 계약 검사는 완료됐다.
- CCD 영상 ID 200까지 확인했으며 실제 저장된 수동 라벨은 74건이다. 목표 200~300건에는 아직 미달이다.
- Stage 2 Kaggle 런타임 스모크와 본 학습은 미완료다. 2026-09-06 진행분에 대해서는 추가 평가를 실행하지 않았다.
- 현재 가장 큰 병목은 외부 영상 수가 아니라 exact manual label 수다.

## Stage 3 — comma2k19 기본, ZOD 도시 행동 보강

### 확정 구성

- comma2k19를 기본 학습 데이터로 사용한다. 첫 프로토타입은 약 10GB 청크 하나(약 200분)로 분포와 학습을 확인한다.
- 첫 청크 이후 다른 차량 청크를 추가해 RAV4와 Civic 양쪽을 포함한 약 400분 구성으로 확대한다.
- comma2k19는 고속도로 중심이라 STOPPED, 출발·정지, 교차로 좌우회전과 저속 큰 조향이 부족할 가능성이 높다.
- ZOD Sequences를 도시·저속·정지·회전·악천후 보강 데이터로 사용한다. 접근 신청 메일은 발송 완료했고 승인 대기 중이다.
- ZOD 승인 후 5~10개 변환 스모크, 100개 분포 검사, 필요 시 300~500개 확대 순으로 진행한다.

### 라벨 생성

- 두 데이터셋 모두 별도의 대규모 수동 라벨링은 하지 않는다.
- 영상 timestamp와 속도·종가속도·조향각·yaw rate를 정렬해 10Hz에서 accel/steer 범주를 자동 생성한다.
- 가속·브레이크 페달은 ZOD의 보조 신호로 사용한다.
- 차량·데이터셋별 조향 영점과 단위가 다르므로 동일한 각도 threshold를 공유하지 않는다.
- 변환 후 정지·출발·가감속·좌우회전·차선 변경 장면을 오버레이로 표본 검수한다.

### 후보 판단

- nuScenes는 기술적으로 적합하지만 CC BY-NC-SA 및 상금 대회 사용 가능성이 불명확해 서면 허가 전에는 사용하지 않는다.
- A2D2는 CC BY-ND와 카메라 domain 차이 때문에 후순위다.
- commaSteeringControl은 대응 영상이 없어 직접 영상 모델 학습에는 사용하지 않는다.

### 검증 상태와 다음 단계

- comma2k19 공식 1분 예제에서 1,200 source frames를 10Hz 600 samples로 변환하고 오버레이 검수까지 PASS했다.
- 첫 10GB 청크 확보와 전체 행동 episode 분포 분석, Stage 3 본 학습은 미완료다.
- ZOD 승인 후 전방 카메라와 100Hz vehicle data의 timestamp·부호·필드를 먼저 실측한다.

## 공통 원칙

- 동일 source/route/drive의 인접 영상과 파생본은 하나의 split에만 둔다.
- frame 단위 무작위 split을 금지하고 source·route·차량 단위로 분할한다.
- 외부 데이터 출처가 클래스의 지름길이 되지 않도록 클래스와 데이터셋 비율을 균형화한다.
- 비공개 평가 데이터로 학습·튜닝·pseudo-labeling하지 않는다.
- 외부 데이터의 공식 URL, 버전, 라이선스, 다운로드 일자와 변환 manifest를 보존한다.
- 2차 평가에 제출할 수 없는 출처·라이선스의 데이터는 본 학습에 포함하지 않는다.

## 실행 우선순위

1. Stage 2 CCD 200~300개 수동 라벨링 (다음 재개 지점: ID 201).
2. Stage 2 4-output Kaggle 런타임 스모크와 최초 학습.
3. comma2k19 첫 10GB 청크의 행동 episode 분포 분석과 Stage 3 프로토타입.
4. ZOD 승인 시 5~10개 sequence 변환 스모크.
5. Stage 1 실행 환경이 준비되면 CCD 전체 합성 및 휴대전화 holdout 평가.
6. 각 프로토타입의 실제 실패 유형을 근거로 Nexar·ZOD 등 보강 데이터를 단계적으로 확대한다.
