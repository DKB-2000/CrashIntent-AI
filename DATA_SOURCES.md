# 외부 데이터 출처 기록 (Dacon 236753)

이 문서는 학습에 사용(검토)한 외부 공개 데이터셋의 출처·라이선스·이용조건·확보 방법을 기록한다.
2차 평가 제출물(학습데이터 구성 보고서)에 그대로 재사용할 목적으로 작성한다 — `dacon-236753-대회안내.md` 8장 참고
("2차 평가 제출물에는 사용한 사전학습 모델/API/외부 데이터의 출처를 명시해야 함").

**작성 원칙**: 실제로 다운로드·사용을 확정한 데이터셋만 "사용" 상태로 표시한다. CCD(1번)는 사용자가 라이선스 리스크를
감수하고 승인하여 **실제 다운로드·검증까지 완료**했다(2026-09-02). 그 외 항목은 아직 **조사 단계**이며 다운로드하지 않았다.

---

## 조사 일자: 2026-09-02 (Stage 2 후보 조사) / 다운로드·검증 일자: 2026-09-02

## 1. CCD (Car Crash Dataset) — 최우선 후보, 데이콘 베이스라인이 실제로 참조한 데이터셋으로 확인됨. **다운로드完·구조 검증完**

> **상태: 사고 영상(Crash-1500) 1,500건 + 라벨 파일 확보 완료.** 정상주행(Normal) 3,000건(BDD100K 파생)은 Stage2
> 4항목 라벨링에 직접 쓰이지 않아 **의도적으로 다운로드하지 않음**(아래 "다운로드 범위" 참고). 아래 표는 실제로 받은
> 파일을 직접 열어 검증한 결과로 갱신했다 — 조사 단계의 추정치와 다른 부분은 볼드로 표시.

| 항목 | 내용 |
|---|---|
| 정식 명칭 | Car Crash Dataset (CCD) |
| 논문 | Bao, Wentao / Yu, Qi / Kong, Yu, *"Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning"*, ACM Multimedia Conference (ACM MM), 2020년 5월 |
| 저장소 (데이터) | https://github.com/Cogito2012/CarCrashDataset |
| 저장소 (코드/모델, UString) | https://github.com/Cogito2012/UString |
| 논문 원문 | https://arxiv.org/abs/2008.00334 |
| 다운로드 | Google Drive, 신청 절차 없이 공개 링크로 직다운로드 확인됨. **실제 사용한 링크(README 원문 재확인 결과, 조사 단계에서 파악한 링크와 폴더ID가 다름 — 동일 데이터셋의 다른 공유 링크로 추정): `https://drive.google.com/drive/folders/1ao-wCdQkWRYJMtWlDLPQEg9_y9d5K803`.** `gdown`으로 폴더 전체를 스캔한 결과 총 4,511개 항목(`vgg16_features/` 4,500개 npz + `videos/`(Crash-1500.zip·Normal.zip·Crash-1500.txt) + `codes/`+`README.md`)으로 구성 — 즉 원본 영상은 압축 zip 2개로만 제공되고, 개별 mp4 파일 형태로 폴더 안에 흩어져 있지 않음 |
| **다운로드 범위 (이번 작업)** | `Crash-1500.zip`(사고 영상, 실측 793,091,291 bytes ≈ 756MiB) + `Crash-1500.txt`(라벨, 실측 276,792 bytes) + `README.md`만 다운로드. **`Normal.zip`(정상주행 3,000건)과 `vgg16_features/`(4,500개 npz, 사전추출 VGG16 특징 — 우리 파이프라인에서 쓸 원본 프레임이 아님)는 의도적으로 미다운로드.** `Normal.zip`은 시도 시점에 Google Drive가 HTML 확인 페이지(2,009 bytes)만 반환해 정상 다운로드가 되지 않는 상태였음(과다 요청으로 인한 일일 다운로드 쿼터 초과로 추정) — 필요 시 추후 재시도 |
| 규모 (실측) | **확보분: 사고 영상 1,500개**(zip 756MiB → 압축 해제 후 825MB, 총 1.6GB 순간 사용 후 원본 zip 삭제해 최종 825MB로 정리). 조사 단계에서 파악한 "총 4,500개(사고 1,500+정상 3,000)"는 CCD 데이터셋 전체 규모로는 맞으나, **이번에 실제로 받은 것은 사고 1,500건만**. `train.txt`/`test.txt`(분할 파일)는 `vgg16_features/` 하위에 있으며 이번엔 받지 않음(원본 영상 분할과 무관, feature 파일용) |
| 영상 형식 (실측, ffprobe 대신 OpenCV `cv2.VideoCapture`로 4개 샘플 검증: `000001/000250/000750/001500.mp4`) | **10.000fps, 해상도 1280×720, 50프레임, 길이 5.00초 — 4개 샘플 전부 동일.** 조사 단계에서 "해상도 미상"이었던 부분이 **1280×720으로 확정됨** |
| 데이터 출처(중요) | **사고 영상 1,500개는 유튜브 채널에서 수집한 실제 교통사고 영상**(트리밍). **정상주행 3,000개는 BDD100K 데이터셋에서 무작위 샘플링**(이번엔 미다운로드) — 즉 CCD 자체가 1차 촬영물이 아니라 두 외부 소스의 재가공/재배포본. README 원문에서 재확인 완료 |
| 대시캠 스타일 유사도 | 사고 영상은 실제 대시캠 촬영물이 다수 포함되어 블랙박스 영상과 유사도 높음. 다만 유튜브 편집본(다양한 카메라 각도·화질·자막 삽입 등 혼재) 가능성 있어 전수 검수 필요 |
| **제공 라벨 (실제 파일 파싱으로 검증 완료)** | `Crash-1500.txt`(1,500줄) 컬럼: `vidname`(영상명, 000001~001500) · `binlabels`(**실측: 전 행 길이 정확히 50, 값은 {0,1} 두 종류만 — 정합**) · `startframe`(원본 유튜브 영상 내 시작 프레임) · `youtubeID` · `timing`(**실측 값셋: {Day, Night}**) · `weather`(**실측 값셋: {Normal, Snowy, Rainy}**) · `egoinvolve`(**실측 값셋: {Yes, No}**). README 원문에 "Current version of CCD only provides the temporal annotations and environmental attributes listed above. For more detailed annotations such as traffic accident reasons and tracklets, we will release them soon."라고 명시 — **조사 단계에서 언급했던 "사고 참가자·원인 서술 annotation"은 현재 배포본에는 없음(추후 릴리즈 예고만 있음)으로 정정** |
| **대회 4항목 커버 범위 (실측 검증)** | `collision_frame`(충돌시점) 관련 정보만 제공 — `binlabels`에서 처음으로 1이 되는 인덱스(=사고 구간 시작)를 1,500건 전수 계산한 결과 **범위 30~49(50프레임 중), 평균 37.2**, 전 영상이 최소 1개 이상의 양성 프레임을 가짐(누락 0건). 데이콘 학습 코드 주석("공개 CCD 5건에는 충돌 구간만 공식 주석") 및 라벨 CSV(`path,t_collision`)와 정확히 일치. **`entry_frame`(진입시점) · `evasion_space`(회피공간 여부) · `entry_side`(진입방향)에 해당하는 공식 라벨은 없음** — README 원문으로 재확인, 직접 라벨링 필요 |
| 라이선스 — 코드 | 저장소(`Cogito2012/CarCrashDataset`)에 **MIT License** 파일 존재(2020, Wentao Bao). 단, LICENSE 파일 문구는 "the Software"(코드·스크립트)에 대한 것으로, 저장소 관행상 **영상 데이터 자체에 대한 재배포 라이선스인지는 불명확** |
| 라이선스 — 데이터(중요 유의점) | README에 **데이터셋 자체에 대한 명시적 라이선스 문구는 없음**(사용 조건 문의는 저자 이메일 `wb6219@rit.edu`로 안내). 원본 사고 영상은 **제3자(유튜브 업로더)가 저작권을 보유한 콘텐츠를 재편집한 것**이고, 정상주행 영상(미다운로드분)은 **BDD100K 라이선스**를 따른다. BDD100K는 "교육·연구·비영리 목적 무료 이용 허용, 상업적 이용은 BDD/BAIR Commons 멤버·제휴사에 한정"이라는 연구용 라이선스 문구가 확인됨(`https://github.com/bdd100k/bdd100k/blob/master/LICENSE`, `bdd-data.berkeley.edu`) — **완전한 "법적 제한 없음"은 아니고 "비영리 연구 목적" 조건부** |
| **대회 요건 부합 여부 (사용자 승인 기록)** | 대회 규칙은 "법적 제한이 없는" 외부 데이터 사용을 허용하며, 대회 자체도 비영리 정부 주관 사업이다. CCD는 (a) MIT 라이선스 코드 + (b) 출처 불명확한 유튜브 재편집 영상 + (c) 비영리 연구 목적 한정인 BDD100K 파생 영상(단, 이번엔 미다운로드)으로 구성되어 "완전히 제약 없는 데이터"는 아니다. **사용자가 이 라이선스 리스크를 인지하고 2026-09-02 다운로드를 명시적으로 승인함.** 2차 평가 제출 시 위 라이선스 불확실성을 학습데이터 구성 보고서에 그대로 명시할 것 |
| 확보 난이도 / 상태 | **완료** — Google Drive 직다운로드로 손쉽게 확보. 로컬 경로: `crashvideo-project/data_raw/ccd/`(git 추적 제외, `.gitignore`에 `/crashvideo-project/data_raw/` 추가 완료) |
| 로컬 파일 구성 (실측) | `data_raw/ccd/Crash-1500.txt`(라벨, 270KB) · `data_raw/ccd/README.md`(원본 README) · `data_raw/ccd/videos/Crash-1500/000001.mp4`~`001500.mp4`(825MB, 1,500개) · `data_raw/ccd/_validation_report.json`(아래 검증 스크립트 출력 요약). 원본 `Crash-1500.zip`은 압축 해제 검증(1,500개/1,500개 일치, 누락·초과 0건) 후 디스크 절약을 위해 삭제함 |
| 다음 액션 | (1) `entry_frame`/`evasion_space`/`entry_side` 직접 라벨링 워크플로 설계(이번 작업 범위 밖), (2) 이 원본을 대회 Stage2 폴더 규격(`data/stage2/images/<ID>/frame_NNNNNN.jpg` + `labels.csv`)으로 변환하는 작업은 `crashvideo-model-expert`/후속 작업에서 별도 진행, (3) 필요 시 `Normal.zip`(BDD100K 파생 3,000건) 재다운로드 시도 — Stage1(재녹화 판별)이나 다른 용도로 필요해지면 그때 쿼터 회복 여부 확인 후 진행 |

---

## 2. DoTA (Detection of Traffic Anomaly) — entry_frame 공백을 메울 후보로 조사. **조사 완료·트랙 종결(2026-09-02)**

> **결론(최종): `entry_frame`/`collision_frame` 대리 라벨은 부적합(N, 확정) — 4,376개 영상 전수 조사로 확정.**
> `entry_side`/`evasion_space`는 **보류** — 좌표만으로는 "신호 후보"가 존재함을 확인했으나(조건부), 실물 영상으로
> 검증을 시도한 결과 **원본 유튜브 소스 영상이 96/96 시도 전부 재생 불가(대부분 업로더에 의해 삭제됨)**로 확인되어
> 이 방법으로는 실물 검증 자체가 불가능함을 확정하고 트랙을 종결한다. 상세: "4) 핵심 검증"(entry_frame), "5)
> 좌표 예비 점검"(신호 존재), "10) 실물 다운로드 시도 결과"(검증 불가 원인 진단), "11) 최종 결론"(종합) 참고.

### 1) 라벨 JSON 다운로드 (실측 — 이미지·영상 55GB는 받지 않음)

GitHub 저장소의 `dataset/` 폴더가 이미지 없이 라벨만 별도로 제공한다는 것을 확인하고 이것만 받았다(전부 `raw.githubusercontent.com`,
Google Drive 쿼터 이슈 없음):

| 파일 | 실측 용량 | 내용 |
|---|---|---|
| `DoTA_annotations.zip` | 10,357,806 bytes (≈9.9MiB) → 압축 해제 시 142MB, **개별 영상당 JSON 1개 × 4,677개** | 영상별 전체 프레임 단위 bbox 궤적 포함 |
| `metadata_train.json` / `metadata_val.json` | 696,987 / 295,463 bytes | 영상별 요약 메타데이터(train/val 분할) |
| `train_split.txt` / `val_split.txt` | 62,225 / 26,638 bytes | 분할 영상ID 목록 |
| `DoTA_urls.txt` / `broken_urls.txt` | 8,096 / 263 bytes | 원본 유튜브 소스 URL 184개(깨진 링크 목록 별도) |

**총 다운로드량 약 11.4MB** — 지시대로 55GB 이미지/영상은 받지 않았다. 로컬 경로: `crashvideo-project/data_raw/dota/`.

### 2) 개별 영상 JSON 실측 구조

한 파일(`0RJPQ_97dcs_000199.json`)을 직접 열어 확인한 top-level 필드: `video_name`, `channel`, `num_frames`, `ignore`,
`ego_involve`, `night`, `anomaly_start`, `anomaly_end`, `video_start`, `video_end`, `accident_id`, `accident_name`, `labels`(프레임별 배열).
`labels[i]`는 `frame_id`, `image_path`, `accident_id`, `accident_name`, `objects`(바운딩박스 배열)로 구성.

**바운딩박스 궤적 구조(실측)**: 좌표계는 **정규화값이 아닌 원본 프레임 픽셀 좌표**(`[x1, y1, x2, y2]`, 예: `[662.3, 165.2, 1279.6, 633.4]` — 1280×720 프레임과 일치). 객체당 `obj_track_id`(프레임 간 추적 ID), `category`(car/truck/rider/motor/bus/person/bike), `category ID`, `trunc`(화면 잘림 여부) 필드 보유. **ego/non-ego 구분은 객체 단위가 아니라 영상 전체 단위**(top-level `ego_involve` 불리언 하나) — 즉 "이 객체가 자차와 충돌한 그 객체인지"를 프레임별로 구분하는 필드는 없다(대부분 영상에서 tracked object가 1개뿐이라 실무상 모호함은 적음).

### 3) 카테고리 실측 건수 (4,677건 전수 집계 — 논문은 차트 이미지로만 제시했던 것을 이번에 숫자로 확정)

| accident_id | accident_name | 건수 | 논문 Table 2 대응(추정) |
|---|---|---|---|
| 5 | **turning**(진입/횡단) | **1,696 (36.3%)** | TC |
| 3 | **lateral**(측면 동일방향) | **726 (15.5%)** | LA |
| 2 | moving_ahead_or_waiting | 663 (14.2%) | AH |
| 4 | oncoming | 478 (10.2%) | OC |
| 9 | leave_to_left | 370 (7.9%) | OO(좌) |
| 8 | leave_to_right | 362 (7.7%) | OO(우) |
| 6 | pedestrian | 100 (2.1%) | VP |
| 1 | start_stop_or_stationary | 95 (2.0%) | ST |
| 7 | obstacle | 95 (2.0%) | VO |
| 10 | unknown | 92 (2.0%) | UK |

**정정 사항**: 논문 Table 2는 "OO(이탈)"을 좌/우 구분 없이 1개 카테고리로 제시했지만, **실제 배포 라벨은 `leave_to_left`/`leave_to_right`로 이미 좌우가 분리되어 있다** — 논문 서술과 실제 배포 스키마가 완전히 같지 않다는 것을 실측으로 확인(사소하지만 "논문만 보고 판단하면 안 된다"는 근거).
추가 통계: `ego_involve` True 2,724 / False 1,953, `night` True 587 / False 4,090, `ignore` True 61(제외 권장) / False 4,616.
바운딩박스 보유 영상 **4,376/4,677건(93.6%)**, 나머지 301건은 궤적 없음. 객체 클래스 전체 분포: car 124,368 · truck 20,785 · rider 10,417 · motor 7,852 · bus 3,469 · person 2,817 · bike 1,031(바운딩박스 인스턴스 수 기준).

**대회 유형과의 겹침 판단**: 대회가 다루는 "피해차량이 피의차량 차선에 진입해 충돌"에 개념적으로 가장 가까운 것은 **turning(1,696건) + lateral(726건) = 2,422건(전체의 51.8%)** — 규모 자체는 충분히 크다. 단, 이는 "영상 단위의 굵은 범주가 개념적으로 겹친다"는 뜻이지, `entry_frame`/`entry_side`/`evasion_space`를 프레임 단위로 정확히 제공한다는 뜻은 아니다(아래 핵심 검증 참고).

### 4) 핵심 검증 — 궤적이 `anomaly_start` 이전에도 존재하는가? (4,376건 전수, 이미지 없이 좌표만으로 확정)

`entry_frame`을 유도하려면 상대 차량이 화면에 나타나 차선에 들어오는 **`anomaly_start` 이전** 구간의 궤적이 필요하다.
바운딩박스가 있는 4,376개 영상 전체를 대상으로 "첫 바운딩박스가 찍힌 프레임 vs `anomaly_start`"를 비교한 결과:

| 항목 | 건수 (비율) |
|---|---|
| 첫 bbox 프레임 < `anomaly_start` (사전 궤적 존재) | **0건 (0.0%)** |
| 첫 bbox 프레임 == `anomaly_start` | 4,352건 (99.5%) |
| 첫 bbox 프레임 > `anomaly_start` | 24건 (0.5%) |
| `anomaly_start` 대비 5프레임 이상 앞선 사전 궤적 보유 | **0건 (0.0%)** |

**결론: 4,376건 전수에서 `anomaly_start` 이전 구간의 바운딩박스는 단 한 건도 존재하지 않는다.** 논문이 "궤적은 `anomaly_start`부터 `anomaly_end`(또는 화면 밖으로 나갈 때까지) 라벨링했다"고 밝힌 것과 정확히 일치하는 실측 결과다. 즉 주석자들은 애초에 "사고 발생 전" 구간에는 바운딩박스를 그리지 않았다 — 원본 영상에 그 장면이 존재하더라도 **라벨 데이터 자체에는 없다.** 따라서 궤적 데이터만으로 `entry_frame`(진입 시점)을 역산하는 것은 **원천적으로 불가능**하며, 이는 이미지를 봐야 판단할 수 있는 사안이 아니라 라벨 스키마 자체의 구조적 한계다. **이 부분은 "부적합"으로 최종 확정하고 추가 검증 없이 조사를 마친다.**

### 5) `entry_side`/`evasion_space` 유도 가능성 — 좌표만으로 예비 점검 (이미지 미검수)

`entry_frame`과 달리 `entry_side`/`evasion_space`는 `anomaly_start` **시점의** 위치 정보만 있으면 되므로(진입 "과정"이 아니라 "그 순간의 상태"), 궤적이 그 시점부터 시작해도 원리적으로는 시도해볼 수 있다. 개념적으로 가장 가까운 turning/lateral 2,412건(bbox 보유분)을 대상으로 좌표만으로 예비 신호를 점검했다:

- `anomaly_start` 시점 바운딩박스 중심 x좌표(프레임 폭 1280 기준): 좌측 40% 미만 1,013건(42.0%) · 중앙 665건(27.6%) · 우측 60% 초과 734건(30.4%) — **분포가 한쪽으로 쏠리지 않고 고르게 퍼져 있어, 위치 정보가 "변별력 있는 신호"일 가능성은 있음**
- `anomaly_start`→+5프레임 이동 방향: 좌로 이동 926건 · 우로 이동 1,004건 · 거의 정지 434건 — 이동 방향도 고르게 분포

이 결과는 "신호가 존재한다(=상수가 아니다)"는 것만 보여줄 뿐, **"이 좌표가 실제 진입방향/회피공간과 맞는지"는 실제 영상을 봐야 확인 가능**하다 — 좌표 계산만으로는 검증 불가능한 영역이라 여기서 멈췄다.

### 6) 이미지/영상 검수 경로 조사 (당시엔 다운로드 전 — 이후 10번에서 실제 시도)

55GB 전체를 받지 않고 소량만 접근하는 방법을 먼저 찾아봤다:
- **Kaggle/HuggingFace 등 개별 클립 단위 미러**: 검색 결과 존재하지 않음(공식 GitHub·Google Drive가 유일한 배포처로 확인됨).
- **55GB Drive(5분할)**: 분할해도 파트당 약 11GB로 추정 — "소량"이 아님.
- **`DoTA_urls.txt`의 원본 유튜브 소스(184개 링크)**: 라벨 파일명(`{유튜브ID}_{프레임오프셋}.json`)과 대조한 결과 4,677개 클립이 **약 190개의 원본 유튜브 영상**(사고 모음/편집 채널)에서 나온 것으로 확인(평균 클립 24.6개/영상) — 즉 원본은 짧은 클립이 아니라 여러 사고를 이어붙인 긴 편집 영상. `yt-dlp` 등으로 필요한 구간(`video_start`~`video_end`, 10fps 프레임 오프셋을 초 단위로 환산)만 잘라 받으면 영상 전체보다는 훨씬 작아질 것으로 예상되나, **실제 다운로드 전이라 정확한 용량은 확정하지 못함**(대략 20~30개 클립 × 5~15초 720p 기준 어림잡아 수백MB대로 추정 — 확정치 아님, 소스 영상 삭제·비공개 가능성도 있음(`broken_urls.txt`에 이미 깨진 링크 존재)).

**→ 지시에 따라 이 이상은 진행하지 않고 여기서 멈춘다.** 실제 이미지/영상이 필요한 부분은 오직 5번(entry_side/evasion_space 좌표 신호의 실물 검증)뿐이며, entry_frame 관련 결론(4번)은 이미지 없이 이미 확정됐다.

### 7) 기본 서지·라이선스 정보 (1차 조사에서 이미 확인, 요약 유지)

| 항목 | 내용 |
|---|---|
| 정식 명칭 | DoTA (Detection of Traffic Anomaly) |
| 논문(원 발표) | Yao, Yu / Wang, Xizi / Xu, Mingze / Pu, Zelin / Atkins, Ella / Crandall, David, *"When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos"*, arXiv:2004.03044, 2020년 4월 |
| 논문(확장 저널판) | Yao, Yu / Wang, Xizi / Xu, Mingze / Pu, Zelin / Wang, Yuchen / Atkins, Ella / Crandall, David, *"DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos"*, IEEE TPAMI, 2022 |
| 공식 저장소 | https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly |
| 규모 (논문+실측 일치 확인) | 총 4,677개 영상(1280×720), 원본 30fps→10fps로 다운샘플링, train 3,275 / test 1,402(VAD 기준) |
| 라이선스 — 코드 | 저장소에 MIT License(2020, Yu Yao) — 코드/스크립트에 대한 것이고 영상 콘텐츠 라이선스는 별도 |
| 라이선스 — 데이터 | 논문 PDF 전문에 라이선스 관련 문구 전혀 없음(검색으로 확인). 유튜브 6,000여 클립을 수집·재배포한 것이라 CCD와 동일한 리스크 패턴 |
| 대회 요건 부합 여부 | "완전히 제약 없는 데이터"는 아니고 "비영리 연구 목적 관행" 수준. 다운로드를 확대하려면 CCD와 동일하게 사용자의 명시적 리스크 인지·승인 필요 |

### 8) 최종 판단 (Y / N / 조건부)

| 대회 라벨 | 판단 | 근거 |
|---|---|---|
| `entry_frame` | **N (부적합, 확정)** | `anomaly_start` 이전 궤적이 4,376건 전수에서 0건 — 라벨 스키마 자체의 구조적 한계, 이미지 검수로도 뒤집을 수 없는 사실 |
| `collision_frame` | **N (부적합)** | `anomaly_end`는 "이상 객체가 화면에서 사라지거나 정지한 시점"으로 정의돼 충돌 시점과 다름(CCD가 이미 이 라벨을 담당하므로 실익도 없음) |
| `entry_side` | **보류(조건부 신호만 확인, 실물 검증 불가로 트랙 종결)** | `anomaly_start` 시점 bbox 중심좌표가 좌/중/우로 고르게 분포(42/28/30%)하고 이동방향도 고르게 분포 — 좌표상 신호는 존재. 하지만 실제 영상으로 검증을 시도한 결과 **원본 유튜브 소스 영상이 거의 전멸**해(10번 참고) 이 신호가 실제 진입방향과 일치하는지 확인할 방법이 없다. Y/N 확정 불가 — "조건부 신호 확인, 실물 미검증"으로 종결 |
| `evasion_space` | **보류(조건부 신호만 확인, 실물 검증 불가로 트랙 종결)** | 같은 이유로 프레임 내 여유 공간을 bbox 점유율로 근사 계산하는 것은 원리적으로 가능하나, 실물 검증이 불가능해 신뢰도를 확정할 수 없다 |

### 9) 다음 액션 제안 (승인 필요)

- **`entry_frame`/`collision_frame` 관련 DoTA 조사는 여기서 종료한다** — 궤적 데이터로 대체 불가능함이 전수 조사로 확정됐고, 추가 이미지 검수를 해도 결론이 바뀌지 않는다.
- **`entry_side`/`evasion_space` 조건부 트랙: 사용자가 실물 검수를 승인해 실제로 시도했으나, 원본 유튜브 소스 영상이 거의 전멸해 실물 검증 자체가 불가능함을 확인하고 트랙을 종결했다.** 상세 과정은 아래 10번 참고.
- 55GB 이미지/영상 전체 다운로드는 **끝까지 진행하지 않았다** — 55GB를 받아도 원본 소스가 소실된 것이 아니라 2020년 수집 당시 Google Drive에 미리 백업해 둔 프레임이므로 이론적으로는 그 안에 이미지가 존재할 수 있으나, 이번 승인 범위(유튜브에서 클립 받기)를 벗어나는 별도 승인이 필요한 대용량(수십GB) 작업이라 이번 트랙에서는 시도하지 않았다(11번 "남는 선택지" 참고).

### 10) 실물 다운로드 실제 시도 결과 (2026-09-02) — 트랙 종결, 정확한 원인 진단

사용자 승인을 받아 `yt-dlp`로 원본 유튜브에서 직접 클립을 받아보았다. 결과: **총 96회의 실제 다운로드 시도(turning/lateral/leave_to_left/leave_to_right 카테고리, 좌/중/우 버킷 균형 배분) 중 성공 0건, 실패 96건.** 이전 라운드의 사전 가용성 점검(서로 다른 소스 영상 60개, `--simulate`만 수행)까지 합치면 **150개 이상의 서로 다른 원본 유튜브 영상을 확인했고 전부 재생 불가였다.**

**원인 진단(추측이 아니라 대조 실험으로 확인)**:
1. **네트워크 차단 아님**: 이 환경에서 YouTube 자체는 정상 접근된다. 임의의 생존 영상(`dQw4w9WgXcQ`)으로 대조 테스트한 결과, `android` 클라이언트로 전환하니 실제 포맷 목록까지 정상적으로 받아왔다(`[info] Downloading 1 format(s): 18`) — 웹페이지·플레이어 API 통신 자체는 문제없이 성공.
2. **yt-dlp 설치 문제 아님**: 위와 같이 도구 자체는 정상 동작. 대조 영상에서 실제로 막힌 지점은 로컬에 `ffmpeg`이 설치돼 있지 않아 구간 자르기(`--download-sections`)를 못한 것뿐이었다(`ERROR: ... ffmpeg is not installed`) — 이는 부차적 문제이고, 아래 3번이 결정적 원인이라 실익이 없어 별도로 고치지 않았다.
3. **레이트리밋이 주원인 아님**: 96회 중 초반 약 85회는 순수하게 `This video is unavailable` / `Video unavailable. This video has been removed by the uploader`로 즉시 실패했다(HTTP 429 없이). 마지막 11회에서만 429(Too Many Requests)가 섞여 나왔는데, 이는 짧은 시간에 반복 요청한 데 따른 부수 효과이지 근본 원인이 아니다.
4. **결정적 원인: DoTA 라벨 파일명에 담긴 원본 유튜브 소스 영상 자체가 대부분 삭제됨.** 같은 `android` 클라이언트로 대조 영상은 포맷까지 받아왔는데 DoTA 소스 영상(`TNZv-NBcV5U`)은 포맷 조회 이전 단계("영상 존재 확인")에서부터 곧바로 "This video is unavailable"로 끝났다 — 이는 우리 쪽 문제가 아니라 **유튜브에 그 영상 자체가 더 이상 없다는 뜻**이다. 다수의 실패 메시지가 `"Video unavailable. This video has been removed by the uploader"`로 명시적 삭제임을 알려준다.

**의미**: 이 데이터셋은 사고 모음/재편집 채널(`AnAn`, `CarCrashesTime`)에서 수집한 영상으로 구성되는데(3번 섹션에서 이미 확인), 이런 "사고 영상 재편집 채널"은 저작권 침해로 삭제되는 사례가 많다 — 2번 섹션에서 이미 "라이선스 — 데이터" 항목에 적어둔 **"CCD와 동일한 유형의 리스크(제3자 유튜브 저작권 미해결 상태)"가 실제로 6년 뒤(2020→2026) 대규모 삭제라는 형태로 현실화된 것을 실측으로 확인**한 셈이다.

**시도 중단(재시도 안 함)**: 지시대로 96회 실패 시점에서 재시도를 멈췄다. 다운로드된 검증용 클립은 **0개**다(`crashvideo-project/data_raw/dota/verification_clips/`는 빈 폴더로 남아있음). 과정 기록은 `data_raw/dota/_candidates.json`(후보 2,908건), `_selected_samples.json`(1차 선정 24건), `_availability.json`(사전 점검 60건), `_download_log.txt`(실제 시도 96건 전체 로그)에 남겨뒀다.

### 11) 최종 결론 및 남는 선택지

**`entry_side`/`evasion_space` 최종 판단: 보류(조건부 신호는 확인, 실물 검증은 이 방법으로는 불가능).** Y도 N도 확정할 근거가 없다 — 좌표 신호는 존재를 확인했지만(5번 섹션), 그 신호가 실제로 맞는지 확인할 자료 자체를 구할 수 없었다.

DoTA를 통한 entry_side/evasion_space 조건부 트랙에서 남는 선택지(모두 이번 작업 범위 밖, 필요 시 별도 승인 후 진행):
- (a) **55GB Google Drive 프레임 아카이브**(2020년 수집 당시 미리 저장된 것이라 유튜브 삭제와 무관하게 이미지가 존재할 가능성이 높음)에서 이번에 선정한 20~30건에 해당하는 이미지만 부분적으로 받을 수 있는지 확인 — 단, 5개 분할 파일 단위(파트당 약 11GB 추정)라 "필요한 것만 콕 집어" 받기 어려울 수 있고, 이번 승인 범위(수백MB급 유튜브 클립)를 크게 초과하는 별도 승인 대상.
- (b) 이 트랙을 여기서 접고 **CCD 기반 수작업 라벨링**(1번 섹션)에 집중 — 원본 영상이 CCD 쪽은 이미 로컬에 1,500건 확보돼 있어 즉시 가능.

---

## 3. 기타 Stage 2 대체/보강 후보 (조사 단계, 미다운로드)

| 데이터셋 | 출처 | 규모 | 라벨 | 대회 4항목 커버 | 라이선스 메모 | 비고 |
|---|---|---|---|---|---|---|
| **DAD** (Dashcam Accident Dataset) | Chan et al., ACCV 2016. Taiwan 6개 도시 대시캠 | 1,750개 영상(사고 620 + 정상 1,130), 100프레임/5초 클립, train 1,284 / test 466 | 사고 여부 + 사고 시점(ToA) 프레임 | collision_frame만 | 학술 공개, 상세 라이선스 문구 미확인 — 재확인 필요 | CCD와 함께 accident-anticipation 연구에서 가장 널리 쓰이는 원조 데이터셋. 대만 현지 대시캠이라 국내 블랙박스와 화각·화질 유사도 상대적으로 높을 가능성 |
| **A3D** (AnAn Accident Detection) | 원래 Traffic Anomaly Detection용, East Asia 도심 대시캠. DoTA 논문이 비교 대상으로 인용(Yao et al., 같은 저자 그룹의 전작) | 1,500개 영상, temporal(시작/종료) 주석만 | 이상구간 시작/종료 — **DoTA 논문에 따르면 A3D는 "충돌이 발생한 시점"을 anomaly start로 정의**(DoTA와 다름, CCD/DAD와 유사) | collision_frame(근사)만 | 미확인 — 재확인 필요 | ego-vehicle 연루 비율 60%+ (DAD 대비 높음) |
| **CADP** (CCTV Accident Detection) | 논문: "CADP: A Novel Dataset for CCTV Traffic Camera based Accident Analysis" | 확인 필요 | 사고 시공간 위치 | collision_frame 근사 | 미확인 | **CCTV 고정 카메라 시점** — 블랙박스(차량 탑재, 이동 시점) 영상과 화각·구도가 근본적으로 다름. 데이터 소스로서 우선순위 낮음 |
| **Nexar Dashcam Collision Prediction Dataset** | Moura, Zhu, Zvitia (Nexar), CVPR 2025 Workshop. https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction | 1,500개 영상(50% 충돌/임박 충돌, 50% 정상). 해상도 1280×720, 30fps, train ~40초/test ~10초 | `label`(충돌여부), `time_of_event`, `time_of_alert`(양성만), `light_conditions`, `weather`, `scene`, `time_to_accident`(test만) | `time_of_event` ≈ collision_frame(시간축) 근접. entry/evasion/side 없음 | 라이선스명만 "nexar-open-data-license"로 확인, **전문 미확인 — 상업적 이용/재배포 제한 여부 불명** | 실제 대시캠 원본(30fps 고해상도)이라 화질·프레임레이트가 CCD/DAD보다 우수하고 블랙박스와 가장 유사. 다만 규모가 작고(1,500개) 라이선스 전문 확인이 선행되어야 함 |

---

## 4. 종합 판단 및 다음 액션 제안

1. **CCD 사고 영상(Crash-1500) 1,500건 + 라벨 확보 완료(2026-09-02)** — 데이콘 베이스라인이 명시적으로 참조("공개 CCD 5건")한 데이터셋이며, `collision_frame` 학습에 그대로 쓸 수 있는 구조(50프레임 클립 + 이진 라벨)가 대회 Stage2 입력 구조(영상별 프레임 폴더)와 변환이 쉽다는 점을 실측으로 재확인(10fps·1280×720·50프레임 전 샘플 일치, 1,500/1,500 파일-라벨 매칭 누락 0건). 로컬 경로: `crashvideo-project/data_raw/ccd/`(1번 표 참고).
2. **`entry_frame`/`evasion_space`/`entry_side` 3개 항목은 CCD·DoTA를 포함해 지금까지 조사된 모든 공개 데이터셋에서 "그대로 쓸 수 있는" 공식 라벨이 존재하지 않는다** — 데이콘 베이스라인 주석과 정확히 일치하는 결과다. 직접 수작업 라벨링(프레임 단위 확인) 워크플로가 여전히 필수다.
3. **DoTA 조사 최종 결론(2026-09-02, 라벨 실측 + 실물 검수 시도까지 완료, 트랙 종결): `entry_frame`/`collision_frame` 대리 라벨은 부적합(N, 확정), `entry_side`/`evasion_space`는 보류.** `anomaly_start` 이전 구간의 바운딩박스 궤적이 4,376건 전수에서 **0건**이라 entry_frame 관련 조사는 종료했다. `entry_side`/`evasion_space`는 좌표상 신호 후보(turning/lateral 2,412건 대상 좌/중/우 42/28/30% 분포)가 있어 실물 검증을 시도했으나, **원본 유튜브 소스 영상 96/96(100%) 다운로드 실패** — 대조 실험으로 네트워크·도구 문제가 아니라 **원본 영상 자체가 삭제된 것**임을 확정했다(사고 재편집 채널 특유의 저작권 삭제 리스크가 6년 뒤 실제로 현실화된 사례). 그 결과 이 방법으로는 Y/N을 확정할 수 없어 "조건부 신호만 확인, 실물 미검증"으로 트랙을 종결한다. 상세 근거·표는 위 2번 섹션(특히 10·11번) 참고.
4. **라이선스 관련 미해결 사항**:
   - CCD/DAD/A3D/DoTA 모두 "저장소 코드는 MIT"이지만 "재배포된 사고 영상 자체(유튜브 원본)"의 저작권 상태는 불명확 — 비영리 연구 목적 관행으로 널리 통용되나 100% 법적으로 확정된 것은 아님. DoTA는 논문 PDF 전문 검색으로도 라이선스 관련 문구가 전혀 없음을 확인(CCD와 동일한 리스크 패턴).
   - BDD100K 파생분(CCD의 정상주행 3,000개)은 "비영리 연구 목적" 조건부 라이선스 확인됨.
   - Nexar 데이터셋은 라이선스 전문을 아직 확인하지 못함.
   - → **사용자에게 이 불확실성을 명시적으로 알리고, 실제 다운로드/학습 사용 전 최종 승인을 받는 것을 권장**(CCD는 이미 이 절차를 거쳐 승인·다운로드함).
5. **DoTA는 라벨 JSON(약 11.4MB)만 다운로드·검증했고, 55GB 이미지/영상 아카이브는 끝까지 받지 않았다.** 사용자 승인을 받아 원본 유튜브에서 20~30건 실물 검수를 시도했으나 소스 영상이 전멸해(96/96 실패) 클립을 하나도 확보하지 못했고, 이 사실 자체가 결정적 결론(트랙 종결)이 되었다. DAD/A3D/CADP/Nexar는 여전히 조사 단계다.
6. 모델 구현(라벨 CSV 변환 `path,t_collision` 등 베이스라인 스키마 매핑, `data/stage2/images/<ID>/frame_NNNNNN.jpg` 규격으로의 변환)은 이번 작업 범위 밖 — `crashvideo-model-expert`와 협의해 후속 진행.

## 참고 링크 (전체)

- CCD 데이터: https://github.com/Cogito2012/CarCrashDataset
- CCD 코드(UString, ACM MM 2020): https://github.com/Cogito2012/UString
- CCD 논문: https://arxiv.org/abs/2008.00334
- BDD100K 라이선스: https://github.com/bdd100k/bdd100k/blob/master/LICENSE , https://bdd-data.berkeley.edu/
- DoTA 저장소: https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly
- DoTA 라벨 파일(dataset/ 폴더, 실제 사용한 다운로드 소스): https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly/tree/master/dataset (`DoTA_annotations.zip`, `metadata_train.json`, `metadata_val.json`, `train_split.txt`, `val_split.txt`, `DoTA_urls.txt`, `broken_urls.txt`)
- DoTA 논문(원 발표, arXiv): https://arxiv.org/abs/2004.03044
- DoTA 논문(확장 저널판, TPAMI 2022): Yao et al., "DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos" (IEEE TPAMI)
- Nexar Collision Prediction: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction , 논문 https://arxiv.org/abs/2503.03848
- DAD/A3D/CADP: 정확한 공식 배포처(GitHub/Drive)는 이번 조사에서 URL까지 확정하지 못함 — 후속 조사 필요

---

## 5. Stage 1 데이터 구축 방침 (확정, 2026-09-03)

- CCD 원본에 무아레, 주사율 간섭/이동 밝기 띠, 프레임 밝기 변화, 초점 흐림, 색·감마 변화, 원근 왜곡,
  반사/빛 번짐, FPS 불일치, 이중 압축을 확률적으로 조합해 `RERECORDED` 파생본을 만든다.
- source 영상 ID를 기준으로 train/validation을 먼저 나눠 동일 장면의 원본과 파생본이 서로 다른 분할에 들어가는 누수를 막는다.
- ORIGINAL 쪽도 코덱·비트레이트를 다양화해 파일 메타데이터가 정답 지름길이 되지 않게 한다.
- 학습에 사용하지 않은 CCD 원본 30개를 휴대전화로 각 3조건 재촬영한 약 90개를 실제 재녹화 검증 세트로 보존한다.
- 데이콘 공개 Stage1의 5개 RERECORDED 파일은 실제 기기 재촬영이 아닌 파생 예제이므로 파이프라인 참고용으로만 사용한다.

## 6. Stage 3 후보 조사 (1차, 2026-09-03)

### comma2k19 — 1순위, 1분 예제 검증 완료

| 항목 | 확인 내용 |
|---|---|
| 공식 저장소 | https://github.com/commaai/comma2k19 |
| 규모 | 캘리포니아 고속도로 주행 33시간, 1분 구간 2,019개, 전체 약 100GB(약 10GB 청크 10개) |
| 영상 | 각 구간에 전방 `video.hevc`, 영상 프레임별 `frame_times` 제공 |
| 종방향 신호 | CAN `car_speed`(m/s), wheel speeds와 IMU forward acceleration 제공 |
| 횡방향 신호 | CAN `steering_angle`(degree), gyro와 camera pose 제공 |
| 동기화 | 각 processed log가 timestamp/value 배열로 구성되어 영상 frame timestamp와 10Hz 정렬 가능 |
| 차량/장면 | Toyota RAV4 2청크 + Honda Civic 8청크, 고속도로 중심이라 도시·저속·정지 표본 부족 가능 |
| 소량 검증 | 공식 GitHub 저장소에 영상과 processed log가 포함된 1분 예제 구간 존재 |
| 라이선스 | 저장소는 MIT지만 문구가 `Software` 대상이므로 데이터 파일까지 명백히 포괄하는지는 추가 확인 필요 |
| 판단 | Stage3 네 범주의 직접 라벨 생성 가능성이 가장 높음. 전체 다운로드 전에 예제 구간으로 검증한다. |

**1분 예제 실측 결과(2026-09-03)**

- `video.hevc`: 1164×874, 실제 프레임 1,200개/약 60초. raw HEVC 헤더는 25fps로 표시되지만 공식
  `frame_times`는 중앙간격 약 0.05초이고 프레임 수도 일치하므로 실제 시간축은 20Hz로 확정했다.
- `frame_times`: 1,200개, 59.949초, 약 20Hz.
- CAN speed/steering: 각각 4,974개, 약 89Hz, 영상과 동일한 약 60초 범위.
- IMU acceleration: 6,256개, 약 104Hz, 영상과 동일한 약 60초 범위.
- 영상과 CAN 시작 차이는 약 0.04초, 종료 차이는 약 0.08초로 timestamp 기반 보간에 충분하다.
- speed 범위 7.97~19.84m/s로 이 예제에는 STOPPED가 없다.
- steering 범위 -4.6~+2.5도, 중앙값 -0.2도로 작은 영점 offset이 관찰됐다.
- 20Hz 영상에서 격 프레임을 선택하고 CAN을 보간해 10Hz 600개 임시 라벨을 생성했다.
- 임시 분포: accel CONSTANT 285 / ACCELERATING 158 / DECELERATING 157,
  steer STRAIGHT 356 / RIGHT 139 / LEFT 105. 사용자가 오버레이를 육안 확인하고 신뢰 가능한 것으로 판단했다.

예정 라벨 변환 원칙(임계값은 데이터 분포를 본 뒤 확정):

- 영상 프레임과 CAN을 timestamp 기준으로 정렬한 뒤 10Hz로 resample한다.
- `STOPPED`는 속도가 작은 구간에 hysteresis를 적용한다.
- `ACCELERATING`/`DECELERATING`/`CONSTANT`는 평활화한 속도의 시간 미분과 IMU 종가속도를 결합한다.
- `LEFT`/`RIGHT`/`STRAIGHT`는 조향각 부호와 dead-zone을 이용하되 차량별 영점 offset을 보정한다.
- 클래스 불균형과 고속도로 편향을 측정한 뒤 nuScenes/A2D2 보강 여부를 결정한다.

### nuScenes CAN bus — 도시·저속 보강 후보

- 20초 길이 1,000 scenes, 6개 카메라와 timestamped CAN/IMU를 제공한다.
- pose는 50Hz로 속도·가속도를, steering feedback은 100Hz로 조향각을 제공해 Stage3 라벨 변환이 가능하다.
- 다양한 도시 장면과 정지/회전 표본 보강에 유리하지만 데이터가 크고 가입과 별도 이용약관 동의가 필요하다.
- 비상업적 이용은 무료다. 실제 사용 전 대회 이용조건과 라이선스를 다시 확인한다.

### A2D2 — 추가 보강 후보

- 동기화된 6-camera 이미지, LiDAR, vehicle bus JSON을 제공한다.
- bus 신호에 acceleration, angular velocity, brake pressure 등이 포함된다.
- 공식 논문은 CC BY-ND 4.0을 언급하지만 세부 조향 신호 스키마와 다운로드 약관은 실제 사용 전에 다시 확인한다.

### Stage 3 다음 액션

1. 완료: 공식 1분 예제의 영상·CAN·IMU 구조, timestamp 범위, 10Hz 변환과 오버레이를 검증했다.
2. 완료: `src/prepare_stage3_comma2k19.py`에 10Hz MP4, 학습/디버그 CSV, 분포 리포트, 오버레이 생성기를 구현했다.
3. 완료(2026-09-03): Kaggle에서 런타임 스모크 PASS. 1,200 source frames→600 output frames,
   timestamp edge offset과 accel/steer 분포가 수동 검증 결과와 정확히 일치했고 MP4·CSV·JSON·오버레이를 모두 생성했다.
4. 사용자 결정(2026-09-03): Stage3 데이터·변환 타당성 검증은 여기서 완료 처리한다. 대용량 청크 확보와
   nuScenes/A2D2 보강은 실제 Stage3 학습에서 데이터 부족이 확인될 때 재개한다.
