# CODEX 인계 가이드 — Dacon 236753 crashvideo-project

이 문서는 Claude Code 세션의 토큰 예산 소진으로 작업을 Codex(또는 다른 에이전트)가 이어받기 위한 인계 문서입니다.
이 문서 하나만 읽어도 프로젝트 전체 맥락과 "지금 어디서부터 이어가야 하는지"를 알 수 있게 구성했습니다.

## 최신 재개 지점 (2026-09-10)

**Stage3 사람 검증셋 도구·안내 준비:** 사용자 직접 검수 요청. src/review_stage3_human.py(구간별조향/운동상태,UNKNOWN,자동저장,재개,해시검증), scripts/Start-Stage3HumanReview.ps1, docs/stage3-human-validation-guide.md 및 docs/templates/stage3-human-sources.csv. 기본공개예제 FPS헤더오류는 명시적 -Assume10Hz 필요. src/score_stage3_human.py는 사람MOVING/확정조향만채점. 2단위테스트+5영상5992시점 사전검사+채점제외스모크PASS. GUI키입력 수동시험은아직. 실제사람라벨 미생성, artifacts 스모크자료는정답아님.


**Stage3 라벨/영상 감사 완료:** docs/stage3-label-audit.md. 112205라벨 생성식 일치,187영상 timestamp frame_times[::2]일치. 뚜렷한 pose회전 방향일치약97.8%, 전반적좌우반전 근거없음. 원본/변환3영상60시점 프레임매핑 일치. 명목10Hz 대비최대0.45초드리프트는 이표본의 라벨정렬오류가 아님. 7클립42프레임 육안표본에서 완만한곡선추종 LEFT/RIGHT 확인, 공식범주 의미동등성 미확정. 전체부호교환/시간이동/추가학습은 보류. 다음은 공개영상 수동검증 기준·표본 구성. 자료 artifacts/stage3-label-audit-20260910/review.html.


**최우선: 공식 Stage3 하락, 추가학습 보류:** 사용자 제출86692/프로토_2026_09_10_002, 2026-09-10 13:56:59,13분40초. Stage1 0.4045598914/Stage2 0.2096390642/Stage3 0.1331824102. 사용자 업로드 경로 artifacts/stage3-submit-candidate-20260910 확인, ZIP재해시 cec269d... 일치. Stage3만교체,7개 핵심함수 AST일치. 직전Stage3 0.1656369753 대비19.59%하락. 외부검증 개선은 공식으로 이전되지 않았음. 기존 추가12에폭은 보류하고 외부proxy/평가분포 차이 및 모델 입력별 오작동 조사 우선. 원인 미확정. 상세 docs/submission-history.md, artifacts/stage3-submission-86692-audit.json.


**제출86692 Stage3 하락(사용자 보고):** 프로토_2026_09_10_002, 13:56:59,13분40초.
Stage1 .4045598914 / Stage2 .2096390642 / Stage3 .1331824102. 직전Stage3 .1656369753보다 하락,
가중합 .21804056804. 최고보고조합은 직전_001(.23102239408). 실제업로드ZIP해시는 미확인.
외부 검증 개선만으로 확대 모델을 승격하지 말고 모델연결·출력동등성·검증분포를 먼저 조사한다.
docs/submission-history.md 참고.


**Stage3 추가12에폭 준비 완료·업로드 차단:** 동일1000클립, 기존확대모델 가중치 시작/AdamW초기화. 로컬2테스트와 기존결과재검증PASS. artifacts/kaggle-stage3-continuation-20260910 및 docs/stage3-continuation.md 참고. 기존비공개 biadis/crashintent-stage3-optimization-assets로 모델+코드 업로드가 자동승인검토의 구체적대상·자료 승인 미확인 사유로 거절됨. GPU 및 watcher 아직 미실행. 준비된 자료에 대한 구체적 사용자 승인 대기.


**Stage1 전체 합성·검증·번들 완료(14:36):**1110원본→3330영상(학습2640/검증690),
전166500프레임·해시·출처분리 검사PASS. artifacts/stage1-full-preparation-20260910/status.json은
BUNDLE_READY_NOT_UPLOADED. artifacts/kaggle-stage1-full-20260910/에 balanced-v2 본 학습 번들
4,411,579,071bytes 준비 완료. SHA256 069fad70203ea83c036205841257b36ef1e1cefc8a7f60705518e82193138758.
전체 번들 업로드·GPU 본 학습은 아직 미실행이다. 아래 생성 중 기록보다 이 완료 상태를 우선한다.


**Stage3 오류 분석 완료:** 사용자 점수는 별도 전달 예정, 공식점수 대기와 독립적으로 클래스·경로·전환별 분석 완료. docs/stage3-error-analysis.md 및 artifacts/stage3-error-analysis-20260910/ 참고. 가속재현율20.16%, 우회전25.86%. 조향4경로 중08-02는0.3712→0.2777 하락, 직진재현율1.75%. 안정구간도 조향정확도46.30%. 동일1000클립 추가12에폭 실험 계획작성(AdamW재초기화 명시), GPU는 아직 미실행. 기존 후보ZIP 보존.


**새 Stage3 제출 후보 GPU 통합 완료:** biadis/crashintent-stage3-candidate-check v1 COMPLETE, 자동 회수 및 로컬 독립검증PASS. 후보 artifacts/stage3-submit-candidate-20260910/submit.zip (303597984 bytes), SHA256 cec269d3975db3d55a734505e40a3c03d8c43775e02fc887c2cfeba132964337. Stage3만 확대모델 교체, 기존Stage1/2 유지. 설치145.5초, Stage1/2 각5행,Stage3 1200행82.41초. docs/stage3-submit-candidate.md 및 artifacts/kaggle-stage3-candidate-20260910/result-validation.json 참고. 아래 RUNNING 기록보다 이 완료 결과 우선. 대회 실제 제출 미실행, selected.json 변경 없음.


**새 Stage3 제출 후보 GPU 검사 실행 중:** 사용자 요청으로 Stage2 개선 후보에서 Stage3만 확대 모델로 교체. artifacts/stage3-submit-candidate-20260910/submit.zip, SHA256 cec269d3975db3d55a734505e40a3c03d8c43775e02fc887c2cfeba132964337, 정적PASS. 기존 비공개 candidate-assets 데이터셋 버전 업로드 완료, biadis/crashintent-stage3-candidate-check v1 RUNNING. 자동 회수·검증 watcher PID26644; artifacts/kaggle-stage3-candidate-20260910/remote-run.json 및 monitor.log 확인, 중복 실행 금지. docs/stage3-submit-candidate.md 참고. 대회 제출 및 selected.json 변경 없음.


**Stage3 1000클립 확대 실험 완료:** biadis/crashintent-stage3-expanded-trial v1 COMPLETE, 자동 결과 회수 및 독립검증PASS. 12에폭/1200업데이트,31영상18603행,98분30초. 검증 가감속F1 0.461828 / 조향0.409353 (이전0.307496/0.293618). 주행 중 RIGHT예측19.3%로 이전70% 쏠림 완화. docs/stage3-expanded-trial.md 완료 절 및 artifacts/kaggle-stage3-expanded-20260910/result-validation.json 참고. 아래 RUNNING 기록보다 이 완료 결과 우선. 다음 작업은 새 Stage3 제출 후보 패키징·GPU 통합검사이며 아직 새 ZIP 없음.


**Stage3 1000클립 확대 실험 실행 중:** 사용자 승인으로 `biadis/crashintent-stage3-expanded-trial` v1 RUNNING 확인. 전체156학습영상/17경로에서 균형1000클립, scratch FP32 batch10 lr3e-5, 12에폭/1200업데이트 계획. 검증은 같은31영상. 예상1.5~2시간, runner최대2시간. 자동 회수·재검산 watcher PID13156 시작; `artifacts/kaggle-stage3-expanded-20260910/remote-run.json`, monitor.log 확인하고 중복 실행 금지. docs/stage3-expanded-trial.md 참고. 기존200클립 결과0.307496/0.293618과 비교 후 개선시 패키징·통합검사 진행. 아직 결과 및 새 제출 ZIP 없음.


**Stage1 후속 검증·번들 준비 실행 중(11:16):** 생성543/1110원본 상태에서
`src/prepare_stage1_full_run.py`를 숨김 실행했다. 완료된 영상부터 전 프레임/해시/출처검사 후
전체3330영상이 통과하면 `artifacts/kaggle-stage1-full-20260910/` 로컬 번들을 자동 생성한다.
상태는 `artifacts/stage1-full-preparation-20260910/status.json`, 오류는 preparation-error.log.
첫15원본/45영상 검사 성공. 외부 업로드·GPU 본 학습은 아직 하지 않는다. 중복 실행하지 않는다.


**Stage1 본 학습 코드 balanced-v2 반영 완료:** 사용자 요청으로 균형 유효배치8(원본4/재녹화4),
정확한 누적손실, lr1e-4 고정·규제비활성·head warmup없음, 최소120업데이트를 반영했다.
전체6에폭 계획은2640 optimizer updates다. 실행번들·노트북·결과검사기까지 연결,7테스트PASS.
GPU 본 학습은 이번 작업에서 실행하지 않았다. docs/stage1-full-training.md의 balanced-v2 절 참고.


**Stage1 쏠림 진단 완료:** Notebook optimization-check version1 COMPLETE, 4조건×120업데이트,
약25분 및 독립검증PASS. 모든 조건에서 학습30클립 F1=1.0. 검증30영상 F1은 기존누적0.7778,
전체가중치정규화0.5833, 균형lr2e-5 0.7205, 균형lr1e-4 0.8295. 검증2출처의 소량 진단이며
기존 손실 방식도 학습에 성공해 원인을 하나로 단정하지 않는다. 전체 합성은 아직 생성 중이다.
docs/stage1-optimization-diagnostic.md의 완료 절을 우선한다. 진단용 제출 가중치는 없다.


**Stage1 작은 표본 진단 준비:** 사용자가 Stage3은 별도 세션에 맡기고 Stage1 쏠림 진단을 요청했다.
기존 microbatch별 weighted mean 누적과 전체 배치 CE가 다름을 로컬 기울기 검사로 확인했다.
같은 표본·초기화의 손실 정규화/균형배치/학습률4조건, 최대120업데이트 GPU 진단을 준비했다.
새 코드·메타데이터7052bytes 전송은 처음 차단됐으나 사용자가 명시적으로 승인해 업로드·GPU 실행을 완료했다.
Notebook biadis/crashintent-stage1-optimization-check version 1 RUNNING, 자동 회수·검증 watcher 실행 중이다.
artifacts/kaggle-stage1-optimization-20260910/remote-run.json과 monitor.log를 확인하고 중복 실행하지 않는다.
아직 GPU 결과는 미확보다. docs/stage1-optimization-diagnostic.md 참고.


**최신 제출 결과(사용자 보고):** 프로토_2026_09_10_001, 표시 시각2026-09-10 09:24:27.
Stage1 0.4045598914 / Stage2 0.2096390642 / Stage3 0.1656369753.
로컬 가중합0.23102239408, 첫 제출 대비+0.01993517008이며 Stage2만+0.0498379252 상승했다.
누적 사용자 보고 제출2회. 업로드 ZIP 해시·서버 제출 ID·소요 시간은 미제공.
docs/submission-history.md 참고. Stage3 우선 개선은 유지한다.

**Stage3 최신 진단 돌파:** 같은10개 학습 클립에서 새 초기화·FP32·규제 비활성·유효 배치10은
lr1e-4 및3e-5 모두120스텝에 가감속/주행 중 조향100% 과적합을 달성했다.
기존 쏠림 모델은 같은 배치·lr3e-5에서도40%/33.3%였다. 독립 로컬 검증PASS
(입력·라벨·체크포인트·코드 해시 및 전 시점 로짓 기반 손실·정답률 재계산).
Notebook biadis/crashintent-stage3-optimization-check version1 COMPLETE, runner약21분50초.
이번 진단은 완료됐고 제출 모델을 바꾸지 않았다. 다음은 새 초기화·큰 유효 배치의
작은 학습 집합과 독립 route 검증으로 일반화 및 쏠림 해소 여부를 확인하는 것이다.
상세 docs/stage3-optimization-diagnostic.md, artifacts/kaggle-stage3-optimization-20260910/result-validation.json.
아래 실험 진행 중·과적합 미달 상태보다 이 완료 결과를 우선한다.

**Stage1 최신 결과(2026-09-10 10:04):** 사전 GPU version 1 COMPLETE, 6에폭/24 optimizer updates,
반환 ZIP 회수 및 `artifacts/kaggle-stage1-trial-20260910/validated/result-validation.json` PASS.
자동 회수 cp949 오류는 UTF-8 설정으로 복구했다. 합성 검증 F1 baseline0.25→0.4지만
새 모델도 전부 RERECORDED로 판별 성능 개선은 미확인이다. 전체 합성은 생성 중이다.
10:03 기준140/1110원본, 합성13시전후·바로 이어 실행 시 첫 본 학습 결과16~17시 조건부 예상.
아래 RUNNING/모니터링 기록보다 이 결과를 우선한다. docs/stage1-full-training.md 참고.

**Stage1 본 학습 착수(사용자 우선순위 3번 요청):** `docs/stage1-full-training.md` 참고.
CCD 1,500개 실재 확인. 휴대전화 holdout 및 같은 출처 형제까지 390개 제외,
학습 880원본/85출처 + 합성검증 230원본/21출처, 총 3,330영상 로컬 생성 중이다.
첫 20원본/60영상의 전 프레임·분할 검사 PASS. FP32 학습 코드와 결과 검사기 준비.
GPU 사전 번들 `artifacts/kaggle-stage1-trial-20260910/` 약190MB 준비 완료.
비공개 Kaggle 전송은 자동 승인 검토에서 구체적 자료·목적지 승인이 없다는 사유로
차단됐으나 사용자가 이 대화에서 약190MB 사전 번들 전송·무료 T4 실행을 명시적으로 승인해 업로드를 재개했다. GPU 완료 모델 확보로 취급하지 않는다.
전체 생성 로그: `artifacts/stage1-full-generation-20260910.log`.
비공개 Dataset ready 및 Notebook `biadis/crashintent-stage1-trial-v1` version 1 RUNNING 확인.
`src/watch_stage1_trial.py`가 결과 자동 회수·검증 중이다. `artifacts/kaggle-stage1-trial-20260910/remote-run.json`과 monitor.log를 확인하고 중복 실행하지 않는다.


**현재 우선 작업:** 사용자가 Stage2 제출 점수는 별도 전달하고 Stage3 쏠림 진단을 먼저
진행하도록 지시했다. 동일 10클립의 FP32 배치·학습률 대조 4조건을 준비했고
기울기 누적 동등성 테스트 2개 PASS. 새 비공개 Kaggle Dataset에 코드 전송은 자동
승인 검토가 처음 거절했으나 사용자가 구체적으로 승인해 비공개 Dataset 생성 및 GPU 실행을 시작했다. 현재 biadis/crashintent-stage3-optimization-check version 1을 추적한다. 중복 실행하지 않는다.
docs/stage3-optimization-diagnostic.md 참고.

**최신 GPU 결과:** Stage2 개선 후보 ZIP의 새 비공개 Kaggle 통합 검사를 완료하고
반환 ZIP 로컬 검증도 PASS했다. Notebook biadis/crashintent-stage2-candidate-check version 1
COMPLETE, 후보 SHA256 c0e8f2e3fc5847650e2e4a63f3e2b48e4058ad34edfbfe24a9a05333a4cf7047.
설치 186.60초, Stage1 5행/Stage2 5행/Stage3 1200행. 검증 보고서는
artifacts/kaggle-stage2-candidate-20260910/result-validation.json. 실제 대회 제출은 미실행.
아래 GPU 준비·승인 대기 기록보다 이 결과를 우선한다. docs/stage2-improved-candidate.md 참고.

**최신 성능 작업:** Stage2 soft_mixed_scene 후보를 독립 재검증했다. 동일 개발 검증
15영상/5출처에서 네 항목 평균 0.6000(수동 라벨 대조군 0.4333), 원본 CSV 해시·출처 분리,
저장 예측·실제 ZIP 모델 클래스 출력 일치 PASS. 원본 1영상 특징 재추출도 정확 일치했다.
별도 ZIP artifacts/stage2-submit-candidate-20260910/submit.zip 생성 및 정적 검사 PASS.
GPU 검사 입력 344.6MB와 노트북을 준비했으나 새 비공개 Kaggle Dataset 전송은
처음 자동 승인 검토에서 거절됐으나 2026-09-10 사용자가 명시적으로 승인해 업로드·GPU 검사를 재개했다. 같은 범위의 비공개 Kaggle 업로드·GPU 검사는 재확인 없이 진행하도록 승인했다. 일일 배포 후보는 변경하지 않았다.
상세 docs/stage2-improved-candidate.md. Stage3 FP32 대조 3조건×240스텝 결과도
독립 검증했고 과적합 실패가 지속된다. docs/stage3-overfit-diagnostic.md의 최신 절 참고.
Stage1 로컬 메타데이터가 과거 Stage3 원격 실행을 가리키므로 Stage1 완료로 취급하지 않는다.

**사용자 확정 목표:** 최종 1차 평가 상위 15위 이내 진입으로 2차 평가에 진출한다.
2026-09-10 사용자 제공 순위표에서 15위 장재홍은 Stage1 0.84684 / Stage2 0.4073 /
Stage3 0.72274이며 표시 점수 가중합은 약 0.621384다. 우리 첫 제출 가중합은 0.211087224다.
세 점수의 순서는 사용자가 Stage1·2·3으로 명시했다. 현재 15위는 비교 기준이며 최종 진출선은 아니다.
상세는 `docs/submission-history.md` 참고.

**최우선 최신 결과:** 첫 제출 후보 `artifacts/first-submit-candidate-v3/submit.zip`(303,672,145 bytes)을 만들고 공개 예제 GPU 통합 PASS했다. Stage1/2는 기존 baseline, Stage3는 혼합학습epoch3. 공개 Stage3 FPS 헤더 검사와 공식 Stage별 model_dir 경로를 수정했고 회귀10테스트 PASS. Stage1 5행/Stage2 5행/Stage3 1200행 검증, ZIP 해시 및 CPU 로짓 일치 PASS. 2026-09-09 사용자 보고로 최초 1회 제출 완료를 기록했다. 2026-09-10 사용자 보고로 1번 제출 평가 완료를 반영했다. 표시 시각 2026-09-09 17:51:28, 점수는 Stage1 0.4045598914 / Stage2 0.159801139 / Stage3 0.1656369753, 평가 소요 시간 13분 54초(834초)다. 사용자가 Stage1·2·3 순서임을 확인했으며 로컬 계산 종합점수는 0.211087224다. 제출 이력은 `docs/submission-history.md` 참고. 상세 `docs/first-submission-candidate.md`. 소량10클립 과적합 진단도 실행했으나180스텝에서 충분히 학습되지 않았고 기존 모델의 입력별 로짓이 거의 같았다. 다음 성능 작업은 FP32/규제 비활성 대조 및 scaler skip 진단이다(`docs/stage3-overfit-diagnostic.md`). 후보는 실행 검증용이며 Stage2 미학습 항목과 Stage3 최빈 클래스 쏠림이 남아 있다. 이번에 실행한 과적합 진단과 후보 통합 GPU 검사는 완료됐다.

**최신 결과:** 영상8개 혼합 배치 추가3에폭(누적2~4)이 완료됐고 결과 ZIP·최적 모델 회수 및 로컬 검증 PASS다. 총17,550스텝, runner3시간42분. 최적은epoch3. 가감속 F1 0.180613, 주행 중 조향 F1 0.221897이지만 예측은 모두 CONSTANT/STRAIGHT로 최빈 클래스 기준과 같다. 단일 클래스 쏠림이 해결되지 않았다. 상세 `docs/stage3-mixed-finetuning-results.md`, `artifacts/kaggle-stage3-mixed-training/result-validation.json` 참고. 다음은 클래스가 섞인 작은 고정 표본 과적합 및 로짓·gradient·가중치 변화 진단이다. 신규 학습은 아직 시작하지 않았다. 데이터·모델·결과는 artifacts에 있어 Git 제외이며, 다른 PC에서는 Kaggle 출력에서 회수한다.

마지막 작업인 **Stage3 comma2k19 Chunk_1 변환과 전수 검증을 완료**했다.
187개 영상 / 21개 route / 112,205개 10Hz 임시 라벨이며 검증은 PASS다.
상세 결과와 재현 명령은 `docs/stage3-comma2k19-runbook.md`,
`artifacts/stage3-comma-chunk1/validation_report.json`을 참고한다.
추가로 조향 보정 실험 v1과 route 분할을 완료했다. 영점 −0.2455도·직진 ±1.5도, 학습 156개/검증 31개다. 상세는 `docs/stage3-calibration-v1.md`와 `artifacts/stage3-comma-chunk1-calibrated-v1/`을 참고한다. Stage3 모델 파이프라인도 구현 완료했다. `docs/stage3-pipeline-runbook.md` 참고. 테스트 6개와 실제 MViTv2-S CPU 학습·저장·재로딩·추론 스모크가 통과했다. GPU 번들은 `artifacts/stage3-gpu-smoke.zip`이며 실행 노트북 `notebooks/Stage3_GPU_Smoke.ipynb`와 반환 ZIP 검사기 `src/validate_stage3_gpu_result.py`도 준비했다. Kaggle CLI 로그인 연결 후 비공개 Dataset 업로드와 T4 GPU 노트북 실행을 완료했고, 반환 ZIP의 로컬 검증도 PASS했다. 학습 1스텝·체크포인트 재로딩·공개 CUDA 추론 함수의 4시점 출력을 확인했다. 결과는 `artifacts/kaggle-stage3-smoke/result-validation.json`, 노트북은 `https://www.kaggle.com/code/biadis/crashintent-stage3-gpu-check`다. 첫 GPU 환경은 torch 2.10.0+cu128 / torchvision 0.25.0+cu128이었다. 추가로 제출 requirements 6개를 독립 환경에 설치해 torch 2.8.0+cu128 / torchvision 0.23.0+cu128에서 GPU 스모크와 로컬 결과 검증도 PASS했다. 설치는 Kaggle에서 166.72초였다. `notebooks/Stage3_GPU_Compatibility.ipynb`와 `artifacts/kaggle-stage3-compatibility/result-validation.json` 참고. 전체 v1 187영상 약 4.91GB 비공개 Dataset 업로드를 완료했다(`biadis/crashintent-stage3-training-v1`). 제출 버전 T4에서 2영상/2영상, stride32, batch2, 1epoch 시험 학습도 PASS했다. 20스텝, 총 176.85초, 최대 할당 2.73GiB. 1,200개 예측으로 지표를 로컬 재계산했고 체크포인트 재로딩 전후도 일치했다. 가감속 F1 0.1068, 주행 중 조향 F1 0.0945이며 단일 클래스 예측으로 쏠린 시험 모델이다. 결과는 `artifacts/kaggle-stage3-training/result-validation.json`, 실행은 `notebooks/Stage3_Training_Trial.ipynb`다. 전체 학습 156영상/검증31영상의 첫 본 학습과 결과 회수를 완료했다. 5,928스텝, 재검증 포함 91.02분. 18,603개 예측·원본 라벨·지표 재계산과 best.pt strict 로딩 PASS다. 하지만 가감속은 모두 DECELERATING(F1 0.069475), 주행 중 조향은 모두 STRAIGHT(F1 0.221897)로 쏠렸다. 조향은 최빈 클래스 기준과 같고 가감속은 그보다 낮다. 원인 확정은 아직이며, 다음은 작은 학습 집합 과적합 검사와 로짓·클래스별 손실/배치 순서를 진단한 뒤 개선 실험이다. 학습 순서 조사에서 stride8 표본 분포는 전체와 유사하지만 마지막100배치 감속43.15%(전체 선택18.70%)로 높았다. 후반 순서 영향은 후보이며 인과관계는 미확정이다. 마지막 영상의 직진은22.67%여서 직진 쏠림까지 설명하지 못한다. 상세 `docs/stage3-full-training-v1-results.md`와 `artifacts/kaggle-stage3-full-training/result-validation.json` 참고. 최종 제출 통합은 남아 있다.
초기 구간별 중앙값 라벨은 보존했고 v1은 별도 생성했다. v1도 센서 proxy와 표본 검수로 정한 실험용이며 공식 정답은 아니다. 전체 첫 에폭의 실행·결과 검증은 완료했으나 단일 클래스 쏠림으로 성능 개선이 필요하다.
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

## 완료: Stage3 짧은 독립 경로 검증 (2026-09-10)

사용자 요청에 따라 `biadis/crashintent-stage3-short-route-trial` v1 GPU 실행 시작, RUNNING 확인. 중복 실행하지 말 것. 기존 승인된 비공개 optimization-assets 데이터셋에 새 코드 버전 업로드 완료. 학습 34영상/17경로/균형 200클립, scratch FP32 batch10 lr3e-5 최대240업데이트, 검증 기존31영상/4경로 전체. 결과 대기 중. 세부 및 다운로드/독립 검증 명령은 `docs/stage3-route-trial.md`; 로컬 실행 기록은 `artifacts/kaggle-stage3-route-trial-20260910/launch.json`. 제출 가중치 및 ZIP 변경 없음.

Stage3 짧은 실험 최신 완료: biadis/crashintent-stage3-short-route-trial v1 COMPLETE, 12에폭/240업데이트, 검증31영상/18603행 독립 재계산 PASS. 가감속 F1 0.180613→0.307496, 조향0.221897→0.293618. 약41분11초. 조향 RIGHT 약70% 편향은 남음. 외부 대리 검증이며 공식 제출 점수 아님. 결과 회수 완료이므로 재실행하지 말 것. docs/stage3-route-trial.md의 완료 결과를 이전 RUNNING 기록보다 우선. 다음은 균형 학습 표본 확대 검증.

## Stage1 전체 업로드 승인 후 자동 본학습 대기 (2026-09-10 14:55)

사용자 명시적 승인으로4.41GB 비공개 full-v1-assets 업로드 시작.
숨김 launcher PID32984가 Dataset ready 후 biadis/crashintent-stage1-full-v1을 한 번 실행하고
결과를 회수·검증한다. GPU 시작 여부는 artifacts/kaggle-stage1-full-20260910/remote-run.json과
Kaggle status로 확인할 것. 아직 작성 시점 GPU RUNNING 미확인. 중복 실행 금지.
상세 docs/stage1-full-training.md 마지막 절. 같은 범위 업로드·무료 T4 실행 재승인 불필요.
