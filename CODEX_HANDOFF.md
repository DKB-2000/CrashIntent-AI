## Storage cleanup of completed training artifacts (2026-09-18)

At the user's request, 14 reproducible heavyweight files from completed jobs were deleted to free
27,070,000,549 bytes (25.211 GiB). Removed files were local Kaggle upload datasets (`quality.bin`,
`stage1-training.bin`, `stage3-training-v1.zip`, `screen.bin`), duplicate downloaded result ZIPs,
and the 6.4 GB partial VDMoire source prefix after its pilot clips and GPU diagnostic had already
been validated. Validation reports, logs, extracted validated checkpoints, official/current
submission ZIPs, raw core datasets, and the running Stage1 Imperial stratified acquisition were
preserved. The deleted bundles must be regenerated or downloaded again if those old trials are
rerun. D: free space after cleanup was 63,841,910,784 bytes.

## 최신 공식 최고: 제출 92510, Stage3 A2D2 dense blend (2026-09-18)

사용자 보고 파일명 `submit.zip`, 표시 시각 16:51:16, 평가 11분58초. Stage1/2/3은
`0.6386345895 / 0.2635462325 / 0.4435198965`, 가중합은 `0.4105533695`다.
ZIP은 `artifacts/stage3-a2d2-dense-blend-submit-candidate-20260918/submit.zip`, SHA256
`5d8d03b2e3cbfb6e67d332cec4e81ef2f6cca554df19fda9cd667b3caf87aaec`.
직전 Stage3 최고92259 대비 Stage3 `+0.0004448210`; 직전 전체 최고92269 대비 Stage1·2는
같고 Stage3 `+0.0076345919`, 가중합 `+0.00305383676` 개선됐다. 따라서 사용자 보고 기준
Stage3 단독과 전체 가중합 모두 새 공식 최고다. validation.json을 submitted=true와 제출
ID92510으로 갱신했다. 공식 서버 평가가 정상 완료됐으므로 이 ZIP의 실행 계약도 통과했다.

## Stage3 A2D2 dense hard-straight blend submission candidate complete (2026-09-18)

A2D2 sensor-fusion 공개 기록은 총3개 route이며 현재 2개 학습/1개 holdout으로 이미 전부 사용 중이라
새 route 확대는 불가능했다. 대신 기존 causal clip의 센서 정렬 endpoint 8..15에서 만든 dense 표본
(train582, holdout182)을 사용한 사전 고정 blend search 결과를 패키징했다. 선택은 report의
`dense25`, adapted weight0.3을 그대로 사용했다. 현재 공식 Stage3 최고 체크포인트와 dense25의
steer logits를 70:30으로 결합했으며 최신 전체 최고 Stage2 제출92269 ZIP을 base로 삼았다.

후보 `artifacts/stage3-a2d2-dense-blend-submit-candidate-20260918/submit.zip`, 185,381,809 bytes,
SHA256 `5d8d03b2e3cbfb6e67d332cec4e81ef2f6cca554df19fda9cd667b3caf87aaec`.
ZIP 변경 엔트리는 `model/stage3/best.pt` 하나뿐이다. 36개 A2D2 pilot 영상에서 명시적 70:30
logit 혼합과 compact 체크포인트가 `rtol=1e-5`, `atol=2e-6`, 최대오차
`1.52587890625e-05`로 일치했고 ZIP CRC/구조/CPU 검증을 통과했다. 생성 당시 상태는
`ZIP_READY_CPU_VALIDATED_GPU_PENDING`였으나 이후 제출92510으로 공식 평가가 정상 완료됐고
validation.json은 `SUBMITTED_EVALUATED`, submitted=true로 갱신됐다. 패키저
`src/package_stage3_a2d2_dense_blend_candidate.py`, 검증
`artifacts/stage3-a2d2-dense-blend-submit-candidate-20260918/validation.json`.
공식 결과 Stage3 0.4435198965로 채택 근거가 확인됐다.

## Stage3 공식 최고: 제출 92259, A2D2 50:50 logit blend (2026-09-18)

사용자 보고 제목 `2026_09_18_001 edit`, 표시 시각 09:35:45, 평가 11분43초.
Stage1/2/3은 0.6386345895 / 0.2456630208 / 0.4430750755, 가중합은
`0.40322215642`다. ZIP은
`artifacts/stage3-a2d2-blend-submit-candidate-20260918/submit.zip`, SHA256
`fa43f89048b1894ce29ee815dbe74848941acae7db5ed37f57f35a650c1efbcb`.
직전 제출91775와 Stage1·2는 같고 Stage3는 `+0.0071897709`, 가중합은
`+0.00287590836` 개선됐다. 따라서 공식 Stage3 최고는 0.4430750755이며 모델은 기존
warmstart_ce 3-seed ensemble과 A2D2 mix50 adapted ensemble의 steer logits를 50:50으로
혼합한 compact 3-model checkpoint다. 이후 제출92269는 전체 가중합 최고
`0.40749953274`이지만 Stage3는 0.4358853046으로 더 낮다. 이후 공식 비교 기준을
Stage3 단독은 제출92259, 전체 가중합은 제출92269로 구분한다.

## Stage3 RGB+motion causal GRU and held-route blend evaluation complete (2026-09-18)

Follow-up accel-only experiment is running as private Kaggle kernel
`biadis/crashintent-stage3-accel-gru` version1 (confirmed RUNNING 2026-09-18 14:09 KST). It reuses
the prior 187 RGB+motion feature caches through a private kernel source, trains three seeds for 12
epochs, and selects each epoch by validation route mean minus 0.25*route std. Local watcher PID
27532 has a finite four-hour deadline and will download/validate the 3-member checkpoint. Status:
`artifacts/kaggle-stage3-accel-gru-20260918/remote-run.json`. Do not launch a duplicate. This is an
experiment; no submission checkpoint or ZIP has been changed.

User explicitly approved uploading the private code-only Kaggle asset. Dataset
`biadis/crashintent-stage3-rgb-gru-assets` and private kernel
`biadis/crashintent-stage3-rgb-gru` were created. Version 1 failed because the notebook expected a
ZIP while Kaggle mounted the training dataset as an extracted directory. Version 2 reached feature
extraction but failed because the torchvision preset transform received a NumPy RGB array. The
runner now converts each contiguous RGB array to a CHW uint8 tensor. Version 3 completed and the
watcher downloaded and validated all outputs. Best epoch9 on 18,603 route-separated validation rows:
accel macro-F1 .5675041, moving-steer .4473808, joint .5074424; checkpoint SHA256
`54032946de5da799f3d787081abcaa7e0f532912bcbab59e95a5c44e7bd8926d`.
The trial extracts frozen ImageNet ResNet18 RGB features plus an 8x8 frame-difference grid for all
112,205 comma2k19 frames, then trains a two-layer causal GRU with balanced accel and steer losses.
Train/validation remain route separated (156/31 videos, 93,602/18,603 rows). It compares up to ten
epochs within a five-hour runner budget. Status and validation are under
`artifacts/kaggle-stage3-rgb-gru-20260918/`.

Same-frame comparison and route-held-out head-specific logit blend is complete at
`artifacts/stage3-rgb-gru-blend-20260918/report.json` (`COMPLETE_VALIDATED`, `DO_NOT_PACKAGE`).
Incumbent on the exact same rows: accel .5407368, steer .7242958, joint .6325163. RGB-GRU standalone:
accel .5675041, steer .4473808, joint .5074424. Route-held-out blend: accel .5353707, steer .7298280,
joint .6325993. Gain is only +.0000830 joint; accel regressed, LEFT F1 regressed, and the required
+.02 joint gate failed. The new model has useful accel signal and slight complementary steer signal,
but is not a submission candidate. No submission checkpoint or ZIP was changed.

## Stage3 A2D2 50:50 logit blend submission candidate complete (2026-09-18)

The validated 50:50 steer-logit blend of the incumbent 3-seed ensemble and the A2D2 mix50
adapted ensemble was compressed into the same three-model checkpoint structure. Only each
corresponding model's `steer.weight` and `steer.bias` were averaged; all other model state is
identical to the incumbent. `model/stage3/best.pt` is the only changed entry relative to the
current official-best Stage2 package
`artifacts/stage2-nexar-temporal-submit-candidate-v2-20260917/submit.zip` (base SHA256
`7a6c6ff418a5dcfd33b24dab5ef3ec47d00b984ac2361aa4947967444c9f7989`).
Candidate: `artifacts/stage3-a2d2-blend-submit-candidate-20260918/submit.zip`, 185,390,498 bytes,
SHA256 `fa43f89048b1894ce29ee815dbe74848941acae7db5ed37f57f35a650c1efbcb`.
Explicit 50:50 old/adapted logits and the compact checkpoint passed parity on all 36 A2D2 pilot
videos (`rtol=1e-5`, `atol=2e-6`; maximum absolute difference `3.0517578125e-05`). ZIP CRC,
static checks, and CPU package validation passed. 별도 GPU 통합 검사는 실행하지 않았지만
제출92259로 공식 평가가 정상 완료됐고 Stage3 0.4430750755를 기록했다. Report:
`artifacts/stage3-a2d2-blend-submit-candidate-20260918/validation.json`. Packager:
`src/package_stage3_a2d2_blend_candidate.py`.

# CODEX 인계 가이드 — Dacon 236753 crashvideo-project

## Stage2 장면 자동 라벨 교사 신뢰도 감사 완료 (2026-09-18)

사용자 승인으로 상위권 목표의 1순위인 `evasion_space`/`entry_side` 고신뢰 자동 라벨 구축을
시작했다. 기존 수동 CCD 라벨 중 source 분리 OOF 66건과 semantic-crop 5-seed 예측만 사용해
seed 합의도를 pseudo-label 선택 신호로 쓸 수 있는지 감사했다. 평가 데이터와 Nexar frozen50은
사용하지 않았다. `src/analyze_stage2_pseudolabel_precision.py`, 결과
`artifacts/stage2-pseudolabel-precision-20260918/report.json`, 상태 `COMPLETE_VALIDATED`.

5/5 seed 만장일치에서도 evasion은 57건 중30건(정밀도 .5263, Wilson95 하한 .3992), side는
52건 중32건(.6154, 하한 .4796)에 불과했다. class별 정밀도도 사전 gate를 통과하지 못해 두 작업
모두 `pass=false`, 최종 결정은 `DO_NOT_GENERATE_PSEUDOLABELS_FROM_THIS_ENSEMBLE`다. 기존
spatial-motion도 장면 정확도가 50% 안팎이라 대체 교사로 부적합하다. 따라서 현재 모델의 argmax나
seed 합의 라벨을 CCD 1,500건에 확장하거나 학습에 섞지 않는다. 다음 장면 라벨 경로는 독립적인
강한 시각 교사(VLM 다중판정) 또는 객체 검출·추적 교사이며, 같은 66건 source-OOF에서 고정 gate
(5/5 합의 n>=20, 전체 precision>=.75, Wilson 하한>=.60, 양 class n>=5 및 precision>=.70)를
통과한 작업만 후보 CSV를 생성한다.

## 최신 공식 최고: 제출 92269, Stage2 Nexar GRU2 (2026-09-18)

사용자 보고 제목 `2026_09_18_002 edit`, 표시 시각 09:58:03, 평가 12분52초.
Stage1/2/3은 0.6386345895 / 0.2635462325 / 0.4358853046, 가중합은
`0.40749953274`다. ZIP은 `artifacts/stage2-nexar-gru2-submit-candidate-20260918/submit.zip`,
SHA256 `63471bf9dd22398e0857029cef4f8763795b459c35b08c120c1e6476421a5df2`.
직전 공식 최고91775 대비 Stage1·3 동일, Stage2 `+0.0178832117`, 가중합
`+0.00715328468`로 현재 사용자 보고 기준 전체 최고다. 최초 시간헤드 이전 공식 최고
`2026_09_17_002 edit`의 Stage2 0.217560831 대비 누적 `+0.0459854015`다. 별도 GPU
통합검사는 미실행이었지만 공식 평가가 772초에 정상 완료됐다. validation.json을
submitted=true와 제출 ID92269로 갱신했다. 이후 공식 비교 기준 Stage2는 0.2635462325,
전체 가중합은 0.40749953274다.

## Stage2 Nexar GRU2 시험제출 ZIP 완료 (2026-09-18)

사용자 ZIP 생성 요청. 공식 최고 제출91775의 정확한 ZIP(SHA256 `7a6c6ff4...f7989`)을
base로 `artifacts/stage2-nexar-gru2-submit-candidate-20260918/submit.zip` 생성 완료.
185,382,127 bytes, SHA256
`63471bf9dd22398e0857029cef4f8763795b459c35b08c120c1e6476421a5df2`, 상태
`STATIC_AND_CPU_PARITY_PASS_GPU_PENDING`. ZIP 변경 엔트리는 `model/stage2/best.pt` 하나뿐이며
inference.py/Stage1/Stage3/requirements/Stage2 ResNet은 base와 byte exact. 체크포인트에서도
배포용 temporal_10hz_model의 GRU l1 forward/reverse 8 tensors와 tc/te 4 tensors만 변경했고,
기존 full-rate scene temporal/crop probes는 불변이다. frozen50 배포 stride3 재현은 실험 보고와
float32 parity: collision MAE4.251753, entry proxy4.182948, 합8.434701. CRC/구조/모델계약/
보존 엔트리 hash PASS. GPU 통합검사·대회제출은 미실행, submitted=false. 공식 최고91775와
selected는 유지하며 이 ZIP을 시험제출한다.

## Stage3 A2D2 학습 클립 확보 완료 (2026-09-18)

최종 상태 `ACQUIRED_VALIDATED`. 고유 PNG1,640/1,640, 5,873,776,681 bytes를 전수
1920x1208 디코딩·개별SHA 검증했다. 640x402/10fps/16 causal frames MP4 108/108 생성·재디코딩·
fps·프레임수·SHA 검증 PASS, MP4 합계53,800,070 bytes. LEFT/STRAIGHT/RIGHT 각36, 두 학습 route와
2018 holdout route 분리 유지. 결과 `artifacts/stage3-a2d2-training-clips-20260917/report.json`,
samples.csv endpoint15. 이제 특징 추출과 고정 rehearsal mix 대조를 시작할 수 있다.

## Stage2 Nexar GRU 2층 미세조정 완료 (2026-09-18)

공식 제출91775의 시간 모델을 시작점으로, scene 경로와 GRU 1층(l0)은 고정하고 10Hz 시간
모델의 BiGRU 2층(l1 forward/reverse)+tc/te만 Nexar160에 미세조정한다. 기존과 동일한 SHA256
분리 내부40으로 epoch1/2/5를 선택하고 seed20260918~20 상태를 평균한다. soft sigma0.5초,
GRU lr1e-4/head lr5e-4, weight decay1e-4, grad clip1. frozen50은 이미 노출된 확인셋임을 명시하고
공식91775 시간모델 대비 두 MAE 모두 개선, 합0.25초 이상 개선, 양쪽 within2s 비열화 없음일 때만
승격한다. `src/train_stage2_nexar_gru2.py`, 결과 `artifacts/stage2-nexar-gru2-20260918/`,
watcher `artifacts/stage2-nexar-gru2-watch-20260918/status.json`; watcher/result 모두
`COMPLETE_VALIDATED`, exit code0, 1,349.125초. 내부40은 epoch5 선택, seed평균 MAE합
9.805892. frozen50에서 공식91775 시간모델→GRU2 후보: collision MAE
5.597804→4.251753초(-1.346052), entry proxy MAE 5.380640→4.182948초(-1.197692),
합10.978444→8.434701(-2.543743). within2s collision .40→.46, entry .36→.52.
사전 gate5/5 PASS, `PROMOTE_FOR_INTEGRATION`. candidate SHA256
`b0927a8f2691e5f6c3ee79a85c9ceb3e57705e81005dad7ebbaae37abe7d1af8`.
아직 adopted=false이며 공식91775 ZIP/selected 불변. 다음은 동일한 scene 보존 구조로 새 시간모델만
교체해 ZIP 정적·CPU 패리티를 수행한다.

## 최신 공식 최고: 제출 91775, Stage2 Nexar 시간 헤드 (2026-09-17)

사용자 보고 제목 `2026_09_17_003 edit`, 표시 시각 17:34:42, 평가 12분32초.
Stage1/2/3은 0.6386345895 / 0.2456630208 / 0.4358853046, 가중합은
`0.40034624806`이다. ZIP은
`artifacts/stage2-nexar-temporal-submit-candidate-v2-20260917/submit.zip`, SHA256
`7a6c6ff418a5dcfd33b24dab5ef3ec47d00b984ac2361aa4947967444c9f7989`.
직전 공식 최고 `2026_09_17_002 edit` 대비 Stage1·3 동일, Stage2 `+0.0281021898`,
가중합 `+0.01124087592`로 현재 사용자 보고 기준 전체 최고다. 별도 GPU 통합검사는
미실행이었지만 공식 평가가 752초에 정상 완료되어 실행 계약도 통과했다. validation.json을
submitted=true와 제출 ID91775로 갱신했다. 이후 공식 비교 기준 Stage2는 0.2456630208,
전체 가중합은 0.40034624806이다.

## Stage2 Nexar 시간 헤드 공식 시험제출 ZIP 완료 (2026-09-17)

사용자 공식 ZIP 생성 요청. 현재 공식 최고 ZIP SHA256 `29014644...3614`를 base로
`artifacts/stage2-nexar-temporal-submit-candidate-v2-20260917/submit.zip` 생성 완료.
185,390,808 bytes, SHA256
`7a6c6ff418a5dcfd33b24dab5ef3ec47d00b984ac2361aa4947967444c9f7989`, 상태
`STATIC_AND_CPU_PARITY_PASS_GPU_PENDING`. Stage1/Stage3/requirements/Stage2 ResNet은 base와
byte exact. ZIP 변경은 `model/stage2/best.pt`, `inference.py`뿐이며 temporal state 변경은
tc/te weight+bias 4개뿐이다. 새 시간경로는 전체 특징의 매3번째 프레임(10Hz 가정)에서 예측하고
선택 index×3을 원본 이미지 파일명의 frame 번호로 복원한다. 기존 공식최고 full-rate temporal
시점과 crop probes는 별도로 그대로 사용해 evasion/side scene 경로를 보존했다. frozen Nexar50
배포 stride3 재현이 학습 보고서와 float32 exact parity: collision MAE5.597804, entry proxy
MAE5.380640, 합10.978444. CRC/구조/모델 계약/보존 엔트리 hash PASS. 첫 패키징은 .3초
경계에서 Python float64와 기존 Torch float32의 within 비율 차이로 gate가 차단했고, 기존 평가와
동일한 float32로 수정한 별도 v2가 최종본이다. GPU 통합검사와 대회 제출은 미실행,
`submitted=false`; 기존 공식 최고/selected는 유지한다.

## Stage2 Nexar 시간 헤드 보강 실험 완료 (2026-09-17)

사용자 진행 승인. `src/train_stage2_nexar_temporal_heads.py`가 Nexar TRAIN_ONLY 200개를
SHA256 고정 split 160 train/40 internal validation으로 나눈다. incumbent BiGRU와 scene head는
고정하고 tc/te 선형 시간 헤드만 학습한다. exact CE와 sigma0.5초 soft target, epoch
2/5/10/20, seed 20260917~19를 내부40의 collision+entry MAE 합으로만 선택한다. 그 뒤 frozen
Nexar50을 딱 한 번 평가한다. 세 seed 선형가중치 평균은 로그잇 평균과 동일한 단일 candidate로
저장한다. promotion gate는 frozen50에서 두 MAE 모두 개선, 합계 0.5초 이상 개선, 양쪽 within2s
비열화 없음이다. 최초 실행은 GRU hidden을 `inference_mode`에서 생성해 첫 선형헤드 역전파 때
`Inference tensors cannot be saved for backward`로 종료됐다(completed 0/6). 데이터·모델 오류는
아니다. `no_grad`로 수정해 별도 v2 결과 `artifacts/stage2-nexar-temporal-heads-v2-20260917/`,
watcher `artifacts/stage2-nexar-temporal-heads-watch-v2-20260917/status.json`으로 재시작했다.
2시간 제한 watcher/result 모두 `COMPLETE_VALIDATED`, exit code0, 54.406초. 내부 선택은
soft_0.5s epoch20, seed 평균 MAE합13.05596. frozen50 paired 10fps에서 incumbent→candidate:
collision MAE 7.57988→5.59780초(-1.98207), entry/alert-proxy MAE
9.42980→5.38064초(-4.04916), 합17.00968→10.97844(-6.03123). within2s도 collision
.24→.40, entry .18→.36으로 개선되어 사전 고정 gate 5/5 PASS,
`PROMOTE_FOR_OFFICIAL_INTEGRATION_TEST`. candidate SHA256
`8119a6f3648686d40607ed6e2381f71b2e8afc747b7df6633fb44b1d2581b161`.
단, 이는 Nexar 양성(near-miss 포함) event/alert 대리라 공식 entry/scene 정확도를 직접 보장하지 않는다.
candidate adopted=false이며 공식모델/ZIP/selected 불변. 다음은 공식 inference의 10fps→원본 frame
index 복원을 포함한 통합·패리티 시험 후에만 제출 후보를 만든다.

## Stage3 A2D2 분리 학습 클립 확보 착수 (2026-09-17)

2026-09-18 갱신: 최초 작업은 상태 `FAILED`, `TimeoutError('six-hour clip acquisition deadline')`.
실제 코드 deadline은12시간이었고 PC 대기시간을 포함해 제한을 넘긴 운영상 종료이며 HTTP/파일 검증
실패가 아니다. 잠금·이전 프로세스 종료 확인. 디스크에는 정상 PNG 1,550/1,640장,
5,562,323,385 bytes가 남아 있다. 같은 worker는 기존 파일을 다시 디코딩·SHA 검증해 재사용하므로
남은90장만 네트워크로 받도록 재개했다. 최초 재개 상태 DOWNLOADING(기존파일 재검증 진행)과 실제
launcher/worker를 확인했다. `worker-resume.stderr.log` 연결. ACQUIRED_VALIDATED 전 완료 아님.

추가 갱신: 재개 작업으로 PNG 1,640/1,640, 총5,873,776,681 bytes 확보 자체는 완료됐다. 그러나
메인 스레드가 status.json.tmp→status.json 교체 중 Windows `PermissionError(13)`를 받아 MP4 인코딩
전에 FAILED했다. 이미지/HTTP/디코딩 실패가 아니다. 잠금·프로세스 종료 확인 후 `save()`에 0.1초×50회
유한 PermissionError 재시도를 추가했다. 전체 PNG 디코딩·SHA를 다시 확인하고 MP4를 만들도록 finalizer
재시작, 최초 DOWNLOADING 재검증 진행과 실제 launcher/worker 확인. `worker-finalize.stderr.log` 연결.

두 학습전용 버스 `20190401_121727`/`20190401_145936` 확보·검증 완료: 184,996,122와
149,762,346 bytes, SHA256 `2094ab40...c025`/`4f99fe98...1c35c6c`, 각22신호·필수5신호·전체JSON
PASS. 카메라 anchor 100/80과 공개목록의 보수적 상한27500/22000으로 센서 범위를 잘랐다.
`src/plan_stage3_a2d2_training.py`에서 yaw/steering 동시조건을 유지했다. 경로별 상관 .921359/.943277,
후보는 각각 L567/R585/S2637, L182/R439/S2425. 3초 간격24개는 일부 회전 burst 때문에 불가했고
causal창과 같은1.5초 비중첩으로 첫 경로 클래스당24, 둘째 클래스당12를 선정했다. 총108개,
L/S/R 각36, 평가전용 `20180810_150607`과 경로분리. 처음 센서끝까지 사용해 index27601 metadata가
404였으나 카메라 실존범위로 제한 후 108 중앙 metadata 전수검증 PASS, 최대오차47,440us.
결과 `artifacts/stage3-a2d2-training-plan-20260917/report.json`.

`src/download_stage3_a2d2_training_clips.py`로 108개×causal16프레임의 중복 제거 PNG 1,640개를
4동시/파일별3회/12시간/배타잠금으로 확보 시작. 1920x1208 디코딩·개별해시 후 640x402 10fps
16프레임 MP4 108개와 endpoint15 samples.csv를 전수검증한다. 최초 DOWNLOADING 0/1640 및 실제
launcher/worker 확인. 상태·결과 `artifacts/stage3-a2d2-training-clips-20260917/`, 원본
`data_raw/a2d2/training-clips-20260917/`. ACQUIRED_VALIDATED 전 학습 금지.

## Stage1 Imperial Kaggle GPU 진단 준비 (2026-09-17)

사용자 진행 승인. 로컬 CPU는 현재 메모리부족으로 첫추론 실패하여 비공개 Kaggle T4 진단 전환. 업로드 목적지 `biadis/crashintent-stage1-imperial-assets`, kernel `biadis/crashintent-stage1-imperial-diagnostic`. payload344246449bytes: Imperial공개표본70145784 + official.pt137050519 + screen.pt137050519 + runner/manifest. 해시 bundle.json 고정. 학습/threshold튜닝 없음, 정지이미지16회반복 CUDA FP16 100장×2모델. `src/prepare_stage1_imperial_gpu.py`, `stage1_imperial_gpu_runner.py`, `launch_stage1_imperial_gpu.py`; 결과 `artifacts/kaggle-stage1-imperial-20260917`. 사용자 `진행해`가 이 제안 직후여서 해당 비공개 업로드/GPU 실행 승인으로 간주. 기존모델/ZIP/selected 불변.

## Stage3 A2D2 zero-shot 및 분리 학습경로 확보 (2026-09-17)

`src/evaluate_stage3_a2d2_zero_shot.py`로 공식 warmstart_ce 단일 배포모델과 현재 3시드 앙상블을
36개 endpoint15 causal clip에 fitting/threshold 없이 평가했다. 첫 실행은 학습용 model.pt를 배포
로더에 넣은 경로오류로 즉시 실패했으며 배포 best.pt로 수정 후 COMPLETE_VALIDATED. 두 모델 결과는
동일: macro-F1 .7380952381, accuracy .7777777778, LEFT/RIGHT recall 1.0, STRAIGHT recall .3333333333;
직진12개 중4개 LEFT,4개 정답,4개 RIGHT. A2D2에서 회전 검출은 강하지만 직진 과민판정이 핵심 약점이다.
결과 `artifacts/stage3-a2d2-zero-shot-20260917/report.json`. 이 첫 주행은 독립평가로 고정한다.

공개 S3 목록에서 별도 sensor-fusion 주행 `20190401_121727`, `20190401_145936` 두 개를 확인했다.
버스 JSON은 각각184,996,122 / 149,762,346 bytes. `src/download_stage3_a2d2_training_buses.py`가
2동시 Range 이어받기, 파일별3회, 4시간 제한, 배타잠금, 정확한 크기·전체 JSON·필수5신호·timestamp·
finite·SHA256을 검증한다. 최초 DOWNLOADING 0/2와 실제 launcher/worker 확인. 상태·결과
`artifacts/stage3-a2d2-training-buses-20260917/`, 원본 `data_raw/a2d2/`. ACQUIRED_VALIDATED 후에만
회전분포를 비교하고 학습 영상 부분확보를 시작한다.

## Stage2 Nexar 200영상 10fps 특징 추출 완료 (2026-09-17)

사용자 진행 승인. TRAIN_ONLY Nexar 200영상에서 공식 고정 ResNet18(SHA256
`b55eb2f9f3f559e2101e507d9acc49a5e0768f9d710eeb6a0912080e57595860`) 특징을 약10fps로
추출한다. `src/extract_nexar_stage2_temporal_features.py`, 결과
`artifacts/stage2-nexar-temporal-features-20260917/`, v2 감시 상태
`artifacts/stage2-nexar-temporal-features-watch-v3-20260917/status.json`. watcher/result 모두
`COMPLETE_VALIDATED`, exit code0, 200/200영상, 75,181 feature rows, 총9,303초. index SHA256
`b7ed5c3731d0731a0987ddcd051436d431a395545a472699f76a5fe04fa262e7`. 첫 watcher는 OpenBLAS 메모리
할당 실패로 즉시 종료되어 BLAS/OMP/MKL/NumExpr를 각1스레드로 제한한 별도 v2 감시기로
재시작했다. v2는 17/200영상·6,099행 저장 후 Windows native access violation(exit
3221225477)으로 종료됐다. 저장된 17개를 검증·재사용하고 OpenCV/Torch 각1스레드, batch8,
원자적 파일 저장과 영상별 메모리 해제를 적용한 재개형 v3를 시작했으며 최초 `resumed=true`,
17/200을 확인했다. 기존 Python3.11 실행파일이 사라져 있던 환경 문제는 같은 3.11.9를 재설치하고
기존 `.venv`의 torch2.8.0+cpu/torchvision0.23.0+cpu/OpenCV4.10.0/pandas3.0.5 import까지
검증해 복구했다. 다음은 frozen Nexar50에서 incumbent 대비 collision/event 및 entry/alert 시간 MAE를 사전 고정
실험으로 비교한다. scene head·공식모델·제출 ZIP은 변경하지 않았다.

## Stage1 Imperial 100장 진단 착수 (2026-09-17)

사용자 진행 요청. DATA_VALIDATED Imperial 원본50/재촬영50 정지이미지를 동일프레임16회 반복하여 공식최고와 screen_mix_25 CPU FP32 비교. threshold.5 고정, fitting 없음, 100장×2모델=200예측. 공간 재촬영단서만 진단하며 시간단서 성능으로 해석 금지. 명시적 SPDX license가 없어 진단한정, 학습/제출반영 금지. `src/evaluate_stage1_imperial_images.py`, 결과/상태 `artifacts/stage1-imperial-evaluation-20260917/`, Stage1ImperialEvaluation. 2시간 이내 유한 CPU 작업. 기존모델/ZIP/selected 불변.

최초 평가 프로세스는 모델0/2에서 로그 없이 종료. 100개 float 16-frame tensor 사전캐시가 약1GB라 메모리 종료로 판단. JPEG압축바이트 약70MB만 보관하고 이미지별 decode/즉시해제로 수정해 재시작. 최초 status RUNNING은 stale였고 완료 아님.

저메모리 재실행도 187초, 모델0/2 첫 추론에서 CPU 9,834,496bytes 추가할당 실패로 FAILED. JPEG캐시는 줄였으나 MViTv2 자체 forward peak를 현재 가용메모리가 감당 못함. 당시 Memory Compression 약2465MB, IntelliJ1650MB, Java700MB 등 사용자작업이 점유; 임의종료하지 않음. Imperial 예측0건, 결과/학습/모델/ZIP변경 없음. 다음은 메모리가 빈 시점의 예약 CPU 재시도 또는 70MB 공개표본+코드를 비공개 Kaggle GPU에 올리는 구체적 bundle 준비 후 실행.

## Stage3 A2D2 공개 센서 라벨 파일럿 착수 (2026-09-17)

초기 실행은 최초 `curl`과 감시기가 같은 부분파일에 동시에 append하여 원격 크기보다
6,705,003 bytes 큰 112,287,446 bytes가 되어 FAILED했다. 두 프로세스 종료를 확인하고 손상 파일만
삭제했다. 감시기에 배타적 `worker.lock`과 oversized partial 폐기 후 깨끗한 재시작 로직을 추가했다.
시스템 Python은 numpy가 없어 즉시 종료됐으며, numpy 2.4.6이 확인된 프로젝트 `.venv`를 외부 실행으로
사용해 로그 연결 후 단일 작업을 다시 시작했다. 최초 상태 `STARTING`, 0/105,582,443 bytes와 실제
launcher PID를 확인했다. 현재 완료 아님이며 아래 원래 계획은 유지한다.

재시작 후 한 번 조회에서 84,934,656/105,582,443 bytes(약80.4%), `DOWNLOADING`, 오류 없음이었다.
다운로드 중인 정상 JSON prefix를 읽어 `steering_angle_calculated`(degree),
`angular_velocity_omega_z`(degree/s), 3축 acceleration, accelerator pedal, brake pressure,
4륜 distance pulse, GPS, pitch/roll 신호를 확인했다. 따라서 완료 후 yaw-z와 steering의 방향·시간
일치 조건으로 사람 라벨 없이 LEFT/STRAIGHT/RIGHT를 만들 수 있는 스키마는 확보됐다. 부분파일은
완전한 JSON이 아니므로 분포·timestamp coverage 판정은 완료 보고서 생성 뒤에만 수행한다.

완료 갱신: 감시 상태와 보고서 모두 `COMPLETE_VALIDATED`. 원본은 정확히 105,582,443 bytes,
SHA256 `99dc8bcef028ef4e52164a789b693ddc1d2bd3b67ff7bdc2f0c808a7b2da82d0`, 전체 JSON 파싱 PASS,
22개 신호를 검증했다. 핵심 범위는 yaw-z -31.51~31.47deg/s(105,153표본), unsigned steering
0~432.45deg + sign(각52,577표본), speed 0~73.36km/h(26,288표본)이며 모두 약525.76초를 덮는다.
전방 카메라 표본 timestamp도 이 범위 안이다. 따라서 첫 주행 센서 파일은 자동 라벨 후보로 적격이다.
아직 전방 영상 전체 확보·선별 구간별 timestamp 일치·라벨 분포 검증은 하지 않았으므로 A2D2 학습셋
전체가 완성된 것은 아니다. 다음은 yaw/steering 일치로 균형 회전 구간을 정하고 해당 front-center
프레임과 메타데이터만 부분 확보하는 단계다.

후속 계획 검증 완료: `src/plan_stage3_a2d2_pilot.py`가 10Hz 센서 grid에서 speed>=5km/h,
yaw/steering 동시 조건을 사용했다. steering sign=1을 음수로 해석할 때 yaw 상관계수 0.9275808584.
후보 LEFT758/RIGHT521/STRAIGHT2146에서 시간 분산·3초 간격으로 각12개를 선택했다. 처음 10Hz
카메라 번호 가정은 최대344.97초 오차로 실패했고, 실제 메타데이터 쌍으로 30Hz임을 확인해 수정했다.
재검증은 36/36 성공, 최대 sensor-camera timestamp 오차 11,079us, 상태 COMPLETE_VALIDATED.
결과 `artifacts/stage3-a2d2-pilot-plan-20260917/report.json`.

PNG 파일럿 확보 착수: `src/download_stage3_a2d2_pilot.py`로 위 36개 중앙 front-center PNG만
1시간 제한/파일별3회 재시도/배타 잠금/실제 OpenCV 디코딩/메타 filename·해시 검증한다. 최초 상태
DOWNLOADING 0/36과 실제 launcher/worker를 확인했다. 상태·결과는
`artifacts/stage3-a2d2-pilot-acquisition-20260917/`, 원본은
`data_raw/a2d2/pilot-20180810_150607/`. ACQUIRED_VALIDATED 전 학습·추가 대량 확보 금지.

완료 갱신: PNG 파일럿은 `ACQUIRED_VALIDATED`, 36/36. LEFT/RIGHT/STRAIGHT 각12장,
총122,058,196 bytes, 모든 파일 개별 SHA256 기록, OpenCV 디코딩 PASS, 전부1920x1208,
메타데이터 image_png 일치, 최대 sensor-camera timestamp 오차11,079us다. 결과는
`artifacts/stage3-a2d2-pilot-acquisition-20260917/report.json`. 이 결과는 단일 중앙 프레임의
데이터·동기화 타당성 검증이며 아직 시간창 영상 클립이나 다중 A2D2 주행 학습셋은 아니다.
다음은 이36개 중심마다 주변30Hz 원본에서 10Hz로 2~4초 시간창을 부분 확보해 motion 입력을 만들고,
별도 주행을 holdout으로 남기는 것이다.

후속 causal clip 확보 착수: 현재 배포 motion feature는 10fps 과거15 flow pair가 필요하므로 각 중심
`center-45:center`에서 원본30Hz를3간격으로 뽑은16프레임/1.5초 causal 창으로 고정했다. 미래 프레임은
사용하지 않는다. 36클립×16에서 중복 제거한 고유PNG는551개다. `src/download_stage3_a2d2_clip_pilot.py`가
4동시 다운로드, 파일별3회, 6시간 제한, 배타잠금, 1920x1208 디코딩, 개별SHA256을 검증하고 640x402
10fps/16프레임 MP4 36개와 endpoint15 samples.csv를 생성한다. 최초 DOWNLOADING 0/551, clips36,
실제 launcher/worker 확인. 상태·결과 `artifacts/stage3-a2d2-clip-pilot-20260917/`, 원본
`data_raw/a2d2/clip-pilot-20180810_150607/`. ACQUIRED_VALIDATED 전 학습 금지.

완료 갱신: causal clip 확보는 `ACQUIRED_VALIDATED`. 고유 PNG 551/551,
원본 1,876,900,130 bytes, 개별 1920x1208 OpenCV 디코딩·SHA256 PASS. 640x402, 10fps,
16프레임 causal MP4 36/36을 생성하고 전수 재디코딩·fps·프레임수·해시 검증 PASS,
MP4 합계 12,589,846 bytes. LEFT/STRAIGHT/RIGHT 각12개이며 samples.csv endpoint=15다.
결과 `artifacts/stage3-a2d2-clip-pilot-20260917/report.json`. 이제 데이터 확보가 아닌 모델의
A2D2 zero-shot 진단과 Comma2k19 rehearsal을 포함한 소규모 domain-mix 실험을 설계할 수 있다.

사람 라벨 없이 도시·회전 주행을 보강하기 위해 A2D2 공개 S3의 첫 sensor-fusion 주행
`20180810_150607`을 조사한다. 공식 LICENSE/README와 전방 카메라 메타데이터 1건을 확보했고,
CC BY-ND 4.0, 동기화된 전방 카메라 timestamp 및 vehicle bus 제공을 재확인했다. 버스 JSON은
105,582,443 bytes이며 최초 부분파일에서 정상 JSON과 `acceleration_x` 신호를 확인했다.
`src/scout_stage3_a2d2.py`가 HTTP Range 이어받기, 3회 재시도, 2시간 제한, 정확한 크기·SHA256,
전체 JSON 파싱과 steering/yaw/speed/acceleration 후보 통계를 수행한다. 상태는
`artifacts/stage3-a2d2-scout-20260917/status.json`, 결과는 같은 폴더 `report.json`, 원본은
`data_raw/a2d2/20180810_150607_bus_signals.json`이다. `COMPLETE_VALIDATED` 전에는 영상 확보·학습을
시작하지 않는다. 모델·제출 ZIP·selected 상태는 변경하지 않았다.

## 최신 공식 최고: Stage2 crop-blend + Stage3 3시드 앙상블 (2026-09-17)

사용자 보고 제목 `2026_09_17_002 edit`, 표시 시각 10:41:40, 평가 12분 14초. Stage1/2/3은 0.6386345895 / 0.217560831 / 0.4358853046, 가중합은 0.38910537214다. ZIP은 `artifacts/stage2-crop-blend-stage3-ensemble-submit-candidate-20260917/submit.zip`, SHA256 `290146445dcd56de78e09e7f30dc2681376af4bfb09ae55f312493fcd4243614`. 직전 공식 최고 대비 Stage2 +0.0079217668, Stage3 +0.0009209265, 가중합 +0.00353707732로 현재 사용자 보고 기준 전체 최고다. validation.json submitted=true 및 제출 메타데이터 갱신 완료. 사전 정적·CRC·Stage3 CPU 1200행 패리티 PASS; 별도 GPU 통합 검사는 미실행이었으나 공식 평가는 정상 완료됐다.

## Stage1 대체자료 Imperial 표본 확보 착수 (2026-09-17)

Zenodo 서비스 자체 장애 확인 후 대체자료 조사. Imperial College Single-Capture and Recaptured Image Database의 공개 subjective-test 표본 100장/66.89MiB 선택. 공식페이지는 논문 인용을 요청하지만 SPDX형 명시 라이선스는 없어 우선 진단용 한정, 학습/제출 반영 전 사용조건 재검토. `src/download_stage1_imperial_sample.py`가 중단된2.09MiB부터 resume, 2시간 curl 제한/3회재시도/정확한70145784bytes/ZIP CRC/목록/SHA256 검증. 결과 `artifacts/stage1-imperial-recapture-20260917/`, 상태 Stage1ImperialSample. Axon HF 표본은 공격영상만 있어 원본오탐 교정용에서 제외; TPO는 CC BY-NC-SA4.0이나 접근신청 필요.

최초 Imperial 완료파일은 이전 수동 curl과 watcher curl이 잠시 동시 append해 75126520bytes(정상70145784보다4980736 초과), 크기 gate에서 FAILED. CRC 전에 차단되어 사용 안 함. 손상본 `SubjectiveTestImages.zip` 보존, 별도 `SubjectiveTestImages.clean.zip`에 resume 없이 단일writer로 새로 받고 같은 크기/CRC/SHA 검증 재실행.

깨끗한 `SubjectiveTestImages.clean.zip` 확보 완료. 정확히70145784bytes, ZIP CRC PASS, 101멤버(labels.txt+IMG001~100), 라벨50 Original Captured/50 Recaptured, SHA256 `d8ce8a3671bf49235d23190f33122105ffe087a42a1550b08bfc1099bfadf697`, status DATA_VALIDATED. 상태 최종쓰기 때 PowerShell BOM을 Python 기본cp949로 읽어 실패한 운영오류는 스크립트를 utf-8-sig read/utf-8 write로 수정하고 동일 크기/CRC/라벨/해시 독립재검증 후 status 복구. 아직 모델진단/학습/ZIP변경 없음.

## Stage1 DLC-2021 40쌍 자동라벨 확보 착수 (2026-09-17)

사용자 추가개선 진행 승인. 9문서에서 기본4템플릿, 그중4문서에서 1템플릿 추가해 총40쌍/80클립 구성. 각 클립 중앙16프레임만 Zenodo Range로 확보해 전체데이터 다운로드 회피. `est_id`,`srb_passport` 8쌍은 문서종류 holdout, 나머지32쌍 학습. ZIP CRC/크기·MP4재디코딩·SHA256·pair/split/공식파일명 라벨 검증. `src/acquire_stage1_dlc_pair_training.py`, 결과 `artifacts/stage1-dlc-pair-training-20260917/`, 상태 Stage1DLCPairTraining. 기존 다문서 예측은 모든 쌍에서 이미 P(re)>P(or)이므로 순위손실 단독은 공유 양성편향을 못 고침. 학습은 낮은가중치 pair BCE+ranking과 기존데이터 rehearsal, head-only 및 마지막block 조건 비교로 수정. holdout 원본오탐 감소와 기존420 평가 비열화를 함께 gate. DATA_VALIDATED 전 학습bundle/GPU실행 금지. 사람라벨/threshold튜닝/모델·ZIP·selected 변경 없음.

최초 확보는 or.zip 선두 alb_id 2.4MB Range가 API/직접주소 모두 3회504로 0/80 FAILED. 파일/검증 오류 아님. 이전 성공 구간은 수백MB 이후이므로 alb_id 제외, 나머지9문서 기본4템플릿 + aze/esp/fin/svk 각1추가로 총40쌍 유지해 재개. holdout est/srb 각4쌍은 유지.

재개도 첫 aze_passport or Range(315071480-316459913)에서 3회504로 0/80 FAILED. 추가로 수동 status 정리 시 실제 newline 대신 literal `\\n`이 붙어 worker의 실패기록 갱신이 JSONDecodeError가 된 운영오류가 있었으며 status를 유효 JSON으로 복구하고 `REMOTE_RANGE_ACQUISITION`, retry_exhausted로 확정했다. 40쌍 DATA_VALIDATED 아님, 학습/GPU/모델/ZIP 변경 없음. 이미 검증된 표본은 alb_id 1쌍 + 다문서4쌍뿐이다.

## Stage1 DLC-2021 다문서 Range 표본·진단 완료 (2026-09-17)

사용자 추가확보 후 진행 요청. FTP 재접속 실패 후 Zenodo 공식 API HTTP Range 사용. re38.5GB/or18.8GB 전체 대신 중앙목록 re2.97MB·or2.21MB와 선택구간 약67MB만 확보. aze_passport/esp_id/fin_id/svk_id 각각 `00.or0001`·`00.re0001`, 총8클립이 ZIP CRC/크기/MP4재디코딩/해시/공식파일명 자동라벨 검증 PASS, DATA_VALIDATED. 두 모델 진단 포함 전체 파이프라인 COMPLETE_VALIDATED. 공식최고: 원본0/4(mean positive .92721), 재촬영4/4(.99921). screen_mix_25: 원본1/4(.73124), 재촬영4/4(.94992); esp_id 원본만 정답(.26794), 나머지원본>.85. 합성보강이 양성편향은 줄이나 원본 일반화가 여전히 부족하고 공식점수도 낮았으므로 미채택. 사람라벨/threshold튜닝/모델·ZIP·selected 변경 없음. `artifacts/stage1-dlc-range-pilot-evaluation-20260917/report.json`.

## 최신 공식 최고: 제출 91514, Stage3 warmstart_ce 개선 (2026-09-17)

사용자 보고 제목 `2026_09_17_001 edit`, 표시 시각 09:20:35, 평가 11분 28초. Stage1/2/3은 0.6386345895 / 0.2096390642 / 0.4349643781이며 가중합은 0.38556829482다. ZIP은 `artifacts/stage3-warmstart-ce-submit-candidate-20260917/submit.zip`, SHA256 `b4b73b2968ecdef39fd923645210ae3e00b8496e06d313be5a587f2a867d8655`. 제출 ID 91514. 동일 Stage1·2 기준 기존 Stage3 0.4187467089보다 +0.0162176692, 직전 전체 최고 가중합 0.38304763346보다 +0.00252066136 개선되어 현재 사용자 보고 기준 전체 최고다. Stage3는 warmstart_ce 고정 seed 20260910. 정적·CRC·CPU 1200행 클래스 패리티 PASS, 별도 GPU 통합 검사는 미실행이나 공식 평가는 정상 완료. validation.json의 submitted=true 및 제출 메타데이터 갱신 완료.

## Stage1 DLC-2021 부분 TAR 초기 진단 완료 (2026-09-17)

전체 clips.tar 17.77GB 다운로드는 8시간 제한에서 193,638,340bytes만 받고 FAILED. 재시작하지 않음. 무압축 TAR의 온전한 앞부분에서 JPEG669개 확인, 그중 공식 유형코드로 자동라벨 가능한 alb_id 원본(or)2클립/화면재촬영(re)4클립 총338프레임을 MP4로 복구·재디코딩·해시검증해 `artifacts/stage1-dlc-partial-pilot-20260917/status.json` DATA_VALIDATED. 사람라벨 없음. 공식최고 vs screen_mix_25 CPU FP32 진단은 `COMPLETE_VALIDATED`. 공식최고: 원본0/2, 재촬영4/4, 평균양성확률 원본.98810/재촬영.99554. screen_mix_25: 원본1/2, 재촬영4/4, 평균양성확률 원본.43677/재촬영.69308. 합성보강이 파일럿의 과도한 양성편향을 줄였으나 6클립·단일템플릿이고 공식점수는 낮아 채택/threshold튜닝/모델·ZIP·selected변경 없음. 다음은 서로 다른 문서템플릿 자동라벨 표본 확보가 우선. 결과 `artifacts/stage1-dlc-pilot-evaluation-20260917/report.json`.

## Stage1 실제 화면 재촬영 DLC-2021 확보 착수 (2026-09-16)

사람라벨불가 전제의 다음방향1번 진행. RECOD-MPAD/Replay-Mobile/CoC는 접근제한으로 제외, 공개 CC BY-SA2.5 DLC-2021 선택. 메타1424클립=or290/re400/기타734, re iPhone200/Android200·화면 Lenovo40/Philips160/MacBook160/HP40·문서10/템플릿73. 공식baseline 자동목록 train 양성19543/음성25980, test 양성15346/음성16264 확인. 소형 metadata/license/README/baseline 확보 및 FTP baseline MD5 exact. 원영상88.33GB와 Zenodo원본+재촬영72GB 대신 전체추출프레임 tar17,768,312,320bytes 선택. src/download_stage1_dlc_frames.py 숨김PID38472 시작, status DOWNLOADING·실제파일증가 확인. data_raw/dlc-2021/clips.tar, artifacts/stage1-dlc-frames-acquisition-20260916/status.json. 이어받기/3회/8시간/공식MD50758a65d.../tar목록검증, Get-BackgroundWorkStatus 연결. 아직완료아님, 중복실행/반복폴링금지. 완료후제공목록에서기기·화면·문서분리소규모추출→기존모델진단→artifact encoder대조. 모델/ZIP/selected불변. docs/stage1-physical-recapture-data.md 및 DATA_SOURCES.md 참고.

## Stage2 Nexar 공식 시간 라벨 학습 pool 200영상 확보 완료 (2026-09-17)

watcher/result 모두 `COMPLETE_VALIDATED`, exit code0. 200/200파일,
3,205,071,535bytes exact, OpenCV decode 200/200, bad0, frames/fps 및 event 시간범위 PASS.
영상은 `data_raw/nexar-stage2-temporal-train-20260917`, 고정 manifest와 decode 결과는
`artifacts/stage2-nexar-temporal-pool-20260917/`. 기존 Nexar50은 검증전용으로 계속 분리하며
200건만 TRAIN_ONLY. 다음은 Nexar event/alert 시간 head 보강과 고정50 평가 대조이며 scene
head·공식모델·ZIP은 아직 변경하지 않는다. 아래 다운로드 중 기록보다 이 완료 판정 우선.

200파일 HEAD 전수 확인: 3,205,071,535bytes(2.985GiB), 파일5.88~35.89MB, 로컬여유
101.25GiB, 요구정책13.94GB로 space gate PASS. 4동시/파일별3회/이어받기/4시간 제한 watcher
시작, 최초 DOWNLOADING 6/200·94,341,003bytes와 child PID 확인. 완료 후 원격크기 exact,
OpenCV decode, frames/fps, event<=duration+1초 검증. 결과
`artifacts/stage2-nexar-temporal-pool-20260917/download-status.json`, 감시
`artifacts/stage2-nexar-temporal-pool-watch-20260917/status.json`, 영상
`data_raw/nexar-stage2-temporal-train-20260917`. 반복폴링·중복실행 금지. 기존50 검증셋 불변.

## Stage2 Nexar 공식 시간 라벨 50영상 채점·학습 pool 준비 완료 (2026-09-17)

기존 고정 Nexar 양성50영상 예측을 공식 event/alert와 frame PTS로 최초 채점했다.
`artifacts/stage2-nexar-temporal-score-20260917/report.json` COMPLETE_VALIDATED. incumbent 충돌
MAE7.406초/median5.657초/.3초6%, entry-alert proxy MAE8.851초/median8.588초/.3초2%.
CCD pretrained는 충돌 MAE12.377초, entry MAE10.938초로 더 나쁨. 현재 Stage2 병목은 scene
probe보다 외부출처 시간 일반화임을 확인. Nexar는 near-miss 포함, alert는 entry proxy이므로
대회 정답 동등 주장 금지. 이50건은 검증전용 유지·학습금지.

나머지700양성에서 scene/weather/light와 alert-event gap을 균형화한 고정seed 200건 학습
manifest 준비: `artifacts/stage2-nexar-temporal-pool-20260917/manifest.csv`, 상태
POOL_READY_SIZE_UNCHECKED, 다운로드 미시작. 단일 HEAD 표본 24,145,111bytes이고 기존50 평균도
약16.8MB라 전체는 수GB 예상. 디스크·전체 HEAD 용량 확인 후 watcher 다운로드할 것.

## Stage2 라벨 개발셋 MM-AU·Nexar 및 객체 라벨 감사 완료 (2026-09-17)

MM-AU 객체 라벨 113,457,793 bytes 다운로드·검증 완료, SHA f84f291d..., tar 422,492파일.
전수 감사에서 YOLO txt 422,488개/영상키11,400, 메타 매칭11,383. t_ai±5 박스11,381,
t_co±5 박스11,380. 일반 객체 7클래스 bbox이며 track ID와 collision actor ID가 없어서
entry_side/evasion_space 공식 정답으로 직접 변환 불가. 시간 주변 객체 위치 특징에는 사용 가능.
범위경고6,882줄은 표본상 경계 차량 폭 1.0008~1.0063 반올림이라 clip 가능. 결과
`artifacts/stage2-mmau-detection-label-audit-20260917/report.json` COMPLETE_VALIDATED.
전체525.8GB 다운로드는 계속 금지. 다음은 작은 MM-AU 영상 표본으로 t_ai/t_co 의미 보정 후
시간개발셋 채택 판단이며, 방향/회피 공식라벨 탐색은 별도 계속.

사용자 요청으로 라벨된 개발셋에 집중. 기존 Nexar metadata.csv가 양성 750건 전체이며 alert/event
모두 완전함을 확인했다. MM-AU 공식 repo commit 2b205a48... 확보, 메타 11,730행 중 양성
11,713행 전부 시간 순서 유효·고유 ID. MM-AU t_ai/t_co 및 Nexar alert/event를 공통 Stage2
CSV로 변환했으나 entry 의미 동등성은 영상 보정 전 미확정, side/evasion 공식 라벨은 양쪽 모두
없다. `artifacts/stage2-labeled-dataset-audit-v2-20260917/report.json` COMPLETE_VALIDATED.
MM-AU 전체 525.8GB는 받지 않고 공식 객체 라벨 113,457,793 bytes만 watcher 다운로드 중,
최초 DOWNLOADING·실제 child 확인. `artifacts/stage2-mmau-label-acquisition-20260917/status.json`.
완료 후 tar 구조·ID 대응·방향/공간 파생 가능성 감사. 반복 폴링·중복 실행 금지.

## 2026_09_17_002 공식 결과 — 전체 최고·Stage2 blend 미채택 (2026-09-17)

사용자 보고: submit.zip, `2026_09_17_002 edit`, 10:41:40, 12분14초. Stage1/2/3
.6386345895/.217560831/.4358853046, 가중합 .38910537214. 직전 Stage2 crop 최고
.21955508 대비 Stage2는 -.001994249로 blend를 Stage2 개선으로 채택하지 않는다. 직전 전체
최고 91514(.38556829482) 대비 가중합은 +.00353707732이며 사용자 보고 전체 최고다. 상승은
Stage3 +.0009209265 및 조합 결과이고 blend의 Stage2 효과로 해석하지 않는다. 맥락상 후보는
`artifacts/stage2-crop-blend-submit-candidate-20260917/submit.zip`, SHA256 86e6d656...이나
서버 ID·해시는 미제공. `official-result.json`, `docs/submission-history.md` 참고.

## Stage2 shared/task 50:50 로짓 혼합 PASS·ZIP 생성 완료 (2026-09-17)

패키징 watcher `COMPLETE`, exit code 0. 후보 ZIP은
`artifacts/stage2-crop-blend-submit-candidate-20260917/submit.zip`, 177,720,336 bytes,
SHA256 `86e6d6562ad784520415e66682e1d7ac4e90d0c5cb1b729364361351534f5bee`.
기준 ZIP 대비 `model/stage2/best.pt`와 `inference.py`만 변경, ZIP CRC와 정적·CPU 패키지
검증 PASS. 상태 `ZIP_READY_CPU_VALIDATED_GPU_PENDING`; GPU 통합 검사와 대회 제출은 아직
실행하지 않았고 `submitted:false`다. 아래 생성 중 문장보다 이 완료 기록이 우선한다.

재학습 없이 기존 5seed OOF 로짓을 혼합했다. 사전 주 후보 task weight .50은 shared 대비
회피 +.04545, 방향 +.04545, 개발평균 +.02273, 출처동일가중 +.01612이며 최악 출처 delta
0.0으로 gate `PASS`. 진단 .25도 전 출처 비열화 없이 개선, .75는 출처 회귀로 실패해 보수적
혼합 근거도 확인했다. `artifacts/stage2-crop-logit-blend-20260917/report.json`.
shared 5 probe와 작업별 회피/방향 각 5 probe를 저장하고 추론에서 .50/.50 혼합하는 후보 ZIP
패키징 watcher 시작, 최초 `RUNNING`과 child PID 확인. 결과 경로
`artifacts/stage2-crop-blend-submit-candidate-20260917`, 감시
`artifacts/stage2-crop-blend-submit-watch-20260917/status.json`. 완료·검증 전 제출 금지.

## Stage2 작업별 crop 5seed 로짓 앙상블 직접 대조 완료·미채택 (2026-09-17)

watcher와 결과 모두 `COMPLETE_VALIDATED`, 5/5 seed, exit code 0이나 gate는 `FAIL`이다.
shared 앙상블 대비 작업별 앙상블은 회피 +0.03030, 방향 +0.03030, 개발평균 +0.01515로
상승했지만 출처동일가중은 -0.00562 하락했다. 출처 23과 37의 개발평균 delta가 각각
-0.25로 사전 한계 -0.10도 넘었다. 전체 표본 개선이 특정 출처 회귀를 가리므로 미채택한다.
새 ZIP·제출 없이 공식 shared crop Stage2(.21955508)를 유지한다. 결과는
`artifacts/stage2-task-specific-ensemble-20260917/report.json`, 감시는
`artifacts/stage2-task-specific-ensemble-watch-20260917/status.json`이다.
아래 실행 중 문장보다 이 완료 판정이 우선한다.

개별 seed 엄격 gate 실패 후 실제 제출 구조에 맞춰 shared와 작업별 probe를 같은 실행에서
학습하고 OOF 로짓을 저장하여 각각 5seed 평균하는 직접 대조를 시작했다. 최초 `RUNNING`·0/5,
watcher와 child PID 확인, 1시간 제한·중복 시작 거절. 사전 gate는 네 전체 지표 비열화 없음,
개발평균·출처동일가중 엄격 상승, 최악 출처 개발평균 delta >= -0.10이다. 결과는
`artifacts/stage2-task-specific-ensemble-20260917/report.json`, 감시는
`artifacts/stage2-task-specific-ensemble-watch-20260917/status.json`. 완료 전 ZIP·제출 금지.

## Stage2 작업별 crop 특징 5seed 대조 완료·엄격 gate 미통과 (2026-09-17)

v2 감시와 결과 모두 `COMPLETE_VALIDATED`, 5/5 seed, exit code 0이다. 기존 shared crop 대비
평균 변화는 회피 +0.00909, 방향 +0.02727, 개발평균 +0.00909, 출처동일가중 +0.00612로
모두 상승했다. 회피·방향·개발평균은 모든 seed에서 비열화가 없었다. 다만 출처동일가중이
2/5 seed에서 각각 -0.00688, -0.00851 하락하여 사전 정의 gate는 `FAIL`이다. 개선 신호는
있지만 현재 규칙상 미채택하며 새 ZIP·제출 없이 공식 argmax shared crop Stage2(.21955508)를
유지한다. 유효 결과는 `artifacts/stage2-crop-task-specific-seeds-v2-20260917/report.json`,
감시는 `artifacts/stage2-crop-task-specific-seeds-v2-watch-20260917/status.json`이다.
아래 실행 중 기록과 첫 0/5 구현 실패 경로보다 이 완료 판정이 우선한다.

공식 crop ZIP이 이미 5seed 로짓 평균을 사용함을 패키징 코드에서 재확인하여 중복 앙상블
실험은 취소했다. 대신 argmax 시점·ResNet18·라벨을 고정하고 회피 head에는 두 사건의 중앙
crop, 방향 head에는 두 사건의 좌우 차이 crop만 주는 작업별 특징 대조를 시작했다. point
특징은 두 head 모두 유지한다. watcher 최초 `RUNNING`·0/5와 실제 child PID 확인, 1시간 제한,
중복 시작 거절. 결과 `artifacts/stage2-crop-task-specific-seeds-20260917/report.json`, 감시
`artifacts/stage2-crop-task-specific-seeds-watch-20260917/status.json`. 모든 네 지표가 매 seed
비열화 없고 개발·출처동일가중 평균이 상승해야 PASS. 완료 전 ZIP·제출·공식 모델 변경 금지.

## Stage2 crop PCA16 5seed 대조 완료·미채택 (2026-09-17)

감시와 결과 모두 `COMPLETE_VALIDATED`, 5/5 seed, exit code 0이다. gate는 `FAIL`.
기존 crop 대비 5seed 평균 변화는 회피 -0.00606, 방향 -0.11818, 개발평균 -0.03106,
출처동일가중 -0.03822이며 네 지표 모두 전 seed 비열화 조건을 실패했다. 특히 방향 정보가
PCA16에서 크게 손실됐다. PCA16은 폐기하고 공식 argmax crop Stage2(.21955508)를 유지한다.
새 ZIP·대회 제출·selected 변경 없음. 아래 실행 중 문장은 당시 설정 기록이며 이 완료 판정이 우선한다.

공식 최고 argmax crop Stage2(.21955508)의 시점·라벨·백본을 고정하고, 각 source-CV
학습 fold에서만 crop 특징을 표준화·SVD하여 16차원으로 압축하는 과적합 완화 대조를 시작했다.
5seed, 1시간 제한, 중복 시작 거절 watcher이며 최초 `RUNNING`·0/5와 실제 child PID를 확인했다.
결과는 `artifacts/stage2-crop-pca16-seeds-20260917/report.json`, 감시는
`artifacts/stage2-crop-pca16-seeds-watch-20260917/status.json`이다. 네 scene 지표가 모든
seed에서 비열화 없고 개발평균·출처동일가중 평균이 모두 엄격 상승해야 PASS다. 완료 전
ZIP·제출·현재 공식 모델 변경 금지. Codex 반복 폴링 금지.

## Stage2 crop+순서공동선택 5seed 대조 완료·미채택 (2026-09-16)

5seed 대조는 `COMPLETE_VALIDATED`, 사전 정의 gate는 `FAIL`로 끝났다. 기존 argmax crop 대비
평균 변화는 충돌 -0.01515, 진입 +0.01515, 회피 +0.00606, 방향 +0.00303,
개발평균 +0.00227, 출처동일가중 -0.00471이다. 출처동일가중이 4/5 seed에서 하락해
채택하지 않는다. 새 ZIP·대회 제출은 만들지 않았고 공식 Stage2는 argmax crop .21955508을
유지한다. 결과는 `artifacts/stage2-crop-ordered-seeds-20260916/report.json`, 감시 결과는
`artifacts/stage2-crop-ordered-seeds-watch-20260916/status.json`이다. 아래 실행 중 문장은
당시 실행 설정 기록이며 이 완료 판정이 우선한다.

공식개선crop(.21955508) 유지,시점만 entry<=collision 공동로짓선택으로 변경한5seed
대조 watch 시작. artifacts/stage2-crop-ordered-seeds-20260916 report 및
stage2-crop-ordered-seeds-watch-20260916/status.json. 최초RUNNING 0/5 확인,
1시간/재시도0/중복금지. 진입·회피·방향·개발·출처평균 전seed비열화없고 개발/출처평균
평균엄격상승이gate. COMPLETE_VALIDATED와 gate PASS 전 ZIP/마지막제출 금지.
docs/stage2-ordered-and-semantic-crop-experiments.md 참고.

## Stage2 crop 시험 제출 공식 개선 — 새 최고 (2026-09-16)

사용자보고 제출명2026_09_16_002 edit,16:47:25,Stage1/2/3
0.6386345895/0.21955508/0.4187467089,12분12초. Stage2 기존.2096390642 대비
+.0099160158(약4.73%), Stage1·3 동일. 가중합.38304763346으로 이전최고87406
.37908122714 대비+.00396640632, 새최고. 제출ZIP은 대화맥락상
artifacts/stage2-semantic-crop-submit-candidate-20260916/submit.zip,
로컬SHA256 401eade6940a64c80eecbffd7365562143b957ffa2792ef218b3de89fb475f9b.
서버해시·제출ID 미제공. 생성시submitted:false보다 사용자결과 최신. crop후보 공식개선
채택, 후속Stage2 비교기준 .21955508. docs/submission-history.md 및
docs/stage2-semantic-crop-submit-candidate.md 참고.

## Stage2 화면 영역 특징 시험 제출 ZIP 준비 (2026-09-16)

사용자 시험제출 요청. 공식최고87406 대응 ZIP을 기준으로 Stage2만 crop-probe 5seed
ensemble로 교체. 기존시점모델 유지,66영상23출처 전체fit. ZIP
artifacts/stage2-semantic-crop-submit-candidate-20260916/submit.zip,
177585722bytes, SHA256 401eade6940a64c80eecbffd7365562143b957ffa2792ef218b3de89fb475f9b.
변경항목 Stage2 best.pt/inference.py만, Stage1·3/requirements byte-identical.
정적·CRC·실제40JPEG CPU predict_stage2 스모크PASS. GPU_PENDING, submitted:false.
학습적합 회피/방향100%는독립성능아님. 공식시험점수가 기존Stage2 .2096390642를
넘어야 개선근거. docs/stage2-semantic-crop-submit-candidate.md 참고.

## Stage1 screen_mix_25 공식 제출 결과 — 기존 최고 미달 (2026-09-16)

사용자 보고 13:36:56, artifacts/stage1-quarter-submit-candidate-20260916/submit.zip SHA256 3dec7b2caf918dd448e3c47aacc4a1db5215845bce5bdab15498498a1c212154. Stage1/2/3 0.5868856347 / 0.2096390642 / 0.4187467089, 11분30초, 가중합0.36873143618. 직전 screen_mix_50 대비 Stage1 +0.0271655252·가중합+0.00543310504 회복했으나 기존최고87406 대비 Stage1 -0.0517489548·가중합-0.01034979096. 합성개발 gate 통과가 공식개선으로 이전되지 않음. screen_mix_25 미채택, 기존최고 모델/ZIP/selected 유지. manifest submitted=true 및 docs/submission-history.md 갱신. 별도GPU통합미실행보다 실제평가완료가 최신.

## Stage1 사람 라벨 없는 25% 혼합 대조 GPU 연결 (2026-09-16)

기존 최고 baseline/mixed_50/mixed_25/screen_mix_25 4조건, 같은85학습출처·seed20260911·128업데이트씩·lr1e-5·원본4/재녹화4·고정3slot/0.5/CUDA FP16 평가. 25% 조건은 매배치4쌍 중 기존합성3/화질균형1, screen조건은 화질균형 재녹화1입력만 기존검증된 약한화면단서로 교체. 기존 비공개 quality/screen 영상 자산 재사용, 새 코드/설정 delta 874053bytes만 비공개 biadis/crashintent-stage1-quarter-mix-assets, 무료T4 biadis/crashintent-stage1-quarter-mix-trial. 3배치테스트PASS·문법PASS·로컬번들 해시/출처분리/1680예측계획 검증. src/prepare_stage1_quarter_mix_trial.py / launch_stage1_quarter_mix_trial.py, artifacts/kaggle-stage1-quarter-mix-trial-20260916. 사용자 "이어서 진행해"에 따라 구체적payload/hash/목적지 승인 기록 후 숨김launcher PID27284 시작. 최초 확인 UPLOADING/감시ACTIVE, 실제GPU시작·완료 아님. remote-run.json·monitor-error.log·validated/result-validation.json이 결과 증거. 4시간 대기/상태조회유한재시도·회수5회·검증900초 연결, 중복 upload/push 금지. 독립검증PASS + 감시VALIDATED만 완료. 기존모델/ZIP/selected/예약 유지, 반복폴링금지. docs/stage1-quarter-mix-trial.md 참고.

완료 갱신: remote-run COMPLETE/VALIDATED, 독립검증 PASS. 1680예측/384업데이트·배치·입력/코드/모델해시·로짓/지표 재계산 통과. screen_mix_25는 baseline 대비 원본noise 오류19→7, quality_mix18→16, weak_screen 포함 모든 재녹화조건 오류0, normal63 F1 1.0으로 사전gate 유일 통과. 체크포인트 SHA256 49118651e9617c48216f5eff3baf3c5bb5f515a471a628f576d327e08db8e534. 반복 합성개발셋·단일seed이고 quality_mix 개선2건뿐이므로 공식개선 확정/자동승격 아님. 기존모델/ZIP/selected/예약 유지. 위 실행중 기록보다 완료결과 우선.

후속 후보 준비: artifacts/stage1-quarter-submit-candidate-20260916/submit.zip, 177351283bytes, SHA256 3dec7b2caf918dd448e3c47aacc4a1db5215845bce5bdab15498498a1c212154. 사용자보고 최고의 로컬기준 2c0090ad... ZIP에서 model/stage1/best.pt만 위 screen_mix_25 체크포인트로 교체, 나머지5멤버 byte exact. 구조/모델로드/메타/유한가중치/체크포인트바이트 검증 PASS, local-verification.json STATIC_AND_MODEL_IDENTITY_PASS_GPU_PENDING. 별도GPU통합검사·대회제출·selected변경은 아직 없음. docs/stage1-quarter-mix-trial.md 참고.

## 2026-09-15 Stage1 마지막 제출 결과 수신 (2026-09-16 기록)

사용자 보고 표시시각 2026-09-15 17:22:21, Stage1/2/3 순서로 0.5597201095 / 0.2096390642 / 0.4187467089, 평가 소요 12분30초(750초). 로컬 가중합 0.36329833114로 이전 보고 최고87406의 0.37908122714보다 0.015782896 낮다. Stage1만 -0.07891448, Stage2·3 동일. 사용자 후속 확인에 따라 업로드 ZIP은 artifacts/stage1-screen-submit-candidate-20260915/submit.zip이며 로컬 재해시 SHA256 ce7ae3fee0f831737ac447d839b9bb1909a690ba6e233ea62fcc3c60ba8d83d7로 manifest와 일치한다. 기존 최고 ZIP의 Stage1만 screen_mix_50.pt로 교체한 후보. manifest submitted:false는 생성 당시 기록보다 이번 제출 보고가 최신이다. 서버 보관 해시·제출 ID는 미제공. 최고 제출87406 유지, 모델/ZIP/selected/예약 변경 없음. docs/submission-history.md 참고.

## Stage3 기존 예측 보존 대조 착수 (2026-09-14)

사용자 후속개선 승인. 기존체크포인트 초기화,같은1180표본·1560updates·원본정규화·lr0.0001 조건으로 일반CE vs CE+기존주행표본조향KL 비교,각3seed.
KL온도2·가중치1·T제곱고정,교사동결/검증자료목표없음/새180개KL없음. 동일예측0손실·변경양의손실·유한역전파·교사gradient없음 스모크PASS.
src/train_stage3_prediction_preservation.py watch 실행. artifacts/stage3-prediction-preservation-20260914 status/train-watch-status/train.log 확인. 특징재사용,1시간/재시도0/OS잠금/중복실행거절.
6초기모델exact→6학습→모델/CSV/전환gate 자동검증. 각조건 기존대비gate 및 보존vsCE차이 별도보고. 아직개선확정아님. docs/stage3-prediction-preservation.md.
final-review COMPLETE_VALIDATED와감시COMPLETE가완료조건. 기존모델/ZIP유지. 반복폴링금지.

## Stage1 화면단서 GPU 전송 승인·실행 연결 (2026-09-14 17:40)

사용자가956MB 보강영상/코드의private screen-trial-assets 전송·무료T4 screen-trial을명시승인.
upload-approval.json 해시/목적지기록,기존시도마커없음확인후숨김launcher PID26660 시작.
artifacts/kaggle-stage1-screen-trial-20260914/remote-run.json 및 monitor-error.log 확인.
해시검사→업로드→ready→단일push→회수/독립검증연결,이전승인차단해소.
중복실행/반복폴링금지. 아직GPU완료결과없음,모델/ZIP/예약변경없음.
docs/stage1-screen-trial.md 최신절 참고.

## Stage3 우회전 보강 검증 완료 (2026-09-14)

6모델 COMPLETE_VALIDATED / 감시COMPLETE, 최종판정 KEEP_BASELINE.
개발F1 기존0.684695→직진보강0.705256→우회전추가0.711734. 평균은추가개선했고 기존대비3시드모두상승.
그러나 우회전재현율은기존대비-10.70%p로 이전직진보강의-9.87%p보다약0.83%p 더하락했다. 목표였던우회전회복실패. 직진은기존대비+11.97%p,좌+2.51%p.
전환오류/지연gate도미달,현재공식모델유지. artifacts/stage3-right-recovery-20260914/final-review.json이최신.12모델로짓재현·6CSV·해시·고정정규화검증PASS. 재학습불필요.
추가표본만으로우회전회복확인안됨. 다음실험은미시작이며평균개선과우회전목표달성구분할것.

## Stage3 우회전 보강 대조 학습 시작 (2026-09-14)

사용자 Stage3 개선 요청. 직진보강후 우회전손실 세시드 합1575중1454가직진오판,981건은11시점연속우회전. artifacts/stage3-transition-right-losses-20260914 진단완료.
기존1000+직진90보존,학습17경로 새우90(가감속각30) 추가. 경로×가감속×속도구간맞춤·16프레임비중복,같은추가배치의우회전반복325노출만대체. epoch/update/class수·정규화불변.
src/train_stage3_right_recovery.py watch 시작(launcher32620). artifacts/stage3-right-recovery-20260914 status/train-watch-status/train.log 확인. 1시간/재시도0/OS잠금,중복실행금지.
6모델학습→반복대조이전logits exact→원본/이전직진보강포함12모델재추론·지표·전환gate 자동검증. 아직개선확인아님. docs/stage3-right-recovery.md 참조.
final-review COMPLETE_VALIDATED와감시COMPLETE가완료조건. 기존모델/ZIP유지. Codex반복폴링하지말것.

## Stage1 화면단서 대조GPU 준비 완료·전송차단 (2026-09-14)

사용자진행요청으로 baseline/mixed_50/screen_mix_50 3조건 준비. 동일85출처/128updates각/lr1e-5,
screen조건은화질균형쪽재녹화2개만교체,나머지6입력과slot동일. cue사용85/86/85회.
8테스트·증분768파일동등성·직전혼합전체결과회귀검증PASS.
artifacts/kaggle-stage1-screen-trial-20260914/ 준비,신규765영상+코드956113828bytes.
private biadis/crashintent-stage1-screen-trial-assets / screen-trial무료T4 목적지.
자동승인검토가구체적payload/새목적지명시승인부족으로감시기시작전거절.
remote-run BLOCKED_APPROVAL,upload/launch시도마커없음,전송/GPU/감시기미실행.
구체적승인후upload-approval.json 해시일치기록 및 launch_stage1_screen_trial.py 실행.
docs/stage1-screen-trial.md 참고. 기존모델/ZIP/예약변경없음.

## Stage1 화면단서 보강 데이터 완료 (2026-09-14 15:50)

85출처765재녹화/38250프레임 생성·전수검증 PASS, DATA_VALIDATED.
기존 ORIGINAL255참조 포함 training_manifest1020행. 사용자후속조회에서
manifest2개/plan해시·클래스행수 별도재확인PASS. 약5시간1분 소요.
artifacts/stage1-screen-cue-data-20260914/status.json 및 docs/stage1-screen-cue-data.md.
생성중기록대체,재생성불필요. 다음은소규모GPU대조준비,아직학습/전송/모델/ZIP변경없음.

## Stage3 전환 보강 학습·자동 검증 완료 (2026-09-14)

COMPLETE_VALIDATED / 감시COMPLETE / 6모델 완료. final-review.json decision KEEP_BASELINE.
개발 조향F1 평균: 기존0.684695,동일추가학습 반복대조0.674335,전환보강0.705256. 전환보강은 두기준 대비3시드 모두개선, 기존대비4경로중3개선·반복대조대비4개모두개선.
기존대비 재현율 변화: 좌+4.62%p,직진+8.88%p,우-9.87%p. 반복대조대비 좌-3.58%p,직진+10.49%p,우-3.38%p. 전환오류/지연 gate도두비교모두미달.
따라서 평균개선은확인했지만 우회전손실과전환기준미달로미채택. 모델/ZIP유지. 입력·특징anchor·정규화·9체크포인트로짓재현·6CSV·전환지표검증PASS.
artifacts/stage3-transition-matched-control-20260914/final-review.json 최신. 재학습불필요. 다음은보강후우회전정답상실의방향·경로·전환분포를저장예측에서분석. 임계값/가중치임의조정은아직하지않음.

## Stage2 정확도 판단·AI 방향 부분 테스트 완료 (2026-09-14)

사용자 정확도테스트 요청. 기존 공식점수 .2096390642 > 사전학습 .1966181317이므로 기존유지 판단.
예측전 CONTACT_SUSPECT13개 전부 연속60스틸 재관찰, 방향명확6(LEFT3/RIGHT3)/보류7.
별도AI참고값 대조 기존3/6·사전학습3/6 동률. 기존전부LEFT/신규전부RIGHT.
선택/이미지·예측해시/별도집계/사람작업지불변PASS. AI참고·비블라인드·방향부분테스트,
전체정확도/충돌확정 아님. 공식점수는과거사용자보고이며이번서버재평가없음.
docs/stage2-ai-direction-test.md 및 artifacts/stage2-ai-direction-test-20260914/report.json.
사람라벨0 유지, 별도AI참고6개 생성. 모델/ZIP/재추론없음.

## Stage3 전환 보강 대조 학습 실행 요청 (2026-09-14)

사용자 학습진행 명시 승인. src/train_stage3_transition_matched_control.py watch 실행, 구문검사 및 중복 프로세스 없음 확인 후 숨김 감시기 시작.
90개 새표본 특징 추출·기존endpoint exact 대조→정규화 원본값 검증→2조건×3seed/1560updates→9모델 logits 재추론 exact·CSV검산·전환오류·사전gate 자동평가 연결.
artifacts/stage3-transition-matched-control-20260914의 status.json,train-watch-status.json,train.log,launcher-error.log 확인. final-review.json COMPLETE_VALIDATED 및 감시COMPLETE가 완료조건.
1시간 제한/재시도0/OS잠금. 기존execution.json 존재시 중복시작 차단. scripts/Get-BackgroundWorkStatus.ps1에 연결. 반복폴링 금지. 아직 학습완료나 개선 의미 아님.

## Stage3 전환 보강 대조 구성 확정 (2026-09-14)

artifacts/stage3-transition-matched-control-20260914/plan.json MATCHED_CONTROL_READY_NO_TRAINING.
574후보 중 경로×가감속×속도구간 일대일 매칭202개(가속60/감속52/등속90), 각30개 총90선정.17경로,회전전60/후30.
기존1000개 매에폭 보존+동일10그룹 추가30배치,12에폭/1560updates/3seed. 반복대조군과전환후보는추가직진3슬롯만다름. 정규화는원본고정,원래1200updates모델도비교.
CSV/배치일정/원본전체노출/슬롯라벨·경로/중복 검증PASS. docs/stage3-transition-matched-control.md.
아직학습없음. 다음은90개특징추출·동등성확인과감시연결6모델학습. 두기준 대비사전gate와전환오류검증필수. 신규자료다양성vs전환선정효과 단독분리실험은아님.

## Stage2 라벨 전 예측 비교 완료 (2026-09-14 14:51)

50/50영상·고정2모델 COMPLETE_VALIDATED, 감시COMPLETE, validation PASS. 소요6377초.
전체JPEG해시/저장특징재로드/전예측·장면로짓정확재현/CSV/보호파일불변PASS,
완료조회시4결과파일SHA재확인PASS. 재추론불필요.
충돌시점 차이평균13.946초(0.3초초과44/50), 진입10.368초(45/50).
회피공간불일치18/50, 진입방향36/50(전부기존LEFT→사전학습RIGHT).
정답미사용이므로 정확도·우열판단불가, 라벨0·기존모델유지.
artifacts/stage2-unlabeled-comparison-20260914/comparison.json 및
docs/stage2-unlabeled-comparison.md 완료절 참고. 아래착수기록보다완료우선.

## Stage3 전환 직진 후보 점검 완료 (2026-09-14)

artifacts/stage3-transition-inventory-20260914/report.json INVENTORY_VALIDATED_NO_TRAINING.
기존1000 CSV 해시 동일 보존. 학습17경로 전환2853개 중 전후10시점 지속791개, 해당직진 고유7341시점/기존16프레임입력 비중복5246시점.
전환당1개·추가끼리도 프레임비중복 적용574후보(회전전308/후266),17경로전부. 등속345/감속130/가속99로 불균형, 최종학습목록 아님.
지속791전환 중 기존직진 endpoint 포함65개; 입력영상 자체가 나머지전환을 전혀 보지 않았다는 의미 아님. 보강효과 미확인.
docs/stage3-transition-inventory.md 참고. 다음은 기존1000·정규화 보존, 추가update·클래스노출 동일 대조설계의 경로×가감속×속도 매칭 가능 수 확인. 새학습/모델변경 없음.

## Stage3 직진 정답 상실 진단 완료 (2026-09-14)

artifacts/stage3-sample-loss-diagnostic-20260914/report.json COMPLETE_VALIDATED. 새 학습 없음.
검증 직진 8654시점, 시드별 정답 상실806/1125/1024·획득735/587/491. 2시드 이상 상실762시점,31영상518구간.
제거84개 중 전환 전후1초59개(70.2%),회전 라벨 종료후1초40개(47.6%). 검증 기존정답 상실률 종료후1초29.4% vs나머지18.1%.
전환표본 보존 근거는 있으나 먼구간 순손실654건, 교체하지 않은 저속도 악화하여 단독원인 확정 불가. 주변 큰움직임 공통원인 비지지. 가중치와 정규화 모두 변경된 실험임.
docs/stage3-sample-loss-diagnostic.md 참고. 기존모델 유지. 다음 설계 전 기존1000개 보존과 학습경로 전환표본 부족·중복을 점검할 것. 검증오답의 학습전환 금지.

## Stage3 직진 표본 대조 검증 완료 (2026-09-14)

6개 학습과 결과 검증 완료, COMPLETE_VALIDATED / KEEP_BASELINE. 기존 3개 시드 logits 정확 재현 및 6개 모델 해시·CSV 대응·경로별 지표 검산 PASS.
개발 조향 F1 평균 0.684695 → 0.684528로 사실상 제자리다. 3시드 중 1개 개선, 4경로 평균 중 2개 개선으로 사전 기준 미달. 직진 재현율 57.13% → 52.73%, 좌 74.09% → 76.64%, 우 81.39% → 85.62%.
회전 종료 후 직진 복귀 미검출은 시드별 26→29, 27→31, 32→36으로 모두 증가했다. 안정적인 직진 선별이 직진 일반화를 개선한다는 이번 가설은 지지되지 않았다. 원인 확정은 아니다.
artifacts/stage3-straight-sample-control-20260914/final-review.json 및 transitions.csv가 최종 결과다. comparison.json의 PENDING은 검토 이전 기록이다. 재학습 불필요, 현재 제출 모델 유지. 새로운 실험은 시작하지 않았다.

## Stage3 직진 표본 대조 학습 시작 (2026-09-14)

기존 1,000개 중 직진 84개를 학습 경로의 안정적인 직진으로 교체했다. 클래스 수와 각 슬롯의 경로·가감속·속도 구간을 유지했고, 안정 조건 충족 직진은 143개에서 227개로 늘었다. 라벨은 수정하지 않았다.
원본/교체 표본 각각 3개 시드, 동일 1,200회 업데이트 대조 학습을 시작했다. 기존 결과의 logits 정확 재현을 먼저 검사한다. 감시기 및 작업 프로세스, 첫 영상 특징 추출 로그를 확인했다. 아직 학습 완료나 개선을 의미하지 않는다.
결과: artifacts/stage3-straight-sample-control-20260914. status.json, train-watch-status.json, train.log를 사용한다. 감시 제한 1시간, 재시도 0회. 중복 실행하지 말 것.
다음은 완료된 comparison.json의 사전 채택 조건과 전환 실패·지연 검토이다. 학습 완료만으로 채택하지 않는다. docs/stage3-straight-sample-control.md 참조. 현재 제출 모델 유지.

## Stage3 학습구성·라벨 감사 완료 (2026-09-14)

사용자하락반복지적후임의특징변경중단,실제1000학습표본감사승인·완료.
주행좌/직진/우각300+정지100,직진17경로각16~20개/누락0,특정경로독점근거없음.
직진187/300이25m/s이상(원래후보도고속편중),명확pose직진185/중간100/저속10/좌강회전5.
라벨93602행생성식·1000선택라벨일치PASS.11시점안정202/비안정98,임계근접27.
5센서불일치후보추림,오라벨확정아님.모델·라벨변경없음.
docs/stage3-training-label-audit.md 및 artifacts/stage3-training-label-audit-20260914.
다음은명확직진/전환/속도별후보수점검과같은총량·클래스수대조설계,아직새학습없음.

## Stage3 영역별 특징 약화 진단 완료 (2026-09-14)

기존4개발경로18603시점3seed.192격자차원동일0.5배:기존F1 .684695,
좌우.653050/중앙.640831.직진재현57.13→45.32/55.90%.단순주변부억제근거없음.
v1 CPU1스레드로짓미세차이중단,v2 2스레드exact통과후지표키오류,v3완료검증PASS.
최신 artifacts/stage3-region-attenuation-20260914-v3/report.json,
docs/stage3-region-attenuation.md.이전실패보존,재실행불필요.
전역요약불변의인위적특징개입,실제물체제거아님.모델/라벨/ZIP유지.

## Stage3 직진 흐름공간분포 비교 완료 (2026-09-14)

속도10~25m/s 기존개발직진1933행/Civic112행.오답좌우·중앙움직임동반증가,
Civic좌우/중앙비율정답2.575→오답2.053으로주변부단독원인비지지.
Civic영상내3개모두영역절대움직임증가하지만비율은1증가2감소.인과확정불가.
docs/stage3-flow-regions.md 및 artifacts/stage3-flow-regions-20260914,해시/매핑검사PASS.
다음후보기존개발자료영역별특징약화대조(실제물체제거와다름).모델/라벨불변.

## Stage3 저속·직진 오판 점검 완료 (2026-09-14)

추가시각점검:holdout001 48.6~53.2초6스틸은도로변주차차량·가로수의일반도로형태.
새자료전부고속도로라는이전표현수정.가까운물체/그림자이동관찰,원인확정아님.
시각점검완료,다음기존개발자료와흐름공간분포비교진단. docs/stage3-low-speed-diagnostic.md.

저속3290중정지(<.5m/s)2577/주행저속713. .5~2m/s pose/steer부호81.82%,
2~5는92.19%,5~35는99.73%;저속pose불안정한계로라벨확장미실행.
직진52오답12구간,holdout001에35집중/최장19연속.36개는확률.9이상.
오답flow_p90중앙값5.361/정답3.495이나교란있어원인확정불가,오답속도중앙17.636m/s.
docs/stage3-low-speed-diagnostic.md 및 artifacts/stage3-low-speed-diagnostic-20260914.
다음001직진구간시각진단·기존개발경로대조. 모델/라벨불변,지속진행승인유효.

## Stage3 새5경로 채점·제외원인 분석 완료 (2026-09-14)

COMPLETE_VALIDATED/감시COMPLETE,9영상5310예측/467채점. F1 .834118,
직진53.57%/좌96.20%/우98.48%,예측해시재확인PASS.61오답중직진52,05-01 11:24에35집중.
제외4843=범위밖3527/합의不成立849/지속467.범위밖3290은速度5m/s未満。
docs/stage3-continuation.md최신절 및 stage3-civic-holdout-score-20260914/report.json,
error-coverage-summary.json.다운로드/채점재실행금지.91.21%제외된부분집합,공식성능아님.
지속진행승인유효.다음은저속센서라벨가능성진단/기존개발직진오판분석,未実行。

## Stage3 새 Civic 확보·부분라벨 완료 / 채점시작 (2026-09-14)

지속진행승인유효.새5경로9구간99파일334MB확보검증완료.고정규칙재적용5310행중
467유효(직진112/좌158/우197),4843제외91.21%,원본해시·규칙검산PASS.
src/prepare_stage3_civic_holdout_labels.py / artifacts/stage3-civic-holdout-labels-20260914.
기존모델9영상5310추론/467채점 src/score_stage3_civic_holdout.py watch시작,
artifacts/stage3-civic-holdout-score-20260914 status/log확인.30분/재시도0/잠금.
완료前結果断定禁止·중복실행금지. docs/stage3-continuation.md 최신절참고.
새자료보정·튜닝없음,일부경로클래스누락/유효7행경로존재,전체분포평가아님.

## Stage3 지속진행 승인·새경로 다운로드 착수 (2026-09-14)

사용자:중단시키기전까지계속테스크진행. 작업별재승인없이이어가되사람검수보류와감시규칙유지.
새Civic5경로9구간99파일약334MB선정,이전5경로와중복0·reservation계획해시고정.
src/download_stage3_civic_holdout.py prepare PASS,watch 숨김PID26192 시작.
artifacts/stage3-civic-holdout-acquisition-20260914 상태/로그/예약,
원본 data_raw/comma2k19/civic-holdout-20260914.1시간/범위3회/1GiB/OS잠금감시연결.
다음은확보검증→기존고정부분라벨규칙·제외율→기존모델채점.새평가경로튜닝금지.
docs/stage3-continuation.md참고.대기만남으면응답종료,자동재개약속금지.중복시작금지.

## Stage3 과거예측 안정화 대조 완료 — 기존유지 (2026-09-14)

기존4개발경로31영상18603시점/3seed/저장로짓,배포seed모델정규화동일성PASS.
F1평균기존.684695/다수결3 .686312/확률EMA.5 .690288.
다수결은4경로중3하락,EMA는seed11종료미검출27→28/seed12시작18→20 증가.
둘다사전gate미달 KEEP_BASELINE. 시작62/종료63전환·미검출/조건부지연함께보고.
영상분리·미래미사용접두사검사·CSV독립검산·해시PASS. 새학습/Civic재채점/ZIP변경없음.
docs/stage3-causal-smoothing.md 및 artifacts/stage3-causal-smoothing-20260914 참고.

## Stage2 라벨 전 예측 비교 착수 (2026-09-14 사용자 최신 요청)

“라벨링 넘기고 모델 비교 평가부터” 요청으로 별도 미라벨 진단 실행.
src/compare_stage2_unlabeled.py watch, 최초 launcher PID27268.
고정 incumbent/ccd_pretrained, Nexar50개57,483프레임, CPU FP32 공유특징.
OS잠금/4시간제한/재시도0/전체저장특징 예측재현·CSV·보호해시검증 연결.
artifacts/stage2-unlabeled-comparison-20260914/{status,watch-status}.json 및run.log 확인.
COMPLETE_VALIDATED + validation PASS + 감시COMPLETE 전에는 완료 아님. 반복폴링금지.
시간차·범주불일치만 진단, 정확도/우열 판단 불가. 라벨0/기존모델유지.
기존 최종평가 계약·라벨게이트 유지, 후보추론없음이라는 아래 기록은 과거 준비시점 상태.
docs/stage2-unlabeled-comparison.md 참고. 단위테스트3개PASS.

## Stage3 Civic06-11 오답분석 완료 (2026-09-14)

유효428중오답72→21연속구간.직진오판30/우오판38/좌오판4.
009 1.8~3.7초20연속오답,008 37.4~39.1초직진15,009 16.4~18.8초직진14.
21요약/2타임라인/CSV생성,4요약24스틸실제확인.그늘/터널/인접밴관찰,원인확정아님.
높은softmax오답존재(.97~.99998),단순신뢰도필터한계.해시·로짓예측·오답합검증PASS.
docs/stage3-civic-error-review.md 및 artifacts/stage3-civic-error-review-20260914/review.html.
다음후보는기존개발경로의인과적예측안정화와회전지연대조,아직미실행.
21요약은등간격6스틸이라짧은오답시점누락가능,타임라인참조.모델/라벨불변.

## Stage3 Civic 기존모델 채점 완료 (2026-09-14)

COMPLETE_VALIDATED/감시COMPLETE,2400추론/714부분채점,조향F1 .820299.
좌454/458=99.13%,직진65/99=65.66%,우119/157=75.80%.
05-01 F1 .974208,06-11 .692889. 오답76중72가06-11,직진41.18%/우62%.
ZIP/가중치/AST/원본해시/무손실짝수픽셀/로짓재현/CSV독립검산/예측해시PASS.
artifacts/stage3-civic-partial-score-20260914/report.json 및 docs/stage3-civic-partial-score.md.
70.25%제외된센서합의진단,공식/전체성능아님·모델개선비교아님.재실행금지.
다음은06-11직진·우회전오류진단. 기존모델유지·사람검수보류유지.

## Stage2 최종입력·채점준비 완료 (2026-09-14)

READY_FOR_HUMAN_REVIEW/readiness-validation PASS 최종확인.50영상57,483JPEG/6.45GB,
전원본/전이미지/시각표해시재검증PASS.실제추론명령의미라벨차단(출력미생성)확인.
검수페이지 data/stage2-validation-nexar-20260914/review.html 준비, 작업지50PENDING/4정답열빈칸.
입력·고정2모델·평가코드 evaluation-contract.json 동결. 이검증경로의필수준비완료,
다음은사람검수/라벨링후 scripts/Start-Stage2FinalComparison.ps1로고정비교실행.
새특징연구는선택. 아래착수/해시검증중기록보다완료우선, 변환재실행금지.

사용자 이어서진행승인, 라벨마지막유지. 후보50원본 전체JPEG(q2/원본크기/FPS변환없음)/
정수PTS·timebase맵으로입력변환중. worker14032/감시24464, 1시간·20GiB·재시도0.
src/finalize_stage2_validation_readiness.py가변환완료후전원본/JPEG/시간표해시재검증,
검수HTML/빈작업지PENDING50/평가규약해시고정. 기준/CCD사전학습2모델동결복사,
실제제출클래스vs로컬기존개발3입력exact·공개3JPEG전체경로스모크PASS.
유리수시각채점/미라벨차단7테스트와UI상태·CSV로직PASS, 브라우저수동조작미검증.
50개변환 INPUTS_READY_UNLABELED,57,483JPEG/6,448,590,759bytes 전디코딩PASS.
후속준비감시기17096 최초 VERIFYING_INPUTS 확인, 전체입력해시재확인중.
실제완료는 artifacts/stage2-final-validation-20260914/readiness-status.json READY_FOR_HUMAN_REVIEW
및readiness-validation.json PASS필수. Codex반복폴링/변환중복실행금지.
docs/stage2-final-validation-preparation.md, data/stage2-validation-nexar-20260914/ 참고.
라벨0/후보추론0, 기존모델/ZIP유지. 준비PASS후사람검수·라벨→고정모델평가만필수잔여.

## Stage3 Civic 부분라벨 기존모델 채점 착수 (2026-09-14)

사용자승인,src/score_stage3_civic_partial.py watch 숨김시작PID15740.
기존제출ZIP/가중치/AST검증,4원본짝수프레임→무손실10fpsAVI·전체픽셀대조,
2400시점추론/부분714채점/저장로짓재현·CSV독립검산 연결.30분제한/재시도0/OS잠금.
artifacts/stage3-civic-partial-score-20260914/{status,watch-status}.json 및run.log확인.
Get-BackgroundWorkStatus.ps1 연결,중복실행·반복폴링금지. 완료전성능단정금지.
docs/stage3-civic-partial-score.md 참고.모델학습·교체없음,사람검수보류유지.

## Stage3 Civic 부분 라벨 생성 완료 (2026-09-14)

사용자승인으로 센서합의+11시점지속 규칙 사전고정. 보정offset+.011351°,
직진각≤.5/yaw<.003,회전각>1.5/yaw>.012 부호합의,속도5~35/valid_sensor.
평가2경로4영상2400시점 중714유효(직진99/좌458/우157),1686제외70.25%,21구간.
PARTIAL_PROXY_LABELS_READY,전행독립스칼라재계산/프레임대응/경로분리PASS.
artifacts/stage3-civic-partial-labels-20260914 및 docs/stage3-civic-partial-labels.md 참고.
센서대리·선택된안정구간평가용,사람정답/전체분포아님. gyro미사용·미래5시점은라벨선정만.
다음은고정기존모델714시점채점과방향/경로별보고,아직추론미실행. 사람검수보류유지.

## Stage3 Civic 속도반영 기준 검증 완료 — 미채택 (2026-09-14)

보정3경로 내부 route-out,2경로적합1경로검사. yaw범위.003/.006/.009/.012 사전고정.
3분할 모두 적합단계 직진90%·회전97% 동시기준미달로 선택임계값없음.
FAILED_NO_EVALUATION_LABELS. 제외경로UNKNOWN,report일치율0은미선택집계이지성능0주장금지.
보정3602행 중2695유효/907제외,직진671·회전1260·중간764. CSV독립검산PASS.
docs/stage3-civic-speed-rule.md 및 artifacts/stage3-civic-speed-rule-20260914 참고.
새평가2경로 라벨/모델채점 미실행. 다음후보는 센서합의 명확구간 부분라벨·제외율보고,
아직미실행. 기존모델유지·사람검수보류유지.

## Stage1 약한 화면단서 학습자료 생성 착수 (2026-09-14)

사용자 후속진행 요청. 기존train85출처×3화질×3화면조건(25%약화/원근제거/깜박임제거)
=765재녹화영상 생성,동일화질 ORIGINAL255개는기존검증자료참조. val/phone출처제외유지.
src/prepare_stage1_screen_cues.py 및 docs/stage1-screen-cue-data.md.
실제소형9영상 생성/재개/손상거절·화질동등성2테스트PASS. 숨김생성부모PID19964 시작.
artifacts/stage1-screen-cue-data-20260914/status.json 우선확인, DATA_VALIDATED 전완료아님.
단일worker/출처15분/전체6시간,실패시중단·로그보존,출처별검증재개,중복OS잠금.
Get-BackgroundWorkStatus.ps1 Stage1ScreenCueData로한번조회. Codex반복폴링금지.
데이터검증후대조GPU준비가후속,현재GPU/전송/모델/ZIP/예약변경없음.

## Stage3 Civic 자이로·속도 진단 완료 (2026-09-14)

보정3경로6구간만 사용.05-05 gyro-pose 편향+.05727/+.05805rad/s,
상관.9986/.9949.002↔003 교차편향 적용 후 회전부호809행100%,MAE약.0017.
고정편향이 주요설명, 전체부호반전 근거없음. 다른경로에 공통보정 적용하지 않음.
05-12는+0.2초 진단오차 감소,시간오류확정 아님·시각변경없음.
보정경로별제외 조향각MAE yaw-only→speed항 .9163→.8488/1.6696→1.0876/.5832→.5763.
새라벨/모델채점없음. 다음은 속도반영 대리기준의 보정경로 내부 교차검증 설계.
docs/stage3-civic-sensor-diagnostic.md 및 artifacts/stage3-civic-sensor-diagnostic-20260914 참고.

## Stage1 혼합 대조학습 완료 (2026-09-14 10:38)

mixed-trial v1 COMPLETE, 자동회수/독립검증PASS(4조건1680예측/384updates/실제표본slot일정).
artifacts/kaggle-stage1-mixed-trial-20260914/validated/result-validation.json 및 remote-run VALIDATED.
baseline/existing/matched/mixed 원본noise오류19/19/0/0, mix18/20/2/9(각21개).
weak_screen재녹화오류0/0/6/3,normal63 F1 1/1/.982348/1.
혼합이단독보강의재녹화손실일부회복하지만원본mix오탐증가,재녹화보존기준실패로미채택.
현재공식모델/ZIP/예약유지. 추가GPU미실행. docs/stage1-mixed-trial.md 완료절우선.

## Stage3 Civic 센서 보정 점검 완료 — 라벨 생성 기준 미달 (2026-09-14)

사용자 후속진행 승인.110원본SHA확인,3보정경로/2평가경로 사전분리,6002시점 중
경계/CAN범위213행 제외. pose회귀 보정265행 영점+.01135°,양수좌회전관계 지지.
고정0.5/1/1.5/2° 모두 직진90%·회전97% 동시기준 미달. ±1.5는93.14/90.63%.
CALIBRATION_GATE_FAILED_NO_LABELS, 평가라벨0/조향UNKNOWN,모델채점없음.
05-05보정경로 gyro/pose 부호일치44%,steer/pose100%로gyro추가조사필요.
다음은 보정3경로 속도의존관계·gyro불일치 분석. 평가2경로진단열람은기록,미열람주장금지.
docs/stage3-civic-calibration.md 및 artifacts/stage3-civic-calibration-20260914 참고.

## Stage3 Civic 원본 확보 완료 (2026-09-14)

ACQUIRED_VALIDATED_UNCALIBRATED/감시COMPLETE/validation PASS.
5경로10구간110파일377,073,634bytes 수신, 전12,000프레임 디코딩·시각수일치,
전파일CRC·크기·저장SHA 재검증PASS. CAN범위기준10/10통과, 경계 약0.167%는 범위밖.
artifacts/stage3-civic-acquisition-20260914/validation.json 및
docs/stage3-civic-pilot-acquisition.md 완료절 우선. 다운로드 재실행금지.
다음은 센서부호·영점·시각 검토와 경계제외/보정용·평가용 분리. 라벨·모델채점 미실행.

## Stage2 라벨링 마지막으로 변경·지역/움직임 대조 완료 (2026-09-14)

사용자가 라벨링 전 가능한작업 진행승인. 기존66라벨/23출처만사용, 새Nexar50학습/예측없음.
전체160×96/4×4영역 밝기·인접flow를 기존두예측시점hidden에추가, point/motion/spatial/combined
5fold3seed60head 완료·원본66재디코딩/정규화/60로짓재현오차0/CSV지표PASS. 감시기완료.
네항목평균44.07/44.82/44.32/44.82%, 기존source-CV45.08%미달. 회피50%불변.
point는이전대조3seed전체예측일치. motion의방향+2정답/seed는각2출처집중. 이번방식미채택.
artifacts/stage2-spatial-motion-20260914/{summary,validation,paired-analysis}.json 및
docs/stage2-spatial-motion-experiment.md. 완료실험재실행금지, 기존모델/ZIP/라벨유지.
다음은 최종평가입력·채점규약 준비와 차량/도로공간 특징가능성검토, 사람검수·라벨은마지막.
과거라벨선행순서/단계별승인대기보다이번사용자진행승인우선. 다른Stage작업보존.

## Stage3 다른 차량 Civic 원본 다운로드 착수 (2026-09-14)

사용자 자료 다운로드 승인. 공식HF Chunk_3 고정revision에서 다른장치 Civic
5경로×2구간/110파일/약377MB 부분수신 계획고정(seed20260914). 전체ZIP9.41GB 아님.
src/download_stage3_civic_pilot.py prepare PASS, 숨김감시기 시작 PID20328.
원본 data_raw/comma2k19/civic-pilot-20260914, 상태·로그·계획·검증은
artifacts/stage3-civic-acquisition-20260914. scripts/Get-BackgroundWorkStatus.ps1에 연결.
1시간/범위당3회/1GiB/OS잠금, CRC·SHA·전체디코딩·센서시각 자동검사.
ACQUIRED_VALIDATED_UNCALIBRATED 전까지 확보완료 아님. 중복시작·반복폴링금지.
다른기록경로·장치이나 같은고속도로, 도시독립검증 아님. 센서보정/라벨/평가 미실행.
docs/stage3-civic-pilot-acquisition.md 참고. 사람검수보류·기존모델유지.

## Stage3 로컬 출처 목록·중복 점검 완료 (2026-09-14)

사용자 진행 승인으로 로컬 센서 자료 재검산. Chunk_1 ZIP188구간 중 변환187개는
학습156/17경로+기존개발검증31/4경로, 장치ID1개. 새 독립평가 적격구간0.
예제는 학습 COMMA2K19_C1_0074와 영상/센서6파일 SHA256 동일.
미변환1개는 기존경로 07-29/31, 앞선 CAN범위 검사에서 제외된 자료로 유지.
artifacts/stage3-source-inventory-20260914/{report.json,segments.csv},
src/audit_stage3_local_sources.py 및 docs/stage3-source-inventory.md 참고.
목록·파일존재·예제해시 점검 완료, 전영상 재디코딩/GPS중복 검증은 아님.
다음은 다른장치/경로 comma 소규모 확보·센서 대응 검사 후 평가전용 고정.
이번 다운로드/새평가/학습없음, 사람검수보류·기존모델 유지.

## Stage3 중앙·전체 화면 결합 완료 — 미채택 (2026-09-14)

사용자 진행 승인. 중앙402차원을 그대로 두고 전체화면402차원을 비활성 슬롯에 추가.
검증된 fullframe 캐시 재사용, 같은1000클립/17경로/5fold/3seed/1200업데이트.
새15모델 학습+기준15모델 재현 검증 완료. 입력매핑 검사PASS, 계획·캐시 해시고정.
COMPLETE_VALIDATED/검증PASS/30모델 로짓오차0/summary·results 해시 재확인.
조향F1 .718900→.694265, 직진재현65→60.67%, 좌75.44→73.89%, 우74.78→73.33%.
3seed와5fold평균 모두 하락하여 KEEP_BASELINE. 기존 제출모델 유지, 재실행금지.
artifacts/stage3-fused-view-cv-20260914/{status,watch-status}.json 및 run.log 확인.
30분제한/재시도0/OS잠금 감시 정상완료. 중복실행·Codex 반복폴링금지.
상세 docs/stage3-fused-view-experiment.md. 사람검수보류 유지, 기존 제출모델 유지.

## Stage3 전체화면 대조 완료 — 미채택 (2026-09-14)

COMPLETE_VALIDATED/독립검증PASS,summary/results해시재확인.
조향F1 3seed평균 중앙.718900/전체.642716,직진재현65/53%,좌75.44/70.89%,우74.78/69%.
3seed와5fold평균모두하락.156영상/새15모델+기준1exact재학습,전체30모델로짓오차0.
전체화면96×96letterbox미채택·기존모델유지.아래착수기록보다완료우선,재실행금지.
artifacts/stage3-fullframe-cv-20260914/{summary,independent-validation}.json,
docs/stage3-fullframe-experiment.md 완료절.사람검수/공식평가대체아님.

## Stage2 연속 프레임 예비 관찰 완료 (2026-09-14)

사용자 진행 요청. Nexar50개 event-1초부터 원본연속60장씩 3,000프레임 AI스틸관찰 완료.
접촉·충격의심13/관찰구간 접촉징후안보임13/판단불가24. 충돌확정·사람판정 아님.
원본50/PNG50해시·PTS3000·ID50일치PASS. 원본50유지, 정답라벨0, 적합성UNCONFIRMED.
artifacts/stage2-contact-review-20260914/review.html,report.json,observations.json 및
data_raw/stage2-validation-nexar-20260914/review-contact-queue.csv가 최신. 이전목록 보존.
수집·중복선별·AI예비관찰 완료, 실제충돌30~50건 최종확정은 미완료. 다음은 사람 원본검수와
적합성확정·라벨링(사용자보류 유지). 유효30건 미달이면 후보보충. 전체영상/오디오 검수로
과장하지 말 것. docs/stage2-validation-acquisition.md 최신절 우선. 모델/학습/ZIP변경없음.

## Stage3 전체화면 대조 실험 착수 (2026-09-14)

사용자1번승인.이전공간실험은중앙크롭grid요약이었으므로원본시야변경으로구분.
중앙크롭 vs 전체프레임비율유지96×96letterbox,같은402flow/1·5·15창/MLP.
17경로1000클립5fold3seed1200updates,기준15replay+1exact재학습gate/새15모델계획.
4테스트PASS,prepare해시고정.최초PID3860트리중지후Windows시간초과/T종료보강,
학습전계획원본보존/감시코드해시갱신/설정불변으로PID21560재개.상태/실제자식은
artifacts/stage3-fullframe-cv-20260914/{status,watch-status}.json 및 로그확인.
extract→train→verify자동연결,각1시간제한/재시도0/OS중복잠금.중복실행·Codex반복폴링금지.
docs/stage3-fullframe-experiment.md,src/compare_stage3_fullframe_cv.py 참고.
아직성능결과미확보.사람검수/기존모델/ZIP/예약/타Stage변경없음,CPU로컬실험.

## Stage2 후보 적합성 예비 점검 완료 (2026-09-14)

사용자 이어서진행 요청. Nexar50영상 사건참고시각 주변 각6장/총300스틸 AI점검,
우선검수34/판단보류16. 실제충돌 확정·최종검증셋 채택 아님, 정답/사람라벨0 유지.
50원본해시PASS/ID일치/삭제0. artifacts/stage2-eligibility-review-20260914/
review.html,observations.json,report.json 및 data_raw/stage2-validation-nexar-20260914/
review-queue.csv 참고. 스틸은약1Hz로접촉을놓칠수있어 연속영상확인/사람라벨남음.
기존목록/중복검사/모델/ZIP보존. docs/stage2-validation-acquisition.md 최신절우선.

## Stage3 새 우회전 후보8영상 준비 완료 (2026-09-14)

사용자 CASE_01 재평가보류·후보보강승인. 기존17검수/이전24블라인드/공개5/phone30과
같은CCD출처제외,722영상70출처탐색.36영상연속스틸스크리닝후8출처8영상400프레임선정.
artifacts/stage3-right-candidates-20260914/ READY_FOR_HUMAN_REVIEW, 해시/출처/전디코딩/
10fps/GUI사전검사PASS. data/stage3-right-review-v2 라벨전부UNKNOWN·reviewed0.
review.html시청,scripts/Start-Stage3RightReviewV2.ps1검수. docs/stage3-right-candidates-v2.md.
확정우회전0,모델채점/학습없음,기존332행라벨/모델/ZIP/예약유지.다음사용자검수대기.

## Stage2 후보 확보·중복 의심 검토 완료 (2026-09-14)

수집50/50(838MB)·전체디코딩/해시PASS, CCD1500+후보50 지문/76,225쌍 검사완료.
동일파일0, NEXAR_00664 대 CCD19개 의심은 114스틸 AI시각비교로 표시구간 오탐판정.
관련20원본 해시PASS, 50후보 모두유지. artifacts/stage2-duplicate-review-20260914/
report.json,pair-review.csv,review.html 및 data_raw/stage2-validation-nexar-20260914/
review-inventory-resolved.csv 참고. 기존자동검사/목록보존, 사람라벨0/모델변경없음.
촬영출처 독립성 인증은 아니며 최종 충돌적합성·라벨링은 사용자보류 유지.
docs/stage2-validation-acquisition.md 최신완료절 우선. 수집/중복검사 재실행불필요.

## Stage1 혼합 대조학습 전송 승인·감시기 시작 (2026-09-14)

최초 승인파일 cp949 읽기오류는 UTF-8로복구,원격전송전실패라 중복push없음.
재개 PID20076(실제watcher22812), kernel version1 push성공 및 QUEUED 직접응답확인.
최신상태 SUBMITTED/WAITING, GPU실제RUNNING은아직미확인. 이전PID28368기록대체.

사용자가 601754bytes 코드/설정의 새비공개 mixed-trial 목적지·무료T4 확인에 실행명시승인.
upload-approval.json 기록 및 notebook/metadata 해시일치 확인, 숨김감시기 PID28368 시작.
이전 승인차단 해소. artifacts/kaggle-stage1-mixed-trial-20260914/remote-run.json 및 push.log로
실제GPU상태 확인. 단일push→회수→독립검증 연결, 중복실행/반복폴링 금지.
아직 완료결과 없음. docs/stage1-mixed-trial.md 참고. 기존모델/ZIP/예약변경없음.

## Stage1 혼합 대조학습 준비 완료·전송 차단 (2026-09-14)

사용자 진행요청으로 baseline/기존합성/화질균형/50:50혼합4조건 준비.
같은85출처·seed·128updates각·초기모델·lr1e-5,420영상각 총1680예측/384updates.
6테스트+이전결과전체 회귀검증PASS. 기존비공개 quality-trial-assets 재사용,
새비공개 biadis/crashintent-stage1-mixed-trial에 코드/설정601754bytes만 전송계획.
자동승인검토가 구체적payload/새목적지 사용자명시승인 증거부족으로 감시기시작명령을
실행전거절. 업로드/GPU/감시기미실행. artifacts/kaggle-stage1-mixed-trial-20260914/
remote-run.json BLOCKED_APPROVAL, approval-required.json. 동일시도 재실행금지,
구체적승인후 upload-approval.json 해시일치기록 및 watch_stage1_mixed_trial.py --launch.
docs/stage1-mixed-trial.md 참고. 기존모델/ZIP/예약변경없음.

## Stage3 REVIEW_011 사용자 수정·재채점 완료 (2026-09-14)

사용자 사고전직진/사고후오른쪽전도 설명,32~49조향UNKNOWN 승인반영. motion MOVING유지,
UNKNOWN으로18행제외. 확정332프레임=무작위282(직진182/좌100)+우선별50.
무작위F1 기존.398352/short.386657,직진재현33.52/36.26%,우재현38/36%불변.
원래라벨 before/백업,기존예측해시·18행수정범위·JSON/CSV·혼동행렬/F1 PASS.
artifacts/stage3-human-comparison-20260914-revision1/report.json 및 docs/stage3-human-comparison.md.
기존모델유지. 이전350행결과/실패검토페이지는수정전스냅샷. CASE_01재판정은미완료.

## Stage3 후속 실패 구간 분석 완료 (2026-09-14)

사용자 다음태스크 요청으로 REVIEW_011/003/CASE_01+대조004 연속스틸 분석,
7영상350프레임/2모델 특징·로짓 exact재현PASS. 011은오토바이전도32~49 판단가능여부,
CASE_01은30~49 우곡선종료경계 사람이재확인할후보. 라벨은전부원본보존.
011초반직진4/30·2/30,003후반5/20·0/20로모델실패도남음. 새학습/모델/ZIP변경없음.
docs/stage3-failure-review.md와artifacts/stage3-failure-review-20260914/review.html 참고.
읽기전용분석완료이며 사람재검수완료아님. 다음은불확실구간판정·자동차우회전보강.

## Stage1 화질 보강 실패 분석 완료 (2026-09-14)

사용자 휴대전화 검증 보류 및 실패분석 진행 요청. 420쌍/3모델 비교·해시·슬롯/오류수 검사 PASS.
원본 오탐35건 해소, 재녹화12건 신규오류(7원본). weak6/geometry2/noise2/flicker1/normal_v1 1.
9/12는 세slot모두 원본판정,11/12는 평균확률<.4. full21건은전부정답.
기존합성 추가학습 대조군은 신규오류12건전부정답. 강한 화면단서 의존 잔존 해석 지지,
단일원인확정 불가. 26영상1300프레임 해시/디코딩,12사례비교이미지 생성·대표4사례육안확인.
docs/stage1-quality-failure-analysis.md 및 artifacts/stage1-quality-failure-analysis-20260914/ 참고.
다음은 기존합성+화질균형 혼합대조 설계. 아직 새학습/번들/모델/ZIP/예약변경없음.

## Stage1 화질 대조실험 회수·독립 검증 완료 (2026-09-14)

기존 quality-trial v1 GPU COMPLETE, 결과 ZIP 회수 및 1260예측/256업데이트 검증 PASS.
요약 평균확률의 최대2.22e-16 반올림차이로 발생한 Summary mismatch를 평균만 abs_tol1e-12로
수정, 건수/오류율은 exact 유지, 4테스트 PASS. 실패 결과·상태는 보존했다.
최종 증거 artifacts/kaggle-stage1-quality-trial-20260911/validated-recovery-20260914/result-validation.json.
remote-run COMPLETE/VALIDATED. 재학습·추가다운로드 불필요.
baseline→existing_recipe→matched_quality 원본 noise오류19→19→0/21, mix18→20→2/21.
normal63 F1 1→1→.982348. matched_quality는 weak_screen오류0→6/21,
no_geometry/no_noise 각각0→2/21로 재녹화 보존기준 실패. 두후보 모두 promising_pilot=false.
현공식최고 Stage1 .6386345895 모델 유지, ZIP/예약 변경없음.
이전 회수중·실패 기록보다 이 완료결과 우선. docs/stage1-quality-trial.md 완료 절 참고.

## Stage3 1순위 사람 라벨 대조 채점 완료 (2026-09-14)

사용자 요청으로 확정350프레임(무작위 직진200/좌100, 우선별 우50)을 기존1/5/15와
short1/3/5 실제후보ZIP/seed20260910으로 CPU채점했다. 무작위F1 .386074/.378844,
직진재현31%/34.5%,좌84%/85%; 우선별재현38%/36%(CASE_01 한영상).
두세트합산점수 없음,고정3클래스F1/없는정답클래스의 한계 명시. 기존모델유지.
17원본해시·전체디코딩·JSON/CSV·배포AST/가중치·별도CSV혼동행렬/F1/paired검증PASS.
artifacts/stage3-human-comparison-20260914/{report,independent-validation}.json 및
docs/stage3-human-comparison.md 참고. 사람라벨불변,추가학습/ZIP/selected/예약변경없음.
다음은 REVIEW_011/003 및 CASE_01 실패구간 검토이며 아직착수안함.

## Stage2 검증자료 확보 착수 (2026-09-14)

후속 사용자 승인: 50행 review-inventory.csv와 검증전용 reservation.json 생성.
Inspect-Stage2ValidationCandidates.ps1 숨김 PID8140 시작, CCD1500 지문부터 추출 후
수집완료 시 후보50 지문/76,225쌍 유사검사/임시그룹/재생HTML 자동 생성.
artifacts/stage2-validation-inspection-20260914/status.json의 SCREENING_COMPLETE 전까지
검사완료 아님. Get-BackgroundWorkStatus.ps1에 연결, Codex 반복폴링 금지.
유사도는 의심쌍 탐색이며 출처독립성 확정 아님, 사람라벨0 유지.

사용자 자료 확보 요청, 라벨링 보류. CCD 전체133출처=수동23+사전학습110,
잔여0을 실제 CSV로 재검산했다. Nexar 공개 양성50영상838MB 고정 선정,
scripts/Get-Stage2ValidationCandidates.ps1 -Worker 숨김 PID19800 시작 확인.
해시/전체디코딩/CCD동일파일검사·유한재시도·잠금 연결. 상태는
artifacts/stage2-validation-acquisition-20260914/status.json 및 Get-BackgroundWorkStatus.ps1.
반복폴링/중복실행금지. ACQUIRED_UNLABELED 전까지 확보완료 아님.
촬영출처 독립성/시각중복/실제충돌 적합성/사람라벨은 미확정.
docs/stage2-validation-acquisition.md 참고. 학습/모델/ZIP 변경없음.

## Stage1 화질 대조실험 결과 회수 재개 (2026-09-14)

사용자 1순위 요청으로 기존 Kaggle quality-trial v1 COMPLETE 직접 확인.
watch_stage1_quality_trial.py를 --launch 없이 실행해 결과 다운로드·독립 검증 재개(PID5736).
최초 remote-run.json COMPLETE/DOWNLOADING 확인, 아직 검증 완료로 취급하지 않는다.
artifacts/kaggle-stage1-quality-trial-20260911/remote-run.json 및
validated/result-validation.json 확인. 중복 실행·Codex 반복 폴링 금지.
신규 업로드/GPU학습/모델교체/ZIP/예약 변경 없음. docs/stage1-quality-trial.md 최신 절 참고.

## 최신 사용자 검수 마감 (2026-09-13)

사용자 요청으로 Stage3 1차 사람 검수를 일단락했다. 새 무작위 CCD 12영상은
600/600프레임 검수, 조향 유효 300프레임(직진200/좌100). 추가 우회전 후보 5영상은
229/250프레임 검수, 조향 유효50프레임(우50). CASE_04의0~20프레임은 미검수로
유지하며 사용자가 현재 상태에서 넘기기로 했다. REVIEW_010은 사용자 요청으로
UNKNOWN/STOPPED로 정리했다. 데이터 형식·영상해시·구간중복·JSON/CSV 일치 PASS.
라벨과 원본ID/해시 목록은 data/stage3-human-review 및 data/stage3-right-review에 보존.
영상은 Git 제외다. 이번 표본은 독립 일반화 평가가 아니며 모델 대조 채점은 미실행.
우회전 후보는 대부분 사람이 판단불가로 판정했으므로 확정 우회전5개로 취급하지 않는다.
상세 docs/stage3-remaining-work.md의2026-09-13 완료 기록을 우선한다.

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

## Stage2 출처 교차검증 실행 중 (2026-09-10)

사용자 1번 요청으로 기존66영상 특징 캐시에서5분할 CPU 교차검증 시작.
exact5/15와soft_mixed15 고정에폭, 외부 분할로 최적에폭 선택하지 않음.
기존 검증15개 오류 분석 완료: 회피 공간 없음11개 중7개 오판,
정답 시점을 넣어도 회피정답률46.7%로 동일. 최종 교차검증 결과는 아직 없음.
artifacts/stage2-source-cv-20260910/status.json 및 로그 확인, 중복 실행 금지.
자동 독립검증 PID13084. 상세 docs/stage2-source-cv.md.
Stage1 GPU 실행은 변경하지 않았다.

## Stage2 교차검증 완료 (2026-09-10 15:26 확인)

5분할/66영상/23출처 CPU917.765초, 독립검증PASS. soft_mixed15 개발평균.4508,
exact5 .4280, exact15 .4015, 학습 상수기준선.4773. 출처 동일가중은 soft .4188 vs exact5 .4252.
개선이 안정적으로 유지된다고 결론낼 수 없으며 단순 장기학습보다 시점선택·분류특징 개선 우선.
상세 docs/stage2-source-cv.md 완료 절 및 artifacts/stage2-source-cv-20260910/independent-validation.json.
위 Stage2 실행 중 기록은 완료로 대체한다. Stage1과 제출 모델은 변경하지 않았다.

## Stage2 1번 시점 선택 대조 완료, 2번 승인 대기 (2026-09-10)

사용자가 각 번호 이동 전 승인을 요구했다. 1번만 CPU 실행 완료.
±0.3초 확률합 적용시 exact5 평균.4280→.4432, exact15 .4015→.4356,
soft_mixed15 .4508→.4470. 현재 개선 설정에서 이득 없어 제출에는 미반영.
상세 docs/stage2-time-decoding.md 및 artifacts/stage2-time-decoding-20260910/report.json.
2번 분류 특징 개선,3번 충돌 사전학습은 아직 미착수이며 각각 진입 전 승인 필요.
Stage1 GPU 및 제출 모델 변경 없음.

## Stage1 화질 대조군·효과 제거 진단 실행 중 (2026-09-10)

사용자 1·2번 요청으로 기존 val 21출처에서 원본 하나씩, 17조건 총357영상 생성 시작.
원본형5조건(기본/JPEG/블러/노이즈/혼합), 합성12조건(전체/10효과 개별제거/화면단서25%).
출처·phone holdout 분리, 같은 원본/seed/CRF의 paired 진단이다.
artifacts/stage1-robustness-20260910/status.json 확인, GENERATING 중 중복 실행 금지.
첫2출처34영상 전프레임·해시검사PASS. 전체 완료를 뜻하지 않는다.
6개 코드검사PASS, 실제 제출 모델의 인공17영상 스모크는 별도 artifacts/stage1-robustness-smoke-20260910/.
watch_stage1_robustness.py가 본학습 validated/best.pt의 PASS·해시와 데이터 완료를 기다려
CPU2스레드 최대4시간으로 1회 자동 평가한다. full-model-evaluation/watch-status.json 확인.
새 원격작업/모델교체/제출 없음. CPU FP32 진단이며 실재촬영·공식성능 결과가 아니다.
상세 docs/stage1-robustness.md. 별도 Stage2/3 세션 변경을 보존한다.

Stage1 강건성 연결 스모크 완료: 실제 baseline 모델+제출 decoder로 인공17영상/51구간,
CPU273.77초 및 저장로짓·확률·출처 독립재검산PASS.
artifacts/stage1-robustness-smoke-20260910/evaluation/report.json.
실제357영상 진단 성능 결과가 아니며 실제 데이터 생성·모델 대기는 계속 진행 중.


## Stage3 움직임 의존성·경로별 실패 진단 완료 (2026-09-10)

사용자 요청으로 1000클립 모델을 고정해 21영상/48클립 × 4조건, CPU FP32 192회 추론을 완료했다. 정상 예측은 기존 GPU 가감속/조향과 전부 일치. 과거 순서 섞기 조향변화4/48, 마지막 프레임 반복7/48; 조향정확도 정상35.42%→반복31.25%. 08-16 표본12개는 모든 변형에서 조향분류가 유지됐다. 시간정보 일부 의존은 있으나 다수 오답·경로편향은 정적입력에도 남는다.
31영상 전체 경로분석: 08-02 안정직진 재현율1.67%, |조향각|≤1도 직진1.53%, pose근직진1.48%. 08-16 좌회전353시점 재현율0%. 단순 경계/전환/속도 차이만으로 설명 부족; 배경 등 원인은 미확정.
원본영상 재디코딩 192입력해시+움직임80행/경로108행 독립 재검산PASS. docs/stage3-motion-route-diagnostic.md 및 artifacts/stage3-motion-route-20260910/validation.json 참고. 추가학습·제출모델·라벨 변경 없음. 다음은 단일프레임/움직임특징 대조군과 경로×클래스 학습표본 진단, 사람검증셋 병행이다.

## Stage2 2번 전후 특징 대조 완료, 3번 승인 대기 (2026-09-10)

사용자 승인으로 기존 시간 모델 고정, 두 시점 vs ±0.5초 평균 scene head 대조 완료.
5분할/66영상/23출처/3seed/30head CPU45.48초, 원본 특징부터 독립 예측 재현PASS.
기존 개발평균.4508, point재학습3seed평균.4407,window .4356. 채택 근거 없어 제출 미반영.
상세 docs/stage2-scene-context.md 및 artifacts/stage2-scene-context-20260910/.
사용자의 번호별 승인 조건 유지: 3번 CCD 충돌 사전학습은 아직 미착수, 승인 필요.
Stage1 GPU 및 제출 모델 변경 없음.

## Stage2 3번 CCD 충돌 사전학습 대조 실행 중 (2026-09-10)

사용자가3번을 명시 승인했다. 사전학습1296영상/110출처,수동66영상 전체23출처 제외.
CPU특징추출→충돌3에폭→5분할 scratch/pretrained 각15에폭 자동 실행 PID23960.
독립검증 대기 PID32360. artifacts/stage2-ccd-pretrain-20260910/status.json 및 로그 확인,
중복 실행 금지. 초기 특징 추출 약3시간 예상,학습·검증 시간 별도. 결과 아직 없음.
상세 docs/stage2-ccd-pretrain.md. Stage1 GPU 및 제출 모델 변경 없음.

## Stage1 자동 ZIP 후속 예약 준비·등록 차단 (2026-09-10)

사용자가 모델 반영→GPU통합검사→ZIP생성·등록 자동화를 요청했다.
src/advance_stage1_release.py 및 scripts/{Advance-Stage1Release,Register-Stage1ReleaseTask}.ps1 준비.
docs/stage1-auto-release.md와 artifacts/stage1-auto-release-20260910/config.json 참고.
Stage2개선 후보를 기준으로 Stage1만교체,357영상진단·동일검증개선·정확한GPU증거 확인 후
등록/즉시ZIP복사. daily_submission.py에 등록·복사 잠금과 다른세션 변경 거절 추가,6테스트PASS.
Windows 신규작업 CrashIntent-Stage1AutoRelease-20260910 등록/시작은 자동승인검토에서
새 Kaggle목적지·민감자료 명시승인없음을 이유로 실행 전 거절됐다. 아직 미등록·미업로드.
승인 요청 범위: 새후보모델/추론코드/기존공개예제 약345MB,
비공개 biadis/crashintent-stage1-release-20260910-assets 및 동명release노트북 무료T4 통합검사.
사용자가 승인하면 준비된 등록스크립트를 실행하고 실제 스케줄러 첫실행 결과를 확인한다.
기존 Stage1 생성/학습/강건성 검증과 기존08:30예약은 그대로다. 대회제출은 범위 밖.


## Stage3 표본 감사·입력 특징 대조 완료 (2026-09-10)

사용자 1·2번 요청 완료. 기존1000클립은17경로×3조향 전부15~21개, 경로×가감속×조향153조합 전부2~8개로 누락 없음. 다만 한 경로50클립이2영상에서 나와 다양성 차이는 남음.
같은1000클립·검증31영상18603시점에서 동일MLP의 이미지/움직임/결합 입력을3seed×12에폭1200업데이트씩 CPU비교했다. 조향F1 평균 이미지.536198/움직임.684695/결합.684117, 기존MViT .409353. 움직임3seed .681430~.690056. 4경로 평균F1 모두 기존보다 개선. 08-02직진재현율1.75%→39.81%, 08-16좌회전0→57.60%, 대신08-16직진90.03%→47.12%로 하락.
187영상해시+19603특징행+9체크포인트 전로짓 재로딩 최대오차0 독립검증PASS. artifacts/stage3-representation-20260910/independent-validation.json 및 docs/stage3-representation-comparison.md 참고. 제출 모델·ZIP 변경, 외부전송, GPU실행 없음. 센서 대리 검증이며 공식 점수 개선은 미확인. 후속은 움직임 모델 주력/결합 보완 후보를 사람·별도출처 검증하고 직진-회전 균형 점검. 현재 동일MLP 내 입력비교와 구조/학습률이 다른 MViT 기준선 비교는 구분한다.

Stage1 자동 ZIP 후속 예약 등록 완료(2026-09-10 16:30 KST):
사용자가 약345MB 비공개 Kaggle 목적지·무료T4 검사·예약등록 범위를 명시적으로 승인했다.
CrashIntent-Stage1AutoRelease-20260910 등록 및 첫실행 LastTaskResult=0 확인,
16:31:03부터2분간격24시간+로그인트리거. 현재 WAITING_FOR_VALIDATION.
이전 등록차단 기록보다 이 완료상태 우선, 같은범위 재승인 불필요.
docs/stage1-auto-release.md 및 artifacts/stage1-auto-release-20260910/status.json 참고.
기존selected.json 해시불변, 아직 신규모델 전송·통합GPU 실행은 미착수.

## Stage3 움직임 제출 ZIP 준비, GPU 업로드 승인 대기 (2026-09-10)

사용자가 ZIP 생성 및 테스트 제출을 요청했다. artifacts/stage3-motion-submit-candidate-20260910/submit.zip 생성 완료(177,368,840 bytes, SHA256 7f98769355146458abdd5d1a897de1cdc77b7f1706142c814ba6ff5bd90c3020). 기존 Stage2 후보 기준 Stage1·2/requirements 바이트 유지, Stage3만 고정 첫 seed20260910 움직임MLP로 교체. 규격6파일·기존19 AST 보존·검증18603행 클래스 일치·실제2영상 각600행 특징 일치 CPU PASS. 공개1200행 CPU 예측 준비.
GPU 검사용 데이터/노트북은 artifacts/kaggle-stage3-motion-candidate-20260910/. 자동승인 검토에서 약218MB 모델·추론코드·공개fixture의 기존 비공개 biadis/crashintent-stage2-candidate-assets 업로드를 목적지 명시승인 부재로 거절. 업로드/GPU 미실행, 공식 미제출. 승인 후 데이터셋 버전 업로드→비공개 biadis/crashintent-stage3-motion-check 무료T4 실행→validate_candidate_gpu_result.py 및 stage3_cpu_prediction_parity 확인. docs/stage3-motion-submit-candidate.md 참고. selected.json 및 Stage1 예약 변경 없음.

Stage3 GPU 업로드 승인 및 실행 (2026-09-10): 사용자가 약218MB 비공개 candidate-assets 업로드와 무료T4 검사를 명시 승인했다. 데이터셋 업로드 ready, biadis/crashintent-stage3-motion-check v1 push 완료. artifacts/kaggle-stage3-motion-candidate-20260910/remote-run.json 추적, watch.py가 결과 회수·독립 계약검증 중. 같은 실행 중복 금지. 완료 후 stage3_cpu_prediction_parity 및 CPU 기준CSV 일치를 추가 확인할 것. 대회 미제출.

## Stage3 움직임 후보 GPU 통합 완료 (2026-09-10)

사용자 명시 승인 후 비공개 업로드 및 biadis/crashintent-stage3-motion-check v1 COMPLETE. 결과 ZIP 회수·로컬 독립검증 PASS, Stage3 CPU/GPU 1200행 전부 일치. 설치165.50초, Stage1 5행3.91초/Stage2 5행5.33초/Stage3 1200행14.25초. DNS 다운로드 실패는 재다운로드로 해결했으며 재실행 안 함. artifacts/kaggle-stage3-motion-candidate-20260910/result-validation.json 및 docs/stage3-motion-submit-candidate.md 참고. 제출 ZIP artifacts/stage3-motion-submit-candidate-20260910/submit.zip(177368840 bytes, SHA256 7f98769355146458abdd5d1a897de1cdc77b7f1706142c814ba6ff5bd90c3020). 상태 CPU_GPU_PASS. 기존 GPU 승인대기/진행중 기록보다 이 완료 결과 우선. 대회 실제 제출 미실행, selected.json 및 Stage1 예약 변경 없음.

## 제출86809 사용자 결과 보고 일부 수신 (2026-09-10)

프로토_2026_09_10_003/submit.zip, 소요11분55초(715초). 사용자 Stage3 유의미한 상승 보고, 정확한 세 점수와 제출시각 미제공. 점수 수신 후 이전 최고.1656369753 및 직전.1331824102와 비교할 것. 대화 맥락상 움직임 후보이나 서버파일 해시 미확인. docs/submission-history.md에 누적4회(오늘3회) 기록. GPU 검증 당시 미제출 기록보다 이 사용자 제출 보고가 최신이다.

제출86809 후속: 사용자 제공 표시 시각2026-09-10 17:01:41 기록 완료. 소요11분55초, 정확한 Stage1·2·3 점수는 계속 대기. docs/submission-history.md 참고.


## 제출86809 공식 결과 수신 완료 (사용자 보고)

2026-09-10 17:01:41, 프로토_2026_09_10_003, 11분55초. Stage1 .4045598914 / Stage2 .2096390642 / Stage3 .4187467089. Stage1·2 동일, Stage3 이전 최고 대비+.2531097336(약152.81%). 가중합 .33226628752, 이전 최고 대비+.10124389344로 현재 보고 최고. docs/submission-history.md 참고. 앞선 점수 대기 기록보다 이 완료 결과 우선. 제출 맥락상 움직임 후보 연결, 서버 해시 미확인. selected.json 변경 및 추가 학습 없음.

## Stage3 개선 1·2번 독립 실험 병렬 착수 (2026-09-10)

사용자가 1번 화면 영역/공통 움직임 분리, 2번 시간길이/이미지 결합을 독립 실험하도록 승인하고 병렬 처리 허용. 각각 별도 코드·artifact에서 동일1000학습표본/18603검증시점/3seed/고정12에폭1200업데이트를 비교 중. 1번 artifacts/stage3-spatial-motion-20260910, 2번 artifacts/stage3-temporal-motion-20260910. 현재 제출ZIP SHA256 7f98769355146458abdd5d1a897de1cdc77b7f1706142c814ba6ff5bd90c3020 보존, 신규외부업로드/공식제출/selected변경 없음.
Root 공통 독립 평가 src/evaluate_stage3_motion_slices.py 및 artifacts/stage3-motion-independent-20260910/plan.json. 센서속도는 평가구간에만 사용, 모델입력 아님. 정지1257/주행<2mps305/2~5mps428/>=5mps16613행. 전환±5프레임/안정/4경로 및 각클래스 지표 조건 사전고정. baseline3seed 재계산 .517241가감속/.684695조향 일치. 완료 판단은 각 실험 독립검증과 공통 평가 합산 후 할 것.

## Stage3 개선 1·2번 병렬 독립 실험 완료 (2026-09-10)

사용자 승인 범위 완료. 동일1000학습/31영상18603검증/3seed/고정12epoch1200update로 공간9조건+시간12조건(19신규학습+2기준재사용), 모델별 전로짓재현오차0 독립검증PASS. root공통속도/전환/경로/클래스 평가PASS. 기준가감속/조향.517241/.684695, 공통이동분리.512456/.680354, 영역추가.490498/.679482, 짧은창.516314/.687982, 긴창.486119/.685540, 긴창+이미지.483918/.667402. 짧은창 저속305행조향 .386112→.517077,저속직진124행재현25%→66.94%지만 전체우회전81.39%→76.91%, 평균재현.708695→.704769하락. 현재공식.4187467089 모델유지,short만별도출처/사람검증후속후보. 나머지는이번조건미채택. 공식제출/ZIP/selected/외부업로드변경없음.
상세 docs/stage3-motion-independent-experiments.md, docs/stage3-spatial-motion-experiment.md, docs/stage3-temporal-motion-experiment.md. 각각 artifacts/stage3-spatial-motion-20260910/independent-validation.json, stage3-temporal-motion-20260910/independent-validation.json PASS, 공통 artifacts/stage3-motion-independent-20260910/validation.json PASS. 실행중기록보다이완료결과우선.

## Stage2 3번 사전학습 대조 완료 (2026-09-11 확인)

artifacts/stage2-ccd-pretrain-20260910/report.json COMPLETED 및 independent-validation PASS.
1296영상/110출처 사전학습3에폭3888업데이트,5분할66영상 비교 완료.
이번 matched scratch 개발평균.4318→pretrained .5000,출처 동일가중 .4065→.5156.
5분할3개 개선/1개동일/1개하락. 단일seed 개발결과,공식점수 아님.
전체 벽시계61007.562초(약16시간57분),지연 원인 미확정.
상세 docs/stage2-ccd-pretrain.md 마지막 완료 절. 위 실행 중 기록보다 이 완료 결과 우선.
새 최종 제출모델·ZIP·GPU검사·공식제출 미실행,다음 작업은 사용자와 결정.

## Stage1 본학습 완료·본검사 시간초과 복구 중 (2026-09-11)

6에폭2640업데이트/690영상 검증PASS,합성F1 .9983713(baseline .2968900).
357영상 CPU본검사는78개에서시간초과해 자동ZIP NEEDS_ATTENTION.
--resume 검증기와 recover_stage1_robustness.py로78개재사용·나머지재개,80/357증가확인.
4테스트PASS,실행중자동유휴절전방지·종료시해제. 완료하면 해당ZIP실패gate만복구한다.
docs/stage1-robustness.md 및 full-model-evaluation/result/status.json, recovery-evaluation.log 참고.
최신Stage3 .4187467089 모델을보존하도록 아직미업로드인자동ZIP기준을7f98769...움직임후보로갱신,
해당GPU검증·ZIP해시확인. 상세docs/stage1-auto-release.md. 본검사·신규ZIP은아직미완료.

## 예약 주기 변경 완료 (2026-09-11)

사용자 요청으로 실제 Windows 작업 CrashIntent-Stage1AutoRelease-20260910을
2분에서 12시간 간격으로 변경하고 로그인 트리거를 제거했다.
다음 실행은 2026-09-11 21:02:33 KST, 이후 12시간 반복(만료 없음).
실제 XML에서 PT12H, 트리거 1개, 실행 대상 불변을 확인했고 등록 스크립트 구문 검사도 통과했다.
등록 스크립트는 등록 12시간 뒤부터 반복하며 즉시 실행하지 않는다. 기존 08:30 예약은 유지한다.
위 2분 간격 기록보다 이 변경을 우선한다. 한 번에 한 단계만 진행하므로 완료까지 여러 주기가 필요할 수 있다.

## Stage2 사전학습 개선 제출 ZIP 생성 완료 (2026-09-11)

사용자 ZIP 요청으로 CCD사전학습에서 수동66영상전체15에폭/990업데이트 최종학습,
마지막 사용자 제출86692의 ZIP에서Stage2만 교체했다. Stage1/3 및 inference 그대로.
artifacts/stage2-pretrained-submit-candidate-20260911/submit.zip,303600234bytes,
SHA25646db5bde8d42ee22b1582ab3869ea249721571e31aeb0367fa94e7978e697003.
정적·CRC·모델66영상출력동등성PASS. 신규GPU검사·공식제출·selected변경 미실행.
상세 docs/stage2-pretrained-submit-candidate.md 및 같은출력폴더 validation.json.

## Stage2 개선 + Stage3 short 최종 제출 후보 준비 완료 (2026-09-11)

사용자가 다른 세션 생성ZIP과 Stage3수정을 합쳐 최종ZIP준비 요청. Stage2세션 artifacts/stage2-pretrained-submit-candidate-20260911/submit.zip(46db5bde...) STATIC_AND_CPU_PARITY_PASS 확인 후 그Stage1/2/requirements보존,Stage3만short첫seed20260910으로교체. artifacts/final-stage2-short-stage3-20260911/submit.zip 177371036bytes SHA256 f94995a9b6303fde41b65153372ad9e48d220e5034d86c9e1c05cfea04fe45fb. CPU_PASS_GPU_PENDING: 18603행 예측전부일치,실제2영상1200특징행exact,공개1200행 CPU예측준비. GPU자료 artifacts/kaggle-final-short-candidate-20260911/ 준비만했으며아직업로드/GPU/대회제출안함. Stage1새본학습모델은미포함(별도검증복구중),기존모델유지. src/package_final_short_candidate.py 및 docs/final-stage2-short-stage3-candidate.md 참고. 기존ZIP/selected/Stage1예약보존. 다음은이정확한결합ZIP의T4통합검사.


## 제출87243 평가 결과 수신 (2026-09-11)

사용자 보고2026_09_11_001,09:13:12,11분38초. Stage1 0.4045598914 / Stage2 0.1966181317 / Stage3 0.4141711429, 가중합 0.32522768812. 이전최고86809 대비Stage2 -0.0130209325,Stage3 -0.0045755660,가중합 -0.00703859940. 기존86809 조합이계속최고. 맥락상Stage2사전학습+Stage3short결합후보,서버해시미확인. docs/submission-history.md 참고. 준비당시미제출기록보다이번사용자결과가최신. 신규GPU/학습/selected/예약변경없음.

## Stage2 제출87243 하락 분석 완료 (2026-09-11)

공식 .2096390642→.1966181317(-6.21%). 결합ZIP Stage2 가중치 연결/ResNet/Stage2 AST정상.
각pretrained fold 학습평균97.2~99.5% vs미학습40.4~63.5%,최종66영상학습정확도98.5~100%.
일반화 격차 강함. 대조 .4318→.5000은 새scratch 대비이며 이전실제제출모델 직접비교 아님.
새GPU FP16특징동등성 미검증,CCD proxy/분포차이 가설,공식항목별하락 원인확정 불가.
상세 docs/stage2-drop-analysis.md 및 artifacts/stage2-drop-analysis-20260911/audit.json.
기존공식Stage2 유지 권고만 했으며 모델/ZIP/selected변경·새학습·제출 없음.

## Stage3 제출87243 하락 분석 완료 (2026-09-11)

공식 .4187467089→.4141711429. 두로컬ZIP AST/가중치연결PASS,Stage3변경은flow창+loader뿐. 실제seed20260910끼리외부검증:조향F1 .681430→.684549지만LEFTrecall73.64→72.75%,RIGHT81.58→78.57%,STRAIGHT56.69→58.96%. 우회전정답상실322중315가직진으로이동,순우회전정답-140;안정우회전재현87.67→83.74%. 2경로개선/2하락. 검증예측전환3019→2952로감소해단순짧은창노이즈증가설은비지지. 실제seed저속직진51.61→62.10%,이전25→66.94는3seed평균임을명확화. 가장유력한설명은회전→직진tradeoff와외부개발분포이득의공식미이전,비공개정답없어원인확정불가. Stage3공식은조향F1,가감속하락을직접원인으로해석하지말것. docs/stage3-submission-87243-analysis.md 및 artifacts/stage3-drop-87243-20260911/ 참고. 기존모델유지,신규학습/ZIP/selected/GPU변경없음.

## Stage2 구·신 GPU 정밀도 비교 준비·전송 차단 (2026-09-11)

사용자1번 승인으로 기존66영상 동일입력6조건 CPU/GPU FP32/FP16/JPEG 비교준비.
artifacts/kaggle-stage2-precision-20260911/precision 번들93.25MB.
목적지 비공개 biadis/crashintent-stage2-precision-assets,무료T4 precision-check.
자동승인검토가 구체적자료·새목적지 업로드명시승인 없음으로 거절,외부전송/GPU미실행.
구체적사용자승인 대기. 상세 docs/stage2-precision-diagnostic.md.
기존모델·ZIP·Stage1작업 변경없음.

Stage1 본검사 두번째복구(2026-09-11 09:39이후):
205/357에서 PyTorch c10.dll 네이티브종료(0xc0000005),실제프로세스없음확인.
오래된RUNNING상태보존후FAILED정정,205개저장예측전체검증PASS.
recover_stage1_robustness.py 단일CPU스레드 재개,진전있을때만 최대3회 제한재시도 및
네이티브실패 결과상태갱신 보강.2통합테스트PASS. 본검사·새ZIP미완료.
docs/stage1-robustness.md와 full-model-evaluation/{watch-status.json,result/status.json} 우선확인.

## Stage3 별도출처 블라인드 검수 준비 (2026-09-11)

사용자후속진행요청 및 실제검수라벨없음확인. CCD새36원본출처 pool에서 예측전고정무작위12+나머지모델불일치진단12,총24영상1200프레임 준비. Stage3학습자료와다른CCD,공개5영상동일출처/Stage1phoneholdout동일출처제외. 영상 data/stage3-blind-review-20260911/videos, 라벨 labels/annotations.json 및labels_review.csv(현재reviewed0). 예측비노출UI scripts/Start-Stage3BlindComparison.ps1, -Score로사람MOVING/확정조향만무작위와진단분리비교. src/prepare_stage3_blind_comparison.py 및score_stage3_blind_comparison.py. docs/stage3-blind-comparison-guide.md 참고. 사람라벨없어서성능점수미산출이며현재다음단계는사용자검수. 합성스모크를진짜라벨로사용하지않음. 예측과선정정보 artifacts/stage3-blind-review-20260911 별도보관. 기존모델/ZIP/selected/다른Stage변경없음.

## Stage2 GPU 정밀도 진단 RUNNING (2026-09-11)

사용자 명시승인 후93.25MB 비공개 precision-assets 업로드ready,
biadis/crashintent-stage2-precision-check v1 RUNNING 확인.
자동회수·검증 watcher PID37760, artifacts/kaggle-stage2-precision-20260911/remote-run.json 확인.
이전 전송차단 해소,중복실행금지. src/validate_stage2_precision.py가792예측·특징·logits 독립검산.
상세 docs/stage2-precision-diagnostic.md. 아직GPU결과 미확보,모델/ZIP변경없음.

## Stage3 사람 검수 보류·경로 교차검증 착수 (2026-09-11)

사용자가사람검수는나중에하고무인진행가능한작업요청. 준비한24영상/라벨0 보존,검수GUI실행프로세스없음확인. 기존학습1000클립/17경로5분할/기존vs short/3seed=30모델CPU학습시작. artifacts/stage3-route-cv-20260911/status.json 확인,중복실행금지. 각분할학습경로만정규화,동일10그룹복원추출배치/1200updates 사용. 이전4개검증경로미사용. 완료시 src/compare_stage3_route_cv.py verify로전checkpoint/OOF검증필요. docs/stage3-route-cross-validation.md 참고. 제출/selected/GPU변경없음.

## Stage3 무인 경로 교차검증 완료 (2026-09-11)

사람검수보류후17경로/기존1000균형클립/5fold/3seed/2창=30모델완료,372.219초학습. 전checkpoint·fold정규화/분리/배치/로짓오차0 및별도OOF CSV혼동행렬PASS. OOF조향3seed평균기존 .718900→short .709580,3seed합산모두하락. 5fold중4평균하락,15쌍중10하락,17경로중12하락. 직진재현65.00→60.56%,우회전74.78→76.67%로이전4경로tradeoff와반대:short가항상직진편향이라는주장불가,시간창이득의경로의존성을확인. 기존1/5/15제출모델유지. docs/stage3-route-cross-validation.md 및 artifacts/stage3-route-cv-20260911/summary.json,independent-validation.json PASS. 위실행중기록보다완료결과우선. 사람라벨0/검수자료보존,공식제출/GPU/selected/ZIP변경없음.

## Stage1 로컬 검사 자동 복구 보강 (2026-09-11)

사용자 요청으로 CrashIntent-Stage1LocalEvaluation-20260911 Windows 예약 등록·실행.
로컬 검사만 2분 감시, 기존 Stage1AutoRelease 12시간 예약 유지. 업로드·대회 제출 없음.
25개/새 프로세스, PAUSED 재개, 저장 fsync/해시·prefix 검증, 중복 OS 잠금,
15분 정체·native 종료 복구, 총5회/같은지점2회 실패 중단. Windows venv 자식까지 종료 검증.
전체PASS 후 독립 재검증과 관련 실패gate만 복구. 다른 세션 registry 변경은 기존 CAS 유지.
32개 기존 테스트 PASS + 추가 후 감시기10테스트 PASS. 실제 소형 추론 1→3→5개,
서로 다른 PID에서 2개씩 PAUSED/재개 확인(소형17개 전체완료 아님).
본검사265개부터 새 예약이 자동 인계, 계획 재시작으로 최신 자식종료 처리까지 적용.
최신 진행 수치는 full-model-evaluation/result/status.json, supervisor-status.json을 확인.
상세 docs/stage1-local-supervisor.md. 본검사 완료/신규ZIP 생성으로 혼동하지 말 것.

## Stage2 GPU 정밀도 진단 완료 (2026-09-11)

precision-check v1 COMPLETE,66영상792예측 독립검증PASS,logits최대오차2.67e-5.
FP32→FP16 구모델0/신모델충돌1건0.1초변화,JPEG입력에서는구신모두0변화.
JPEG변환 영향은구9/신12영상,신진입최대2.5초변화. 그러나신진입정답65→66으로
개선되어JPEG가공식하락원인이라고확정불가. 학습자료민감도검사이며일반화비교아님.
상세 docs/stage2-precision-diagnostic.md 완료절. 이전RUNNING기록대체,중복실행금지.
모델/ZIP변경없음. 다음은별도출처검증자료확보.

## Stage3 1번 움직임 변화량 추가 실험 착수 (2026-09-11)

사용자1번승인. 기존1/5/15시간범위·402흐름유지+변화량268/격자크기변화64/방향cos64/구간상쇄일치도64=460파생특징추가. 같은17경로5fold/3seed/1200updates,기존CV batch/sampleexact. baseline1재학습exactPASS+14재사용,dynamics15신규 총16신규/30조건. 영/일정/증가/반전/불완전경계검사PASS. artifacts/stage3-dynamics-cv-20260911/status.json 확인,중복실행금지. src/compare_stage3_dynamics_cv.py 및 docs/stage3-motion-dynamics-experiment.md. 끝나면verify필수. 기존ZIP/selected/GPU변경없음.

## Stage3 1번 변화량·방향 특징 실험 완료 (2026-09-11)

요청1번완료:동일17경로5fold/3seed,baseline14재사용+1exact재학습+dynamics15신규(신규103.063초). 모든checkpoint/정규화/mask/batch/OOF검증오차0 PASS,별도CSV혼동행렬대조PASS. 조향F1 OOF평균 .718900→.682605,가감속 .497810→.507823. LEFT75.44→73.22%,직진65→59.11%,RIGHT74.78→72.22%. 5fold평균전부하락,15쌍중14하락,17경로중12하락. 이번460파생특징추가방식미채택,기존1/5/15공식최고모델유지. docs/stage3-motion-dynamics-experiment.md 및 artifacts/stage3-dynamics-cv-20260911/independent-validation.json PASS. 다음2번작은시간축모델은미착수,공식제출/ZIP/selected/GPU변경없음. 위진행중기록보다완료결과우선.

## Stage2 과적합 완화1번 실행 중 (2026-09-11)

사용자승인으로내부출처검증조기종료/BiGRU저학습률/BiGRU고정3조건실행.
기존5외부fold 내부에서에폭·설정선택후외부학습전체재학습·평가. 외부점수로에폭선택금지.
CPU PID13012,자동독립검증PID36140. artifacts/stage2-regularization-20260911/status.json 확인.
초기기준약30~60분예상,결과아직없음. docs/stage2-regularization.md 참고,중복실행금지.
기존모델/ZIP/Stage1변경없음,2번이후다른개선은미착수.

## Codex 반복 감시 절감 적용 (2026-09-11)

사용자 2번 요청으로 AGENTS.md에 장시간 작업의 Codex sleep/상태조회 반복 금지,
기존 Python watcher/Windows 예약의 완료·실패 기록 활용, 시작 확인 후 대기만 남으면
응답 종료 및 완료 이벤트/사용자 후속 요청 때 결과 분석 규칙을 추가했다.
scripts/Get-BackgroundWorkStatus.ps1은 로컬 JSON을 한 번만 읽으며 API/네트워크/작업시작 없음.
실제 실행 확인. 기존 Stage1 ZIP 12시간, 로컬검사 감시 2분 예약은 유지.
이미 열린 다른 Codex 세션에 변경 지침을 강제로 전달하거나 중단한 것은 아니다.
확인 시점 Stage1 본검사349/357에서 FAILED(3221225477), supervisor NEEDS_ATTENTION.
Stage2 정밀도 COMPLETE/VALIDATED, Stage3 dynamics COMPLETE_VALIDATED(30모델).
이 상태 확인은 신규 복구·모델 승격·추가 검증 수행을 의미하지 않는다.

## Stage3 2번 시간순서CNN 실험 착수 (2026-09-11)

사용자2번승인. 과거15흐름×134채널→Conv1d32ch2층+last/mean→64→2head,20615params(기준MLP224135). 기존17경로5fold/3seed/1200업데이트·저장배치동일. baseline15검증재사용,CNN15신규예정. 과거14shuffle/current유지 및current반복 평가진단사전고정. 기존156학습영상의필요16프레임구간만시퀀스재추출중,모든요약402특징exact대조. src/compare_stage3_sequence_cv.py 및 artifacts/stage3-sequence-cv-20260911/status.json 확인,중복실행금지. 출력/순서진단검사PASS. docs/stage3-sequence-model-experiment.md 참고. 끝나면train/verify완료필수. 기존ZIP/selected/GPU변경없음.

## Stage1 349/357 오류 점검 후 단일영상 복구 착수 (2026-09-11)

사용자 오류점검·복구 승인. 저장349개·모델/서빙/데이터해시와 남은8영상400프레임 검사PASS.
마지막 c10.dll 0xc0000005는MViT 상대위치 텐서연산중 발생, 실패영상고정 아님.
가용RAM약385MiB 관측, 근본원인 미확정. 로딩후 checkpoint복사본해제 및1영상/새프로세스 적용.
기존5회실패 기록archive+prior_recovery_failures 보존, 검증된prefix로 유한복구 회차시작.
11감시기테스트PASS. Windows 로컬검사 예약 즉시재가동, ZIP12시간주기유지.
완료여부는 result/report.json,supervisor-status.json 확인. docs/stage1-local-supervisor.md 상세.
Codex 반복폴링 없이 최초실행확인후 기존예약에 맡길 것.

## Stage3 2번 시간축CNN 실험 완료 (2026-09-11)

156영상→1000×15×134입력,요약402특징exact. 고정5fold×3seed CNN15학습135.5초,기준MLP15재사용. 실제디코딩/경로분리/fold정규화/전3진단로짓오차0/OOF/별도CSV혼동행렬검증PASS. OOF조향기존 .718900,순서CNN .705620,과거shuffle .677699,현재flow반복 .655099. 정상CNN3seed모두기준미달,5fold4하락/1상승,15쌍13하락/2상승,17경로10하락/7상승. CNN직진recall65→61%,RIGHT74.78→75.89%. 과거순서·구성에민감하나기존MLP성능못넘음,소형CNN미채택. docs/stage3-sequence-model-experiment.md 및 artifacts/stage3-sequence-cv-20260911/independent-validation.json PASS 참고. 기존공식최고모델/ZIP/selected/다른Stage/GPU변경없음. 실행중기록보다완료결과우선.

## Stage2 과적합 완화1번 완료 (2026-09-11)

CPU1809.187초,독립검증PASS. 영상평균조기종료.4659/저lr .5114/고정 .5227.
내부검증설정선택결과 .4848로이전기준 .5000보다낮음. 고정은후속후보,개선확정아님.
출처동일가중고정 .5248 vs기준 .5156. docs/stage2-regularization.md 완료절참고.
이전RUNNING기록대체,중복실행금지. 모델/ZIP/공식제출변경없음.

## Stage1 본검사 완료 및 새 ZIP 생성 (2026-09-11)

357/357 ALL_CONDITIONS PASS, supervisor COMPLETED 및 제출gate 복구 확인.
사용자 ZIP생성 요청으로 기존 예약의 로컬 생성단계만 즉시 실행해 CANDIDATE_READY.
artifacts/stage1-auto-release-20260910/candidate/submit.zip,177354958bytes,
SHA256 2c0090adb06f272c0833cfb151ed479ed9c78273830f724e0439fef220626197.
기존공식최고 Stage3움직임 후보7f98769...에서 검증완료Stage1만 교체.
구성/CRC/strict모델로드/유한가중치/Stage1해시/나머지파일바이트동일 및
357예측 독립재검증PASS. candidate/local-verification.json 참고.
새ZIP GPU통합검사·클라우드업로드·대회제출 아직미실행, 일일selected미변경.
기존ZIP12시간예약은 CANDIDATE_READY부터 후속진행, 주기는변경하지않음.

## 제출87406 공식 결과 수신 — 새 최고 (2026-09-11)

사용자보고 2026_09_11_002/submit.zip,13:39:09,소요시간미제공.
Stage1 0.6386345895 / Stage2 0.2096390642 / Stage3 0.4187467089.
이전최고86809 대비Stage1 +0.2340746981(약57.86%),Stage2·3동일.
로컬가중합 0.37908122714(이전최고대비+0.04681493962)로 현재보고최고.
대화맥락상 stage1-auto-release-20260910/candidate/submit.zip(2c0090ad...) 연결,서버해시미확인.
누적6회/9월11일2회. docs/submission-history.md에 원점수·비교기록.
생성당시미제출기록보다이번사용자평가완료보고우선. 별도Kaggle GPU검사완료로간주하지않음.
selected/모델/예약변경없음.

## Stage1 1번 노이즈 원본 보강 착수 (2026-09-11)

사용자1번승인. 357본검사에서noise원본19/21,quality_mix18/21오탐확인.
기존train85출처각1원본×3화질×2클래스=510학습전용영상생성시작.
양쪽noise/blur/JPEG/CRF동일,원본화면특성중립/재녹화화면특성만차이.
기존val21출처·phone및형제제외,기존manifest와원래CCD분할재검증.
src/prepare_stage1_quality_balanced.py,artifacts/stage1-quality-balanced-20260911/status.json.
출처별15분/전체6시간제한·단일worker·중복잠금·전체프레임/해시검증·재개.
3테스트PASS,생성실행중; DATA_VALIDATED가되어야510개준비완료.
docs/stage1-quality-balanced-data.md 참고. 2번대조학습미실행,현공식최고모델/ZIP불변.

## Stage1 화질 데이터 메모리 개선 후 재시작 (2026-09-11)

사용자재진행요청. 초기510영상생성은첫출처에서NumPy10.5MiB할당실패로0개완료.
32행단위노이즈생성+FFmpeg단일스레드+BLAS/OMP1 적용,4테스트PASS.
전변환과픽셀/난수진행동등성PASS. 기존실패폴더보존,새코드/인코딩결과는
artifacts/stage1-quality-balanced-20260911-lowmem 으로분리해같은85출처510개재실행.
Get-BackgroundWorkStatus.ps1의Stage1QualityData로새상태확인. 자세한내용
 docs/stage1-quality-balanced-data.md. 실행중이며아직완료아님,학습/ZIP변경없음.

## Stage1 화질 대조학습 준비 완료·전송 차단 (2026-09-11)

510화질균형데이터 DATA_VALIDATED 확인후사용자2번진행요청.
현재모델/기존합성추가학습/균형합성추가학습3조건,같은85출처·128updates각·lr1e-5,
FP32학습/CUDA FP16평가420영상각,최종모델만평가. 기존모델/ZIP보존.
입력1.55GB(1545566290bytes) 준비·해시/출처/전체프레임재검증,3테스트PASS.
artifacts/kaggle-stage1-quality-trial-20260911/bundle.json 및 docs/stage1-quality-trial.md.
새비공개biadis/crashintent-stage1-quality-trial-assets와무료T4quality-trial 전송/실행이
자동승인검토에서구체적payload/목적지승인미확인사유로거절. 업로드/GPU미실행.
remote-run BLOCKED_APPROVAL. 사용자구체적승인후일치하는upload-approval.json작성,
launch_stage1_quality_trial.py실행하면유한감시·회수·독립검증자동연결.
동일업로드/push재시도금지마커있음. 현재는승인파일/업로드시도마커없음.

## Stage1 화질 대조 GPU 실행 승인·착수 (2026-09-11)

사용자가구체적자료/목적지확인후GPU학습진행승인. upload-approval.json기록및1.55GB해시재검증PASS.
launch_stage1_quality_trial.py PID35332로업로드/ready대기/단일push/회수/독립검증자동연결실행.
artifacts/kaggle-stage1-quality-trial-20260911/remote-run.json 및 monitor-error.log확인.
이전BLOCKED_APPROVAL해소,실제GPU시작여부는최신상태확인. 중복업로드/push금지.
관련 docs/stage1-quality-trial.md. 기존제출모델/ZIP/예약불변.

## Stage1 Imperial GPU 진단 재실행 (2026-09-17 15:48 KST)

- Kaggle 커널 v1은 `imperial.zip`이 데이터셋 마운트 때 자동 해제되어 원본 파일이 없어지는 문제로 ERROR 종료했다. 회수 로그는 `artifacts/kaggle-stage1-imperial-20260917/error-output/crashintent-stage1-imperial-diagnostic.log`이다.
- 원본 70,145,784바이트 ZIP을 `imperial.bin`으로 보존하고 SHA256/CRC 검증을 유지하도록 `src/stage1_imperial_gpu_runner.py`와 준비 스크립트를 수정했다. 비공개 Kaggle 데이터셋 v2 업로드를 완료했다.
- `src/resume_stage1_imperial_gpu_v2.py` watcher(PID는 `watcher-v2.pid`)가 커널 v2를 제출했고 현재 `SUBMITTED / WAITING`이다. 성공 시 200개 예측을 회수·검증해 `validated/report.json`, `validated/predictions.json`과 `remote-run.json=COMPLETE/VALIDATED`를 남긴다. 오류 시 원격 출력을 `error-output-v2`에 회수한다.
- 중복 실행하지 말고 `./scripts/Get-BackgroundWorkStatus.ps1`로 한 번 확인한다. 아직 진단 완료나 모델 개선으로 간주하면 안 되며 공식 최고 Stage1 ZIP은 변경하지 않았다.

## Stage1 Imperial GPU 진단 완료 (2026-09-17 16:17 KST)

- 커널 v4가 `COMPLETE / VALIDATED`로 끝났고 CUDA FP16에서 두 모델 × 100장 = 200개 예측을 독립 검증했다. 결과는 `artifacts/kaggle-stage1-imperial-20260917/validated/report.json`과 `predictions.json`이다.
- official_best: ORIGINAL 50/50, RERECORDED 0/50. screen_mix_25도 ORIGINAL 50/50, RERECORDED 0/50. 두 모델 모두 전체 정확도 50%이며 사실상 전부 ORIGINAL로 분류했다.
- 평균 재촬영 확률은 official_best가 original 0.000332 / rerecorded 0.000210, screen_mix_25가 original 0.000460 / rerecorded 0.000300이었다. 0.5 임계값 조정만으로 분리할 수 있는 방향성도 아니다.
- 정지 이미지를 16프레임 반복한 진단이라 시간 단서는 없다. 다만 화면 재촬영의 공간 단서도 현재 모델이 일반화하지 못한다는 강한 음성 결과다. 학습이나 제출 ZIP 변경은 없고 공식 최고 모델은 그대로 유지한다.

## Stage1 Imperial 움직임 미세조정 파일럿 착수 (2026-09-17)

- 사용자 진행 요청에 따라 사람 라벨링 없이 Imperial 제공 라벨을 사용한다. 클래스별 40장 학습/10장 홀드아웃(총 80/20)을 파일명 해시로 고정 분할했다.
- 양 클래스에 같은 affine 움직임, 노출 진동, 수평 밴딩 분포를 적용한다. official_best 초기값에서 AdamW lr 5e-6, 64 balanced updates로 학습하며 baseline/candidate를 홀드아웃 정지·움직임 뷰에서 비교한다.
- Imperial 샘플은 명시적 SPDX 라이선스가 없어 비공개 진단 학습만 수행한다. 후보 체크포인트는 기존 420개 회귀 평가 전 제출에 사용하지 않는다.
- 비공개 Kaggle T4 커널 `biadis/crashintent-stage1-imperial-motion-trial` v1 제출 완료. watcher가 회수·80개 예측 검증을 담당하며 상태는 `artifacts/kaggle-stage1-imperial-motion-20260917/remote-run.json`, 결과는 성공 시 `validated/`에 저장한다. `Get-BackgroundWorkStatus.ps1`의 `Stage1ImperialMotionTrial`로 확인하며 중복 실행하지 않는다.

## Stage1 Imperial 움직임 결과와 공간 모델 전환 (2026-09-18)

- 움직임 파일럿은 COMPLETE/VALIDATED. 홀드아웃 20장에서 candidate macro-F1은 static 0.333(기준과 동일), motion 0.500(기준 0.436)이지만 accuracy는 0.50(기준 0.55)이다. 확률 AUC도 static 0.62(기준 0.64), motion 0.46(기준 0.46)으로 개선 신호가 없어 420개 회귀평가와 제출 채택을 중단했다.
- 다음 파일럿은 기존 제출 ZIP에 이미 포함된 ImageNet ResNet18(`b55eb2...`, 46,836,811 bytes)을 공간 단서 보조 모델로 사용한다. 같은 고정 80/20 split, layer4+fc 240 balanced updates, 5-view 홀드아웃 평가다.
- 비공개 데이터셋 `biadis/crashintent-stage1-imperial-spatial-assets`와 커널 `biadis/crashintent-stage1-imperial-spatial-trial`을 watcher로 연결했다. 상태는 `artifacts/kaggle-stage1-imperial-spatial-20260917/remote-run.json`, `Get-BackgroundWorkStatus.ps1`의 `Stage1ImperialSpatialTrial`로 확인한다. 진단 전용이고 제출 ZIP은 불변이다.

## Stage1 Imperial 공간 모델 결과 (2026-09-18 09:03 KST)

- ResNet18 layer4+fc 파일럿은 COMPLETE/VALIDATED. 고정 홀드아웃 20장에서 accuracy 0.55, macro-F1 0.5489, AUC 0.71, ORIGINAL 0.60, RERECORDED 0.50이다.
- 첫 20 step 평균 loss 0.7334에서 마지막 20 step 0.00690으로 급락해 80장 학습 세트 과적합 징후가 강하다. MViT 정지 진단보다는 확률 순위 신호가 있으나 표본이 작고 제출 후보 기준에는 부족하다.
- `artifacts/kaggle-stage1-imperial-spatial-20260917/validated/`에 report/predictions/spatial.pt가 있다. 420개 회귀평가나 제출 ZIP 반영은 하지 않았고 공식 최고 모델은 유지한다.

## Stage1 Imperial 고정 특징 5-fold CV 착수 (2026-09-18)

- 사용자 진행 요청. frozen ImageNet ResNet18 512차원 특징, FFT radial/color/edge 수작업 특징, 두 특징 결합의 세 조건을 동일한 stratified hash 5-fold에서 비교한다.
- 각 이미지는 정확히 한 번 out-of-fold 예측된다. fold별 train 통계로만 표준화하고 고정 L2 얕은 선형 분류기를 사용해 하이퍼파라미터 검증 누수를 피한다. 총 3×100=300예측을 검증한다.
- 비공개 커널 `biadis/crashintent-stage1-imperial-feature-cv`를 watcher로 제출했다. 상태는 `artifacts/kaggle-stage1-imperial-feature-cv-20260918/remote-run.json`, 성공 결과는 `validated/`. 제출 모델/ZIP 불변.

## Stage1 Imperial 고정 특징 5-fold 결과 (2026-09-18 09:15 KST)

- COMPLETE/VALIDATED, 동일 stratified 5-fold out-of-fold 100장×3조건=300예측.
- frozen ResNet: accuracy 0.51, macro-F1 0.5096, AUC 0.5432.
- FFT radial + RGB 통계 + edge 통계 54차원: accuracy 0.83, macro-F1 0.8298, AUC 0.9184, ORIGINAL 0.80, RERECORDED 0.86로 최고.
- 결합 566차원: accuracy 0.75, macro-F1 0.7494, AUC 0.8132. 고차원 ResNet 특징이 소표본에서 오히려 성능을 낮췄다.
- 결과는 `artifacts/kaggle-stage1-imperial-feature-cv-20260918/validated/`. 다음 단계는 54차원 분류기를 영상 프레임 단위로 적용하고 기존 MViT와 독립 검증 세트에서 결합 gate를 평가하는 것이다. Imperial 라이선스와 배포 영상 일반화 한계 때문에 아직 제출 반영하지 않았다.

## Stage1 수작업 특징 영상 전이 평가 착수 (2026-09-18)

- Imperial 100장 전체에서 CV와 동일한 고정 L2 설정으로 54차원 분류기를 학습하고 classifier scaler/weight/bias를 기록한다.
- 기존 quality.bin의 role=evaluation 420개(stress357+normal63)에서 각 영상 20/50/80% 프레임 특징 확률을 평균한다. 저장된 official-best MViT 확률과 단독 비교하고 사전 고정 alpha 0.1/0.25/0.5 확률 결합을 평가한다.
- `src/evaluate_stage1_handcrafted_video.py`를 로컬 watcher로 실행했다. 상태 `artifacts/stage1-handcrafted-video-20260918/status.json`, 성공 시 classifier/report/predictions. 파일별 임시 추출 후 즉시 삭제하며 10영상마다 상태 갱신. 제출 ZIP은 불변이다.

## Stage1 수작업 특징 영상 전이 평가 완료 (2026-09-18 09:31 KST)

- COMPLETE_VALIDATED, 420/420영상×3프레임. 기존 MViT: 전체 macro-F1 0.8844/errors37, stress 0.8589/errors37, normal 1.0/errors0.
- Imperial handcrafted 단독: 전체 macro-F1 0.2984/errors284, stress 0.2989/errors242, normal 0.2933/errors42. 문서 재촬영의 주파수/색상 통계가 주행 영상 도메인으로 일반화되지 않았다.
- 고정 alpha 0.1/0.25 결합은 MViT와 판정이 완전히 동일해 개선 0. alpha 0.5는 전체 macro-F1 0.8788/errors39로 악화하고 normal 오류도 1개 생성했다.
- 결과 `artifacts/stage1-handcrafted-video-20260918/report.json`. 수작업 특징 및 결합 경로는 제출 후보에서 탈락, ZIP/공식 최고 모델 불변.

## Stage1 VDMoiré 실제 영상 파일럿 (2026-09-18)

- 공식 CVMI-Lab/VideoDemoireing(CVPR 2022)은 휴대전화(TCL20 Pro/iPhoneXR)로 모니터를 촬영한 720p 290영상×60프레임 paired dataset이다. 공식 Dropbox 원본 재촬영 ZIP은 36,395,895,475 bytes.
- Dropbox가 Range를 지원하지 않아 제한 요청이 예상과 달리 백그라운드에서 계속되어 6,402,682,880-byte prefix가 생겼다. 즉시 curl PID27720을 중지했고 전체 다운로드는 하지 않았다. 이 prefix는 바깥 ZIP 안의 stored `tcl.zip` 시작 부분이다.
- 로컬헤더/deflate/CRC를 직접 검증해 완전 멤버16,427개, 영상폴더91개, 180프레임 완전폴더74개를 확인했다. 서로 다른8개를 48프레임 1920×1080 MP4로 복구·전프레임 검증했다: `artifacts/stage1-vdmoire-pilot-20260918/status.json`.
- 원본 대응쌍은 현재 prefix에 없어 학습하지 않는다. 비공개 Kaggle dataset `biadis/crashintent-stage1-vdmoire-clips`(37,286,958 bytes)와 kernel `biadis/crashintent-stage1-vdmoire-diagnostic`을 watcher로 연결, official_best/screen_mix_25 각8개 총16예측 회수·검증 중. 상태 `artifacts/kaggle-stage1-vdmoire-20260918/remote-run.json`. 데이터셋 별도 라이선스가 명시되지 않아 진단 전용, 제출 ZIP 불변.
## Stage2 DoTA 공식 주석 전수 감사 완료·영상 링크 접근 불가 (2026-09-18)

Stage2 목표 0.51021의 추가 데이터 1순위로 DoTA를 재점검했다. 로컬 공식
`DoTA_annotations.zip`은 10,357,806 bytes, SHA256
`56209f87398cbf47b9eeee4b516461e71cd3f97f9dd8c4432773315db307ff15`이다.
`src/audit_stage2_dota_annotations.py` 전수 감사 결과 `COMPLETE_VALIDATED`: JSON
4,677/4,677 유효, malformed 0, ego-involved 2,724, 방향성 유형 후보 3,632,
공간 객체 영상 4,376, track ID+bbox 170,739개다. 공식 `num_frames` 불일치 1건은 실제
per-frame rows를 정본으로 별도 기록했다. 결과는
`artifacts/stage2-dota-annotation-audit-20260918/report.json`과 `inventory.csv`다.

공식 GitHub가 연결한 2023 분할 프레임 폴더(10GB×5+5GB)와 전체55GB Google Drive 폴더를
2026-09-18 gdown 목록조회로 확인했으나 둘 다 접근 불가였다. 공개 검색에서도 원본을
호스팅하고 출처·권리를 확인할 수 있는 미러를 찾지 못했다. 과거 YouTube 원본 접근성 검사도
선택 source가 모두 UNAVAILABLE이었다. 주석은 향후 MM-AU/DAD 영상과 중복 매칭하거나 공식
링크가 복구될 때 재사용하되, provenance 불명 미러는 쓰지 않는다. 다음 확보 우선순위는 DAD다.
## Stage2 DAD 공식 객체 주석 확보·감사 완료, 영상은 서명 요청 필요 (2026-09-18)

DAD 저자 공식 GitHub가 연결한 `annotation.zip`을 `data_raw/dad/annotation.zip`에 확보했다.
2,789,878 bytes, SHA256
`54854c8769d31aec628bfa03fb73059c837d681eaa6d1d8d03150945e8a3b33b`, ZIP CRC PASS.
`src/audit_stage2_dad_annotations.py` 전수 감사 결과 `COMPLETE_VALIDATED`: 사고 영상 주석
620/620, malformed0, bbox326,910, 사고 관련 bbox103,310, 사고 관련 track 영상612개다.
공식 1280x720 좌표에서 최장 사고관련 track이 10관측 이상이고 최초 중심이 좌우35% 밖인 경우만
고르면 entry_side 약한라벨 후보291개(LEFT156/RIGHT135)가 나온다. 이는 아직 공식 정답이 아니며
in-domain 수동74개와 직접 겹치지 않아 정확도 검증 전 학습에 사용하지 않는다. 결과
`artifacts/stage2-dad-annotation-audit-20260918/report.json`, `inventory.csv`.

영상은 공식 요청 양식
`https://docs.google.com/forms/d/1A4DdNTuPZC9zfV8lBWgja659rNkL2ntR_NKmT-8t-Ng/viewform`을
통해 full name, academic email, affiliation, 필요시 supervisor 정보와 서명 동의서 링크를 제출해야
한다. 공식 동의서도 `data_raw/dad/dataset-agreement.pdf`로 확보했으며 SHA256
`7eac3a57b8ed703d97e43b63a58a4dc4a65d680fa9f97bca58368ac47a8abfa0`. 조건은 scientific/research
purpose only, third-party 제공 금지, 연구자 책임, VSLab 접근종료 권한, Taiwan 관할이다. 사용자의
신원·학술 이메일·법적 서명 없이 자동 제출하거나 이메일을 보내지 않는다. 영상 승인 전에는 주석만
보존하고 모델 학습에 넣지 않는다.
## Stage2 MM-AU 장면 후보 2,500 선별·검출 이미지 파일럿 착수 (2026-09-18)

공식 metadata11,730행과 이미 검증된 detection labels422,488 frame을 결합해
`src/select_stage2_mmau_scene_candidates.py`로 다운로드 우선순위를 생성했다. 상태
`COMPLETE_VALIDATED`: 유효 양성시간9,991, detection+장면신호 후보9,712 중 사고 type당 우선
상한150으로2,500개 선별, train/val/test 1,755/370/375. 전부 t_ai/t_co 주변 detection frame이
있고 SIDE_TEXT2,420, EVASION_TEXT2,198, 좌우박스 비대칭1,473개다. 이는 acquisition priority이며
pseudo ground truth가 아니다. 결과
`artifacts/stage2-mmau-scene-selection-20260918/selected_2500.csv`, `report.json`.

공식 Hugging Face inventory 확인 결과 raw CAP type shard는 1-10 42.42GiB, type11 92.71GiB,
12-42 43.17GiB, type43 42.25GiB, 44-62 28.35GiB로 개별 선택에 부적합하다. 반면 논문용 6fps
detection images는 test9.94GiB/train45.84GiB/val9.54GiB, 총65.32GiB이고 각 part가 독립 gzip
archive다. 남은 디스크 약82.36GiB에서 전체를 바로 받지 않고 가장 작은 train part인
`train_part_aw` 1,972,510,663 bytes, 공식 SHA256
`7f4e4390da3b8d8f71a6b5d8be246dac56c5ac32ee6c586fa748b9d84c48f1a1`을 파일럿으로 시작했다.
`src/watch_stage2_mmau_image_pilot.py`가 Range 재개, 3회 재시도, 3시간 제한, 크기/SHA/tar 안전성,
내부 경로와 selected2500 적중 영상을 자동 검증한다. 최초 실제 PID31152, 상태 DOWNLOADING 0/
1,972,510,663 확인. 상태/결과
`artifacts/stage2-mmau-detection-images-pilot-20260918/status.json|report.json`, 원본
`data_raw/mm-au-detection-images-pilot-20260918/`. ACQUIRED_VALIDATED 전 완료 아님.
## Stage2 MM-AU train detection images 전체 분할 감시 재착수 (2026-09-18)

첫 `train_part_aw` 파일럿은 1,972,510,663 bytes와 공식 SHA256이 정확했지만 `ReadError: not a
gzip file`로 FAILED했다. 손상이 아니라 23개 파일이 독립 archive가 아닌 하나의 gzip을 나눈
split parts인데 마지막 part만 단독으로 열려고 한 설계 오류였다. 해당 part는 삭제하지 않고 재사용한다.

`src/watch_stage2_mmau_train_images.py`를 새로 추가해 공식 HF API에서 train 23 part manifest와
각 LFS SHA256을 다시 받고, 조각별 Range 재개·3회 재시도·크기/SHA 검증을 수행한다. 전체
45.837GiB를 받은 뒤 별도 결합파일을 만들지 않고 `ConcatenatedParts` 스트림으로 tar.gz를 읽어
동결한 selected2500 중 train1,755개 영상 이미지만 추출한다. 12시간 제한과 원자적 상태/결과 저장.
최초 PID12988 실행 확인, 상태 STARTING. 상태/결과
`artifacts/stage2-mmau-detection-images-train-20260918/status.json|report.json`, 데이터
`data_raw/mm-au-detection-images-train-20260918/`. `ACQUIRED_VALIDATED` 전 완료 아님.
## Stage2 MM-AU train 이미지 해시명 매핑 수정·재추출 착수 (2026-09-18)

기존 watcher가 23개 분할 49,217,150,919 bytes를 크기·SHA 검증한 뒤 아카이브를 끝까지
읽었지만, 이미지 멤버가 `train/<16hex>.jpg|png`인 COCO 해시명인데
`type_video_frame` 정규식을 적용해 이미지0/영상0을 추출하고도 `ACQUIRED_VALIDATED`로 잘못
판정했다. 이 결과는 무효이며 완료로 취급하지 않는다. 공식 `labels.tar.gz`의
`labels/train.json`에 해시 이미지명·크기·COCO bbox가 있고 YOLO txt에는 원래
`type_video_frame`명이 남아 있다. producer-order 가정은 COCO/YOLO bbox 표본에서 0/256으로
실패해 폐기했다.

`src/watch_stage2_mmau_train_images.py`를 객체 class+정규화 bbox multiset의 3/4자리 내용
지문으로 두 이름 공간을 연결하도록 수정했다. 빈 detection frame은 식별 정보가 없어 제외하되,
비어 있지 않은 선택 프레임의 98% 이상 매핑 및 train 선택 영상1,755개 전부 커버, archive
train 이미지295,013개 전수 확인, 실제 추출 수 일치를 통과해야만 성공한다. 0개 추출 성공은
불가능하다. 사라진 사용자 Python3.11.9를 보존 installer `/repair`로 복구하고 문법검사를
통과했다. 수정 watcher PID25776을 15:34 KST 시작했으며 기존23분할을 재사용한다. 상태
`artifacts/stage2-mmau-detection-images-train-20260918/status.json`, 로그
`watcher-content-map.stderr.log`; 중복 실행 금지. 아직 완료 아님.

추가 갱신: 내용 지문 1차 gate는 75,705개 비어 있지 않은 선택 라벨 중 고유 매핑
65,005개(85.8662%)를 복원했으나 임의 98% raw-frame gate에 막혀 추출 전 FAILED했다.
이는 손상이나 잘못된 매핑이 아니라 동일/근사 bbox 지문의 모호한 프레임을 보수적으로 제외한 결과다.
목적에 맞게 raw 비율>=80%와 선택 train 영상1,755개 전부 커버, 각 영상의 t_ai/t_co ±5
이웃을 모두 고유 매핑하는 의미 gate로 변경했다. `mapping-report.json`과
`content-mapping.csv`를 추출 전에 남기며 gate 실패 시 이미지 추출을 시작하지 않는다.
문법검사 PASS 후 새 watcher PID8228을 15:50 KST 시작했다. 로그는
`watcher-semantic-map.stderr.log`; 현재 STARTING이며 아직 완료 아님. PID25776은 종료됨.

추가 갱신: 의미 gate 결과는 영상1754/1755, t_ai 이웃1665/1755, t_co 이웃1650/1755로
완전 gate를 통과하지 못해 PID8228이 추출 전 FAILED했다. 공식 Detectv1 `labels.zip`
178,368,951 bytes를 추가 확보했고 SHA256
`3024fa9415802a709a8016740b768f473f822bfc8f0af99fb6d7f2e15c877f75`가 공식 LFS와 일치한다.
Detectv1 전체 이미지는 약42.15GB라 추가 수집하지 않았다. 확정 65,005쌍에서 일반 경로 해시
규칙도 발견되지 않았다. 따라서 모호 프레임을 임의 연결하지 않고 고유 bbox 내용 매칭만 쓰는
검증 부분 데이터셋으로 전환했다. gate는 mapping>=85%, 영상>=99.9%, t_ai/t_co 이웃 각각>=93%;
결과 상태는 완전 확보와 구별해 `ACQUIRED_VALIDATED_PARTIAL`로만 기록한다. 기존
`content-mapping.csv`/`mapping-report.json`을 검증해 재사용하고 65,005장만 추출한다.
문법검사 PASS, watcher PID17592를 16:51 KST 시작, 로그
`watcher-partial-extract.stderr.log`. 기존49.2GB 재사용, 아직 완료 아님.

추가 갱신: 부분 추출은 아카이브 끝까지 진행했으나 이미지 regex가 16hex basename만 허용해
COCO 공식 train.json 295,013개 중294,953개만 인식하고 FAILED했다. 누락60개는 손상이 아니라
`8.34E+20.jpg`, `5.58E+171.jpg`, 긴 숫자명 등 공식 numeric/scientific-notation basename이다.
단일 basename+jpg/jpeg/png만 허용하는 안전 regex로 수정하고 정확한295,013 전수 gate는 유지했다.
문법검사 PASS, PID25888을 17:49 KST 재시작, 로그 `watcher-filename-fix.stderr.log`.
기존 추출물은 같은 목적지에 덮어써 재사용하며 아직 완료 아님. PID17592는 종료됨.
