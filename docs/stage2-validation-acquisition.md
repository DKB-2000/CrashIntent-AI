# Stage2 검증 후보 확보 (2026-09-14)

## 연속 프레임 접촉 징후 점검 완료 (2026-09-14)

50개 전체에서 참고 event 시각 1초 전부터 원본 연속 60프레임씩 총 3,000프레임을
AI가 스틸 격자로 확인했다. 접촉·충격 의심 13 / 확인 구간에서 접촉 징후 안 보임 13 /
판단 불가 24. 이전 우선검수34/보류16은 접근 과정의 가시성 분류이며 이번 접촉 관찰과
다른 기준이다. 둘 다 확정 충돌 수나 최종 검증셋 채택 수가 아니다.

- 최신 페이지: [관찰 기록](../artifacts/stage2-contact-review-20260914/review.html)
- 최신 대기열: `data_raw/stage2-validation-nexar-20260914/review-contact-queue.csv`
- 근거: `artifacts/stage2-contact-review-20260914/observations.json`, `contact-review.csv`,
  `evidence-index.json`, 영상별 PNG/디코딩 로그, `report.json`
- 재현: `scripts/Prepare-Stage2ContactReview.ps1`로 근거 추출,
  AI 관찰 기록을 입력으로 `scripts/Complete-Stage2ContactReview.ps1`로 검증·목록 생성.

접촉 의심 13개: 00363,00085,00381,00541,00463,00528,00478,00298,00262,
00000,00141,00223,00927 (NEXAR 접두사 생략). 화면 흔들림은 제동·회피일 수도 있어
직접 접촉 확정으로 사용하지 않는다. 접촉 징후가 안 보인 13개 역시 전체 영상의 비충돌이나
근접사고 확정이 아니다. 가림·야간·구도 문제 24개를 포함해 원본50개 모두 유지했다.

원본50개와 근거PNG50개 SHA256, 영상별60개 PTS 단조증가, ID50개 일치 PASS.
기존 inventory/예비 대기열 보존, human_review=NOT_STARTED 및 stage2_eligibility=UNCONFIRMED
전부 유지. 이번 검수는 짧은 연속프레임 스틸 관찰이며 전체 영상 재생·오디오·사람 판정이 아니다.

자료 수집·중복 선별·AI 예비 관찰은 완료했다. 실제 충돌과 Stage2 진입 기준을 사람이 확인한
30~50건의 최종 검증셋은 아직 확보 완료로 볼 수 없다. 사람 검수 가능 시 최신 페이지에서
의심13 → 판단불가24 → 징후 안 보임13 순으로 원본을 확인하고 적합성 확정·라벨링한다.
그 결과 유효 수가30건 미만이면 외부 후보를 보충한다. 라벨링은 사용자 요청대로 보류한다.

## 50개 후보 적합성 예비 점검 완료 (2026-09-14)

사용자 이어서 진행 요청으로 사건 참고시각 주변 영상별6장(약1Hz), 총300스틸을
AI가 비교해 우선 검수34개/판단 보류16개로 정리했다. 실제 충돌 확정 수나 최종
검증셋 채택 수를 의미하지 않는다. Stage2 정답 및 사람 라벨은 생성하지 않았다.
50개 원본 해시 전부 검사PASS, 기존 목록과50개ID 일치, 삭제0.

우선 검수는 교차·차선 변경·근접 차량의 접근 과정이 보이는 사례다. 보류는 앞차 접근만
보여 별도 진입이 불명확하거나 비/어둠/카메라 각도로 접촉·차로 확인이 어려운 사례다.
보류16개: 00363,00819,00406,00777,00463,00333,00528,00792,
00262,00809,00472,00327,00678,00630,00065,00383 (NEXAR 접두사 생략).
단순 추돌을 영구 제외한 것이 아니며 후속 연속영상 검수로 재판정할 수 있다.

- 페이지: artifacts/stage2-eligibility-review-20260914/review.html
  (우선/보류 그룹, 스틸, 영문 관찰 메모, 원본 이벤트 부근 재생; 읽기전용)
- 최신 대기열: data_raw/stage2-validation-nexar-20260914/review-queue.csv
- 관찰·증거: 같은 artifacts 폴더 observations.json/evidence-index.csv/report.json
- 재현: scripts/Prepare-Stage2EligibilityReview.ps1

스틸 시작은 round(참고event-3)초이며1Hz 샘플링이라 정확한 접촉순간을 놓칠 수 있다.
관찰300장은 연속300프레임이나 프레임단위 검수완료가 아니다. 충돌유무는 모두
UNCONFIRMED로 남긴다. 다음에는 연속영상으로 실제충돌/근접사고·진입기준 적용 가능성을
확인한 후 사람라벨이 필요하다. 검수 우선순위를 모델성능 기반 제외 기준으로 사용하지 않는다.

## 완료 결과와 중복 의심 후속 판정 (2026-09-14)

수집50/50 ACQUIRED_UNLABELED, 지문1550/1550 및76,225쌍 SCREENING_COMPLETE.
동일파일0, 자동 의심19쌍은 모두 NEXAR_00664 대 CCD19영상이었다.
사용자 이어서 진행 요청으로 각 의심구간의 0/0.5/1초 간격3장씩을 추출해
5개 비교판의 총114스틸을 AI가 시각 비교했다. 사용자 사람 검수나 Stage2 정답 라벨이 아니다.

19쌍 모두 표시된 구간에서 다른 장면으로 판단했다. Nexar는 저층상가·주차차량·전신주가
있는 역광 도시 도로이며, CCD 참조들은 숲길/설경/야간/아파트/다른 구조의 도로 등으로
구별된다. 저해상도 밝기·도로 구도 유사성에 따른 지문 오탐 해석을 지지한다.
해당 후보를 제외할 중복 근거가 없어 50개 모두 미라벨 검증 후보로 유지한다.

- 판정: artifacts/stage2-duplicate-review-20260914/pair-review.csv (19쌍·개별 사유)
- 비교판: 같은 폴더 review.html 및 board-0.png~board-4.png
- 검증: 같은 폴더 report.json, 관련20원본 해시 전부 기존 지문 출처기록과 일치 PASS
- 최신 목록: data_raw/stage2-validation-nexar-20260914/review-inventory-resolved.csv
- 기존 review-inventory.csv·suspected-pairs.json·검사보고서는 덮어쓰지 않고 보존

이 판정은 자동 표시된 구간만 해소한 것이며 모든 재편집/촬영출처/route 중복의 부재를
보장하지 않는다. 최종30~50건 확정에는 실제 충돌/근접사고 및 진입 판단 가능성 검수가
남는다. 현재 라벨링 보류 유지, 학습·모델예측·ZIP 변경 없음.

## 후속 중복 검사·검수 목록 (2026-09-14)

사용자 후속 진행 요청으로 `scripts/Inspect-Stage2ValidationCandidates.ps1`을 추가했다.
수집 중복 실행 없이 기존 수집기를 유지하고, 로컬 CCD 지문 추출부터 독립 진행한다.
최초 실행 PID8140. 수집 완료를 기다리는 일은 이 로컬 스크립트가 담당한다.
OS 파일 잠금, 전체 4시간 제한, 영상당 FFmpeg 30초 제한, 실패 상태 기록을 적용했다.
실패 후 재실행하면 검사 처음부터 다시 수행하며 자동 무한 재시도하지 않는다.

- 준비 완료: `data_raw/stage2-validation-nexar-20260914/review-inventory.csv` 50행.
  원본 URL/revision/해시/용량/환경/참고 이벤트시각, VALIDATION_ONLY 및 NOT_STARTED 표시.
  수집/검사 완료 전 길이와 유사 그룹은 비어 있고 acquisition=PENDING이다.
- 예약 목록: `artifacts/stage2-validation-inspection-20260914/reservation.json`.
  향후 학습 자료 구성 시 이 50개 ID와 동일장면을 제외해야 한다.
- 검사 상태·로그: 같은 artifacts 폴더의 status.json 및 worker-error.log.
- 완료 후: report.json, suspected-pairs.json, fingerprint-provenance.json,
  갱신된 review-inventory.csv와 같은 데이터 폴더의 review.html.

영상별 초당2장, 9×8 회색조 dHash를 생성하고 연속3장의 Hamming 거리가 각각8이하,
합18이하인 구간을 중복 의심으로 표시한다. 후보↔CCD75,000쌍과 후보↔후보1,225쌍,
총76,225쌍을 검사한다. 파일 전체 SHA256 일치도 별도로 검사한다.
연결된 유사 쌍은 임시 검수 그룹으로만 묶는다. source/vehicle/route 확정 라벨이 아니다.
커널의 동일시퀀스·시간오프셋 매칭과 반대 비트 음성 대조 검사 PASS.
실제 원본 해시·수집 디코딩 증거·메타데이터1:1매핑·전체 비교 수를 검사한다.

`SCREENING_COMPLETE`가 되기 전까지 중복 검사 완료가 아니다.
이 방식은 전체 프레임 저해상도 시각 검사라 크롭·좌우반전·속도변경·강한 편집을
놓칠 수 있고 비슷한 도로에서 오탐할 수 있다. 의심쌍 0도 출처 독립성 증명이 아니다.
충돌/근접사고 판정이나 Stage2 정답 생성은 수행하지 않는다.

사용자 요청 범위는 검증자료 확보이며 라벨링은 보류한다.

## CCD 출처 감사

실제 Crash-1500.txt와 사전학습 CSV, 수동학습 CV fold-0 train/validation을
문자열 ID로 대조했다. 1,500영상/133출처 중 사전학습 1,296영상/110출처와
수동 66영상/23출처의 합집합은 전체 133출처다. 남는 출처와 영상은 0이다.
같은 출처에서 다른 클립을 고르는 것으로 독립 검증셋을 만들 수 없다.
입력 SHA256과 출처 목록: artifacts/stage2-validation-acquisition-20260914/ccd-source-audit.json.

## 새 후보

공식 Nexar 공개 train/positive에서 50영상을 선정했다. 총 838,265,916 bytes.
저장소 revision aa97deda5a59f00bb7187739053b7c72e14374df를 고정했다.
선정은 SHA256("stage2-holdout-20260914:" + 파일 경로)의 오름차순 첫 50개로,
모델 예측·정답·크기를 이용하지 않는다. metadata.csv는 원본 참고자료이며
Stage2 정답 CSV로 변환하지 않는다. 양성에는 실제 충돌과 근접사고가 섞여 있다.

- 영상·원문 README·LICENSE·metadata: data_raw/stage2-validation-nexar-20260914/
- 고정 목록·URL·공식 LFS SHA256·크기: artifacts/stage2-validation-acquisition-20260914/manifest.json
- 상태: 같은 폴더 status.json, 통과 목록 validated-files.json
- 로그: worker.log, worker-error.log, download.log, 영상별 decode.log

수집기 scripts/Get-Stage2ValidationCandidates.ps1 -Worker를 숨김 실행했다.
최초 PID 19800 확인. 최대 3시간(파일 처리 사이 확인), 다운로드 요청당 180초,
디코딩당 60초, 파일별 3회 유한 재시도, OS 파일 잠금으로 중복 실행 방지.
크기·공식 SHA256·CCD 전체 파일 SHA256 중복·FFmpeg 전체 프레임 디코딩을 검사한다.
ACQUIRED_UNLABELED가 되어야 50개 수집·기계 검사가 끝난 것이다.
현재 완료 여부는 상태 파일을 확인하고 실제 실행 여부는 PID와 함께 확인한다.
Codex는 완료 대기 반복 조회를 하지 않는다. PC가 켜져 있어야 수집이 계속된다.

상태 한 번 조회: `./scripts/Get-BackgroundWorkStatus.ps1`.
실패 시 worker-error.log/download.log 확인 후 같은 `-Worker`로 재개할 수 있다.
기존 파일도 다시 해시·디코딩하며 후보 목록을 다시 선정하지 않는다.

## 남은 검증과 사용 제한

50개 후보는 확정된 독립 50출처 또는 라벨 완료 50건을 뜻하지 않는다.
공개 메타데이터에는 촬영 route/vehicle 식별자가 없어 내부 촬영 출처 독립성을
확정하지 않았다. CCD와 다른 배포 출처이지만 재인코딩·재편집된 동일 장면은
파일 해시로 배제할 수 없으므로 시각적 중복 검사가 남아 있다.
과거 Stage1 Nexar 소량 검수 4건의 로컬 ID는 확보하지 못했으며 대조가 남는다.
Nexar 영상을 기존 Stage2 학습에 사용한 기록은 확인되지 않았다.
최종 30~50건은 실제 충돌·진입 판정 가능성과 중복 검수 후 확정해야 한다.
현재 수동 라벨 생성, 모델 추론·학습·특징 개선·ZIP 변경은 하지 않는다.
이 폴더를 향후 학습 목록에 넣지 말고 검증 후보로 보존한다.

출처: https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction
라이선스: Nexar Open Data License. 원문 고지와 출처표시를 보존한다.
자세한 사용 조건은 DATA_SOURCES.md 및 내려받은 LICENSE를 따른다.
