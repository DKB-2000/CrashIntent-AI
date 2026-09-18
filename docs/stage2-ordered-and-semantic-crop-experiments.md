# Stage2 순서 선택·화면 영역 특징 대조 (2026-09-16)

기존 공식 Stage2 모델과 제출 ZIP은 유지한다. 이 문서의 수치는 기존 CCD 수동 라벨
66영상·23출처를 반복 사용한 개발 교차검증이며 공식 점수가 아니다. Nexar 50개는
학습·설정 선택에 사용하지 않았다.

## 순서 제약 결과 — 완료

기존 5fold 체크포인트와 특징 캐시를 동결했다. 먼저 기존 argmax OOF 예측을
전항목 exact 재현한 뒤, `collision_logit[c] + entry_logit[e]` 최대쌍을
`e <= c` 조건에서 선택하고 변경된 두 시점으로 기존 scene head를 다시 평가했다.
출력은 `artifacts/stage2-ordered-joint-20260916/report.json` 및 조건별 CSV다.

| 학습 설정 | 기존 네 항목 개발평균 | 순서 공동 선택 | 변경된 충돌/진입/회피/방향 |
|---|---:|---:|---:|
| exact_5 | .428030 | .431818 | 0/2/0/1 |
| exact_15 | .401515 | .405303 | 3/4/0/3 |
| soft_mixed_15 | .450758 | .454545 | 1/5/0/1 |

세 조건 모두 출력 순서 위반0. `soft_mixed_15` 출처동일가중 평균은
.418841→.422464. 차이가 작고 회피 출력은 그대로이므로 제출 적용 근거가 아니다.
기존예측·출처분리·66 OOF·정규화된 원본특징해시 및 checkpoint 검증은 스크립트에
포함돼 있다. 수동 라벨과 CCD 충돌 proxy의 공식 기준 차이는 남아 있다.

## 상황 특징 대조 — 완료

감시기와 결과 모두 `COMPLETE_VALIDATED`, 5fold·66 OOF 완료, 기존 scene 출력
exact 재현 PASS다. 재시도 없이 정상 종료했다.

| 조건 | 충돌 | 진입 | 회피 | 방향 | 개발평균 | 출처동일가중 |
|---|---:|---:|---:|---:|---:|---:|
| 기존 scene head | .5000 | .3182 | .5000 | .4848 | .4508 | .4188 |
| 동일 선형 point probe | .5000 | .3182 | .5152 | .5152 | .4621 | .4543 |
| 좌·우·중앙 crop 추가 | .5000 | .3182 | .5152 | .5758 | .4773 | .4659 |

crop 조건은 기존 대비 개발평균 +.0265, 출처동일가중 +.0471이며 개선은 방향
정확도 +.0909에 집중됐다. 회피는 point probe와 같아 공간 특징이 회피를 개선했다는
근거는 없다. 한 seed·기존66개 반복개발이고 고차원 probe 과적합 위험이 있어 제출
모델로 승격하지 않는다. 다음 단계는 같은 고정 프로토콜의 seed 반복 안정성 및
독립 라벨 검증이다.

`src/compare_stage2_semantic_crops.py`는 기존 `soft_mixed_15`의 시점 argmax와
ResNet18 백본을 고정한다. 원래 hidden 두 시점 768차원으로 학습한 동일한 선형
scene probe(`point`)와 여기에 사건 프레임의 좌·우·중앙 하단 crop ResNet 특징을
추가한 probe(`semantic_crop`)를 fold별 학습 출처만으로 비교한다. 좌우 특징 차이와
중앙 특징을 충돌·진입 두 시점에서 붙인다. 네 항목 점수와 출처동일가중 점수를
보고하며 기존 scene head 예측도 exact 대조한다.

화면 crop은 차량·차선 분할 정답을 제공하는 모델이 아니다. ImageNet 고정 백본의
영역별 표현을 대조하는 탐색 실험이다. 학습 라벨은 기존66개뿐이고, fold별 scene
probe는 학습50여건의 고차원 특징을 사용하므로 과적합 위험이 크다. 결과가 좋아도
새 독립 정답과 공식 점수 상승으로 해석하지 않는다.

로컬 Python 3.11.9 휴대용 실행 파일을 프로젝트 `.tools/python311/runtime`에
설치해 기존 `.venv/Lib/site-packages`를 읽는다. Python ZIP은 공식 python.org
파일 11,249,023 bytes, SHA256
`009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b`.
기존 `.venv`는 수정하지 않았다. ResNet18 가중치 내용 해시는 baseline,
사전학습 ZIP, Stage1 최고 ZIP에서 모두
`b55eb2f9f3f559e2101e507d9acc49a5e0768f9d710eeb6a0912080e57595860`으로
일치했다. 기존 `DATA_SOURCES.md`의 `f37072...` 표기는 재확인 필요하다.

감시기 `src/watch_stage2_semantic_crops.py`는 중복시작 거절,
30분 제한·자동재시도0, 로그·완료/실패 상태를 기록한다.
`artifacts/stage2-semantic-crops-watch-20260916/status.json`과
`artifacts/stage2-semantic-crops-20260916/report.json`을 모두 확인해야 한다.
Codex가 반복 조회하지 않는다. 아직 모델 개선이나 제출 후보를 확정하지 않는다.

## 여러 seed 안정성 대조 — 완료

감시기와 결과가 모두 `COMPLETE_VALIDATED`, 5/5 seed 완료, exit code 0,
재시도0이다. 기존 scene OOF SHA는 전 seed에서 동일했다.

| 지표 | 기존 scene | crop 5seed 평균 | crop 최소~최대 | 전 seed 기존 초과 |
|---|---:|---:|---:|---:|
| 회피 정확도 | .5000 | .5242 | .5000~.5455 | 아니오 |
| 방향 정확도 | .4848 | .5848 | .5758~.5909 | 예 |
| 네 항목 개발평균 | .4508 | .4818 | .4773~.4848 | 예 |
| 출처동일가중 | .4188 | .4718 | .4659~.4871 | 예 |

방향 개선은 5seed 모두 유지됐고 개발평균은 평균 +.0311, 출처동일가중은
평균 +.0530이다. 회피는 한 seed가 기존과 같아 전 seed 개선이 아니다.
같은66개 반복개발이라는 한계 때문에 제출 승격은 보류한다. 독립 검증 라벨이 없으면
다음 단계는 동일 특징의 정규화·차원축소 또는 seed 앙상블을 기존 분할 안에서
진단할 수 있지만, 그 결과도 독립 성능 증거는 아니다.

## 작업별 crop 5seed 로짓 앙상블 직접 대조 — 완료·미채택 (2026-09-17)

5/5 seed와 watcher가 `COMPLETE_VALIDATED`로 끝났고 gate는 `FAIL`이다.

| 지표 | shared 앙상블 | 작업별 앙상블 | 변화 |
|---|---:|---:|---:|
| 회피 | 0.5000 | 0.5303 | +0.0303 |
| 방향 | 0.5758 | 0.6061 | +0.0303 |
| 개발평균 | 0.4735 | 0.4886 | +0.0152 |
| 출처동일가중 | 0.4714 | 0.4658 | -0.0056 |

출처 23과 37의 개발평균이 각각 -0.25 하락해 사전 한계 -0.10을 넘었다. 전체 표본 지표는
좋아졌지만 특정 출처 회귀와 출처동일가중 하락 때문에 미채택한다. 공식 shared crop
Stage2(.21955508)와 기존 ZIP을 유지한다.

실제 제출 구조에 맞춰 같은 fold 실행에서 shared와 작업별 probe 로짓을 모두 저장하고, 각 영상의
5seed 로짓을 평균해 앙상블끼리 비교한다. 사전 gate는 네 전체 지표 비열화 없음, 개발평균·
출처동일가중 엄격 상승, 최악 출처 개발평균 변화 `>= -0.10`이다. 결과는
`artifacts/stage2-task-specific-ensemble-20260917/report.json`, 감시는
`artifacts/stage2-task-specific-ensemble-watch-20260917/status.json`이다.

## 작업별 crop 특징 분리 — 완료·엄격 gate 미통과 (2026-09-17)

수정된 v2 실행은 5/5 seed와 watcher 모두 `COMPLETE_VALIDATED`이며 gate는 `FAIL`이다.

| 지표 | shared crop 평균 | 작업별 특징 평균 | 변화 | 전 seed 비열화 없음 |
|---|---:|---:|---:|:---:|
| 회피 | 0.5242 | 0.5333 | +0.0091 | 예 |
| 방향 | 0.5848 | 0.6121 | +0.0273 | 예 |
| 개발평균 | 0.4818 | 0.4909 | +0.0091 | 예 |
| 출처동일가중 | 0.4718 | 0.4780 | +0.0061 | 아니요 |

출처동일가중은 평균 상승했지만 2/5 seed에서 -0.00688, -0.00851 하락했다. 사전 정의한
전 seed 비열화 없음 조건 때문에 미채택한다. 개선 신호는 후속 안정화 후보로 보존하며 공식
argmax shared crop Stage2(.21955508)와 기존 ZIP을 유지한다. 유효 결과는
`artifacts/stage2-crop-task-specific-seeds-v2-20260917/report.json`이다.

현재 공식 crop 패키지가 이미 5seed probe 로짓 평균을 사용하므로 앙상블 재실험은 중복이다.
후속으로 회피 head에는 중앙 crop, 방향 head에는 좌우 차이 crop만 입력하고 point 특징은 양쪽에
유지하는 5seed 대조를 실행 중이다. argmax 사건 시점, 백본, source fold와 학습 설정은 고정한다.
결과는 `artifacts/stage2-crop-task-specific-seeds-20260917/report.json`, 감시는
`artifacts/stage2-crop-task-specific-seeds-watch-20260917/status.json`이다.

## crop 특징 PCA16 과적합 완화 — 완료·미채택 (2026-09-17)

5/5 seed와 감시가 `COMPLETE_VALIDATED`로 끝났고 gate는 `FAIL`이다.

| 지표 | 기존 crop 평균 | PCA16 평균 | 변화 |
|---|---:|---:|---:|
| 회피 | 0.5242 | 0.5182 | -0.0061 |
| 방향 | 0.5848 | 0.4667 | -0.1182 |
| 개발평균 | 0.4818 | 0.4508 | -0.0311 |
| 출처동일가중 | 0.4718 | 0.4336 | -0.0382 |

네 지표 모두 전 seed 비열화 조건을 통과하지 못했다. PCA16 압축은 방향 판별 정보를 크게
손실하므로 폐기한다. 새 ZIP과 제출은 만들지 않고 공식 argmax crop Stage2(.21955508)를 유지한다.

공식 개선을 낸 argmax crop 조건을 고정하고, 각 학습 fold에서만 표준화와 정확 SVD를
적합해 crop 특징을 16차원으로 압축한다. 검증 fold는 학습 fold의 평균·표준편차·주성분만
적용하므로 정보 누출이 없다. point 특징은 기존처럼 유지한다.

`src/run_stage2_crop_pca16_seeds.py`와 `src/watch_stage2_crop_pca16_seeds.py`로 5seed
대조를 실행 중이다. 결과는 `artifacts/stage2-crop-pca16-seeds-20260917/report.json`, 감시는
`artifacts/stage2-crop-pca16-seeds-watch-20260917/status.json`이다. 모든 seed에서 회피·방향·
개발평균·출처동일가중 비열화가 없고 두 평균이 엄격히 상승할 때만 채택한다.

## 공식 개선 crop + 순서 공동 선택 — 완료·미채택

5seed 대조는 `COMPLETE_VALIDATED`로 끝났고 사전 정의 gate는 `FAIL`이었다.

| 지표 | argmax crop | 순서 공동 선택 | 변화 |
|---|---:|---:|---:|
| 충돌 ±0.3초 | 0.5000 | 0.4848 | -0.0152 |
| 진입 ±0.3초 | 0.3182 | 0.3333 | +0.0152 |
| 회피 | 0.5242 | 0.5303 | +0.0061 |
| 방향 | 0.5848 | 0.5879 | +0.0030 |
| 개발평균 | 0.4818 | 0.4841 | +0.0023 |
| 출처동일가중 | 0.4718 | 0.4671 | -0.0047 |

출처동일가중이 평균 하락했고 4/5 seed에서 나빠졌으므로 채택하지 않는다. 새 ZIP과 제출은
만들지 않았고, 공식 개선을 확인한 기존 argmax crop Stage2(.21955508)를 유지한다.

공식 Stage2 `.21955508` 개선 확인 후, 현재 crop 조건에서 시점 argmax만
`collision_logit[c] + entry_logit[e]`, `e <= c` 공동 선택으로 바꾼 5seed
대조를 시작했다. 같은 5개 출처 분할·66개·frozen backbone/temporal을 사용하며,
공동 선택 위치에서 crop을 다시 추출하고 probe를 다시 학습한다.

사전 gate는 진입·회피·방향·개발평균·출처동일가중이 모든 seed에서 기존 crop보다
낮지 않고, 개발평균과 출처동일가중의 seed 평균이 엄격히 상승하는 것이다.
`src/run_stage2_crop_ordered_seeds.py`, `src/watch_stage2_crop_ordered_seeds.py`.
결과 `artifacts/stage2-crop-ordered-seeds-20260916/report.json`, 감시
`artifacts/stage2-crop-ordered-seeds-watch-20260916/status.json`. 1시간 제한,
재시도0, 최초 `RUNNING`·0/5·자식 실행 확인. gate PASS 전 ZIP/제출 없음.

사용자 후속 진행 요청으로 seed `20260825`, `20260826`, `20260827`, `20260916`,
`20260917`의 5조건을 고정했다. 각 seed는 같은 5개 출처 분할, 기존
`soft_mixed_15` 시점, frozen ResNet18 crop 특징, 80epoch·AdamW 설정을 사용한다.
기존 scene 출력의 seed간 SHA 동일성, seed별 point/crop OOF와 네 지표,
평균·최소·최대 및 기존보다 모든 seed가 높은지를 자동 검증한다.

`src/run_stage2_semantic_crop_seeds.py`와
`src/watch_stage2_semantic_crop_seeds.py`를 사용한다. 감시기는 1시간 제한,
재시도0, OS 수준 단일 시작 파일을 남긴다. 최초 상태 `RUNNING`, 완료0/5와
자식 프로세스 실행을 확인했다. 결과는
`artifacts/stage2-semantic-crop-seeds-20260916/report.json`, 감시는
`artifacts/stage2-semantic-crop-seeds-watch-20260916/status.json`이다.
두 상태가 `COMPLETE_VALIDATED`가 되기 전에는 완료나 안정적 개선으로 보고하지 않는다.
