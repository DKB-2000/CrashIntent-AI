# Stage3 짧은 독립 경로 검증 실험

2026-09-10: Kaggle kernel `biadis/crashintent-stage3-short-route-trial` version 1 완료, 독립 검증 PASS.

- 초기화: scratch MViT, FP32 학습, learning rate 3e-5, effective batch 10 (microbatch 2), 정규화 비활성화.
- 학습: 17개 경로의 34개 영상, 클래스 조합별 20개씩 총 200개 클립. 매 배치 9개 이동 조합 + STOPPED 1개.
- 최대 12 epoch / 240 optimizer updates. 검증 점수에 따른 checkpoint 선택 없이 마지막 모델 평가.
- 검증: 학습과 겹치지 않는 기존 4개 경로의 31개 영상 전체, dense FP16 추론. STOPPED 정답은 조향 F1에서 제외.
- 예산: runner 약 55분 이내 제한. 시간 초과 또는 일부 평가만 완료되면 PARTIAL로 기록.
- 목표: 기존 CONSTANT/STRAIGHT 예측의 검증 moving-steer macro F1 약 0.221897과 비교. 외부 데이터 대리 검증이며 공식 제출 점수와 다름.

사전 검증: 실제 데이터에서 샘플 200개 / 20개 균형 배치 / train-validation 경로 분리 확인, 누적 gradient 테스트 2개 PASS. 결과 회수 및 검증 완료, 제출 ZIP 변경 없음.

실행 자료: `artifacts/kaggle-stage3-route-trial-20260910/launch.json`, `sample-plan.json`, `dataset/route-trial-assets.json`.

결과 다운로드(실행 완료 후):
```
.\.venv\Scripts\kaggle.exe kernels output biadis/crashintent-stage3-short-route-trial/1 -p artifacts/kaggle-stage3-route-trial-20260910/result --file-pattern 'stage3-route-trial-result.zip$' -q
.\.venv\Scripts\python.exe src/validate_stage3_route_trial.py --root artifacts/kaggle-stage3-route-trial-20260910
```


## 완료 결과 (2026-09-10)

Kaggle version 1 COMPLETE, runner 2470.75초(약41분11초), 12에폭/240업데이트 완료. 검증31영상/18603행 전체 독립 점수 재계산 PASS, 소스·데이터·체크포인트 해시 일치. 분할 목록은 번들 생성 시 경로와 행 순서가 바뀌므로 원본 번들 해시 및 ID 정렬 후 경로 외 필드와 영상 파일명을 비교했다. 라벨 CSV는 바이트 단위 동일.

| 검증 macro F1 | 기존 CONSTANT/STRAIGHT | 이번 실험 |
|---|---:|---:|
| 가감속 | 0.180613 | 0.307496 |
| 주행 중 조향 | 0.221897 | 0.293618 |

학습 F1은 가감속0.604433/조향0.527938. 검증 조향은 모든 클래스를 예측하지만 주행 중 RIGHT 12147/17346(약70%)로 편향이 남는다. 공식 제출 점수가 아닌 외부 데이터 대리 검증이다. 다음 권장 단계는 균형 배치를 유지하며 학습 표본을 확대하고 동일 분리 경로에서 재검증하는 것. 제출 ZIP 변경 없음.

결과: `artifacts/kaggle-stage3-route-trial-20260910/result-validation.json` 및 `result/stage3-route-trial-result.zip`.
