# Stage3 표본 확대 실험

2026-09-10 사용자 승인으로 `biadis/crashintent-stage3-expanded-trial` version 1 실행, RUNNING 확인. `src/watch_stage3_expanded.py` PID13156 자동 회수·검증 감시기 시작. 중복 실행 금지.

- 학습: 기존17경로의 전체156영상에서 1000클립(10조합별100개, 중복 샘플 없음).
- scratch MViT, FP32, batch10/microbatch2, lr3e-5, weight decay0 및 dropout/stochastic depth 비활성 유지.
- 12에폭/1200업데이트, 이전200클립 실험보다 표본과 업데이트 모두5배. 표본 확대 효과만 분리하는 대조 실험은 아님.
- 검증: 동일31영상/4경로 전체, dense FP16. 마지막 체크포인트를 고정 평가하며 validation 점수에 따른 선택 없음.
- 비교 기준: 이전 가감속 F1 0.307496, 조향0.293618, 주행 중 RIGHT 비율 약70%. 외부 임시 라벨이므로 공식 제출 성능은 별도 확인 필요.
- 예상1.5~2시간, runner최대7200초. 학습 전처리1200초 제한, 학습 종료시점최대4800초, 마지막 검증시간 확보. 시간 내 목표 미완료면 PARTIAL.
- 메모리: uint8 클립 배열 사전 할당, 전체 복제 없이 해시 스트리밍.
- 사전 검증: 실제1000표본/100균형배치/경로 분리PASS, gradient누적2테스트PASS, 기존 결과 재검산PASS.

실행자료와 상태: `artifacts/kaggle-stage3-expanded-20260910/launch.json`, `remote-run.json`, `monitor.log`, `sample-plan.json`.
완료시 `result/stage3-route-trial-result.zip` 자동 다운로드 후 `result-validation.json`에 검사 결과 저장.
감시기는 결과 회수·검증만 하며 제출 ZIP 교체나 새 GPU 실행은 하지 않음.

개선 및 편향 감소 여부를 확인한 후 제출 후보 패키징과 GPU 통합검사 여부를 결정한다.


## 완료 결과

Kaggle version1 COMPLETE. 자동 다운로드 및 독립 F1 재계산 PASS. 12에폭/1200업데이트, 전체31검증영상18603행 완료. Runner5909.65초(98분30초).

- 검증 가감속 F1: 200클립0.307496 → 1000클립0.461828.
- 검증 주행 중 조향 F1: 0.293618 → 0.409353.
- 주행 중 예측: LEFT6038, STRAIGHT7961, RIGHT3347 / 총17346. RIGHT비율70.0%→19.3%; 실제RIGHT비율26.8%이므로 편향 전체 해소로 해석하지 않는다.
- 학습 F1: 가감속0.635836, 조향0.505335.
- 체크포인트SHA256: 90ecb3a7b0e578ab05cfa250b74df5b072559a11317f4a12464ee22409abf8dc.

기존 RUNNING 기록보다 본 완료 결과를 우선한다. 외부 대리 검증상 개선 확인. 다음 단계는 별도 제출 후보 ZIP 패키징 및 GPU 통합검사. 아직 제출 ZIP 변경 없음.
