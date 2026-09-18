# Stage3 Civic 저속 보정 경로 탐색 (2026-09-16)

기존 Civic 보정5경로와 평가5경로를 제외한 공식 comma2k19 Chunk_3의 남은11경로를 예약했다. 고정 revision `4bff77c7254c654c28d4c2726186b4e825adccee`의 ZIP 중앙목록을 부분 조회해 각 경로에서 최대2구간, 총22구간을 선정했다. 선정은 속도 값을 읽기 전에 `artifacts/stage3-civic-low-speed-scout-20260916/plan.json`에 고정했다.

`src/scout_stage3_civic_low_speed.py watch`로 2~5m/s 주행 시점·연속길이를 **프레임 시각과 CAN 속도만** 받아 확인하는 감시기를 시작했다. 영상·자이로·조향·모델 예측·라벨은 받거나 사용하지 않는다. 첫 실행 때 watcher와 자식 Python 프로세스, 첫 구간 로그를 실제로 확인했다. 감시 제한30분, 전송상한1GiB, 범위당 유한3회, 작업 재시도0, OS 잠금·중복실행 거절이다. Codex 반복 폴링은 하지 않는다.

## 센서 탐색 완료

22/22구간 `COMPLETE_VALIDATED`, watcher `COMPLETE`. 실제 읽은 네트워크 자료는2,612,200bytes다. 센서 시각·속도 값 유한성 및 정렬, 기존 경로와 중복0, 재계산한 목록 SHA256 일치를 확인했다. 2~5m/s가 최소100시점이고 연속10시점 이상인 서로 다른 기록 경로5개를 찾았다. 경로별 표본은225/205/194/189/109시점이다. 결과는 `artifacts/stage3-civic-low-speed-scout-20260916/report.json`과 `segment-speed-inventory.csv`에 있다. 이는 속도 센서 확보 결과이며 영상·전체센서 확보나 라벨 검증이 아니다.

## 저속 원본 확보 착수

탐색 점수 상위3경로를 **보정**, 다음2경로를 **신규 평가**로 속도 센서 조사 후, 조향·자세·자이로 값이나 모델 예측을 읽기 전에 예약했다. 기존 Civic 보정/평가10경로와 겹침0, 각 경로2구간 총10구간110파일, 예상 압축 자료368,729,696bytes다. 계획은 `artifacts/stage3-civic-low-speed-acquisition-20260916/plan.json`에 고정됐다.

## 저속 원본 확보·보정 검증 완료

10/10구간110파일 원본 확보 완료. 감시 `COMPLETE`, `validation.json` `PASS`, `status.json` `ACQUIRED_VALIDATED_UNCALIBRATED`. 실제 수신368,744,393bytes, 전10구간 CAN 범위 합격, 저장110파일 SHA256 독립 재확인 불일치0이다. 영상 전체 디코딩·센서시각 검증은 획득 watcher가 수행했다.

`src/validate_stage3_civic_low_speed_proxy.py`로 **보정3경로6영상만** 원본 신호를 분석했다. 2~5m/s 유효624시점에서 고정 3센서·11시점 후보 라벨은 좌31/우27/직진0이었다. 경로별 뚜렷한 저속 회전 수, 다른 영상의 자이로 편향 전송 일치율97%·고속 오차0.005rad/s 이하, 각 클래스30시점·경로별 두 클래스 조건을 함께 요구하는 사전 gate가 실패했다. 한 영상의 저속 자이로·자세 방향 일치는92.95%, 다른 영상의 고속 편향 전송 오차는0.00774rad/s였다. 결과는 `artifacts/stage3-civic-low-speed-proxy-20260916/calibration-review.json`에 있다. 저장 CSV SHA256 재확인PASS. **예약된 평가2경로의 원본 신호를 읽거나 저속 평가 라벨을 생성하지 않았다.** 기존 모델·제출ZIP·기존5~35m/s 규칙을 유지한다.

다음은 속도표본 수만으로 경로를 선택하는 방식을 중단하고, 남은 Chunk3 경로를 영상·모델 예측 없이 센서로 먼저 조사해 저속 좌/직진/우 구간과 다른 영상 간 자이로 편향 전송이 함께 확보되는지 확인하는 것이다.

## 남은 경로의 클래스 균형 센서 조사 시작

기존 보정·평가 및 방금 원본 확보한5경로를 제외한 Chunk3의 남은6경로12구간을 `artifacts/stage3-civic-low-speed-class-scout-20260916/plan.json`에 예약했다. 실패한 저속 보정 실험의 각도·자세·자이로 임계값과 11시점 지속 조건을 그대로 적용하며, 다른 영상의 고속 센서로 편향을 구하는 방식도 고정했다. 영상·모델 예측·예약 평가 경로 신호는 사용하지 않는다.

`src/scout_stage3_civic_low_speed_classes.py watch`를 30분 제한·1GiB 상한·재시도0·OS 잠금의 로컬 감시기로 시작했다. 최초 watcher·자식 Python 프로세스와 실행 마커를 확인했다. 결과는 아직 없으며 중복 실행하지 않는다. 로컬 한 번 조회는 `./scripts/Get-BackgroundWorkStatus.ps1`의 `Stage3CivicLowSpeedClassScout`다. 완료 조건은 같은 출력 폴더의 `watch-status.json` `COMPLETE`와 `report.json` `COMPLETE_VALIDATED`이다. 후보가 없으면 저속 자동 정답 확대를 주장하지 않고 남은 자료의 분포 한계를 기록한다.

독립 로컬 RAV4 학습17경로 센서 진단은 `artifacts/stage3-low-speed-sensor-agreement-20260916/report.json`에 완료됐다. 같은 고정 조향각·자세 강회전 조건에서 2~5m/s **715/787=90.85%**, 5~35m/s **30,294/30,405=99.63%** 방향 일치였다. 저속 부호 불일치가 커 기존 고속 임계값을 저속으로 이전하지 않는다. RAV4와 Civic은 차량이 다르며 이 센서 일치는 공식 정답 정확도도 아니다.

탐색이 완료되면 충분한 2~5m/s 연속 구간을 가진 **서로 다른 기록 경로**를 보정용과 신규 평가용으로 먼저 예약한다. 이후 속도·조향·자세·자이로·영상의 시간 대응을 검증한 뒤 보정 경로에서만 저속 부분라벨 규칙을 고정한다. 기존 Stage3 모델·11시점 라벨·제출 ZIP은 유지한다.
