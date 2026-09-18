# Stage3 Civic 소규모 원본 확보 (2026-09-14)

## 확보 완료

최종 상태 ACQUIRED_VALIDATED_UNCALIBRATED, 감시 COMPLETE, validation.json PASS.
5경로10구간/110파일, 실제 부분수신377,073,634bytes. 전영상12,000프레임 디코딩과
프레임 시각 개수 일치, 전파일 CRC32·길이·저장 SHA256 재검증을 통과했다.
10구간 모두 CAN 범위 기준 통과. 각 구간10Hz 표본의 약0.167%는 범위 밖이므로
후속 라벨·평가 구성 시 경계 제외 처리가 필요하다. 센서 부호·영점 보정과 라벨 생성은 미완료다.
아래 착수 기록보다 이 완료 결과를 우선하며 다운로드를 재실행하지 않는다.

사용자 자료 다운로드 승인에 따라 공식 comma.ai 배포 Chunk_3에서 5경로×2구간,
총10구간의 영상과 센서110파일을 확보하는 감시기를 시작했다.
압축 전송 예상량377,058,948bytes(약377MB). 전체9.41GB ZIP 다운로드가 아닌 HTTP Range 추출이다.
최종 확보 여부는 `artifacts/stage3-civic-acquisition-20260914/status.json`을 따른다.

## 출처와 선정

- 원본: https://huggingface.co/datasets/commaai/comma2k19/tree/main/raw_data
- 구조·차량 설명: https://github.com/commaai/comma2k19#dataset-structure
- 공식 설명에서 Chunk1–2는 RAV4, Chunk3–10은 Civic. 새 장치ID는 `99c94dc769b5d96e`.
- 다운로드 revision: `4bff77c7254c654c28d4c2726186b4e825adccee`.
- 전체 ZIP의 공개 LFS SHA256: `09f959b9ea2d98087ba84e8d356c7833bbc0a8b39de7f1878cf592caa48de636`.
  부분 추출이므로 전체 ZIP 해시 검증을 수행했다고 주장하지 않는다.

기존 Chunk_1의 장치ID·기록경로와 겹치는 후보를 제외한 뒤 seed20260914로
5경로와 각2구간을 선정했다. 예측·센서 수치를 보기 전에 plan.json으로 고정했다.
5경로 날짜는 05-01,05-05,05-12,05-13,06-11이다. 세부 구간은 plan에 보존한다.
같은 고속도로 데이터셋이므로 지리적으로 독립된 도시 주행 검증셋은 아니다.

구간당 영상, 프레임 시각, CAN 속도/조향 값과 시각, 자세·위치·속도, IMU gyro 값/시각을 받는다.
Windows에서 사용할 수 없는 원본 경로의 `|`는 로컬 폴더명에 사용하지 않고
`CIVIC_PILOT_000`부터009까지 매핑하여 원래 segment를 메타데이터에 보존한다.

## 자동 감시·검증

`src/download_stage3_civic_pilot.py`의 prepare→watch/run을 사용한다.
원격206 응답·Content-Range·길이를 검사하며 범위를 무시한 전체응답은 수신하지 않는다.
다운로드 범위당 최대3회, 전체 작업1시간 제한, 전송예산1GiB, OS 중복 잠금,
숨김 프로세스와 파일 로그를 사용한다. 시간 초과 시 Windows 자식 트리를 종료한다.
파일은 CRC32·길이를 확인한 뒤 원자적으로 저장하고 SHA256을 기록한다.
기존 파일은 크기·CRC가 일치할 때만 재사용하며 완료 작업은 중복 실행하지 않는다.

매 구간 영상 전체를 디코딩해 프레임 시각 배열 길이와 맞추고, 센서 값의 유한성·
시간 순서·CAN 범위 밖 비율을 기록한다. 완료 시110파일을 재읽어 SHA256·CRC·길이를 대조한다.
CAN 범위 합격 여부는 별도 필드이며 저장 성공과 라벨 유효성을 혼동하지 않는다.
조향 영점·부호·단위의 의미 검증과 라벨 생성은 다음 작업이다.

## 위치와 다음 확인

- 원본: `data_raw/comma2k19/civic-pilot-20260914/`
- 계획/상태/로그/검증: `artifacts/stage3-civic-acquisition-20260914/`
- 한 번 상태 조회: `scripts/Get-BackgroundWorkStatus.ps1`의 Stage3CivicAcquisition
- 최종 상태: `ACQUIRED_VALIDATED_UNCALIBRATED`; 실패 시 FAILED와 run.log 확인

평가 후보 확보 용도이며 아직 학습·예측·확정 조향 라벨은 없다. 센서 정합성·보정 검토 후
보정용과 평가용을 구분하고 고정 모델을 평가한다. 사람 검수 보류는 유지한다.
