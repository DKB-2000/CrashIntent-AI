# Stage1 실제 화면 재촬영 공개 데이터 확보 (2026-09-16)

사람 라벨 없이 생성·수집 메타데이터로 ORIGINAL/RERECORDED를 확정할 수 있는 공개 자료를 조사했다. RECOD-MPAD, Replay-Mobile, CoC는 모두 접근 요청 또는 계정 승인이 필요한 제한 자료라 즉시 사용하지 않는다. DLC-2021은 공개 CC BY-SA 2.5이며 합성 신분증을 사용해 개인정보 위험이 낮고, 원본과 실제 화면 재촬영 유형이 명시돼 있어 1순위로 선택했다.

공식 출처는 DLC-2021 논문과 Zenodo part1/part2, 원저자 FTP다. part2 화면 재촬영 ZIP은 38.5GB, part1 원본·회색복사 ZIP은 33.9GB다. 원저자 FTP는 모든 유형에서 추출한 프레임 `clips.tar` 17,768,312,320bytes와 원본 영상 `clips_video.tar` 88,328,994,816bytes를 제공한다. 이번에는 Stage1 파일 전체 학습 전 artifact encoder 소규모 실험에 필요한 프레임 묶음만 선택했다.

메타데이터 1,424행은 cc484/cg250/or290/re400클립이다. 화면 재촬영400클립은 iPhone200/Android200, 화면 Lenovo40/Philips160/MacBook Pro160/HP40, 문서종류10·템플릿73이다. 공식 baseline 목록은 화면 재촬영 학습 양성19,543장/원본 음성25,980장, 테스트 양성15,346장/음성16,264장이다. 라벨과 분할은 제공 목록에서 자동으로 읽으며 사람이 내용을 판정하지 않는다.

확보한 소형 파일은 `artifacts/stage1-physical-recapture-inventory-20260916/`의 `dlc-2021.csv`, `license.txt`, `README-part2.md`, `experimental_baseline.zip`, 추출된 baseline 목록과 `inventory.json`이다. Zenodo baseline 88MB 다운로드는 0바이트 정체로 중단했고, 원저자 FTP의 동일 목적 공식 baseline 887,710bytes를 받아 MD5 `D444201C7239C654A6AFAD7606C7EA46`가 FTP `md5.txt`와 일치했다.

## 부분 TAR 파일럿 (2026-09-17)

17,768,312,320바이트 전체 TAR 확보는 8시간 제한에서 실패했으며 193,638,340바이트까지만 받았다. TAR은 무압축이고 앞부분은 온전하므로 이를 폐기하지 않고 자동 복구했다. 완전한 JPEG 669개 중 Stage1에 직접 대응하는 `or` 2클립과 `re` 4클립, 총 338프레임을 MP4 6개로 변환하고 전 파일을 재디코딩·해시 검증했다. 상태와 manifest는 `artifacts/stage1-dlc-partial-pilot-20260917/`에 있다. 라벨은 픽셀 검토가 아니라 DLC-2021 공식 파일명 유형 코드(`or`=원본, `re`=화면 재촬영)에서 정해진다.

이 표본은 한 문서 종류·템플릿에 묶인 상관 표본이므로 모델 선택이나 임계값 조정에 사용하지 않는다. 기존 공식 최고 모델과 `screen_mix_25`의 실제 화면 재촬영 방향성이 맞는지 보는 초기 진단만 수행한다. 결과는 `artifacts/stage1-dlc-pilot-evaluation-20260917/`에 기록한다.

진단은 `COMPLETE_VALIDATED`로 완료됐다. 공식 최고 모델은 재촬영 4/4를 맞혔지만 원본 0/2로, 원본의 평균 재촬영 확률이 0.98810이었다. `screen_mix_25`는 재촬영 4/4를 유지하고 원본 1/2를 맞혔으며 평균 재촬영 확률은 원본 0.43677, 재촬영 0.69308이었다. 따라서 합성 화면 보강이 이 파일럿에서는 과도한 양성 편향을 크게 줄인 방향성은 확인됐다. 다만 6클립·단일 템플릿 결과이고 공식 점수는 기존 최고보다 낮았으므로 모델·ZIP·임계값을 변경하지 않는다. 다음 실험에는 서로 다른 문서 템플릿의 자동 라벨 표본이 필요하다.

## 다문서 Range 파일럿 결과 (2026-09-17)

Zenodo 공식 split ZIP의 중앙 디렉터리와 선택 항목만 HTTP Range로 받아 전체 57.3GB를 내려받지 않았다. `aze_passport`, `esp_id`, `fin_id`, `svk_id`에서 각각 `00.or0001`과 `00.re0001`을 확보했다. 총 8클립은 ZIP CRC·비압축 크기, MP4 재디코딩, SHA-256 및 공식 파일명 자동 라벨 검증을 통과했고, 파이프라인 상태는 `COMPLETE_VALIDATED`다.

공식 최고 모델은 재촬영 4/4, 원본 0/4였고 평균 재촬영 확률은 각각 0.99921, 0.92721이었다. `screen_mix_25`는 재촬영 4/4, 원본 1/4였고 평균 확률은 각각 0.94992, 0.73124였다. 원본 중 `esp_id`만 `screen_mix_25`가 0.26794로 맞혔으며 나머지 세 원본은 0.85 이상이었다. 첫 파일럿과 같은 방향으로 합성 보강이 양성 편향을 줄였지만, 실제 원본 일반화는 여전히 부족하다. 공식 점수가 더 낮았던 `screen_mix_25`를 채택하지 않고 기존 모델·ZIP·임계값을 유지한다. 결과는 `artifacts/stage1-dlc-range-pilot-evaluation-20260917/report.json`과 `predictions.json`에 있다.

`src/download_stage1_dlc_frames.py`가 17.77GB `clips.tar`를 원저자 FTP에서 내려받는 중이다. 중복 잠금, 이어받기, 최대3회, 전체8시간, 공식 크기17,768,312,320bytes, MD5 `0758a65d3ddccdd24eba25a98e4ba3c6`, tar 전체목록 검사를 적용한다. 상태는 `artifacts/stage1-dlc-frames-acquisition-20260916/status.json`, 원본은 `data_raw/dlc-2021/clips.tar`다. 최초 프로세스와 파일 증가를 확인했으며 아직 완료가 아니다. 완료 후 baseline 목록에서 출처·기기·화면 분리 표본만 추출하고, 기존 Stage1 모델 평가와 artifact encoder 소규모 대조를 준비한다. 기존 모델·ZIP·selected는 변경하지 않는다.
