# Stage2 사전학습 개선 모델 제출 후보 (2026-09-11)

> 현재 상태 (2026-09-11): ZIP 생성·CPU 검사 완료. 이후 공식 점수는 하락해 현재 기준 모델로 채택하지 않는다. [진행 계획](stage2-next-steps.md) 참고.

사용자 요청으로 Stage2 개선 결과를 반영한 ZIP을 생성한다.
기준은 사용자가 제출86692에 사용했다고 확인한
artifacts/stage3-submit-candidate-20260910/submit.zip이다.
기준 SHA256 cec269d3975db3d55a734505e40a3c03d8c43775e02fc887c2cfeba132964337.
Stage1과Stage3는 해당 기준 ZIP 그대로이며 다른 세션의 신규 후보를 자동 통합하지 않는다.

개선 근거: 5분할 Stage2 CCD 사전학습 대조 .4318→.5000, 독립검증PASS.
이는 최종 단일 모델의 독립 성능이나 공식점수가 아니다.
최종 모델은1296영상 충돌3에폭 사전학습에서 시작해 수동66영상/23출처 전체로
soft_mixed15에폭,lr2e-4,seed20260825,990업데이트 학습한다.
전체 라벨을 사용하므로 최종 모델에 별도 독립 검증 세트는 없다.
교차검증 최적 fold를 고르는 방식이 아니라 고정 설정으로 단일 모델을 재학습한다.

재현 src/package_stage2_pretrained_candidate.py.
출력 artifacts/stage2-pretrained-submit-candidate-20260911/.
상태 status.json, provenance.json에 입력과 코드 해시·학습 설정 기록.
학습 종료 시 best.pt,training-history.json,training-predictions.csv 및submit.zip 생성.
validation.json에서 STATIC_AND_CPU_PARITY_PASS 확인 전 완성 ZIP으로 취급하지 않는다.

검사: 전체66캐시에서 실제ZIP 모델클래스와 출력 완전일치,strict로딩·유한가중치,
ResNet 가중치 텐서 일치,필수6파일·함수계약·CRC·크기 제한,
Stage2 best.pt 외5파일 바이트 불변. 기존 .pt는 보존하고 별도 폴더에 생성.
이 신규 가중치에 대한 GPU 통합검사는 아직 미실행. 실제 대회 제출 및 배포selected 변경 없음.

## ZIP 생성 완료 (2026-09-11)

15에폭/990업데이트 학습 및 ZIP생성431.27초 완료.
validation.json STATIC_AND_CPU_PARITY_PASS, status ZIP_READY.
ZIP303600234bytes(약303.6MB).
SHA25646db5bde8d42ee22b1582ab3869ea249721571e31aeb0367fa94e7978e697003.
기준ZIP과 비교해 model/stage2/best.pt 하나만 변경됐음을 전수 바이트 비교 확인.
실제 제출 클래스와66영상 출력 완전일치,backbone 텐서일치,CRC·구조·함수계약PASS.
최종 파일 artifacts/stage2-pretrained-submit-candidate-20260911/submit.zip.
새 체크포인트의 GPU 검사는 아직 미실행. 실제 대회 제출 및selected 변경 없음.
