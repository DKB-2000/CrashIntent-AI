# Stage2 수동 라벨 74건 개발 평가 (2026-09-07)

CPU / torch 2.8.0+cpu / torchvision 0.23.0+cpu. ImageNet ResNet18 고정 특징 + BiGRU 네 출력 헤드, seed 20260825, 5에폭 최종 가중치 저장·재로딩 후 검증.

74건 중 보류 8건(000007, 000030, 000066, 000130, 000134, 000140, 000174, 000190)을 제외했다. CCD 출처 기준 학습 51건 / 검증 15건, 출처 중복 없음. 원본 라벨 변경 없음.

| 항목 | 결과 |
|---|---|
| 충돌 ±0.3초 | 6/15 (40.0%), MAE 0.587초 |
| 진입 ±0.3초 | 4/15 (26.7%), MAE 0.713초 |
| 방향 | 8/15 (53.3%), Macro-F1 0.444 |
| 회피 공간 | 3/15 (20.0%), Macro-F1 0.196 |

학습 다수 클래스 기준선: 방향 LEFT 60.0%, 회피 공간 있음 26.7%. 두 분류 항목 모두 기준선보다 낮다. 검증 표본이 작고 CCD 충돌 기준·수동 라벨의 불확실성이 남아 있다. 공식 대회 점수가 아니며, 기존 44건 실험과 분할이 달라 직접 성능 증감을 비교할 수 없다.

결과: artifacts/stage2-manual-200-20260907/ (audit.json, 원본 스냅샷, 분할 CSV, 모델, validation_details.csv, test_report.json).

재현: .\.venv\Scripts\python.exe src/test_stage2_manual.py --output-dir artifacts/stage2-manual-200-20260907 --exclude-id 000007 --epochs 5 --device cpu

재실행 시 새 출력 경로를 사용한다. 사전학습 가중치 출처는 DATA_SOURCES.md 참고. 제출 ZIP은 생성하지 않았다. 다음 단계는 보류 표본 재검토와 적합 라벨 200~300건 확보다.
