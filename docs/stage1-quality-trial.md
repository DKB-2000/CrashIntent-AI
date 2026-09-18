# Stage1 화질 균형 소규모 대조 학습

사용자가 510개 생성·검증 완료 후 2번 대조 학습 진행 승인(2026-09-11).
현재 공식 모델 Stage1 0.6386345895(제출87406), 체크포인트92b10c99...를 시작점으로 고정.
데이터 생성 DATA_VALIDATED:85출처510영상25500프레임, manifest d29ab2c5... .
이번 결과로 제출 모델을 자동 교체하지 않는다.

## 사전 고정 비교

- baseline: 현재 모델 그대로 평가.
- existing_recipe: 같은85개 학습 원본의 기존 생성255영상(원본85/재녹화170)으로 추가학습.
- matched_quality: 새510영상(기본/노이즈/복합화질,각클래스255)으로 추가학습.

두 학습 조건은 같은시작가중치·seed20260911·128업데이트·학습률1e-5.
FP32 AdamW, weight_decay0, dropout/stochastic-depth0,gradient clip1.
유효배치8(원본4/재녹화4),microbatch1,매업데이트출처4개복원추출.
같은출처·슬롯일정 공유,클래스쌍에동일슬롯. 최종모델만평가하고검증점수로에폭선택하지않음.
기존조건의재녹화v1/v2는profile인덱스로분배한다. 새데이터는화면합성세부도다르므로
두학습조건차이를노이즈한요인만의인과효과로단정하지않는다.

## 평가와 해석

기존357조건진단영상+같은21검증출처원본에대응하는기존합성63영상=420개.
3모델모두같은제출decoder,16프레임×3slot,CUDA FP16softmax평균,임계값0.5.
총1260영상예측/3780slot. 새모델256updates. 기존690검증전체평가가아닌소규모표본이다.
진단21출처는이미약점을분석한개발셋이다. phone검증/새출처블라인드검증이아니다.

파일생성이끝났다는PASS와성능개선판정을분리한다.
사전조건:noise/quality_mix원본오류각각감소,각재녹화조건손실최대1/21,
normal63개Macro-F1하락최대0.02. 통과해도단일seed개발파일럿후보일뿐자동승격없음.
현재모델·기존추가학습·새추가학습조건별표를모두보관하고실험끝에해석한다.

## 실행·회수

src/prepare_stage1_quality_trial.py가자료/출처/해시를재검증하고비공개GPU번들생성.
artifacts/kaggle-stage1-quality-trial-20260911/{bundle.json,plan.json,manifest.json}.
목적지비공개 biadis/crashintent-stage1-quality-trial-assets,
무료T4노트북 biadis/crashintent-stage1-quality-trial.
전송자료는기존CCD학습·검증영상1185개,Stage1시작모델,실험/추론코드/설정이다.
크기는bundle.json의실측값을따른다. 외부전송/실행여부는remote-run.json확인.
노트북패키지설치600초,runner10200초제한. 업로드/노트북push는무작정재시도하지않음.
watch_stage1_quality_trial.py는ready대기1시간/결과대기4시간,조회오류5회/다운로드5회제한.
결과회수후validate_stage1_quality_trial.py로입력·코드·학습schedule/128step·가중치·
1260예측전체softmax/평균/판정/지표를독립재검산한다. 모든모델출력을CPU에서다시추론하는검사는아님.
원격오류도결과ZIP을회수해보존. 현재모델/selected/기존ZIP예약은변경하지않는다.

## 준비 완료·구체적 전송 승인 대기

1545566290bytes(약1.55GB), SHA256 d31d0d1ea076b5f1bef718e96ebccd72b6327171b20785ac393da40ac4b1bff6.
3테스트PASS(동일학습일정/출처누수·표본구성/예측검산오류차단), 모든실행코드구문검사PASS.
학습765영상(기존255+새510),검증420영상,시작Stage1모델과코드포함.
자동승인검토가해당payload와새Kaggle목적지에대한구체적사용자승인없음을이유로
외부전송·원격실행을거절했다. 업로드/GPU미실행,remote-run.json BLOCKED_APPROVAL.
launch_stage1_quality_trial.py는준비된payload SHA와dataset에일치하는
upload-approval.json의approved=true가있어야실행한다. 승인파일은아직없다.
승인후에만업로드→ready대기→1회push→회수→독립검증을시작한다.

## 사용자 승인 후 업로드·GPU 자동 실행 연결 (2026-09-11)

사용자가 구체적1.55GB/새비공개목적지/무료T4 확인질문에 GPU학습진행 승인.
upload-approval.json에dataset/payload SHA일치승인기록,원본bundle해시재확인PASS.
launch_stage1_quality_trial.py 숨김실행. 업로드→ready대기→단일GPU push→결과회수·
독립검증을한감시기에연결. 이전전송차단은해소됨. 실제GPU상태는remote-run.json을따른다.
최초launcher PID35332,중복실행금지. 현재모델/ZIP/12시간예약변경없음.

## 기존 실험 결과 회수 재개 (2026-09-14)

사용자 1순위 진행 요청으로 기존 kernel version 1의 Kaggle 상태를 직접 조회해
`KernelWorkerStatus.COMPLETE`를 확인했다. 9월 11일의 로컬 RUNNING 기록은 오래된 기록이다.
기존 PID 28752/35332는 로컬 프로세스 조회에서 발견되지 않았다.
`watch_stage1_quality_trial.py`를 `--launch` 없이 실행해 결과 다운로드와 독립 검증을 재개했다.
새 업로드나 GPU 학습은 실행하지 않았다. 최초 재개 PID는 5736이며
`remote-run.json`의 `COMPLETE / DOWNLOADING` 전환을 확인했다.

GPU 완료와 결과 검증 완료는 구분한다. 검증 완료 증거는
`artifacts/kaggle-stage1-quality-trial-20260911/validated/result-validation.json`의 PASS 및
`remote-run.json`의 `monitor_status=VALIDATED`다. 회수·검증은 기존 유한 감시기에 맡기며
중복 실행하지 않는다. 상태의 기존 `last_error`와 `remote_response`는 과거 기록이 남아 있을 수 있다.
현재 제출 모델·ZIP·기존 예약은 변경하지 않았다.

## 결과 회수·독립 검증 완료 (2026-09-14)

기존 GPU 실행은 2195.59초(약36분36초), Tesla T4에서 3조건을 완료했다.
결과 ZIP SHA256: `f71ba375ca696ebbbca87129a90da1496643d239cc9a1bf819bbdde155b03b84`.
1260예측·256업데이트·코드/입력/일정/모델해시·유한가중치·저장로짓 재계산 PASS.
검증기는 요약 평균의 최대2.22e-16 부동소수점 차이를 완전일치로 비교해 실패했다.
평균확률만 abs_tol1e-12/rel_tol0으로 비교하고 건수·오류율·조건/필드는 엄격 비교를 유지했다.
반올림 허용 및 실제 손상 거절을 포함한 테스트4개 PASS. GPU 재학습은 하지 않았다.
기존 실패폴더와 remote-run-before-validation-recovery-20260914.json을 보존했다.

| 항목 | baseline | existing_recipe | matched_quality |
|---|---:|---:|---:|
| 노이즈 원본 오탐 /21 | 19 | 19 | 0 |
| 복합화질 원본 오탐 /21 | 18 | 20 | 2 |
| 화면 단서 약화 재녹화 오류 /21 | 0 | 0 | 6 |
| normal 63영상 Macro-F1 | 1.000000 | 1.000000 | 0.982348 |

matched_quality는 no_geometry/no_noise 재녹화에서도 오류가 각각0→2/21로 증가했다.
원본 오탐과 normal F1 기준은 통과하지만 재녹화 조건별 오류증가 최대1/21 기준은 실패했다.
existing_recipe는 원본 오탐 개선 기준 실패. 두 조건 모두 promising_pilot=false이며
현 공식 최고 Stage1 .6386345895 모델을 유지한다. 이는 단일seed 개발셋 결과로,
실제 휴대전화 재촬영 또는 공식 점수 개선을 검증한 것이 아니다.

최종 검증: `artifacts/kaggle-stage1-quality-trial-20260911/validated-recovery-20260914/result-validation.json`.
remote-run.json은 COMPLETE/VALIDATED 및 위 검증경로로 갱신했다.
현재 제출 모델·ZIP·예약 변경 없음. 앞선 회수 중/검증 실패 기록보다 이 완료 결과를 우선한다.
