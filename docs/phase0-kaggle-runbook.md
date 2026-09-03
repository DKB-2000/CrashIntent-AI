# Phase 0 — Kaggle Baseline 실행 가이드

목표는 공식 학습 노트북과 추론 노트북을 변경하지 않고 순서대로 실행해 `submit.zip` 생성과 공개 예제 스모크 테스트를 확인하는 것이다.
실제 데이콘 제출은 이 단계에서 하지 않는다.

## 1. 로컬 사전 검증

프로젝트 루트의 PowerShell에서 실행한다.

```powershell
.\scripts\Test-Phase0Baseline.ps1
```

`PASS`가 출력돼야 한다. 이 검사는 노트북 JSON, requirements, CSV 컬럼, Stage별 예제 영상 존재 여부를 확인한다.

## 2. Kaggle 업로드

1. `Baseline/` 폴더 전체를 비공개 Kaggle Dataset으로 업로드한다.
2. 새 Kaggle Notebook에 해당 Dataset을 Input으로 연결한다.
3. Accelerator는 GPU를 선택한다. T4 한 장이면 충분하며 두 장을 요구하지 않는다.
4. 학습 중 ResNet18 ImageNet 가중치를 처음 내려받아야 하므로 Internet을 켠다. 생성된 제출물의 추론 코드는 외부 다운로드 없이 동작해야 한다.

Kaggle Input은 읽기 전용이므로 첫 셀에서 Baseline 전체를 작업 디렉터리로 복사한다. 아래의 `<dataset-slug>`는 실제 Input 폴더명으로 바꾼다.

```python
from pathlib import Path
import shutil

source = Path('/kaggle/input/<dataset-slug>')
work = Path('/kaggle/working/Baseline')
if work.exists():
    shutil.rmtree(work)
shutil.copytree(source, work)
%cd /kaggle/working/Baseline
```

업로드 방식에 따라 `source` 바로 아래에 `Baseline/`이 한 번 더 들어 있으면 `source = source / 'Baseline'`으로 조정한다.

## 3. 실행 순서

1. `[Baseline_Train]_3Stage_학습.ipynb`의 셀을 위에서 아래로 전부 실행한다.
2. 아래 모델 파일 네 개가 생성됐는지 확인한다.

```python
required = [
    'model/stage1/best.pt',
    'model/stage2/best.pt',
    'model/stage2/resnet18-f37072fd.pth',
    'model/stage3/best.pt',
]
for path in required:
    p = Path(path)
    print(path, p.exists(), p.stat().st_size if p.exists() else None)
assert all(Path(path).is_file() for path in required)
```

3. 같은 `/kaggle/working/Baseline`에서 `[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb`의 셀을 전부 실행한다.
4. Stage 1/2/3 공개 예제 예측이 모두 끝나고 `submit.zip` 최종 검사가 통과하는지 확인한다.
5. Kaggle Output에서 `submit.zip`과 실행 로그를 다운로드한다.

## 4. 로컬 사후 검증

다운로드한 ZIP 경로를 지정한다.

```powershell
.\scripts\Test-Phase0Baseline.ps1 -SubmitZip '.\Baseline\submit.zip'
```

필수 파일 여섯 개가 정확한 경로에 들어 있으면 `PASS`가 출력된다. ZIP은 검증 기준선으로 보관하되 데이콘에는 아직 제출하지 않는다.

## 실패 시 기록할 정보

- 실패한 노트북과 셀 번호
- 오류 전문
- Kaggle GPU 종류와 Python/torch/torchvision 버전
- `model/` 아래 생성된 파일 목록과 크기
- 추론 스모크 테스트 중 어느 Stage에서 실패했는지
