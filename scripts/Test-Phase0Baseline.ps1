[CmdletBinding()]
param(
    [string]$BaselineDir,
    [string]$SubmitZip
)

$ErrorActionPreference = 'Stop'
if (-not $BaselineDir) { $BaselineDir = Join-Path $PSScriptRoot '..\Baseline' }
$baseline = (Resolve-Path -LiteralPath $BaselineDir).Path

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

$trainNotebook = Get-ChildItem -LiteralPath $baseline -Filter '*Train*.ipynb' | Select-Object -First 1
$inferenceNotebook = Get-ChildItem -LiteralPath $baseline -Filter '*Inference*.ipynb' | Select-Object -First 1
Assert-Condition ($null -ne $trainNotebook) '학습 노트북을 찾지 못했습니다.'
Assert-Condition ($null -ne $inferenceNotebook) '추론 노트북을 찾지 못했습니다.'

$trainJson = Get-Content -Raw -LiteralPath $trainNotebook.FullName | ConvertFrom-Json
$inferenceJson = Get-Content -Raw -LiteralPath $inferenceNotebook.FullName | ConvertFrom-Json
Assert-Condition ($null -ne $trainJson.cells) '학습 노트북 JSON이 올바르지 않습니다.'
Assert-Condition ($null -ne $inferenceJson.cells) '추론 노트북 JSON이 올바르지 않습니다.'

$requirements = Join-Path $baseline 'requirements.txt'
Assert-Condition (Test-Path -LiteralPath $requirements -PathType Leaf) 'requirements.txt가 없습니다.'

$expectedColumns = @{
    'stage1' = @('path', 'label')
    'stage2' = @('path', 't_collision')
    'stage3' = @('ID', 'frame_index', 'accel_label', 'steer_label')
}
foreach ($stage in $expectedColumns.Keys) {
    $csvPath = Join-Path $baseline "data\$stage\labels.csv"
    Assert-Condition (Test-Path -LiteralPath $csvPath -PathType Leaf) "$stage labels.csv가 없습니다."
    $rows = @(Import-Csv -LiteralPath $csvPath)
    Assert-Condition ($rows.Count -gt 0) "$stage labels.csv가 비어 있습니다."
    $columns = @($rows[0].PSObject.Properties.Name)
    foreach ($column in $expectedColumns[$stage]) {
        Assert-Condition ($columns -contains $column) "$stage labels.csv에 '$column' 컬럼이 없습니다."
    }
}

$stage1Videos = @(Get-ChildItem -File -LiteralPath (Join-Path $baseline 'data\stage1\original')) +
    @(Get-ChildItem -File -LiteralPath (Join-Path $baseline 'data\stage1\rerecorded'))
$stage2Videos = @(Get-ChildItem -File -LiteralPath (Join-Path $baseline 'data\stage2\videos'))
$stage3Videos = @(Get-ChildItem -File -LiteralPath (Join-Path $baseline 'data\stage3\videos'))
Assert-Condition ($stage1Videos.Count -gt 0) 'Stage1 예제 영상이 없습니다.'
Assert-Condition ($stage2Videos.Count -gt 0) 'Stage2 예제 영상이 없습니다.'
Assert-Condition ($stage3Videos.Count -gt 0) 'Stage3 예제 영상이 없습니다.'

$inferenceText = Get-Content -Raw -LiteralPath $inferenceNotebook.FullName
$markerCount = ([regex]::Matches($inferenceText, 'BASELINE_INFERENCE_PART')).Count
Assert-Condition ($markerCount -ge 3) '추론 코드 추출 마커가 충분하지 않습니다.'

if ($SubmitZip) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipPath = (Resolve-Path -LiteralPath $SubmitZip).Path
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
    try {
        $entries = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
        $required = @(
            'inference.py',
            'requirements.txt',
            'model/stage1/best.pt',
            'model/stage2/best.pt',
            'model/stage2/resnet18-f37072fd.pth',
            'model/stage3/best.pt'
        )
        foreach ($entry in $required) {
            Assert-Condition ($entries -contains $entry) "submit.zip에 '$entry'가 없습니다."
        }
    }
    finally {
        $archive.Dispose()
    }
}

Write-Host "PASS: Phase 0 정적 검증 완료"
Write-Host "  Stage1 videos: $($stage1Videos.Count)"
Write-Host "  Stage2 videos: $($stage2Videos.Count)"
Write-Host "  Stage3 videos: $($stage3Videos.Count)"
if ($SubmitZip) { Write-Host "  ZIP: $SubmitZip" }
