[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $root 'src\stage2_pipeline.py'
$runbook = Join-Path $root 'docs\stage2-smoke-runbook.md'
$labels = Join-Path $root 'Baseline\data\stage2\labels.csv'

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

Assert-Condition (Test-Path -LiteralPath $source -PathType Leaf) 'Stage2 pipeline script is missing.'
Assert-Condition (Test-Path -LiteralPath $runbook -PathType Leaf) 'Stage2 smoke runbook is missing.'
Assert-Condition (Test-Path -LiteralPath $labels -PathType Leaf) 'Baseline Stage2 labels are missing.'

$text = Get-Content -Raw -LiteralPath $source
foreach ($token in @(
    'class Stage2Temporal',
    'def training_outputs',
    'F.cross_entropy(collision_logits',
    'F.cross_entropy(entry_logits',
    'scene_logits[:, :2]',
    'scene_logits[:, 2:]',
    'torch.save(backbone.state_dict()',
    'reloaded.load_state_dict',
    'OUTPUT_COLUMNS'
)) {
    Assert-Condition $text.Contains($token) "Pipeline contract token is missing: $token"
}

$rows = @(Import-Csv -LiteralPath $labels)
Assert-Condition ($rows.Count -gt 0) 'Baseline Stage2 labels are empty.'
foreach ($column in @('ID', 'path', 't_collision')) {
    Assert-Condition ($rows[0].PSObject.Properties.Name -contains $column) "Missing source column: $column"
}
foreach ($row in $rows) {
    $video = Join-Path (Join-Path $root 'Baseline\data\stage2') $row.path
    Assert-Condition (Test-Path -LiteralPath $video -PathType Leaf) "Missing Stage2 video: $video"
}

Write-Host 'PASS: Stage2 four-output pipeline static contract'
Write-Host "  Source samples: $($rows.Count)"
Write-Host '  Runtime smoke test: run docs/stage2-smoke-runbook.md on Kaggle GPU'
