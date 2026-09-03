[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $root 'src\stage1_pipeline.py'

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

Assert-Condition (Test-Path -LiteralPath $source -PathType Leaf) 'Stage1 pipeline is missing.'
$text = Get-Content -Raw -LiteralPath $source
foreach ($token in @(
    'class Stage1MViT',
    'mvit_v2_s',
    'self.net.head[1]',
    'ORIGINAL',
    'RERECORDED',
    'torch.save(',
    'model.net.state_dict()',
    'reloaded.net.load_state_dict',
    'rerecorded_probability >= args.threshold',
    'OUTPUT_COLUMNS = ["ID", "answer"]',
    'stage1_predictions.csv'
)) {
    Assert-Condition $text.Contains($token) "Pipeline contract token is missing: $token"
}

Write-Host 'PASS: Stage1 MViTv2-S pipeline static contract'
Write-Host '  Runtime smoke test: run docs/stage1-pipeline-runbook.md on Kaggle GPU'
