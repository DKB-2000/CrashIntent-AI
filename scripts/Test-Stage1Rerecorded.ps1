[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $root 'src\prepare_stage1_rerecorded.py'

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

Assert-Condition (Test-Path -LiteralPath $source -PathType Leaf) 'Stage1 generator is missing.'
$text = Get-Content -Raw -LiteralPath $source
foreach ($token in @(
    'class H264Writer',
    'bgr24',
    'libx264',
    'def _stable_fraction',
    'def _synthetic_frame',
    'moire_amplitude',
    'flicker_amplitude',
    'band_amplitude',
    'perspective_x',
    'temporal_jitter_probability',
    'f"labels_{split}.csv"',
    'generation_manifest.csv',
    'source_splits.csv',
    'source leakage detected'
)) {
    Assert-Condition $text.Contains($token) "Generator contract token is missing: $token"
}

Write-Host 'PASS: Stage1 rerecorded generator static contract'
Write-Host '  Runtime smoke test: run docs/stage1-rerecorded-runbook.md on Kaggle'
