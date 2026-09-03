[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $root 'src\prepare_stage3_comma2k19.py'

function Assert-Condition {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

Assert-Condition (Test-Path -LiteralPath $source -PathType Leaf) 'Stage3 converter is missing.'
$text = Get-Content -Raw -LiteralPath $source
foreach ($token in @(
    'global_pose/frame_times',
    'processed_log/CAN/speed/t',
    'processed_log/CAN/steering_angle/t',
    'target_times = frame_times[::stride]',
    'ACCELERATING',
    'DECELERATING',
    'CONSTANT',
    'STOPPED',
    'LEFT',
    'STRAIGHT',
    'RIGHT',
    'TRAIN_COLUMNS',
    'conversion_report.json',
    'overlay frame mismatch'
)) {
    Assert-Condition $text.Contains($token) "Converter contract token is missing: $token"
}

Write-Host 'PASS: comma2k19 Stage3 converter static contract'
Write-Host '  Runtime smoke test: run the command in docs/stage3-comma2k19-runbook.md on Kaggle'
