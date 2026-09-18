$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$logDir = Join-Path $projectRoot 'artifacts/stage1-robustness-20260910/full-model-evaluation/supervisor-logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ((Get-Date -Format 'yyyy-MM-dd_HHmmss') + '.log')
& (Join-Path $projectRoot '.venv/Scripts/python.exe') (Join-Path $projectRoot 'src/supervise_stage1_robustness.py') *> $logPath
if ($LASTEXITCODE -ne 0) { throw "Local Stage1 supervision needs attention: $logPath" }
