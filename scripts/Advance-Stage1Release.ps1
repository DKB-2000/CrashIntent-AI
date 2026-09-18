$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$logDir = Join-Path $projectRoot 'artifacts/stage1-auto-release-20260910/logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ((Get-Date -Format 'yyyy-MM-dd_HHmmss') + '.log')
& (Join-Path $projectRoot '.venv/Scripts/python.exe') (Join-Path $projectRoot 'src/advance_stage1_release.py') *> $logPath
if ($LASTEXITCODE -ne 0) { throw "Stage1 release tick needs attention: $logPath" }
