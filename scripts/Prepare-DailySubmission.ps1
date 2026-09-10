$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$logDir = Join-Path $projectRoot 'artifacts/daily-submissions/logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logPath = Join-Path $logDir ((Get-Date -Format 'yyyy-MM-dd_HHmmss') + '.log')
& (Join-Path $projectRoot '.venv/Scripts/python.exe') (Join-Path $projectRoot 'src/daily_submission.py') publish *> $logPath
if ($LASTEXITCODE -ne 0) { throw "Daily ZIP preparation failed. See $logPath" }
