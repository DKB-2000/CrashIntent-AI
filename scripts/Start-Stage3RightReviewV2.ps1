$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
& (Join-Path $projectRoot '.venv/Scripts/python.exe') (Join-Path $projectRoot 'src/review_stage3_human.py') --video-dir (Join-Path $projectRoot 'artifacts/stage3-right-candidates-20260914/videos') --output-dir (Join-Path $projectRoot 'data/stage3-right-review-v2')
if ($LASTEXITCODE -ne 0) { throw "Stage3 review exited with $LASTEXITCODE" }
