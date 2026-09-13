param([switch]$Check)
$taskRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $taskRoot
$taskArgs = @('src/review_stage3_human.py', '--video-dir', 'artifacts/stage3-local-review-20260913/videos', '--output-dir', 'data/stage3-human-review')
if ($Check) { $taskArgs += '--check' }
& "$taskRoot/.venv/Scripts/python.exe" @taskArgs
exit $LASTEXITCODE
