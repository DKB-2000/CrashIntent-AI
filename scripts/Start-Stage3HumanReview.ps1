param([string]$VideoDir = 'Baseline/data/stage3/videos', [string]$OutputDir = 'data/stage3-human-review', [switch]$Assume10Hz)
$taskRoot=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $taskRoot
$taskArgs=@("$taskRoot/src/review_stage3_human.py",'--video-dir',$VideoDir,'--output-dir',$OutputDir)
if ($Assume10Hz) {$taskArgs+='--assume-10hz'}
& "$taskRoot/.venv/Scripts/python.exe" @taskArgs
