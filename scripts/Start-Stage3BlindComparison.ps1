param([switch]$Score)
$taskRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $taskRoot
if ($Score) {
    & "$taskRoot/.venv/Scripts/python.exe" "$taskRoot/src/score_stage3_blind_comparison.py"
} else {
    & "$taskRoot/.venv/Scripts/python.exe" "$taskRoot/src/review_stage3_human.py" --video-dir "$taskRoot/data/stage3-blind-review-20260911/videos" --output-dir "$taskRoot/data/stage3-blind-review-20260911/labels"
}
