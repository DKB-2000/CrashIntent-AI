param(
    [string]$Labels = 'data/stage2-validation-nexar-20260914/review-decisions.csv',
    [string]$Output = 'artifacts/stage2-final-comparison-20260914'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$labelPath = (Resolve-Path -LiteralPath $Labels).Path
$outputPath = [IO.Path]::GetFullPath((Join-Path $root $Output))
if (-not $outputPath.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must stay in project workspace' }
if (Test-Path -LiteralPath $outputPath) { throw 'Output already exists' }
$python = Join-Path $root '.venv/Scripts/python.exe'
& $python src/score_stage2_final_validation.py check --labels $labelPath
if ($LASTEXITCODE -ne 0) { throw 'Input or label validation failed' }
$state = Get-Content artifacts/stage2-final-validation-20260914/label-readiness.json -Raw -Encoding UTF8 | ConvertFrom-Json
if ($state.status -ne 'READY_TO_SCORE') { throw 'Human review and labels are incomplete; no inference started' }
$argsList = @('-u', 'src/watch_stage2_final_comparison.py', '--labels', ('"' + $labelPath + '"'), '--output', ('"' + $outputPath + '"'))
$out = Join-Path $root 'artifacts/stage2-final-validation-20260914'
$process = Start-Process -FilePath $python -ArgumentList $argsList -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out 'comparison-launcher.log') -RedirectStandardError (Join-Path $out 'comparison-launcher-error.log')
$process | Select-Object Id, ProcessName
