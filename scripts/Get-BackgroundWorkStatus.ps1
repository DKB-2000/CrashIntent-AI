# One local snapshot only. No polling, remote API, or job launch.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sources = [ordered]@{
    Stage3RightRecovery = 'artifacts/stage3-right-recovery-20260914/status.json'
    Stage1ScreenTrial = 'artifacts/kaggle-stage1-screen-trial-20260914/remote-run.json'
    Stage3TransitionMatchedControl = 'artifacts/stage3-transition-matched-control-20260914/status.json'
    Stage3StraightSampleControl = 'artifacts/stage3-straight-sample-control-20260914/status.json'
    Stage1ScreenCueData = 'artifacts/stage1-screen-cue-data-20260914/status.json'
    Stage1MixedTrial = 'artifacts/kaggle-stage1-mixed-trial-20260914/remote-run.json'
    Stage1QualityTrial = 'artifacts/kaggle-stage1-quality-trial-20260911/remote-run.json'
    Stage1QualityData = 'artifacts/stage1-quality-balanced-20260911-lowmem/status.json'
    Stage1Supervisor = 'artifacts/stage1-robustness-20260910/full-model-evaluation/supervisor-status.json'
    Stage1Evaluation = 'artifacts/stage1-robustness-20260910/full-model-evaluation/result/status.json'
    Stage1Release = 'artifacts/stage1-auto-release-20260910/status.json'
    Stage2Precision = 'artifacts/kaggle-stage2-precision-20260911/remote-run.json'
    Stage2SpatialMotion = 'artifacts/stage2-spatial-motion-20260914/status.json'
    Stage2FinalInputs = 'artifacts/stage2-final-validation-20260914/status.json'
    Stage2FinalReadiness = 'artifacts/stage2-final-validation-20260914/readiness-status.json'
    Stage2UnlabeledComparison = 'artifacts/stage2-unlabeled-comparison-20260914/status.json'
    Stage2ValidationAcquisition = 'artifacts/stage2-validation-acquisition-20260914/status.json'
    Stage2ValidationInspection = 'artifacts/stage2-validation-inspection-20260914/status.json'
    Stage3Dynamics = 'artifacts/stage3-dynamics-cv-20260911/status.json'
    Stage3FullFrame = 'artifacts/stage3-fullframe-cv-20260914/status.json'
    Stage3FusedView = 'artifacts/stage3-fused-view-cv-20260914/status.json'
    Stage3CivicAcquisition = 'artifacts/stage3-civic-acquisition-20260914/status.json'
    Stage3CivicPartialScore = 'artifacts/stage3-civic-partial-score-20260914/status.json'
    Stage3CivicHoldoutAcquisition = 'artifacts/stage3-civic-holdout-acquisition-20260914/status.json'
    Stage3CivicHoldoutScore = 'artifacts/stage3-civic-holdout-score-20260914/status.json'
}
foreach ($entry in $sources.GetEnumerator()) {
    $path = Join-Path $projectRoot $entry.Value
    $state = $null
    $updated = $null
    $readError = $null
    if (Test-Path -LiteralPath $path) {
        try {
            $state = Get-Content -Raw -Encoding utf8 -LiteralPath $path | ConvertFrom-Json
            $updated = (Get-Item -LiteralPath $path).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
        } catch { $readError = $_.Exception.Message }
    } else { $readError = 'State file missing' }
    [pscustomobject]@{
        Job = $entry.Key
        RecordedStatus = if ($readError) { 'UNAVAILABLE' } else { $state.status }
        MonitorStatus = $state.monitor_status
        Completed = if ($null -ne $state.completed_videos) { $state.completed_videos } else { $state.completed }
        Expected = if ($null -ne $state.expected_videos) { $state.expected_videos } else { $state.total }
        Models = $state.models
        UpdatedLocal = $updated
        Error = if ($readError) { $readError } elseif ($state.last_error) { $state.last_error } else { $state.error }
        StateFile = $entry.Value
    }
}
