# One local snapshot only. No polling, remote API, or job launch.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sources = [ordered]@{
    Stage3AccelGRUGPU = 'artifacts/kaggle-stage3-accel-gru-20260918/remote-run.json'
    Stage1ImperialStratified = 'artifacts/stage1-imperial-stratified-20260918/status.json'
    Stage3RGBGRUGPU = 'artifacts/kaggle-stage3-rgb-gru-20260918/remote-run.json'
    Stage1VDMoireGPU = 'artifacts/kaggle-stage1-vdmoire-20260918/remote-run.json'
    Stage1HandcraftedVideo = 'artifacts/stage1-handcrafted-video-20260918/status.json'
    Stage1ImperialFeatureCV = 'artifacts/kaggle-stage1-imperial-feature-cv-20260918/remote-run.json'
    Stage1ImperialSpatialTrial = 'artifacts/kaggle-stage1-imperial-spatial-20260917/remote-run.json'
    Stage1ImperialMotionTrial = 'artifacts/kaggle-stage1-imperial-motion-20260917/remote-run.json'
    Stage1ImperialGPU = 'artifacts/kaggle-stage1-imperial-20260917/remote-run.json'
    Stage1ImperialEvaluation = 'artifacts/stage1-imperial-evaluation-20260917/status.json'
    Stage1ImperialSample = 'artifacts/stage1-imperial-recapture-20260917/status.json'
    Stage1DLCPairTraining = 'artifacts/stage1-dlc-pair-training-20260917/status.json'
    Stage1DLCRangePipeline = 'artifacts/stage1-dlc-range-pipeline-20260917/status.json'
    Stage1DLCRangePilot = 'artifacts/stage1-dlc-range-pilot-20260917/status.json'
    Stage1DLCPilot = 'artifacts/stage1-dlc-pilot-evaluation-20260917/status.json'
    Stage1DLCFrames = 'artifacts/stage1-dlc-frames-acquisition-20260916/status.json'
    Stage1QuarterMixTrial = 'artifacts/kaggle-stage1-quarter-mix-trial-20260916/remote-run.json'
    Stage3PredictionPreservation = 'artifacts/stage3-prediction-preservation-20260914/status.json'
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
    Stage2MMAUTrainImages = 'artifacts/stage2-mmau-detection-images-train-20260918/status.json'
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
    Stage3CivicLowSpeedScout = 'artifacts/stage3-civic-low-speed-scout-20260916/status.json'
    Stage3CivicLowSpeedAcquisition = 'artifacts/stage3-civic-low-speed-acquisition-20260916/status.json'
    Stage3CivicLowSpeedClassScout = 'artifacts/stage3-civic-low-speed-class-scout-20260916/status.json'
    Stage3FrozenCivicHoldout = 'artifacts/stage3-frozen-civic-holdout-20260916/status.json'
    Stage3Civic535ClassScout = 'artifacts/stage3-civic-5-35-class-scout-20260916/status.json'
    Stage3Civic535Acquisition = 'artifacts/stage3-civic-535-balanced-acquisition-20260916/status.json'
    Stage3Civic535Evaluation = 'artifacts/stage3-civic-535-balanced-evaluation-20260917/status.json'
    Stage3CivicTrainingAcquisition = 'artifacts/stage3-civic-535-training-acquisition-20260917/status.json'
    Stage3CivicDomainAdaptation = 'artifacts/stage3-civic-domain-adaptation-20260917/status.json'
    Stage3A2D2Scout = 'artifacts/stage3-a2d2-scout-20260917/status.json'
    Stage3A2D2PilotAcquisition = 'artifacts/stage3-a2d2-pilot-acquisition-20260917/status.json'
    Stage3A2D2ClipPilot = 'artifacts/stage3-a2d2-clip-pilot-20260917/status.json'
    Stage3A2D2TrainingBuses = 'artifacts/stage3-a2d2-training-buses-20260917/status.json'
    Stage3A2D2TrainingClips = 'artifacts/stage3-a2d2-training-clips-20260917/status.json'
    Stage3A2D2Rehearsal = 'artifacts/stage3-a2d2-rehearsal-20260918/status.json'
    Stage3A2D2EnsembleRehearsal = 'artifacts/stage3-a2d2-ensemble-rehearsal-20260918/status.json'
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
        Expected = if ($null -ne $state.expected_videos) { $state.expected_videos } elseif ($null -ne $state.expected) { $state.expected } else { $state.total }
        Models = $state.models
        UpdatedLocal = $updated
        Error = if ($readError) { $readError } elseif ($state.last_error) { $state.last_error } else { $state.error }
        StateFile = $entry.Value
    }
}
