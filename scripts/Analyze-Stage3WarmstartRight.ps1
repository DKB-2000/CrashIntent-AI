param([string]$OutputDir = 'artifacts/stage3-warmstart-right-audit-20260916')

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path '.').Path
$valid = @(Import-Csv 'artifacts/stage3-representation-20260910/validation-samples.csv')
$out = Join-Path $root $OutputDir
New-Item -ItemType Directory -Path $out -Force | Out-Null
$rows = @()
foreach ($seed in 20260910, 20260911, 20260912) {
    $arms = @{}
    $paths = @{
        original = "artifacts/stage3-representation-20260910/$seed-motion/validation-predictions.csv"
        warmstart_ce = "artifacts/stage3-prediction-preservation-20260914/warmstart_ce/$seed/predictions.csv"
        warmstart_preserve = "artifacts/stage3-prediction-preservation-20260914/warmstart_preserve/$seed/predictions.csv"
    }
    foreach ($arm in $paths.Keys) {
        $arms[$arm] = @(Import-Csv $paths[$arm])
        if ($arms[$arm].Count -ne $valid.Count) { throw "Row count mismatch: $arm $seed" }
        for ($i = 0; $i -lt $valid.Count; $i++) {
            $point = if ($arm -eq 'original') { $arms[$arm][$i].sample_index } else { $arms[$arm][$i].endpoint }
            if ($arms[$arm][$i].ID -ne $valid[$i].ID -or $point -ne $valid[$i].endpoint) {
                throw "Key mismatch: $arm $seed row $i"
            }
        }
    }
    foreach ($arm in 'warmstart_ce', 'warmstart_preserve') {
        $groups = @{}
        for ($i = 0; $i -lt $valid.Count; $i++) {
            if ($valid[$i].accel -eq '3' -or $valid[$i].steer -ne '2') { continue }
            $route = $valid[$i].route
            foreach ($scope in 'ALL', $route) {
                $key = "$scope|$arm"
                if (-not $groups.ContainsKey($key)) {
                    $groups[$key] = [ordered]@{seed=$seed; arm=$arm; route=$scope; right_rows=0; baseline_correct=0; candidate_correct=0; lost=0; gained=0; lost_to_straight=0; lost_to_left=0}
                }
                $g = $groups[$key]
                $g.right_rows++
                $before = $arms.original[$i].steer_label -eq 'RIGHT'
                $after = $arms[$arm][$i].steer -eq '2'
                if ($before) { $g.baseline_correct++ }
                if ($after) { $g.candidate_correct++ }
                if ($before -and -not $after) {
                    $g.lost++
                    if ($arms[$arm][$i].steer -eq '1') { $g.lost_to_straight++ } else { $g.lost_to_left++ }
                }
                if (-not $before -and $after) { $g.gained++ }
            }
        }
        foreach ($g in $groups.Values) {
            if ($g.arm -eq $arm) {
                $g['recall_delta'] = [math]::Round(($g.candidate_correct - $g.baseline_correct) / $g.right_rows, 6)
                $rows += [pscustomobject]$g
            }
        }
    }
}
$csv = Join-Path $out 'right-loss-by-route.csv'
$rows | Sort-Object arm,seed,route | Export-Csv -Path $csv -NoTypeInformation -Encoding utf8
$summary = $rows | Where-Object route -eq 'ALL' | Sort-Object arm,seed | Select-Object seed,arm,right_rows,baseline_correct,candidate_correct,lost,gained,lost_to_straight,lost_to_left,recall_delta
$review = Get-Content 'artifacts/stage3-prediction-preservation-20260914/final-review.json' -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($arm in 'warmstart_ce', 'warmstart_preserve') {
    $expected = $review.gates.$arm.recall_delta[2]
    $measured = ($summary | Where-Object arm -eq $arm | Measure-Object recall_delta -Average).Average
    if ([math]::Abs($expected - $measured) -gt 0.000001) { throw "Independent recall mismatch: $arm" }
}
$summary | Format-Table -AutoSize
$report = [ordered]@{
    status = 'COMPLETE_VALIDATED'
    scope = 'Existing external development routes; no new training or competition labels'
    seeds = @(20260910,20260911,20260912)
    rows_per_seed = $valid.Count
    validation = 'All row keys match; right-recall deltas independently agree with final-review.json'
    summary = @($summary)
    csv = $csv
}
$report | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $out 'report.json') -Encoding utf8
