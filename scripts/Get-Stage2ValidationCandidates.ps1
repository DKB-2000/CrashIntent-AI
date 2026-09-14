param([switch]$Worker)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$out = Join-Path $root 'data_raw/stage2-validation-nexar-20260914'
$stateDir = Join-Path $root 'artifacts/stage2-validation-acquisition-20260914'
New-Item -ItemType Directory -Force -Path $out,$stateDir | Out-Null
function Save-Json($path, $value) {
    $value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath "$path.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$path.tmp" -Destination $path -Force
}
$lock = $null
try {
    $lock = [IO.File]::Open((Join-Path $stateDir 'acquisition.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    if (!$Worker) {
        $ccd = @(Get-Content data_raw/ccd/Crash-1500.txt | ForEach-Object {
            if ($_ -notmatch '^(\d+),\[[^\]]+\],\d+,([^,]+),') { throw 'Invalid CCD row' }
            [pscustomobject]@{ID=$Matches[1]; source_id=$Matches[2]}
        })
        $pre = @(Import-Csv artifacts/stage2-ccd-pretrain-20260910/pretrain-labels.csv)
        $manual = @(Import-Csv artifacts/stage2-source-cv-20260910/fold-0/train.csv) + @(Import-Csv artifacts/stage2-source-cv-20260910/fold-0/validation.csv)
        $used = @($pre.source_id + $manual.source_id | Sort-Object -Unique)
        foreach ($row in @($pre) + @($manual)) {
            if (@($ccd | Where-Object {$_.ID -ceq $row.ID -and $_.source_id -ceq $row.source_id}).Count -ne 1) { throw 'Source mapping mismatch' }
        }
        $remaining = @($ccd | Where-Object {$_.source_id -cnotin $used})
        Save-Json (Join-Path $stateDir 'ccd-source-audit.json') @{
            status='PASS'; total_videos=$ccd.Count; total_sources=@($ccd.source_id | Sort-Object -Unique).Count
            manual_videos=$manual.Count; manual_sources=@($manual.source_id | Sort-Object -Unique).Count
            pretrain_videos=$pre.Count; pretrain_sources=@($pre.source_id | Sort-Object -Unique).Count
            used_sources=$used; remaining_videos=$remaining.Count; remaining=$remaining
            input_sha256=@( 'data_raw/ccd/Crash-1500.txt','artifacts/stage2-ccd-pretrain-20260910/pretrain-labels.csv','artifacts/stage2-source-cv-20260910/fold-0/train.csv','artifacts/stage2-source-cv-20260910/fold-0/validation.csv' | ForEach-Object { @{path=$_; sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash} })
        }
        if (Test-Path (Join-Path $stateDir 'manifest.json')) { throw 'Manifest already exists; use -Worker to resume without reselection' }
        $repo = 'https://huggingface.co/api/datasets/nexar-ai/nexar_collision_prediction'
        $revision = (Invoke-RestMethod -Uri $repo -TimeoutSec 30).sha
        $files = Invoke-RestMethod -Uri "$repo/tree/$revision/train/positive?limit=1000" -TimeoutSec 30
        $base = "https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/$revision"
        foreach ($name in 'README.md','LICENSE','train/positive/metadata.csv') {
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$name" -OutFile (Join-Path $out ([IO.Path]::GetFileName($name))) -TimeoutSec 60
        }
        # Hash ordering gives a fixed sample independent of labels, predictions and download size.
        $hash = [Security.Cryptography.SHA256]::Create()
        $ranked = @($files | Where-Object {$_.path.EndsWith('.mp4')} | ForEach-Object {
            $key = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes("stage2-holdout-20260914:$($_.path)"))).Replace('-','')
            [pscustomobject]@{path=$_.path; bytes=$_.size; sha256=$_.lfs.oid; rank=$key; url="$base/$($_.path)"}
        } | Sort-Object rank)
        $hash.Dispose()
        $selected = @($ranked | Select-Object -First 50)
        if ($selected.Count -ne 50 -or @($selected | Where-Object {!$_.sha256}).Count) { throw 'Incomplete manifest' }
        Save-Json (Join-Path $stateDir 'manifest.json') @{revision=$revision; selected=$selected; total_bytes=($selected | Measure-Object bytes -Sum).Sum; purpose='Unlabeled validation candidates only; positive includes near misses; source identity and visual duplicate audit pending'}
        Save-Json (Join-Path $stateDir 'status.json') @{status='PREPARED'; completed=0; total=50; updated=(Get-Date -Format o)}
        Write-Output "Prepared 50 candidates; bytes=$(($selected | Measure-Object bytes -Sum).Sum)"
    } else {
        $manifest = Get-Content (Join-Path $stateDir 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        $ffmpeg = Join-Path $root '.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
        if (!(Test-Path -LiteralPath $ffmpeg)) { throw 'FFmpeg missing' }
        $start = Get-Date
        $done = @()
        Save-Json (Join-Path $stateDir 'status.json') @{status='RUNNING'; pid=$PID; completed=0; total=50; updated=(Get-Date -Format o)}
        $ccdHashes = @{}
        Get-ChildItem data_raw/ccd/videos/Crash-1500 -Filter '*.mp4' | ForEach-Object { $ccdHashes[(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash] = $_.Name }
        foreach ($item in $manifest.selected) {
            if (((Get-Date)-$start).TotalHours -gt 3) { throw 'Three hour total timeout' }
            $target = Join-Path $out ([IO.Path]::GetFileName($item.path))
            $valid = $false
            for ($attempt=1; $attempt -le 3; $attempt++) {
                try {
                    if (!(Test-Path -LiteralPath $target)) {
                        Invoke-WebRequest -UseBasicParsing -Uri $item.url -OutFile "$target.part" -TimeoutSec 180
                        Move-Item -LiteralPath "$target.part" -Destination $target -Force
                    }
                    $digest = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
                    if ((Get-Item -LiteralPath $target).Length -ne $item.bytes -or $digest -ine $item.sha256) {
                        Move-Item -LiteralPath $target -Destination "$target.invalid-$attempt" -Force
                        throw 'Size/hash mismatch'
                    }
                    if ($ccdHashes.ContainsKey($digest)) { throw 'Exact CCD duplicate' }
                    $log = Join-Path $stateDir (([IO.Path]::GetFileNameWithoutExtension($target)) + '-decode.log')
                    $proc = Start-Process -FilePath $ffmpeg -ArgumentList @('-v','error','-xerror','-threads','1','-i',('"'+$target+'"'),'-map','0:v:0','-f','null','-') -PassThru -WindowStyle Hidden -RedirectStandardError $log
                    if (!$proc.WaitForExit(60000)) { $proc.Kill(); throw 'Decode timeout' }
                    $proc.Refresh()
                    if ($proc.ExitCode -ne 0 -or (Get-Item -LiteralPath $log).Length -gt 0) { throw 'Decode failed' }
                    $valid=$true; break
                } catch {
                    "$(Get-Date -Format o) $($item.path) attempt=$attempt $_" | Add-Content (Join-Path $stateDir 'download.log')
                    if ($attempt -eq 3) { throw }
                    Start-Sleep -Seconds 5
                }
            }
            if (!$valid) { throw 'Unvalidated file' }
            $done += [pscustomobject]@{path=$item.path; sha256=$digest; bytes=$item.bytes; full_decode='PASS'; exact_ccd_duplicate=$false}
            Save-Json (Join-Path $stateDir 'validated-files.json') $done
            Save-Json (Join-Path $stateDir 'status.json') @{status='RUNNING'; pid=$PID; completed=$done.Count; total=50; updated=(Get-Date -Format o)}
        }
        Save-Json (Join-Path $stateDir 'status.json') @{status='ACQUIRED_UNLABELED'; completed=$done.Count; total=50; updated=(Get-Date -Format o); limitations='Collision eligibility, independent recording identities, perceptual duplicates and human labels pending. Do not train on these candidates.'}
    }
} catch {
    if ($lock) { Save-Json (Join-Path $stateDir 'status.json') @{status='FAILED'; pid=$PID; error="$_"; updated=(Get-Date -Format o)} }
    throw
} finally { if ($lock) {$lock.Dispose()} }
