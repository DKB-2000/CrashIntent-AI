param([switch]$PrepareOnly)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$acq=Join-Path $root 'artifacts/stage2-validation-acquisition-20260914'
$out=Join-Path $root 'artifacts/stage2-validation-inspection-20260914'
$data=Join-Path $root 'data_raw/stage2-validation-nexar-20260914'
New-Item -ItemType Directory -Force -Path $out,(Join-Path $out 'fingerprints') | Out-Null
function Save($name,$value) {
    $path=Join-Path $out $name
    $value | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath "$path.tmp" -Encoding UTF8
    Move-Item -LiteralPath "$path.tmp" -Destination $path -Force
}
$lock=$null
try {
    $lock=[IO.File]::Open((Join-Path $out 'inspection.lock'),'OpenOrCreate','ReadWrite','None')
    $manifest=Get-Content (Join-Path $acq 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $metadata=Import-Csv (Join-Path $data 'metadata.csv')
    $rows=@(foreach($item in $manifest.selected) {
        $file=[IO.Path]::GetFileName($item.path)
        $m=@($metadata | Where-Object {$_.file_name -ceq $file})
        if($m.Count -ne 1){throw "Metadata mapping: $file"}
        [pscustomobject]@{ID="NEXAR_$([IO.Path]::GetFileNameWithoutExtension($file))"; file=$file;
            purpose='VALIDATION_ONLY'; source_dataset='nexar-ai/nexar_collision_prediction'; revision=$manifest.revision;
            source_url=$item.url; expected_sha256=$item.sha256; expected_bytes=$item.bytes;
            duration_seconds=''; reference_event_seconds=$m[0].time_of_event; reference_alert_seconds=$m[0].time_of_alert;
            light_conditions=$m[0].light_conditions; weather=$m[0].weather; scene=$m[0].scene;
            acquisition='PENDING'; similarity_group=''; suspected_matches=0; recording_source='UNKNOWN'; human_review='NOT_STARTED'}
    })
    if($rows.Count -ne 50 -or @($rows.ID | Sort-Object -Unique).Count -ne 50){throw 'Manifest IDs invalid'}
    $rows | Export-Csv (Join-Path $data 'review-inventory.csv') -NoTypeInformation -Encoding UTF8
    Save 'reservation.json' @{purpose='VALIDATION_ONLY'; IDs=$rows.ID; manifest_sha256=(Get-FileHash (Join-Path $acq 'manifest.json')).Hash; human_labels_created=0}
    if($PrepareOnly){Save 'status.json' @{status='PREPARED'; total=1550; completed=0}; return}
    $start=Get-Date
    function Status($phase,$count) { Save 'status.json' @{status='RUNNING'; phase=$phase; completed=$count; total=1550; pid=$PID; updated=(Get-Date -Format o)} }
    function Deadline {if(((Get-Date)-$start).TotalHours -gt 4){throw 'Four hour inspection timeout'}}
    Add-Type -Path (Join-Path $root 'src/Stage2DuplicateFingerprint.cs')
    # Sanity checks for the actual matching kernel (temporal offset and negative case).
    $hit=[Stage2DuplicateFingerprint]::Match([uint64[]]@(1,2,4),[uint64[]]@([uint64]::MaxValue,1,2,4))
    $miss=[Stage2DuplicateFingerprint]::Match([uint64[]]@(0,0,0),[uint64[]]@([uint64]::MaxValue,[uint64]::MaxValue,[uint64]::MaxValue))
    if($hit[0] -ne 0 -or $hit[1] -ne 1 -or $hit[2] -ne 0 -or $miss[0] -ne -1){throw 'Fingerprint kernel check failed'}
    $ffmpeg=Join-Path $root '.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    function Fingerprint($path,$id) {
        Deadline
        $digest=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        $raw=Join-Path $out "fingerprints/$id.gray"
        $log=Join-Path $out "fingerprints/$id.log"
        $p=Start-Process -FilePath $ffmpeg -ArgumentList @('-hide_banner','-nostdin','-y','-threads','1','-i',('"'+$path+'"'),'-an','-vf','fps=2,scale=9:8:flags=area,format=gray','-threads','1','-f','rawvideo',('"'+$raw+'"')) -PassThru -WindowStyle Hidden -RedirectStandardError $log -RedirectStandardOutput (Join-Path $out 'ffmpeg-output.log')
        if(!$p.WaitForExit(30000)){$p.Kill();throw "Fingerprint timeout: $id"}
        $p.Refresh(); if($p.ExitCode -ne 0){throw "Fingerprint failed: $id"}
        $hashes=[Stage2DuplicateFingerprint]::Read($raw)
        $txt=Get-Content -LiteralPath $log -Raw
        if($txt -notmatch 'Duration: (\d+):(\d+):(\d+\.\d+)'){throw "Duration missing: $id"}
        $seconds=3600*[int]$Matches[1]+60*[int]$Matches[2]+[double]::Parse($Matches[3],[Globalization.CultureInfo]::InvariantCulture)
        [pscustomobject]@{ID=$id; sha256=$digest; hashes=$hashes; duration=$seconds; fingerprint_sha256=(Get-FileHash $raw).Hash}
    }
    $ccdFiles=@(Get-ChildItem data_raw/ccd/videos/Crash-1500 -Filter '*.mp4' | Sort-Object Name)
    if($ccdFiles.Count -ne 1500){throw 'CCD reference count mismatch'}
    $ccd=@(); Status 'CCD_FINGERPRINTS' 0
    foreach($file in $ccdFiles){$ccd+=Fingerprint $file.FullName "CCD_$($file.BaseName)"; Status 'CCD_FINGERPRINTS' $ccd.Count}
    while($true) {
        Deadline
        $state=Get-Content (Join-Path $acq 'status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        if($state.status -eq 'ACQUIRED_UNLABELED'){break}
        if($state.status -eq 'FAILED'){throw "Acquisition failed: $($state.error)"}
        if($state.pid -and !(Get-Process -Id $state.pid -ErrorAction SilentlyContinue)){throw 'Acquisition process absent'}
        Status 'WAITING_FOR_ACQUISITION' 1500
        Start-Sleep -Seconds 15
    }
    $validated=Get-Content (Join-Path $acq 'validated-files.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if(@($validated).Count -ne 50){throw 'Acquisition evidence incomplete'}
    $nexar=@()
    foreach($row in $rows) {
        $v=@($validated | Where-Object {$_.sha256 -ieq $row.expected_sha256 -and $_.full_decode -eq 'PASS'})
        if($v.Count -ne 1){throw 'Acquisition evidence mismatch'}
        $fp=Fingerprint (Join-Path $data $row.file) $row.ID
        if($fp.sha256 -ine $row.expected_sha256 -or (Get-Item (Join-Path $data $row.file)).Length -ne $row.expected_bytes){throw 'Candidate hash/size mismatch'}
        $row.duration_seconds=$fp.duration; $row.acquisition='HASH_AND_DECODE_PASS'; $nexar+=$fp
        Status 'NEXAR_FINGERPRINTS' (1500+$nexar.Count)
    }
    $pairs=@(); $comparisons=0
    for($i=0;$i -lt $nexar.Count;$i++) {
        Deadline
        $a=$nexar[$i]
        $others=@($ccd)+@($nexar | Select-Object -Skip ($i+1))
        foreach($b in $others) {
            $match=[Stage2DuplicateFingerprint]::Match($a.hashes,$b.hashes);$comparisons++
            $exact=$a.sha256 -ceq $b.sha256
            if($exact -or $match[0] -ge 0){$pairs+=[pscustomobject]@{candidate=$a.ID; reference=$b.ID; exact_file=$exact; candidate_seconds=$match[0]/2.0; reference_seconds=$match[1]/2.0; distance_sum=$match[2]; verdict='SUSPECT_REQUIRES_REVIEW'}}
        }
        Status 'COMPARING' (1500+$i+1)
    }
    if($comparisons -ne 76225){throw 'Comparison coverage mismatch'}
    # Connected components are provisional review groups, never confirmed source identities.
    $groups=@{}; foreach($row in $rows){$groups[$row.ID]=$row.ID}
    foreach($pair in $pairs){if(!$groups.ContainsKey($pair.reference)){$groups[$pair.reference]=$pair.reference}}
    foreach($pair in $pairs){$old=$groups[$pair.reference];$new=$groups[$pair.candidate];foreach($key in @($groups.Keys)){if($groups[$key] -eq $old){$groups[$key]=$new}}}
    foreach($row in $rows){$row.suspected_matches=@($pairs | Where-Object {$_.candidate -eq $row.ID -or $_.reference -eq $row.ID}).Count;if($row.suspected_matches){$row.similarity_group=$groups[$row.ID]}}
    $rows | Export-Csv (Join-Path $data 'review-inventory.csv') -NoTypeInformation -Encoding UTF8
    Save 'suspected-pairs.json' @($pairs)
    Save 'fingerprint-provenance.json' @(@($ccd)+@($nexar) | Select-Object ID,sha256,duration,fingerprint_sha256)
    $cards=@(foreach($row in $rows){$id=[Net.WebUtility]::HtmlEncode($row.ID);$file=$row.file;"<article><h2>$id</h2><p>Duration: $($row.duration_seconds)s; reference event (NOT ground truth): $($row.reference_event_seconds)s; suspected matches: $($row.suspected_matches)</p><video controls preload='none' width='640' src='$file'></video></article>"})
    "<!doctype html><meta charset='utf-8'><title>Stage2 validation candidates</title><h1>Unlabeled validation candidates</h1><p>Not for training. Positive includes near misses. Recording identity unknown. No annotations are saved by this page.</p>$($cards -join "`n")" | Set-Content (Join-Path $data 'review.html') -Encoding UTF8
    Save 'report.json' @{status='SCREENING_COMPLETE'; candidates=50; ccd_references=1500; pair_comparisons=$comparisons; suspected_pairs=$pairs.Count; exact_file_duplicates=@($pairs | Where-Object exact_file).Count; human_labels=0; method='2fps 9x8 grayscale dHash; 3 consecutive frames <=8 bits each and <=18 total; full frame only'; limitations='No proven recording identities. Crops, speed changes, mirrors, heavy edits and shared routes may evade this screen. Similar roads may false-match. Past Stage1 Nexar 4 IDs unavailable.'; inventory_sha256=(Get-FileHash (Join-Path $data 'review-inventory.csv')).Hash; script_sha256=(Get-FileHash $PSCommandPath).Hash; kernel_sha256=(Get-FileHash (Join-Path $root 'src/Stage2DuplicateFingerprint.cs')).Hash}
    Save 'status.json' @{status='SCREENING_COMPLETE'; completed=1550; total=1550; suspected_pairs=$pairs.Count; human_labels=0; updated=(Get-Date -Format o)}
} catch {
    if($lock){Save 'status.json' @{status='FAILED'; error="$_"; pid=$PID; updated=(Get-Date -Format o)}}
    throw
} finally {if($lock){$lock.Dispose()}}
