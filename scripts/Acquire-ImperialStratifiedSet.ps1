param(
    [string]$ScoutDir = 'artifacts/stage1-imperial-full-scout-20260918',
    [string]$OutputDir = 'artifacts/stage1-imperial-stratified-20260918',
    [int]$ContentCount = 20
)
$ErrorActionPreference = 'Stop'
$singleUrl = 'https://www.commsp.ee.ic.ac.uk/~pld/research/Rewind/Recapture/TestImages/SingleCaptureImages.zip'
$recapUrl = 'https://www.commsp.ee.ic.ac.uk/~pld/research/Rewind/Recapture/TestImages/RecapturedImages.zip'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$statusPath = Join-Path $OutputDir 'status.json'
$manifestPath = Join-Path $OutputDir 'manifest.json'
$single = Get-Content -Raw (Join-Path $ScoutDir 'single-index.json') | ConvertFrom-Json
$recap = Get-Content -Raw (Join-Path $ScoutDir 'recaptured-index.json') | ConvertFrom-Json

function Get-ContentId([string]$Name,[bool]$IsRecap) {
    if ($IsRecap -and $Name -match '-0*(\d+)\.(png|jpg)$') { return [int]$Matches[1] }
    if (-not $IsRecap -and $Name -match 'DS-05-0*(\d+)-S%') { return [int]$Matches[1] }
    return $null
}
function Get-Sha256([string]$Path) {
    $sha=[Security.Cryptography.SHA256]::Create(); $stream=[IO.File]::OpenRead((Resolve-Path $Path))
    try { return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLower() }
    finally { $stream.Dispose(); $sha.Dispose() }
}
function Save-State([string]$State,[int]$Completed,[int]$Expected,[object]$Extra=$null) {
    $o=[ordered]@{status=$State;completed=$Completed;expected=$Expected;updated=(Get-Date).ToString('o');manifest=$manifestPath}
    if ($null -ne $Extra) { $o.detail=$Extra }
    $o|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $statusPath -Encoding UTF8
}
function Expand-RemoteEntry([object]$Entry,[string]$Url,[string]$Destination) {
    $offset=[long]$Entry.local_header_offset; $compressed=[long]$Entry.compressed_size
    $end=$offset+$compressed+1023; $segment=$Destination+'.segment'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $required=$end-$offset+1; $attempt=0
    while ((-not (Test-Path -LiteralPath $segment)) -or (Get-Item -LiteralPath $segment).Length -lt $required) {
        $have=$(if(Test-Path -LiteralPath $segment){(Get-Item -LiteralPath $segment).Length}else{0})
        $rangeStart=$offset+$have; $part=$segment+'.part'; if(Test-Path $part){Remove-Item $part -Force}
        & curl.exe --fail --location --max-time 300 --range "$rangeStart-$end" --output $part $Url
        $got=$(if(Test-Path -LiteralPath $part){(Get-Item -LiteralPath $part).Length}else{0})
        if($got -gt 0){$src=[IO.File]::OpenRead((Resolve-Path $part));$dst=[IO.File]::Open($segment,[IO.FileMode]::Append,[IO.FileAccess]::Write);try{$src.CopyTo($dst)}finally{$src.Dispose();$dst.Dispose()};Remove-Item $part -Force}
        $attempt++; if($attempt -ge 144){throw "bounded range retries exhausted: $($Entry.name)"}
        if($got -eq 0){Start-Sleep -Seconds ([math]::Min(30,2*$attempt))}
    }
    $b=[IO.File]::ReadAllBytes((Resolve-Path $segment))
    if ([BitConverter]::ToUInt32($b,0) -ne 0x04034b50) { throw "bad local header: $($Entry.name)" }
    $method=[BitConverter]::ToUInt16($b,8); $nameLen=[BitConverter]::ToUInt16($b,26); $extraLen=[BitConverter]::ToUInt16($b,28)
    $start=30+$nameLen+$extraLen
    if ($method -ne 8) { throw "unsupported method ${method}: $($Entry.name)" }
    $input=[IO.MemoryStream]::new($b,$start,$compressed,$false)
    $deflate=[IO.Compression.DeflateStream]::new($input,[IO.Compression.CompressionMode]::Decompress)
    $output=[IO.File]::Create($Destination)
    try { $deflate.CopyTo($output) } finally { $output.Dispose(); $deflate.Dispose(); $input.Dispose() }
    Remove-Item -LiteralPath $segment -Force
    $actual=(Get-Item -LiteralPath $Destination).Length
    if ($actual -ne [long]$Entry.uncompressed_size) { throw "size mismatch: $($Entry.name)" }
    Add-Type -AssemblyName System.Drawing
    $img=[Drawing.Image]::FromFile((Resolve-Path $Destination))
    try { $width=$img.Width; $height=$img.Height } finally { $img.Dispose() }
    return [ordered]@{content_id=0;archive_name=$Entry.name;path=$Destination;bytes=$actual;width=$width;height=$height;sha256=(Get-Sha256 $Destination);crc32_declared=$Entry.crc32}
}

try {
    $recapEntries=@($recap.entry_details|Where-Object {$_.name -match '\.(png|jpg)$'})
    $allIds=@($recapEntries|ForEach-Object {Get-ContentId $_.name $true}|Sort-Object -Unique)
    if ($allIds.Count -lt $ContentCount) { throw 'not enough content IDs' }
    $selected=[Collections.Generic.List[int]]::new()
    for($i=0;$i -lt $ContentCount;$i++) {
        $index=[math]::Round($i*($allIds.Count-1)/[math]::Max(1,$ContentCount-1))
        $selected.Add([int]$allIds[$index])
    }
    $plan=[Collections.Generic.List[object]]::new()
    foreach($id in $selected) {
        $orig=@($single.entry_details|Where-Object {(Get-ContentId $_.name $false) -eq $id -and $_.name -notmatch '/not used/'})
        $re=@($recapEntries|Where-Object {(Get-ContentId $_.name $true) -eq $id})
        if ($orig.Count -ne 1 -or $re.Count -ne 8) { throw "pair cardinality failed for content ${id}: original=$($orig.Count), recap=$($re.Count)" }
        $plan.Add([pscustomobject]@{id=$id;kind='original';entry=$orig[0];url=$singleUrl})
        foreach($e in $re) { $plan.Add([pscustomobject]@{id=$id;kind='recaptured';entry=$e;url=$recapUrl}) }
    }
    $records=[Collections.Generic.List[object]]::new(); $done=0
    if(Test-Path -LiteralPath $manifestPath){@(Get-Content -Raw $manifestPath|ConvertFrom-Json)|ForEach-Object{$records.Add($_)}}
    Save-State 'RUNNING' $done $plan.Count ([ordered]@{selected_content_ids=$selected})
    foreach($item in $plan) {
        $device=($item.entry.name -split '/')[1]
        $ext=[IO.Path]::GetExtension($item.entry.name).ToLower()
        $dest=Join-Path $OutputDir (Join-Path ('content_{0:d4}' -f $item.id) ("$($item.kind)_$device$ext"))
        if (Test-Path -LiteralPath $dest) { $done++; continue }
        $record=Expand-RemoteEntry $item.entry $item.url $dest
        $record.content_id=$item.id; $record.kind=$item.kind; $record.device=$device
        $records.Add([pscustomobject]$record); $done++
        $records|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $manifestPath -Encoding UTF8
        Save-State 'RUNNING' $done $plan.Count ([ordered]@{current=$item.entry.name;selected_content_ids=$selected})
    }
    Save-State 'COMPLETE_VALIDATED' $done $plan.Count ([ordered]@{selected_content_ids=$selected;decoded=$done;content_groups=$ContentCount})
} catch {
    Save-State 'FAILED' $done $(if($null-ne$plan){$plan.Count}else{0}) ([ordered]@{error=$_.Exception.Message})
    throw
}
