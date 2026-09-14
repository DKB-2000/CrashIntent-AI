$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$out=Join-Path $root 'artifacts/stage2-contact-review-20260914'
New-Item -ItemType Directory -Force -Path $out | Out-Null
function Save($name,$v){$p=Join-Path $out $name;$v | ConvertTo-Json -Depth 7 | Set-Content "$p.tmp" -Encoding UTF8;Move-Item -LiteralPath "$p.tmp" -Destination $p -Force}
$lock=$null
try {
 $lock=[IO.File]::Open((Join-Path $out 'preparation.lock'),'OpenOrCreate','ReadWrite','None')
 $rows=Import-Csv data_raw/stage2-validation-nexar-20260914/review-queue.csv
 $ff=Join-Path $root '.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
 $started=Get-Date;$index=@()
 foreach($r in $rows){
  if(((Get-Date)-$started).TotalMinutes -gt 15){throw 'Preparation timeout'}
  $t=[Math]::Max(0.0,([double]::Parse($r.reference_event_seconds,[Globalization.CultureInfo]::InvariantCulture)-1.0))
  $path=Join-Path $root ('data_raw/stage2-validation-nexar-20260914/'+$r.file)
  if((Get-FileHash $path).Hash -ine $r.expected_sha256){throw 'Source changed'}
  $png=Join-Path $out ($r.ID+'.png');$log=Join-Path $out ($r.ID+'.log')
  # No fps filter: first 60 decoded frames after seek, in their original order.
  $args=@('-hide_banner','-nostdin','-y','-threads','1','-ss',$t.ToString([Globalization.CultureInfo]::InvariantCulture),'-i',('"'+$path+'"'),'-an','-vf','"trim=end_frame=60,showinfo,scale=256:144,tile=5x12"','-frames:v','1','-threads','1',('"'+$png+'"'))
  $p=Start-Process $ff -ArgumentList $args -WindowStyle Hidden -PassThru -RedirectStandardError $log -RedirectStandardOutput (Join-Path $out 'stdout.log')
  if(!$p.WaitForExit(30000)){$p.Kill();throw 'FFmpeg timeout'}
  $p.Refresh();if($p.ExitCode -ne 0){throw "Decode failed: $($r.ID)"}
  $lines=@(Get-Content $log | Where-Object {$_ -match '\bn:\s*\d+\s+pts:'})
  if($lines.Count -ne 60){throw "Expected 60 frames: $($r.ID), got $($lines.Count)"}
  $pts=@(foreach($line in $lines){if($line -notmatch 'pts_time:([0-9.]+)'){throw 'PTS missing'};[double]::Parse($Matches[1],[Globalization.CultureInfo]::InvariantCulture)})
  for($k=1;$k -lt $pts.Count;$k++){if($pts[$k] -le $pts[$k-1]){throw 'Nonmonotonic PTS'}}
  $index+=[pscustomobject]@{ID=$r.ID; video_sha256=$r.expected_sha256;seek_seconds=$t;relative_pts=$pts;decoded_frames=60;image_sha256=(Get-FileHash $png).Hash}
  Save 'evidence-index.json' $index
  Save 'status.json' @{status='PREPARING';completed=$index.Count;total=50;pid=$PID;updated=(Get-Date -Format o)}
 }
 Save 'status.json' @{status='EVIDENCE_READY';completed=50;total=50;decoded_frames=3000;review_complete=$false;updated=(Get-Date -Format o)}
 '50 native-frame sequences prepared; 3000 frames; PTS and source hashes PASS'
}catch{if($lock){Save 'status.json' @{status='FAILED';error="$_";pid=$PID;updated=(Get-Date -Format o)}};throw}finally{if($lock){$lock.Dispose()}}
