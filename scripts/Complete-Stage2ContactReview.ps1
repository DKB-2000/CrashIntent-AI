$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$out=Join-Path $root 'artifacts/stage2-contact-review-20260914'
$data=Join-Path $root 'data_raw/stage2-validation-nexar-20260914'
$obs=Get-Content (Join-Path $out 'observations.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$evidence=Get-Content (Join-Path $out 'evidence-index.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$rows=@(Import-Csv (Join-Path $data 'review-queue.csv'))
if($obs.Count -ne 50 -or $evidence.Count -ne 50 -or $rows.Count -ne 50){throw 'Expected 50 records'}
foreach($set in @($obs,$evidence,$rows)){if(@($set.ID | Sort-Object -Unique).Count -ne 50){throw 'Duplicate IDs'}}
if(@(Compare-Object $rows.ID $obs.ID).Count -or @(Compare-Object $rows.ID $evidence.ID).Count){throw 'ID mismatch'}
$original=Join-Path $data 'review-inventory.csv'
$originalHash=(Get-FileHash $original).Hash
if($originalHash -ne 'F3D8A1323A7818122A07A282084CEC8C54C59D5295A178E814AEC65D49348B04'){throw 'Original inventory changed'}
$beforeQueueHash=(Get-FileHash (Join-Path $data 'review-queue.csv')).Hash
$counts=[ordered]@{CONTACT_SUSPECT=0;NO_CONTACT_VISIBLE=0;UNRESOLVED=0}
$queue=@()
foreach($r in $rows){
 $o=@($obs | Where-Object ID -eq $r.ID)[0]
 $e=@($evidence | Where-Object ID -eq $r.ID)[0]
 if(!$counts.Contains($o.ai_contact_screen)){throw 'Unknown screening value'}
 $counts[$o.ai_contact_screen]++
 if($r.human_review -ne 'NOT_STARTED' -or $r.stage2_eligibility -ne 'UNCONFIRMED'){throw 'Unexpected label state'}
 if((Get-FileHash (Join-Path $data $r.file)).Hash -ine $e.video_sha256 -or $e.video_sha256 -ine $r.expected_sha256){throw 'Video hash mismatch'}
 if((Get-FileHash (Join-Path $out ($r.ID+'.png'))).Hash -ine $e.image_sha256){throw 'Evidence hash mismatch'}
 if($e.decoded_frames -ne 60 -or $e.relative_pts.Count -ne 60){throw 'Frame count mismatch'}
 for($i=1;$i -lt 60;$i++){if($e.relative_pts[$i] -le $e.relative_pts[$i-1]){throw 'PTS ordering mismatch'}}
 $r | Add-Member NoteProperty ai_contact_screen $o.ai_contact_screen
 $r | Add-Member NoteProperty ai_contact_note $o.ai_contact_note
 $r | Add-Member NoteProperty contact_window_start_seconds ($e.seek_seconds+$e.relative_pts[0])
 $r | Add-Member NoteProperty contact_window_end_seconds ($e.seek_seconds+$e.relative_pts[59])
 $rank=switch($o.ai_contact_screen){CONTACT_SUSPECT{1};UNRESOLVED{2};NO_CONTACT_VISIBLE{3}}
 $r | Add-Member NoteProperty contact_review_order $rank
 $queue+=$r
}
$queue=@($queue | Sort-Object contact_review_order,ID)
$queue | Export-Csv (Join-Path $data 'review-contact-queue.csv') -NoTypeInformation -Encoding UTF8
$queue | Select-Object ID,ai_contact_screen,ai_contact_note,contact_window_start_seconds,contact_window_end_seconds,contact_review_order,human_review,stage2_eligibility | Export-Csv (Join-Path $out 'contact-review.csv') -NoTypeInformation -Encoding UTF8
function Encode-Html($v){[Net.WebUtility]::HtmlEncode([string]$v)}
$names=@{CONTACT_SUSPECT='접촉·충격 의심';UNRESOLVED='판단 불가';NO_CONTACT_VISIBLE='확인 구간에서 접촉 징후 안 보임'}
$html=New-Object Text.StringBuilder
[void]$html.AppendLine('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Stage2 검증 후보 관찰 기록</title><style>body{font:16px sans-serif;max-width:1100px;margin:32px auto;padding:16px;line-height:1.6}article{border-top:1px solid #bbb;padding:20px 0}img{max-width:100%}video{width:640px;max-width:100%}summary,select{cursor:pointer;padding:8px}</style><h1>Stage2 검증 후보 50개 — 관찰 기록</h1><p>AI가 사건 참고시각 1초 전부터 원본 연속 60프레임씩 총 3,000프레임을 스틸 격자로 확인했습니다. 전체 영상 재생·오디오 검수나 사람 판정은 아닙니다. 접촉 의심 13 / 접촉 징후 안 보임 13 / 판단 불가 24. 모든 후보의 충돌 여부와 최종 적합성은 미확정이며 정답 라벨은 작성하지 않았습니다.</p><p>흔들림은 제동·회피·카메라 움직임일 수도 있습니다. 접촉 징후가 보이지 않아도 전체 영상의 비충돌을 뜻하지 않습니다. 원본 50개를 모두 보존합니다.</p><label>관찰 그룹 <select id="filter"><option value="">전체 50</option value="CONTACT_SUSPECT">접촉·충격 의심 13</option><option value="UNRESOLVED">판단 불가 24</option><option value="NO_CONTACT_VISIBLE">접촉 징후 안 보임 13</option></select></label>')
foreach($r in $queue){
 $start=[Math]::Max(0,([double]$r.reference_event_seconds-3)).ToString([Globalization.CultureInfo]::InvariantCulture)
 $url='../../data_raw/stage2-validation-nexar-20260914/'+$r.file+'#t='+$start
 [void]$html.AppendLine(('<article data-group="{0}"><h2>{1} · {2}</h2><p>{3}</p><p>관찰 구간 {4:F3}–{5:F3}초 · 사람 검수 미시작 · 최종 적합성 미확정</p><video controls preload="none" src="{6}"></video><p><a href="{6}">원본 영상 열기</a></p><details><summary>연속 60프레임 펼치기 (왼쪽→오른쪽, 위→아래)</summary><a href="{1}.png"><img loading="lazy" src="{1}.png" alt="{1} 연속 프레임"></a></details></article>' -f $r.ai_contact_screen,$r.ID,(Encode-Html $names[$r.ai_contact_screen]),(Encode-Html $r.ai_contact_note),[double]$r.contact_window_start_seconds,[double]$r.contact_window_end_seconds,(Encode-Html $url)))
}
[void]$html.AppendLine('<script>document.getElementById("filter").addEventListener("change",function(){document.querySelectorAll("article").forEach(a=>{a.hidden=!!this.value&&a.dataset.group!==this.value;if(a.hidden)a.querySelector("video").pause()})})</script></html>')
[IO.File]::WriteAllText((Join-Path $out 'review.html'),$html.ToString(),(New-Object Text.UTF8Encoding($false)))
$report=[ordered]@{
 status='CONTACT_SCREENING_COMPLETE';reviewer='AI visual screening';reviewed_videos=50;consecutive_frames=3000
 method='First 60 native decoded frames after max(0, reference event - 1 second), 5x12 contact sheet at 256x144 per frame; no fps resampling'
 counts=$counts;confirmed_collision_count=$null;final_eligible_count=$null;human_labels_created=0;retained_candidates=50
 limitations=@('Short event-centered contact sheets only; no full-video playback or audio adjudication','Camera shake is indirect evidence and may reflect braking or avoidance','NO_CONTACT_VISIBLE refers only to the inspected window, not confirmed near miss','Recording-source independence and final Stage2 eligibility remain unconfirmed')
 verification=@{video_sha256='PASS_50';image_sha256='PASS_50';frame_count_and_monotonic_pts='PASS_3000';ids='PASS_50_UNIQUE';original_inventory_sha256=$originalHash;prior_queue_sha256=$beforeQueueHash;human_review='NOT_STARTED_50';stage2_eligibility='UNCONFIRMED_50'}
 updated=(Get-Date -Format o)
}
$report | ConvertTo-Json -Depth 7 | Set-Content (Join-Path $out 'report.json') -Encoding UTF8
@{status='CONTACT_SCREENING_COMPLETE';completed=50;total=50;decoded_frames=3000;review_complete=$true;reviewer='AI';human_review_complete=$false;counts=$counts;updated=(Get-Date -Format o)} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $out 'status.json') -Encoding UTF8
if((Get-FileHash $original).Hash -ne $originalHash -or (Get-FileHash (Join-Path $data 'review-queue.csv')).Hash -ne $beforeQueueHash){throw 'Prior inventory modified'}
'PASS: 50 videos, 3000 frame timestamps, 50 evidence hashes; 13 suspect / 13 no visible contact / 24 unresolved; labels untouched'
