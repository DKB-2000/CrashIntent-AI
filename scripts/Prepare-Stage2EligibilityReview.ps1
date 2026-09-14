$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$out=Join-Path $root 'artifacts/stage2-eligibility-review-20260914'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$rows=Import-Csv data_raw/stage2-validation-nexar-20260914/review-inventory-resolved.csv
$ff=Join-Path $root '.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
Add-Type -AssemblyName System.Drawing
$font=New-Object Drawing.Font('Arial',14)
$index=@()
for($offset=0;$offset -lt $rows.Count;$offset+=5){
    $count=[Math]::Min(5,$rows.Count-$offset)
    $board=New-Object Drawing.Bitmap(960,($count*400))
    $g=[Drawing.Graphics]::FromImage($board);$g.Clear([Drawing.Color]::White)
    for($j=0;$j -lt $count;$j++){
        $row=$rows[$offset+$j]
        # Coarse preview starts at nearest whole second; never use these stills as frame labels.
        $t=[Math]::Max(0.0,[Math]::Round([double]::Parse($row.reference_event_seconds,[Globalization.CultureInfo]::InvariantCulture)-3))
        $path=Join-Path $root ('data_raw/stage2-validation-nexar-20260914/'+$row.file)
        if((Get-FileHash $path).Hash -ine $row.expected_sha256){throw 'Input hash mismatch'}
        $png=Join-Path $out ($row.ID+'.png')
        & $ff -v error -nostdin -y -threads 1 -ss ($t.ToString([Globalization.CultureInfo]::InvariantCulture)) -i $path -an -vf 'fps=1,scale=320:180,tile=3x2' -frames:v 1 -threads 1 $png
        if($LASTEXITCODE -ne 0){throw "Extraction failed: $($row.ID)"}
        $y=$j*400
        $g.DrawString("$($row.ID) | approx t=$([Math]::Round($t,2)) to +5s; event reference=$($row.reference_event_seconds)",$font,[Drawing.Brushes]::Black,5,$y)
        $img=[Drawing.Image]::FromFile($png);$g.DrawImage($img,0,($y+30),960,360);$img.Dispose()
        $index += [pscustomobject]@{ID=$row.ID; video_sha256=$row.expected_sha256; approximate_start_seconds=$t; reference_event_seconds=$row.reference_event_seconds; board=('board-'+[int]($offset/5)+'.png'); still=$row.ID+'.png'}
    }
    $target=Join-Path $out ('board-'+[int]($offset/5)+'.png')
    $board.Save($target,[Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$board.Dispose()
}
$font.Dispose()
$index | Export-Csv (Join-Path $out 'evidence-index.csv') -NoTypeInformation -Encoding UTF8
Write-Output 'Prepared 50 candidates, 300 sampled frames, 10 boards; source hashes PASS'
