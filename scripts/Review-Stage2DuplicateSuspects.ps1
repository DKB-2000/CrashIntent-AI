$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $root
$out=Join-Path $root 'artifacts/stage2-duplicate-review-20260914'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$pairs=Get-Content artifacts/stage2-validation-inspection-20260914/suspected-pairs.json -Raw | ConvertFrom-Json
$ff=Join-Path $root '.venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
Add-Type -AssemblyName System.Drawing
$font=New-Object Drawing.Font('Arial',14)
$brush=[Drawing.Brushes]::Black
$boards=@()
for($offset=0;$offset -lt $pairs.Count;$offset+=4){
    $count=[Math]::Min(4,$pairs.Count-$offset)
    $board=New-Object Drawing.Bitmap(960,($count*420))
    $g=[Drawing.Graphics]::FromImage($board);$g.Clear([Drawing.Color]::White)
    for($j=0;$j -lt $count;$j++){
        $pair=$pairs[$offset+$j]
        for($k=0;$k -lt 2;$k++){
            if($k -eq 0){$id=$pair.candidate;$t=$pair.candidate_seconds;$path=Join-Path $root ('data_raw/stage2-validation-nexar-20260914/'+$id.Replace('NEXAR_','')+'.mp4')}
            else{$id=$pair.reference;$t=$pair.reference_seconds;$path=Join-Path $root ('data_raw/ccd/videos/Crash-1500/'+$id.Replace('CCD_','')+'.mp4')}
            $png=Join-Path $out ('pair-'+($offset+$j)+'-'+$k+'.png')
            & $ff -v error -nostdin -y -threads 1 -ss ([string]$t) -i $path -an -vf 'fps=2,scale=320:180,tile=3x1' -frames:v 1 -threads 1 $png
            if($LASTEXITCODE -ne 0){throw "Frame extraction failed: $id"}
            $y=$j*420+$k*210
            $g.DrawString("$id | t=$t, +0.5, +1.0 seconds",$font,$brush,5,$y)
            $img=[Drawing.Image]::FromFile($png);$g.DrawImage($img,0,($y+28),960,180);$img.Dispose()
        }
    }
    $target=Join-Path $out ('board-'+[int]($offset/4)+'.png')
    $board.Save($target,[Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$board.Dispose();$boards+=$target
}
$font.Dispose()
$boards | ConvertTo-Json | Set-Content (Join-Path $out 'boards.json') -Encoding UTF8
Write-Output ('Prepared '+$pairs.Count+' comparisons across '+$boards.Count+' boards')
