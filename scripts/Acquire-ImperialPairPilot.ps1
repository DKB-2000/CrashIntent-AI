param([string]$OutputDir = 'artifacts/stage1-imperial-full-pilot-20260918')
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$statusPath = Join-Path $OutputDir 'status.json'
@{status='RUNNING';updated=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content $statusPath
function Get-FirstImage([string]$Url,[long]$End,[string]$OutName) {
    $prefix = Join-Path $OutputDir ($OutName + '.prefix.bin')
    $dest=Join-Path $OutputDir $OutName
    if (Test-Path -LiteralPath $dest) {
        $sha=[Security.Cryptography.SHA256]::Create(); $stream=[IO.File]::OpenRead((Resolve-Path $dest)); $digest=$sha.ComputeHash($stream); $stream.Dispose(); $sha.Dispose()
        return @{archive_name='reused_verified_extraction';path=$dest;bytes=(Get-Item $dest).Length;sha256=([BitConverter]::ToString($digest).Replace('-','').ToLower())}
    }
    & curl.exe --fail --location --retry 3 --max-time 900 --range "0-$End" --output $prefix $Url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $OutName" }
    $b = [IO.File]::ReadAllBytes((Resolve-Path $prefix))
    $p = 0
    while ([BitConverter]::ToUInt32($b,$p) -eq 0x04034b50) {
        $method=[BitConverter]::ToUInt16($b,$p+8); $size=[BitConverter]::ToUInt32($b,$p+18)
        $nameLen=[BitConverter]::ToUInt16($b,$p+26); $extraLen=[BitConverter]::ToUInt16($b,$p+28)
        $name=[Text.Encoding]::UTF8.GetString($b,$p+30,$nameLen); $start=$p+30+$nameLen+$extraLen
        if ($name -match '\.(JPG|png)$') {
            if ($method -ne 8) { throw "unexpected compression method $method" }
            $input=[IO.MemoryStream]::new($b,$start,$size,$false)
            $deflate=[IO.Compression.DeflateStream]::new($input,[IO.Compression.CompressionMode]::Decompress)
            $output=[IO.File]::Create($dest); $deflate.CopyTo($output); $output.Dispose(); $deflate.Dispose(); $input.Dispose()
            $sha=[Security.Cryptography.SHA256]::Create(); $stream=[IO.File]::OpenRead((Resolve-Path $dest)); $digest=$sha.ComputeHash($stream); $stream.Dispose(); $sha.Dispose()
            return @{archive_name=$name;path=$dest;bytes=(Get-Item $dest).Length;sha256=([BitConverter]::ToString($digest).Replace('-','').ToLower())}
        }
        $p=$start+$size
    }
    throw "no image entry found in prefix"
}
try {
    $original=Get-FirstImage 'https://www.commsp.ee.ic.ac.uk/~pld/research/Rewind/Recapture/TestImages/SingleCaptureImages.zip' 2700000 'original_015.jpg'
    $recaptured=Get-FirstImage 'https://www.commsp.ee.ic.ac.uk/~pld/research/Rewind/Recapture/TestImages/RecapturedImages.zip' 3550000 'recaptured_015.png'
    @{status='ACQUIRED';updated=(Get-Date).ToString('o');content_id=15;original=$original;recaptured=$recaptured} | ConvertTo-Json -Depth 4 | Set-Content $statusPath
} catch {
    @{status='FAILED';updated=(Get-Date).ToString('o');error=$_.Exception.Message} | ConvertTo-Json | Set-Content $statusPath
    throw
}
