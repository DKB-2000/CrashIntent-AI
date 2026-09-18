param(
    [Parameter(Mandatory=$true)][string]$TailPath,
    [Parameter(Mandatory=$true)][long]$TotalSize,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$bytes = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $TailPath))
$base = $TotalSize - $bytes.LongLength
$eocd = -1
for ($i = $bytes.Length - 22; $i -ge 0; $i--) {
    if ($bytes[$i] -eq 0x50 -and $bytes[$i+1] -eq 0x4b -and $bytes[$i+2] -eq 0x05 -and $bytes[$i+3] -eq 0x06) { $eocd = $i; break }
}
if ($eocd -lt 0) { throw 'ZIP EOCD not present in tail' }
$entries = [BitConverter]::ToUInt16($bytes, $eocd + 10)
$cdSize = [BitConverter]::ToUInt32($bytes, $eocd + 12)
$cdOffset = [BitConverter]::ToUInt32($bytes, $eocd + 16)
if ($entries -eq [uint16]::MaxValue -or $cdOffset -eq [uint32]::MaxValue) {
    $locator = $eocd - 20
    if ($locator -lt 0 -or [BitConverter]::ToUInt32($bytes, $locator) -ne 0x07064b50) { throw 'ZIP64 locator missing' }
    $zip64Absolute = [BitConverter]::ToUInt64($bytes, $locator + 8)
    $zip64 = [int]($zip64Absolute - $base)
    if ($zip64 -lt 0 -or [BitConverter]::ToUInt32($bytes, $zip64) -ne 0x06064b50) { throw 'ZIP64 EOCD missing from tail' }
    $entries = [BitConverter]::ToUInt64($bytes, $zip64 + 32)
    $cdSize = [BitConverter]::ToUInt64($bytes, $zip64 + 40)
    $cdOffset = [BitConverter]::ToUInt64($bytes, $zip64 + 48)
}
$names = [Collections.Generic.List[string]]::new()
$entryDetails = [Collections.Generic.List[object]]::new()
if ($cdOffset -ge $base) {
    $pos = [int]($cdOffset - $base)
    $cdEnd = $pos + [int]$cdSize
    while ($pos -lt $cdEnd) {
        if ([BitConverter]::ToUInt32($bytes, $pos) -ne 0x02014b50) { throw "Bad central directory at $pos" }
        $nameLen = [BitConverter]::ToUInt16($bytes, $pos + 28)
        $extraLen = [BitConverter]::ToUInt16($bytes, $pos + 30)
        $noteLen = [BitConverter]::ToUInt16($bytes, $pos + 32)
        $name = [Text.Encoding]::UTF8.GetString($bytes, $pos + 46, $nameLen)
        $names.Add($name)
        $compressed = [uint64][BitConverter]::ToUInt32($bytes, $pos + 20)
        $uncompressed = [uint64][BitConverter]::ToUInt32($bytes, $pos + 24)
        $localOffset = [uint64][BitConverter]::ToUInt32($bytes, $pos + 42)
        if ($compressed -eq [uint32]::MaxValue -or $uncompressed -eq [uint32]::MaxValue -or $localOffset -eq [uint32]::MaxValue) {
            $extraPos = $pos + 46 + $nameLen
            $extraEnd = $extraPos + $extraLen
            while ($extraPos + 4 -le $extraEnd) {
                $tag = [BitConverter]::ToUInt16($bytes, $extraPos)
                $fieldLen = [BitConverter]::ToUInt16($bytes, $extraPos + 2)
                if ($tag -eq 1) {
                    $valuePos = $extraPos + 4
                    if ($uncompressed -eq [uint32]::MaxValue) { $uncompressed = [BitConverter]::ToUInt64($bytes, $valuePos); $valuePos += 8 }
                    if ($compressed -eq [uint32]::MaxValue) { $compressed = [BitConverter]::ToUInt64($bytes, $valuePos); $valuePos += 8 }
                    if ($localOffset -eq [uint32]::MaxValue) { $localOffset = [BitConverter]::ToUInt64($bytes, $valuePos) }
                    break
                }
                $extraPos += 4 + $fieldLen
            }
        }
        $entryDetails.Add([ordered]@{
            name = $name
            method = [BitConverter]::ToUInt16($bytes, $pos + 10)
            crc32 = ('{0:x8}' -f [BitConverter]::ToUInt32($bytes, $pos + 16))
            compressed_size = $compressed
            uncompressed_size = $uncompressed
            local_header_offset = $localOffset
        })
        $pos += 46 + $nameLen + $extraLen + $noteLen
    }
}
$report = [ordered]@{
    status = $(if ($cdOffset -ge $base) {'CENTRAL_DIRECTORY_PARSED'} else {'NEED_MORE_TAIL'})
    total_size = $TotalSize
    tail_size = $bytes.LongLength
    central_directory_offset = $cdOffset
    central_directory_size = $cdSize
    entries_declared = $entries
    entries_parsed = $names.Count
    required_tail_bytes = $TotalSize - $cdOffset
    names = $names
    entry_details = $entryDetails
}
$parent = Split-Path -Parent $OutputPath
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$report | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
$report | Select-Object * -ExcludeProperty names | Format-List
