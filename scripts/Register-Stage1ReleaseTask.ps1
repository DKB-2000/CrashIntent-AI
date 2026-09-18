$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$taskName = 'CrashIntent-Stage1AutoRelease-20260910'
$scriptPath = Join-Path $PSScriptRoot 'Advance-Stage1Release.ps1'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw "Task already exists; inspect it before changing: $taskName"
}
$action = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $scriptPath + '"') -WorkingDirectory $projectRoot
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddHours(12) -RepetitionInterval (New-TimeSpan -Hours 12)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 45) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $repeat -Settings $settings -Principal $principal -Description 'After verified Stage1 training and robustness: private T4 integration, local submit.zip, guarded daily registration. Never submit to competition.' | Out-Null
$stage1StatePath = Join-Path $projectRoot 'artifacts/stage1-auto-release-20260910/status.json'
$stage1State = Get-Content -LiteralPath $stage1StatePath -Raw | ConvertFrom-Json
$stage1State | Add-Member -NotePropertyName scheduler_registered -NotePropertyValue $true -Force
$stage1State | Add-Member -NotePropertyName activation_status -NotePropertyValue 'REGISTERED' -Force
$stage1State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $stage1StatePath -Encoding utf8
Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $projectRoot 'artifacts/stage1-auto-release-20260910/scheduled-task.xml') -Encoding utf8
Get-ScheduledTaskInfo -TaskName $taskName | Select-Object LastRunTime,LastTaskResult,NextRunTime

