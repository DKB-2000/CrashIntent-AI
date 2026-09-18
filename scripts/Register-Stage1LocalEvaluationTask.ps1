$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$taskName = 'CrashIntent-Stage1LocalEvaluation-20260911'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw "Task already exists: $taskName" }
$scriptPath = Join-Path $PSScriptRoot 'Supervise-Stage1Robustness.ps1'
$action = New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -File "' + $scriptPath + '"') -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 2)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 24) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Local Stage1 evaluation only: verified 25-video chunks, bounded crash/stall recovery. No uploads; preserve separate 12-hour ZIP release task.' | Out-Null
Export-ScheduledTask -TaskName $taskName | Set-Content (Join-Path $projectRoot 'artifacts/stage1-robustness-20260910/full-model-evaluation/supervisor-task.xml') -Encoding utf8
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTaskInfo -TaskName $taskName | Select-Object LastRunTime,LastTaskResult,NextRunTime
