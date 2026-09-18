$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$worker = Join-Path $root 'src\evaluate_stage3_civic_535_balanced.py'
$taskName = 'CrashIntent-Stage3-Civic535-Evaluation'
$action = New-ScheduledTaskAction -Execute $python -Argument ('"' + $worker + '" watch') -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 35) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Frozen sensor-proxy evaluation of four Stage3 families over three seeds.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTaskInfo -TaskName $taskName | Select-Object TaskName,LastRunTime,LastTaskResult,NextRunTime
