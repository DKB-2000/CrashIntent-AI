$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$worker = Join-Path $root 'src\scout_stage3_civic_5_35_classes.py'
$taskName = 'CrashIntent-Stage3-Civic535-Scout'

if (-not (Test-Path -LiteralPath $python)) { throw "Python not found: $python" }
if (-not (Test-Path -LiteralPath $worker)) { throw "Worker not found: $worker" }

$action = New-ScheduledTaskAction -Execute $python -Argument ('"' + $worker + '" watch') -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 35) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Finite resumable sensor-only Stage3 Civic 5-35 class scout.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTaskInfo -TaskName $taskName | Select-Object TaskName,LastRunTime,LastTaskResult,NextRunTime
