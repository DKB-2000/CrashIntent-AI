$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$worker=Join-Path $root 'scripts\Acquire-ImperialStratifiedSet.ps1'
$taskName='CrashIntent-Stage1-Imperial-Stratified'
$action=New-ScheduledTaskAction -Execute 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+$worker+'"') -WorkingDirectory $root
$trigger=New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 36) -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Resumable bounded HTTP Range acquisition and validation of 180 Imperial Stage1 images.' -Force|Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName|Select-Object TaskName,State
