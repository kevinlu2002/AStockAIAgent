$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TaskName = "AStockAIAgentWeb"
$ScriptPath = Join-Path $ProjectRoot "scripts\start_web_app.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Start AStockAIAgent web app after Windows sign-in." `
    -Force

Write-Output "Registered Windows startup task: $TaskName"
