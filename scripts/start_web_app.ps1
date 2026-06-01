param(
    [int]$Port = 7860
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:ASHARE_PORT = [string]$Port
$env:ASHARE_AUTO_NEWS = "1"
$env:ASHARE_NEWS_INTERVAL_SECONDS = "900"
$env:ASHARE_AUTO_KNOWLEDGE = "1"
$env:ASHARE_KNOWLEDGE_INTERVAL_SECONDS = "86400"
$env:ASHARE_AUTO_RETRAIN = "1"
$env:ASHARE_RETRAIN_TIME = "16:30"

Start-Process -FilePath "python" `
    -ArgumentList ".\scripts\run_web_app.py" `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden

Write-Output "AStockAIAgent web app started on http://127.0.0.1:$Port"
