param(
    [int]$Port = 7860
)

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if (-not $connections) {
    Write-Output "No process is listening on port $Port."
    exit 0
}

$pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($processId in $pids) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Write-Output "Stopped process $processId on port $Port."
}
