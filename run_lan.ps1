# Launch the AI Study Assistant on the local network.
# Any device on your Wi-Fi/Ethernet can open the URL printed below.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Detect the primary LAN IPv4 (skip loopback / link-local / VMware adapters)
$ip = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike '127.*' -and
        $_.IPAddress -notlike '169.*' -and
        $_.InterfaceAlias -notmatch 'VMware|Loopback|vEthernet'
    } |
    Sort-Object -Property InterfaceMetric |
    Select-Object -First 1 -ExpandProperty IPAddress

if (-not $ip) { $ip = "0.0.0.0" }

Write-Host ""
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor DarkYellow
Write-Host "  🎓  AI Study Assistant — LAN deployment" -ForegroundColor Yellow
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor DarkYellow
Write-Host ("  Local       : http://localhost:8502")
Write-Host ("  Network     : http://{0}:8502" -f $ip) -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the server"
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor DarkYellow
Write-Host ""

$env:PYTHONIOENCODING = "utf-8"

python -m streamlit run app.py `
    --server.headless=true `
    --server.address=0.0.0.0 `
    --server.port=8502 `
    --browser.serverAddress=$ip
