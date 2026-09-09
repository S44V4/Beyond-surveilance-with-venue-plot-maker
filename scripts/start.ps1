param([string]$Python = "", [switch]$Dev)
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot
if (-not $Python) {
    $candidates = @("$projectRoot\.venv\Scripts\python.exe", "C:\Users\abhij\Documents\Beyond Surveillance\.venv\Scripts\python.exe")
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Python) { $Python = "python" }
}
if (-not (Test-Path -LiteralPath "$projectRoot\models\balanced_ddpf.pt")) { throw "Import the supplied DDPF checkpoint with scripts/import-checkpoint.py first." }
if (-not $Dev -and -not (Test-Path -LiteralPath "$projectRoot\dist\index.html")) { & npm.cmd run build; if ($LASTEXITCODE) { throw "Dashboard build failed" } }
Write-Host "Beyond Surveillance: http://127.0.0.1:8010"
Write-Host "The API starts its persistent worker automatically. Ctrl+C stops both."
& $Python -m uvicorn backend.api:app --host 127.0.0.1 --port 8010

