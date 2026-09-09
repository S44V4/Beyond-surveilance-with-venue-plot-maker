param([string]$Python = "C:\Users\abhij\Documents\Beyond Surveillance\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot
& $Python -u -m research.run
