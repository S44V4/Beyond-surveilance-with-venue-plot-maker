param([ValidateSet('transport','direct','state_only')][string]$Mode = 'transport')
$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
foreach ($seed in @(23037,23038,23039)) {
    $runFolder = Join-Path $packageRoot "runs\$Mode-$seed"
    if (Test-Path -LiteralPath (Join-Path $runFolder 'run.lock')) {
        New-Item -ItemType File -Path (Join-Path $runFolder 'PAUSE') -Force | Out-Null
        Write-Host "Pause requested for $Mode-$seed. Wait for the training window to say Paused before shutting down."
    }
}
