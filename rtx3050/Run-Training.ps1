param(
    [string]$Data = 'data',
    [ValidateSet('transport','direct','state_only')][string]$Mode = 'transport',
    [int[]]$Seeds = @(23037,23038,23039),
    [string]$Python = '',
    [string]$Config = 'configs\forecast_v2_rtx3050.json'
)
$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $packageRoot
if (-not $Python) { $Python = Join-Path $packageRoot '.venv-forecast\Scripts\python.exe' }
foreach ($seed in $Seeds) {
    $runFolder = Join-Path 'runs' "$Mode-$seed"
    & $Python -m forecast_v2.cli train --data $Data --run $runFolder --config $Config --mode $Mode --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Training stopped with an error in $runFolder. Last saved checkpoint is preserved." }
    $state = Get-Content -Raw -LiteralPath (Join-Path $runFolder 'status.json') | ConvertFrom-Json
    if ($state.stage -eq 'paused') { Write-Host 'Paused. Rerun this same command to continue.'; return }
}
Write-Host 'Selected runs completed. Review validation histories before freezing the model for calibration.'
