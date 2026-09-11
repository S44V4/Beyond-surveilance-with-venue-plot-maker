param([string]$Python = 'py', [string]$TorchIndex = 'https://download.pytorch.org/whl/cu128')
$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $packageRoot
if (-not (Test-Path -LiteralPath '.venv-forecast\Scripts\python.exe')) {
    & $Python -3.11 -m venv .venv-forecast
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 x64, then rerun Setup.ps1.' }
}
$trainingPython = Join-Path $packageRoot '.venv-forecast\Scripts\python.exe'
& $trainingPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $trainingPython -m pip install torch --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed. Consult the official PyTorch Windows/CUDA selector.' }
& $trainingPython -m pip install 'numpy>=1.26,<3'
if ($LASTEXITCODE -ne 0) { throw 'NumPy installation failed' }
& $trainingPython -c "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver and PyTorch installation'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed' }
& $trainingPython -m pip freeze | Set-Content -LiteralPath 'rtx3050\installed-environment.txt'
Write-Host 'Setup finished. Next run the hardware check and then Run-Training.ps1.'
