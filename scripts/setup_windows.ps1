# ReSynth setup for Windows 10/11.
# Creates .venv in the project folder and installs everything in the README's order.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# (run_windows.bat calls this automatically the first time.)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Find-Python {
    foreach ($cand in @(@("py", "-3.13"), @("python"))) {
        try {
            $exe = $cand[0]
            $rest = @($cand | Select-Object -Skip 1)
            $ver = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver -eq "3.13") { return ,$cand }
        } catch { }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host ""
    Write-Host "Python 3.13 was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this again."
    exit 1
}
$pyExe = $py[0]
$pyArgs = @($py | Select-Object -Skip 1)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating the virtual environment (.venv)..."
    & $pyExe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "could not create .venv" }
}
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "Upgrading pip..."
& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "Installing CPU-only PyTorch (for stem separation; this is a large download)..."
& $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

Write-Host "Installing ReSynth and its dependencies..."
& $venvPy -m pip install -e ".[dev,analysis,separation,web,claude]"
if ($LASTEXITCODE -ne 0) { throw "ReSynth install failed" }

Write-Host "Installing basic-pitch (note detection) without its TensorFlow requirement..."
& $venvPy -m pip install --no-deps basic-pitch==0.4.0
if ($LASTEXITCODE -ne 0) { throw "basic-pitch install failed" }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "Note: ffmpeg was not found. WAV, FLAC and MP3 work without it; M4A/AAC uploads need it." -ForegroundColor Yellow
    Write-Host "See 'Windows setup' in README.md for how to install it."
}
if (-not $env:RESYNTH_ANTHROPIC_KEY) {
    Write-Host ""
    Write-Host "Note: RESYNTH_ANTHROPIC_KEY is not set. ReSynth works without it, but skips gear research." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Setup finished. Double-click scripts\run_windows.bat to start ReSynth." -ForegroundColor Green
