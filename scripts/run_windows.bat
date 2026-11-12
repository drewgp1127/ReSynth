@echo off
rem ReSynth for Windows: double-click to start. The first run installs everything (takes a while).
rem Pass --lan to let other devices on your network open the app, e.g.  run_windows.bat --lan
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo First run: installing ReSynth. This downloads about 1 GB and can take 10-20 minutes.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
    if errorlevel 1 (
        echo.
        echo Setup failed. Read the messages above, fix the problem, then double-click this file again.
        pause
        exit /b 1
    )
)

if "%RESYNTH_PORT%"=="" set RESYNTH_PORT=8000
echo Starting ReSynth. Leave this window open while you use the app; close it to stop ReSynth.
set RESYNTH_OPEN_BROWSER=1
".venv\Scripts\python.exe" -m app.main %*
echo.
echo ReSynth stopped.
pause
