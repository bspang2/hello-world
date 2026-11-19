@echo off
REM Rogue Sniper AI - start backend + folder monitor and open the dashboard.
setlocal
cd /d "%~dp0\..\.."
if not exist .venv\Scripts\python.exe (
  echo Run scripts\windows\install.bat first.
  pause & exit /b 1
)
start "" http://127.0.0.1:8765/
.venv\Scripts\python.exe -m rogue_sniper serve --host 127.0.0.1 --port 8765
pause
