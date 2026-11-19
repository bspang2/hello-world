@echo off
setlocal
cd /d "%~dp0\..\.."
.venv\Scripts\python.exe -m rogue_sniper env-check
.venv\Scripts\python.exe -m rogue_sniper env-check --json > env-report.json
echo.
echo Full report saved to env-report.json
pause
