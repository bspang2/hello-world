@echo off
REM Rogue Sniper AI - one-time setup. Run from anywhere; double-click works.
setlocal
cd /d "%~dp0\..\.."

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")
%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)" 2>nul
if errorlevel 1 (
  echo [!] Python 3.11+ not found. Install it with:  winget install Python.Python.3.12
  echo     then run this script again.
  pause & exit /b 1
)

if not exist .venv (
  echo Creating virtual environment...
  %PY% -m venv .venv || (echo [!] venv creation failed & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
python -m pip install -r requirements-dev.txt || (echo [!] pip install failed & pause & exit /b 1)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo.
  echo [!] FFmpeg not found on PATH. Install it with:  winget install Gyan.FFmpeg
  echo     then close and reopen this window and run install.bat again.
  pause & exit /b 1
)

echo.
echo === Creating ROGUE SNIPER workspace and detecting ShadowPlay / OBS folders ===
python -m rogue_sniper init --add-detected
echo.
echo === Environment check (tests GPU encoders, takes ~10s) ===
python -m rogue_sniper env-check
echo.
echo Setup complete. Start the app with scripts\windows\start.bat
pause
