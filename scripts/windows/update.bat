@echo off
REM Rogue Sniper AI - download the latest version from GitHub and install it over this folder.
REM Your settings, database, templates and recordings live elsewhere and are not touched.
setlocal
cd /d "%~dp0\..\.."
echo Close start.bat before updating. Downloading latest version...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $z=Join-Path $env:TEMP 'rogue_sniper_update.zip'; $d=Join-Path $env:TEMP 'rogue_sniper_update'; Invoke-WebRequest 'https://github.com/bspang2/hello-world/archive/refs/heads/claude/rogue-sniper-ai-fortnite-tm3smx.zip' -OutFile $z; if (Test-Path $d) { Remove-Item $d -Recurse -Force }; Expand-Archive $z $d; $src=Get-ChildItem $d | Select-Object -First 1; Copy-Item (Join-Path $src.FullName '*') (Get-Location) -Recurse -Force; Write-Host 'Files updated.'" && .venv\Scripts\python.exe -m pip install -q -r requirements.txt && echo Update complete. You can start the app again. & pause & exit /b
