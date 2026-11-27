@echo off
title Dojo 24
cd /d "%~dp0"
if not exist "engine\venv\Scripts\python.exe" (
  echo The training engine is not installed yet. Run setup.bat first.
  pause
  exit /b 1
)
"engine\venv\Scripts\python.exe" -m app.server --open
if errorlevel 1 pause
