@echo off
rem Glide desktop app. No console window; it lives in the tray when its window is closed.
cd /d "%~dp0"
if "%~1"=="" (
  start "" ".venv\Scripts\pythonw.exe" -m glide
) else (
  ".venv\Scripts\python.exe" -m glide %*
)
