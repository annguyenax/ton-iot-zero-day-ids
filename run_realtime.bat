@echo off
echo Starting Real-time Detector...
cd /d "%~dp0\src"
..\.venv\Scripts\python.exe realtime_simple.py
pause
