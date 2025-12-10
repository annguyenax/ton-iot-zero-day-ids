@echo off
echo ============================================================
echo Starting Zero-Day IoT Attack Detection Dashboard
echo ============================================================
echo.
echo Dashboard will open in your browser at http://localhost:8501
echo.
echo Features:
echo   - Real-time Network Monitoring
echo   - CSV Upload and Analysis
echo   - Manual Testing Mode
echo.
echo Press Ctrl+C to stop the dashboard
echo ============================================================
echo.

cd src
streamlit run dashboard_zeroday.py

pause
