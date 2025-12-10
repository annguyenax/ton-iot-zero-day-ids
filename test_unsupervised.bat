@echo off
echo ====================================================================
echo    TESTING UNSUPERVISED MODELS
echo ====================================================================
echo.
echo Testing zero-day detection capability:
echo   - Normal samples should NOT be flagged
echo   - Attack samples should be DETECTED as zero-day
echo.
echo ====================================================================
echo.

cd src
python test_unsupervised.py

echo.
pause
