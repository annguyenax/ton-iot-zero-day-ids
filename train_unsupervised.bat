@echo off
echo ====================================================================
echo    UNSUPERVISED TRAINING - TRUE ZERO-DAY DETECTION
echo ====================================================================
echo.
echo Approach:
echo   - Train ONLY on normal traffic
echo   - Model learns "what is normal"
echo   - ANY deviation is detected as attack/zero-day
echo.
echo This will take approximately 15-20 minutes...
echo ====================================================================
echo.

cd src
python train_unsupervised.py

echo.
echo ====================================================================
echo Training complete!
echo.
echo Next steps:
echo   1. Test models: test_unsupervised.bat
echo   2. Run dashboard: run_dashboard_unsupervised.bat
echo ====================================================================
pause
