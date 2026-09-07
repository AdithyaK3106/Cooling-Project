@echo off
:: Check for Administrator privileges (required for Toolkit)
net session >nul 2>&1
if %errorLevel% == 0 (
    goto :run
) else (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process '%~dpnx0' -Verb RunAs"
    exit /b
)

:run
cd /d "%~dp0"
echo ======================================================
echo THERVO - TWO-NODE DEMONSTRATION LAUNCHER (MAIN)
echo ======================================================
echo Ensure Node 2 is connected to this laptop's Personal Hotspot!
echo Starting THERVO Brain and Dashboard...
python scripts\demo_brain_node1.py
pause
