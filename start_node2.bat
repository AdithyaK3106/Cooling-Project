@echo off
:: Check for Administrator privileges (required for Lenovo Toolkit CLI)
net session >nul 2>&1
if %errorLevel% == 0 (
    goto :run
) else (
    echo Requesting Administrator privileges to interact with Lenovo Legion Toolkit...
    powershell -Command "Start-Process '%~dpnx0' -ArgumentList '%*' -Verb RunAs"
    exit /b
)

:run
cd /d "%~dp0"
echo ======================================================
echo THERVO - NODE 2 (NEIGHBOR AGENT) LAUNCHER
echo ======================================================
echo Ensure you are connected to Node 1's Personal Hotspot or Local Network!
echo Starting THERVO Node 2 Agent...
python scripts\demo_agent_node2.py %*
pause
