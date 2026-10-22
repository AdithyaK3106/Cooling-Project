#!/usr/bin/env bash
# ==============================================================================
# THERVO - Dual-Node Demonstration Launcher (Main Node / Node 1)
# Cross-platform: Git Bash, WSL, macOS, and Linux
# ==============================================================================

set -e

# Resolve project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo " THERVO - DUAL-NODE MISSION CONTROL & MAIN NODE (1)"
echo "======================================================"
echo "[*] Working directory: $SCRIPT_DIR"

# Detect Python interpreter (.venv on Windows vs Unix or system python)
if [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_CMD="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "[!] Error: Python not found. Please install Python or set up .venv."
    exit 1
fi

echo "[*] Using Python: $($PYTHON_CMD --version) ($PYTHON_CMD)"
echo "[*] Launching Node 1 Brain, Hardware Telemetry Engine & Dashboard..."
echo "[*] Dashboard will be available at: http://localhost:3000/"
echo "[*] Telemetry API will listen at:    http://localhost:8080/"
echo "------------------------------------------------------"
echo "[*] If connecting Node 2, ensure it is on this machine's network/hotspot."
echo "[*] Press Ctrl+C to stop all services."
echo "======================================================"

exec "$PYTHON_CMD" scripts/demo_brain_node1.py "$@"
