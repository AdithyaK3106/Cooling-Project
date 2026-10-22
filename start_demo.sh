#!/usr/bin/env bash
# ==============================================================================
# THERVO - Single-Node Full Demonstration Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo " THERVO - MISSION CONTROL DEMONSTRATION LAUNCHER"
echo "======================================================"

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

echo "[*] Using Python: $($PYTHON_CMD --version)"
echo "[*] Launching THERVO demonstration..."
exec "$PYTHON_CMD" scripts/run_demo.py "$@"
