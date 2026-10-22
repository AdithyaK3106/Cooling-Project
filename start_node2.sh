#!/usr/bin/env bash
# ==============================================================================
# THERVO - Node 2 (Neighbor Agent) Launcher
# Cross-platform: Git Bash, WSL, macOS, and Linux
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo " THERVO - NODE 2 (NEIGHBOR AGENT) LAUNCHER"
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

echo "[*] Using Python: $($PYTHON_CMD --version) ($PYTHON_CMD)"
echo "[*] Ensure you are connected to Node 1's network or hotspot."
echo "[*] Starting THERVO Node 2 Agent..."
echo "======================================================"

exec "$PYTHON_CMD" scripts/demo_agent_node2.py "$@"
