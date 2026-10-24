import sys
import os
import time
import requests
import psutil
import subprocess
import argparse
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fan_controller import HardwareFanController
from src.hardware.thermal_mode_controller import ThermalModeController

def collect_basic_telemetry():
    """Lightweight telemetry collection for Node 2."""
    sys_cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    
    gpu_util = 0.0
    gpu_temp = 0.0
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1
        )
        if res.returncode == 0:
            parts = res.stdout.strip().split(",")
            if len(parts) >= 2:
                gpu_util = float(parts[0].strip())
                gpu_temp = float(parts[1].strip())
    except Exception:
        pass

    # Approximate CPU temp (since WMI is slow, use simple synthetic correlation or sensors)
    cpu_temp = 40.0 + 0.4 * sys_cpu

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": round(sys_cpu, 2),
        "gpu": round(gpu_util, 2),
        "memory": round(mem, 2),
        "cpu_temp": round(cpu_temp, 1),
        "gpu_temp": round(gpu_temp, 1),
        "disk_io": 0.0,
        "network_io": 0.0,
        "cpu_power": 15.0,
        "gpu_power": 20.0,
        "cpu_util": round(sys_cpu, 2),
        "gpu_util": round(gpu_util, 2),
        "mem_util": round(mem, 2),
        "power_draw": 35.0
    }

def main():
    parser = argparse.ArgumentParser(description="THERVO - Node 2 Agent (Neighbor Device)")
    parser.add_argument("--host", default=os.environ.get("THERVO_MAIN_IP", "192.168.137.1"), help="Node 1 Brain IP address (default: 192.168.137.1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("THERVO_MAIN_PORT", 8080)), help="Node 1 Brain API port (default: 8080)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Safe-command fallback timeout in seconds (default: 5.0)")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/neighbor/sync"

    print("======================================================")
    print(" THERVO - NODE 2 AGENT (NEIGHBOR / SECONDARY NODE)")
    print("======================================================")
    print(f"[*] Targeting Node 1 (Brain) at: {api_url}")
    print(f"[*] Safe-Command Fallback Timeout: {args.timeout}s")
    
    fan_controller = HardwareFanController()
    try:
        thermal_controller = ThermalModeController()
    except Exception as e:
        thermal_controller = None
        print(f"[*] Note: Hardware thermal mode controller unavailable ({e}); running fan emulation mode.")

    last_valid_command_time = time.monotonic()
    is_fallback = False
    
    while True:
        telemetry = collect_basic_telemetry()
        
        try:
            # Sync with Node 1: Send telemetry, get command
            response = requests.post(api_url, json=telemetry, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                fan_target = data.get("fan_percent")
                mode = data.get("mode", "BASELINE")
                risk_score = data.get("risk_score", 0.5)
                
                last_valid_command_time = time.monotonic()
                if is_fallback:
                    print(f"[{telemetry['timestamp']}] Reconnected to Node 1. Resuming THERVO control.")
                    is_fallback = False
                
                print(f"[{telemetry['timestamp']}] Sent Telemetry | CPU: {telemetry['cpu']}% | Received Command: Fan {fan_target}% ({mode})")
                
                # Apply cooling command
                if fan_target is not None:
                    fan_controller.write_target(
                        future_risk=risk_score,
                        target_fan_percent=fan_target,
                        policy_state=mode,
                        stabilization_active=False
                    )
                    
                    # Also actually apply the Lenovo toolkit mode if available
                    if thermal_controller is not None:
                        toolkit_mode = "BALANCED"
                        if fan_target > 70:
                            toolkit_mode = "PERFORMANCE"
                        elif fan_target < 30:
                            toolkit_mode = "QUIET"
                            
                        try:
                            thermal_controller.set_mode(toolkit_mode, reason="THERVO_SYNC", severity="HIGH" if toolkit_mode == "PERFORMANCE" else "LOW")
                            thermal_controller.reconcile()
                        except Exception:
                            pass
                        
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            # Network or server error
            elapsed_since_valid = time.monotonic() - last_valid_command_time
            print(f"[{telemetry['timestamp']}] Sync Failed ({e}). Time since last valid: {elapsed_since_valid:.1f}s")
            
            if elapsed_since_valid > args.timeout and not is_fallback:
                print(f"!!! SAFE-COMMAND FALLBACK TRIGGERED !!!")
                print(f"Lost connection to Node 1 for >{args.timeout}s. Relinquishing fan control to BIOS.")
                is_fallback = True
                fan_controller.write_target(
                    future_risk=1.0,
                    target_fan_percent=100.0,
                    policy_state="FALLBACK",
                    stabilization_active=False
                )
                if thermal_controller is not None:
                    try:
                        thermal_controller.set_mode("FAILSAFE", reason="DISCONNECT_TIMEOUT", severity="CRITICAL")
                        thermal_controller.reconcile()
                    except Exception:
                        pass
                
        time.sleep(1.0)

if __name__ == "__main__":
    main()
