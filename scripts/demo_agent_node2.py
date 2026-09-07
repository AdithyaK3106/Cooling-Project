import sys
import os
import time
import requests
import psutil
import subprocess
import json
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fan_controller import HardwareFanController

# Configure Node 1 IP here (will be the Personal Hotspot IP, usually 192.168.137.1)
NODE1_IP = "192.168.137.1"
NODE1_PORT = 8080
API_URL = f"http://{NODE1_IP}:{NODE1_PORT}/neighbor/sync"
FALLBACK_TIMEOUT = 5.0 # Seconds

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

    # Approximate CPU temp (since WMI is slow, we use a simple synthetic correlation or you can add OHM reading)
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
    print("======================================================")
    print(" THERVO - NODE 2 AGENT (DUMB TERMINAL)")
    print("======================================================")
    print(f"[*] Targeting Node 1 (Brain) at: {API_URL}")
    print(f"[*] Safe-Command Fallback Timeout: {FALLBACK_TIMEOUT}s")
    
    fan_controller = HardwareFanController()
    last_valid_command_time = time.monotonic()
    is_fallback = False
    
    while True:
        telemetry = collect_basic_telemetry()
        
        try:
            # Sync with Node 1: Send telemetry, get command
            response = requests.post(API_URL, json=telemetry, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                fan_target = data.get("fan_percent")
                mode = data.get("mode", "BASELINE")
                
                last_valid_command_time = time.monotonic()
                if is_fallback:
                    print(f"[{telemetry['timestamp']}] Reconnected to Node 1. Resuming THERVO control.")
                    is_fallback = False
                
                print(f"[{telemetry['timestamp']}] Sent Telemetry | CPU: {telemetry['cpu']}% | Received Command: Fan {fan_target}% ({mode})")
                
                # Apply cooling command
                if fan_target is not None:
                    # In real life this calls Toolkit. Here we use our HardwareFanController
                    fan_controller.write_target(risk_score=data.get("risk_score", 0.5), target_percent=fan_target, policy_state=mode, is_stabilizing=False)
                    
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            # Network or server error
            elapsed_since_valid = time.monotonic() - last_valid_command_time
            print(f"[{telemetry['timestamp']}] Sync Failed ({e}). Time since last valid: {elapsed_since_valid:.1f}s")
            
            if elapsed_since_valid > FALLBACK_TIMEOUT and not is_fallback:
                print(f"!!! SAFE-COMMAND FALLBACK TRIGGERED !!!")
                print(f"Lost connection to Node 1 for >{FALLBACK_TIMEOUT}s. Relinquishing fan control to BIOS.")
                is_fallback = True
                # Call fan_controller with a safe state
                fan_controller.write_target(risk_score=1.0, target_percent=100.0, policy_state="FALLBACK", is_stabilizing=False)
                
        time.sleep(1.0)

if __name__ == "__main__":
    main()
