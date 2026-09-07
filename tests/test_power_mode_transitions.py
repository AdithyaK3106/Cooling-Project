"""
TEST: Dynamic Power/Thermal Mode Transitions Based on Telemetry Risk
Validates that ThermalModeController dynamically escalates and de-escalates modes
(QUIET <-> BALANCED <-> PERFORMANCE <-> FAILSAFE) in response to telemetry without getting stuck.
"""

import sys
import time
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.thermal_mode_controller import ThermalModeController, ThermalMode, WorkloadPhase

def test_dynamic_mode_transitions():
    print("\n" + "=" * 70)
    print("TEST: Dynamic Thermal/Power Mode Transitions")
    print("=" * 70)

    controller = ThermalModeController()
    # Mock hardware query during testing so physical host state doesn't interfere
    controller.hardware_controller.get_current_mode_from_hardware = lambda: controller.hardware_controller.desired_mode

    def reset_hold_timers():
        t = time.time() - 60.0
        controller.last_switch_time = t
        controller.hardware_controller.last_transition_time = t
        controller.hardware_controller.last_attempt_time = t

    # Force initial state to QUIET for predictable testing
    controller.active_mode = ThermalMode.QUIET
    controller.hardware_controller.actual_hardware_mode = "QUIET"
    controller.hardware_controller.desired_mode = "QUIET"
    reset_hold_timers()

    # 1. Baseline IDLE telemetry -> should stay QUIET
    idle_telemetry = {"cpu": 5, "gpu": 0, "cpu_temp": 40, "gpu_temp": 38, "power_draw": 15.0}
    fan_pct, mode_str, _, diag = controller.update(future_risk=0.10, current_telemetry=idle_telemetry)
    assert mode_str == "QUIET", f"Expected QUIET, got {mode_str}"
    print("[PASS] Low telemetry risk maintains QUIET mode")

    # 2. Moderate risk telemetry without massive transient -> should transition QUIET -> BALANCED
    reset_hold_timers()
    controller.gpu_power_history.clear()
    moderate_telemetry = {"cpu": 45, "gpu": 20, "cpu_temp": 55, "gpu_temp": 50, "power_draw": 20.0}
    for _ in range(5):
        fan_pct, mode_str, _, diag = controller.update(future_risk=0.55, current_telemetry=moderate_telemetry)
    assert mode_str == "BALANCED", f"Expected BALANCED mode under moderate load, got {mode_str}"
    print("[PASS] Moderate telemetry load escalates QUIET -> BALANCED mode")

    # 3. High sustained load telemetry (e.g. GPU heavy > 70%) -> should transition BALANCED -> PERFORMANCE
    reset_hold_timers()
    heavy_telemetry = {"cpu": 85, "gpu": 90, "cpu_temp": 75, "gpu_temp": 80, "power_draw": 120.0}
    for _ in range(5):
        fan_pct, mode_str, _, diag = controller.update(future_risk=0.90, current_telemetry=heavy_telemetry)
    assert mode_str == "PERFORMANCE", f"Expected PERFORMANCE mode under high load, got {mode_str}"
    print("[PASS] High telemetry load escalates BALANCED -> PERFORMANCE mode")

    # 4. Workload relaxes back to moderate -> PERFORMANCE -> BALANCED / SILENT_RECOVERY
    reset_hold_timers()
    for _ in range(10):
        fan_pct, mode_str, _, diag = controller.update(future_risk=0.45, current_telemetry=moderate_telemetry)
    assert mode_str in ("BALANCED", "SILENT_RECOVERY"), f"Expected BALANCED or SILENT_RECOVERY on load relaxation, got {mode_str}"
    print(f"[PASS] Relaxing load transitions PERFORMANCE -> {mode_str}")

    # 5. Workload returns to true IDLE -> mode de-escalates to QUIET
    reset_hold_timers()
    for _ in range(15):
        fan_pct, mode_str, _, diag = controller.update(future_risk=0.10, current_telemetry=idle_telemetry)
    assert mode_str == "QUIET", f"Expected QUIET mode after returning to idle baseline, got {mode_str}"
    print("[PASS] Telemetry returning to idle baseline successfully de-escalates mode to QUIET")

    print("\n" + "=" * 70)
    print("ALL POWER MODE TRANSITION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_dynamic_mode_transitions()
