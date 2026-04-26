import time
import threading
import random
import os
import requests
import sys

# -----------------------------
# CONFIG
# -----------------------------
PHASE_DURATION = 60
BURST_CHANCE = 0.2

# Path for temporary disk stress (aligned with project structure)
TEMP_STRESS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "workload_temp.txt")

# -----------------------------
# CPU LOAD
# -----------------------------
def cpu_stress(duration):
    """Generates CPU load by performing continuous calculations."""
    end = time.time() + duration
    while time.time() < end:
        _ = sum(i*i for i in range(10000))


# -----------------------------
# MEMORY LOAD
# -----------------------------
def memory_stress(duration):
    """Generates Memory load by allocating and deallocating large strings."""
    end = time.time() + duration
    data = []
    while time.time() < end:
        data.append("X" * 5_000_000)
        if len(data) > 10:
            data.pop(0)


# -----------------------------
# DISK LOAD
# -----------------------------
def disk_stress(duration):
    """Generates Disk I/O by writing and deleting a temporary file."""
    end = time.time() + duration
    os.makedirs(os.path.dirname(TEMP_STRESS_FILE), exist_ok=True)
    while time.time() < end:
        try:
            with open(TEMP_STRESS_FILE, "w") as f:
                f.write("A" * 5_000_000)
            os.remove(TEMP_STRESS_FILE)
        except OSError:
            pass


# -----------------------------
# NETWORK LOAD
# -----------------------------
def network_stress(duration):
    """Generates Network I/O by downloading a large test file."""
    end = time.time() + duration
    url = "https://speed.hetzner.de/100MB.bin"
    while time.time() < end:
        try:
            # stream=True and iterating ensures actual network traffic is generated
            response = requests.get(url, timeout=5, stream=True)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if time.time() >= end:
                    break
        except Exception:
            pass


# -----------------------------
# GPU LOAD (ADAPTIVE)
# -----------------------------
def gpu_stress(duration):
    """Generates GPU load using PyTorch if available."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
            
        device = "cuda"
        x = None

        # Try progressively safer sizes
        for size in [2048, 1536, 1024]:
            try:
                x = torch.randn(size, size, device=device)
                print(f"[GPU] Stress active with tensor size: {size}x{size}")
                break
            except Exception:
                continue

        if x is None:
            raise RuntimeError("GPU allocation failed")

        end = time.time() + duration
        while time.time() < end:
            x = torch.matmul(x, x)

    except Exception as e:
        print(f"[GPU] Skipping stress: {e}")
        time.sleep(duration)


# -----------------------------
# BURST WRAPPER
# -----------------------------
def burst_wrapper(func, duration):
    end = time.time() + duration
    while time.time() < end:
        if random.random() < BURST_CHANCE:
            func(2)
        else:
            time.sleep(0.5)


# -----------------------------
# PHASE RUNNER
# -----------------------------
def run_phase(name, funcs):
    print(f"\n>>> STARTING PHASE: {name} ({PHASE_DURATION}s)")

    threads = []
    for func in funcs:
        t = threading.Thread(target=func, args=(PHASE_DURATION,))
        t.daemon = True
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


# -----------------------------
# MAIN SCENARIO
# -----------------------------
def run_scenario():
    print("=" * 60)
    print("  SYSTEM WORKLOAD GENERATOR — AI PREDICTIVE COOLING")
    print("=" * 60)
    print("  [IMPORTANT] Ensure src/telemetry_logger.py is running!")
    print(f"  Targeting features: cpu, gpu, memory, disk_io, network_io")
    print("=" * 60)

    print("\n=== IDLE PHASE (Baselines) ===")
    time.sleep(PHASE_DURATION)

    run_phase("CPU STRESS", [cpu_stress])
    time.sleep(20)

    run_phase("GPU STRESS", [gpu_stress])
    time.sleep(20)

    run_phase("MEMORY STRESS", [memory_stress])
    time.sleep(20)

    run_phase("DISK I/O STRESS", [disk_stress])
    run_phase("NETWORK I/O STRESS", [network_stress])

    run_phase("MIXED LOAD (CPU/GPU/MEM)", [
        cpu_stress,
        gpu_stress,
        memory_stress
    ])

    run_phase("BURST CHAOS", [
        lambda d: burst_wrapper(cpu_stress, d),
        lambda d: burst_wrapper(gpu_stress, d),
        lambda d: burst_wrapper(disk_stress, d),
        lambda d: burst_wrapper(network_stress, d)
    ])

    print("\n" + "=" * 60)
    print("✅ Workload Scenario Complete")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_scenario()
    except KeyboardInterrupt:
        print("\n\n[INFO] Workload generator stopped by user.")
        # Cleanup temp file if it exists
        if os.path.exists(TEMP_STRESS_FILE):
            os.remove(TEMP_STRESS_FILE)
        sys.exit(0)