import multiprocessing
import time
import sys

def cpu_burner():
    """Tight loop to burn CPU cycles and generate heat."""
    while True:
        _ = [x**2 for x in range(10000)]

def run_stress_test(duration=30):
    print("======================================================")
    print(" THERVO - NODE 1 STRESS TEST")
    print("======================================================")
    print(f"[*] Starting CPU Burner across all cores for {duration} seconds...")
    
    processes = []
    # Use max cores - 1 to leave room for the OS and our agent
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    for i in range(num_cores):
        p = multiprocessing.Process(target=cpu_burner)
        p.start()
        processes.append(p)
        
    print(f"[*] Spawning {num_cores} processes. Watch the dashboard for heat propagation!")
    
    try:
        # Wait for the specified duration with a progress bar
        for i in range(duration):
            sys.stdout.write(f"\rStress Test Active: {duration - i}s remaining...")
            sys.stdout.flush()
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] Stress test aborted by user.")
    
    print("\n[*] Stopping all stress processes...")
    for p in processes:
        p.terminate()
        p.join()
        
    print("[*] Stress test complete. Node 1 should begin cooling down.")

if __name__ == "__main__":
    run_stress_test(60) # Run for 60 seconds
