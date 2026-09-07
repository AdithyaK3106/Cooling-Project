import multiprocessing
import time
import sys

def cpu_burner():
    """Tight loop to burn CPU cycles and generate heat."""
    while True:
        _ = [x**2 for x in range(10000)]

def gpu_burner():
    """Use PyTorch to load the GPU to ~50%."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n[!] CUDA not available. Skipping GPU stress test.")
            return
            
        # Create a reasonably large tensor
        a = torch.randn(8192, 8192, device='cuda')
        b = torch.randn(8192, 8192, device='cuda')
        
        while True:
            t0 = time.time()
            # Matrix multiplication to stress CUDA cores
            c = torch.matmul(a, b)
            # Synchronize to ensure execution finishes before timing
            torch.cuda.synchronize()
            t1 = time.time()
            
            compute_time = t1 - t0
            # To get 50% utilization, we sleep for the exact same duration as the compute took
            time.sleep(compute_time)
            
    except ImportError:
        print("\n[!] PyTorch not installed. Skipping GPU stress test.")
    except Exception as e:
        print(f"\n[!] GPU stress error: {e}")

def run_stress_test(duration=60):
    print("======================================================")
    print(" THERVO - NODE 1 STRESS TEST (CPU + GPU)")
    print("======================================================")
    
    num_cores = 15
    print(f"[*] Starting {num_cores} CPU Burner processes...")
    
    processes = []
    
    # Start CPU processes
    for i in range(num_cores):
        p = multiprocessing.Process(target=cpu_burner)
        p.start()
        processes.append(p)
        
    # Start GPU process
    print("[*] Starting GPU Burner process (~50% utilization)...")
    gpu_p = multiprocessing.Process(target=gpu_burner)
    gpu_p.start()
    processes.append(gpu_p)
        
    print(f"\n[*] All stress processes running. Watch the dashboard for heat propagation!")
    
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
