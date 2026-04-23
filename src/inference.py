import pandas as pd
import numpy as np
import pickle
import time
import os
import psutil
import subprocess
from datetime import datetime
from features import FeatureProcessor

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join("..", "models", "cooling_model.pkl")
PREPROCESSOR_STATE = os.path.join("..", "models", "preprocessor_state.pkl")
OUTPUT_LOG = os.path.join("..", "data", "inference_logs.csv")
INTERVAL = 1.0

# ─────────────────────────────────────────────
#  INFERENCE ENGINE
# ─────────────────────────────────────────────

class InferenceEngine:
    def __init__(self, model_path, state_path):
        self.model_path = model_path
        self.state_path = state_path
        
        # 1. Load Model
        self.model = self._load_model()
        
        # 2. Setup Unified Feature Processor
        self.processor = FeatureProcessor()
        self.processor.load(self.state_path)
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)

    def collect_telemetry(self, disk_prev, net_prev):
        """Collects raw metrics using the same logic as the logger."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        
        # GPU Check
        gpu = 0.0
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 
                                 capture_output=True, text=True, timeout=1)
            if res.returncode == 0:
                gpu = float(res.stdout.strip())
        except: pass

        # I/O Deltas
        now = time.monotonic()
        dk = psutil.disk_io_counters()
        disk_raw = (dk.read_bytes + dk.write_bytes) if dk else 0
        disk_rate = (disk_raw - disk_prev['val']) / (now - disk_prev['time']) if disk_prev else 0.0
        
        nk = psutil.net_io_counters()
        net_raw = (nk.bytes_sent + nk.bytes_recv) if nk else 0
        net_rate = (net_raw - net_prev['val']) / (now - net_prev['time']) if net_prev else 0.0

        raw_data = {
            "timestamp": ts,
            "cpu": cpu,
            "gpu": gpu,
            "memory": mem,
            "disk_io": max(0, disk_rate),
            "network_io": max(0, net_rate)
        }
        return raw_data, {"val": disk_raw, "time": now}, {"val": net_raw, "time": now}

    def get_risk_level(self, score):
        if score < 0.35: return "LOW"
        if score < 0.55: return "MED"
        if score < 0.75: return "HIGH"
        return "CRITICAL"

    def log_result(self, raw_data, risk_score, risk_level):
        os.makedirs(os.path.dirname(OUTPUT_LOG), exist_ok=True)
        file_exists = os.path.isfile(OUTPUT_LOG)
        
        log_row = {**raw_data, "risk_score": round(float(risk_score), 4), "risk_level": risk_level}
        pd.DataFrame([log_row]).to_csv(OUTPUT_LOG, mode='a', index=False, header=not file_exists)

def main():
    print("--- Starting Unified Inference Engine ---")
    try:
        engine = InferenceEngine(MODEL_PATH, PREPROCESSOR_STATE)
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    # I/O Baselines
    dk_prev = {"val": psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes, "time": time.monotonic()}
    nk_prev = {"val": psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv, "time": time.monotonic()}
    
    print(f"Running inference at {INTERVAL}s intervals...")
    
    try:
        while True:
            start_tick = time.monotonic()
            
            # 1. Collect
            raw_data, dk_prev, nk_prev = engine.collect_telemetry(dk_prev, nk_prev)
            
            # 2. Preprocess (SHARED LOGIC)
            features = engine.processor.process_single(raw_data)
            
            # 3. Predict
            risk_score = engine.model.predict(features)[0]
            risk_level = engine.get_risk_level(risk_score)
            
            # 4. Output
            print(f"[{raw_data['timestamp']}] Score: {risk_score:.3f} | Level: {risk_level:<8}")
            engine.log_result(raw_data, risk_score, risk_level)
            
            time.sleep(max(0.01, INTERVAL - (time.monotonic() - start_tick)))
            
    except KeyboardInterrupt:
        print("\nStopping Engine.")

if __name__ == "__main__":
    main()
