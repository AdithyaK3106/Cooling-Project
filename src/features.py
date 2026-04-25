import numpy as np
import pandas as pd
import pickle
import os
from collections import deque

class FeatureProcessor:
    """
    Unified Feature Engineering Engine (Source of Truth).
    Handles log-scaling, temporal features, and strict vector validation.
    """
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.buffer = deque(maxlen=history_size) # Buffer for rolling stats
        
        # Scaling stats persisted from training
        self.stats = {
            'max_disk_log': 1.0,
            'max_net_log': 1.0
        }
        
        # Mandatory feature order (Length == 15)
        self.feature_names = [
            'cpu_norm', 'gpu_norm', 'mem_norm',
            'disk_io_norm', 'net_io_norm',
            'heat_norm',
            'cpu_roll_5', 'gpu_roll_5', 'heat_roll_5',
            'cpu_roll_10', 'gpu_roll_10', 'heat_roll_10',
            'cpu_delta', 'gpu_delta', 'heat_delta'
        ]

    def fit(self, df):
        """Calculates log-scaling bounds from training data."""
        # disk_log = log(1 + disk_io)
        self.stats['max_disk_log'] = float(np.log1p(df['disk_io']).max())
        self.stats['max_net_log'] = float(np.log1p(df['network_io']).max())
        
        # Prevent division by zero
        if self.stats['max_disk_log'] <= 0: self.stats['max_disk_log'] = 1.0
        if self.stats['max_net_log'] <= 0: self.stats['max_net_log'] = 1.0
        
        print(f"Processor fitted: MaxDiskLog={self.stats['max_disk_log']:.4f}, MaxNetLog={self.stats['max_net_log']:.4f}")

    def save(self, path):
        """Persists fitted scaling bounds."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)
        print(f"Preprocessing state saved to {path}")

    def load(self, path):
        """Loads scaling bounds for inference."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing preprocessor state: {path}")
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        print(f"Preprocessing state loaded from {path}")

    def process_single(self, raw_point):
        """
        Transforms a single telemetry point into a validated 15-feature vector.
        """
        # 1. Base Normalization & Log-Scaling
        cpu_n = raw_point.get('cpu', 0) / 100.0
        gpu_n = raw_point.get('gpu', 0) / 100.0
        mem_n = raw_point.get('memory', 0) / 100.0
        
        # Log scaling: disk_io_norm = log(1 + disk_io) / max_disk_log
        disk_log = np.log1p(float(raw_point.get('disk_io', 0)))
        disk_io_n = disk_log / self.stats['max_disk_log']
        
        net_log = np.log1p(float(raw_point.get('network_io', 0)))
        net_io_n = net_log / self.stats['max_net_log']
        
        # Heat proxy
        heat_n = (0.6 * raw_point.get('cpu', 0) + 0.4 * raw_point.get('gpu', 0)) / 100.0
        
        # 2. Stateful Feature Management
        current_step = {
            'cpu_norm': cpu_n, 'gpu_norm': gpu_n, 'mem_norm': mem_n,
            'disk_io_norm': disk_io_n, 'net_io_norm': net_io_n, 'heat_norm': heat_n
        }
        
        # Add to buffer
        self.buffer.append(current_step)
        hist_df = pd.DataFrame(list(self.buffer))
        
        # 3. Temporal Features (Graceful degradation for warmup)
        # rolling_mean on tail(N) works correctly even if len(buffer) < N
        roll_5 = hist_df.tail(5).mean()
        roll_10 = hist_df.tail(10).mean()
        
        # Deltas
        delta_cpu = 0.0
        delta_gpu = 0.0
        delta_heat = 0.0
        if len(self.buffer) > 1:
            delta_cpu = self.buffer[-1]['cpu_norm'] - self.buffer[-2]['cpu_norm']
            delta_gpu = self.buffer[-1]['gpu_norm'] - self.buffer[-2]['gpu_norm']
            delta_heat = self.buffer[-1]['heat_norm'] - self.buffer[-2]['heat_norm']

        # 4. Assembly & Validation Layer
        # Order matters! Must match self.feature_names
        features = [
            cpu_n, gpu_n, mem_n,           # [0-2]
            disk_io_n, net_io_n,           # [3-4]
            heat_n,                        # [5]
            roll_5['cpu_norm'], roll_5['gpu_norm'], roll_5['heat_norm'], # [6-8]
            roll_10['cpu_norm'], roll_10['gpu_norm'], roll_10['heat_norm'], # [9-11]
            delta_cpu, delta_gpu, delta_heat # [12-14]
        ]
        
        # Convert to numpy and validate
        vector = np.array(features)
        
        # Assert no NaNs
        if np.any(np.isnan(vector)):
             vector = np.nan_to_num(vector, nan=0.0)
        
        # CLIP ALL FEATURES TO [0, 1]
        vector = np.clip(vector, 0.0, 1.0)
        
        # Assert feature count == 15
        assert len(vector) == 15, f"Expected 15 features, got {len(vector)}"
        
        return vector.reshape(1, -1)
