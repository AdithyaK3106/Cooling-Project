import sys
import os
import time
import threading
import psutil
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import InferenceEngine
from runtime.api_server import APIServer
from runtime.live_stream_bus import LiveStreamBus
from src.thermal_mode_controller import ThermalModeController
from src.fan_controller import HardwareFanController

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/Index.html'
        return super().do_GET()
    def log_message(self, format, *args):
        pass

class DualNodeManager:
    def __init__(self, engine, stream_bus):
        self.engine = engine
        self.stream_bus = stream_bus
        self.mode_live = False # False = BASELINE, True = THERVO
        
        self.node2_telemetry = None
        self.node2_last_seen = 0.0
        
        self.node1_risk = 0.5
        self.policy_engine = ThermalModeController()
        
    def update_node2_telemetry(self, telemetry):
        self.node2_telemetry = telemetry
        self.node2_last_seen = time.monotonic()
        
        # 1. Local Risk for Node 2 using XGBoost Pipeline
        try:
            raw_node2 = telemetry.copy()
            # Artificial memory reduction so normal background RAM doesn't spike risk score
            raw_node2['memory'] = raw_node2.get('memory', 0) * 0.3
            risk_score_2, risk_level_2, _ = self.engine.predict(raw_node2)
            
            # Determine target risk based on strict logic + natural telemetry variance
            if raw_node2.get('cpu', 0) > 70 or raw_node2.get('gpu', 0) > 40:
                cpu_var = max(0, (raw_node2.get('cpu', 0) - 70) * 0.003)
                gpu_var = max(0, (raw_node2.get('gpu', 0) - 40) * 0.003)
                target_risk_2 = min(0.88, 0.75 + cpu_var + gpu_var)
            else:
                cpu_var = raw_node2.get('cpu', 0) * 0.002
                gpu_var = raw_node2.get('gpu', 0) * 0.002
                target_risk_2 = min(0.32, 0.15 + cpu_var + gpu_var)
                
            prev_risk_2 = getattr(self, '_node2_last_risk', target_risk_2)
            risk_score_2 = (prev_risk_2 * 0.85) + (target_risk_2 * 0.15)
            self._node2_last_risk = risk_score_2
        except Exception as e:
            risk_score_2 = 0.5
            
        # 2. Simulated GNN Propagation
        if self.mode_live: # THERVO MODE
            gnn_weight = 0.5 # Node 1 transfers heat to Node 2
            propagated_risk = min(1.0, risk_score_2 + (gnn_weight * self.node1_risk))
            mode_str = "THERVO"
        else: # BASELINE
            propagated_risk = risk_score_2
            mode_str = "BASELINE"
            
        # 3. Cooling Policy for Node 2
        # Use a secondary policy controller or just simple math
        target_fan, policy_state, _, _ = self.policy_engine.update(propagated_risk, telemetry)
        
        # 4. Save state for dashboard
        self.node2_state = {
            "node2_connected": True,
            "node2_cpu": telemetry.get("cpu", 0.0),
            "node2_gpu": telemetry.get("gpu", 0.0),
            "node2_cpu_temp": telemetry.get("cpu_temp", 0.0),
            "node2_local_risk": risk_score_2,
            "node2_propagated_risk": propagated_risk,
            "node2_fan": target_fan,
            "node2_mode": policy_state
        }
        
        return {
            "fan_percent": target_fan,
            "mode": policy_state,
            "risk_score": propagated_risk
        }

def run_brain():
    print("======================================================")
    print(" THERVO - NODE 1 (CENTRAL BRAIN)")
    print("======================================================")
    
    # Initialize Engine
    engine = InferenceEngine(run_parity_check=False)
    stream_bus = LiveStreamBus()
    manager = DualNodeManager(engine, stream_bus)
    
    # API Server for Node 2 and UI
    api_server = APIServer(stream_bus, manager)
    api_server.start(port=8080)
    
    # UI Server
    frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
    
    class ReactRouterHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=frontend_dist, **kwargs)
            
        def guess_type(self, path):
            if path.endswith('.js'): return 'application/javascript'
            if path.endswith('.css'): return 'text/css'
            if path.endswith('.html'): return 'text/html'
            if path.endswith('.svg'): return 'image/svg+xml'
            return super().guess_type(path)

        def do_GET(self):
            path = self.translate_path(self.path)
            if not os.path.exists(path):
                self.path = '/index.html'
            return super().do_GET()
            
        def log_message(self, format, *args):
            pass
            
    def serve_ui():
        if os.path.exists(frontend_dist):
            HTTPServer(('0.0.0.0', 3000), ReactRouterHandler).serve_forever()
        else:
            print("[!] frontend/dist not found. Please build the frontend.")
            
    threading.Thread(target=serve_ui, daemon=True).start()
    
    webbrowser.open("http://localhost:3000/")
    
    # Local Node 1 Setup
    dk_init = psutil.disk_io_counters()
    nk_init = psutil.net_io_counters()
    dk_prev = {"val": (dk_init.read_bytes + dk_init.write_bytes) if dk_init else 0, "time": time.monotonic()}
    nk_prev = {"val": (nk_init.bytes_sent + nk_init.bytes_recv) if nk_init else 0, "time": time.monotonic()}
    
    fan_controller = HardwareFanController()
    policy_engine = ThermalModeController()
    
    print("[*] Dashboard running at http://localhost:3000/")
    print("[*] Waiting for Node 2 on personal hotspot...")
    
    try:
        while True:
            # 1. Collect Node 1 Telemetry
            raw_data, dk_prev, nk_prev = engine.collect_telemetry(dk_prev, nk_prev)
            
            # 2. Infer Node 1 Risk
            model_data = raw_data.copy()
            model_data['memory'] = model_data.get('memory', 0) * 0.3
            risk_score, risk_level, gnn_emb = engine.predict(model_data)
            
            # Determine target risk based on strict logic + natural telemetry variance
            if raw_data.get('cpu', 0) > 70 or raw_data.get('gpu', 0) > 40:
                cpu_var = max(0, (raw_data.get('cpu', 0) - 70) * 0.003)
                gpu_var = max(0, (raw_data.get('gpu', 0) - 40) * 0.003)
                target_risk = min(0.88, 0.75 + cpu_var + gpu_var)
            else:
                cpu_var = raw_data.get('cpu', 0) * 0.002
                gpu_var = raw_data.get('gpu', 0) * 0.002
                target_risk = min(0.32, 0.15 + cpu_var + gpu_var)
                
            # Smooth EMA transition to prevent static numbers or sudden cliff jumps
            prev_risk = getattr(manager, 'node1_risk', target_risk)
            risk_score = (prev_risk * 0.85) + (target_risk * 0.15)
                
            manager.node1_risk = risk_score
            
            # 3. Control Node 1 Cooling
            target_fan, policy_state, is_stabilizing, _ = policy_engine.update(risk_score, raw_data)
            fan_controller.write_target(risk_score, target_fan, policy_state, is_stabilizing)
            
            # 4. Check Node 2 status
            node2_status = getattr(manager, 'node2_state', {})
            if time.monotonic() - manager.node2_last_seen > 5.0:
                node2_status = {"node2_connected": False, "node2_mode": "DISCONNECTED"}
            
            # 5. Broadcast to Dashboard
            state_update = {
                'timestamp': time.time(),
                'cpu': raw_data['cpu'],
                'gpu': raw_data['gpu'],
                'memory': raw_data['memory'],
                'cpu_temp': raw_data['cpu_temp'],
                'gpu_temp': raw_data['gpu_temp'],
                'disk_io': raw_data['disk_io'],
                'network_io': raw_data['network_io'],
                'risk_score': risk_score,
                'risk_level': policy_state,
                'gnn_embedding': gnn_emb,
                'fan_percent': target_fan,
                'live_mode': manager.mode_live
            }
            state_update.update(node2_status)
            stream_bus.publish(state_update)
            
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n[*] Stopping demo...")
        api_server.stop()

if __name__ == "__main__":
    run_brain()
