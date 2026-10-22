import sys
import os
import time
import threading
import webbrowser

# Ensure import path is correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.live_runtime_manager import LiveRuntimeManager
from http.server import SimpleHTTPRequestHandler, HTTPServer

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")

class DashboardHandler(SimpleHTTPRequestHandler):
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
        pass  # Disable logging to avoid console spam

def run_demo():
    print("======================================================")
    print(" THERVO - Hyperscale Orchestration Mission Control")
    print(" FINAL PROTOTYPE DEMONSTRATION")
    print("======================================================")
    print("[*] Initializing predictive runtime & orchestration layers...")
    
    # 1. Start backend runtime & API Server
    manager = LiveRuntimeManager()
    manager.mode_live = True  # Enable real telemetry from PC
    manager.start()
    
    # Start HTTP server on port 3000
    print("[*] Starting Dashboard Web Server on port 3000...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def serve_dashboard():
        os.chdir(root_dir)
        server = HTTPServer(('0.0.0.0', 3000), DashboardHandler)
        server.serve_forever()
        
    t = threading.Thread(target=serve_dashboard)
    t.daemon = True
    t.start()
    
    # 2. Open dashboard
    print("[*] Launching Mission Control Dashboard at: http://localhost:3000/")
    webbrowser.open("http://localhost:3000/")
    
    print("[*] Dashboard launched. Running live local telemetry orchestration loop...")
    print("[*] The system is now reading local hardware telemetry and dynamically")
    print("    adjusting Lenovo Legion Toolkit thermal profiles in real-time.")
    print("------------------------------------------------------")
    print("[*] Press Ctrl+C to stop the orchestration.")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Stopping demonstration and exporting artifacts...")
        manager.stop()
        print("[*] Demonstration offline. Goodbye.")
        
if __name__ == "__main__":
    run_demo()
