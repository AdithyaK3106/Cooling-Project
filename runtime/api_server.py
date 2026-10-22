import json
import threading
import os
import base64
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

class TelemetryHandler(BaseHTTPRequestHandler):
    bus = None  # Will be injected
    manager = None  # Will be injected

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/telemetry':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if self.bus:
                state = dict(self.bus.get_latest_state())
                state["live_mode"] = getattr(self.manager, "mode_live", False)
            else:
                state = {"live_mode": False}
                
            self.wfile.write(json.dumps(state).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/toggle-mode':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if self.manager:
                self.manager.mode_live = not self.manager.mode_live
                res_val = self.manager.mode_live
            else:
                res_val = False
                
            self.wfile.write(json.dumps({"live_mode": res_val}).encode('utf-8'))
            
        elif self.path == '/neighbor/sync':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                try:
                    node2_telemetry = json.loads(body.decode('utf-8'))
                    if hasattr(self.manager, 'update_node2_telemetry'):
                        command = self.manager.update_node2_telemetry(node2_telemetry)
                    else:
                        command = {"fan_percent": 50, "mode": "BASELINE"}
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(command).encode('utf-8'))
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
            else:
                self.send_response(400)
                self.end_headers()

        elif self.path == '/api/investigate':
            content_length = int(self.headers.get('Content-Length', 0))
            payload = {}
            if content_length > 0:
                try:
                    payload = json.loads(self.rfile.read(content_length).decode('utf-8'))
                except Exception:
                    payload = {}
            
            rack_id = payload.get('rackId', 'RACK-A001')
            risk_score = payload.get('riskScore', 0.5)
            risk_level = payload.get('riskLevel', 'ELEVATED')
            cpu = float(payload.get('cpu', 50))
            gpu = float(payload.get('gpu', 50))
            temp = float(payload.get('temperature', 35.0))
            cooling_state = payload.get('coolingState', 'standby')
            explanation = payload.get('explanation', '')
            query = payload.get('query', f"server cooling high workload thermal management {rack_id}")

            serpapi_key = os.environ.get("SERPAPI_KEY", "").strip()
            response_data = None
            
            if serpapi_key:
                try:
                    url = f"https://serpapi.com/search.json?engine=google&q={urllib.parse.quote(query)}&api_key={serpapi_key}&num=3"
                    req = urllib.request.Request(url, headers={'User-Agent': 'THERVO-Copilot/1.0'})
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        res_json = json.loads(resp.read().decode('utf-8'))
                        organic = res_json.get('organic_results', [])
                        if organic and len(organic) > 0:
                            top = organic[0]
                            link = top.get('link', 'https://www.google.com')
                            parsed_domain = urllib.parse.urlparse(link).netloc or 'google.com'
                            response_data = {
                                "status": "success",
                                "is_mock": False,
                                "message": "EXTERNAL CONTEXT READY",
                                "finding": top.get('snippet') or top.get('title') or "External technical finding retrieved.",
                                "source_title": top.get('title', 'Technical Reference Documentation'),
                                "source_domain": parsed_domain,
                                "source_link": link
                            }
                except Exception as ex:
                    print(f"[Copilot SerpAPI Error] {ex}")

            if not response_data:
                # Development mock fallback (clearly labelled)
                primary = "GPU workload" if gpu > cpu else "CPU multi-thread computation"
                response_data = {
                    "status": "fallback",
                    "is_mock": True,
                    "message": "SerpAPI is not configured." if not serpapi_key else "External search temporarily unavailable.",
                    "label": "DEVELOPMENT / MOCK DATA",
                    "finding": f"High {primary} load on {rack_id} ({gpu:.0f}% GPU, {cpu:.0f}% CPU) triggers localized die-level heat flux. Standard practice calls for dynamic CFM increase and predictive baseline shifting before junction throttle limits.",
                    "source_title": "ASHRAE TC 9.9 High-Density Thermal Guidelines for Data Centers",
                    "source_domain": "tc0909.ashraetcs.org",
                    "source_link": "https://tc0909.ashraetcs.org/resources.html"
                }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        elif self.path == '/api/voice-brief':
            content_length = int(self.headers.get('Content-Length', 0))
            payload = {}
            if content_length > 0:
                try:
                    payload = json.loads(self.rfile.read(content_length).decode('utf-8'))
                except Exception:
                    payload = {}
            
            text = payload.get('text', 'Operational briefing. Systems nominal.')
            elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
            response_data = None

            if elevenlabs_key:
                try:
                    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
                    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                    req_payload = json.dumps({
                        "text": text,
                        "model_id": "eleven_monolingual_v1",
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75
                        }
                    }).encode('utf-8')
                    req = urllib.request.Request(
                        tts_url, 
                        data=req_payload, 
                        headers={
                            'xi-api-key': elevenlabs_key,
                            'Content-Type': 'application/json',
                            'Accept': 'audio/mpeg'
                        }
                    )
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        audio_bytes = resp.read()
                        b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                        response_data = {
                            "status": "success",
                            "is_mock": False,
                            "message": "ElevenLabs audio synthesized successfully.",
                            "text": text,
                            "audio_base64": b64_audio
                        }
                except Exception as ex:
                    print(f"[Copilot ElevenLabs Error] {ex}")

            if not response_data:
                # Development mock fallback (clearly labelled)
                response_data = {
                    "status": "fallback",
                    "is_mock": True,
                    "message": "ElevenLabs is not configured." if not elevenlabs_key else "Voice synthesis service temporarily unavailable.",
                    "label": "DEVELOPMENT / MOCK DATA",
                    "text": text,
                    "audio_base64": None
                }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def log_message(self, format, *args):
        print(f"[API HTTP] {format % args}")

class APIServer:
    def __init__(self, bus, manager=None):
        self.bus = bus
        self.manager = manager
        self.server = None
        self._thread = None

    def start(self, port=8080):
        TelemetryHandler.bus = self.bus
        TelemetryHandler.manager = self.manager
        self.server = HTTPServer(('0.0.0.0', port), TelemetryHandler)
        self._thread = threading.Thread(target=self.server.serve_forever)
        self._thread.daemon = True
        self._thread.start()
        print(f"API Server listening on port {port}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self._thread:
            self._thread.join()
        print("API Server stopped")
