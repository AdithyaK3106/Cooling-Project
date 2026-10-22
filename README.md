# THERVO — AI-Driven Sensor-Free Predictive Cooling System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-r185-black.svg)](https://threejs.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Accelerated-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**THERVO** is a closed-loop, predictive thermal orchestration platform for high-density computing clusters and enterprise client hardware. By forecasting die-level thermal risk up to **4.5 seconds before junction temperature spikes occur**, THERVO eliminates reactive fan lag, curbs acoustic oscillation, and reduces data center cooling overhead.

---

## ⚡ Key Highlights & Architecture

### 1. 🚀 Cinematic Landing Page (`/`)
- Cybernetic particle network canvas fallback with real-time resizing.
- Ambient video background loop with dynamic contrast and brightness grading.
- One-click instant navigation to live mission control.

### 2. 💻 Dual-Node Demonstration (Main PC as Node 1)
- **Node 1 (Central Brain & Main Node)**: Ingests real-time hardware telemetry (`psutil`, NVIDIA SMI, WMI, Lenovo Legion Toolkit) from your active workstation.
- **Node 2 (Neighbor Agent)**: Secondary node connecting via local network / hotspot that receives distributed GNN thermal propagation payloads.
- **Interactive 3D Twin**: Real-time 3D laptop models with dynamic thermal radiation pedestal glows and animated inter-node GNN particle streams.
- **Zero-Hold Actuation**: Proactively shifts power modes and fan RPMs ahead of heat spikes without reactive delay.

### 3. 🏢 Interactive 3D Datacenter Digital Twin (`/thermal-map`)
- High-fidelity Three.js room geometry (`room_server.glb`) with 25-rack server room layout, raised access floors, and perimeter containment walls.
- Multi-layer visual inspection:
  - **Thermal Layer**: Heat flux mapping with gradient threshold warnings.
  - **GNN Layer**: Inter-rack thermal adjacency propagation graphs.
  - **Airflow Layer**: Simulated CFD velocity vector particles.

### 4. 🧠 Explainable AI (XAI) & Operations Copilot
- **SHAP Feature Attribution**: Breakdown of CPU thread load, GPU kernel utilization, memory pressure, and I/O rates.
- **AI Operations Copilot**: Live root-cause diagnostic briefs with contextual recommendations.

---

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.10+ (Recommended: Python 3.12)
- **Node.js**: v20+ / npm v10+ (for building frontend assets)
- **Operating System**: Windows (Full hardware fan actuation support via Lenovo Legion Toolkit / WMI) or Linux/macOS (Simulation & Telemetry fallback mode)

---

### One-Click Launchers

#### Option A: Cross-Platform Shell Scripts (Bash / WSL / macOS / Git Bash)
```bash
# 1. Start the Dual-Node Demonstration (Main Node + Dashboard)
chmod +x start_dual_demo.sh start_node2.sh start_demo.sh
./start_dual_demo.sh

# 2. (Optional) Connect a second machine as Node 2
./start_node2.sh --host <NODE_1_IP>
```

#### Option B: Windows Batch Files (Run as Administrator for Hardware Control)
```cmd
:: 1. Start the Dual-Node Demonstration (Main Node + Dashboard)
start_dual_demo.bat

:: 2. (Optional) Connect a second laptop as Node 2
start_node2.bat --host <NODE_1_IP>
```

---

## 🌐 Network & Port Map

| Endpoint | Port | Description |
| :--- | :--- | :--- |
| **Mission Control Dashboard & Landing Page** | `3000` | High-performance React + Three.js SPA web interface |
| **THERVO Node 1 Brain API** | `8080` | Real-time hardware telemetry stream, inference engine & Node 2 sync |
| **Simulated Datacenter Engine** | `8000` | 25-rack synthetic cluster simulation with anomaly injection |

---

## 🗺️ Application Routes

- **[`/`](http://localhost:3000/)** — Cinematic Landing Page & Welcome Screen
- **[`/dual-node`](http://localhost:3000/dual-node)** — Live 3D Dual-Node Orchestration (PC Main Node + Neighbor Agent)
- **[`/thermal-map`](http://localhost:3000/thermal-map)** — 3D Datacenter Digital Twin & Layer Switcher
- **[`/pipeline`](http://localhost:3000/pipeline)** — Real-Time ML Feature Transformation & Distilled GNN Embedding Stage
- **[`/overview`](http://localhost:3000/overview)** — Cluster-Wide Telemetry Overview & Actuation Summary
- **[`/racks`](http://localhost:3000/racks)** — Rack-Level Thermal Telemetry Inspector
- **[`/predictions`](http://localhost:3000/predictions)** — XGBoost Inference Horizon & Accuracy Benchmarks
- **[`/events`](http://localhost:3000/events)** — Real-Time Anomaly & Thermal Event Audit Stream

---

## 🛠️ Repository Structure

```text
Cooling-Project/
├── frontend/                     # React + Three.js Mission Control SPA
│   ├── dist/                     # Optimized production bundle
│   ├── public/                   # Static assets & 3D models (laptop.glb, room_server.glb, use_this.mp4)
│   ├── src/
│   │   ├── app/App.tsx           # Route definitions & landing gate
│   │   ├── features/             # Copilot, TopNav, bottom controls
│   │   ├── pages/                # DualNodeDemo, LandingPage, ThermalMapTab, etc.
│   │   ├── services/             # Telemetry queries & cluster APIs
│   │   └── three/                # R3F canvas components, layers, and geometry
│   ├── package.json
│   └── vite.config.ts
├── models/                       # Pre-trained models and feature transformers
│   ├── cooling_model.pkl         # Production XGBoost thermal risk model
│   └── preprocessor_state.pkl    # Serialized feature normalization bounds
├── runtime/                      # Low-latency serving and API server
│   ├── api_server.py             # HTTP API (telemetry, toggle-mode, neighbor sync)
│   └── live_stream_bus.py        # Lock-free in-memory pub/sub bus
├── scripts/                      # Demonstration entrypoints & test runners
│   ├── demo_brain_node1.py       # Main PC telemetry collector & orchestration brain
│   ├── demo_agent_node2.py       # Secondary agent syncing with Node 1
│   ├── demo_stress_node1.py      # Multi-core workload generator for live demonstration
│   └── run_demo.py               # Standalone single-node demonstration runner
├── src/                          # Machine learning runtime and hardware controllers
│   ├── fan_controller.py         # Hardware fan control abstraction
│   ├── features.py               # FeatureProcessor single source of truth
│   ├── inference.py              # Real-time inference engine
│   └── thermal_mode_controller.py# Predictive hysteresis policy engine
├── start_dual_demo.sh            # Shell launcher for Dual-Node demo (Unix / Git Bash)
├── start_dual_demo.bat           # Windows launcher for Dual-Node demo
├── start_node2.sh                # Shell launcher for Node 2 agent
├── start_node2.bat               # Windows launcher for Node 2 agent
└── requirements.txt              # Production Python dependencies
```

---

## 🔬 Model Training & Validation

To retrain the thermal prediction model or verify offline-to-online parity:

```powershell
# 1. Feature normalization and preprocessing
python src/preprocess.py

# 2. Train the XGBoost thermal risk model
python training/xgboost_model.py

# 3. Verify training/serving parity
python tests/test_training_serving_parity.py
```

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
