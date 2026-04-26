# AI-Driven Sensor-Free Predictive Cooling System

This project implements a real-time telemetry logging and ML preprocessing pipeline for predictive cooling analysis.

## Directory Structure

```text
Cooling-Project/
├── data/
│   ├── raw/                # Original telemetry CSV logs
│   └── processed/          # Cleaned dataset, X (features), and y (labels)
├── docs/                   # Documentation, PRDs, and READMEs
├── logs/                   # System and logger runtime logs
├── scripts/                # Helper scripts (data generation, etc.)
└── src/                    # Core source code (logger, preprocessor)
```

## Getting Started

### 1. Data Collection
To start collecting real-time telemetry from your system:
```bash
cd src
python telemetry_logger.py
```

### 2. Preprocessing
To clean data and engineer features for ML training:
```bash
cd src
python preprocess.py
```

### 3. Testing (Synthetic Data)
To generate a sample dataset for testing the pipeline:
```bash
cd scripts
python generate_test_data.py
```

### 4. Testing (Live Load)
To generate real system load for the telemetry logger to capture:
```bash
cd scripts
python workload_generator.py
```

## Requirements
- Python 3.8+
- `psutil`
- `pandas`
- `scikit-learn`
- `numpy`
