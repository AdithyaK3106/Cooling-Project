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
├── models/                 # Saved models and preprocessor states
├── scripts/                # Helper scripts (data generation, workload, etc.)
├── src/                    # Core source code (production runtime)
│   ├── core/               # Core logic
│   ├── features.py         # Feature definitions and processing
│   ├── inference.py        # Inference pipeline
│   ├── preprocess.py       # Preprocessing logic
│   └── telemetry_logger.py # Real-time telemetry logger
└── training/               # Model training pipelines (non-production)
    ├── data_processing.py  # Data processing for training
    ├── gnn_model.py        # GNN model definition
    ├── inference_pipeline.py # Training validation pipeline
    └── xgboost_model.py    # XGBoost model training
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

### 3. Model Training
To train the XGBoost model on processed data:
```bash
cd training
python xgboost_model.py
```

### 4. Testing (Synthetic Data)
To generate a sample dataset for testing the pipeline:
```bash
cd scripts
python generate_test_data.py
```

### 5. Testing (Live Load)
To generate real system load for the telemetry logger to capture:
```bash
cd scripts
python workload_generator.py
```

## Requirements
See [requirements.txt](requirements.txt) for the full list of dependencies.
- Python 3.8+
- Core: `psutil`, `pandas`, `numpy`, `scikit-learn`, `xgboost`
- Research (Optional): `torch`
