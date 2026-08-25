# XAI-NIDS: Explainable Graph-Based Intelligent Network Intrusion Detection System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch 2.6+cu124](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-red.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-zone)
[![PyG 2.8](https://img.shields.io/badge/PyTorch_Geometric-2.8-purple.svg)](https://pyg.org/)

---

## 📌 Project Overview

**XAI-NIDS** (*Explainable Graph-Based Intelligent Network Intrusion Detection and Threat Analysis System*) is a research prototype inspired by the **GTAE-IDS** framework (*Graph Transformer-Based Autoencoder Framework for Real-Time Network Intrusion Detection*).

The system addresses critical challenges in modern network security:
1. **Graph Topological Context**: Encodes spatial and relational host communications using Graph Neural Networks and Graph Transformers.
2. **Reconstruction-Based Threat Detection**: Identifies subtle, zero-day, and multi-vector anomalies via autoencoder reconstruction error.
3. **Transparent Explainability (XAI)**: Attributes threat decisions through GNNExplainer, SHAP, and attention mechanisms.
4. **Hardware Optimized**: Tuned specifically for resource-efficient execution on a **6 GB RTX 3060 Laptop GPU** with PyTorch CUDA 12.4.

---

## 📂 Project Architecture

```
XAI-NIDS/
├── data/
│   ├── raw/                # Original CIC-IDS2017 Parquet dataset files
│   ├── processed/          # Preprocessed, scaled, and feature-engineered datasets
│   └── samples/            # Verified synthetic sample datasets for testing and CI
├── src/
│   ├── __init__.py
│   ├── config.py           # Project constants, paths, and GPU guardrails
│   ├── data_loader.py      # Dataset discovery, schema inspection, and chunked loading
│   ├── preprocessing.py    # Leakage-free scalers, imputers, and label normalization
│   ├── graph_builder.py    # [Phase 2] Flow-to-Graph conversion pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py     # Random Forest empirical benchmark model
│   │   ├── graph_encoder.py # [Phase 2] Spatial GNN Encoder
│   │   ├── graph_transformer.py # [Phase 2/3] Global Graph Transformer
│   │   └── autoencoder.py  # [Phase 2/3] Graph Autoencoder (GTAE)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_baseline.py # Baseline training, evaluation, and artifact exporter
│   │   └── train_gtae.py   # [Phase 3] GTAE model training pipeline
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py     # Anomaly detector engine
│   │   ├── risk_engine.py  # Threat risk scoring engine
│   │   └── inference.py    # Unified end-to-end inference API
│   └── explainability/
│       ├── __init__.py
│       └── explainer.py    # Explainability engine (Ablation & Reconstruction XAI)
├── dashboard/
│   └── app.py              # [Phase 6] Streamlit monitoring dashboard
├── tests/                  # Pytest test suite
├── models/                 # Serialized model weights and preprocessors (.joblib/.pt)
├── results/
│   ├── figures/            # High-resolution confusion matrices & plots
│   └── metrics/            # Evaluation metrics JSON & classification reports
├── notebooks/              # Exploratory data analysis notebooks
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Hardware Environment

| Component | Specification |
|---|---|
| **Python** | `3.11.9` |
| **GPU** | NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM) |
| **CUDA Driver** | `596.36` (Supports CUDA 13.2) |
| **PyTorch** | `2.6.0+cu124` |
| **PyTorch Geometric** | `2.8.0.post1` |

---

## 🚀 Getting Started (Phase 1)

### 1. Inspect Hardware & CUDA Status

Run the environment utility to verify CPU, GPU, VRAM, and PyTorch CUDA availability:

```bash
python -m src.utils.gpu_info
```

### 2. Dataset Setup

Place the official CIC-IDS2017 Parquet files into the `data/raw/` directory:

- `Benign-Monday-no-metadata.parquet`
- `Bruteforce-Tuesday-no-metadata.parquet`
- `DoS-Wednesday-no-metadata.parquet`
- `WebAttacks-Thursday-no-metadata.parquet`
- `Infiltration-Thursday-no-metadata.parquet`
- `DDoS-Friday-no-metadata.parquet`
- `Botnet-Friday-no-metadata.parquet`
- `Portscan-Friday-no-metadata.parquet`

*(Note: If raw dataset files are not yet in `data/raw/`, the data loader automatically generates a compliant synthetic CIC-IDS2017 sample dataset in `data/samples/` to enable end-to-end local testing and pipeline validation).*

### 3. Run Phase 1 Baseline Training

Train the Random Forest baseline model, evaluate on unseen test data, and export metrics:

```bash
# Binary classification (BENIGN vs ATTACK)
python -m src.training.train_baseline --mode binary --sample_size 25000

# Multiclass classification (BENIGN, DoS, DDoS, PortScan, etc.)
python -m src.training.train_baseline --mode multiclass --sample_size 25000
```

### 4. Run Automated Test Suite

Execute all unit and integration tests:

```bash
pytest tests/ -v
```

---

## 📊 Phase 1 Deliverables

- **Model Artifact**: `models/baseline_rf.joblib`
- **Fitted Preprocessor**: `models/preprocessor.joblib`
- **Metrics Summary**: `results/metrics/baseline_metrics.json`
- **Text Classification Report**: `results/metrics/baseline_classification_report.txt`
- **Confusion Matrix Heatmap**: `results/figures/baseline_confusion_matrix.png`

---

## 🗺️ Project Roadmap

- [x] **Phase 1: Project Foundation & Baseline Pipeline**
- [x] **Phase 2: Graph Construction & GNN Architecture**
- [x] **Phase 3: Graph Transformer Autoencoder (GTAE-IDS) Training**
- [x] **Phase 4: Real-time Detection & Threat Risk Engine**
- [x] **Phase 5: Explainable AI Engine (Ablation & Reconstruction XAI)**
- [ ] **Phase 6: Interactive Streamlit Dashboard & End-to-End Evaluation**
