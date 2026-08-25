"""
Configuration settings for the XAI-NIDS project.
"""

import os
from pathlib import Path
from typing import Dict, List

# -----------------------------------------------------------------------------
# Base Directories
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
# Search in data/raw/cicids2017 first if present, else fallback to data/raw
if (DATA_DIR / "raw" / "cicids2017").exists():
    RAW_DATA_DIR = DATA_DIR / "raw" / "cicids2017"
else:
    RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Ensure runtime directories exist
for directory in [
    DATA_DIR,
    DATA_DIR / "raw",
    PROCESSED_DATA_DIR,
    SAMPLES_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Reproducibility & Environment
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# CIC-IDS2017 Dataset Files Specification
# -----------------------------------------------------------------------------
EXPECTED_PARQUET_FILES: List[str] = [
    "Benign-Monday-no-metadata.parquet",
    "Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata.parquet",
]

# -----------------------------------------------------------------------------
# Label Normalization Mappings for Real CIC-IDS2017
# -----------------------------------------------------------------------------
# Maps all raw label string variations (including unicode encodings) to unified families
RAW_LABEL_TO_CATEGORY: Dict[str, str] = {
    # Benign
    "BENIGN": "BENIGN",
    "Benign": "BENIGN",
    "benign": "BENIGN",
    
    # DoS Family
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    
    # DDoS
    "DDoS": "DDoS",
    
    # PortScan
    "PortScan": "PortScan",
    
    # Brute Force Family
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    
    # Botnet
    "Bot": "Botnet",
    "Botnet": "Botnet",
    
    # Web Attack Family (handling unicode replacement and dashes)
    "Web Attack – Brute Force": "Web Attack",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack \ufffd Brute Force": "Web Attack",
    "Web Attack ? Brute Force": "Web Attack",
    
    "Web Attack – XSS": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack \ufffd XSS": "Web Attack",
    "Web Attack ? XSS": "Web Attack",
    
    "Web Attack – Sql Injection": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Web Attack \ufffd Sql Injection": "Web Attack",
    "Web Attack ? Sql Injection": "Web Attack",
    
    # Infiltration
    "Infiltration": "Infiltration",
}

# Binary Classification Mapping
BINARY_LABEL_MAP: Dict[str, int] = {
    "BENIGN": 0,
    "ATTACK": 1,
}

# Multi-Class Classification Mapping (8 Core Categories)
MULTICLASS_LABEL_MAP: Dict[str, int] = {
    "BENIGN": 0,
    "DoS": 1,
    "DDoS": 2,
    "PortScan": 3,
    "Brute Force": 4,
    "Botnet": 5,
    "Web Attack": 6,
    "Infiltration": 7,
}

MULTICLASS_INDEX_TO_NAME: Dict[int, str] = {v: k for k, v in MULTICLASS_LABEL_MAP.items()}

# -----------------------------------------------------------------------------
# Hardware & Memory Guardrails (Targeting 6 GB VRAM Laptop GPU)
# -----------------------------------------------------------------------------
GPU_CONFIG = {
    "device": "cuda" if os.environ.get("USE_CPU", "0") != "1" else "cpu",
    "batch_size": 256,
    "eval_batch_size": 512,
    "max_memory_reserved_gb": 4.5,
    "num_workers": 0,
}

# -----------------------------------------------------------------------------
# Baseline Model Defaults
# -----------------------------------------------------------------------------
BASELINE_CONFIG = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
