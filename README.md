# XAI-NIDS: Explainable Graph-Based Network Intrusion Detection System

XAI-NIDS is a professional, end-to-end Network Intrusion Detection System built for evaluating the CIC-IDS2017 dataset. It integrates Graph Transformer Autoencoders (GTAE-IDS) with Explainable AI (XAI) to provide both highly accurate threat detection and transparent, feature-level insights into its decisions.

## Architecture

The project pipeline consists of:
1. **Preprocessing Layer:** Scales numerical network-flow features and builds a k-NN similarity graph from raw traffic.
2. **GTAE-IDS (Graph Transformer Autoencoder):** 
   - Uses a supervised classification head to identify known attack families.
   - Uses an unsupervised reconstruction decoder to detect novel/zero-day threats as anomalies.
3. **Risk Engine:** Computes an aggregated risk score based on anomaly severity and classification confidence.
4. **XAI Engine:** Applies reconstruction feature importance, feature ablation, and neighborhood analysis to explain predictions.
5. **Interactive Dashboard:** A professional Streamlit application that visualizes threats, flow explanations, and evaluation metrics.

## Methodology

**GTAE-IDS** uses a dual-objective training approach (Supervised Hybrid):
- **Classification Loss (Cross-Entropy):** Optimizes the latent representation to separate known attack classes (e.g., DoS, DDoS, Botnet, PortScan, Brute Force, Web Attack).
- **Reconstruction Loss (Smooth L1):** Forces the model to reconstruct normal traffic behavior, isolating out-of-distribution novel attacks that fail to reconstruct properly.

## Dataset Information

This project relies on the **CIC-IDS2017** dataset, containing realistic background traffic and updated attack profiles. 
- **Predictable Classes:** 8 (BENIGN + 7 attack families)
- **Input Features:** 67 numerical network flow features

> [!IMPORTANT]  
> **Raw Dataset Not Included:** Due to size constraints, the raw CIC-IDS2017 `.parquet` or `.csv` files are **not included** in this repository. You must obtain them separately and place them in the `data/raw/cicids2017/` directory to run inference on new flows.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/xai-nids.git
   cd xai-nids
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Launching the Dashboard

The dashboard provides the main user interface for the NIDS:
```bash
python -m streamlit run dashboard/app.py
```

### Running Tests

The project includes an extensive test suite validating the entire pipeline (data, models, detection, risk, explainability).
```bash
python -m pytest tests -q
```

## Model Artifacts

Pre-trained model artifacts required to run the dashboard are included in the repository:
- `models/gtae_ids.pt`: Normal production model weights
- `models/gtae_config.json`: Production model configuration
- `models/preprocessor.joblib`: Scaler and graph builder state

### Zero-Day Experiment

The repository also includes artifacts for a **Zero-Day/Held-Out Experiment** under `results/zero_day_experiment/`. 
- **Methodology:** The model was trained entirely without samples from the **Infiltration** attack class. 
- **Goal:** To demonstrate the system's ability to detect novel, unseen attacks purely through reconstruction-based anomaly detection.
- **Separation:** The zero-day checkpoint is independent and does *not* replace or overwrite the normal production checkpoint (`models/gtae_ids.pt`).

## Current Verified Results
- **Pytest:** 55/55 tests passing cleanly.
- **Dashboard UI:** 7 comprehensive tabs running optimally via Streamlit.

## Directory Structure
```
xai-nids/
├── dashboard/               # Streamlit application UI and rendering logic
├── data/                    # Dataset storage (raw data not tracked)
├── models/                  # Production model checkpoints and preprocessors
├── notebooks/               # Exploratory and prototyping Jupyter notebooks
├── results/
│   └── zero_day_experiment/ # Independent held-out model artifacts
├── src/                     # Core library
│   ├── data/                # Data loading and preprocessing pipelines
│   ├── detection/           # Inference API and risk engine
│   ├── explainability/      # Feature importance and ablation engine
│   └── models/              # PyTorch graph transformer definitions
├── tests/                   # Unit and integration test suite
├── .gitignore
├── README.md
└── requirements.txt
```
