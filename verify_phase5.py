"""
Verification script for Phase 5: Explainability, Detection, and Risk Engine.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import json
import matplotlib.pyplot as plt

from src.config import RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.detection.inference import InferenceAPI
from src.explainability.explainer import IntrusionExplainer
from src.visualization.explain_plots import plot_feature_importance, plot_risk_distribution

def main():
    print("=== Phase 5 Verification ===")
    
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    model_path = MODELS_DIR / "gtae_ids.pt"
    
    if not preprocessor_path.exists() or not model_path.exists():
        print(f"Error: Required models not found. Please train GTAE first.")
        print(f"Paths checked: {preprocessor_path}, {model_path}")
        sys.exit(1)

    print(f"Loading Inference API...")
    api = InferenceAPI(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        anomaly_threshold=2.0,
        confidence_threshold=0.60
    )

    # Use a small chunk of a raw dataset for inference
    dataset_file = RAW_DATA_DIR / "Benign-Monday-no-metadata.parquet"
    if not dataset_file.exists():
        print(f"Raw data file {dataset_file} not found. Please check paths.")
        sys.exit(1)
        
    print(f"Loading sample data from {dataset_file}...")
    df = pd.read_parquet(dataset_file).head(100)  # Just take 100 flows for quick test
    
    print("Running detection pipeline...")
    results = api.predict(df)
    
    # Save results
    results_file = RESULTS_DIR / "metrics" / "phase5_inference_sample.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved detection results to {results_file}")
    
    # Generate risk distribution plot
    plot_file = RESULTS_DIR / "figures" / "risk_distribution.png"
    plot_risk_distribution(results, save_path=plot_file)
    print(f"Saved risk distribution plot to {plot_file}")
    
    # Explainability
    print("Running Explainability Engine...")
    explainer = IntrusionExplainer(api.model, api.preprocessor.feature_names_)
    
    # Preprocess and prepare graphs for explainer
    x_scaled = api.preprocessor.transform(df)
    data = api.graph_builder.build_graph(x_scaled)
    edge_index = data.edge_index
    edge_weights = data.edge_weight
    
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(api.device)
    edge_index_tensor = edge_index.to(api.device)
    
    reconstruction_importance = explainer.compute_reconstruction_importance(
        x_tensor, edge_index_tensor, top_k=5
    )
    
    # Plot feature importance for the first node
    first_node_features = reconstruction_importance[0]["top_features"]
    feat_plot_file = RESULTS_DIR / "figures" / "feature_importance_node0.png"
    plot_feature_importance(first_node_features, title="Top Anomalous Features (Node 0)", save_path=feat_plot_file)
    print(f"Saved feature importance plot to {feat_plot_file}")
    
    # Ablation importance
    print("Computing ablation importance for Node 0, Target Class 0 (BENIGN)...")
    ablation_importance = explainer.compute_ablation_importance(
        x_tensor, edge_index_tensor, node_idx=0, target_class=0, top_k=5
    )
    print(f"Ablation Importance: {ablation_importance}")

    print("Phase 5 Verification Complete!")

if __name__ == "__main__":
    main()
