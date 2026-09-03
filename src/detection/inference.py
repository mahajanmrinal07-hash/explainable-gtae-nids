"""
Unified Inference API for XAI-NIDS.
Loads GTAE checkpoint and preprocessor for end-to-end inference.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch

from src.models.gtae_ids import GTAE_IDS
from src.preprocessing import Preprocessor
from src.graph_builder import NetworkGraphBuilder
from src.detection.detector import IntrusionDetector
from src.config import GPU_CONFIG, MULTICLASS_INDEX_TO_NAME

class InferenceAPI:
    """
    End-to-end inference pipeline for new network flows.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        preprocessor_path: Union[str, Path],
        anomaly_threshold: float = 1.188638,
        confidence_threshold: float = 0.6,
        device: str = GPU_CONFIG["device"]
    ):
        """
        Loads necessary assets.
        """
        self.device = torch.device(device)
        
        print(f"Loading preprocessor from {preprocessor_path}...")
        self.preprocessor = Preprocessor.load(preprocessor_path)
        
        print(f"Loading GTAE model from {model_path}...")
        self.model = GTAE_IDS.load(model_path, map_location=device)
        self.model.to(self.device)
        self.model.eval()

        self.graph_builder = NetworkGraphBuilder(k_neighbors=5)
        self.detector = IntrusionDetector(
            anomaly_threshold=anomaly_threshold,
            confidence_threshold=confidence_threshold,
            class_names=MULTICLASS_INDEX_TO_NAME
        )

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Runs end-to-end prediction on a dataframe of raw flows.

        Args:
            df: DataFrame containing raw flow features.

        Returns:
            List of detection results with risk scores.
        """
        # 1. Preprocess
        # The preprocessor handles removing labels if present, clipping, and scaling.
        print("Preprocessing features...")
        x_scaled = self.preprocessor.transform(df)
        
        # 2. Build Graph
        print("Building similarity graph...")
        data = self.graph_builder.build_graph(x_scaled)
        edge_index = data.edge_index
        edge_weights = data.edge_weight

        # 3. Model Inference
        print("Running GTAE inference...")
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
        edge_index_tensor = edge_index.to(self.device)
        edge_weight_tensor = edge_weights.to(self.device)

        preds, probs, anomaly_scores = self.model.predict(
            x_tensor, edge_index_tensor, edge_weight_tensor
        )

        # 4. Detection Logic & Risk Scoring
        print("Scoring threats and risk...")
        results = self.detector.detect_batch(preds, probs, anomaly_scores)

        return results

