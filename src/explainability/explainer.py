"""
Explainability Engine for XAI-NIDS.
Provides model-agnostic and reconstruction-based feature importance,
and graph neighborhood explanations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

class IntrusionExplainer:
    """
    Generates local explanations for detected intrusion events.
    Supports feature importance via perturbation/ablation and reconstruction errors.
    """

    def __init__(self, model: nn.Module, feature_names: List[str]):
        """
        Args:
            model: Trained GTAE_IDS model.
            feature_names: List of string names for features.
        """
        self.model = model
        self.feature_names = feature_names
        self.model.eval()

    @torch.no_grad()
    def compute_reconstruction_importance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Ranks features based on their absolute reconstruction error for each node.
        High error indicates the feature is anomalous/unexpected.

        Args:
            x: Input node features (N, F).
            edge_index: Graph edge indices.
            edge_weight: Graph edge weights.
            top_k: Number of top features to return per node.

        Returns:
            List of length N. Each element is a dict with 'node_id' and 'top_features'
            where 'top_features' is a list of tuples (feature_name, error_value).
        """
        outputs = self.model(x, edge_index, edge_weight)
        x_hat = outputs["reconstruction"]

        # Calculate absolute error per feature
        errors = torch.abs(x - x_hat).detach().cpu().numpy()  # (N, F)

        results = []
        for i in range(errors.shape[0]):
            node_errors = errors[i]
            # Get indices of top k errors
            top_indices = np.argsort(node_errors)[-top_k:][::-1]
            
            top_features = [
                (self.feature_names[idx], float(node_errors[idx]))
                for idx in top_indices
            ]
            results.append({
                "node_id": i,
                "top_features": top_features,
                "anomaly_score": float(outputs["node_reconstruction_error"][i].item())
            })

        return results

    @torch.no_grad()
    def compute_ablation_importance(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        target_class: int,
        edge_weight: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Computes feature importance for a specific node by ablating (zeroing out)
        each feature and measuring the drop in predicted probability for target_class.

        Args:
            x: Input node features (N, F).
            edge_index: Graph edge indices.
            node_idx: Index of the node to explain.
            target_class: Class probability to track.
            top_k: Number of top features to return.

        Returns:
            List of tuples (feature_name, importance_score).
        """
        # Baseline probability
        baseline_outputs = self.model(x, edge_index, edge_weight)
        baseline_prob = baseline_outputs["probabilities"][node_idx, target_class].item()

        importances = []
        
        # Ablate each feature
        for feat_idx in range(x.shape[1]):
            x_ablated = x.clone()
            x_ablated[node_idx, feat_idx] = 0.0  # Zero out feature (assuming robust scaled, 0 is median)
            
            ablated_outputs = self.model(x_ablated, edge_index, edge_weight)
            ablated_prob = ablated_outputs["probabilities"][node_idx, target_class].item()
            
            # Importance is drop in probability (positive means it was contributing to the class)
            importance = baseline_prob - ablated_prob
            importances.append((self.feature_names[feat_idx], importance))

        # Sort by absolute importance (impact)
        importances.sort(key=lambda item: abs(item[1]), reverse=True)
        return importances[:top_k]

    @torch.no_grad()
    def explain_neighborhood(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        target_class: int,
        k_hops: int = 1
    ) -> Dict[str, Any]:
        """
        Identifies which neighbors contribute most to the prediction by ablating edges.
        """
        # Baseline probability
        baseline_outputs = self.model(x, edge_index)
        baseline_prob = baseline_outputs["probabilities"][node_idx, target_class].item()

        # Find 1-hop edges connected to node_idx
        src, dst = edge_index
        mask = (src == node_idx) | (dst == node_idx)
        connected_edges = mask.nonzero(as_tuple=True)[0]

        edge_importances = []
        for e_idx in connected_edges:
            # Ablate edge by dropping it
            ablated_edge_index = torch.cat([edge_index[:, :e_idx], edge_index[:, e_idx+1:]], dim=1)
            
            ablated_outputs = self.model(x, ablated_edge_index)
            ablated_prob = ablated_outputs["probabilities"][node_idx, target_class].item()
            
            importance = baseline_prob - ablated_prob
            neighbor_node = dst[e_idx].item() if src[e_idx].item() == node_idx else src[e_idx].item()
            
            edge_importances.append((neighbor_node, importance))

        edge_importances.sort(key=lambda item: abs(item[1]), reverse=True)

        return {
            "node_idx": node_idx,
            "target_class": target_class,
            "baseline_prob": baseline_prob,
            "top_influential_neighbors": edge_importances[:5]
        }
