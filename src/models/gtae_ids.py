"""
Hybrid GTAE-IDS (Graph Transformer Autoencoder with Multi-Class Intrusion Classifier).

Combines:
1. Graph Transformer Encoder: relational topology and feature message passing
2. Autoencoder Reconstruction Branch: decodes features and computes anomaly scores
3. Classification Head: 8-class MLP predicting network attack families

Supports joint multi-task learning, benign-focused manifold learning,
and novel/unknown attack anomaly scoring.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.autoencoder import GraphAutoencoder
from src.models.graph_encoder import GraphEncoder
from src.models.graph_transformer import GraphTransformer


class ClassificationHead(nn.Module):
    """
    Multi-Class Intrusion Classification Head.
    Maps latent node embeddings to attack family class logits.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, probabilities).
        """
        logits = self.mlp(z)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class GTAE_IDS(nn.Module):
    """
    Unified Hybrid Graph Transformer Autoencoder Intrusion Detection System.

    Parameters
    ----------
    in_features : int
        Number of input flow features (e.g., 67).
    num_classes : int
        Number of attack classes (default: 8).
    hidden_dim : int
        Hidden dimension in encoder, decoder, and classifier (default: 128).
    latent_dim : int
        Dimension of latent graph node embeddings (default: 64).
    encoder_type : str
        Encoder backbone: 'transformer', 'sage', 'gcn', 'graphconv', or 'gat' (default: 'transformer').
    num_heads : int
        Number of attention heads in Graph Transformer (default: 4).
    num_encoder_layers : int
        Number of message passing layers (default: 2).
    dropout : float
        Dropout probability (default: 0.2).
    loss_type : str
        Reconstruction loss type: 'mse' or 'smooth_l1' (default: 'mse').
    lambda_rec : float
        Weight multiplier for reconstruction loss in joint training (default: 0.5).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        encoder_type: str = "transformer",
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.2,
        loss_type: str = "mse",
        lambda_rec: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type.lower()
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.loss_type = loss_type.lower()
        self.lambda_rec = lambda_rec

        # 1. Graph Autoencoder Core (Encoder + Decoder)
        self.autoencoder = GraphAutoencoder(
            in_features=in_features,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            loss_type=loss_type,
        )

        # 2. Multi-Class Classification Head
        self.classifier = ClassificationHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes latent node embeddings."""
        return self.autoencoder.encode(x, edge_index, edge_weight=edge_weight)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs flow features from latent embeddings."""
        return self.autoencoder.decode(z)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full Forward Pass:
        x, edge_index -> latent z -> (x_hat, classification_logits, probabilities, node_errors).

        Returns
        -------
        dict with keys:
            'embedding': (N, latent_dim) latent representation
            'reconstruction': (N, in_features) reconstructed flow features
            'classification_logits': (N, num_classes) raw class logits
            'probabilities': (N, num_classes) softmax probabilities
            'node_reconstruction_error': (N,) node-level anomaly score
        """
        # Encode
        z = self.encode(x, edge_index, edge_weight=edge_weight)

        # Decode & Reconstruct
        x_hat = self.decode(z)
        node_error = self.autoencoder.reconstruction_error(x, x_hat, reduction="none")

        # Classify
        logits, probs = self.classifier(z)

        return {
            "embedding": z,
            "reconstruction": x_hat,
            "classification_logits": logits,
            "probabilities": probs,
            "node_reconstruction_error": node_error,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        lambda_rec: Optional[float] = None,
        training_mode: str = "supervised_hybrid",
        classification_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes hybrid multi-task loss.

        Parameters
        ----------
        outputs : dict
            Output dictionary from forward pass.
        x : torch.Tensor of shape (N, in_features)
            Ground truth normalized node features.
        y : Optional[torch.Tensor] of shape (N,)
            Ground truth class labels.
        class_weights : Optional[torch.Tensor] of shape (num_classes,)
            Weights for handling class imbalance in CrossEntropyLoss.
        lambda_rec : Optional[float]
            Weight for reconstruction loss (defaults to self.lambda_rec).
        training_mode : str
            'supervised_hybrid': reconstruction on all flows + classification
            'benign_autoencoder': reconstruction solely on BENIGN flows (y == 0) + classification
        classification_mask : Optional[torch.Tensor] of shape (N,)
            Boolean mask indicating which nodes should contribute to classification loss
            (e.g., to exclude held-out novel attack classes from training).

        Returns
        -------
        dict with keys:
            'total_loss', 'classification_loss', 'reconstruction_loss'
        """
        lam = self.lambda_rec if lambda_rec is None else lambda_rec

        # 1. Reconstruction Loss
        x_hat = outputs["reconstruction"]
        if training_mode == "benign_autoencoder" and y is not None:
            benign_mask = (y == 0)
            rec_loss = self.autoencoder.compute_loss(x, x_hat, mask=benign_mask)
        else:
            rec_loss = self.autoencoder.compute_loss(x, x_hat, mask=None)

        # 2. Classification Loss
        if y is not None:
            logits = outputs["classification_logits"]
            if classification_mask is not None:
                if classification_mask.sum() > 0:
                    cls_loss = F.cross_entropy(
                        logits[classification_mask],
                        y[classification_mask],
                        weight=class_weights,
                    )
                else:
                    cls_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            else:
                cls_loss = F.cross_entropy(logits, y, weight=class_weights)
        else:
            cls_loss = torch.tensor(0.0, device=x.device)

        total_loss = cls_loss + (lam * rec_loss)

        return {
            "total_loss": total_loss,
            "classification_loss": cls_loss,
            "reconstruction_loss": rec_loss,
        }

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs evaluation inference.

        Returns
        -------
        Tuple of (predicted_class_ids, class_probabilities, anomaly_scores).
        """
        self.eval()
        outputs = self.forward(x, edge_index, edge_weight=edge_weight)
        probs = outputs["probabilities"].detach().cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        anomaly_scores = outputs["node_reconstruction_error"].detach().cpu().numpy()
        return preds, probs, anomaly_scores

    @torch.no_grad()
    def get_anomaly_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Computes per-node reconstruction errors as anomaly scores.
        """
        self.eval()
        outputs = self.forward(x, edge_index, edge_weight=edge_weight)
        return outputs["node_reconstruction_error"].detach().cpu().numpy()

    def save(self, file_path: Union[str, Path]) -> Path:
        """
        Saves model weights and architecture configuration.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "in_features": self.in_features,
                "num_classes": self.num_classes,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "encoder_type": self.encoder_type,
                "num_heads": self.num_heads,
                "num_encoder_layers": self.num_encoder_layers,
                "dropout": self.dropout,
                "loss_type": self.loss_type,
                "lambda_rec": self.lambda_rec,
            },
        }
        torch.save(checkpoint, str(path))
        return path

    @classmethod
    def load(
        cls,
        file_path: Union[str, Path],
        map_location: Optional[str] = None,
    ) -> "GTAE_IDS":
        """
        Loads a saved GTAE_IDS model checkpoint.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(str(path), map_location=map_location or "cpu", weights_only=False)
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        return model
