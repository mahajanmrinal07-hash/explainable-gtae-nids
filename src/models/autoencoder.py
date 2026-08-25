"""
Graph Autoencoder (GTAE) Module for Reconstruction-Based Anomaly Detection.

Encodes graph nodes into a compact latent representation using Graph Transformer / GNN,
and reconstructs the original numerical flow features via an MLP decoder.
Calculates per-node reconstruction error vectors and global reconstruction loss.
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph_encoder import GraphEncoder
from src.models.graph_transformer import GraphTransformer


class GraphAutoencoder(nn.Module):
    """
    Graph Autoencoder (GTAE) for Network Flow Anomaly Detection.

    Parameters
    ----------
    in_features : int
        Number of input flow features (e.g., 67).
    latent_dim : int
        Dimension of latent node representation (default: 64).
    hidden_dim : int
        Hidden dimension in encoder and decoder (default: 128).
    encoder_type : str
        Type of encoder: 'transformer', 'sage', 'gcn', 'graphconv', or 'gat' (default: 'transformer').
    num_heads : int
        Number of attention heads if using transformer (default: 4).
    num_encoder_layers : int
        Number of encoder message-passing layers (default: 2).
    decoder_hidden_dims : Optional[List[int]]
        Hidden dimensions for decoder MLP (default: [128]).
    dropout : float
        Dropout rate (default: 0.2).
    loss_type : str
        Reconstruction loss function: 'mse' or 'smooth_l1' (default: 'mse').
    """

    def __init__(
        self,
        in_features: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        encoder_type: str = "transformer",
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        decoder_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type.lower()
        self.loss_type = loss_type.lower()
        self.dropout = dropout

        # 1. Encoder instantiation
        if self.encoder_type == "transformer":
            self.encoder = GraphTransformer(
                in_channels=in_features,
                hidden_dim=hidden_dim,
                out_channels=latent_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                dropout=dropout,
            )
        else:
            self.encoder = GraphEncoder(
                in_channels=in_features,
                hidden_channels=hidden_dim,
                out_channels=latent_dim,
                num_layers=num_encoder_layers,
                layer_type=self.encoder_type,
                dropout=dropout,
            )

        # 2. Decoder instantiation (Multi-layer Perceptron)
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [hidden_dim]

        decoder_layers: List[nn.Module] = []
        current_dim = latent_dim
        for dec_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(current_dim, dec_dim))
            decoder_layers.append(nn.LayerNorm(dec_dim))
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.Dropout(dropout))
            current_dim = dec_dim

        # Final reconstruction layer back to in_features
        decoder_layers.append(nn.Linear(current_dim, in_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes graph nodes into latent representation."""
        return self.encoder(x, edge_index, edge_weight=edge_weight)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent representation back into reconstructed flow features."""
        return self.decoder(z)

    def reconstruction_error(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        """
        Computes reconstruction error between input and reconstructed features.

        Parameters
        ----------
        x : torch.Tensor of shape (N, in_features)
            Ground truth normalized flow features.
        x_hat : torch.Tensor of shape (N, in_features)
            Reconstructed flow features.
        reduction : str
            'none': returns per-node error vector of shape (N,)
            'mean': returns scalar mean error
            'sum': returns scalar sum error

        Returns
        -------
        torch.Tensor
            Node-level error vector or scalar loss.
        """
        if self.loss_type == "smooth_l1":
            diff = F.smooth_l1_loss(x_hat, x, reduction="none", beta=1.0)
        else:  # 'mse'
            diff = (x_hat - x) ** 2

        if reduction == "none":
            # Mean error across feature dimensions for each individual node -> shape (N,)
            return torch.mean(diff, dim=-1)
        elif reduction == "mean":
            return torch.mean(diff)
        elif reduction == "sum":
            return torch.sum(diff)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates scalar reconstruction loss, optionally filtered on a subset mask.
        """
        node_errors = self.reconstruction_error(x, x_hat, reduction="none")
        if mask is not None:
            if mask.sum() > 0:
                return node_errors[mask].mean()
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        return node_errors.mean()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full Autoencoder forward pass: x -> z -> x_hat.

        Returns
        -------
        dict with keys:
            'embedding': (N, latent_dim) latent representation
            'reconstruction': (N, in_features) reconstructed features
            'node_reconstruction_error': (N,) per-node error
            'reconstruction_loss': scalar mean reconstruction loss
        """
        z = self.encode(x, edge_index, edge_weight=edge_weight)
        x_hat = self.decode(z)
        node_err = self.reconstruction_error(x, x_hat, reduction="none")
        rec_loss = node_err.mean()

        return {
            "embedding": z,
            "reconstruction": x_hat,
            "node_reconstruction_error": node_err,
            "reconstruction_loss": rec_loss,
        }
