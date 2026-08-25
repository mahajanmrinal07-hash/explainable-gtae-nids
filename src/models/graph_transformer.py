"""
Graph Transformer Module for Multi-Head Relational Attention.

Captures global and local flow-similarity context using PyTorch Geometric
TransformerConv layers with multi-head attention, residual connections, and normalization.
Optimized for 6 GB VRAM GPU hardware constraints.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
except ImportError:
    TransformerConv = None


class GraphTransformerLayer(nn.Module):
    """
    Single Graph Transformer Block:
    Multi-Head Attention (TransformerConv) + Residual + LayerNorm + Feed-Forward MLP + Residual + LayerNorm.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
        edge_dim: Optional[int] = 1,
        beta: bool = True,
    ):
        super().__init__()
        if TransformerConv is None:
            raise ImportError("torch_geometric is required for GraphTransformerLayer.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.edge_dim = edge_dim

        # Ensure head_dim evenly divides out_dim
        head_dim = max(out_dim // heads, 1)
        self.actual_out_dim = head_dim * heads

        # Multi-Head Graph Attention
        self.conv = TransformerConv(
            in_channels=in_dim,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            beta=beta,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # Norm & Dropout for Attention output
        self.norm1 = nn.LayerNorm(self.actual_out_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Skip connection projection if dimensions differ
        if in_dim != self.actual_out_dim:
            self.res_proj1 = nn.Linear(in_dim, self.actual_out_dim)
        else:
            self.res_proj1 = nn.Identity()

        # Feed-Forward Network (FFN)
        ffn_dim = self.actual_out_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(self.actual_out_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, self.actual_out_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(self.actual_out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for single Graph Transformer block.
        """
        h_res = self.res_proj1(x)

        # Attention Message Passing
        h_conv = self.conv(x, edge_index, edge_attr=edge_attr)
        h_attn = self.dropout1(h_conv)
        h = self.norm1(h_attn + h_res)

        # Feed-Forward Block
        h_ffn = self.ffn(h)
        h = self.norm2(h + h_ffn)
        return h


class GraphTransformer(nn.Module):
    """
    Multi-layer Graph Transformer capturing neighborhood and relational structure.

    Parameters
    ----------
    in_channels : int
        Dimension of input node features (e.g., 67).
    hidden_dim : int
        Hidden dimension for transformer layers (default: 128).
    out_channels : int
        Dimension of final node embedding (default: 64).
    num_heads : int
        Number of attention heads (default: 4).
    num_layers : int
        Number of stacked Transformer layers (default: 2).
    dropout : float
        Dropout probability (default: 0.2).
    edge_dim : Optional[int]
        Dimension of edge features/weights (default: 1 for cosine similarity weights).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        edge_dim: Optional[int] = 1,
    ):
        super().__init__()
        if TransformerConv is None:
            raise ImportError("torch_geometric is required for GraphTransformer.")

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim

        # Input feature projection to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GraphTransformerLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_channels),
            nn.LayerNorm(out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Graph Transformer.

        Parameters
        ----------
        x : torch.Tensor of shape (N, in_channels)
            Node features.
        edge_index : torch.Tensor of shape (2, E)
            COO edge indices.
        edge_weight : Optional[torch.Tensor] of shape (E,) or (E, 1)
            Edge cosine similarity weights.

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Node embeddings.
        """
        # Prepare edge_attr
        edge_attr = None
        if self.edge_dim is not None:
            if edge_weight is not None:
                if edge_weight.dim() == 1:
                    edge_attr = edge_weight.unsqueeze(-1).to(torch.float32)
                else:
                    edge_attr = edge_weight.to(torch.float32)
            else:
                # Default edge weights of 1.0 if not provided
                num_edges = edge_index.size(1) if edge_index.dim() == 2 else 0
                edge_attr = torch.ones((num_edges, self.edge_dim), dtype=torch.float32, device=x.device)

        # Project input features
        h = self.input_proj(x)

        # Pass through Graph Transformer blocks
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr=edge_attr)

        # Project to latent embedding
        embeddings = self.out_proj(h)
        return embeddings
