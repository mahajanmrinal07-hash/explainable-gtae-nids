"""
Graph Neural Network Encoder Module for Spatial Neighborhood Message Passing.

Encodes flow-similarity graph nodes into dense latent representations using
PyTorch Geometric message passing layers (SAGEConv, GCNConv, GraphConv, or GATConv).
Designed for memory-constrained environments (6 GB VRAM GPU).
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv
except ImportError:
    GCNConv = GraphConv = SAGEConv = GATConv = None


class GraphEncoder(nn.Module):
    """
    Modular Graph Neural Network Encoder.

    Parameters
    ----------
    in_channels : int
        Dimension of input node features (e.g., 67 preprocessed flow features).
    hidden_channels : int
        Dimension of hidden representation (default: 128).
    out_channels : int
        Dimension of output node embeddings (default: 64).
    num_layers : int
        Number of GNN message-passing layers (default: 2).
    layer_type : str
        Type of GNN convolution: 'sage', 'gcn', 'graphconv', or 'gat' (default: 'sage').
    dropout : float
        Dropout probability (default: 0.2).
    activation : str
        Non-linear activation function: 'relu', 'gelu', or 'elu' (default: 'relu').
    residual : bool
        Whether to add residual/skip connections between matching layer dimensions (default: True).
    norm_type : Optional[str]
        Normalization type: 'layer', 'batch', or None (default: 'layer').
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        layer_type: str = "sage",
        dropout: float = 0.2,
        activation: str = "relu",
        residual: bool = True,
        norm_type: Optional[str] = "layer",
    ):
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphEncoder.")

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layer_type = layer_type.lower()
        self.dropout = dropout
        self.residual = residual
        self.norm_type = norm_type

        # Activation function
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build GNN layers and normalization layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projections = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = in_channels if layer_idx == 0 else hidden_channels
            out_dim = hidden_channels

            # GNN convolution layer
            conv = self._create_conv_layer(self.layer_type, in_dim, out_dim)
            self.convs.append(conv)

            # Normalization layer
            if norm_type == "layer":
                self.norms.append(nn.LayerNorm(out_dim))
            elif norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(out_dim))
            else:
                self.norms.append(nn.Identity())

            # Skip projection if input and output dimensions differ
            if residual:
                if in_dim != out_dim:
                    self.res_projections.append(nn.Linear(in_dim, out_dim))
                else:
                    self.res_projections.append(nn.Identity())
            else:
                self.res_projections.append(None)

        # Final projection to output embedding dimension
        if hidden_channels != out_channels:
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.LayerNorm(out_channels) if norm_type == "layer" else nn.Identity(),
            )
        else:
            self.out_proj = nn.Identity()

    def _create_conv_layer(self, layer_type: str, in_dim: int, out_dim: int) -> nn.Module:
        """Instantiates the specified PyG convolution layer."""
        if layer_type == "sage":
            return SAGEConv(in_dim, out_dim)
        elif layer_type == "gcn":
            return GCNConv(in_dim, out_dim)
        elif layer_type == "graphconv":
            return GraphConv(in_dim, out_dim)
        elif layer_type == "gat":
            # 2 heads internally for intermediate GAT, concatenated to out_dim
            heads = 2
            head_dim = max(out_dim // heads, 1)
            return GATConv(in_dim, head_dim, heads=heads, concat=True)
        else:
            raise ValueError(f"Unknown layer_type: '{layer_type}'. Supported: 'sage', 'gcn', 'graphconv', 'gat'.")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Graph Encoder.

        Parameters
        ----------
        x : torch.Tensor of shape (N, in_channels)
            Node feature matrix.
        edge_index : torch.Tensor of shape (2, E)
            Graph connectivity matrix in COO format.
        edge_weight : Optional[torch.Tensor] of shape (E,)
            Edge weights (cosine similarity).

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
            Node embeddings in the latent space.
        """
        h = x

        for idx, conv in enumerate(self.convs):
            h_in = h

            # Pass edge_weight to conv layers that accept it
            if self.layer_type in ["gcn", "graphconv"] and edge_weight is not None:
                h_conv = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h_conv = conv(h, edge_index)

            # Normalization
            h_norm = self.norms[idx](h_conv)

            # Activation
            h_act = self.act(h_norm)

            # Dropout
            h_drop = F.dropout(h_act, p=self.dropout, training=self.training)

            # Residual connection
            if self.residual and self.res_projections[idx] is not None:
                res = self.res_projections[idx](h_in)
                h = h_drop + res
            else:
                h = h_drop

        # Final projection to out_channels
        embeddings = self.out_proj(h)
        return embeddings
