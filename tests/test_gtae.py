"""
Unit and Integration Tests for Graph Encoder, Graph Transformer, Autoencoder,
Hybrid GTAE-IDS Architecture, Anomaly Detection, and Training Pipeline.

Tests:
1. GraphEncoder forward pass
2. GraphTransformer forward pass
3. GraphAutoencoder forward pass
4. Hybrid GTAE forward pass
5. Output dimensions
6. Reconstruction output dimensions
7. Classification output dimensions
8. Node anomaly scores
9. GPU execution when CUDA is available
10. Small graph training step
11. Model save/load
12. No labels leaking into x
13. Held-out class evaluation logic
"""

import os
from pathlib import Path
import numpy as np
import pytest
import torch

from src.graph_builder import NetworkGraphBuilder
from src.models.autoencoder import GraphAutoencoder
from src.models.graph_encoder import GraphEncoder
from src.models.graph_transformer import GraphTransformer
from src.models.gtae_ids import GTAE_IDS, ClassificationHead


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_graph_data():
    """Generates a synthetic flow-similarity graph with 40 nodes, 10 features, 3 classes."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 10).astype(np.float32)
    y = rng.randint(0, 3, size=40).astype(np.int64)

    builder = NetworkGraphBuilder(k_neighbors=4, random_state=42)
    graph = builder.build_graph(X, labels=y)
    return graph


# ---------------------------------------------------------------------------
# 1. GraphEncoder forward pass
# ---------------------------------------------------------------------------

def test_graph_encoder_forward(mock_graph_data):
    encoder = GraphEncoder(
        in_channels=10,
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        layer_type="sage",
    )
    embeddings = encoder(mock_graph_data.x, mock_graph_data.edge_index)
    assert embeddings.shape == (40, 16)
    assert not torch.isnan(embeddings).any()

    # Test with edge weights and GCN
    gcn_encoder = GraphEncoder(
        in_channels=10,
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        layer_type="gcn",
    )
    embeddings_gcn = gcn_encoder(
        mock_graph_data.x,
        mock_graph_data.edge_index,
        edge_weight=mock_graph_data.edge_weight,
    )
    assert embeddings_gcn.shape == (40, 16)


# ---------------------------------------------------------------------------
# 2. GraphTransformer forward pass
# ---------------------------------------------------------------------------

def test_graph_transformer_forward(mock_graph_data):
    transformer = GraphTransformer(
        in_channels=10,
        hidden_dim=32,
        out_channels=16,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )
    embeddings = transformer(
        mock_graph_data.x,
        mock_graph_data.edge_index,
        edge_weight=mock_graph_data.edge_weight,
    )
    assert embeddings.shape == (40, 16)
    assert not torch.isnan(embeddings).any()


# ---------------------------------------------------------------------------
# 3. GraphAutoencoder forward pass
# ---------------------------------------------------------------------------

def test_graph_autoencoder_forward(mock_graph_data):
    autoencoder = GraphAutoencoder(
        in_features=10,
        latent_dim=16,
        hidden_dim=32,
        encoder_type="transformer",
        num_heads=2,
    )
    outputs = autoencoder(
        mock_graph_data.x,
        mock_graph_data.edge_index,
        edge_weight=mock_graph_data.edge_weight,
    )

    assert "embedding" in outputs
    assert "reconstruction" in outputs
    assert "node_reconstruction_error" in outputs
    assert "reconstruction_loss" in outputs

    assert outputs["embedding"].shape == (40, 16)
    assert outputs["reconstruction"].shape == (40, 10)
    assert outputs["node_reconstruction_error"].shape == (40,)
    assert outputs["reconstruction_loss"].item() >= 0.0


# ---------------------------------------------------------------------------
# 4. Hybrid GTAE forward pass
# ---------------------------------------------------------------------------

def test_gtae_hybrid_forward(mock_graph_data):
    model = GTAE_IDS(
        in_features=10,
        num_classes=8,
        hidden_dim=32,
        latent_dim=16,
        num_heads=2,
        num_encoder_layers=2,
    )
    outputs = model(
        mock_graph_data.x,
        mock_graph_data.edge_index,
        edge_weight=mock_graph_data.edge_weight,
    )

    assert "embedding" in outputs
    assert "reconstruction" in outputs
    assert "classification_logits" in outputs
    assert "probabilities" in outputs
    assert "node_reconstruction_error" in outputs


# ---------------------------------------------------------------------------
# 5. Output dimensions
# ---------------------------------------------------------------------------

def test_output_dimensions(mock_graph_data):
    latent_dim = 24
    model = GTAE_IDS(
        in_features=10,
        num_classes=8,
        hidden_dim=32,
        latent_dim=latent_dim,
    )
    outputs = model(mock_graph_data.x, mock_graph_data.edge_index)
    assert outputs["embedding"].shape == (40, latent_dim)


# ---------------------------------------------------------------------------
# 6. Reconstruction output dimensions
# ---------------------------------------------------------------------------

def test_reconstruction_output_dimensions(mock_graph_data):
    model = GTAE_IDS(in_features=10, num_classes=8, hidden_dim=32, latent_dim=16)
    outputs = model(mock_graph_data.x, mock_graph_data.edge_index)
    assert outputs["reconstruction"].shape == (40, 10)
    assert outputs["node_reconstruction_error"].shape == (40,)


# ---------------------------------------------------------------------------
# 7. Classification output dimensions
# ---------------------------------------------------------------------------

def test_classification_output_dimensions(mock_graph_data):
    num_classes = 8
    model = GTAE_IDS(in_features=10, num_classes=num_classes, hidden_dim=32, latent_dim=16)
    outputs = model(mock_graph_data.x, mock_graph_data.edge_index)
    assert outputs["classification_logits"].shape == (40, num_classes)
    assert outputs["probabilities"].shape == (40, num_classes)

    # Probabilities should sum to ~1.0 per node
    prob_sums = outputs["probabilities"].sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(prob_sums, np.ones(40), atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Node anomaly scores
# ---------------------------------------------------------------------------

def test_node_anomaly_scores(mock_graph_data):
    model = GTAE_IDS(in_features=10, num_classes=8, hidden_dim=32, latent_dim=16)
    scores = model.get_anomaly_scores(
        mock_graph_data.x,
        mock_graph_data.edge_index,
        edge_weight=mock_graph_data.edge_weight,
    )
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (40,)
    assert (scores >= 0.0).all()


# ---------------------------------------------------------------------------
# 9. GPU execution when CUDA is available
# ---------------------------------------------------------------------------

def test_gpu_execution_if_cuda(mock_graph_data):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this environment.")

    device = torch.device("cuda")
    model = GTAE_IDS(in_features=10, num_classes=8, hidden_dim=32, latent_dim=16).to(device)

    x_gpu = mock_graph_data.x.to(device)
    edge_index_gpu = mock_graph_data.edge_index.to(device)
    edge_weight_gpu = mock_graph_data.edge_weight.to(device)

    outputs = model(x_gpu, edge_index_gpu, edge_weight=edge_weight_gpu)
    assert outputs["embedding"].is_cuda
    assert outputs["reconstruction"].is_cuda
    assert outputs["classification_logits"].is_cuda
    assert outputs["node_reconstruction_error"].is_cuda


# ---------------------------------------------------------------------------
# 10. Small graph training step
# ---------------------------------------------------------------------------

def test_small_graph_training_step(mock_graph_data):
    model = GTAE_IDS(
        in_features=10,
        num_classes=8,
        hidden_dim=32,
        latent_dim=16,
        lambda_rec=0.5,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    outputs = model(
        mock_graph_data.x,
        mock_graph_data.edge_index,
        edge_weight=mock_graph_data.edge_weight,
    )
    loss_dict = model.compute_loss(
        outputs=outputs,
        x=mock_graph_data.x,
        y=mock_graph_data.y,
        training_mode="supervised_hybrid",
    )

    initial_loss = loss_dict["total_loss"].item()
    optimizer.zero_grad()
    loss_dict["total_loss"].backward()

    # Verify gradients exist and are non-zero
    has_grads = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in model.parameters())
    assert has_grads is True

    optimizer.step()

    # Second step to check loss computation works repeatedly
    outputs2 = model(mock_graph_data.x, mock_graph_data.edge_index)
    loss_dict2 = model.compute_loss(outputs2, mock_graph_data.x, mock_graph_data.y)
    assert not torch.isnan(loss_dict2["total_loss"])


# ---------------------------------------------------------------------------
# 11. Model save/load
# ---------------------------------------------------------------------------

def test_model_save_and_load(mock_graph_data, tmp_path):
    model = GTAE_IDS(
        in_features=10,
        num_classes=8,
        hidden_dim=32,
        latent_dim=16,
        dropout=0.0,
    )
    model.eval()

    save_path = tmp_path / "gtae_model.pt"
    model.save(save_path)
    assert save_path.exists()

    loaded_model = GTAE_IDS.load(save_path)
    loaded_model.eval()

    with torch.no_grad():
        orig_out = model(mock_graph_data.x, mock_graph_data.edge_index)
        load_out = loaded_model(mock_graph_data.x, mock_graph_data.edge_index)

    torch.testing.assert_close(orig_out["embedding"], load_out["embedding"])
    torch.testing.assert_close(orig_out["reconstruction"], load_out["reconstruction"])
    torch.testing.assert_close(orig_out["classification_logits"], load_out["classification_logits"])


# ---------------------------------------------------------------------------
# 12. No labels leaking into x
# ---------------------------------------------------------------------------

def test_no_labels_leaking_into_x(mock_graph_data):
    # Ensure graph.x only contains features, not y
    assert mock_graph_data.x.shape[1] == 10
    assert mock_graph_data.y is not None
    assert mock_graph_data.y.shape[0] == mock_graph_data.x.shape[0]


# ---------------------------------------------------------------------------
# 13. Held-out class evaluation logic
# ---------------------------------------------------------------------------

def test_held_out_class_evaluation_logic(mock_graph_data):
    # Suppose class 2 is the held-out attack
    holdout_id = 2
    y = mock_graph_data.y

    cls_mask = (y != holdout_id)
    model = GTAE_IDS(in_features=10, num_classes=8, hidden_dim=32, latent_dim=16)

    outputs = model(mock_graph_data.x, mock_graph_data.edge_index)
    loss_dict = model.compute_loss(
        outputs=outputs,
        x=mock_graph_data.x,
        y=y,
        classification_mask=cls_mask,
        training_mode="supervised_hybrid",
    )

    assert not torch.isnan(loss_dict["total_loss"])

    # Anomaly scoring on held-out vs benign
    errors = outputs["node_reconstruction_error"].detach().numpy()
    benign_errors = errors[y.numpy() == 0]
    holdout_errors = errors[y.numpy() == holdout_id]

    threshold = float(np.percentile(benign_errors, 95.0)) if len(benign_errors) > 0 else 1.0
    is_anomaly = holdout_errors > threshold
    assert isinstance(is_anomaly, np.ndarray)
