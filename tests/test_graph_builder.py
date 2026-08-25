"""
Unit tests for the flow-similarity graph construction module.

Covers:
1.  Correct number of nodes
2.  Correct feature dimensions
3.  edge_index shape is [2, num_edges]
4.  No self-loops unless explicitly requested
5.  k-neighbor construction works
6.  Edge weights are valid
7.  Labels are NOT included in x
8.  Deterministic output with fixed seed
9.  Works with numpy and torch inputs
10. Small graph creation completes successfully
"""

import numpy as np
import pytest
import torch

from src.graph_builder import NetworkGraphBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_features():
    """50 flows × 10 features, reproducible."""
    rng = np.random.RandomState(42)
    return rng.randn(50, 10).astype(np.float32)


@pytest.fixture
def sample_labels():
    """50 integer labels across 3 classes."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 3, size=50).astype(np.int64)


@pytest.fixture
def builder():
    return NetworkGraphBuilder(k_neighbors=5, random_state=42)


# ---------------------------------------------------------------------------
# 1. Correct number of nodes
# ---------------------------------------------------------------------------

def test_correct_number_of_nodes(builder, sample_features):
    graph = builder.build_graph(sample_features)
    assert graph.num_nodes == 50
    assert graph.x.shape[0] == 50


# ---------------------------------------------------------------------------
# 2. Correct feature dimensions
# ---------------------------------------------------------------------------

def test_correct_feature_dimensions(builder, sample_features):
    graph = builder.build_graph(sample_features)
    assert graph.x.shape == (50, 10)


# ---------------------------------------------------------------------------
# 3. edge_index shape is [2, num_edges]
# ---------------------------------------------------------------------------

def test_edge_index_shape(builder, sample_features):
    graph = builder.build_graph(sample_features)
    assert graph.edge_index.dim() == 2
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0


# ---------------------------------------------------------------------------
# 4. No self-loops unless explicitly requested
# ---------------------------------------------------------------------------

def test_no_self_loops_by_default(builder, sample_features):
    graph = builder.build_graph(sample_features)
    src, dst = graph.edge_index[0], graph.edge_index[1]
    assert (src != dst).all(), "Found self-loops when include_self_loops=False"


def test_self_loops_when_requested(sample_features):
    builder_sl = NetworkGraphBuilder(
        k_neighbors=5, include_self_loops=True, random_state=42
    )
    graph = builder_sl.build_graph(sample_features)
    src, dst = graph.edge_index[0], graph.edge_index[1]
    # At least some self-loops should exist
    assert (src == dst).any(), "Expected self-loops when include_self_loops=True"


# ---------------------------------------------------------------------------
# 5. k-neighbor construction works
# ---------------------------------------------------------------------------

def test_k_neighbor_construction(sample_features):
    for k in [1, 3, 5, 10]:
        builder_k = NetworkGraphBuilder(k_neighbors=k, random_state=42)
        graph = builder_k.build_graph(sample_features)
        n = sample_features.shape[0]
        effective_k = min(k, n - 1)
        # Each node has exactly effective_k outgoing edges
        expected_edges = n * effective_k
        assert graph.edge_index.shape[1] == expected_edges, (
            f"k={k}: expected {expected_edges} edges, got {graph.edge_index.shape[1]}"
        )


# ---------------------------------------------------------------------------
# 6. Edge weights are valid
# ---------------------------------------------------------------------------

def test_edge_weights_valid(builder, sample_features):
    graph = builder.build_graph(sample_features)
    assert hasattr(graph, "edge_weight")
    assert graph.edge_weight is not None
    assert graph.edge_weight.shape[0] == graph.edge_index.shape[1]
    # Cosine similarity weights should be in [0, 1] for normalized data
    # For unnormalized data they should still be roughly in [-1, 1] range
    # but we clamp to [0, 1]
    assert (graph.edge_weight >= 0.0).all()
    assert (graph.edge_weight <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# 7. Labels are NOT included in x
# ---------------------------------------------------------------------------

def test_labels_not_in_features(builder, sample_features, sample_labels):
    graph = builder.build_graph(sample_features, labels=sample_labels)
    # x should have same feature count as input (no label column appended)
    assert graph.x.shape[1] == sample_features.shape[1]
    # y should be stored separately
    assert graph.y is not None
    assert graph.y.shape[0] == sample_features.shape[0]
    # y values should match input labels
    np.testing.assert_array_equal(graph.y.numpy(), sample_labels)


def test_graph_without_labels(builder, sample_features):
    graph = builder.build_graph(sample_features)
    assert not hasattr(graph, "y") or graph.y is None


# ---------------------------------------------------------------------------
# 8. Deterministic output with fixed seed
# ---------------------------------------------------------------------------

def test_deterministic_output(sample_features, sample_labels):
    builder_a = NetworkGraphBuilder(k_neighbors=5, random_state=42)
    builder_b = NetworkGraphBuilder(k_neighbors=5, random_state=42)

    graph_a = builder_a.build_graph(sample_features, labels=sample_labels)
    graph_b = builder_b.build_graph(sample_features, labels=sample_labels)

    torch.testing.assert_close(graph_a.x, graph_b.x)
    torch.testing.assert_close(graph_a.edge_index, graph_b.edge_index)
    torch.testing.assert_close(graph_a.edge_weight, graph_b.edge_weight)
    torch.testing.assert_close(graph_a.y, graph_b.y)


# ---------------------------------------------------------------------------
# 9. Works with numpy and torch inputs
# ---------------------------------------------------------------------------

def test_numpy_input(builder, sample_features, sample_labels):
    graph = builder.build_graph(sample_features, labels=sample_labels)
    assert graph.x.shape == (50, 10)
    assert graph.y.shape == (50,)


def test_torch_input(builder, sample_features, sample_labels):
    X_t = torch.tensor(sample_features)
    y_t = torch.tensor(sample_labels)
    graph = builder.build_graph(X_t, labels=y_t)
    assert graph.x.shape == (50, 10)
    assert graph.y.shape == (50,)
    torch.testing.assert_close(graph.x, torch.tensor(sample_features))


def test_pandas_input(builder, sample_features, sample_labels):
    import pandas as pd
    # Use default integer columns to avoid pyarrow string_arrow crash on Windows
    df = pd.DataFrame(sample_features)
    graph = builder.build_graph(df, labels=sample_labels)
    assert graph.x.shape == (50, 10)


# ---------------------------------------------------------------------------
# 10. Small graph creation completes successfully
# ---------------------------------------------------------------------------

def test_small_graph_creation():
    rng = np.random.RandomState(99)
    X = rng.randn(5, 3).astype(np.float32)
    builder_small = NetworkGraphBuilder(k_neighbors=2, random_state=99)
    graph = builder_small.build_graph(X)
    assert graph.num_nodes == 5
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == 10  # 5 nodes × 2 neighbors


def test_single_node_graph():
    """A single node should produce a graph with zero edges."""
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    builder_one = NetworkGraphBuilder(k_neighbors=5, random_state=42)
    graph = builder_one.build_graph(X)
    assert graph.num_nodes == 1
    assert graph.edge_index.shape[1] == 0


# ---------------------------------------------------------------------------
# Batch graph construction
# ---------------------------------------------------------------------------

def test_batch_graphs(sample_features, sample_labels):
    builder_batch = NetworkGraphBuilder(
        k_neighbors=3, graph_size=20, random_state=42
    )
    graphs = builder_batch.build_batch_graphs(
        sample_features, labels=sample_labels, batch_size=20
    )
    # 50 samples / 20 per batch = 3 batches (20, 20, 10)
    assert len(graphs) == 3
    total_nodes = sum(g.num_nodes for g in graphs)
    assert total_nodes == 50
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        assert g.x.shape[1] == 10


# ---------------------------------------------------------------------------
# Graph summary
# ---------------------------------------------------------------------------

def test_graph_summary(builder, sample_features, sample_labels):
    graph = builder.build_graph(sample_features, labels=sample_labels)
    summary = NetworkGraphBuilder.graph_summary(graph)
    assert summary["num_nodes"] == 50
    assert summary["num_features"] == 10
    assert summary["num_edges"] > 0
    assert "avg_degree" in summary
    assert "class_distribution" in summary
    assert "edge_weight_stats" in summary


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(builder, sample_features, sample_labels, tmp_path):
    graph = builder.build_graph(sample_features, labels=sample_labels)
    save_path = tmp_path / "test_graph.pt"
    builder.save_graph(graph, save_path)
    loaded = NetworkGraphBuilder.load_graph(save_path)

    torch.testing.assert_close(graph.x, loaded.x)
    torch.testing.assert_close(graph.edge_index, loaded.edge_index)
    torch.testing.assert_close(graph.edge_weight, loaded.edge_weight)
    torch.testing.assert_close(graph.y, loaded.y)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_k_neighbors():
    with pytest.raises(ValueError):
        NetworkGraphBuilder(k_neighbors=0)


def test_label_length_mismatch(builder, sample_features):
    bad_labels = np.array([0, 1, 2])  # wrong length
    with pytest.raises(ValueError, match="Length mismatch"):
        builder.build_graph(sample_features, labels=bad_labels)


# ---------------------------------------------------------------------------
# Fit method
# ---------------------------------------------------------------------------

def test_fit_method(builder, sample_features):
    builder.fit(sample_features)
    assert builder._is_fitted is True
