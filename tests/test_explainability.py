import pytest
import torch
from src.explainability.explainer import IntrusionExplainer
from src.models.gtae_ids import GTAE_IDS

@pytest.fixture
def mock_model():
    model = GTAE_IDS(in_features=10, num_classes=3, hidden_dim=16, latent_dim=8)
    model.eval()
    return model

@pytest.fixture
def sample_data():
    x = torch.rand((5, 10))
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    features = [f"feat_{i}" for i in range(10)]
    return x, edge_index, features

def test_reconstruction_importance(mock_model, sample_data):
    x, edge_index, features = sample_data
    explainer = IntrusionExplainer(mock_model, features)
    
    results = explainer.compute_reconstruction_importance(x, edge_index, top_k=3)
    
    assert len(results) == 5
    assert len(results[0]["top_features"]) == 3
    assert isinstance(results[0]["top_features"][0][0], str)
    assert isinstance(results[0]["top_features"][0][1], float)
    assert "anomaly_score" in results[0]

def test_ablation_importance(mock_model, sample_data):
    x, edge_index, features = sample_data
    explainer = IntrusionExplainer(mock_model, features)
    
    importances = explainer.compute_ablation_importance(x, edge_index, node_idx=0, target_class=1, top_k=4)
    
    assert len(importances) == 4
    assert isinstance(importances[0][0], str)
    assert isinstance(importances[0][1], float)

def test_explain_neighborhood(mock_model, sample_data):
    x, edge_index, features = sample_data
    explainer = IntrusionExplainer(mock_model, features)
    
    result = explainer.explain_neighborhood(x, edge_index, node_idx=1, target_class=2)
    
    assert result["node_idx"] == 1
    assert result["target_class"] == 2
    assert "baseline_prob" in result
    assert "top_influential_neighbors" in result
    assert isinstance(result["top_influential_neighbors"], list)
