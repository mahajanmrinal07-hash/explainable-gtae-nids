"""
Focused unit tests for the live traffic capture module (tools/live_capture.py).
Validates NFStream-to-67-feature mapping, Preprocessor integration, and InferenceAPI
pipeline execution offline without requiring live network traffic.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR
from src.detection.inference import InferenceAPI
from src.preprocessing import Preprocessor
from tools.live_capture import (
    CIC_67_FEATURES,
    UNAVAILABLE_CIC_FEATURES,
    extract_flow_metadata,
    get_available_interfaces,
    map_nflow_to_cic_features,
    resolve_interface,
)


class MockNFlow:
    """Mock NFStream flow object mimicking an active TCP/TLS flow."""

    def __init__(
        self,
        protocol: int = 6,
        duration_ms: float = 250.0,
        src_ip: str = "192.168.1.100",
        src_port: int = 54321,
        dst_ip: str = "93.184.216.34",
        dst_port: int = 443,
        src2dst_pkts: int = 15,
        dst2src_pkts: int = 10,
        src2dst_bytes: int = 2400,
        dst2src_bytes: int = 8000,
    ):
        self.id = 1
        self.protocol = protocol
        self.ip_version = 4
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port

        self.bidirectional_duration_ms = duration_ms
        self.bidirectional_packets = src2dst_pkts + dst2src_pkts
        self.bidirectional_bytes = src2dst_bytes + dst2src_bytes
        self.src2dst_duration_ms = duration_ms * 0.9
        self.src2dst_packets = src2dst_pkts
        self.src2dst_bytes = src2dst_bytes
        self.dst2src_duration_ms = duration_ms * 0.8
        self.dst2src_packets = dst2src_pkts
        self.dst2src_bytes = dst2src_bytes

        self.bidirectional_min_ps = 54.0
        self.bidirectional_mean_ps = 416.0
        self.bidirectional_stddev_ps = 450.0
        self.bidirectional_max_ps = 1460.0

        self.src2dst_min_ps = 54.0
        self.src2dst_mean_ps = 160.0
        self.src2dst_stddev_ps = 200.0
        self.src2dst_max_ps = 800.0

        self.dst2src_min_ps = 54.0
        self.dst2src_mean_ps = 800.0
        self.dst2src_stddev_ps = 500.0
        self.dst2src_max_ps = 1460.0

        self.bidirectional_min_piat_ms = 0.5
        self.bidirectional_mean_piat_ms = 10.0
        self.bidirectional_stddev_piat_ms = 15.0
        self.bidirectional_max_piat_ms = 45.0

        self.src2dst_min_piat_ms = 1.0
        self.src2dst_mean_piat_ms = 16.0
        self.src2dst_stddev_piat_ms = 20.0
        self.src2dst_max_piat_ms = 50.0

        self.dst2src_min_piat_ms = 2.0
        self.dst2src_mean_piat_ms = 24.0
        self.dst2src_stddev_piat_ms = 25.0
        self.dst2src_max_piat_ms = 60.0

        self.src2dst_psh_packets = 2
        self.bidirectional_fin_packets = 1
        self.bidirectional_syn_packets = 1
        self.bidirectional_rst_packets = 0
        self.bidirectional_psh_packets = 4
        self.bidirectional_ack_packets = 24
        self.bidirectional_urg_packets = 0
        self.bidirectional_ece_packets = 0

        self.application_name = "TLS"
        self.application_category_name = "Web"
        self.bidirectional_first_seen_ms = 1700000000000
        self.bidirectional_last_seen_ms = 1700000000250


def test_map_nflow_produces_exact_67_features():
    """Verify that mapping produces all 67 required feature keys."""
    mock_flow = MockNFlow()
    mapped = map_nflow_to_cic_features(mock_flow)

    assert len(mapped) == 67
    assert list(mapped.keys()) == CIC_67_FEATURES

    # Verify duration and IAT units are converted from ms to us
    assert mapped["Flow Duration"] == mock_flow.bidirectional_duration_ms * 1000.0
    assert mapped["Flow IAT Mean"] == mock_flow.bidirectional_mean_piat_ms * 1000.0
    assert mapped["Flow Bytes/s"] > 0
    assert mapped["Flow Packets/s"] > 0
    assert mapped["Down/Up Ratio"] == pytest.approx(10 / 15)
    assert mapped["Packet Length Variance"] == pytest.approx(mock_flow.bidirectional_stddev_ps ** 2)


def test_map_nflow_matches_preprocessor_trained_features():
    """Verify feature alignment directly against the production preprocessor."""
    prep_path = MODELS_DIR / "preprocessor.joblib"
    assert prep_path.exists(), f"Preprocessor checkpoint missing at {prep_path}"

    preprocessor = Preprocessor.load(prep_path)
    assert len(preprocessor.feature_names_) == 67
    assert preprocessor.feature_names_ == CIC_67_FEATURES

    mock_flow = MockNFlow()
    features = map_nflow_to_cic_features(mock_flow)
    df = pd.DataFrame([features])

    # Transform through fitted Preprocessor
    X_scaled = preprocessor.transform(df)

    assert X_scaled.shape == (1, 67)
    assert not np.isnan(X_scaled).any(), "NaN values found in preprocessor output!"
    assert not np.isinf(X_scaled).any(), "Infinite values found in preprocessor output!"


def test_unavailable_features_reported_and_imputed():
    """Verify unavailable features are tracked and properly imputed with medians."""
    mock_flow = MockNFlow()
    features = map_nflow_to_cic_features(mock_flow)

    # Check that all features in UNAVAILABLE_CIC_FEATURES are set to NaN in raw mapping
    for unavail_col in UNAVAILABLE_CIC_FEATURES.keys():
        assert unavail_col in features
        assert np.isnan(features[unavail_col]), f"Expected NaN for unavailable feature: {unavail_col}"

    preprocessor = Preprocessor.load(MODELS_DIR / "preprocessor.joblib")
    df = pd.DataFrame([features])
    X_scaled = preprocessor.transform(df)

    # Ensure no NaN remains after transform
    assert not np.isnan(X_scaled).any()


def test_end_to_end_inference_on_mapped_flow():
    """Verify that InferenceAPI successfully scores a DataFrame of mapped flows."""
    model_path = MODELS_DIR / "gtae_ids.pt"
    prep_path = MODELS_DIR / "preprocessor.joblib"
    assert model_path.exists(), f"GTAE model missing at {model_path}"

    inference_api = InferenceAPI(
        model_path=model_path,
        preprocessor_path=prep_path,
        device="cpu",
    )

    # Batch of 2 distinct mock flows
    flows = [
        MockNFlow(protocol=6, duration_ms=100.0, src2dst_pkts=10, dst2src_pkts=8),
        MockNFlow(protocol=17, duration_ms=50.0, src2dst_pkts=2, dst2src_pkts=2),
    ]
    df = pd.DataFrame([map_nflow_to_cic_features(f) for f in flows])

    results = inference_api.predict(df)

    assert len(results) == 2
    for r in results:
        assert "category" in r
        assert "detected_type" in r
        assert "classifier_prob" in r
        assert "anomaly_score" in r
        assert "risk_score" in r
        assert "severity" in r
        assert "is_anomalous" in r
        assert isinstance(r["risk_score"], (int, float))
        assert r["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_extract_flow_metadata():
    """Verify flow metadata extraction correctly captures endpoints and app details."""
    mock_flow = MockNFlow(
        src_ip="10.0.0.1",
        src_port=12345,
        dst_ip="10.0.0.2",
        dst_port=80,
    )
    meta = extract_flow_metadata(mock_flow)

    assert meta["src_ip"] == "10.0.0.1"
    assert meta["src_port"] == 12345
    assert meta["dst_ip"] == "10.0.0.2"
    assert meta["dst_port"] == 80
    assert meta["protocol"] == 6
    assert meta["application_name"] == "TLS"


def test_zero_duration_and_edge_cases():
    """Verify division-by-zero protection when flow duration or packet count is zero."""
    zero_flow = MockNFlow(
        duration_ms=0.0,
        src2dst_pkts=0,
        dst2src_pkts=0,
        src2dst_bytes=0,
        dst2src_bytes=0,
    )
    mapped = map_nflow_to_cic_features(zero_flow)

    assert mapped["Flow Bytes/s"] == 0.0
    assert mapped["Flow Packets/s"] == 0.0
    assert mapped["Fwd Packets/s"] == 0.0
    assert mapped["Bwd Packets/s"] == 0.0
    assert mapped["Down/Up Ratio"] == 0.0


def test_interface_enumeration_and_resolution():
    """Verify interface detection returns non-empty list and resolves valid indices."""
    interfaces = get_available_interfaces()
    assert isinstance(interfaces, list)
    assert len(interfaces) > 0, "Expected at least one network interface on Windows system"

    first_iface = interfaces[0]
    assert "index" in first_iface
    assert "name" in first_iface
    assert "description" in first_iface

    # Test resolving by index 0
    resolved = resolve_interface(0)
    assert resolved == first_iface["name"]

    # Test resolving by invalid name raises ValueError
    with pytest.raises(ValueError):
        resolve_interface("NonExistentInterface_123456789")
