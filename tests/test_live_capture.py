"""
Focused unit tests for the live traffic capture module (tools/live_capture.py).
Validates NFStream-to-67-feature mapping, Preprocessor integration, and InferenceAPI
pipeline execution offline without requiring live network traffic.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import struct
import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR
from src.detection.inference import InferenceAPI
from src.preprocessing import Preprocessor
from tools.live_capture import (
    CIC_67_FEATURES,
    UNAVAILABLE_CIC_FEATURES,
    CICFeaturePlugin,
    extract_flow_metadata,
    extract_tcp_window,
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


# -----------------------------------------------------------------------------
# Tests for Dynamic CIC-IDS2017 Feature Extraction & CICFeaturePlugin
# -----------------------------------------------------------------------------
class MockPacket:
    """Mock NFStream packet matching pythonize_packet fields."""

    def __init__(
        self,
        direction: int = 0,
        ip_size: float = 1500.0,
        transport_size: float = 1480.0,
        payload_size: float = 1448.0,
        protocol: int = 6,
        ip_version: int = 4,
        syn: int = 0,
        ip_packet: bytes = None,
    ):
        self.direction = direction
        self.ip_size = ip_size
        self.transport_size = transport_size
        self.payload_size = payload_size
        self.protocol = protocol
        self.ip_version = ip_version
        self.syn = syn
        self.ip_packet = ip_packet


def test_forward_backward_header_accumulation():
    """Verify forward and backward header length accumulation via CICFeaturePlugin."""
    plugin = CICFeaturePlugin()
    flow = SimpleNamespace(udps=SimpleNamespace())

    # Packet 1 (fwd): ip_size=100, payload_size=60 -> header = 40
    pkt1 = MockPacket(direction=0, ip_size=100.0, transport_size=80.0, payload_size=60.0)
    plugin.on_init(pkt1, flow)

    # Packet 2 (fwd): ip_size=1500, payload_size=1460 -> header = 40
    pkt2 = MockPacket(direction=0, ip_size=1500.0, transport_size=1480.0, payload_size=1460.0)
    plugin.on_update(pkt2, flow)

    # Packet 3 (bwd): ip_size=200, payload_size=148 -> header = 52
    pkt3 = MockPacket(direction=1, ip_size=200.0, transport_size=180.0, payload_size=148.0)
    plugin.on_update(pkt3, flow)

    assert flow.udps.fwd_header_len == 80.0  # 40 + 40
    assert flow.udps.bwd_header_len == 52.0  # 52


def test_tcp_initial_window_extraction():
    """Verify advertised TCP window is correctly parsed from raw IPv4 and IPv6 SYN packets."""
    # 1. IPv4 SYN with window size 64240
    ip_hdr_v4 = bytearray(20)
    ip_hdr_v4[0] = 0x45  # IPv4, IHL=5 (20 bytes)
    ip_hdr_v4[9] = 6    # Protocol TCP
    tcp_hdr_v4 = bytearray(20)
    struct.pack_into("!H", tcp_hdr_v4, 14, 64240)  # Offset 14-15 = Window
    pkt_v4 = MockPacket(
        direction=0,
        protocol=6,
        ip_version=4,
        syn=1,
        ip_packet=bytes(ip_hdr_v4 + tcp_hdr_v4),
    )

    win_v4 = extract_tcp_window(pkt_v4)
    assert win_v4 == 64240.0

    # 2. IPv6 SYN with window size 28960
    ip_hdr_v6 = bytearray(40)
    ip_hdr_v6[0] = 0x60  # IPv6
    tcp_hdr_v6 = bytearray(20)
    struct.pack_into("!H", tcp_hdr_v6, 14, 28960)
    pkt_v6 = MockPacket(
        direction=1,
        protocol=6,
        ip_version=6,
        syn=1,
        ip_packet=bytes(ip_hdr_v6 + tcp_hdr_v6),
    )

    win_v6 = extract_tcp_window(pkt_v6)
    assert win_v6 == 28960.0

    # 3. Truncated packet returns None safely
    pkt_short = MockPacket(direction=0, protocol=6, ip_version=4, syn=1, ip_packet=b"\x45\x00")
    assert extract_tcp_window(pkt_short) is None


def test_non_tcp_initial_window_is_minus_one():
    """Verify non-TCP protocols (UDP, QUIC, IGMP) strictly set initial window bytes to -1.0."""
    # UDP / QUIC (Protocol 17)
    udp_flow = MockNFlow(protocol=17)
    features_udp = map_nflow_to_cic_features(udp_flow)
    assert features_udp["Init Fwd Win Bytes"] == -1.0
    assert features_udp["Init Bwd Win Bytes"] == -1.0

    # IGMP (Protocol 2)
    igmp_flow = MockNFlow(protocol=2)
    features_igmp = map_nflow_to_cic_features(igmp_flow)
    assert features_igmp["Init Fwd Win Bytes"] == -1.0
    assert features_igmp["Init Bwd Win Bytes"] == -1.0


def test_forward_active_data_packet_counting():
    """Verify forward active data packet counts only packets carrying non-zero payload."""
    plugin = CICFeaturePlugin()
    flow = SimpleNamespace(udps=SimpleNamespace())

    # Packet 1 (fwd): SYN control packet (payload_size = 0)
    pkt1 = MockPacket(direction=0, payload_size=0.0)
    plugin.on_init(pkt1, flow)

    # Packet 2 (fwd): Data packet (payload_size = 500)
    pkt2 = MockPacket(direction=0, payload_size=500.0)
    plugin.on_update(pkt2, flow)

    # Packet 3 (fwd): ACK keepalive packet (payload_size = 0)
    pkt3 = MockPacket(direction=0, payload_size=0.0)
    plugin.on_update(pkt3, flow)

    # Packet 4 (bwd): Return data (direction = 1, should not increment fwd counter)
    pkt4 = MockPacket(direction=1, payload_size=1000.0)
    plugin.on_update(pkt4, flow)

    assert flow.udps.fwd_act_data_packets == 1.0


def test_forward_segment_minimum():
    """Verify minimum forward segment size (transport_size - payload_size) is tracked."""
    plugin = CICFeaturePlugin()
    flow = SimpleNamespace(udps=SimpleNamespace())

    # Packet 1 (fwd): transport_size=40, payload_size=0 -> seg_sz = 40
    pkt1 = MockPacket(direction=0, transport_size=40.0, payload_size=0.0)
    plugin.on_init(pkt1, flow)

    # Packet 2 (fwd): transport_size=1020, payload_size=1000 -> seg_sz = 20
    pkt2 = MockPacket(direction=0, transport_size=1020.0, payload_size=1000.0)
    plugin.on_update(pkt2, flow)

    # Packet 3 (fwd): transport_size=532, payload_size=500 -> seg_sz = 32
    pkt3 = MockPacket(direction=0, transport_size=532.0, payload_size=500.0)
    plugin.on_update(pkt3, flow)

    assert flow.udps.fwd_seg_size_min == 20.0


def test_handshake_missing_fallback_behavior():
    """Verify robust fallback behavior when handshake packets (SYN/SYN-ACK) are unobserved."""
    # Flow with plugin udps but no observed SYN or SYN-ACK
    tcp_flow = MockNFlow(protocol=6, src2dst_pkts=5, dst2src_pkts=5)
    tcp_flow.udps = SimpleNamespace(
        fwd_header_len=160.0,
        bwd_header_len=160.0,
        init_fwd_win_bytes=None,  # SYN missed
        init_bwd_win_bytes=None,  # SYN-ACK missed
        fwd_act_data_packets=3.0,
        fwd_seg_size_min=32.0,
    )

    features = map_nflow_to_cic_features(tcp_flow)

    assert features["Fwd Header Length"] == 160.0
    assert features["Bwd Header Length"] == 160.0
    assert features["Init Fwd Win Bytes"] == 1834.5  # Training-compatible TCP fallback
    assert features["Init Bwd Win Bytes"] == 131.0   # Training-compatible TCP fallback
    assert features["Fwd Act Data Packets"] == 3.0
    assert features["Fwd Seg Size Min"] == 32.0
