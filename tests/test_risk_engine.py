import pytest
from src.detection.risk_engine import ThreatRiskEngine

def test_risk_engine_initialization():
    engine = ThreatRiskEngine()
    assert engine.anomaly_weight == 0.3
    assert engine.base_weight == 0.7

def test_risk_score_benign():
    engine = ThreatRiskEngine()
    result = engine.compute_risk_score("BENIGN", anomaly_score=0.5, classifier_prob=0.99)
    # base = 5.0 * 1.0 (BENIGN has no prob scaling) = 5.0
    # anomaly = (0.5 / 10.0) * 100 = 5.0
    # score = 0.7 * 5.0 + 0.3 * 5.0 = 3.5 + 1.5 = 5.0
    assert result["risk_score"] == 5.0
    assert result["severity"] == "LOW"

def test_risk_score_known_attack_high_conf():
    engine = ThreatRiskEngine()
    result = engine.compute_risk_score("DDoS", anomaly_score=8.0, classifier_prob=0.9)
    # base = 80.0 * 0.9 = 72.0
    # anomaly = (8.0 / 10.0) * 100 = 80.0
    # score = 0.7 * 72.0 + 0.3 * 80.0 = 50.4 + 24.0 = 74.4
    assert result["risk_score"] == 74.4
    assert result["severity"] == "HIGH"

def test_risk_score_unknown_novel():
    engine = ThreatRiskEngine()
    result = engine.compute_risk_score("UNKNOWN_NOVEL", anomaly_score=15.0, classifier_prob=0.4)
    # base = 85.0 * 1.0 = 85.0
    # anomaly = min(15.0 / 10.0, 1.0) * 100 = 100.0
    # score = 0.7 * 85.0 + 0.3 * 100.0 = 59.5 + 30.0 = 89.5
    assert result["risk_score"] == 89.5
    assert result["severity"] == "CRITICAL"

def test_severity_levels():
    engine = ThreatRiskEngine()
    assert engine.get_severity_level(20.0) == "LOW"
    assert engine.get_severity_level(45.0) == "MEDIUM"
    assert engine.get_severity_level(65.0) == "HIGH"
    assert engine.get_severity_level(85.0) == "CRITICAL"
