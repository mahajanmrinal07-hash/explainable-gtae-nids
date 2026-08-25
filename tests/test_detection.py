import pytest
import numpy as np
from src.detection.detector import IntrusionDetector

def test_detector_initialization():
    detector = IntrusionDetector(anomaly_threshold=2.5, confidence_threshold=0.7)
    assert detector.anomaly_threshold == 2.5
    assert detector.confidence_threshold == 0.7

def test_detector_benign_normal():
    detector = IntrusionDetector()
    preds = np.array([0]) # BENIGN
    probs = np.array([[0.9, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    anomaly = np.array([0.5]) # Low anomaly
    
    results = detector.detect_batch(preds, probs, anomaly)
    assert len(results) == 1
    assert results[0]["category"] == "BENIGN"
    assert results[0]["detected_type"] == "BENIGN"
    assert not results[0]["is_anomalous"]

def test_detector_benign_anomalous():
    detector = IntrusionDetector()
    preds = np.array([0]) # BENIGN
    probs = np.array([[0.9, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])
    anomaly = np.array([5.0]) # High anomaly -> UNKNOWN_NOVEL
    
    results = detector.detect_batch(preds, probs, anomaly)
    assert results[0]["category"] == "UNKNOWN_NOVEL"
    assert results[0]["detected_type"] == "UNKNOWN_NOVEL"
    assert results[0]["is_anomalous"]

def test_detector_known_attack_high_conf():
    detector = IntrusionDetector()
    preds = np.array([1]) # DoS
    probs = np.array([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
    anomaly = np.array([3.0]) 
    
    results = detector.detect_batch(preds, probs, anomaly)
    assert results[0]["category"] == "KNOWN_ATTACK"
    assert results[0]["detected_type"] == "DoS"
    assert results[0]["is_anomalous"]

def test_detector_known_attack_low_conf_anomalous():
    detector = IntrusionDetector()
    preds = np.array([1]) # DoS
    probs = np.array([[0.4, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]) # prob=0.5 < 0.6
    anomaly = np.array([3.0]) # High anomaly -> Novel variant
    
    results = detector.detect_batch(preds, probs, anomaly)
    assert results[0]["category"] == "UNKNOWN_NOVEL"
    assert results[0]["detected_type"] == "UNKNOWN_NOVEL"
    assert results[0]["is_anomalous"]
