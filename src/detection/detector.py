"""
Real-time intrusion detector engine.
Categorizes traffic into BENIGN, KNOWN_ATTACK, or UNKNOWN_NOVEL.
"""

from typing import Dict, Any, List
import numpy as np

from src.detection.risk_engine import ThreatRiskEngine

class IntrusionDetector:
    """
    Evaluates real-time network flow graph embeddings and scores threat severity.
    Uses GTAE outputs: classification probabilities + node reconstruction errors.
    """

    def __init__(
        self, 
        anomaly_threshold: float = 2.0, 
        confidence_threshold: float = 0.6,
        class_names: Dict[int, str] = None
    ):
        """
        Args:
            anomaly_threshold: Cutoff for reconstruction error to be considered anomalous.
            confidence_threshold: Minimum classifier probability to accept a KNOWN_ATTACK prediction.
            class_names: Mapping from class integer to class name.
        """
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold
        self.risk_engine = ThreatRiskEngine()
        
        # Default mapping if none provided
        self.class_names = class_names or {
            0: "BENIGN", 1: "DoS", 2: "DDoS", 3: "PortScan", 
            4: "Brute Force", 5: "Botnet", 6: "Web Attack", 7: "Infiltration"
        }

    def detect_batch(
        self, 
        predicted_classes: np.ndarray, 
        class_probabilities: np.ndarray, 
        anomaly_scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a batch of flows.

        Args:
            predicted_classes: (N,) array of predicted class IDs.
            class_probabilities: (N, C) array of softmax probabilities.
            anomaly_scores: (N,) array of reconstruction errors.

        Returns:
            List of detection result dictionaries.
        """
        results = []
        for i in range(len(predicted_classes)):
            pred_id = int(predicted_classes[i])
            pred_name = self.class_names.get(pred_id, "ATTACK")
            prob = float(class_probabilities[i, pred_id])
            err = float(anomaly_scores[i])

            # Decision Logic
            is_anomalous = err > self.anomaly_threshold
            
            if pred_name == "BENIGN":
                if is_anomalous:
                    # High anomaly but classified as benign -> Unknown/Novel Threat Candidate
                    category = "UNKNOWN_NOVEL"
                    detected_type = "UNKNOWN_NOVEL"
                else:
                    category = "BENIGN"
                    detected_type = "BENIGN"
            else:
                # Classified as an attack
                if prob < self.confidence_threshold and is_anomalous:
                    # Low confidence in known attack, but high anomaly -> Novel variant
                    category = "UNKNOWN_NOVEL"
                    detected_type = "UNKNOWN_NOVEL"
                else:
                    category = "KNOWN_ATTACK"
                    detected_type = pred_name

            # Risk Score
            risk_info = self.risk_engine.compute_risk_score(
                attack_type=detected_type,
                anomaly_score=err,
                classifier_prob=prob
            )

            results.append({
                "node_id": i,
                "category": category,
                "detected_type": detected_type,
                "classifier_prob": prob,
                "anomaly_score": err,
                "risk_score": risk_info["risk_score"],
                "severity": risk_info["severity"],
                "is_anomalous": is_anomalous
            })
            
        return results
