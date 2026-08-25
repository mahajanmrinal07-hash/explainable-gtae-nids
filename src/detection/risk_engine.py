"""
Threat Risk Scoring and Severity Categorization Engine.
"""

from typing import Dict, Union

class ThreatRiskEngine:
    """
    Computes unified threat risk index combining anomaly confidence, classifier confidence, 
    and attack family severity.
    """

    # Base severity scores for different families
    BASE_SEVERITY = {
        "BENIGN": 5.0,
        "PortScan": 40.0,
        "Brute Force": 60.0,
        "DoS": 70.0,
        "DDoS": 80.0,
        "Botnet": 85.0,
        "Web Attack": 90.0,
        "Infiltration": 95.0,
        "UNKNOWN_NOVEL": 85.0,
        "ATTACK": 70.0
    }

    def __init__(self, anomaly_weight: float = 0.3, base_weight: float = 0.7):
        self.anomaly_weight = anomaly_weight
        self.base_weight = base_weight

    def get_severity_level(self, risk_score: float) -> str:
        """
        Categorizes 0-100 score into severity levels.
        """
        if risk_score <= 24.99:
            return "LOW"
        elif risk_score <= 49.99:
            return "MEDIUM"
        elif risk_score <= 74.99:
            return "HIGH"
        else:
            return "CRITICAL"

    def compute_risk_score(
        self,
        attack_type: str,
        anomaly_score: float,
        classifier_prob: float,
        max_expected_anomaly: float = 10.0
    ) -> Dict[str, Union[float, str]]:
        """
        Computes a deterministic 0-100 risk score.
        
        Formula:
        Base = BASE_SEVERITY[attack_type] * classifier_prob (if known attack, else 1.0 for BENIGN/UNKNOWN)
        Anomaly factor = min(anomaly_score / max_expected_anomaly, 1.0) * 100
        
        Risk Score = (base_weight * Base) + (anomaly_weight * Anomaly Factor)
        Capped at 100.0, min 0.0.
        """
        base_score = self.BASE_SEVERITY.get(attack_type, 50.0)
        
        if attack_type not in ["BENIGN", "UNKNOWN_NOVEL"]:
            # For known attacks, factor in model confidence
            base_component = base_score * max(classifier_prob, 0.5) 
        else:
            base_component = base_score

        # Normalize anomaly score to 0-100 range
        anomaly_factor = min(anomaly_score / max_expected_anomaly, 1.0) * 100.0

        # Weighted combination
        raw_score = (self.base_weight * base_component) + (self.anomaly_weight * anomaly_factor)
        
        final_score = max(0.0, min(100.0, float(raw_score)))
        severity = self.get_severity_level(final_score)

        return {
            "risk_score": round(final_score, 2),
            "severity": severity,
            "components": {
                "base_contribution": round(self.base_weight * base_component, 2),
                "anomaly_contribution": round(self.anomaly_weight * anomaly_factor, 2)
            }
        }
