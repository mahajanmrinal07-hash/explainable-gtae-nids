"""
Baseline Machine Learning Model (Random Forest Classifier) for Network Intrusion Detection.
Supports class-balanced weighting to handle severe class imbalance on real CIC-IDS2017 data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import BASELINE_CONFIG, RANDOM_SEED


class BaselineRandomForest:
    """
    Standard Random Forest benchmark for tabular network intrusion detection.
    """

    def __init__(
        self,
        n_estimators: int = BASELINE_CONFIG["n_estimators"],
        max_depth: Optional[int] = BASELINE_CONFIG["max_depth"],
        min_samples_split: int = BASELINE_CONFIG["min_samples_split"],
        class_weight: Optional[str] = BASELINE_CONFIG.get("class_weight", "balanced_subsample"),
        n_jobs: int = BASELINE_CONFIG["n_jobs"],
        random_state: int = RANDOM_SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineRandomForest":
        """
        Trains the Random Forest model on feature matrix X and label vector y.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates class predictions for input samples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generates class probability estimates for input samples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict_proba().")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates comprehensive classification metrics on the provided test dataset.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() before evaluate().")

        y_pred = self.predict(X)

        accuracy = float(accuracy_score(y, y_pred))
        precision_macro = float(precision_score(y, y_pred, average="macro", zero_division=0))
        recall_macro = float(recall_score(y, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y, y_pred, average="macro", zero_division=0))

        precision_weighted = float(precision_score(y, y_pred, average="weighted", zero_division=0))
        recall_weighted = float(recall_score(y, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y, y_pred, average="weighted", zero_division=0))

        # Per-class metrics
        per_class_precision = precision_score(y, y_pred, average=None, zero_division=0).tolist()
        per_class_recall = recall_score(y, y_pred, average=None, zero_division=0).tolist()
        per_class_f1 = f1_score(y, y_pred, average=None, zero_division=0).tolist()

        cm = confusion_matrix(y, y_pred).tolist()
        report_str = classification_report(
            y,
            y_pred,
            target_names=target_names,
            zero_division=0,
        )
        report_dict = classification_report(
            y,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        per_class_metrics = {}
        if target_names:
            for idx, name in enumerate(target_names):
                if idx < len(per_class_precision):
                    per_class_metrics[name] = {
                        "precision": round(per_class_precision[idx], 5),
                        "recall": round(per_class_recall[idx], 5),
                        "f1_score": round(per_class_f1[idx], 5),
                    }

        return {
            "accuracy": round(accuracy, 5),
            "precision_macro": round(precision_macro, 5),
            "recall_macro": round(recall_macro, 5),
            "f1_macro": round(f1_macro, 5),
            "precision_weighted": round(precision_weighted, 5),
            "recall_weighted": round(recall_weighted, 5),
            "f1_weighted": round(f1_weighted, 5),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm,
            "classification_report_str": report_str,
            "classification_report_dict": report_dict,
        }

    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Returns sorted list of feature importances.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        importances = self.model.feature_importances_
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        ranked = sorted(
            [{"feature": name, "importance": round(float(imp), 6)} for name, imp in zip(feature_names, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )
        return ranked

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Persists the trained model artifact to disk.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))

    @staticmethod
    def load(file_path: Union[str, Path]) -> "BaselineRandomForest":
        """
        Loads a persisted BaselineRandomForest model from disk.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")
        return joblib.load(str(path))
