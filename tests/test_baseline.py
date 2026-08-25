"""
Unit tests for baseline Random Forest model and evaluation metrics.
"""

import numpy as np
from src.data_loader import generate_synthetic_sample_dataset, load_parquet_file
from src.models.baseline import BaselineRandomForest
from src.preprocessing import prepare_splits


def test_baseline_random_forest_lifecycle(tmp_path):
    sample_file = tmp_path / "sample.parquet"
    generate_synthetic_sample_dataset(output_path=sample_file, num_samples=500, random_state=42)
    df = load_parquet_file(sample_file)

    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, class_map = prepare_splits(
        df, label_column="Label", mode="binary"
    )

    rf = BaselineRandomForest(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    assert rf.is_fitted is True

    preds = rf.predict(X_test)
    assert len(preds) == len(y_test)

    eval_dict = rf.evaluate(X_test, y_test)
    assert "accuracy" in eval_dict
    assert "precision_macro" in eval_dict
    assert "recall_macro" in eval_dict
    assert "f1_macro" in eval_dict
    assert "confusion_matrix" in eval_dict
    assert 0.0 <= eval_dict["accuracy"] <= 1.0

    # Test persistence
    model_path = tmp_path / "rf.joblib"
    rf.save(model_path)
    loaded_rf = BaselineRandomForest.load(model_path)
    assert loaded_rf.is_fitted is True
    np.testing.assert_array_equal(loaded_rf.predict(X_test), preds)
