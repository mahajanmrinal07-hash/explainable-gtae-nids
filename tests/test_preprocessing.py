"""
Unit tests for data preprocessing, label normalization, and data leakage safeguards.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_sample_dataset, load_parquet_file
from src.preprocessing import Preprocessor, normalize_labels, prepare_splits


def test_normalize_labels():
    raw_labels = ["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator", "Infiltration", "RandomUnknownAttack"]

    # Binary mode
    binary_encoded, bin_map = normalize_labels(raw_labels, mode="binary")
    assert binary_encoded[0] == 0  # BENIGN
    assert (binary_encoded[1:] == 1).all()  # ATTACKS

    # Multiclass mode
    multi_encoded, multi_map = normalize_labels(raw_labels, mode="multiclass")
    assert multi_encoded[0] == 0  # BENIGN
    assert multi_encoded[1] == 1  # DoS
    assert multi_encoded[2] == 3  # PortScan
    assert multi_encoded[3] == 4  # Brute Force
    assert multi_encoded[4] == 7  # Infiltration


def test_preprocessor_no_data_leakage(tmp_path):
    sample_file = tmp_path / "sample.parquet"
    generate_synthetic_sample_dataset(output_path=sample_file, num_samples=600, random_state=42)
    df = load_parquet_file(sample_file)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocessor,
        class_map,
    ) = prepare_splits(df, label_column="Label", test_size=0.2, val_size=0.1)

    assert preprocessor.is_fitted_ is True
    assert len(X_train) == 420
    assert len(X_val) == 60
    assert len(X_test) == 120

    # Ensure no NaN or infinite values remain
    assert not np.isnan(X_train).any()
    assert not np.isinf(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isinf(X_test).any()

    # Test save and reload
    save_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(save_path)
    loaded = Preprocessor.load(save_path)
    assert loaded.is_fitted_ is True
    assert loaded.feature_names_ == preprocessor.feature_names_

    # Verify transform output is identical
    transformed = loaded.transform(df.drop(columns=["Label"]))
    assert transformed.shape[1] == len(preprocessor.feature_names_)
