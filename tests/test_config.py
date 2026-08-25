"""
Unit tests for project configuration.
"""

from src.config import (
    BINARY_LABEL_MAP,
    DATA_DIR,
    EXPECTED_PARQUET_FILES,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    MULTICLASS_LABEL_MAP,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SAMPLES_DATA_DIR,
)


def test_directories_exist():
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()
    assert SAMPLES_DATA_DIR.exists()
    assert MODELS_DIR.exists()
    assert FIGURES_DIR.exists()
    assert METRICS_DIR.exists()


def test_config_constants():
    assert RANDOM_SEED == 42
    assert len(EXPECTED_PARQUET_FILES) == 8
    assert "BENIGN" in BINARY_LABEL_MAP
    assert "ATTACK" in BINARY_LABEL_MAP
    assert "DoS" in MULTICLASS_LABEL_MAP
    assert "DDoS" in MULTICLASS_LABEL_MAP
