"""
Unit tests for data loader and sample dataset generator.
"""

from pathlib import Path
import pytest
from src.data_loader import (
    CIC_IDS2017_COLUMNS,
    discover_raw_files,
    generate_synthetic_sample_dataset,
    inspect_parquet_schema,
    load_parquet_file,
)


def test_discover_raw_files_structure(tmp_path):
    discovery = discover_raw_files(raw_dir=tmp_path)
    assert "found_count" in discovery
    assert "missing_files" in discovery
    assert discovery["found_count"] == 0
    assert discovery["expected_count"] == 8


def test_synthetic_sample_dataset_generation(tmp_path):
    sample_file = tmp_path / "test_sample.parquet"
    target = generate_synthetic_sample_dataset(output_path=sample_file, num_samples=500, random_state=42)

    assert target.exists()
    schema_info = inspect_parquet_schema(target)
    assert schema_info["num_rows"] == 500
    assert schema_info["num_columns"] == len(CIC_IDS2017_COLUMNS)
    assert schema_info["has_label"] is True

    # Test loading with sampling
    df = load_parquet_file(target, max_rows=100)
    assert len(df) == 100
    assert "Label" in df.columns
