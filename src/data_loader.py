"""
Data loading and dataset discovery module for CIC-IDS2017 Parquet datasets.
Supports real dataset discovery, schema validation, chunked/sampled loading,
and stratified sampling for class imbalance management.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    EXPECTED_PARQUET_FILES,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SAMPLES_DATA_DIR,
)

# Standard CIC-IDS2017 feature column names (77 features + Label)
CIC_IDS2017_COLUMNS: List[str] = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Avg Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Act Data Packets",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
]


def generate_synthetic_sample_dataset(
    output_path: Optional[Union[str, Path]] = None,
    num_samples: int = 1000,
    random_state: int = RANDOM_SEED,
) -> Path:
    """
    Generates a synthetic parquet file matching CIC-IDS2017 schema for offline unit tests.
    """
    rng = np.random.RandomState(random_state)
    n_features = len(CIC_IDS2017_COLUMNS) - 1
    feat_data = rng.randn(num_samples, n_features).astype(np.float32)

    labels = ["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator", "Infiltration", "Bot", "Web Attack - XSS"]
    assigned_labels = rng.choice(labels, size=num_samples)

    df_dict = {col: feat_data[:, i] for i, col in enumerate(CIC_IDS2017_COLUMNS[:-1])}
    df_dict["Label"] = assigned_labels
    df = pd.DataFrame(df_dict)

    if output_path is None:
        target = SAMPLES_DATA_DIR / "synthetic_sample.parquet"
    else:
        target = Path(output_path)

    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(target), index=False)
    return target



def discover_raw_files(raw_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Scans the raw data directory to identify existing and missing CIC-IDS2017 Parquet files.
    Checks candidate directories (data/raw/cicids2017 and data/raw).

    Returns:
        Dict containing found files, missing files, total size, and inspection status.
    """
    candidate_dirs = []
    if raw_dir:
        candidate_dirs.append(Path(raw_dir))
    else:
        candidate_dirs.append(RAW_DATA_DIR)
        candidate_dirs.append(RAW_DATA_DIR.parent / "cicids2017")
        candidate_dirs.append(RAW_DATA_DIR.parent)

    target_dir = candidate_dirs[0]
    for d in candidate_dirs:
        if d.exists() and len(list(d.glob("*.parquet"))) >= len(EXPECTED_PARQUET_FILES):
            target_dir = d
            break

    target_dir.mkdir(parents=True, exist_ok=True)

    found_files = []
    missing_files = []
    total_bytes = 0

    for fname in EXPECTED_PARQUET_FILES:
        target = target_dir / fname
        if target.exists() and target.is_file():
            size = target.stat().st_size
            total_bytes += size
            found_files.append({
                "filename": fname,
                "path": str(target),
                "size_mb": round(size / (1024 * 1024), 2),
            })
        else:
            missing_files.append(fname)

    # Check for any other parquet files in the folder
    other_files = [
        str(f.name) for f in target_dir.glob("*.parquet")
        if f.name not in EXPECTED_PARQUET_FILES
    ]

    return {
        "directory": str(target_dir),
        "found_count": len(found_files),
        "expected_count": len(EXPECTED_PARQUET_FILES),
        "found_files": found_files,
        "missing_files": missing_files,
        "other_parquet_files": other_files,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "is_complete": len(found_files) == len(EXPECTED_PARQUET_FILES),
    }


def inspect_parquet_schema(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspects columns, types, row count, and schema of a Parquet file without loading all into memory.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found at: {path}")

    parquet_file = pq.ParquetFile(str(path))
    schema = parquet_file.schema_arrow
    num_rows = parquet_file.metadata.num_rows
    num_columns = parquet_file.metadata.num_columns
    num_row_groups = parquet_file.metadata.num_row_groups

    column_names = schema.names

    return {
        "file_path": str(path),
        "num_rows": num_rows,
        "num_columns": num_columns,
        "num_row_groups": num_row_groups,
        "columns": column_names,
        "has_label": any(c.strip().lower() == "label" for c in column_names),
    }


def load_parquet_file(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    sample_fraction: Optional[float] = None,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Loads a single Parquet file with optional column selection and memory-conscious row sampling.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    table = pq.read_table(str(path), columns=columns)
    df = table.to_pandas()

    # Clean whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=random_state)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)

    return df.reset_index(drop=True)


def load_dataset_combined(
    file_paths: Optional[List[Union[str, Path]]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
    sample_per_file: Optional[int] = None,
    max_total_rows: Optional[int] = None,
    deduplicate: bool = True,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Loads and concatenates multiple Parquet files into a single unified DataFrame.
    """
    if file_paths is None:
        discovery = discover_raw_files(raw_dir)
        file_paths = [f["path"] for f in discovery["found_files"]]

    if not file_paths:
        raise FileNotFoundError(
            "No Parquet files found to load. Please verify dataset files exist in data/raw/cicids2017/."
        )

    dfs = []
    for fp in file_paths:
        df = load_parquet_file(
            fp,
            max_rows=sample_per_file,
            random_state=random_state,
        )
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    if deduplicate:
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    if max_total_rows is not None and len(combined_df) > max_total_rows:
        combined_df = combined_df.sample(n=max_total_rows, random_state=random_state)

    return combined_df.reset_index(drop=True)


def load_stratified_real_dataset(
    raw_dir: Optional[Union[str, Path]] = None,
    max_benign_samples: int = 60000,
    max_attack_samples_per_class: int = 15000,
    deduplicate: bool = True,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Loads the real CIC-IDS2017 dataset with an intelligent stratified sampling strategy:
    - Retains 100% of all rare attacks (Infiltration, Heartbleed, Web Attacks, Sql Injection, Botnet).
    - Caps oversized attack classes (DoS Hulk, DDoS) to `max_attack_samples_per_class`.
    - Samples Benign flows down to `max_benign_samples` to balance class distribution.
    - Prevents data leakage by dropping cross-file duplicate flows.
    """
    discovery = discover_raw_files(raw_dir)
    if discovery["found_count"] == 0:
        raise FileNotFoundError(
            f"No raw Parquet files found in: {discovery['directory']}"
        )

    dfs = []
    for f_info in discovery["found_files"]:
        df = load_parquet_file(f_info["path"])
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    if deduplicate:
        full_df = full_df.drop_duplicates().reset_index(drop=True)

    label_col = [c for c in full_df.columns if c.lower() == "label"][0]

    # Stratified per-class sampling
    sampled_groups = []
    for label, group in full_df.groupby(label_col):
        is_benign = "benign" in str(label).lower()
        if is_benign:
            n_take = min(len(group), max_benign_samples)
        else:
            n_take = min(len(group), max_attack_samples_per_class)

        sampled = group.sample(n=n_take, random_state=random_state)
        sampled_groups.append(sampled)

    stratified_df = pd.concat(sampled_groups, ignore_index=True)
    # Shuffle
    stratified_df = stratified_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return stratified_df
