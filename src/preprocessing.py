"""
Data preprocessing, cleaning, label normalization, and feature transformation module.
Strictly prevents data leakage by fitting all scalers and imputers solely on training data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import (
    BINARY_LABEL_MAP,
    MULTICLASS_INDEX_TO_NAME,
    MULTICLASS_LABEL_MAP,
    RANDOM_SEED,
    RAW_LABEL_TO_CATEGORY,
)


def normalize_labels(
    labels: Union[pd.Series, np.ndarray, List[str]],
    mode: str = "multiclass",
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Normalizes raw CIC-IDS2017 string labels into integer class IDs.
    Handles non-standard unicode artifacts (e.g. '\\ufffd', dashes) and category groupings.

    Args:
        labels: Iterable of raw string labels.
        mode: 'binary' (0=BENIGN, 1=ATTACK) or 'multiclass' (0=BENIGN, 1=DoS, 2=DDoS, etc.)

    Returns:
        Tuple of (encoded_labels_array, class_name_mapping_dict).
    """
    def sanitize_label(val: str) -> str:
        s = str(val).strip()
        # Clean unicode replacement characters
        s = s.replace("\ufffd", "-").replace("–", "-").replace("—", "-")
        # Direct lookup
        if s in RAW_LABEL_TO_CATEGORY:
            return RAW_LABEL_TO_CATEGORY[s]
        
        # Fuzzy category matching
        s_lower = s.lower()
        if "benign" in s_lower:
            return "BENIGN"
        elif "hulk" in s_lower or "goldeneye" in s_lower or "slowloris" in s_lower or "slowhttp" in s_lower or "heartbleed" in s_lower or "dos" in s_lower and "ddos" not in s_lower:
            return "DoS"
        elif "ddos" in s_lower:
            return "DDoS"
        elif "portscan" in s_lower:
            return "PortScan"
        elif "patator" in s_lower or "brute" in s_lower and "web" not in s_lower:
            return "Brute Force"
        elif "bot" in s_lower:
            return "Botnet"
        elif "web" in s_lower or "xss" in s_lower or "sql" in s_lower:
            return "Web Attack"
        elif "infiltration" in s_lower:
            return "Infiltration"
        return "ATTACK"

    series = pd.Series(labels).map(sanitize_label)

    if mode == "binary":
        encoded = np.where(series == "BENIGN", BINARY_LABEL_MAP["BENIGN"], BINARY_LABEL_MAP["ATTACK"])
        mapping = {0: "BENIGN", 1: "ATTACK"}
    elif mode == "multiclass":
        encoded = np.array([
            MULTICLASS_LABEL_MAP.get(cat, MULTICLASS_LABEL_MAP.get("BENIGN", 0))
            for cat in series
        ], dtype=np.int64)
        mapping = MULTICLASS_INDEX_TO_NAME
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'binary' or 'multiclass'.")

    return encoded, mapping


class Preprocessor:
    """
    Stateful preprocessor for tabular network flow data.
    Computes imputations, feature drops, and scalers strictly on training data.
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        clip_outliers: bool = True,
        clip_quantile: float = 0.999,
    ):
        self.scaler_type = scaler_type
        self.clip_outliers = clip_outliers
        self.clip_quantile = clip_quantile

        self.feature_names_: List[str] = []
        self.dropped_constant_columns_: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}
        self.scaler_ = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.is_fitted_: bool = False

    def _extract_and_clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes known metadata columns and ensures numeric typing.
        """
        df_clean = df.copy()
        df_clean.columns = [c.strip() for c in df_clean.columns]

        # Filter out metadata/identifier non-numeric columns if present
        meta_cols = ["Timestamp", "Flow ID", "Source IP", "Destination IP", "Label", "label", "target"]
        cols_to_drop = [c for c in meta_cols if c in df_clean.columns]
        df_clean = df_clean.drop(columns=cols_to_drop)

        # Replace infinite values with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        # Cast to float32 for numeric stability and memory efficiency
        return df_clean.astype(np.float32)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "Preprocessor":
        """
        Fits imputer medians, identifies constant features, and fits the scaler on training features only.
        """
        X_clean = self._extract_and_clean_columns(X)

        # Compute medians for NaN imputation
        self.medians_ = X_clean.median().to_dict()

        # Impute temporarily to identify constant columns
        X_imputed = X_clean.fillna(self.medians_)

        # Identify and drop constant columns (std == 0)
        stds = X_imputed.std(axis=0)
        self.dropped_constant_columns_ = list(stds[stds == 0].index)
        X_filtered = X_imputed.drop(columns=self.dropped_constant_columns_)

        self.feature_names_ = list(X_filtered.columns)

        # Outlier quantile bounding
        if self.clip_outliers:
            self.upper_bounds_ = X_filtered.quantile(self.clip_quantile).to_dict()
            for col in self.feature_names_:
                ub = self.upper_bounds_[col]
                X_filtered[col] = X_filtered[col].clip(upper=ub)

        # Fit scaler
        self.scaler_.fit(X_filtered.values)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame using the parameters fitted on training data.
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")

        X_clean = self._extract_and_clean_columns(X)

        # Apply training medians for imputation
        for col, med in self.medians_.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(med)

        # Drop constant columns identified during fit
        existing_drop = [c for c in self.dropped_constant_columns_ if c in X_clean.columns]
        X_filtered = X_clean.drop(columns=existing_drop)

        # Ensure exact column alignment
        for col in self.feature_names_:
            if col not in X_filtered.columns:
                X_filtered[col] = self.medians_.get(col, 0.0)
        X_filtered = X_filtered[self.feature_names_]

        # Apply upper bound clipping
        if self.clip_outliers:
            for col, ub in self.upper_bounds_.items():
                if col in X_filtered.columns:
                    X_filtered[col] = X_filtered[col].clip(upper=ub)

        # Scale
        X_scaled = self.scaler_.transform(X_filtered.values)
        if self.clip_outliers:
            X_scaled = np.clip(X_scaled, -10.0, 10.0)
        return X_scaled.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fits preprocessor and transforms input DataFrame.
        """
        return self.fit(X, y).transform(X)

    def save(self, file_path: Union[str, Path]) -> None:
        """
        Saves the fitted preprocessor object to disk using joblib.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))

    @staticmethod
    def load(file_path: Union[str, Path]) -> "Preprocessor":
        """
        Loads a saved Preprocessor object from disk.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at: {path}")
        return joblib.load(str(path))


def prepare_splits(
    df: pd.DataFrame,
    label_column: str = "Label",
    mode: str = "multiclass",
    test_size: float = 0.2,
    val_size: float = 0.1,
    scaler_type: str = "robust",
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Preprocessor, Dict[int, str]]:
    """
    Performs stratified train/val/test splitting and strictly isolated feature preprocessing.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, class_names_map)
    """
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    matched_label_col = None
    for col in df_clean.columns:
        if col.lower() == label_column.lower():
            matched_label_col = col
            break

    if matched_label_col is None:
        raise KeyError(f"Label column '{label_column}' not found in dataset columns: {list(df_clean.columns)}")

    raw_labels = df_clean[matched_label_col]
    y, class_map = normalize_labels(raw_labels, mode=mode)

    # Features (Drop label column strictly)
    X_df = df_clean.drop(columns=[matched_label_col])

    # Check minimum class counts for safe stratified splitting
    counts = pd.Series(y).value_counts()
    min_count = counts.min()

    stratify_arg = y if min_count >= 3 else None

    # First split: Train+Val vs Test
    X_temp, X_test_df, y_temp, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    # Second split: Train vs Val
    val_relative_size = val_size / (1.0 - test_size)
    counts_temp = pd.Series(y_temp).value_counts()
    stratify_val_arg = y_temp if counts_temp.min() >= 2 else None

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=stratify_val_arg,
    )

    # Fit preprocessor ONLY on training features
    preprocessor = Preprocessor(scaler_type=scaler_type)
    X_train = preprocessor.fit_transform(X_train_df)

    # Transform Val and Test features using the fitted preprocessor
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocessor,
        class_map,
    )
