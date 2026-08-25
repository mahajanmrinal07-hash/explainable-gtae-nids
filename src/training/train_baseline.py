"""
Baseline Model Training and Evaluation Pipeline for Real CIC-IDS2017 Dataset.
Loads all 8 real parquet files, performs leakage-free preprocessing, trains Random Forest,
and exports comprehensive performance metrics, per-class evaluations, and confusion matrices.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
)
from src.data_loader import (
    discover_raw_files,
    load_stratified_real_dataset,
)
from src.models.baseline import BaselineRandomForest
from src.preprocessing import prepare_splits


def plot_and_save_confusion_matrix(
    cm: list,
    class_names: list,
    output_path: Path,
    title: str = "Baseline Confusion Matrix",
) -> None:
    """
    Renders and saves a high-resolution heatmap of the confusion matrix.
    """
    plt.figure(figsize=(10, 8), dpi=300)
    cm_array = np.array(cm)

    # Normalize by true class
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_array.astype("float") / cm_array.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, labelpad=8)
    plt.ylabel("True Label", fontsize=12, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300)
    plt.close()


def run_baseline_training(
    data_dir: Optional[str] = None,
    max_benign_samples: int = 60000,
    max_attack_samples_per_class: int = 15000,
    mode: str = "multiclass",
    n_estimators: int = 100,
    max_depth: int = 20,
    random_state: int = RANDOM_SEED,
) -> dict:
    """
    Executes the end-to-end baseline training, evaluation, and artifact export on the REAL dataset.
    """
    print("=" * 75)
    print("      XAI-NIDS: BASELINE RANDOM FOREST ON REAL CIC-IDS2017 DATASET")
    print("=" * 75)

    # 1. Dataset Resolution & Discovery
    discovery = discover_raw_files(data_dir)
    print(f"[*] Discovered {discovery['found_count']} / {discovery['expected_count']} raw Parquet files in: {discovery['directory']}")
    if discovery["found_count"] == 0:
        raise FileNotFoundError(f"No raw Parquet files found in: {discovery['directory']}")

    for f_info in discovery["found_files"]:
        print(f"  • {f_info['filename']:40} ({f_info['size_mb']} MB)")

    # 2. Stratified Loading with Class Imbalance Management & Deduplication
    print("\n[*] Loading real dataset with stratified sampling & cross-file deduplication...")
    print(f"    (Max benign: {max_benign_samples:,}, Max attack per class: {max_attack_samples_per_class:,})")
    
    df = load_stratified_real_dataset(
        raw_dir=discovery["directory"],
        max_benign_samples=max_benign_samples,
        max_attack_samples_per_class=max_attack_samples_per_class,
        deduplicate=True,
        random_state=random_state,
    )

    print(f"[+] Loaded cleaned dataset : {len(df):,} rows, {df.shape[1]} columns")
    print(f"[*] Target mode            : {mode.upper()}")

    # 3. Stratified Preprocessing & Splitting (Fit only on Train)
    print("\n[*] Preprocessing features and performing train/val/test split (70/10/20)...")
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, class_map = prepare_splits(
        df,
        label_column="Label",
        mode=mode,
        test_size=0.2,
        val_size=0.1,
        scaler_type="robust",
        random_state=random_state,
    )

    unique_classes_in_data = np.unique(np.concatenate([y_train, y_val, y_test]))
    class_names = [class_map.get(i, str(i)) for i in unique_classes_in_data]

    print(f" • Training set shape   : {X_train.shape}")
    print(f" • Validation set shape : {X_val.shape}")
    print(f" • Test set shape       : {X_test.shape}")
    print(f" • Input features used  : {len(preprocessor.feature_names_)}")
    print(f" • Target classes ({len(class_names)}): {class_names}")

    # Display class counts in split
    train_counts = {class_map.get(k, str(k)): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    test_counts = {class_map.get(k, str(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
    print(f" • Train class distribution: {train_counts}")
    print(f" • Test class distribution : {test_counts}")

    # 4. Model Training
    print(f"\n[*] Training Random Forest Classifier (n_estimators={n_estimators}, max_depth={max_depth}, class_weight='balanced_subsample')...")
    rf_model = BaselineRandomForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    rf_model.fit(X_train, y_train)
    print("[+] Model training completed successfully.")

    # 5. Evaluation on Unseen Test Set
    print("\n[*] Evaluating baseline model on unseen Test Set...")
    eval_results = rf_model.evaluate(X_test, y_test, target_names=class_names)

    # 6. Save Artifacts
    print("\n[*] Persisting trained artifacts and performance reports...")
    model_save_path = MODELS_DIR / "baseline_rf.joblib"
    preprocessor_save_path = MODELS_DIR / "preprocessor.joblib"
    rf_model.save(model_save_path)
    preprocessor.save(preprocessor_save_path)

    # Save metrics JSON
    metrics_path = METRICS_DIR / "baseline_metrics.json"
    metrics_to_export = {
        "dataset": "CIC-IDS2017 (Real)",
        "model_type": "RandomForestClassifier (Balanced Subsample)",
        "mode": mode,
        "n_samples_total": int(len(df)),
        "n_samples_train": int(len(y_train)),
        "n_samples_val": int(len(y_val)),
        "n_samples_test": int(len(y_test)),
        "num_features": int(len(preprocessor.feature_names_)),
        "feature_names": preprocessor.feature_names_,
        "classes": class_names,
        "accuracy": eval_results["accuracy"],
        "precision_macro": eval_results["precision_macro"],
        "recall_macro": eval_results["recall_macro"],
        "f1_macro": eval_results["f1_macro"],
        "precision_weighted": eval_results["precision_weighted"],
        "recall_weighted": eval_results["recall_weighted"],
        "f1_weighted": eval_results["f1_weighted"],
        "per_class_metrics": eval_results["per_class_metrics"],
        "confusion_matrix": eval_results["confusion_matrix"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_export, f, indent=4)

    # Save text report
    report_text_path = METRICS_DIR / "baseline_classification_report.txt"
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("XAI-NIDS Baseline Random Forest Classification Report (Real CIC-IDS2017)\n")
        f.write("=" * 75 + "\n\n")
        f.write(eval_results["classification_report_str"])
        f.write("\n\nTop 20 Most Predictive Network Flow Features:\n")
        f.write("-" * 60 + "\n")
        importances = rf_model.get_feature_importances(preprocessor.feature_names_)[:20]
        for rank, item in enumerate(importances, 1):
            f.write(f" {rank:2d}. {item['feature']:35}: {item['importance']:.6f}\n")

    # Save confusion matrix plot
    cm_plot_path = FIGURES_DIR / "baseline_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        cm=eval_results["confusion_matrix"],
        class_names=class_names,
        output_path=cm_plot_path,
        title=f"Real CIC-IDS2017 Baseline Random Forest ({mode.capitalize()}) - Normalized Confusion Matrix",
    )

    # 7. Print Summary
    print("\n" + "=" * 75)
    print("                     EVALUATION RESULTS SUMMARY")
    print("=" * 75)
    print(f" Overall Accuracy   : {eval_results['accuracy']:.4f}")
    print(f" Macro Precision    : {eval_results['precision_macro']:.4f}")
    print(f" Macro Recall       : {eval_results['recall_macro']:.4f}")
    print(f" Macro F1-Score     : {eval_results['f1_macro']:.4f}")
    print(f" Weighted F1-Score  : {eval_results['f1_weighted']:.4f}")
    print("-" * 75)
    print("Classification Report:")
    print(eval_results["classification_report_str"])
    print("-" * 75)
    print("Saved Artifacts:")
    print(f" • Model Artifact   : {model_save_path}")
    print(f" • Preprocessor     : {preprocessor_save_path}")
    print(f" • Metrics JSON     : {metrics_path}")
    print(f" • Report Text      : {report_text_path}")
    print(f" • Confusion Matrix : {cm_plot_path}")
    print("=" * 75)

    return metrics_to_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline Random Forest model on real CIC-IDS2017 dataset.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing raw parquet files.")
    parser.add_argument("--max_benign", type=int, default=60000, help="Max benign samples to load.")
    parser.add_argument("--max_attack", type=int, default=15000, help="Max attack samples per class.")
    parser.add_argument("--mode", type=str, choices=["binary", "multiclass"], default="multiclass", help="Classification mode.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees.")
    parser.add_argument("--max_depth", type=int, default=20, help="Maximum tree depth.")

    args = parser.parse_args()
    run_baseline_training(
        data_dir=args.data_dir,
        max_benign_samples=args.max_benign,
        max_attack_samples_per_class=args.max_attack,
        mode=args.mode,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
