"""
GTAE-IDS Training and Anomaly Detection Pipeline for Real CIC-IDS2017 Dataset.

Performs:
1. Stratified real data loading and isolated feature preprocessing.
2. Flow-similarity graph snapshot construction (NetworkGraphBuilder).
3. Hybrid multi-task training (Classification + Autoencoder Feature Reconstruction).
4. Mixed-precision CUDA execution with automatic CPU fallback.
5. Benign-calibrated reconstruction anomaly thresholding for novel/unknown attack detection.
6. Zero-day / held-out attack evaluation.
7. Comprehensive performance metrics export and high-resolution figure generation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    MULTICLASS_INDEX_TO_NAME,
    MULTICLASS_LABEL_MAP,
    RANDOM_SEED,
    RAW_DATA_DIR,
)
from src.data_loader import discover_raw_files, load_stratified_real_dataset
from src.graph_builder import NetworkGraphBuilder
from src.models.gtae_ids import GTAE_IDS
from src.preprocessing import prepare_splits


def plot_and_save_confusion_matrix(
    cm: list,
    class_names: list,
    output_path: Path,
    title: str = "GTAE-IDS Normalized Confusion Matrix",
) -> None:
    """
    Renders and exports a publication-grade normalized confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8), dpi=300)
    cm_array = np.array(cm, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_array.sum(axis=1)[:, np.newaxis]
        cm_norm = np.where(row_sums > 0, cm_array / row_sums, 0.0)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="#e0e0e0",
    )
    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Attack Class", fontsize=12, labelpad=8)
    plt.ylabel("True Attack Class", fontsize=12, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300)
    plt.close()


def plot_and_save_reconstruction_distribution(
    benign_errors: np.ndarray,
    attack_errors: np.ndarray,
    threshold: float,
    output_path: Path,
    heldout_errors: Optional[np.ndarray] = None,
    heldout_name: Optional[str] = None,
    title: str = "Reconstruction Error Distribution (Anomaly Detection)",
) -> None:
    """
    Renders and exports reconstruction error distribution plots comparing Benign, Attack,
    and optional Held-Out Attack flows against the calibrated anomaly threshold.
    """
    plt.figure(figsize=(10, 6), dpi=300)

    # Use 99th percentile for plot x-limit to prevent extreme visual distortion
    max_val = np.percentile(np.concatenate([benign_errors, attack_errors]), 99) * 1.5
    bins = np.linspace(0, max_val, 100)

    plt.hist(
        np.clip(benign_errors, 0, max_val),
        bins=bins,
        alpha=0.6,
        color="#2ecc71",
        label=f"Benign (N={len(benign_errors):,})",
        density=True,
    )
    plt.hist(
        np.clip(attack_errors, 0, max_val),
        bins=bins,
        alpha=0.6,
        color="#e74c3c",
        label=f"Known Attacks (N={len(attack_errors):,})",
        density=True,
    )

    if heldout_errors is not None and len(heldout_errors) > 0:
        plt.hist(
            np.clip(heldout_errors, 0, max_val),
            bins=bins,
            alpha=0.6,
            color="#9b59b6",
            label=f"Held-Out ({heldout_name}) (N={len(heldout_errors):,})",
            density=True,
        )

    plt.axvline(
        threshold,
        color="#2c3e50",
        linestyle="--",
        linewidth=2,
        label=f"Threshold (95th %ile = {threshold:.4f})",
    )

    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Node Reconstruction Error (Smooth L1 / MSE)", fontsize=12, labelpad=8)
    plt.ylabel("Density", fontsize=12, labelpad=8)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300)
    plt.close()


def calculate_class_weights(
    y_train: np.ndarray,
    num_classes: int = 8,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Computes smoothed inverse-frequency class weights solely from the training set.
    """
    counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    weights = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total / (num_classes * (counts[i] ** (1.0 - smoothing)))
        else:
            weights[i] = 1.0

    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_graphs(
    model: GTAE_IDS,
    graphs: List[Any],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Evaluates model across a list of PyG graph snapshots.

    Returns concatenated predictions, ground truth labels, probabilities, and anomaly scores.
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_rec_errors: List[np.ndarray] = []

    with torch.no_grad():
        for graph in graphs:
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            edge_weight = graph.edge_weight.to(device) if hasattr(graph, "edge_weight") and graph.edge_weight is not None else None

            outputs = model(x, edge_index, edge_weight=edge_weight)

            logits = outputs["classification_logits"]
            probs = outputs["probabilities"].cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            node_errors = outputs["node_reconstruction_error"].cpu().numpy()

            all_preds.append(preds)
            all_probs.append(probs)
            all_rec_errors.append(node_errors)
            if hasattr(graph, "y") and graph.y is not None:
                all_y.append(graph.y.cpu().numpy())

    return {
        "preds": np.concatenate(all_preds) if all_preds else np.array([]),
        "y": np.concatenate(all_y) if all_y else np.array([]),
        "probs": np.concatenate(all_probs) if all_probs else np.array([]),
        "reconstruction_errors": np.concatenate(all_rec_errors) if all_rec_errors else np.array([]),
    }


def train_gtae_pipeline(
    data_dir: Optional[str] = None,
    max_benign_samples: int = 30000,
    max_attack_samples_per_class: int = 5000,
    graph_size: int = 1000,
    k_neighbors: int = 5,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    num_heads: int = 4,
    num_encoder_layers: int = 2,
    encoder_type: str = "transformer",
    training_mode: str = "supervised_hybrid",
    lambda_rec: float = 0.5,
    loss_type: str = "mse",
    epochs: int = 5,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    dropout: float = 0.2,
    anomaly_percentile: float = 95.0,
    holdout_class: Optional[Union[str, int]] = None,
    device_name: Optional[str] = None,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Executes end-to-end GTAE-IDS training, anomaly threshold calibration, and evaluation.
    """
    start_time = time.time()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    print("=" * 78)
    print("       XAI-NIDS: GRAPH TRANSFORMER AUTOENCODER (GTAE-IDS) PIPELINE")
    print("=" * 78)

    # 1. Device Resolution
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware Execution Device : {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"    • GPU Model               : {gpu_name} ({vram_total:.2f} GB VRAM)")

    # 2. Dataset Discovery & Loading
    discovery = discover_raw_files(data_dir)
    print(f"\n[*] Discovered {discovery['found_count']} / {discovery['expected_count']} raw Parquet files in: {discovery['directory']}")
    if discovery["found_count"] == 0:
        raise FileNotFoundError(f"No raw Parquet files found in: {discovery['directory']}")

    print(f"[*] Loading stratified subset from real dataset...")
    print(f"    (Max benign: {max_benign_samples:,}, Max attack per class: {max_attack_samples_per_class:,})")
    df = load_stratified_real_dataset(
        raw_dir=discovery["directory"],
        max_benign_samples=max_benign_samples,
        max_attack_samples_per_class=max_attack_samples_per_class,
        deduplicate=True,
        random_state=random_state,
    )
    print(f"[+] Loaded cleaned dataset    : {len(df):,} flows, {df.shape[1]} columns")

    # 3. Stratified Preprocessing & Splitting
    print("\n[*] Splitting and preprocessing features (70/10/20 train/val/test)...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocessor,
        class_map,
    ) = prepare_splits(
        df,
        label_column="Label",
        mode="multiclass",
        test_size=0.2,
        val_size=0.1,
        scaler_type="robust",
        random_state=random_state,
    )

    num_features = X_train.shape[1]
    num_classes = len(MULTICLASS_LABEL_MAP)
    class_names = [MULTICLASS_INDEX_TO_NAME.get(i, f"Class_{i}") for i in range(num_classes)]

    print(f" • Input feature dimension   : {num_features} numerical flow features")
    print(f" • Training flows            : {len(X_train):,}")
    print(f" • Validation flows          : {len(X_val):,}")
    print(f" • Test flows                : {len(X_test):,}")
    print(f" • Classes ({num_classes})             : {class_names}")

    # Handle Held-Out Class if requested
    holdout_id: Optional[int] = None
    holdout_name_str: Optional[str] = None
    if holdout_class is not None:
        if isinstance(holdout_class, str):
            for k, v in MULTICLASS_LABEL_MAP.items():
                if k.lower() == holdout_class.lower():
                    holdout_id = v
                    holdout_name_str = k
                    break
            if holdout_id is None:
                raise ValueError(f"Holdout class '{holdout_class}' not recognized. Valid: {list(MULTICLASS_LABEL_MAP.keys())}")
        else:
            holdout_id = int(holdout_class)
            holdout_name_str = MULTICLASS_INDEX_TO_NAME.get(holdout_id, f"Class_{holdout_id}")

        print(f"\n[!] ZERO-DAY EXPERIMENT ACTIVE: Holding out class '{holdout_name_str}' (ID={holdout_id}) from training.")

    # 4. Graph Construction (NetworkGraphBuilder)
    print(f"\n[*] Constructing flow-similarity graph snapshots (k={k_neighbors}, max_nodes={graph_size})...")
    builder = NetworkGraphBuilder(
        k_neighbors=k_neighbors,
        similarity_metric="cosine",
        include_self_loops=False,
        graph_size=graph_size,
        random_state=random_state,
    )

    train_graphs = builder.build_batch_graphs(X_train, labels=y_train, batch_size=graph_size, shuffle=True)
    val_graphs = builder.build_batch_graphs(X_val, labels=y_val, batch_size=graph_size, shuffle=False)
    test_graphs = builder.build_batch_graphs(X_test, labels=y_test, batch_size=graph_size, shuffle=False)

    print(f" • Training graph snapshots   : {len(train_graphs)} graphs")
    print(f" • Validation graph snapshots : {len(val_graphs)} graphs")
    print(f" • Test graph snapshots       : {len(test_graphs)} graphs")

    # 5. Class Weights Calculation (solely on training set)
    class_weights = calculate_class_weights(y_train, num_classes=num_classes).to(device)

    # 6. Model Initialization
    print(f"\n[*] Initializing GTAE-IDS Model:")
    print(f"    • Backbone Encoder       : {encoder_type.upper()} ({num_encoder_layers} layers, {num_heads} heads)")
    print(f"    • Hidden / Latent Dims   : {hidden_dim} / {latent_dim}")
    print(f"    • Training Mode          : {training_mode}")
    print(f"    • Loss Lambda (rec)      : {lambda_rec}")

    model = GTAE_IDS(
        in_features=num_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        encoder_type=encoder_type,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        loss_type=loss_type,
        lambda_rec=lambda_rec,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Use Mixed Precision if CUDA is available
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 7. Training Loop
    print(f"\n[*] Starting training loop ({epochs} epochs)...")
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_rec_loss = 0.0
        num_nodes_total = 0

        for g_idx, graph in enumerate(train_graphs):
            x = graph.x.to(device)
            y = graph.y.to(device) if hasattr(graph, "y") and graph.y is not None else None
            edge_index = graph.edge_index.to(device)
            edge_weight = graph.edge_weight.to(device) if hasattr(graph, "edge_weight") and graph.edge_weight is not None else None

            # Classification mask (exclude held-out novel attack if requested)
            cls_mask = None
            if holdout_id is not None and y is not None:
                cls_mask = (y != holdout_id)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(x, edge_index, edge_weight=edge_weight)
                loss_dict = model.compute_loss(
                    outputs=outputs,
                    x=x,
                    y=y,
                    class_weights=class_weights,
                    lambda_rec=lambda_rec,
                    training_mode=training_mode,
                    classification_mask=cls_mask,
                )
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            epoch_total_loss += loss.item() * bs
            epoch_cls_loss += loss_dict["classification_loss"].item() * bs
            epoch_rec_loss += loss_dict["reconstruction_loss"].item() * bs
            num_nodes_total += bs

        scheduler.step()

        train_loss = epoch_total_loss / max(num_nodes_total, 1)
        train_cls = epoch_cls_loss / max(num_nodes_total, 1)
        train_rec = epoch_rec_loss / max(num_nodes_total, 1)

        # Validation Step
        val_eval = evaluate_graphs(model, val_graphs, device)
        val_preds = val_eval["preds"]
        val_y = val_eval["y"]
        val_acc = accuracy_score(val_y, val_preds) if len(val_y) > 0 else 0.0
        val_f1 = f1_score(val_y, val_preds, average="macro", zero_division=0) if len(val_y) > 0 else 0.0

        print(
            f" Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} (Cls: {train_cls:.4f}, Rec: {train_rec:.4f}) | "
            f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}"
        )

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)

    # 8. Anomaly Threshold Calibration from Validation Benign Flows
    print("\n[*] Calibrating Anomaly Detection Threshold on BENIGN validation flows...")
    val_eval = evaluate_graphs(model, val_graphs, device)
    val_y = val_eval["y"]
    val_errors = val_eval["reconstruction_errors"]

    benign_val_mask = (val_y == 0)
    benign_val_errors = val_errors[benign_val_mask]

    if len(benign_val_errors) > 0:
        anomaly_threshold = float(np.percentile(benign_val_errors, anomaly_percentile))
    else:
        anomaly_threshold = float(np.percentile(val_errors, anomaly_percentile))

    print(f"[+] Calibrated Anomaly Threshold ({anomaly_percentile}th %ile): {anomaly_threshold:.6f}")

    # 9. Evaluation on Unseen Test Graphs
    print("\n[*] Evaluating GTAE-IDS on unseen Test graph snapshots...")
    test_eval = evaluate_graphs(model, test_graphs, device)
    test_y = test_eval["y"]
    test_preds = test_eval["preds"]
    test_rec_errors = test_eval["reconstruction_errors"]

    # Known classification evaluation (exclude heldout from supervised metric if heldout active)
    eval_mask = (test_y != holdout_id) if holdout_id is not None else np.ones_like(test_y, dtype=bool)

    y_eval = test_y[eval_mask]
    preds_eval = test_preds[eval_mask]

    present_classes = np.unique(np.concatenate([y_eval, preds_eval]))
    eval_class_names = [MULTICLASS_INDEX_TO_NAME.get(i, f"Class_{i}") for i in present_classes]

    acc = float(accuracy_score(y_eval, preds_eval))
    prec_macro = float(precision_score(y_eval, preds_eval, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_eval, preds_eval, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_eval, preds_eval, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_eval, preds_eval, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_eval, preds_eval, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_eval, preds_eval, average="weighted", zero_division=0))

    cm = confusion_matrix(y_eval, preds_eval, labels=present_classes).tolist()
    cls_rep = classification_report(y_eval, preds_eval, target_names=eval_class_names, zero_division=0)

    # Per-class metrics
    p_per, r_per, f_per, s_per = precision_recall_fscore_support(
        y_eval, preds_eval, labels=present_classes, zero_division=0
    )
    per_class_metrics = {}
    for idx, c_id in enumerate(present_classes):
        c_name = MULTICLASS_INDEX_TO_NAME.get(c_id, f"Class_{c_id}")
        per_class_metrics[c_name] = {
            "precision": float(p_per[idx]),
            "recall": float(r_per[idx]),
            "f1-score": float(f_per[idx]),
            "support": int(s_per[idx]),
        }

    # 10. Anomaly Detection Performance Evaluation
    benign_test_mask = (test_y == 0)
    known_attack_test_mask = (test_y != 0) & (test_y != holdout_id) if holdout_id is not None else (test_y != 0)

    benign_test_errors = test_rec_errors[benign_test_mask]
    known_attack_test_errors = test_rec_errors[known_attack_test_mask]

    # Benign False Positive Rate (FPR)
    benign_fp_count = int(np.sum(benign_test_errors > anomaly_threshold))
    benign_fpr = float(benign_fp_count / max(len(benign_test_errors), 1))

    # Known Attack Anomaly Detection Rate (Recall / TPR)
    attack_detected_count = int(np.sum(known_attack_test_errors > anomaly_threshold))
    attack_anomaly_rate = float(attack_detected_count / max(len(known_attack_test_errors), 1))

    # Held-Out Novel Attack Anomaly Detection Rate
    heldout_test_errors = np.array([])
    heldout_anomaly_rate = None
    if holdout_id is not None:
        heldout_mask = (test_y == holdout_id)
        heldout_test_errors = test_rec_errors[heldout_mask]
        if len(heldout_test_errors) > 0:
            heldout_detected = int(np.sum(heldout_test_errors > anomaly_threshold))
            heldout_anomaly_rate = float(heldout_detected / len(heldout_test_errors))
            print(f"[!] Held-Out Attack ({holdout_name_str}) Detection Rate : {heldout_anomaly_rate:.2%} ({heldout_detected}/{len(heldout_test_errors)})")

    # Reconstruction Error Statistics per class
    error_stats_by_class = {}
    for c_id in np.unique(test_y):
        c_name = MULTICLASS_INDEX_TO_NAME.get(c_id, f"Class_{c_id}")
        errs = test_rec_errors[test_y == c_id]
        error_stats_by_class[c_name] = {
            "mean": float(np.mean(errs)),
            "std": float(np.std(errs)),
            "median": float(np.median(errs)),
            "max": float(np.max(errs)),
            "min": float(np.min(errs)),
            "count": int(len(errs)),
        }

    # 11. Artifact Persistence
    print("\n[*] Persisting trained GTAE-IDS artifacts, metrics, and visualization figures...")
    model_save_path = MODELS_DIR / "gtae_ids.pt"
    config_save_path = MODELS_DIR / "gtae_config.json"
    preprocessor_save_path = MODELS_DIR / "preprocessor.joblib"
    metrics_save_path = METRICS_DIR / "gtae_metrics.json"
    cm_plot_path = FIGURES_DIR / "gtae_confusion_matrix.png"
    dist_plot_path = FIGURES_DIR / "reconstruction_error_distribution.png"

    # Save model weights
    model.save(model_save_path)
    preprocessor.save(preprocessor_save_path)

    # Save model config JSON
    config_dict = {
        "model_architecture": "GTAE_IDS",
        "encoder_type": encoder_type,
        "in_features": num_features,
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_heads": num_heads,
        "num_encoder_layers": num_encoder_layers,
        "dropout": dropout,
        "loss_type": loss_type,
        "lambda_rec": lambda_rec,
        "training_mode": training_mode,
        "graph_size": graph_size,
        "k_neighbors": k_neighbors,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "anomaly_percentile": anomaly_percentile,
        "anomaly_threshold": anomaly_threshold,
        "holdout_class": holdout_name_str,
    }
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    # Save metrics JSON
    metrics_to_export = {
        "dataset": "CIC-IDS2017 (Real)",
        "model_name": "GTAE-IDS (Graph Transformer Autoencoder)",
        "execution_device": str(device),
        "training_time_sec": round(time.time() - start_time, 2),
        "total_flows": len(df),
        "train_flows": len(X_train),
        "val_flows": len(X_val),
        "test_flows": len(X_test),
        "num_features": num_features,
        "classes": eval_class_names,
        "classification_metrics": {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
            "per_class": per_class_metrics,
            "confusion_matrix": cm,
        },
        "anomaly_detection_metrics": {
            "anomaly_threshold": anomaly_threshold,
            "anomaly_percentile": anomaly_percentile,
            "benign_false_positive_rate": benign_fpr,
            "known_attack_detection_rate": attack_anomaly_rate,
            "heldout_attack_class": holdout_name_str,
            "heldout_attack_detection_rate": heldout_anomaly_rate,
            "reconstruction_error_statistics": error_stats_by_class,
        },
        "configuration": config_dict,
    }
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_export, f, indent=4)

    # Save Confusion Matrix figure
    plot_and_save_confusion_matrix(
        cm=cm,
        class_names=eval_class_names,
        output_path=cm_plot_path,
        title=f"GTAE-IDS Normalized Confusion Matrix ({encoder_type.upper()})",
    )

    # Save Reconstruction Error Distribution figure
    plot_and_save_reconstruction_distribution(
        benign_errors=benign_test_errors,
        attack_errors=known_attack_test_errors,
        threshold=anomaly_threshold,
        output_path=dist_plot_path,
        heldout_errors=heldout_test_errors if holdout_id is not None else None,
        heldout_name=holdout_name_str,
        title="GTAE Reconstruction Anomaly Scoring Distribution",
    )

    # 12. Print Summary
    print("\n" + "=" * 78)
    print("                    GTAE-IDS PERFORMANCE SUMMARY")
    print("=" * 78)
    print(f" Overall Accuracy             : {acc:.4f}")
    print(f" Macro Precision              : {prec_macro:.4f}")
    print(f" Macro Recall                 : {rec_macro:.4f}")
    print(f" Macro F1-Score               : {f1_macro:.4f}")
    print(f" Weighted F1-Score            : {f1_weighted:.4f}")
    print("-" * 78)
    print("Anomaly / Novelty Detection:")
    print(f" • Calibrated Anomaly Threshold : {anomaly_threshold:.6f}")
    print(f" • Benign False Positive Rate   : {benign_fpr:.2%}")
    print(f" • Known Attack Detection Rate  : {attack_anomaly_rate:.2%}")
    if holdout_name_str is not None and heldout_anomaly_rate is not None:
        print(f" • Held-Out ({holdout_name_str}) Detection Rate: {heldout_anomaly_rate:.2%}")
    print("-" * 78)
    print("Classification Report:")
    print(cls_rep)
    print("-" * 78)
    print("Saved Artifacts:")
    print(f" • Model Weights       : {model_save_path}")
    print(f" • Model Config        : {config_save_path}")
    print(f" • Metrics JSON        : {metrics_save_path}")
    print(f" • Confusion Matrix    : {cm_plot_path}")
    print(f" • Reconstruction Plot : {dist_plot_path}")
    print("=" * 78)

    return metrics_to_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GTAE-IDS on real CIC-IDS2017 data.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing raw parquet files.")
    parser.add_argument("--max_benign", type=int, default=30000, help="Max benign flows to sample.")
    parser.add_argument("--max_attack", type=int, default=5000, help="Max attack flows per class.")
    parser.add_argument("--graph_size", type=int, default=1000, help="Nodes per graph snapshot.")
    parser.add_argument("--k_neighbors", type=int, default=5, help="Nearest neighbors for similarity graph.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent embedding dimension.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--encoder", type=str, default="transformer", choices=["transformer", "sage", "gcn", "graphconv", "gat"], help="Encoder architecture.")
    parser.add_argument("--mode", type=str, default="supervised_hybrid", choices=["supervised_hybrid", "benign_autoencoder"], help="Training loss mode.")
    parser.add_argument("--lambda_rec", type=float, default=0.5, help="Reconstruction loss multiplier.")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "smooth_l1"], help="Reconstruction loss.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--percentile", type=float, default=95.0, help="Anomaly threshold percentile.")
    parser.add_argument("--holdout_class", type=str, default=None, help="Hold out attack class (e.g. Infiltration).")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()
    train_gtae_pipeline(
        data_dir=args.data_dir,
        max_benign_samples=args.max_benign,
        max_attack_samples_per_class=args.max_attack,
        graph_size=args.graph_size,
        k_neighbors=args.k_neighbors,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=args.heads,
        num_encoder_layers=args.layers,
        encoder_type=args.encoder,
        training_mode=args.mode,
        lambda_rec=args.lambda_rec,
        loss_type=args.loss_type,
        epochs=args.epochs,
        learning_rate=args.lr,
        dropout=args.dropout,
        anomaly_percentile=args.percentile,
        holdout_class=args.holdout_class,
        device_name=args.device,
    )
