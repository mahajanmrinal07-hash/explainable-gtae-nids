import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Set stdout encoding if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

from src.detection.inference import InferenceAPI


def normalize_ground_truth(label):
    if pd.isna(label):
        return "UNKNOWN_LABEL"

    value = str(label).strip()

    if value.lower() == "benign":
        return "BENIGN"

    if value in [
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
        "Heartbleed",
    ]:
        return "DoS"

    if value == "DDoS":
        return "DDoS"

    if value == "PortScan":
        return "PortScan"

    if value in ["FTP-Patator", "SSH-Patator"]:
        return "Brute Force"

    if value == "Bot":
        return "Botnet"

    if value.startswith("Web Attack"):
        return "Web Attack"

    if value == "Infiltration":
        return "Infiltration"

    return "UNKNOWN_LABEL"


def load_attack_dataset(raw_dir: Path, target_classes: list, sample_size: int = 100, random_state: int = 42):
    file_mapping = {
        "Botnet": "Botnet-Friday-no-metadata.parquet",
        "Brute Force": "Bruteforce-Tuesday-no-metadata.parquet",
        "DDoS": "DDoS-Friday-no-metadata.parquet",
        "DoS": "DoS-Wednesday-no-metadata.parquet",
        "Infiltration": "Infiltration-Thursday-no-metadata.parquet",
        "PortScan": "Portscan-Friday-no-metadata.parquet",
        "Web Attack": "WebAttacks-Thursday-no-metadata.parquet",
    }

    class_dfs = []
    sample_counts = {}

    for cls in target_classes:
        filename = file_mapping.get(cls)
        if not filename:
            print(f"Warning: No file mapping for class {cls}")
            continue

        file_path = raw_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        df_full = pd.read_parquet(file_path)
        if "Label" not in df_full.columns:
            raise ValueError(f"No 'Label' column found in {file_path}")

        normalized_labels = df_full["Label"].apply(normalize_ground_truth)
        df_class = df_full[normalized_labels == cls].copy()
        df_class["GroundTruth"] = cls

        if len(df_class) == 0:
            print(f"Warning: No flows found for {cls} in {filename}")
            continue

        if len(df_class) > sample_size:
            df_sampled = df_class.sample(n=sample_size, random_state=random_state)
        else:
            df_sampled = df_class.copy()

        sample_counts[cls] = len(df_sampled)
        class_dfs.append(df_sampled)

    combined_df = pd.concat(class_dfs, ignore_index=True)
    return combined_df, sample_counts


def main():
    print("=" * 100)
    print("XAI-NIDS: Controlled Attack-Side Feature Ablation Analysis")
    print("=" * 100)

    model_path = "models/gtae_ids.pt"
    preprocessor_path = "models/preprocessor.joblib"
    raw_dir = Path("data/raw/cicids2017")

    if not Path(model_path).exists() or not Path(preprocessor_path).exists():
        print("Error: Required model artifacts not found.")
        return

    print("Loading Inference API and Preprocessor...")
    api = InferenceAPI(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        anomaly_threshold=2.0,
        confidence_threshold=0.60
    )

    target_classes = [
        "DoS",
        "DDoS",
        "PortScan",
        "Brute Force",
        "Botnet",
        "Web Attack",
        "Infiltration"
    ]

    print("\nLoading attack datasets from raw CIC-IDS2017 parquet files...")
    df_raw, sample_counts = load_attack_dataset(
        raw_dir=raw_dir,
        target_classes=target_classes,
        sample_size=100,
        random_state=42
    )

    print(f"Total attack flows loaded: {len(df_raw)}")
    print("Samples per attack class:")
    for cls, count in sample_counts.items():
        print(f"  - {cls}: {count}")

    feature_names = api.preprocessor.feature_names_

    # Get training means from preprocessor
    training_means = {}
    scaler = api.preprocessor.scaler_
    if hasattr(scaler, 'mean_'):
        for i, f in enumerate(feature_names):
            training_means[f] = scaler.mean_[i]
    else:
        for f in feature_names:
            if f in df_raw.columns:
                training_means[f] = df_raw[f].mean()

    # Exact feature grouping matching logic from tools/feature_ablation_analysis.py
    tcp_flags = [
        f for f in feature_names
        if 'Flag' in f or f in [
            'URG Flag Count', 'FIN Flag Count', 'ACK Flag Count',
            'SYN Flag Count', 'PSH Flag Count', 'RST Flag Count',
            'CWE Flag Count', 'ECE Flag Count'
        ]
    ]
    packet_rate = [f for f in feature_names if 'Packets/s' in f or 'Packets/s' in f]
    packet_length = [f for f in feature_names if 'Packet Length' in f or 'Segment Size' in f]

    experiments = {
        "1. BASELINE": [],
        "2. PACKET_LENGTH": packet_length,
        "3. COMBINED": list(set(tcp_flags + packet_rate + packet_length))
    }

    ground_truth = df_raw["GroundTruth"].tolist()
    total_samples = len(df_raw)

    overall_summaries = []
    per_class_detection = {cls: {} for cls in target_classes}
    per_class_correct = {cls: {} for cls in target_classes}

    for exp_name, features_to_ablate in experiments.items():
        print(f"\nEvaluating configuration: {exp_name}...")
        df_exp = df_raw.copy()

        # Ablate features by replacing with training mean
        ablated_count = 0
        for f in features_to_ablate:
            if f in df_exp.columns:
                df_exp[f] = training_means.get(f, 0.0)
                ablated_count += 1

        print(f" - Ablated features count: {ablated_count}")

        # Run inference (exclude GroundTruth metadata column if present)
        inference_df = df_exp[[c for c in df_exp.columns if c != "GroundTruth"]]
        results = api.predict(inference_df)

        detected_types = [r["detected_type"] for r in results]
        confidences = [r["classifier_prob"] for r in results]
        anomalies = [r["anomaly_score"] for r in results]

        # Calculate metrics
        attack_detected_count = sum(1 for d in detected_types if d != "BENIGN")
        correct_class_count = sum(1 for d, gt in zip(detected_types, ground_truth) if d == gt)
        benign_misclass_count = sum(1 for d in detected_types if d == "BENIGN")
        novel_count = sum(1 for d in detected_types if d == "UNKNOWN_NOVEL")
        other_attack_count = sum(
            1 for d, gt in zip(detected_types, ground_truth)
            if d != "BENIGN" and d != "UNKNOWN_NOVEL" and d != gt
        )

        attack_detected_pct = (attack_detected_count / total_samples) * 100.0
        correct_class_pct = (correct_class_count / total_samples) * 100.0
        benign_misclass_pct = (benign_misclass_count / total_samples) * 100.0
        novel_pct = (novel_count / total_samples) * 100.0
        mean_conf = np.mean(confidences)
        mean_anom = np.mean(anomalies)

        overall_summaries.append({
            "Experiment": exp_name,
            "Samples": total_samples,
            "Attack Detected %": f"{attack_detected_pct:.2f}%",
            "Correct Class %": f"{correct_class_pct:.2f}%",
            "BENIGN Misclass %": f"{benign_misclass_pct:.2f}%",
            "UNKNOWN_NOVEL %": f"{novel_pct:.2f}%",
            "Other Attack %": f"{(other_attack_count / total_samples) * 100.0:.2f}%",
            "Mean Confidence": f"{mean_conf:.4f}",
            "Mean Anomaly": f"{mean_anom:.4f}"
        })

        # Per-class calculation
        for cls in target_classes:
            cls_indices = [i for i, gt in enumerate(ground_truth) if gt == cls]
            cls_total = len(cls_indices)
            cls_detected = sum(1 for i in cls_indices if detected_types[i] != "BENIGN")
            cls_correct = sum(1 for i in cls_indices if detected_types[i] == cls)

            det_rate = (cls_detected / cls_total) * 100.0 if cls_total > 0 else 0.0
            cor_rate = (cls_correct / cls_total) * 100.0 if cls_total > 0 else 0.0

            per_class_detection[cls][exp_name] = f"{det_rate:.2f}% ({cls_detected}/{cls_total})"
            per_class_correct[cls][exp_name] = f"{cor_rate:.2f}% ({cls_correct}/{cls_total})"

    # Format Tables
    print("\n" + "=" * 120)
    print("OVERALL SUMMARY TABLE")
    print("=" * 120)
    summary_df = pd.DataFrame(overall_summaries)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("TABLE 1: ATTACK DETECTION RATE (detected_type != 'BENIGN')")
    print("=" * 120)
    t1_rows = []
    for cls in target_classes:
        t1_rows.append({
            "Class": cls,
            "Samples": sample_counts.get(cls, 0),
            "Baseline Detection": per_class_detection[cls]["1. BASELINE"],
            "Packet-Length Detection": per_class_detection[cls]["2. PACKET_LENGTH"],
            "Combined Detection": per_class_detection[cls]["3. COMBINED"]
        })
    t1_df = pd.DataFrame(t1_rows)
    print(t1_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("TABLE 2: CORRECT ATTACK CLASS IDENTIFICATION RATE (detected_type == GroundTruth)")
    print("=" * 120)
    t2_rows = []
    for cls in target_classes:
        t2_rows.append({
            "Class": cls,
            "Samples": sample_counts.get(cls, 0),
            "Baseline Correct Class": per_class_correct[cls]["1. BASELINE"],
            "Packet-Length Correct Class": per_class_correct[cls]["2. PACKET_LENGTH"],
            "Combined Correct Class": per_class_correct[cls]["3. COMBINED"]
        })
    t2_df = pd.DataFrame(t2_rows)
    print(t2_df.to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    main()
