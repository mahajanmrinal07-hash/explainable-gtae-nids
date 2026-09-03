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

        if len(df_class) > sample_size:
            df_sampled = df_class.sample(n=sample_size, random_state=random_state)
        else:
            df_sampled = df_class.copy()

        sample_counts[cls] = len(df_sampled)
        class_dfs.append(df_sampled)

    combined_df = pd.concat(class_dfs, ignore_index=True)
    return combined_df, sample_counts


def apply_policy(policy_name: str, pred_result: dict) -> str:
    """
    Applies counterfactual decision policy to an inference result item.
    Returns the resolved detected_type string.
    """
    orig_type = pred_result["detected_type"]
    prob = pred_result["classifier_prob"]
    err = pred_result["anomaly_score"]
    anomaly_threshold = 2.0

    if policy_name == "POLICY A (Current)":
        return orig_type

    elif policy_name == "POLICY B (Low-Conf Gate < 0.60)":
        if orig_type != "BENIGN" and prob < 0.60 and err <= anomaly_threshold:
            return "BENIGN"
        return orig_type

    elif policy_name == "POLICY C (Conservative Gate < 0.70)":
        if orig_type != "BENIGN" and prob < 0.70 and err <= anomaly_threshold:
            return "BENIGN"
        return orig_type

    elif policy_name == "POLICY D (Strict Gate < 0.80)":
        if orig_type != "BENIGN" and prob < 0.80 and err <= anomaly_threshold:
            return "BENIGN"
        return orig_type

    elif policy_name == "POLICY E (Conf + Anomaly Validation)":
        if orig_type != "BENIGN":
            if prob < 0.60:
                if err <= anomaly_threshold:
                    return "BENIGN"
                else:
                    return "UNKNOWN_NOVEL"
            else:
                return orig_type
        else:
            # If originally BENIGN
            return orig_type

    return orig_type


def main():
    print("=" * 110)
    print("XAI-NIDS: Controlled Decision-Policy Experiment")
    print("=" * 110)

    model_path = "models/gtae_ids.pt"
    preprocessor_path = "models/preprocessor.joblib"
    raw_dir = Path("data/raw/cicids2017")
    benign_path = raw_dir / "Benign-Monday-no-metadata.parquet"

    if not Path(model_path).exists() or not Path(preprocessor_path).exists():
        print("Error: Required model artifacts not found.")
        return

    print("Loading Inference API...")
    api = InferenceAPI(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        anomaly_threshold=2.0,
        confidence_threshold=0.60
    )

    # 1. Load Benign Dataset
    print("\nLoading Benign-Monday random sample (n=100, random_state=42)...")
    df_benign_raw = pd.read_parquet(benign_path).sample(n=100, random_state=42).reset_index(drop=True)
    df_benign_raw["GroundTruth"] = "BENIGN"

    # 2. Load Attack Dataset
    target_classes = [
        "DoS",
        "DDoS",
        "PortScan",
        "Brute Force",
        "Botnet",
        "Web Attack",
        "Infiltration"
    ]
    print("Loading genuine attack samples (up to 100 per class, random_state=42)...")
    df_attack_raw, attack_counts = load_attack_dataset(
        raw_dir=raw_dir,
        target_classes=target_classes,
        sample_size=100,
        random_state=42
    )

    print(f"Total benign samples : {len(df_benign_raw)}")
    print(f"Total attack samples : {len(df_attack_raw)}")
    print("Attack samples per class:")
    for cls, cnt in attack_counts.items():
        print(f"  - {cls}: {cnt}")

    # Run raw inference once
    print("\nRunning inference on Benign dataset...")
    inference_benign = df_benign_raw[[c for c in df_benign_raw.columns if c != "GroundTruth"]]
    benign_results = api.predict(inference_benign)

    print("Running inference on Attack dataset...")
    inference_attack = df_attack_raw[[c for c in df_attack_raw.columns if c != "GroundTruth"]]
    attack_results = api.predict(inference_attack)

    policies = [
        "POLICY A (Current)",
        "POLICY B (Low-Conf Gate < 0.60)",
        "POLICY C (Conservative Gate < 0.70)",
        "POLICY D (Strict Gate < 0.80)",
        "POLICY E (Conf + Anomaly Validation)"
    ]

    # Evaluate Benign Metrics for each policy
    benign_summaries = []
    benign_gt = df_benign_raw["GroundTruth"].tolist()
    total_benign = len(benign_gt)

    for pol in policies:
        resolved_types = [apply_policy(pol, r) for r in benign_results]
        pred_benign_cnt = sum(1 for d in resolved_types if d == "BENIGN")
        fp_cnt = total_benign - pred_benign_cnt
        fpr = (fp_cnt / total_benign) * 100.0

        botnet_fp = sum(1 for d in resolved_types if d == "Botnet")
        web_fp = sum(1 for d in resolved_types if d == "Web Attack")
        brute_fp = sum(1 for d in resolved_types if d == "Brute Force")
        other_fp = sum(1 for d in resolved_types if d not in ["BENIGN", "Botnet", "Web Attack", "Brute Force"])

        converted_cnt = sum(
            1 for orig, res in zip(benign_results, resolved_types)
            if orig["detected_type"] != "BENIGN" and res == "BENIGN"
        )

        remaining_fp_confs = [
            r["classifier_prob"] for r, res in zip(benign_results, resolved_types)
            if res != "BENIGN"
        ]
        mean_fp_conf = np.mean(remaining_fp_confs) if remaining_fp_confs else 0.0

        benign_summaries.append({
            "Policy": pol,
            "Total": total_benign,
            "Pred BENIGN": pred_benign_cnt,
            "FPs": fp_cnt,
            "Benign FPR": f"{fpr:.2f}%",
            "Botnet FP": botnet_fp,
            "Web FP": web_fp,
            "Brute FP": brute_fp,
            "Other FP": other_fp,
            "Converted to BENIGN": converted_cnt,
            "Mean FP Conf": f"{mean_fp_conf:.4f}" if remaining_fp_confs else "N/A"
        })

    # Evaluate Attack Metrics for each policy
    attack_gt = df_attack_raw["GroundTruth"].tolist()
    total_attack = len(attack_gt)
    attack_summaries = []
    per_class_correct = {cls: {} for cls in target_classes}

    for pol in policies:
        resolved_types = [apply_policy(pol, r) for r in attack_results]

        attack_detected = sum(1 for d in resolved_types if d != "BENIGN")
        correct_class = sum(1 for d, gt in zip(resolved_types, attack_gt) if d == gt)
        attack_fn = sum(1 for d in resolved_types if d == "BENIGN")
        novel_cnt = sum(1 for d in resolved_types if d == "UNKNOWN_NOVEL")
        wrong_attack = sum(
            1 for d, gt in zip(resolved_types, attack_gt)
            if d != "BENIGN" and d != "UNKNOWN_NOVEL" and d != gt
        )

        det_rate = (attack_detected / total_attack) * 100.0
        cor_rate = (correct_class / total_attack) * 100.0
        fn_rate = (attack_fn / total_attack) * 100.0
        novel_rate = (novel_cnt / total_attack) * 100.0
        wrong_rate = (wrong_attack / total_attack) * 100.0

        # Match with benign FPR for main summary table
        b_fpr = next(b["Benign FPR"] for b in benign_summaries if b["Policy"] == pol)

        attack_summaries.append({
            "Policy": pol,
            "Benign FPR": b_fpr,
            "Attack Detection %": f"{det_rate:.2f}%",
            "Correct Attack Class %": f"{cor_rate:.2f}%",
            "Attack FN %": f"{fn_rate:.2f}%",
            "Wrong Attack Class %": f"{wrong_rate:.2f}%",
            "UNKNOWN_NOVEL %": f"{novel_rate:.2f}%"
        })

        # Per-class calculation
        for cls in target_classes:
            cls_indices = [i for i, gt in enumerate(attack_gt) if gt == cls]
            cls_total = len(cls_indices)
            cls_cor = sum(1 for i in cls_indices if resolved_types[i] == cls)
            cls_cor_rate = (cls_cor / cls_total) * 100.0 if cls_total > 0 else 0.0
            per_class_correct[cls][pol] = f"{cls_cor_rate:.2f}% ({cls_cor}/{cls_total})"

    # Print Summary Tables
    print("\n" + "=" * 130)
    print("POLICY COMPARISON SUMMARY TABLE")
    print("=" * 130)
    df_main_summary = pd.DataFrame(attack_summaries)
    print(df_main_summary.to_string(index=False))

    print("\n" + "=" * 130)
    print("BENIGN-SIDE FALSE POSITIVE BREAKDOWN TABLE")
    print("=" * 130)
    df_benign_table = pd.DataFrame(benign_summaries)
    print(df_benign_table.to_string(index=False))

    print("\n" + "=" * 130)
    print("PER-CLASS CORRECT ATTACK-CLASS RATE TABLE")
    print("=" * 130)
    per_class_rows = []
    for cls in target_classes:
        per_class_rows.append({
            "Class": cls,
            "Samples": attack_counts.get(cls, 0),
            "Current Correct %": per_class_correct[cls]["POLICY A (Current)"],
            "Policy B (<0.60)": per_class_correct[cls]["POLICY B (Low-Conf Gate < 0.60)"],
            "Policy C (<0.70)": per_class_correct[cls]["POLICY C (Conservative Gate < 0.70)"],
            "Policy D (<0.80)": per_class_correct[cls]["POLICY D (Strict Gate < 0.80)"],
            "Policy E (Anomaly Val)": per_class_correct[cls]["POLICY E (Conf + Anomaly Validation)"]
        })
    df_per_class = pd.DataFrame(per_class_rows)
    print(df_per_class.to_string(index=False))

    # Focused Analysis of the 8 Known High-Confidence Botnet False Positives
    print("\n" + "=" * 130)
    print("FOCUSED ANALYSIS: 8 HIGH-CONFIDENCE BENIGN -> BOTNET FALSE POSITIVES")
    print("=" * 130)
    high_conf_indices = [
        i for i, r in enumerate(benign_results)
        if r["detected_type"] == "Botnet" and r["classifier_prob"] >= 0.60
    ]

    print(f"Total identified high-confidence Botnet FPs in Benign-Monday sample: {len(high_conf_indices)}")
    for idx in high_conf_indices:
        r = benign_results[idx]
        print(f" - Flow #{idx:02d}: Prob={r['classifier_prob']:.4f}, Anomaly={r['anomaly_score']:.4f} | "
              f"Pol A: {apply_policy(policies[0], r)} | "
              f"Pol B: {apply_policy(policies[1], r)} | "
              f"Pol C: {apply_policy(policies[2], r)} | "
              f"Pol D: {apply_policy(policies[3], r)} | "
              f"Pol E: {apply_policy(policies[4], r)}")

    print("\nHigh-Confidence Botnet FPs remaining by policy:")
    for pol in policies:
        remaining = sum(
            1 for idx in high_conf_indices
            if apply_policy(pol, benign_results[idx]) != "BENIGN"
        )
        converted = len(high_conf_indices) - remaining
        print(f"  * {pol}: {remaining}/8 remaining ({converted} converted to BENIGN)")
    print("=" * 130)


if __name__ == "__main__":
    main()
