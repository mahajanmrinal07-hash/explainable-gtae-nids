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


def apply_threshold_gate(pred_result: dict, threshold: float, anomaly_cutoff: float = 2.0) -> str:
    orig_type = pred_result["detected_type"]
    prob = pred_result["classifier_prob"]
    err = pred_result["anomaly_score"]

    if orig_type != "BENIGN" and prob < threshold and err <= anomaly_cutoff:
        return "BENIGN"
    return orig_type


def main():
    print("=" * 110)
    print("XAI-NIDS: Final Read-Only Confidence Threshold Calibration Experiment")
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

    # Run inference once per dataset
    print("\nRunning inference on Benign dataset...")
    inference_benign = df_benign_raw[[c for c in df_benign_raw.columns if c != "GroundTruth"]]
    benign_results = api.predict(inference_benign)

    print("Running inference on Attack dataset...")
    inference_attack = df_attack_raw[[c for c in df_attack_raw.columns if c != "GroundTruth"]]
    attack_results = api.predict(inference_attack)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    benign_gt = df_benign_raw["GroundTruth"].tolist()
    total_benign = len(benign_gt)
    attack_gt = df_attack_raw["GroundTruth"].tolist()
    total_attack = len(attack_gt)

    main_table_rows = []
    benign_breakdown_rows = []
    per_class_table_rows = []

    for t in thresholds:
        # Benign evaluation
        b_resolved = [apply_threshold_gate(r, threshold=t) for r in benign_results]
        b_pred_benign = sum(1 for d in b_resolved if d == "BENIGN")
        b_fps = total_benign - b_pred_benign
        b_fpr = (b_fps / total_benign) * 100.0

        botnet_fp = sum(1 for d in b_resolved if d == "Botnet")
        web_fp = sum(1 for d in b_resolved if d == "Web Attack")
        brute_fp = sum(1 for d in b_resolved if d == "Brute Force")
        other_fp = sum(1 for d in b_resolved if d not in ["BENIGN", "Botnet", "Web Attack", "Brute Force"])
        converted_cnt = sum(
            1 for orig, res in zip(benign_results, b_resolved)
            if orig["detected_type"] != "BENIGN" and res == "BENIGN"
        )
        remaining_fp_confs = [
            r["classifier_prob"] for r, res in zip(benign_results, b_resolved)
            if res != "BENIGN"
        ]
        mean_fp_conf = np.mean(remaining_fp_confs) if remaining_fp_confs else 0.0

        benign_breakdown_rows.append({
            "Threshold": f"{t:.2f}",
            "Benign FPR": f"{b_fpr:.2f}%",
            "Total FPs": b_fps,
            "Converted to BENIGN": converted_cnt,
            "Botnet FP": botnet_fp,
            "Web FP": web_fp,
            "Brute FP": brute_fp,
            "Other FP": other_fp,
            "Mean FP Conf": f"{mean_fp_conf:.4f}" if remaining_fp_confs else "N/A"
        })

        # Attack evaluation
        a_resolved = [apply_threshold_gate(r, threshold=t) for r in attack_results]
        attack_detected = sum(1 for d in a_resolved if d != "BENIGN")
        correct_class = sum(1 for d, gt in zip(a_resolved, attack_gt) if d == gt)
        attack_fn = sum(1 for d in a_resolved if d == "BENIGN")
        wrong_attack = sum(
            1 for d, gt in zip(a_resolved, attack_gt)
            if d != "BENIGN" and d != "UNKNOWN_NOVEL" and d != gt
        )
        novel_cnt = sum(1 for d in a_resolved if d == "UNKNOWN_NOVEL")

        det_rate = (attack_detected / total_attack) * 100.0
        cor_rate = (correct_class / total_attack) * 100.0
        fn_rate = (attack_fn / total_attack) * 100.0
        wrong_rate = (wrong_attack / total_attack) * 100.0

        main_table_rows.append({
            "Threshold": f"{t:.2f}",
            "Benign FPR": f"{b_fpr:.2f}%",
            "Attack Detection %": f"{det_rate:.2f}%",
            "Correct Attack Class %": f"{cor_rate:.2f}%",
            "Attack FN %": f"{fn_rate:.2f}%",
            "Wrong Attack Class %": f"{wrong_rate:.2f}%"
        })

        # Per-class correct class rates
        p_row = {"Threshold": f"{t:.2f}"}
        for cls in target_classes:
            cls_indices = [i for i, gt in enumerate(attack_gt) if gt == cls]
            cls_total = len(cls_indices)
            cls_cor = sum(1 for i in cls_indices if a_resolved[i] == cls)
            cls_rate = (cls_cor / cls_total) * 100.0 if cls_total > 0 else 0.0
            p_row[cls] = f"{cls_rate:.2f}%"
        per_class_table_rows.append(p_row)

    # Display Tables
    print("\n" + "=" * 120)
    print("MAIN TABLE: CONFIDENCE THRESHOLD CALIBRATION SUMMARY")
    print("=" * 120)
    df_main = pd.DataFrame(main_table_rows)
    print(df_main.to_string(index=False))

    print("\n" + "=" * 120)
    print("BENIGN FALSE POSITIVE BREAKDOWN TABLE")
    print("=" * 120)
    df_benign = pd.DataFrame(benign_breakdown_rows)
    print(df_benign.to_string(index=False))

    print("\n" + "=" * 120)
    print("PER-CLASS CORRECT ATTACK-CLASS RATE TABLE (Recall per class)")
    print("=" * 120)
    df_per_class = pd.DataFrame(per_class_table_rows)
    print(df_per_class.to_string(index=False))

    # Identify Key Decision Thresholds
    print("\n" + "=" * 120)
    print("CRITICAL THRESHOLD BOUNDARY FINDINGS")
    print("=" * 120)

    # 1. First threshold where Botnet detection degrades
    botnet_indices = [i for i, gt in enumerate(attack_gt) if gt == "Botnet"]
    bot_baseline_cor = sum(1 for i in botnet_indices if attack_results[i]["detected_type"] == "Botnet")
    first_bot_deg = None
    for t in thresholds:
        a_res = [apply_threshold_gate(r, threshold=t) for r in attack_results]
        bot_cor = sum(1 for i in botnet_indices if a_res[i] == "Botnet")
        if bot_cor < bot_baseline_cor:
            first_bot_deg = (t, bot_cor, len(botnet_indices))
            break
    if first_bot_deg:
        print(f"- First threshold where Botnet correct classification degrades: {first_bot_deg[0]:.2f} "
              f"(drops to {first_bot_deg[1]}/{first_bot_deg[2]} = {(first_bot_deg[1]/first_bot_deg[2])*100:.1f}%)")
    else:
        print("- Botnet classification does not degrade across tested range.")

    # 2. First threshold where overall attack detection falls below 95%
    first_det_below_95 = None
    for row, t in zip(main_table_rows, thresholds):
        det_val = float(row["Attack Detection %"].replace("%", ""))
        if det_val < 95.0:
            first_det_below_95 = (t, det_val)
            break
    if first_det_below_95:
        print(f"- First threshold where Overall Attack Detection falls below 95%: {first_det_below_95[0]:.2f} "
              f"({first_det_below_95[1]:.2f}%)")

    # 3. First threshold where correct attack class falls below 94%
    first_cor_below_94 = None
    for row, t in zip(main_table_rows, thresholds):
        cor_val = float(row["Correct Attack Class %"].replace("%", ""))
        if cor_val < 94.0:
            first_cor_below_94 = (t, cor_val)
            break
    if first_cor_below_94:
        print(f"- First threshold where Correct Attack-Class Rate falls below 94%: {first_cor_below_94[0]:.2f} "
              f"({first_cor_below_94[1]:.2f}%)")

    print("=" * 120)


if __name__ == "__main__":
    main()
