import os
import warnings
import pandas as pd
import numpy as np

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

from src.detection.inference import InferenceAPI

def main():
    print("Initializing Feature Ablation Analysis...")
    
    # Paths
    model_path = "models/gtae_ids.pt"
    preprocessor_path = "models/preprocessor.joblib"
    dataset_path = "data/raw/cicids2017/Benign-Monday-no-metadata.parquet"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
        
    api = InferenceAPI(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        anomaly_threshold=2.0,
        confidence_threshold=0.60
    )
    
    print("Loading 100-flow Benign-Monday random sample...")
    df_raw = pd.read_parquet(dataset_path).sample(n=100, random_state=42).reset_index(drop=True)
    
    # Get feature names from preprocessor
    feature_names = api.preprocessor.feature_names_
    
    # Attempt to get training means from preprocessor
    # If the scaler is StandardScaler, it has .mean_
    training_means = {}
    scaler = api.preprocessor.scaler_
    if hasattr(scaler, 'mean_'):
        for i, f in enumerate(feature_names):
            training_means[f] = scaler.mean_[i]
    else:
        # Fallback to the mean of the current 100 sample (benign)
        for f in feature_names:
            if f in df_raw.columns:
                training_means[f] = df_raw[f].mean()
                
    # Define Feature Groups
    tcp_flags = [f for f in feature_names if 'Flag' in f or f in ['URG Flag Count', 'FIN Flag Count', 'ACK Flag Count', 'SYN Flag Count', 'PSH Flag Count', 'RST Flag Count', 'CWE Flag Count', 'ECE Flag Count']]
    packet_rate = [f for f in feature_names if 'Packets/s' in f or 'Packets/s' in f]
    packet_length = [f for f in feature_names if 'Packet Length' in f or 'Segment Size' in f]
    
    experiments = {
        "1. BASELINE": [],
        "2. TCP_FLAGS": tcp_flags,
        "3. PACKET_RATE": packet_rate,
        "4. PACKET_LENGTH": packet_length,
        "5. COMBINED": list(set(tcp_flags + packet_rate + packet_length))
    }
    
    results_summary = []
    
    for exp_name, features_to_ablate in experiments.items():
        print(f"\nRunning {exp_name}...")
        df_exp = df_raw.copy()
        
        # Ablate features by replacing with training mean
        ablated_count = 0
        for f in features_to_ablate:
            if f in df_exp.columns:
                df_exp[f] = training_means.get(f, 0.0)
                ablated_count += 1
                
        print(f" - Ablated {ablated_count} features")
        
        # Run inference
        results = api.predict(df_exp)
        
        # Calculate metrics
        total = len(results)
        pred_benign = sum(1 for r in results if r['detected_type'] == 'BENIGN')
        pred_attack = total - pred_benign
        
        # Breakdowns
        counts = {'Botnet': 0, 'Web Attack': 0, 'Brute Force': 0, 'DoS': 0, 'DDoS': 0, 'PortScan': 0, 'Infiltration': 0}
        confidences = []
        anomalies = []
        
        for r in results:
            dt = r['detected_type']
            confidences.append(r['classifier_prob'])
            anomalies.append(r['anomaly_score'])
            if dt != 'BENIGN' and dt != 'UNKNOWN_NOVEL':
                # Map exact or substring matches
                for k in counts.keys():
                    if k in dt:
                        counts[k] += 1
                        break
        
        mean_conf = sum(confidences)/total if total else 0
        mean_anom = sum(anomalies)/total if total else 0
        
        fp_rate = pred_attack / total
        
        results_summary.append({
            'Experiment': exp_name,
            'Total': total,
            'BENIGN': pred_benign,
            'ATTACK_FP': pred_attack,
            'Botnet': counts['Botnet'],
            'Web_Attack': counts['Web Attack'],
            'Brute_Force': counts['Brute Force'],
            'DoS': counts['DoS'],
            'DDoS': counts['DDoS'],
            'PortScan': counts['PortScan'],
            'Infil': counts['Infiltration'],
            'Mean_Conf': mean_conf,
            'Mean_Anom': mean_anom,
            'FPR': fp_rate
        })
        
    print("\n" + "="*140)
    print("Ablation Analysis Results")
    print("="*140)
    df_res = pd.DataFrame(results_summary)
    
    # Format and print table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*140)
    print("Note: Benign-Monday contains entirely BENIGN ground truth. All predicted attacks are False Positives.")

if __name__ == "__main__":
    main()
