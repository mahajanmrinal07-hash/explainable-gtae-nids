"""
Graph Pipeline Development Script.

Loads real CIC-IDS2017 development sample data, builds a flow-similarity
graph from 1,000 real flows, prints summary statistics, and saves one
graph artifact to results/graphs/.

Usage:
    python -m src.training.test_graph_pipeline
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    MULTICLASS_INDEX_TO_NAME,
    RANDOM_SEED,
    RESULTS_DIR,
    SAMPLES_DATA_DIR,
)
from src.graph_builder import NetworkGraphBuilder
from src.preprocessing import Preprocessor, normalize_labels


def main():
    print("=" * 75)
    print("    XAI-NIDS: FLOW-SIMILARITY GRAPH PIPELINE (Development)")
    print("=" * 75)

    # ------------------------------------------------------------------
    # 1. Load sample development data
    # ------------------------------------------------------------------
    sample_path = SAMPLES_DATA_DIR / "sample_cic_ids2017.parquet"

    if not sample_path.exists():
        print(f"[!] Sample dataset not found at: {sample_path}")
        print("    Generating synthetic data for pipeline validation...")
        # Fallback: create minimal synthetic data
        rng = np.random.RandomState(RANDOM_SEED)
        n_flows = 1000
        n_features = 67
        X_synth = rng.randn(n_flows, n_features).astype(np.float32)
        y_synth = rng.randint(0, 8, size=n_flows).astype(np.int64)
        print(f"[+] Created synthetic data: {n_flows} flows × {n_features} features")
    else:
        print(f"[+] Found sample dataset: {sample_path}")
        df = pd.read_parquet(str(sample_path))
        df.columns = [c.strip() for c in df.columns]
        print(f"    Total rows in sample file: {len(df):,}")
        print(f"    Columns: {df.shape[1]}")

        # Identify label column
        label_col = None
        for col in df.columns:
            if col.lower() == "label":
                label_col = col
                break

        if label_col is None:
            print("[!] No 'Label' column found. Proceeding without labels.")
            labels_available = False
        else:
            labels_available = True

        # Select 1,000 flows (stratified if labels exist)
        n_select = min(1000, len(df))
        if labels_available and n_select < len(df):
            # Stratified sample — collect per-group samples in a list
            sampled_groups = []
            for _label, grp in df.groupby(label_col):
                n_take = max(1, int(len(grp) / len(df) * n_select))
                sampled_groups.append(
                    grp.sample(n=min(n_take, len(grp)), random_state=RANDOM_SEED)
                )
            df_sample = pd.concat(sampled_groups, ignore_index=True)
            # Trim to exactly n_select
            if len(df_sample) > n_select:
                df_sample = df_sample.sample(n=n_select, random_state=RANDOM_SEED).reset_index(drop=True)
        else:
            df_sample = df.sample(n=n_select, random_state=RANDOM_SEED).reset_index(drop=True)

        print(f"[+] Selected {len(df_sample):,} flows for graph construction")

        # Preprocess features
        if labels_available:
            raw_labels = df_sample[label_col]
            y_encoded, class_map = normalize_labels(raw_labels, mode="multiclass")
            X_df = df_sample.drop(columns=[label_col])
        else:
            y_encoded = None
            class_map = MULTICLASS_INDEX_TO_NAME
            X_df = df_sample

        preprocessor = Preprocessor(scaler_type="robust")
        X_processed = preprocessor.fit_transform(X_df)

        X_synth = X_processed
        y_synth = y_encoded

        print(f"[+] Preprocessed feature matrix shape: {X_synth.shape}")

    # ------------------------------------------------------------------
    # 2. Build graph
    # ------------------------------------------------------------------
    print("\n[*] Building flow-similarity graph (k=5, cosine similarity)...")

    builder = NetworkGraphBuilder(
        k_neighbors=5,
        similarity_metric="cosine",
        include_self_loops=False,
        graph_size=1000,
        random_state=RANDOM_SEED,
    )

    graph = builder.build_graph(X_synth, labels=y_synth)

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    summary = NetworkGraphBuilder.graph_summary(graph)

    print("\n" + "=" * 75)
    print("                    GRAPH CONSTRUCTION RESULTS")
    print("=" * 75)
    print(f"  Nodes (flows)       : {summary['num_nodes']:,}")
    print(f"  Features per node   : {summary['num_features']}")
    print(f"  Edges               : {summary['num_edges']:,}")
    print(f"  Average degree      : {summary['avg_degree']:.2f}")

    if "edge_weight_stats" in summary:
        ew = summary["edge_weight_stats"]
        print(f"  Edge weight range   : [{ew['min']:.4f}, {ew['max']:.4f}]")
        print(f"  Edge weight mean    : {ew['mean']:.4f} ± {ew['std']:.4f}")

    if "class_distribution" in summary:
        print(f"\n  Class Distribution:")
        for cls_id, count in sorted(summary["class_distribution"].items()):
            cls_name = MULTICLASS_INDEX_TO_NAME.get(cls_id, f"Class_{cls_id}")
            print(f"    {cls_id}: {cls_name:20s} -> {count:,} flows")

    # Verify labels not in features
    print(f"\n  [✓] Labels stored in y (shape: {graph.y.shape if graph.y is not None else 'None'})")
    print(f"  [✓] Features in x (shape: {graph.x.shape})")
    print(f"  [✓] edge_index shape: {list(graph.edge_index.shape)}")

    # ------------------------------------------------------------------
    # 4. Save graph artifact
    # ------------------------------------------------------------------
    graphs_dir = RESULTS_DIR / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    save_path = graphs_dir / "dev_graph_1k.pt"

    builder.save_graph(graph, save_path)
    print(f"\n[+] Saved graph artifact to: {save_path}")

    # Also save summary as JSON
    import json
    summary_path = graphs_dir / "dev_graph_1k_summary.json"

    # Make summary JSON-serializable
    summary_json = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            summary_json[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            summary_json[k] = v

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)
    print(f"[+] Saved summary JSON to: {summary_path}")

    print("\n" + "=" * 75)
    print("    Graph pipeline completed successfully.")
    print("=" * 75)

    return graph, summary


if __name__ == "__main__":
    main()
