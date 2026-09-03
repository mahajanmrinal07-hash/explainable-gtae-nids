"""
XAI-NIDS Interactive Security Dashboard

Phase 6:
- Real CIC-IDS2017 inference
- GTAE-IDS classification
- Reconstruction-based anomaly detection
- Threat risk scoring
- XAI feature importance
- Graph-neighborhood explanation
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import GPU_CONFIG, MULTICLASS_INDEX_TO_NAME
from src.detection.inference import InferenceAPI
from src.explainability.explainer import IntrusionExplainer
from dashboard.evaluation import evaluate_predictions


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "gtae_ids.pt"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.joblib"
DATA_PATH = ROOT / "data" / "raw" / "cicids2017"

ZERO_DAY_CONFIG_PATH = ROOT / "results" / "zero_day_experiment" / "gtae_config_zero_day.json"
ZERO_DAY_METRICS_PATH = ROOT / "results" / "zero_day_experiment" / "gtae_metrics_zero_day.json"


DATASETS = {
    "Benign-Monday": "Benign-Monday-no-metadata.parquet",
    "Botnet-Friday": "Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday": "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS-Friday": "DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday": "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday": "Infiltration-Thursday-no-metadata.parquet",
    "Portscan-Friday": "Portscan-Friday-no-metadata.parquet",
    "WebAttacks-Thursday": "WebAttacks-Thursday-no-metadata.parquet",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="XAI-NIDS | Intrusion Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 0.15rem;
        }

        .subtitle {
            color: #555;
            font-size: 1.05rem;
            line-height: 1.5;
            margin-bottom: 1.6rem;
        }

        .risk-critical {
            color: #b00020;
            font-weight: 700;
        }

        .risk-high {
            color: #d35400;
            font-weight: 700;
        }

        .risk-medium {
            color: #9a7d0a;
            font-weight: 700;
        }

        .risk-low {
            color: #287a35;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL / API LOADING
# ============================================================

@st.cache_resource
def load_inference_api() -> InferenceAPI:
    """Load the trained GTAE model and fitted preprocessor once."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GTAE model not found:\n{MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:\n{PREPROCESSOR_PATH}"
        )

    device = GPU_CONFIG.get("device", "cpu")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return InferenceAPI(
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        anomaly_threshold=2.0,
        confidence_threshold=0.6,
        device=device,
    )


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_parquet(path: str, sample_size: int) -> pd.DataFrame:
    """Load a limited sample of a CIC-IDS2017 parquet file."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{file_path}"
        )

    df = pd.read_parquet(file_path)

    if sample_size < len(df):
        df = df.sample(
            n=sample_size,
            random_state=42,
        )

    return df.reset_index(drop=True)


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    api: InferenceAPI,
    df: pd.DataFrame,
) -> list:
    """Run the existing XAI-NIDS inference pipeline."""

    return api.predict(df)


# ============================================================
# RESULT DATAFRAME
# ============================================================

def results_to_dataframe(results: list) -> pd.DataFrame:

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    st.sidebar.title("XAI-NIDS")

    st.sidebar.caption(
        "Network Intrusion Detection"
    )

    st.sidebar.divider()

    st.sidebar.subheader("Detection Configuration")

    source = st.sidebar.radio(
        "Input source",
        [
            "Built-in CIC-IDS2017",
            "Upload CSV",
            "Upload Parquet",
        ],
    )

    sample_size = st.sidebar.slider(
        "Number of flows",
        min_value=10,
        max_value=5000,
        value=100,
        step=10,
    )

    selected_dataset = None
    uploaded_file = None

    if source == "Built-in CIC-IDS2017":

        selected_dataset = st.sidebar.selectbox(
            "Dataset",
            list(DATASETS.keys()),
        )

    else:

        uploaded_file = st.sidebar.file_uploader(
            "Upload network flow data",
            type=["csv", "parquet"],
        )

    st.sidebar.divider()

    st.sidebar.subheader("System")

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        st.sidebar.success(
            f"CUDA available\n\n{gpu_name}"
        )

    else:

        st.sidebar.warning(
            "CUDA unavailable — using CPU"
        )

    st.sidebar.divider()

    st.sidebar.caption(
        "Model: GTAE-IDS"
    )

    st.sidebar.caption(
        "Dataset: CIC-IDS2017"
    )

    return (
        source,
        selected_dataset,
        uploaded_file,
        sample_size,
    )


# ============================================================
# OVERVIEW
# ============================================================

def render_overview(results_df: pd.DataFrame):

    st.subheader("System Overview")

    if results_df.empty:
        st.info("No inference results available.")
        return

    st.caption("These values represent model predictions. Ground-truth labels are used separately for evaluation.")
    st.write("")

    total = len(results_df)

    benign = int(
        (results_df["category"] == "BENIGN").sum()
    )

    known_attack = int(
        (results_df["category"] == "KNOWN_ATTACK").sum()
    )

    unknown = int(
        (results_df["category"] == "UNKNOWN_NOVEL").sum()
    )

    high_critical = int(
        results_df["severity"].isin(
            ["HIGH", "CRITICAL"]
        ).sum()
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Flows Analyzed",
        total,
    )

    c2.metric(
        "Predicted Benign",
        benign,
    )

    c3.metric(
        "Predicted Attacks",
        known_attack,
    )

    c4.metric(
        "Predicted Novel",
        unknown,
    )

    c5.metric(
        "High / Critical",
        high_critical,
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Detection Categories")

        category_counts = (
            results_df["category"]
            .value_counts()
        )

        st.bar_chart(category_counts, height=350)

    with col2:

        st.markdown("### Predicted Attack Families")

        type_counts = (
            results_df["detected_type"]
            .value_counts()
        )

        st.bar_chart(type_counts, height=350)

    st.write("")
    st.divider()
    st.write("")

    col3, col4 = st.columns(2)

    with col3:

        st.markdown("### Severity")

        severity_counts = (
            results_df["severity"]
            .value_counts()
        )

        st.bar_chart(severity_counts, height=300)

    with col4:

        st.markdown("### Anomaly Score Distribution")
        st.caption("Anomaly scores represent reconstruction-based deviation from learned network-flow patterns.")

        st.line_chart(
            results_df[
                ["anomaly_score"]
            ],
            height=300
        )


# ============================================================
# SINGLE FLOW
# ============================================================

def render_single_flow(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
):

    st.subheader("Single Flow Analysis")

    if results_df.empty:
        st.info("Run inference first.")
        return

    st.caption("Investigate individual network flows to review the model's prediction, anomaly detection, and risk assessment.")

    # 1. FLOW SELECTION
    st.markdown("### 1. Flow Selection")
    node_id = st.selectbox(
        "Select Flow ID",
        results_df["node_id"].tolist(),
    )

    result = results_df[
        results_df["node_id"] == node_id
    ].iloc[0]

    st.divider()

    # 2. DETECTION RESULT
    st.markdown("### 2. Detection Result")
    
    category = result["category"]

    if category == "UNKNOWN_NOVEL":
        st.error("Model Prediction: UNKNOWN / NOVEL THREAT")
    elif category == "KNOWN_ATTACK":
        st.warning(f"Model Prediction: KNOWN ATTACK ({result['detected_type']})")
    else:
        st.success("Model Prediction: BENIGN TRAFFIC")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Class", result["detected_type"])
    c2.metric("Confidence", f"{result['classifier_prob']:.2%}")
    c3.metric("Anomaly Status", "Anomalous" if result.get("is_anomalous", False) else "Normal")

    c4, c5, c6 = st.columns(3)
    c4.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
    c5.metric("Risk Score", f"{result['risk_score']:.2f}/100")
    c6.metric("Severity", result["severity"])

    st.divider()

    # 3. SECURITY ASSESSMENT
    st.markdown("### 3. Security Assessment")
    st.info(
        "- **Classification Result:** The graph-transformer predicts if the flow belongs to a known attack class or is benign traffic.\n"
        "- **Anomaly Detection:** The reconstruction-decoder flags the flow if its patterns deviate significantly from learned normal behavior.\n"
        "- **Risk Assessment:** Combines classification confidence, anomaly score, and threat type to prioritize the alert."
    )
    
    st.write("Detailed feature-level explanations for this flow can be generated in the **XAI** tab. The explanation highlights which input features contributed most to the model's reconstruction/prediction behavior.")

    st.divider()

    # 4. FLOW FEATURES
    st.markdown("### 4. Flow Features")
    with st.expander("View Raw Network Flow Features", expanded=False):
        flow = df.iloc[int(node_id)]
        st.dataframe(
            flow.to_frame("Feature Value"),
            use_container_width=True,
        )


# ============================================================
# THREAT ANALYSIS
# ============================================================

def render_threat_analysis(
    results_df: pd.DataFrame,
):

    st.subheader("Threat Analysis")

    if results_df.empty:
        st.info("No results available.")
        return

    st.caption("Risk assessment combines the detection outputs already produced by the XAI-NIDS pipeline. These are model predictions, not ground-truth labels.")
    st.write("")

    critical_count = len(results_df[results_df["severity"] == "CRITICAL"])
    high_count = len(results_df[results_df["severity"] == "HIGH"])
    avg_risk = results_df["risk_score"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Critical Threats", critical_count)
    c2.metric("High Severity Threats", high_count)
    c3.metric("Average Risk Score", f"{avg_risk:.2f}")

    st.divider()
    st.markdown("### Threat Assessment Details")

    filter_option = st.selectbox(
        "Filter Predictions",
        [
            "All",
            "Predicted Benign",
            "Predicted Known Attack",
            "Predicted Novel",
            "High Risk",
            "Critical Risk",
        ],
    )

    filtered = results_df.copy()

    if filter_option == "Predicted Benign":
        filtered = filtered[filtered["category"] == "BENIGN"]
    elif filter_option == "Predicted Known Attack":
        filtered = filtered[filtered["category"] == "KNOWN_ATTACK"]
    elif filter_option == "Predicted Novel":
        filtered = filtered[filtered["category"] == "UNKNOWN_NOVEL"]
    elif filter_option == "High Risk":
        filtered = filtered[filtered["severity"].isin(["HIGH", "CRITICAL"])]
    elif filter_option == "Critical Risk":
        filtered = filtered[filtered["severity"] == "CRITICAL"]

    filtered = filtered.sort_values(
        "risk_score",
        ascending=False,
    )

    display_columns = [
        "node_id",
        "category",
        "detected_type",
        "is_anomalous",
        "anomaly_score",
        "severity",
        "risk_score",
        "classifier_prob",
    ]

    display_df = filtered[display_columns].copy()
    display_df.rename(columns={
        "node_id": "Flow ID",
        "category": "Threat Category",
        "detected_type": "Predicted Attack",
        "is_anomalous": "Anomaly Status",
        "anomaly_score": "Anomaly Score",
        "severity": "Severity",
        "risk_score": "Risk Score",
        "classifier_prob": "Confidence",
    }, inplace=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Threat Assessment CSV",
        data=display_df.to_csv(index=False),
        file_name="xai_nids_threat_analysis.csv",
        mime="text/csv",
    )


# ============================================================
# XAI
# ============================================================

def render_xai(
    api: InferenceAPI,
    df: pd.DataFrame,
    results_df: pd.DataFrame,
):

    st.subheader("Explainable AI")

    if results_df.empty:
        st.info("Run inference first.")
        return

    node_id = st.selectbox(
        "Select node to explain",
        results_df["node_id"].tolist(),
        key="xai_node",
    )

    result = results_df[
        results_df["node_id"] == node_id
    ].iloc[0]

    st.markdown("### Selected Flow Details")
    st.caption("The XAI component identifies which network-flow features contributed most strongly to the model's decision.")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Class", result['detected_type'])
    c2.metric("Confidence", f"{result['classifier_prob']:.2%}")
    c3.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
    c4.metric("Severity", result['severity'])

    st.divider()
    st.markdown("### How to interpret")
    st.info("The charts below show the relative importance of different features. Features with higher scores had a larger impact on the model's classification or anomaly detection for this specific network flow.")


    try:

        # ----------------------------------------------------
        # Recreate preprocessing and graph
        # ----------------------------------------------------

        x_scaled = api.preprocessor.transform(df)

        graph = api.graph_builder.build_graph(
            x_scaled
        )

        x_tensor = torch.tensor(
            x_scaled,
            dtype=torch.float32,
            device=api.device,
        )

        edge_index = graph.edge_index.to(
            api.device
        )

        edge_weight = graph.edge_weight.to(
            api.device
        )

        feature_names = list(
            api.preprocessor.feature_names_
        )

        explainer = IntrusionExplainer(
            model=api.model,
            feature_names=feature_names,
        )

        # ----------------------------------------------------
        # Reconstruction importance
        # ----------------------------------------------------

        st.markdown(
            "### 1. Reconstruction Feature Importance"
        )

        reconstruction = (
            explainer.compute_reconstruction_importance(
                x_tensor,
                edge_index,
                edge_weight=edge_weight,
                top_k=10,
            )
        )

        node_explanation = reconstruction[
            int(node_id)
        ]

        feature_scores = pd.DataFrame(
            node_explanation["top_features"],
            columns=[
                "Feature",
                "Reconstruction Error",
            ],
        )

        st.bar_chart(
            feature_scores.set_index(
                "Feature"
            ),
            height=350
        )

        st.dataframe(
            feature_scores,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Ablation importance
        # ----------------------------------------------------

        st.markdown(
            "### 2. Feature Ablation Importance"
        )

        predicted_class = int(
            api.model.predict(
                x_tensor,
                edge_index,
                edge_weight,
            )[0][int(node_id)]
        )

        ablation = (
            explainer.compute_ablation_importance(
                x_tensor,
                edge_index,
                node_idx=int(node_id),
                target_class=predicted_class,
                edge_weight=edge_weight,
                top_k=10,
            )
        )

        ablation_df = pd.DataFrame(
            ablation,
            columns=[
                "Feature",
                "Probability Impact",
            ],
        )

        st.bar_chart(
            ablation_df.set_index(
                "Feature"
            ),
            height=350
        )

        st.dataframe(
            ablation_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Neighborhood explanation
        # ----------------------------------------------------

        st.markdown(
            "### 3. Graph Neighborhood Explanation"
        )

        neighborhood = (
            explainer.explain_neighborhood(
                x_tensor,
                edge_index,
                node_idx=int(node_id),
                target_class=predicted_class,
            )
        )

        neighbors = neighborhood[
            "top_influential_neighbors"
        ]

        if neighbors:

            neighborhood_df = pd.DataFrame(
                neighbors,
                columns=[
                    "Neighbor Node",
                    "Probability Impact",
                ],
            )

            st.dataframe(
                neighborhood_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No connected neighbors available for explanation."
            )

    except Exception as exc:

        st.error(
            "Unable to generate explanation for this flow."
        )

        st.exception(exc)


# ============================================================
# MODEL INFORMATION
# ============================================================

def render_model_information():
    import json
    from pathlib import Path
    
    st.subheader("Model Information")
    st.caption("Detailed overview of the GTAE-IDS architecture, hyperparameters, and training configuration.")

    config = {}
    config_path = Path("models/gtae_config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception:
            pass

    st.markdown("### 1. Model Overview")
    st.write("**Graph Transformer Autoencoder Intrusion Detection System (GTAE-IDS)**")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Architecture", config.get("model_architecture", "GTAE-IDS").replace("_", "-"))
    c2.metric("Encoder Type", str(config.get("encoder_type", "transformer")).title())
    c3.metric("Training Mode", str(config.get("training_mode", "supervised_hybrid")).replace("_", " ").title())
    c4.metric("Number of Classes", config.get("num_classes", 8))

    st.divider()

    st.markdown("### 2. Architecture Flow")
    st.info(
        "**Flow Features** &rarr; **Similarity Graph** &rarr; **Graph Transformer** &rarr; "
        "**Latent Representation** &rarr; **Classification Head** &rarr; **Reconstruction Decoder**"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 3. Model Parameters")
        params_df = pd.DataFrame({
            "Parameter": [
                "Hidden Dimension", "Latent Dimension", "Attention Heads", 
                "Encoder Layers", "Dropout", "Graph Size", "k-Neighbors"
            ],
            "Value": [
                config.get("hidden_dim", 128), config.get("latent_dim", 64), config.get("num_heads", 4),
                config.get("num_encoder_layers", 2), config.get("dropout", 0.2), 
                config.get("graph_size", 1000), config.get("k_neighbors", 5)
            ]
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 4. Training Configuration")
        train_df = pd.DataFrame({
            "Parameter": [
                "Epochs", "Learning Rate", "Loss Type", 
                "Reconstruction-Loss Weight", "Anomaly Percentile", "Anomaly Threshold"
            ],
            "Value": [
                config.get("epochs", 5), config.get("learning_rate", 0.001), 
                str(config.get("loss_type", "smooth_l1")).replace("_", " ").title(), 
                config.get("lambda_rec", 0.5), f"{config.get('anomaly_percentile', 95.0)}th",
                f"{config.get('anomaly_threshold', 1.1886):.4f}"
            ]
        })
        st.dataframe(train_df, use_container_width=True, hide_index=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 5. Dataset")
        st.write("**CIC-IDS2017 Dataset**")
        st.write(f"- Input features: **{config.get('in_features', 67)}**")
        st.write(f"- Predictable classes: **{config.get('num_classes', 8)}**")
        st.caption("The dataset encompasses both benign traffic and distinct network attack families.")

    with col4:
        st.markdown("### 6. Zero-Day Separation")
        st.info(
            "The **Infiltration** held-out experiment (presented in the Zero-Day tab) uses a separate, "
            "independent experimental checkpoint. It does not replace or modify the normal production model shown here."
        )


# ============================================================
# ZERO-DAY EXPERIMENT
# ============================================================

def render_zero_day_experiment():
    st.subheader("Zero-Day / Held-Out Attack Experiment")

    st.markdown(
        "Infiltration was deliberately excluded from classification training and evaluated separately as an unseen attack class."
    )

    import json

    config = {}
    metrics = {}

    if ZERO_DAY_CONFIG_PATH.exists():
        try:
            with open(ZERO_DAY_CONFIG_PATH, "r") as f:
                config = json.load(f)
        except Exception:
            pass

    if ZERO_DAY_METRICS_PATH.exists():
        try:
            with open(ZERO_DAY_METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass

    heldout_class = metrics.get("heldout_attack", config.get("holdout_class", "Infiltration"))
    test_samples = metrics.get("test_samples", 7)
    detected = metrics.get("detected_as_anomaly", 4)
    missed = test_samples - detected
    detection_rate_val = metrics.get("detection_rate", 0.5714)
    detection_rate_pct = f"{detection_rate_val * 100:.2f}%" if detection_rate_val <= 1.0 else f"{detection_rate_val:.2f}%"
    anomaly_threshold = config.get("anomaly_threshold", metrics.get("anomaly_threshold", 4.360150))

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Held-Out Class", heldout_class)
    c2.metric("Test Samples", test_samples)
    c3.metric("Detected as Anomaly", f"{detected} ({detection_rate_pct})")
    c4.metric("Anomaly Threshold", f"{anomaly_threshold:.6f}")

    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Context & Explanation")
        st.write(
            "This tab presents the evaluation details for a dedicated zero-day simulation experiment. "
            "In this experiment, the model was trained completely without any samples from the "
            "**Infiltration** attack class. The purpose is to evaluate the model's capacity to "
            "detect entirely novel, unseen attack methods via reconstruction-based anomaly detection."
        )

        st.markdown("**Important Notes:**")
        st.info(
            "- **Model Separation:** The normal production model remains separate and unaffected by this experiment.\n"
            "- **Evaluation Only:** This is an experimental held-out evaluation to showcase the GTAE's zero-day detection capability.\n"
            "- **Infiltration Inclusion:** This experiment does not claim that the normal production checkpoint was trained without Infiltration."
        )

    with col2:
        st.markdown("### Anomaly Detection Summary")

        summary_df = pd.DataFrame({
            "Metric": ["Detected as Anomaly", "Missed"],
            "Count": [detected, missed],
            "Percentage": [f"{detection_rate_pct}", f"{(1 - (detected / test_samples)) * 100:.2f}%"]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        chart_df = pd.DataFrame({
            "Result": ["Detected", "Missed"],
            "Count": [detected, missed]
        }).set_index("Result")

        st.bar_chart(chart_df)


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.markdown(
        '<div class="main-title">XAI-NIDS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Explainable Graph-Based Network Intrusion Detection System<br>"
        "<span style='font-size: 0.88rem; color: #888;'>GTAE-IDS &bull; CIC-IDS2017</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    (
        source,
        selected_dataset,
        uploaded_file,
        sample_size,
    ) = render_sidebar()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        api = load_inference_api()

    except Exception as exc:

        st.error(
            "Unable to load the XAI-NIDS model."
        )

        st.exception(exc)

        st.stop()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    try:

        if source == "Built-in CIC-IDS2017":

            dataset_path = (
                DATA_PATH
                / DATASETS[selected_dataset]
            )

            df = load_parquet(
                str(dataset_path),
                sample_size,
            )

        elif uploaded_file is not None:

            if uploaded_file.name.lower().endswith(
                ".csv"
            ):

                df = pd.read_csv(
                    uploaded_file
                )

            else:

                df = pd.read_parquet(
                    uploaded_file
                )

            if len(df) > sample_size:

                df = df.sample(
                    n=sample_size,
                    random_state=42,
                )

            df = df.reset_index(
                drop=True
            )

        else:

            st.info(
                "Select a dataset or upload a file "
                "from the sidebar."
            )

            st.stop()

    except Exception as exc:

        st.error(
            "Unable to load the selected data."
        )

        st.exception(exc)

        st.stop()

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    if st.sidebar.button(
        "Run Detection",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Running GTAE-IDS detection..."
        ):

            try:

                results = run_inference(
                    api,
                    df,
                )

                st.session_state[
                    "results"
                ] = results

                st.session_state[
                    "data"
                ] = df

                st.success(
                    f"Analyzed {len(df):,} network flows."
                )

            except Exception as exc:

                st.error(
                    "Detection pipeline failed."
                )

                st.exception(exc)

    # --------------------------------------------------------
    # Retrieve session results
    # --------------------------------------------------------

    results = st.session_state.get(
        "results",
        None,
    )

    analyzed_df = st.session_state.get(
        "data",
        None,
    )

    if results is None:

        st.info(
            "Choose a dataset and click "
            "**Run Detection** to begin."
        )

        render_model_information()

        return

    results_df = results_to_dataframe(
        results
    )

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "Overview",
            "Single Flow",
            "Threat Analysis",
            "Evaluation",
            "XAI",
            "Model Information",
            "Zero-Day Experiment",
        ]
    )

    with tabs[0]:

        render_overview(
            results_df
        )

    with tabs[1]:

        render_single_flow(
            analyzed_df,
            results_df,
        )

    with tabs[2]:

        render_threat_analysis(
            results_df
        )

    with tabs[3]:

        st.subheader("Ground-Truth Model Evaluation")
        st.caption("These metrics evaluate model predictions against known ground-truth labels for this specific dataset. They serve as a benchmark and do not guarantee identical real-world performance.")
        st.write("")

        if "Label" not in analyzed_df.columns:

            st.warning("Ground-truth Label column is not available for this dataset.")

        else:

            evaluation = evaluate_predictions(analyzed_df, results_df)

            st.markdown("### Performance Summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{evaluation.get('accuracy', 0):.2%}")
            c2.metric("Macro Precision", f"{evaluation.get('macro_precision', 0):.2%}")
            c3.metric("Macro Recall", f"{evaluation.get('macro_recall', 0):.2%}")
            c4.metric("Macro F1", f"{evaluation.get('macro_f1', 0):.2%}")
            
            if 'weighted_f1' in evaluation:
                c5.metric("Weighted F1", f"{evaluation['weighted_f1']:.2%}")
            else:
                c5.metric("Benign FPR", f"{evaluation.get('benign_fpr', 0):.2%}")

            st.divider()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### How to interpret these metrics")
                st.info(
                    "- **Accuracy** measures overall correct predictions.\n"
                    "- **Precision** measures how reliable positive predictions are.\n"
                    "- **Recall** measures how many actual instances are detected.\n"
                    "- **F1** balances precision and recall.\n"
                    "- The **confusion matrix** shows true labels versus model predictions."
                )

                st.markdown("### Anomaly Detection")
                if 'benign_fpr' in evaluation:
                    st.metric("Benign False-Positive Rate", f"{evaluation['benign_fpr']:.2%}")
                if 'anomaly_threshold' in evaluation:
                    st.metric("Anomaly Threshold", f"{evaluation['anomaly_threshold']:.4f}")
                if 'known_attack_dr' in evaluation:
                    st.metric("Known Attack Detection Rate", f"{evaluation['known_attack_dr']:.2%}")

            with col2:
                st.markdown("### Confusion Matrix")
                st.dataframe(evaluation["confusion_matrix"], use_container_width=True)

            st.divider()

            st.markdown("### Per-Class Classification Performance")
            st.dataframe(evaluation["classification_report"], use_container_width=True, hide_index=True)

            st.markdown("### Incorrect Predictions")
            incorrect = evaluation["evaluation_dataframe"][~evaluation["evaluation_dataframe"]["correct_prediction"]]
            st.dataframe(incorrect, use_container_width=True, hide_index=True)
    with tabs[4]:

        render_xai(
            api,
            analyzed_df,
            results_df,
        )

    with tabs[5]:

        render_model_information()

    with tabs[6]:

        render_zero_day_experiment()


if __name__ == "__main__":
    main()






