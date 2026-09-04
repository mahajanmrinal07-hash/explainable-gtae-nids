"""
Real-time live network traffic monitoring tab for XAI-NIDS Streamlit Dashboard.

Discovers local Npcap-compatible network interfaces, provides configurable
finite-duration packet capture via NFStream, runs end-to-end inference and
risk scoring using the trained GTAE-IDS model, and visualizes live detection results.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.config import GPU_CONFIG, MODELS_DIR

# Safe import of live capture functionality to prevent dashboard crash if Npcap is missing
try:
    from tools.live_capture import get_available_interfaces, run_live_capture
    HAS_LIVE_CAPTURE = True
    LIVE_CAPTURE_ERROR = None
except Exception as exc:
    HAS_LIVE_CAPTURE = False
    LIVE_CAPTURE_ERROR = str(exc)
    get_available_interfaces = None
    run_live_capture = None

# Default asset paths
MODEL_PATH = MODELS_DIR / "gtae_ids.pt"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def _format_interface_label(iface: Dict[str, Any]) -> str:
    """Format human-friendly interface description with IP addresses."""
    idx = iface.get("index", 0)
    desc = iface.get("description", "Interface")
    ips = iface.get("ips", [])
    ip_str = f" | IP: {', '.join(ips)}" if ips else " | No IPv4"
    return f"[{idx}] {desc}{ip_str}"


def _find_default_interface_index(interfaces: List[Dict[str, Any]]) -> int:
    """Find the best default interface index (preferring Wi-Fi or Ethernet with IP)."""
    for i, iface in enumerate(interfaces):
        desc = iface.get("description", "").lower()
        ips = iface.get("ips", [])
        if ips and ("wi-fi" in desc or "ethernet" in desc or "wlan" in desc):
            return i
    for i, iface in enumerate(interfaces):
        if iface.get("ips"):
            return i
    return 0


def render_live_monitoring() -> None:
    """Renders the Real-Time Live Network Monitoring interface."""
    st.subheader("Real-Time Live Network Traffic Monitoring")
    st.caption(
        "Capture live network packets from local interfaces via Npcap & NFStream, "
        "reconstruct flow embeddings using the GTAE-IDS autoencoder, and score active threats."
    )
    st.write("")

    # 1. Dependency and Environment Validation
    if not HAS_LIVE_CAPTURE:
        st.error(
            f"❌ Live capture module unavailable: {LIVE_CAPTURE_ERROR}\n\n"
            "Please ensure Npcap (with WinPcap API compatibility) and NFStream are installed, "
            "and the Npcap packet capture driver service is running."
        )
        return

    # 2. Check model artifacts
    model_ready = True
    missing_assets = []
    if not MODEL_PATH.exists():
        missing_assets.append(f"GTAE-IDS Model checkpoint: `{MODEL_PATH}`")
        model_ready = False
    if not PREPROCESSOR_PATH.exists():
        missing_assets.append(f"Preprocessor artifact: `{PREPROCESSOR_PATH}`")
        model_ready = False

    if not model_ready:
        st.warning(
            "⚠️ Missing required model artifacts for live threat scoring:\n- "
            + "\n- ".join(missing_assets)
            + "\n\nPlease ensure models are trained and saved before initiating live capture."
        )

    # 3. Discover Available Interfaces
    try:
        interfaces = get_available_interfaces()
    except Exception as exc:
        interfaces = []
        st.error(f"Failed to query network interfaces: {exc}")

    if not interfaces:
        st.error(
            "❌ No network interfaces detected on the host system. "
            "Please check that Npcap is installed and running with appropriate permissions."
        )
        return

    # 4. Capture Controls Form / Inputs
    st.markdown("### ⚙️ Capture Configuration")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

    with ctrl_col1:
        default_idx = _find_default_interface_index(interfaces)
        selected_label = st.selectbox(
            "Select Network Interface",
            options=[_format_interface_label(iface) for iface in interfaces],
            index=default_idx,
            help="Select the physical or virtual adapter on which to monitor network flows.",
        )
        # Match back to the selected interface dict
        selected_index = int(selected_label.split("]")[0].replace("[", ""))
        selected_iface = next(
            (iface for iface in interfaces if iface.get("index") == selected_index),
            interfaces[0],
        )

    with ctrl_col2:
        duration_sec = st.slider(
            "Capture Duration (seconds)",
            min_value=3,
            max_value=60,
            value=10,
            step=1,
            help="Observation window in seconds before expiring flows and running model inference.",
        )

    with ctrl_col3:
        bpf_filter = st.text_input(
            "BPF Filter (optional)",
            value="",
            placeholder="e.g. tcp, port 443, ip",
            help="Berkeley Packet Filter string to restrict captured packets.",
        )

    # Interface metadata display
    with st.expander("ℹ️ Interface Details", expanded=False):
        st.markdown(f"- **Device ID**: `{selected_iface.get('name', 'N/A')}`")
        st.markdown(f"- **Description**: {selected_iface.get('description', 'N/A')}")
        st.markdown(
            f"- **Assigned IPv4**: {', '.join(selected_iface.get('ips', [])) if selected_iface.get('ips') else 'None'}"
        )

    # 5. Start Capture Action
    start_capture_btn = st.button(
        "🚀 Start Live Capture",
        type="primary",
        disabled=not model_ready,
        help="Initiates real-time packet capture on the selected interface for the specified duration.",
    )

    if start_capture_btn:
        filter_str = bpf_filter.strip() if bpf_filter.strip() else None
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            with st.spinner(
                f"Capturing traffic on {selected_iface.get('description', 'interface')} "
                f"for {duration_sec} seconds... Generating or browsing traffic will produce completed flows."
            ):
                try:
                    capture_result = run_live_capture(
                        interface=selected_iface["name"],
                        duration=float(duration_sec),
                        bpf_filter=filter_str,
                        model_path=MODEL_PATH,
                        preprocessor_path=PREPROCESSOR_PATH,
                        device=GPU_CONFIG.get("device", "cpu"),
                    )
                    st.session_state["live_capture_results"] = capture_result
                    st.session_state["live_capture_timestamp"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                except Exception as exc:
                    st.error(f"❌ Error during live packet capture or model inference: {exc}")

        progress_placeholder.empty()

    # 6. Render Captured Results
    results = st.session_state.get("live_capture_results")
    if not results:
        st.info("👋 Select an interface and click **Start Live Capture** to begin real-time detection.")
        return

    st.divider()

    # 7. Summary & Metrics
    summary = results.get("summary", {})
    total_flows = summary.get("total_flows", 0)
    category_counts = summary.get("category_counts", {})
    severity_counts = summary.get("severity_counts", {})
    alerts_count = summary.get("alerts_count", 0)

    benign_count = category_counts.get("BENIGN", 0)
    known_attack_count = category_counts.get("KNOWN_ATTACK", 0)
    unknown_novel_count = category_counts.get("UNKNOWN_NOVEL", 0)
    high_critical_count = severity_counts.get("HIGH", 0) + severity_counts.get("CRITICAL", 0)

    st.markdown("### 📊 Live Traffic Detection Summary")
    last_timestamp = st.session_state.get("live_capture_timestamp", "Recent")
    st.caption(f"Last capture executed at: **{last_timestamp}** | Monitored Interface: `{summary.get('interface', 'N/A')}`")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Flows Captured", total_flows)
    m2.metric("BENIGN", benign_count)
    m3.metric(
        "KNOWN ATTACK",
        known_attack_count,
        delta="Threat" if known_attack_count > 0 else None,
        delta_color="inverse",
    )
    m4.metric(
        "UNKNOWN / NOVEL",
        unknown_novel_count,
        delta="Novel Threat" if unknown_novel_count > 0 else None,
        delta_color="inverse",
    )
    m5.metric(
        "HIGH / CRITICAL",
        high_critical_count,
        delta="Urgent" if high_critical_count > 0 else None,
        delta_color="inverse",
    )
    m6.metric(
        "Total Alerts",
        alerts_count,
        delta="Alert" if alerts_count > 0 else None,
        delta_color="inverse",
    )

    # Advisory Note
    st.info(
        "ℹ️ **Security Guidance Note**: Model predictions, reconstruction anomaly scores, and threat risk ratings "
        "are statistical estimates computed by the GTAE-IDS autoencoder and classifier. "
        "They provide operational triage assistance and do **not constitute definitive proof** of an actual compromise. "
        "High anomaly scores may reflect benign protocol variations, bursts, or unseen legitimate applications."
    )

    # Handle zero flows case
    if total_flows == 0:
        st.warning(
            "⚠️ No completed network flows were detected during the capture interval. "
            "Suggestions:\n"
            "- Ensure the selected network interface has active traffic (e.g. browse web pages or stream media).\n"
            "- Increase the capture duration (e.g., 15–30 seconds).\n"
            "- Check that any custom BPF filter is not too restrictive."
        )
        return

    # 8. Detailed Flows Table
    st.markdown("### 🔍 Captured Flow Inspections")

    flow_records = results.get("flows", [])
    table_data = []

    for f in flow_records:
        meta = f.get("flow_info", {})
        proto_num = meta.get("protocol", 0)
        proto_name = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(proto_num, f"Proto {proto_num}")

        category = f.get("predicted_category", "BENIGN")
        if category == "BENIGN":
            cat_badge = "🟢 BENIGN"
        elif category == "KNOWN_ATTACK":
            cat_badge = "🔴 KNOWN_ATTACK"
        elif category == "UNKNOWN_NOVEL":
            cat_badge = "🟣 UNKNOWN_NOVEL"
        else:
            cat_badge = category

        table_data.append({
            "Flow ID": meta.get("flow_id", 0),
            "Source": f"{meta.get('src_ip', '0.0.0.0')}:{meta.get('src_port', 0)}",
            "Destination": f"{meta.get('dst_ip', '0.0.0.0')}:{meta.get('dst_port', 0)}",
            "Protocol": proto_name,
            "Application": meta.get("application_name", "Unknown"),
            "Packets": meta.get("total_packets", 0),
            "Bytes": meta.get("total_bytes", 0),
            "Duration (ms)": meta.get("duration_ms", 0.0),
            "Category": cat_badge,
            "Raw Category": category,
            "Detected Type": f.get("detected_type", "BENIGN"),
            "Severity": f.get("severity", "NONE"),
            "Risk Score": f.get("risk_score", 0.0),
            "Anomaly Score": f.get("anomaly_score", 0.0),
            "Confidence": f"{f.get('classifier_confidence', 0.0):.1%}",
            "Status": f.get("detection_status", "NORMAL"),
        })

    df_flows = pd.DataFrame(table_data)

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        cat_filter = st.selectbox(
            "Filter by Category",
            options=["All", "BENIGN", "KNOWN_ATTACK", "UNKNOWN_NOVEL"],
            index=0,
        )
    with f_col2:
        sev_filter = st.selectbox(
            "Filter by Severity",
            options=["All", "CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"],
            index=0,
        )
    with f_col3:
        sort_by = st.selectbox(
            "Sort By",
            options=["Risk Score (High to Low)", "Anomaly Score (High to Low)", "Flow ID (Ascending)"],
            index=0,
        )

    filtered_df = df_flows.copy()
    if cat_filter != "All":
        filtered_df = filtered_df[filtered_df["Raw Category"] == cat_filter]
    if sev_filter != "All":
        filtered_df = filtered_df[filtered_df["Severity"] == sev_filter]

    if sort_by == "Risk Score (High to Low)":
        filtered_df = filtered_df.sort_values(by="Risk Score", ascending=False)
    elif sort_by == "Anomaly Score (High to Low)":
        filtered_df = filtered_df.sort_values(by="Anomaly Score", ascending=False)
    else:
        filtered_df = filtered_df.sort_values(by="Flow ID", ascending=True)

    display_cols = [
        "Flow ID",
        "Source",
        "Destination",
        "Protocol",
        "Application",
        "Packets",
        "Bytes",
        "Duration (ms)",
        "Category",
        "Detected Type",
        "Severity",
        "Risk Score",
        "Anomaly Score",
        "Confidence",
        "Status",
    ]

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # 9. Individual Flow Deep Dive
    with st.expander("🔎 Deep Flow Inspection", expanded=False):
        selected_flow_id = st.selectbox(
            "Select Flow to Inspect",
            options=df_flows["Flow ID"].tolist(),
            format_func=lambda fid: f"Flow #{fid} - {df_flows.loc[df_flows['Flow ID'] == fid, 'Source'].values[0]} -> {df_flows.loc[df_flows['Flow ID'] == fid, 'Destination'].values[0]} ({df_flows.loc[df_flows['Flow ID'] == fid, 'Detected Type'].values[0]})",
        )

        flow_entry = next(
            (f for f in flow_records if f.get("flow_info", {}).get("flow_id") == selected_flow_id),
            None,
        )

        if flow_entry:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("#### Flow Attributes")
                st.json(flow_entry.get("flow_info", {}))
            with d_col2:
                st.markdown("#### Threat & Anomaly Evaluation")
                eval_details = {
                    "predicted_category": flow_entry.get("predicted_category"),
                    "detected_type": flow_entry.get("detected_type"),
                    "classifier_confidence": flow_entry.get("classifier_confidence"),
                    "anomaly_score": flow_entry.get("anomaly_score"),
                    "is_anomalous": flow_entry.get("is_anomalous"),
                    "severity": flow_entry.get("severity"),
                    "risk_score": flow_entry.get("risk_score"),
                    "detection_status": flow_entry.get("detection_status"),
                }
                st.json(eval_details)
