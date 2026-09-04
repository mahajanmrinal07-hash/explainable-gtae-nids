"""
Real-time network traffic capture and threat detection module for XAI-NIDS.

Captures live network flows using NFStream, maps them to the exact 67-feature
CIC-IDS2017 schema expected by the Preprocessor, and executes end-to-end inference
via InferenceAPI with the trained GTAE-IDS model and threat risk engine.
"""

import argparse
import ctypes
import json
import os
import platform
import queue
import struct
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from nfstream import NFPlugin, NFStreamer
from nfstream.meter import meter_workflow
from nfstream.system import match_flow_conn
from nfstream.utils import NFEvent, NFMode, set_affinity

# Add project root to sys.path to ensure src imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GPU_CONFIG, MODELS_DIR, RESULTS_DIR
from src.detection.inference import InferenceAPI
from src.preprocessing import Preprocessor

# Default capture outputs directory
LIVE_CAPTURE_DIR = RESULTS_DIR / "live_capture"
LIVE_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 67-Feature CIC-IDS2017 Specification
# -----------------------------------------------------------------------------
# The exact 67 features expected by models/preprocessor.joblib:
CIC_67_FEATURES: List[str] = [
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Avg Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Fwd Act Data Packets",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# Features that cannot be natively measured by NFStream and are imputed
# via Preprocessor training medians rather than fabricating data.
# (Note: Header lengths, window bytes, active data packets, and min segment size
# are now dynamically measured per-packet via CICFeaturePlugin).
UNAVAILABLE_CIC_FEATURES: Dict[str, str] = {
    "Active Mean": "Active burst duration statistics require micro-interval connection state machines",
    "Active Std": "Active burst duration statistics require micro-interval connection state machines",
    "Active Max": "Active burst duration statistics require micro-interval connection state machines",
    "Active Min": "Active burst duration statistics require micro-interval connection state machines",
    "Idle Mean": "Idle interval duration statistics require micro-interval connection state machines",
    "Idle Std": "Idle interval duration statistics require micro-interval connection state machines",
    "Idle Max": "Idle interval duration statistics require micro-interval connection state machines",
    "Idle Min": "Idle interval duration statistics require micro-interval connection state machines",
}


# -----------------------------------------------------------------------------
# Interface Discovery
# -----------------------------------------------------------------------------
class PcapIf(ctypes.Structure):
    pass


PcapIf._fields_ = [
    ("next", ctypes.POINTER(PcapIf)),
    ("name", ctypes.c_char_p),
    ("description", ctypes.c_char_p),
    ("addresses", ctypes.c_void_p),
    ("flags", ctypes.c_uint),
]


def get_available_interfaces() -> List[Dict[str, Any]]:
    """
    Discovers network interfaces available for packet capture on Windows via Npcap
    and psutil.

    Returns:
        List of dicts with interface index, device name, description, and IP addresses.
    """
    # 1. Map psutil addresses by adapter name
    psutil_addrs: Dict[str, List[str]] = {}
    try:
        for iface_name, addrs in psutil.net_if_addrs().items():
            ipv4s = [
                a.address for a in addrs if getattr(a.family, "name", "") == "AF_INET"
            ]
            psutil_addrs[iface_name] = ipv4s
    except Exception:
        pass

    interfaces: List[Dict[str, Any]] = []

    # 2. Query Npcap/WinPcap devices via wpcap.dll
    try:
        wpcap = ctypes.cdll.LoadLibrary("wpcap.dll")
        alldevs = ctypes.POINTER(PcapIf)()
        errbuf = ctypes.create_string_buffer(256)
        res = wpcap.pcap_findalldevs(ctypes.byref(alldevs), errbuf)
        if res == 0 and alldevs:
            curr = alldevs
            idx = 0
            while curr:
                d = curr.contents
                dev_name = (
                    d.name.decode("utf-8", errors="ignore") if d.name else f"dev_{idx}"
                )
                desc = (
                    d.description.decode("utf-8", errors="ignore")
                    if d.description
                    else ""
                )

                # Match with psutil friendly name / IP
                matched_ips = []
                for p_name, ips in psutil_addrs.items():
                    if (
                        p_name.lower() in desc.lower()
                        or desc.lower() in p_name.lower()
                    ):
                        matched_ips.extend(ips)
                    elif "loopback" in dev_name.lower() and "loopback" in p_name.lower():
                        matched_ips.extend(ips)
                    elif "wi-fi" in desc.lower() and "wi-fi" in p_name.lower():
                        matched_ips.extend(ips)

                interfaces.append({
                    "index": idx,
                    "name": dev_name,
                    "description": desc or dev_name,
                    "ips": list(set(matched_ips)),
                })
                idx += 1
                curr = d.next
            wpcap.pcap_freealldevs(alldevs)
    except Exception:
        # Fallback to psutil if wpcap.dll direct call fails
        for idx, (p_name, ips) in enumerate(psutil_addrs.items()):
            interfaces.append({
                "index": idx,
                "name": p_name,
                "description": p_name,
                "ips": ips,
            })

    return interfaces


def resolve_interface(interface_arg: Union[str, int]) -> str:
    """
    Resolves an interface argument (index, description substring, or device name)
    to a valid device identifier accepted by NFStream.

    Args:
        interface_arg: String or integer identifier.

    Returns:
        Resolved device string (e.g., '\\Device\\NPF_{...}').
    """
    available = get_available_interfaces()
    if not available:
        raise RuntimeError("No network interfaces detected on the system.")

    target = str(interface_arg).strip()

    # Try numeric index
    if target.isdigit():
        idx = int(target)
        for iface in available:
            if iface["index"] == idx:
                return iface["name"]
        raise ValueError(
            f"Interface index {idx} not found. Available indices: 0 to {len(available) - 1}."
        )

    # Try exact device name match
    for iface in available:
        if iface["name"].lower() == target.lower():
            return iface["name"]

    # Try description / friendly name substring match
    for iface in available:
        if target.lower() in iface["description"].lower():
            return iface["name"]

    # Try IP address match
    for iface in available:
        if target in iface.get("ips", []):
            return iface["name"]

    raise ValueError(
        f"Could not match interface '{target}' to any available network interface."
    )


# -----------------------------------------------------------------------------
# Timed NFStreamer Engine (Finite Duration Guardrail)
# -----------------------------------------------------------------------------
class TimedNFStreamer(NFStreamer):
    """
    Subclasses NFStreamer to guarantee finite-duration live packet capture on Windows
    without blocking indefinitely when network traffic is quiet.
    """

    def __init__(self, *args, duration: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration = duration

    def __iter__(self):
        """
        Custom iterator loop with non-blocking queue polling and timeout checks.
        """
        lock = self._mp_context.Lock()
        lock.acquire()
        meters = []
        performances = []
        n_terminated = 0
        child_error = None
        conn_cache = {}

        n_meters = self.n_meters
        idx_generator = self._mp_context.Value("i", 0)
        for i in range(n_meters):
            performances.append([
                self._mp_context.Value("I", 0),
                self._mp_context.Value("I", 0),
                self._mp_context.Value("I", 0),
            ])
        channel = self._mp_context.Queue(maxsize=32767)
        group_id = os.getpid() + self._idx
        start_time = time.time()

        try:
            for i in range(n_meters):
                meters.append(
                    self._mp_context.Process(
                        target=meter_workflow,
                        args=(
                            self.source,
                            self.snapshot_length,
                            self.decode_tunnels,
                            self.bpf_filter,
                            self.promiscuous_mode,
                            n_meters,
                            i,
                            self._mode,
                            self.idle_timeout * 1000,
                            self.active_timeout * 1000,
                            self.accounting_mode,
                            self.udps,
                            self.n_dissections,
                            self.statistical_analysis,
                            self.splt_analysis,
                            channel,
                            performances[i],
                            lock,
                            group_id,
                            self.system_visibility_mode,
                            self.socket_buffer_size,
                        ),
                    )
                )
                meters[i].daemon = True
                meters[i].start()

            while True:
                # Check finite duration limit
                if (
                    self.duration is not None
                    and (time.time() - start_time) >= self.duration
                ):
                    for i in range(n_meters):
                        if meters[i].is_alive():
                            meters[i].terminate()
                    break

                try:
                    recv = channel.get(timeout=0.5)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    for i in range(n_meters):
                        if meters[i].is_alive():
                            meters[i].terminate()
                    break

                if recv is None:
                    n_terminated += 1
                    if n_terminated == n_meters:
                        break
                else:
                    if recv.id == NFEvent.ERROR:
                        for i in range(n_meters):
                            if meters[i].is_alive():
                                meters[i].terminate()
                        child_error = recv.message
                        break
                    elif recv.id == NFEvent.ALL_AFFINITY_SET:
                        set_affinity(0)
                    elif recv.id == NFEvent.SOCKET_CREATE:
                        conn_cache[recv.key] = [recv.process_name, recv.process_pid]
                    elif recv.id == NFEvent.SOCKET_REMOVE:
                        del conn_cache[recv.key]
                    else:
                        recv.id = idx_generator.value
                        idx_generator.value = idx_generator.value + 1
                        if self._mode == NFMode.INTERFACE and self.system_visibility_mode:
                            recv = match_flow_conn(conn_cache, recv)
                        yield recv
                        if self.max_nflows and recv.id == self.max_nflows:
                            for i in range(n_meters):
                                if meters[i].is_alive():
                                    meters[i].terminate()
                            break

            for i in range(n_meters):
                if meters[i].is_alive():
                    meters[i].join(timeout=1.0)
            channel.close()
            channel.join_thread()
            if child_error is not None:
                raise ValueError(child_error)
        except ValueError as observer_error:
            raise ValueError(observer_error)


# -----------------------------------------------------------------------------
# TCP Window Extraction & Dynamic Feature Measurement Plugin
# -----------------------------------------------------------------------------
def extract_tcp_window(packet: Any) -> Optional[float]:
    """
    Extracts the 16-bit TCP advertised window size from a SYN packet's IP payload.
    Returns None if packet is malformed, truncated, or not IPv4/IPv6 TCP.
    """
    ip_pkt = getattr(packet, "ip_packet", None)
    if not ip_pkt or len(ip_pkt) < 20:
        return None

    try:
        ip_ver = getattr(packet, "ip_version", 4)
        if ip_ver == 6:
            ip_header_len = 40
        else:
            ip_header_len = (ip_pkt[0] & 0x0F) * 4

        if len(ip_pkt) >= ip_header_len + 16:
            window = struct.unpack("!H", ip_pkt[ip_header_len + 14 : ip_header_len + 16])[0]
            return float(window)
    except (IndexError, struct.error, TypeError):
        pass

    return None


class CICFeaturePlugin(NFPlugin):
    """
    NFStream plugin to dynamically measure per-flow CIC-IDS2017 features
    that are not natively tracked in standard NFlow summary slots:
      - Fwd Header Length (cumulative across forward packets)
      - Bwd Header Length (cumulative across backward packets)
      - Init Fwd Win Bytes (TCP forward SYN window)
      - Init Bwd Win Bytes (TCP backward SYN/SYN-ACK window)
      - Fwd Act Data Packets (forward packets with payload_size > 0)
      - Fwd Seg Size Min (minimum transport header size in forward direction)
    """

    def on_init(self, packet: Any, flow: Any) -> None:
        flow.udps.fwd_header_len = 0.0
        flow.udps.bwd_header_len = 0.0
        flow.udps.init_fwd_win_bytes = None
        flow.udps.init_bwd_win_bytes = None
        flow.udps.fwd_act_data_packets = 0.0
        flow.udps.fwd_seg_size_min = None
        self._process_packet(packet, flow)

    def on_update(self, packet: Any, flow: Any) -> None:
        self._process_packet(packet, flow)

    def _process_packet(self, packet: Any, flow: Any) -> None:
        direction = getattr(packet, "direction", 0)
        ip_sz = float(getattr(packet, "ip_size", 0.0))
        pay_sz = float(getattr(packet, "payload_size", 0.0))
        hdr_sz = max(0.0, ip_sz - pay_sz)

        if direction == 0:
            flow.udps.fwd_header_len += hdr_sz
            if pay_sz > 0:
                flow.udps.fwd_act_data_packets += 1.0

            trans_sz = float(getattr(packet, "transport_size", 0.0))
            seg_sz = max(0.0, trans_sz - pay_sz)
            if seg_sz > 0:
                if (
                    flow.udps.fwd_seg_size_min is None
                    or seg_sz < flow.udps.fwd_seg_size_min
                ):
                    flow.udps.fwd_seg_size_min = seg_sz
        else:
            flow.udps.bwd_header_len += hdr_sz

        # TCP initial window extraction
        protocol = getattr(packet, "protocol", 0)
        if protocol == 6 and getattr(packet, "syn", 0):
            win = extract_tcp_window(packet)
            if win is not None:
                if direction == 0 and flow.udps.init_fwd_win_bytes is None:
                    flow.udps.init_fwd_win_bytes = win
                elif direction == 1 and flow.udps.init_bwd_win_bytes is None:
                    flow.udps.init_bwd_win_bytes = win

    def on_expire(self, flow: Any) -> None:
        pass


# -----------------------------------------------------------------------------
# NFStream Flow to 67-Feature CIC-IDS2017 Mapping
# -----------------------------------------------------------------------------
def map_nflow_to_cic_features(flow: Any) -> Dict[str, float]:
    """
    Converts an NFStream NFlow object into the exact 67-feature schema required
    by the XAI-NIDS Preprocessor.

    Derives features dynamically via CICFeaturePlugin measurements where available,
    applying protocol-aware fallbacks for missing handshakes or non-TCP flows.
    Remaining unmetered features (active/idle statistics) are assigned np.nan for
    Preprocessor training median imputation.

    Args:
        flow: NFStream NFlow instance or mock flow object.

    Returns:
        Dictionary mapping all 67 feature names to their numeric float values.
    """
    protocol = int(getattr(flow, "protocol", 0))

    # 1. Base counts and duration (NFStream uses ms; CIC-IDS2017 uses microseconds)
    duration_ms = float(getattr(flow, "bidirectional_duration_ms", 0.0))
    duration_us = duration_ms * 1000.0

    duration_sec = duration_ms / 1000.0 if duration_ms > 0 else 0.0

    total_fwd_pkts = float(getattr(flow, "src2dst_packets", 0.0))
    total_bwd_pkts = float(getattr(flow, "dst2src_packets", 0.0))
    total_pkts = float(getattr(flow, "bidirectional_packets", 0.0))

    total_fwd_bytes = float(getattr(flow, "src2dst_bytes", 0.0))
    total_bwd_bytes = float(getattr(flow, "dst2src_bytes", 0.0))
    total_bytes = float(getattr(flow, "bidirectional_bytes", 0.0))

    # Rates
    flow_bytes_per_sec = (total_bytes / duration_sec) if duration_sec > 0 else 0.0
    flow_pkts_per_sec = (total_pkts / duration_sec) if duration_sec > 0 else 0.0
    fwd_pkts_per_sec = (total_fwd_pkts / duration_sec) if duration_sec > 0 else 0.0
    bwd_pkts_per_sec = (total_bwd_pkts / duration_sec) if duration_sec > 0 else 0.0

    # Down/Up ratio
    down_up_ratio = (
        (total_bwd_pkts / total_fwd_pkts) if total_fwd_pkts > 0 else 0.0
    )

    # Packet size variance
    b_stddev_ps = float(getattr(flow, "bidirectional_stddev_ps", 0.0))
    packet_length_variance = b_stddev_ps**2

    # Plugin-extracted or protocol-aware features
    udps = getattr(flow, "udps", None)

    # 1. Fwd Header Length
    if udps is not None and getattr(udps, "fwd_header_len", None) is not None:
        fwd_header_length = float(udps.fwd_header_len)
    else:
        fwd_header_length = (
            total_fwd_pkts * (32.0 if protocol == 6 else 28.0)
            if total_fwd_pkts > 0
            else 0.0
        )

    # 2. Bwd Header Length
    if udps is not None and getattr(udps, "bwd_header_len", None) is not None:
        bwd_header_length = float(udps.bwd_header_len)
    else:
        bwd_header_length = (
            total_bwd_pkts * (32.0 if protocol == 6 else 28.0)
            if total_bwd_pkts > 0
            else 0.0
        )

    # 3. Init Fwd Win Bytes (-1.0 for non-TCP; SYN advertised window or fallback for TCP)
    if protocol != 6:
        init_fwd_win_bytes = -1.0
    else:
        if udps is not None and getattr(udps, "init_fwd_win_bytes", None) is not None:
            init_fwd_win_bytes = float(udps.init_fwd_win_bytes)
        else:
            init_fwd_win_bytes = 1834.5  # Training-compatible TCP fallback

    # 4. Init Bwd Win Bytes (-1.0 for non-TCP; SYN-ACK advertised window or fallback for TCP)
    if protocol != 6:
        init_bwd_win_bytes = -1.0
    else:
        if udps is not None and getattr(udps, "init_bwd_win_bytes", None) is not None:
            init_bwd_win_bytes = float(udps.init_bwd_win_bytes)
        else:
            init_bwd_win_bytes = 131.0  # Training-compatible TCP fallback

    # 5. Fwd Act Data Packets (forward packets with payload > 0)
    if udps is not None and getattr(udps, "fwd_act_data_packets", None) is not None:
        fwd_act_data_packets = float(udps.fwd_act_data_packets)
    else:
        if protocol != 6:
            fwd_act_data_packets = total_fwd_pkts
        else:
            fwd_act_data_packets = (
                max(1.0, total_fwd_pkts - 2.0) if total_fwd_pkts > 2 else total_fwd_pkts
            )

    # 6. Fwd Seg Size Min (minimum transport header size in forward direction)
    if udps is not None and getattr(udps, "fwd_seg_size_min", None) is not None:
        fwd_seg_size_min = float(udps.fwd_seg_size_min)
    else:
        if protocol == 6:
            fwd_seg_size_min = 32.0
        elif protocol == 17:
            fwd_seg_size_min = 20.0
        else:
            fwd_seg_size_min = 20.0

    # Map features
    features: Dict[str, float] = {
        "Protocol": float(protocol),
        "Flow Duration": duration_us,
        "Total Fwd Packets": total_fwd_pkts,
        "Total Backward Packets": total_bwd_pkts,
        "Fwd Packets Length Total": total_fwd_bytes,
        "Bwd Packets Length Total": total_bwd_bytes,
        "Fwd Packet Length Max": float(getattr(flow, "src2dst_max_ps", 0.0)),
        "Fwd Packet Length Min": float(getattr(flow, "src2dst_min_ps", 0.0)),
        "Fwd Packet Length Mean": float(getattr(flow, "src2dst_mean_ps", 0.0)),
        "Fwd Packet Length Std": float(getattr(flow, "src2dst_stddev_ps", 0.0)),
        "Bwd Packet Length Max": float(getattr(flow, "dst2src_max_ps", 0.0)),
        "Bwd Packet Length Min": float(getattr(flow, "dst2src_min_ps", 0.0)),
        "Bwd Packet Length Mean": float(getattr(flow, "dst2src_mean_ps", 0.0)),
        "Bwd Packet Length Std": float(getattr(flow, "dst2src_stddev_ps", 0.0)),
        "Flow Bytes/s": flow_bytes_per_sec,
        "Flow Packets/s": flow_pkts_per_sec,
        "Flow IAT Mean": float(getattr(flow, "bidirectional_mean_piat_ms", 0.0)) * 1000.0,
        "Flow IAT Std": float(getattr(flow, "bidirectional_stddev_piat_ms", 0.0)) * 1000.0,
        "Flow IAT Max": float(getattr(flow, "bidirectional_max_piat_ms", 0.0)) * 1000.0,
        "Flow IAT Min": float(getattr(flow, "bidirectional_min_piat_ms", 0.0)) * 1000.0,
        "Fwd IAT Total": float(getattr(flow, "src2dst_duration_ms", 0.0)) * 1000.0,
        "Fwd IAT Mean": float(getattr(flow, "src2dst_mean_piat_ms", 0.0)) * 1000.0,
        "Fwd IAT Std": float(getattr(flow, "src2dst_stddev_piat_ms", 0.0)) * 1000.0,
        "Fwd IAT Max": float(getattr(flow, "src2dst_max_piat_ms", 0.0)) * 1000.0,
        "Fwd IAT Min": float(getattr(flow, "src2dst_min_piat_ms", 0.0)) * 1000.0,
        "Bwd IAT Total": float(getattr(flow, "dst2src_duration_ms", 0.0)) * 1000.0,
        "Bwd IAT Mean": float(getattr(flow, "dst2src_mean_piat_ms", 0.0)) * 1000.0,
        "Bwd IAT Std": float(getattr(flow, "dst2src_stddev_piat_ms", 0.0)) * 1000.0,
        "Bwd IAT Max": float(getattr(flow, "dst2src_max_piat_ms", 0.0)) * 1000.0,
        "Bwd IAT Min": float(getattr(flow, "dst2src_min_piat_ms", 0.0)) * 1000.0,
        "Fwd PSH Flags": float(getattr(flow, "src2dst_psh_packets", 0)),
        "Fwd Header Length": fwd_header_length,
        "Bwd Header Length": bwd_header_length,
        "Fwd Packets/s": fwd_pkts_per_sec,
        "Bwd Packets/s": bwd_pkts_per_sec,
        "Packet Length Min": float(getattr(flow, "bidirectional_min_ps", 0.0)),
        "Packet Length Max": float(getattr(flow, "bidirectional_max_ps", 0.0)),
        "Packet Length Mean": float(getattr(flow, "bidirectional_mean_ps", 0.0)),
        "Packet Length Std": b_stddev_ps,
        "Packet Length Variance": packet_length_variance,
        "FIN Flag Count": float(getattr(flow, "bidirectional_fin_packets", 0)),
        "SYN Flag Count": float(getattr(flow, "bidirectional_syn_packets", 0)),
        "RST Flag Count": float(getattr(flow, "bidirectional_rst_packets", 0)),
        "PSH Flag Count": float(getattr(flow, "bidirectional_psh_packets", 0)),
        "ACK Flag Count": float(getattr(flow, "bidirectional_ack_packets", 0)),
        "URG Flag Count": float(getattr(flow, "bidirectional_urg_packets", 0)),
        "ECE Flag Count": float(getattr(flow, "bidirectional_ece_packets", 0)),
        "Down/Up Ratio": down_up_ratio,
        "Avg Packet Size": float(getattr(flow, "bidirectional_mean_ps", 0.0)),
        "Avg Fwd Segment Size": float(getattr(flow, "src2dst_mean_ps", 0.0)),
        "Avg Bwd Segment Size": float(getattr(flow, "dst2src_mean_ps", 0.0)),
        "Subflow Fwd Packets": total_fwd_pkts,
        "Subflow Fwd Bytes": total_fwd_bytes,
        "Subflow Bwd Packets": total_bwd_pkts,
        "Subflow Bwd Bytes": total_bwd_bytes,
        "Init Fwd Win Bytes": init_fwd_win_bytes,
        "Init Bwd Win Bytes": init_bwd_win_bytes,
        "Fwd Act Data Packets": fwd_act_data_packets,
        "Fwd Seg Size Min": fwd_seg_size_min,
        "Active Mean": np.nan,
        "Active Std": np.nan,
        "Active Max": np.nan,
        "Active Min": np.nan,
        "Idle Mean": np.nan,
        "Idle Std": np.nan,
        "Idle Max": np.nan,
        "Idle Min": np.nan,
    }

    return features


def extract_flow_metadata(flow: Any) -> Dict[str, Any]:
    """
    Extracts transport identification and metadata from an NFStream flow.

    Args:
        flow: NFStream NFlow instance or mock flow.

    Returns:
        Dictionary of flow identification attributes.
    """
    return {
        "flow_id": int(getattr(flow, "id", 0)),
        "src_ip": str(getattr(flow, "src_ip", "0.0.0.0")),
        "src_port": int(getattr(flow, "src_port", 0)),
        "dst_ip": str(getattr(flow, "dst_ip", "0.0.0.0")),
        "dst_port": int(getattr(flow, "dst_port", 0)),
        "protocol": int(getattr(flow, "protocol", 0)),
        "ip_version": int(getattr(flow, "ip_version", 4)),
        "duration_ms": round(float(getattr(flow, "bidirectional_duration_ms", 0.0)), 2),
        "total_packets": int(getattr(flow, "bidirectional_packets", 0)),
        "total_bytes": int(getattr(flow, "bidirectional_bytes", 0)),
        "application_name": str(getattr(flow, "application_name", "Unknown")),
        "application_category": str(getattr(flow, "application_category_name", "Unknown")),
        "first_seen_ms": int(getattr(flow, "bidirectional_first_seen_ms", 0)),
        "last_seen_ms": int(getattr(flow, "bidirectional_last_seen_ms", 0)),
    }


# -----------------------------------------------------------------------------
# Live Capture Runner
# -----------------------------------------------------------------------------
def run_live_capture(
    interface: str,
    duration: float = 10.0,
    output_path: Optional[Union[str, Path]] = None,
    bpf_filter: Optional[str] = None,
    model_path: Union[str, Path] = MODELS_DIR / "gtae_ids.pt",
    preprocessor_path: Union[str, Path] = MODELS_DIR / "preprocessor.joblib",
    device: str = GPU_CONFIG["device"],
    idle_timeout: int = 1,
    active_timeout: int = 1,
) -> Dict[str, Any]:
    """
    Executes real-time network traffic capture for a finite duration and runs
    end-to-end inference and risk scoring on all completed flows.

    Args:
        interface: Windows interface device path or resolved name.
        duration: Duration in seconds to capture (finite limit).
        output_path: Destination JSON filepath to write results.
        bpf_filter: Optional BPF filter string.
        model_path: Path to trained GTAE-IDS PyTorch checkpoint.
        preprocessor_path: Path to fitted Preprocessor joblib artifact.
        device: 'cpu' or 'cuda'.
        idle_timeout: Flow idle expiry timeout in seconds.
        active_timeout: Flow active expiry timeout in seconds.

    Returns:
        Complete capture result dictionary containing summary and flow records.
    """
    print(f"\n[+] Initializing XAI-NIDS live capture on interface: {interface}")
    print(f"[+] Capture duration: {duration:.1f} seconds | BPF: {bpf_filter or 'None'}")

    # 1. Initialize Inference API
    inference_api = InferenceAPI(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        device=device,
    )

    # 2. Start Timed NFStreamer with dynamic CIC feature measurement plugin
    streamer = TimedNFStreamer(
        source=interface,
        duration=duration,
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
        statistical_analysis=True,
        bpf_filter=bpf_filter,
        udps=[CICFeaturePlugin()],
        n_meters=1,
    )

    captured_flows: List[Any] = []
    flow_meta_list: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, float]] = []

    start_time_stamp = datetime.now().isoformat()
    t_start = time.time()
    print(f"[+] Listening for flows for {duration:.1f} seconds... (Press Ctrl+C to stop early)\n")

    try:
        for flow in streamer:
            captured_flows.append(flow)
            flow_meta = extract_flow_metadata(flow)
            features = map_nflow_to_cic_features(flow)
            flow_meta_list.append(flow_meta)
            feature_rows.append(features)
            print(
                f"    -> Captured flow: {flow_meta['src_ip']}:{flow_meta['src_port']} "
                f"--> {flow_meta['dst_ip']}:{flow_meta['dst_port']} "
                f"({flow_meta['application_name']}, {flow_meta['total_bytes']} bytes)"
            )
    except KeyboardInterrupt:
        print("\n[!] Capture interrupted by user.")

    actual_duration = round(time.time() - t_start, 2)
    total_flows = len(captured_flows)
    print(f"\n[+] Capture finished: {total_flows} flows completed in {actual_duration:.2f} seconds.")

    # 3. Run Inference if any flows were captured
    inferred_flows: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}
    detected_type_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}

    if total_flows > 0:
        print("[+] Running GTAE-IDS detection & risk scoring pipeline...")
        df_features = pd.DataFrame(feature_rows)[CIC_67_FEATURES]
        detection_results = inference_api.predict(df_features)

        for meta, res in zip(flow_meta_list, detection_results):
            record = {
                "flow_info": meta,
                "predicted_category": res["category"],
                "detected_type": res["detected_type"],
                "classifier_confidence": round(float(res["classifier_prob"]), 4),
                "anomaly_score": round(float(res["anomaly_score"]), 4),
                "risk_score": round(float(res["risk_score"]), 2),
                "severity": res["severity"],
                "is_anomalous": bool(res["is_anomalous"]),
                "detection_status": "ALERT" if res["category"] != "BENIGN" else "NORMAL",
            }
            inferred_flows.append(record)

        category_counts = dict(Counter(r["predicted_category"] for r in inferred_flows))
        detected_type_counts = dict(Counter(r["detected_type"] for r in inferred_flows))
        severity_counts = dict(Counter(r["severity"] for r in inferred_flows))
    else:
        print("[i] No flows observed on the selected interface during the capture window.")

    # 4. Construct Summary and Output
    summary: Dict[str, Any] = {
        "timestamp": start_time_stamp,
        "interface": interface,
        "configured_duration_seconds": duration,
        "actual_duration_seconds": actual_duration,
        "total_flows": total_flows,
        "category_counts": category_counts,
        "detected_type_counts": detected_type_counts,
        "severity_counts": severity_counts,
        "alerts_count": sum(c for k, c in category_counts.items() if k != "BENIGN"),
        "unavailable_cic_features": list(UNAVAILABLE_CIC_FEATURES.keys()),
        "feature_mapping_limitations": UNAVAILABLE_CIC_FEATURES,
    }

    full_results: Dict[str, Any] = {
        "metadata": {
            "module": "XAI-NIDS Live Capture",
            "version": "1.0",
            "model_path": str(model_path),
            "preprocessor_path": str(preprocessor_path),
        },
        "summary": summary,
        "flows": inferred_flows,
    }

    # 5. Persist Results
    if output_path is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = LIVE_CAPTURE_DIR / f"capture_{timestamp_str}.json"
        summary_file = LIVE_CAPTURE_DIR / f"summary_{timestamp_str}.json"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file = output_file.parent / f"{output_file.stem}_summary.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Results successfully saved to:")
    print(f"    - Full Flow Data: {output_file}")
    print(f"    - Concise Summary: {summary_file}")
    print(f"\n--- Capture Summary ---")
    print(f"Total Flows : {total_flows}")
    print(f"Categories  : {category_counts}")
    print(f"Severities  : {severity_counts}")
    print(f"Alerts      : {summary['alerts_count']}")
    print("-----------------------")

    return full_results


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="XAI-NIDS Real-Time Traffic Capture & Threat Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # List all detected network interfaces:
  python tools/live_capture.py --list-interfaces

  # Capture on Wi-Fi for 10 seconds:
  python tools/live_capture.py --interface "Wi-Fi" --duration 10

  # Capture on interface index 0 with custom output:
  python tools/live_capture.py --interface 0 --duration 15 --output results/live_capture/test.json
""",
    )

    parser.add_argument(
        "--interface",
        "-i",
        type=str,
        default=None,
        help="Interface index (e.g. 0, 1), name substring (e.g. 'Wi-Fi'), or device name",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="Capture duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save output JSON results (default: results/live_capture/capture_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--bpf",
        type=str,
        default=None,
        help="Optional BPF capture filter (e.g. 'tcp', 'port 80')",
    )
    parser.add_argument(
        "--list-interfaces",
        "-l",
        action="store_true",
        help="List available network interfaces and exit",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / "gtae_ids.pt"),
        help="Path to trained GTAE-IDS PyTorch checkpoint",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=str,
        default=str(MODELS_DIR / "preprocessor.joblib"),
        help="Path to fitted Preprocessor artifact",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=GPU_CONFIG["device"],
        help="Device to run inference on ('cpu' or 'cuda')",
    )

    args = parser.parse_args()

    # If --list-interfaces requested or --interface omitted
    if args.list_interfaces or args.interface is None:
        print("\nAvailable Network Interfaces:")
        print("=" * 80)
        interfaces = get_available_interfaces()
        if not interfaces:
            print("No interfaces detected. Please verify Npcap service is running.")
            sys.exit(1)

        for iface in interfaces:
            ip_str = ", ".join(iface["ips"]) if iface["ips"] else "No IPv4"
            print(f"[{iface['index']}] {iface['description']}")
            print(f"    Device : {iface['name']}")
            print(f"    IPv4   : {ip_str}\n")
        print("=" * 80)

        if args.interface is None and not args.list_interfaces:
            print("To capture traffic, specify an interface using --interface <INDEX or NAME>.")
            print("Example: python tools/live_capture.py --interface 0 --duration 10\n")
            sys.exit(0)
        return

    # Resolve interface
    resolved_iface = resolve_interface(args.interface)

    # Run capture
    run_live_capture(
        interface=resolved_iface,
        duration=args.duration,
        output_path=args.output,
        bpf_filter=args.bpf,
        model_path=args.model_path,
        preprocessor_path=args.preprocessor_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
