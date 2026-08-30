# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
Reward functions for TWT WiFi scheduling RL environment.

ALL 15 METRICS USED - organized into 5 logical components.

METRIC CORRELATION INSIGHTS (from EDA):
- bsr_queue_index ≈ queue_size_bytes (both 111 unique values) → merged into QUEUE
- duty_cycle = awake / (awake + sleep) → redundant, use sleep ratio directly
- airtime correlates with bytes_transmitted → merged into THROUGHPUT
- fcs_error_count, rx_fragment_count → merged into CHANNEL QUALITY
- packets_transmitted, packets_enqueued → drain ratio in QUEUE

5 COMPONENTS (no metric left behind):
1. THROUGHPUT: bytes_transmitted, packets_transmitted, airtime_used
2. QUEUE: queue_size_bytes/packets, bsr_queue_index, packets_enqueued, drain ratio
3. ENERGY: energy_mj, awake_time, sleep_time, duty_cycle
4. DROPS: drops_expired (with fairness analysis)
5. CHANNEL: fcs_error_count, rx_fragment_count, last_rx_timestamp

NORMALIZATION: From metric_quality_analysis.json (per-step = cumulative/38)
"""

from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import json
import os

# --- SCHEDULE TABLE - Load TWT schedule durations for airtime penalty ---


def _load_schedule_table() -> Dict[int, int]:
    """Load schedule table and return mapping of schedule_idx -> total_duration_ms."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schedule_path = os.path.join(
        script_dir, "..", "exploration-scripts", "schedule_table.json"
    )

    try:
        with open(schedule_path, "r") as f:
            data = json.load(f)
        schedules = data.get("schedules", [])
        return {s["schedule_id"]: s["total_duration_ms"] for s in schedules}
    except Exception as e:
        print(f"Warning: Could not load schedule table: {e}")
        # Default: assume 90ms for all schedules
        return {i: 90 for i in range(20)}


# Global schedule duration lookup
SCHEDULE_DURATIONS = _load_schedule_table()
BEACON_INTERVAL_MS = 102.4  # Standard beacon interval


# --- NORMALIZATION CONSTANTS ---
# Updated from training data (ppo_twt_20260127_182725, 249 steps).
# Original source: metric_quality_analysis.json.
# Updated 2026-01-29 based on training run analysis.

NORM = {
    # === THROUGHPUT METRICS ===
    "delta_bytes_transmitted": {"mean": 98306.0, "std": 34883.0, "max": 143865.0},
    "delta_packets_transmitted": {"mean": 78.9, "std": 26.9, "max": 120.9},
    "delta_airtime_used_us": {"mean": 15999.0, "std": 5650.0, "max": 23100.0},
    # === QUEUE METRICS ===
    "queue_size_bytes": {"mean": 23470.0, "std": 5916.0, "max": 38944.0},
    "queue_size_packets": {"mean": 16.3, "std": 4.2, "max": 27.4},
    "bsr_queue_index": {"mean": 91.7, "std": 23.1, "max": 152.1},
    "delta_packets_enqueued": {
        "mean": 85.0,
        "std": 29.0,
        "max": 130.0,
    },  # Estimated from drain ratio
    # === ENERGY METRICS ===
    "delta_energy_mj": {"mean": 35.5, "std": 16.7, "max": 80.9},
    "delta_awake_time_ms": {"mean": 64.5, "std": 38.2, "max": 175.4},  # Kept from prior
    "delta_sleep_time_ms": {
        "mean": 211.5,
        "std": 107.0,
        "max": 423.8,
    },  # Kept from prior
    "duty_cycle": {
        "mean": 0.30,
        "std": 0.06,
        "max": 0.60,
    },  # Updated: empirical avg across presets
    # === DROP METRICS ===
    "delta_drops_expired": {"mean": 13.1, "std": 13.5, "max": 78.7},
    # === CHANNEL METRICS ===
    "fcs_error_count": {"mean": 114.0, "std": 55.4, "max": 315.8},
    "rx_fragment_count": {"mean": 9.9, "std": 4.8, "max": 26.9},
    "last_rx_timestamp_us": {
        "mean": 18694564.0,
        "std": 6298365.0,
        "max": 28160000.0,
    },  # Kept
}

# --- PREVIOUS NORM (from EDA, pre-training) - kept for reference ---
# NORM_LEGACY = {
#     # === THROUGHPUT METRICS ===
#     'delta_bytes_transmitted': {'mean': 71170.0, 'std': 79635.0, 'max': 346287.0},
#     'delta_packets_transmitted': {'mean': 58.1, 'std': 55.9, 'max': 258.6},
#     'delta_airtime_used_us': {'mean': 13909.0, 'std': 22947.0, 'max': 60000.0},
#     # === QUEUE METRICS ===
#     'queue_size_bytes': {'mean': 25776.0, 'std': 29288.0, 'max': 65024.0},
#     'queue_size_packets': {'mean': 18.0, 'std': 20.9, 'max': 46.0},
#     'bsr_queue_index': {'mean': 100.7, 'std': 114.4, 'max': 254.0},
#     'delta_packets_enqueued': {'mean': 64.5, 'std': 72.1, 'max': 293.3},
#     # === ENERGY METRICS ===
#     'delta_energy_mj': {'mean': 33.0, 'std': 65.0, 'max': 150.0},
#     'delta_awake_time_ms': {'mean': 64.5, 'std': 38.2, 'max': 175.4},
#     'delta_sleep_time_ms': {'mean': 211.5, 'std': 107.0, 'max': 423.8},
#     'duty_cycle': {'mean': 0.28, 'std': 0.10, 'max': 0.50},
#     # === DROP METRICS ===
#     'delta_drops_expired': {'mean': 12.0, 'std': 20.7, 'max': 122.0},
#     # === CHANNEL METRICS ===
#     'fcs_error_count': {'mean': 102.0, 'std': 190.0, 'max': 500.0},
#     'rx_fragment_count': {'mean': 7.5, 'std': 12.0, 'max': 73.0},
#     'last_rx_timestamp_us': {'mean': 18694564.0, 'std': 6298365.0, 'max': 28160000.0},
# }

# Derived constants (updated based on new means)
EXPECTED_SLEEP_RATIO = 0.70  # Updated to match duty_cycle=0.30
EXPECTED_DRAIN_RATIO = 0.928  # packets_tx / packets_enqueued = 78.9 / 85.0
NUM_STA = 16


# --- PRESET WEIGHTS - 5 components, all sum to 1.0 ---

PRESET_WEIGHTS = {
    "throughput": {
        "throughput": 0.35,  # PRIMARY
        "queue": 0.20,  # SECONDARY (queue affects throughput)
        "drops": 0.15,  # SECONDARY (drops = lost throughput)
        "energy": 0.10,  # TERTIARY
        "airtime": 0.15,  # NEW: Penalize long TWT schedules
        "channel": 0.05,  # TERTIARY
    },
    "energy": {
        "energy": 0.35,  # PRIMARY
        "throughput": 0.20,  # SECONDARY (need some throughput)
        "drops": 0.10,  # SECONDARY (retx waste energy)
        "queue": 0.10,  # TERTIARY
        "airtime": 0.20,  # NEW: Shorter schedules save energy
        "channel": 0.05,  # TERTIARY
    },
    "queue": {
        "queue": 0.35,  # PRIMARY (latency proxy)
        "drops": 0.20,  # SECONDARY (drops = queue overflow)
        "throughput": 0.20,  # SECONDARY (need TX to drain)
        "energy": 0.10,  # TERTIARY
        "airtime": 0.10,  # NEW: Shorter schedules = lower latency
        "channel": 0.05,  # TERTIARY
    },
}


# --- HELPER FUNCTIONS ---


def get_sta_values(sta_deltas: List[Dict[str, float]], key: str) -> np.ndarray:
    """Extract a metric across all STAs."""
    return np.array([sta.get(key, 0.0) for sta in sta_deltas])


def jains_fairness(values: np.ndarray) -> float:
    """Jain's fairness index [0, 1]. 1.0 = perfect fairness."""
    if len(values) == 0:
        return 1.0
    sum_x = np.sum(values)
    if sum_x == 0:
        return 1.0
    sum_x2 = np.sum(values**2)
    if sum_x2 == 0:
        return 1.0
    return (sum_x**2) / (len(values) * sum_x2)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division."""
    return a / b if b > 0 else default


def normalize(value: float, metric_name: str) -> float:
    """
    Normalize a value to ~[0, 1] range using mean from NORM constants.
    This is just initial scaling so metrics contribute proportionally.
    """
    if metric_name not in NORM:
        return value
    mean = NORM[metric_name]["mean"]
    if mean <= 0:
        return value
    return value / mean


def soft_clip(x: float, scale: float = 1.0) -> float:
    """
    Soft saturation using tanh to avoid hard clipping.
    Maps x ∈ (-∞, +∞) → (-scale, +scale) smoothly.
    - x = 0 → 0
    - x = ±1 → ±0.76 * scale
    - x = ±2 → ±0.96 * scale
    - x → ±∞ → ±scale (asymptotically)
    """
    return np.tanh(x) * scale


def zscore(value: float, metric_name: str) -> float:
    """
    Compute z-score: (value - mean) / std.
    Returns 0 if metric not found or std is 0.
    """
    if metric_name not in NORM:
        return 0.0
    mean = NORM[metric_name]["mean"]
    std = NORM[metric_name]["std"]
    if std <= 0:
        return 0.0
    return (value - mean) / std


def zscore_reward(
    value: float, metric_name: str, scale: float = 1.0, invert: bool = False
) -> float:
    """
    Convert raw value to reward using z-score normalization + soft clipping.

    - value at mean → 0
    - value 1 std above mean → tanh(1) * scale ≈ 0.76 * scale
    - value 2 std above mean → tanh(2) * scale ≈ 0.96 * scale

    Args:
        value: Raw metric value
        metric_name: Key in NORM dict
        scale: Max reward magnitude
        invert: If True, lower values are better (e.g., energy, drops, queue)
    """
    z = zscore(value, metric_name)
    if invert:
        z = -z  # Lower value → positive reward
    return soft_clip(z, scale)


def ratio_reward(
    value: float, expected: float, std: float, scale: float = 1.0
) -> float:
    """
    Convert ratio to reward with proper z-score normalization.

    Args:
        value: Actual value (e.g., total_bytes)
        expected: Expected value (e.g., mean * n_sta)
        std: Standard deviation for normalization
        scale: Max reward magnitude
    """
    if std <= 0:
        return 0.0
    z = (value - expected) / std
    return soft_clip(z, scale)


def inverse_ratio_reward(
    value: float, expected: float, std: float, scale: float = 1.0
) -> float:
    """
    Inverse reward with proper z-score normalization (lower value = better).
    """
    if std <= 0:
        return 0.0
    z = (value - expected) / std
    return soft_clip(-z, scale)  # Invert: lower is better


# --- COMPONENT 1: THROUGHPUT ---
# Metrics: delta_bytes_transmitted, delta_packets_transmitted, delta_airtime_used_us


def compute_throughput_component(
    sta_deltas: List[Dict[str, float]], n_sta: int
) -> Dict:
    """
    Throughput: Maximize bytes transmitted efficiently.

    Uses:
    - delta_bytes_transmitted: Primary throughput measure
    - delta_packets_transmitted: Packet count
    - delta_airtime_used_us: Efficiency = bytes/airtime
    """
    bytes_tx = get_sta_values(sta_deltas, "delta_bytes_transmitted")
    packets_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")
    airtime = get_sta_values(sta_deltas, "delta_airtime_used_us")

    total_bytes = np.sum(bytes_tx)
    total_packets = np.sum(packets_tx)
    total_airtime = np.sum(airtime)

    # --- Bytes throughput reward (z-score normalized) ---
    # For aggregate: expected = mean * n_sta, std scales by sqrt(n_sta) for sum
    expected_bytes = NORM["delta_bytes_transmitted"]["mean"] * n_sta
    std_bytes = NORM["delta_bytes_transmitted"]["std"] * np.sqrt(n_sta)
    bytes_reward = ratio_reward(total_bytes, expected_bytes, std_bytes, scale=0.4)

    # --- Packets throughput reward ---
    expected_packets = NORM["delta_packets_transmitted"]["mean"] * n_sta
    std_packets = NORM["delta_packets_transmitted"]["std"] * np.sqrt(n_sta)
    packets_reward = ratio_reward(
        total_packets, expected_packets, std_packets, scale=0.15
    )

    # --- Airtime efficiency (bytes per microsecond) ---
    if total_airtime > 0:
        efficiency = total_bytes / total_airtime
        expected_eff = NORM["delta_bytes_transmitted"]["mean"] / (
            NORM["delta_airtime_used_us"]["mean"] + 1
        )
        # Estimate std for efficiency ratio (approximation)
        std_eff = expected_eff * 0.5  # ~50% variation expected
        eff_reward = ratio_reward(efficiency, expected_eff, std_eff, scale=0.15)
    else:
        eff_reward = -0.1 if total_bytes == 0 else 0.0

    # --- Fairness across STAs (Jain's index: 0-1, typical ~0.7-0.9) ---
    fairness = jains_fairness(bytes_tx)
    # Fairness std is ~0.1, center at 0.7
    fairness_bonus = soft_clip((fairness - 0.7) / 0.1, scale=0.1)

    # --- Starvation penalty (count-based, max 16) ---
    starvation_thresh = NORM["delta_bytes_transmitted"]["mean"] * 0.1
    starving = int(np.sum(bytes_tx < starvation_thresh))
    # Normalize by n_sta: 0 starving=0, all starving → -1
    starvation_penalty = soft_clip(-starving / (n_sta * 0.25), scale=0.2)

    reward = (
        bytes_reward + packets_reward + eff_reward + fairness_bonus + starvation_penalty
    )

    return {
        "reward": soft_clip(reward, scale=1.0),
        "total_bytes": total_bytes,
        "total_packets": total_packets,
        "total_airtime": total_airtime,
        "efficiency": safe_div(total_bytes, total_airtime),
        "fairness": fairness,
        "starving_stas": starving,
        "bytes_reward": bytes_reward,
        "packets_reward": packets_reward,
        "eff_reward": eff_reward,
        "fairness_bonus": fairness_bonus,
    }


# --- COMPONENT 2: QUEUE (Latency Proxy) ---
# Metrics: queue_size_bytes, queue_size_packets, bsr_queue_index, delta_packets_enqueued, delta_packets_transmitted


def compute_queue_component(sta_deltas: List[Dict[str, float]], n_sta: int) -> Dict:
    """
    Queue: Minimize queue buildup (latency proxy).

    Uses:
    - queue_size_bytes: Primary queue measure
    - queue_size_packets: Secondary (validates bytes)
    - bsr_queue_index: BSR report (0-254, correlates with queue)
    - delta_packets_enqueued + delta_packets_transmitted: Drain ratio
    """
    queue_bytes = get_sta_values(sta_deltas, "queue_size_bytes")
    queue_packets = get_sta_values(sta_deltas, "queue_size_packets")
    bsr_index = get_sta_values(sta_deltas, "bsr_queue_index")
    packets_enq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    packets_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")

    avg_queue = np.mean(queue_bytes)
    max_queue = np.max(queue_bytes) if len(queue_bytes) > 0 else 0
    avg_queue_packets = np.mean(queue_packets)

    # --- Queue level reward (lower is better, z-score normalized) ---
    # For average: std scales by 1/sqrt(n_sta)
    expected_queue = NORM["queue_size_bytes"]["mean"]
    std_queue = NORM["queue_size_bytes"]["std"] / np.sqrt(n_sta)
    queue_reward = inverse_ratio_reward(
        avg_queue, expected_queue, std_queue, scale=0.35
    )

    # --- Queue packets reward (secondary validation) ---
    expected_packets = NORM["queue_size_packets"]["mean"]
    std_packets = NORM["queue_size_packets"]["std"] / np.sqrt(n_sta)
    packets_reward = inverse_ratio_reward(
        avg_queue_packets, expected_packets, std_packets, scale=0.05
    )

    # --- BSR consistency check (should correlate with queue) ---
    avg_bsr = np.mean(bsr_index)
    expected_bsr = NORM["bsr_queue_index"]["mean"]
    std_bsr = NORM["bsr_queue_index"]["std"] / np.sqrt(n_sta)
    bsr_reward = inverse_ratio_reward(avg_bsr, expected_bsr, std_bsr, scale=0.1)

    # --- Drain ratio (TX vs enqueue) ---
    total_enq = np.sum(packets_enq)
    total_tx = np.sum(packets_tx)
    if total_enq > 0:
        drain_ratio = total_tx / total_enq
        # Typical drain ratio ~0.9, std ~0.1
        drain_reward = soft_clip((drain_ratio - 1.0) / 0.2, scale=0.3)
    else:
        drain_reward = 0.1 if total_tx > 0 else 0.0

    # --- Worst-case STA penalty ---
    # Use std for normalization
    if max_queue > expected_queue + NORM["queue_size_bytes"]["std"]:
        excess_z = (max_queue - expected_queue) / NORM["queue_size_bytes"]["std"]
        max_penalty = soft_clip(-excess_z / 2, scale=0.3)  # /2 to be less aggressive
    else:
        max_penalty = 0.0

    # --- High-queue STA count ---
    high_thresh = expected_queue + NORM["queue_size_bytes"]["std"]
    high_queue_stas = int(np.sum(queue_bytes > high_thresh))
    # Normalize: 0 = good, all 16 high = bad
    high_queue_penalty = soft_clip(-high_queue_stas / (n_sta * 0.25), scale=0.2)

    reward = (
        queue_reward
        + packets_reward
        + bsr_reward
        + drain_reward
        + max_penalty
        + high_queue_penalty
    )

    return {
        "reward": soft_clip(reward, scale=1.0),
        "avg_queue_bytes": avg_queue,
        "avg_queue_packets": avg_queue_packets,
        "max_queue_bytes": max_queue,
        "avg_bsr_index": avg_bsr,
        "drain_ratio": safe_div(total_tx, total_enq),
        "high_queue_stas": high_queue_stas,
        "queue_reward": queue_reward,
        "packets_reward": packets_reward,
        "bsr_reward": bsr_reward,
        "drain_reward": drain_reward,
        "max_penalty": max_penalty,
    }


# --- COMPONENT 3: ENERGY ---
# Metrics: delta_energy_mj, delta_awake_time_ms, delta_sleep_time_ms, duty_cycle


def compute_energy_component(sta_deltas: List[Dict[str, float]], n_sta: int) -> Dict:
    """
    Energy: Minimize consumption, maximize sleep.

    Uses:
    - delta_energy_mj: Direct energy consumption
    - delta_awake_time_ms, delta_sleep_time_ms: Sleep efficiency
    - duty_cycle: Verification (should = awake/(awake+sleep))
    """
    energy = get_sta_values(sta_deltas, "delta_energy_mj")
    awake = get_sta_values(sta_deltas, "delta_awake_time_ms")
    sleep = get_sta_values(sta_deltas, "delta_sleep_time_ms")
    duty = get_sta_values(sta_deltas, "duty_cycle")

    total_energy = np.sum(energy)

    # --- Sleep ratio per STA ---
    total_time = awake + sleep + 1e-8  # Epsilon to avoid divide by zero
    sleep_ratios = sleep / total_time
    avg_sleep_ratio = np.mean(sleep_ratios)
    min_sleep_ratio = np.min(sleep_ratios) if len(sleep_ratios) > 0 else 0

    # --- Sleep efficiency reward (z-score: std ~0.1 for ratio) ---
    sleep_reward = soft_clip((avg_sleep_ratio - EXPECTED_SLEEP_RATIO) / 0.1, scale=0.4)

    # --- Energy consumption penalty (z-score normalized) ---
    expected_energy = NORM["delta_energy_mj"]["mean"] * n_sta
    std_energy = NORM["delta_energy_mj"]["std"] * np.sqrt(n_sta)
    energy_reward = inverse_ratio_reward(
        total_energy, expected_energy, std_energy, scale=0.25
    )

    # --- Duty cycle consistency (lower = more sleep = better, z-score) ---
    avg_duty = np.mean(duty)
    std_duty = NORM["duty_cycle"]["std"] / np.sqrt(n_sta)
    duty_reward = soft_clip(
        (NORM["duty_cycle"]["mean"] - avg_duty) / std_duty, scale=0.1
    )

    # --- Sleep fairness (Jain's: 0-1, typical ~0.8-0.95) ---
    sleep_fairness = jains_fairness(sleep)
    fairness_bonus = soft_clip((sleep_fairness - 0.8) / 0.1, scale=0.1)

    # --- Worst-case STA (low sleep, z-score: expect ~0.1 std for ratio) ---
    if min_sleep_ratio < EXPECTED_SLEEP_RATIO:
        worst_z = (EXPECTED_SLEEP_RATIO - min_sleep_ratio) / 0.15
        worst_penalty = soft_clip(-worst_z, scale=0.15)
    else:
        worst_penalty = 0.0

    reward = sleep_reward + energy_reward + duty_reward + fairness_bonus + worst_penalty

    return {
        "reward": soft_clip(reward, scale=1.0),
        "total_energy_mj": total_energy,
        "avg_sleep_ratio": float(avg_sleep_ratio),
        "min_sleep_ratio": float(min_sleep_ratio),
        "avg_duty_cycle": float(avg_duty),
        "sleep_fairness": sleep_fairness,
        "sleep_reward": sleep_reward,
        "energy_reward": energy_reward,
        "duty_reward": duty_reward,
        "worst_penalty": worst_penalty,
    }


# --- COMPONENT 4: DROPS ---
# Metrics: delta_drops_expired


def compute_drops_component(sta_deltas: List[Dict[str, float]], n_sta: int) -> Dict:
    """
    Drops: Minimize packet drops.

    Uses:
    - delta_drops_expired: Primary drop measure
    - Per-STA analysis for worst-case and fairness
    """
    drops = get_sta_values(sta_deltas, "delta_drops_expired")

    total_drops = np.sum(drops)
    max_drops = np.max(drops) if len(drops) > 0 else 0

    # --- Drop penalty (lower is better, z-score normalized) ---
    expected_drops = NORM["delta_drops_expired"]["mean"] * n_sta
    std_drops = NORM["delta_drops_expired"]["std"] * np.sqrt(n_sta)
    drop_reward = inverse_ratio_reward(
        total_drops, expected_drops, std_drops, scale=0.5
    )

    # --- Worst-case STA (z-score normalized) ---
    if max_drops > NORM["delta_drops_expired"]["mean"]:
        worst_z = (max_drops - NORM["delta_drops_expired"]["mean"]) / NORM[
            "delta_drops_expired"
        ]["std"]
        worst_penalty = soft_clip(-worst_z / 2, scale=0.25)  # /2 to be less aggressive
    else:
        worst_penalty = 0.0

    # --- Drop fairness (Jain's: 0-1, typical ~0.7-0.9) ---
    if total_drops > 0:
        drop_fairness = jains_fairness(drops + 1)  # +1 to avoid all-zero issues
        fairness_penalty = soft_clip((drop_fairness - 0.8) / 0.1, scale=0.1)
    else:
        fairness_penalty = 0.1  # Bonus for no drops

    # --- STAs with high drops (count-based, normalized) ---
    high_drop_thresh = (
        NORM["delta_drops_expired"]["mean"] + NORM["delta_drops_expired"]["std"]
    )
    high_drop_stas = int(np.sum(drops > high_drop_thresh))
    high_drop_penalty = soft_clip(-high_drop_stas / (n_sta * 0.25), scale=0.15)

    reward = drop_reward + worst_penalty + fairness_penalty + high_drop_penalty

    return {
        "reward": soft_clip(reward, scale=1.0),
        "total_drops": total_drops,
        "max_drops": max_drops,
        "high_drop_stas": high_drop_stas,
        "drop_reward": drop_reward,
        "worst_penalty": worst_penalty,
        "fairness_penalty": fairness_penalty,
    }


# --- COMPONENT 5: CHANNEL QUALITY ---
# Metrics: fcs_error_count, rx_fragment_count, last_rx_timestamp_us


def compute_channel_component(sta_deltas: List[Dict[str, float]], n_sta: int) -> Dict:
    """
    Channel quality: Monitor errors and activity.

    Uses:
    - fcs_error_count: FCS errors (lower = better channel)
    - rx_fragment_count: RX activity (more = active link)
    - last_rx_timestamp_us: Recency (recent = active)
    """
    fcs = get_sta_values(sta_deltas, "fcs_error_count")
    rx_frags = get_sta_values(sta_deltas, "rx_fragment_count")
    rx_ts = get_sta_values(sta_deltas, "last_rx_timestamp_us")

    avg_fcs = np.mean(fcs)
    avg_rx = np.mean(rx_frags)

    # --- FCS error penalty (lower is better, z-score normalized) ---
    expected_fcs = NORM["fcs_error_count"]["mean"]
    std_fcs = NORM["fcs_error_count"]["std"] / np.sqrt(n_sta)
    fcs_reward = inverse_ratio_reward(avg_fcs, expected_fcs, std_fcs, scale=0.4)

    # --- RX activity bonus (more fragments = active link, z-score) ---
    expected_rx = NORM["rx_fragment_count"]["mean"]
    std_rx = NORM["rx_fragment_count"]["std"] / np.sqrt(n_sta)
    rx_reward = ratio_reward(avg_rx, expected_rx, std_rx, scale=0.3)

    # --- Recency (z-score normalized) ---
    avg_recency = np.mean(rx_ts)
    expected_recency = NORM["last_rx_timestamp_us"]["mean"]
    std_recency = NORM["last_rx_timestamp_us"]["std"] / np.sqrt(n_sta)
    recency_reward = ratio_reward(avg_recency, expected_recency, std_recency, scale=0.1)

    # --- Inactive STAs (count-based, normalized) ---
    inactive_stas = int(np.sum(rx_frags < 1))
    inactive_penalty = soft_clip(-inactive_stas / (n_sta * 0.25), scale=0.1)

    reward = fcs_reward + rx_reward + recency_reward + inactive_penalty

    return {
        "reward": soft_clip(reward, scale=1.0),
        "avg_fcs_errors": avg_fcs,
        "avg_rx_fragments": avg_rx,
        "inactive_stas": inactive_stas,
        "fcs_reward": fcs_reward,
        "rx_reward": rx_reward,
        "recency_reward": recency_reward,
    }


# --- COMPONENT 6: AIRTIME (Schedule Duration Efficiency) ---
# Penalize long TWT schedules - encourage efficient use of beacon interval.


def compute_airtime_component(schedule_idx: Optional[int], n_sta: int) -> Dict:
    """
    Airtime efficiency: Penalize long TWT schedule durations.

    Shorter schedules are better because:
    - More time available for other traffic (non-TWT STAs, management)
    - Lower latency for responsive scheduling
    - Energy efficiency (less total wake time)

    Args:
        schedule_idx: Index into SCHEDULE_DURATIONS (0-19)
        n_sta: Number of STAs (unused but kept for consistency)

    Returns:
        Dict with reward and metrics
    """
    if schedule_idx is None:
        # No action provided, return neutral reward
        return {
            "reward": 0.0,
            "duration_ms": 0,
            "utilization": 0.0,
            "duration_reward": 0.0,
        }

    # Get schedule duration
    duration_ms = SCHEDULE_DURATIONS.get(schedule_idx, 90)

    # Calculate beacon interval utilization (lower is better)
    # utilization = duration / beacon_interval
    utilization = duration_ms / BEACON_INTERVAL_MS

    # --- Duration reward: Always push towards 0ms ---
    # Linear reward: 0ms → +1.0, 90ms → 0.0
    # The shorter the better, no neutral point
    max_duration = 90.0
    # Normalized: 0ms=1.0, 90ms=0.0
    duration_normalized = 1.0 - (duration_ms / max_duration)
    # Scale and shift so 90ms gets slight penalty
    # 0ms → +0.7, 20ms → +0.54, 40ms → +0.38, 60ms → +0.22, 90ms → -0.1
    duration_reward = soft_clip(duration_normalized * 1.5 - 0.35, scale=0.8)

    return {
        "reward": duration_reward,
        "duration_ms": duration_ms,
        "utilization": utilization,
        "duration_reward": duration_reward,
    }


# --- MAIN REWARD FUNCTION ---


def compute_reward(
    sta_deltas: List[Dict[str, float]],
    preset: str = "throughput",
    action: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Compute weighted reward using all 15 metrics in 5 components.

    Args:
        sta_deltas: List of per-STA metric dicts (15 metrics each)
        preset: 'throughput', 'energy', or 'queue'

    Returns:
        Dict with 'total' reward and 'components' breakdown
    """
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}}

    n_sta = len(sta_deltas)
    weights = PRESET_WEIGHTS.get(preset, PRESET_WEIGHTS["throughput"])

    # Compute all 6 components
    throughput = compute_throughput_component(sta_deltas, n_sta)
    queue = compute_queue_component(sta_deltas, n_sta)
    energy = compute_energy_component(sta_deltas, n_sta)
    drops = compute_drops_component(sta_deltas, n_sta)
    channel = compute_channel_component(sta_deltas, n_sta)

    # Airtime component - uses action's schedule_idx
    schedule_idx = action[0] if action is not None else None
    airtime = compute_airtime_component(schedule_idx, n_sta)

    # Weighted sum
    w_throughput = weights["throughput"] * throughput["reward"]
    w_queue = weights["queue"] * queue["reward"]
    w_energy = weights["energy"] * energy["reward"]
    w_drops = weights["drops"] * drops["reward"]
    w_channel = weights["channel"] * channel["reward"]
    w_airtime = weights["airtime"] * airtime["reward"]

    total = w_throughput + w_queue + w_energy + w_drops + w_channel + w_airtime

    return {
        "total": total,
        "components": {
            # Weighted contributions
            "w_throughput": w_throughput,
            "w_queue": w_queue,
            "w_energy": w_energy,
            "w_drops": w_drops,
            "w_channel": w_channel,
            "w_airtime": w_airtime,
            # Raw component rewards
            "raw_throughput": throughput["reward"],
            "raw_queue": queue["reward"],
            "raw_energy": energy["reward"],
            "raw_drops": drops["reward"],
            "raw_channel": channel["reward"],
            "raw_airtime": airtime["reward"],
        },
        # Detailed breakdown for logging/plotting
        "details": {
            # Throughput sub-rewards
            "throughput_bytes_reward": throughput.get("bytes_reward", 0),
            "throughput_packets_reward": throughput.get("packets_reward", 0),
            "throughput_eff_reward": throughput.get("eff_reward", 0),
            "throughput_fairness_bonus": throughput.get("fairness_bonus", 0),
            # Queue sub-rewards
            "queue_reward": queue.get("queue_reward", 0),
            "queue_packets_reward": queue.get("packets_reward", 0),
            "queue_bsr_reward": queue.get("bsr_reward", 0),
            "queue_drain_reward": queue.get("drain_reward", 0),
            "queue_max_penalty": queue.get("max_penalty", 0),
            # Energy sub-rewards
            "energy_sleep_reward": energy.get("sleep_reward", 0),
            "energy_consumption_reward": energy.get("energy_reward", 0),
            "energy_duty_reward": energy.get("duty_reward", 0),
            "energy_worst_penalty": energy.get("worst_penalty", 0),
            # Drops sub-rewards
            "drops_reward": drops.get("drop_reward", 0),
            "drops_worst_penalty": drops.get("worst_penalty", 0),
            "drops_fairness_penalty": drops.get("fairness_penalty", 0),
            # Channel sub-rewards
            "channel_fcs_reward": channel.get("fcs_reward", 0),
            "channel_rx_reward": channel.get("rx_reward", 0),
            "channel_recency_reward": channel.get("recency_reward", 0),
            # Airtime sub-rewards
            "airtime_duration_reward": airtime.get("duration_reward", 0),
        },
        # Raw metrics for plotting
        "metrics": {
            "total_bytes": throughput["total_bytes"],
            "total_packets": throughput["total_packets"],
            "total_airtime": throughput["total_airtime"],
            "throughput_efficiency": throughput["efficiency"],
            "throughput_fairness": throughput["fairness"],
            "starving_stas": throughput["starving_stas"],
            "avg_queue_bytes": queue["avg_queue_bytes"],
            "avg_queue_packets": queue["avg_queue_packets"],
            "max_queue_bytes": queue["max_queue_bytes"],
            "avg_bsr_index": queue["avg_bsr_index"],
            "drain_ratio": queue["drain_ratio"],
            "high_queue_stas": queue["high_queue_stas"],
            "total_energy_mj": energy["total_energy_mj"],
            "avg_sleep_ratio": energy["avg_sleep_ratio"],
            "min_sleep_ratio": energy["min_sleep_ratio"],
            "avg_duty_cycle": energy["avg_duty_cycle"],
            "sleep_fairness": energy["sleep_fairness"],
            "total_drops": drops["total_drops"],
            "max_drops": drops["max_drops"],
            "high_drop_stas": drops["high_drop_stas"],
            "avg_fcs_errors": channel["avg_fcs_errors"],
            "avg_rx_fragments": channel["avg_rx_fragments"],
            "inactive_stas": channel["inactive_stas"],
            "schedule_duration_ms": airtime["duration_ms"],
            "beacon_utilization": airtime["utilization"],
        },
        # Z-scores for checking if metrics are out of expected bounds.
        # z > 2 or z < -2 indicates unusual values.
        "zscores": {
            # Throughput z-scores (sum metrics: std * sqrt(n))
            "total_bytes_z": (
                throughput["total_bytes"]
                - NORM["delta_bytes_transmitted"]["mean"] * n_sta
            )
            / (NORM["delta_bytes_transmitted"]["std"] * np.sqrt(n_sta) + 1e-8),
            "total_packets_z": (
                throughput["total_packets"]
                - NORM["delta_packets_transmitted"]["mean"] * n_sta
            )
            / (NORM["delta_packets_transmitted"]["std"] * np.sqrt(n_sta) + 1e-8),
            "total_airtime_z": (
                throughput["total_airtime"]
                - NORM["delta_airtime_used_us"]["mean"] * n_sta
            )
            / (NORM["delta_airtime_used_us"]["std"] * np.sqrt(n_sta) + 1e-8),
            # Queue z-scores (avg metrics: std / sqrt(n))
            "avg_queue_bytes_z": (
                queue["avg_queue_bytes"] - NORM["queue_size_bytes"]["mean"]
            )
            / (NORM["queue_size_bytes"]["std"] / np.sqrt(n_sta) + 1e-8),
            "avg_queue_packets_z": (
                queue["avg_queue_packets"] - NORM["queue_size_packets"]["mean"]
            )
            / (NORM["queue_size_packets"]["std"] / np.sqrt(n_sta) + 1e-8),
            "avg_bsr_index_z": (
                queue["avg_bsr_index"] - NORM["bsr_queue_index"]["mean"]
            )
            / (NORM["bsr_queue_index"]["std"] / np.sqrt(n_sta) + 1e-8),
            # Energy z-scores
            "total_energy_z": (
                energy["total_energy_mj"] - NORM["delta_energy_mj"]["mean"] * n_sta
            )
            / (NORM["delta_energy_mj"]["std"] * np.sqrt(n_sta) + 1e-8),
            "avg_duty_cycle_z": (energy["avg_duty_cycle"] - NORM["duty_cycle"]["mean"])
            / (NORM["duty_cycle"]["std"] / np.sqrt(n_sta) + 1e-8),
            # Drops z-scores
            "total_drops_z": (
                drops["total_drops"] - NORM["delta_drops_expired"]["mean"] * n_sta
            )
            / (NORM["delta_drops_expired"]["std"] * np.sqrt(n_sta) + 1e-8),
            # Channel z-scores
            "avg_fcs_errors_z": (
                channel["avg_fcs_errors"] - NORM["fcs_error_count"]["mean"]
            )
            / (NORM["fcs_error_count"]["std"] / np.sqrt(n_sta) + 1e-8),
            "avg_rx_fragments_z": (
                channel["avg_rx_fragments"] - NORM["rx_fragment_count"]["mean"]
            )
            / (NORM["rx_fragment_count"]["std"] / np.sqrt(n_sta) + 1e-8),
        },
        "weights": weights,
        "n_sta": n_sta,
    }


# --- REWARD FUNCTION CLASS ---


class RewardFunction:
    """Wrapper for reward computation."""

    def __init__(self, preset: str = "throughput"):
        if preset not in PRESET_WEIGHTS:
            raise ValueError(
                f"Unknown preset: {preset}. Use: {list(PRESET_WEIGHTS.keys())}"
            )
        self.preset = preset
        self.weights = PRESET_WEIGHTS[preset]

    def __call__(
        self,
        sta_deltas: List[Dict[str, float]],
        action: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Compute reward with optional action for airtime penalty.

        Args:
            sta_deltas: Per-STA metric deltas
            action: Optional (schedule_idx, assignment_idx) tuple for airtime penalty
        """
        return compute_reward(sta_deltas, self.preset, action=action)

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()


# --- REGISTRY ---

# --- LINEAR THROUGHPUT REWARD (diagnostic preset — no soft_clip saturation, no airtime, no energy, no queue, no drops) ---
# Tests whether PPO can find the bytes-maximizing schedule when the gradient landscape is monotonic.
#
# Per-step reward = (sum_delta_bytes_transmitted / SCALE_BYTES) - BASELINE_OFFSET
# Calibrated from diag_linear_throughput on seed 200000, 73 steps:
#   transformer (sched 4, 20ms): ~3.9 MB/step bytes_tx
#   fixed_s00   (sched 0, 90ms): ~12.4 MB/step bytes_tx
# With scale=10MB, baseline=0.8 we get:
#   transformer  -> 3.9/10 - 0.8 = -0.41 per step  (episode ~ -30)
#   fixed_s00    ->12.4/10 - 0.8 = +0.44 per step  (episode ~ +32)
# Symmetric around zero, similar magnitude to the soft-clipped presets so PPO's default advantage normalization stays well-behaved.
# No tanh / clip / penalty: the reward is strictly monotone in bytes_tx, so PPO gradient pressure pushes the policy toward the bytes-maximizing schedule.

LINEAR_THROUGHPUT_SCALE_BYTES = 10_000_000  # 10 MB per reward unit
LINEAR_THROUGHPUT_BASELINE = 0.8  # subtract 0.8 (~8 MB/step neutral)


def compute_linear_throughput_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Pure-throughput diagnostic reward. Linear in summed bytes_tx, no clip."""
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}
    bytes_tx = get_sta_values(sta_deltas, "delta_bytes_transmitted")
    total_bytes = float(np.sum(bytes_tx))
    reward = (total_bytes / LINEAR_THROUGHPUT_SCALE_BYTES) - LINEAR_THROUGHPUT_BASELINE
    return {
        "total": reward,
        "components": {
            "w_throughput": reward,
            "w_queue": 0.0,
            "w_energy": 0.0,
            "w_drops": 0.0,
            "w_channel": 0.0,
            "w_airtime": 0.0,
            "raw_throughput": reward,
            "raw_queue": 0.0,
            "raw_energy": 0.0,
            "raw_drops": 0.0,
            "raw_channel": 0.0,
            "raw_airtime": 0.0,
        },
        "details": {},
        "metrics": {"total_bytes": total_bytes},
        "weights": {
            "throughput": 1.0,
            "queue": 0.0,
            "energy": 0.0,
            "drops": 0.0,
            "channel": 0.0,
            "airtime": 0.0,
        },
        "n_sta": len(sta_deltas),
    }


class LinearThroughputRewardFunction:
    """Wrapper matching RewardFunction's interface."""

    def __init__(self):
        self.preset = "throughput_linear"
        self.weights = {
            "throughput": 1.0,
            "queue": 0.0,
            "energy": 0.0,
            "drops": 0.0,
            "channel": 0.0,
            "airtime": 0.0,
        }

    def __call__(self, sta_deltas, action=None):
        return compute_linear_throughput_reward(sta_deltas, action=action)

    def get_weights(self):
        return self.weights.copy()


# --- LINEAR THROUGHPUT + DROP PENALTY (Option B from the offline reward-design study) ---
# Validated against the 1.4M-row exploration dataset:
#   - corr(reward, total_bytes_tx)  ≈ +0.98  (rewards throughput strongly)
#   - corr(reward, total_drops)     ≈ -0.40  (penalizes drops)
# The drop penalty fixes the throughput_linear pathology where the trained policy got 95% of fixed_s00's bytes but 2× more drops.
#
# Per-step reward = (total_bytes / SCALE) - BASELINE - w_drop * mean_drop_rate
# where mean_drop_rate = mean_i min(1, drops_i / max(enqueued_i, 1))

LIN_DROP_W_DROP = 0.5  # weight: at mean_drop_rate=1.0 (everyone dropping all packets), reward gets a -0.5 penalty.
# At p90 mean_drop_rate=0.5, penalty is -0.25.
# Anchored on the offline study where this gave corr(reward, drops) = -0.40.


def compute_linear_throughput_with_drop_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Throughput_linear + mean_i(drop_rate_i) penalty."""
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}
    bytes_tx = get_sta_values(sta_deltas, "delta_bytes_transmitted")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")
    enq = get_sta_values(sta_deltas, "delta_packets_enqueued")

    total_bytes = float(np.sum(bytes_tx))
    thr = (total_bytes / LINEAR_THROUGHPUT_SCALE_BYTES) - LINEAR_THROUGHPUT_BASELINE
    # Per-STA drop rate, clipped to [0,1]; ignore STAs with zero enqueued.
    rates = np.where(enq > 0, np.minimum(1.0, drops / np.maximum(enq, 1)), 0.0)
    mean_drop_rate = float(np.mean(rates))
    drop_pen = LIN_DROP_W_DROP * mean_drop_rate
    total = thr - drop_pen

    return {
        "total": total,
        "components": {
            "w_throughput": thr,
            "w_drop": -drop_pen,
            "w_queue": 0.0,
            "w_energy": 0.0,
            "w_channel": 0.0,
            "w_airtime": 0.0,
            "raw_throughput": thr,
            "raw_drop": -drop_pen,
            "raw_drop_rate": mean_drop_rate,
            "raw_queue": 0.0,
            "raw_energy": 0.0,
            "raw_channel": 0.0,
            "raw_airtime": 0.0,
        },
        "details": {},
        "metrics": {
            "total_bytes": total_bytes,
            "total_drops": float(np.sum(drops)),
            "mean_drop_rate": mean_drop_rate,
        },
        "weights": {
            "throughput": 1.0,
            "drops": LIN_DROP_W_DROP,
            "queue": 0.0,
            "energy": 0.0,
            "channel": 0.0,
            "airtime": 0.0,
        },
        "n_sta": len(sta_deltas),
    }


class LinearThroughputWithDropRewardFunction:
    """Wrapper matching RewardFunction's interface."""

    def __init__(self):
        self.preset = "throughput_linear_with_drop"
        self.weights = {
            "throughput": 1.0,
            "drops": LIN_DROP_W_DROP,
            "queue": 0.0,
            "energy": 0.0,
            "channel": 0.0,
            "airtime": 0.0,
        }

    def __call__(self, sta_deltas, action=None):
        return compute_linear_throughput_with_drop_reward(sta_deltas, action=action)

    def get_weights(self):
        return self.weights.copy()


# --- TWT-EFFICIENT REWARD (the one that actually models TWT's purpose) ---
#
# Motivation: throughput_linear and throughput_linear_with_drop both let `fixed_s00` (90ms wake, no real TWT) win because there was no real energy cost.
# The trained transformer with throughput_linear_with_drop converged byte-for-byte to fixed_s00 — i.e., it learned NOT to use TWT.
# That's fundamentally wrong: TWT exists to save energy by letting STAs sleep.
#
# This preset bakes the energy cost in as a HINGE penalty so that:
#   - Below the hinge (duty <= 0.20), energy is free and the agent only fights for throughput / low drops.
#   - Above the hinge, the cost grows LINEARLY and quickly outstrips any marginal throughput gain; fixed_s00's duty cycle (~0.88) lies far above the hinge and pays a large penalty.
#
# Three terms only, all linear (no soft_clip), all scale-free (no NORM dependency — only protocol-level constants):
#
#   reward_per_step =
#       W_THR · (sum_bytes_tx / PEAK_PHY_PER_BI)         # throughput ratio in [0, ~1]
#     − W_ENG · max(0, mean_duty_cycle − DUTY_HINGE)     # hinge above 20% duty
#     − W_DRP · mean_i min(1, drops_i / max(enq_i, 1))   # drop rate in [0, 1]
#
# Each input is intrinsic (peak-PHY bytes per BI is a radio constant; duty cycle is awake/(awake+sleep); drop rate is drops/enqueued).
# No deployment-tuned `NORM` lookup — works the same on any traffic mix.
#
# Why this beats `balanced_pf` (which failed): the components are NOT anti-correlated.
# The throughput term grows monotonically with wake until the energy hinge clamps down.
# There is an interior optimum where state-conditional decisions actually matter.
# The agent's job: pick the moderate wake schedule that maximizes throughput while staying under the duty-cycle hinge — and decide per-state which STAs are in which group to handle their traffic.

# Protocol-level constants (NOT deployment-tuned).
# One agent "step" = TWT_UPDATE_INTERVAL_BI=25 beacon intervals = 2.56s.
# Peak PHY bytes per step = single-BI-peak × 25 ≈ 45.9 MB.
# delta_bytes_transmitted is aggregated over the same step, so dividing by this yields a ratio in [0, 1] (with some headroom for OFDMA multi-user multiplexing).
_TWT_EFF_PHY_PEAK_BYTES_PER_STEP = int(
    143.4 * 1e6 / 8 * (102.4 / 1000.0) * 25
)  # ≈ 45_888_000
_TWT_EFF_DUTY_HINGE = 0.20  # below this, energy is "free"
_TWT_EFF_W_THR = 1.0  # throughput scale: per-step thr_ratio (typical ~0.27) → +0.27
_TWT_EFF_W_ENG = 1.5  # energy scale: per-step hinge_excess (fixed_s00: 0.66) → −0.99
_TWT_EFF_W_DRP = (
    0.5  # drop scale: per-step drop_rate (typical 0.02-0.10) → −0.01..−0.05
)
# Calibrated against an actual NS-3 run of fixed_s00 (seed 200000):
#   thr_ratio  = 0.27 (12.4 MB / 45.9 MB)
#   mean_duty  = 0.86 (essentially full BI awake)
#   drop_rate  = 0.02
#   reward = +0.27 − 1.5·0.66 − 0.5·0.02 = +0.27 − 0.99 − 0.01 = −0.73 per step
# Compared to a hypothetical 8-group / 11ms / mean_duty=0.107 schedule:
#   thr_ratio ≈ 0.20 (less aggregate wake), duty_excess = 0 (below hinge)
#   reward = +0.20 − 0 − 0.025 = +0.175 per step
# So fixed_s00 is now ~−0.9/step BELOW a moderate-wake schedule.
# PPO has explicit incentive to spread STAs across groups (reducing mean_duty) while keeping enough total wake budget for throughput.


def compute_twt_efficient_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, int]] = None,
) -> Dict:
    """The reward TWT actually needs: throughput minus an energy hinge."""
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}

    bytes_tx = get_sta_values(sta_deltas, "delta_bytes_transmitted")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")
    enq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    duty = get_sta_values(sta_deltas, "duty_cycle")

    total_bytes = float(np.sum(bytes_tx))
    thr_ratio = total_bytes / max(_TWT_EFF_PHY_PEAK_BYTES_PER_STEP, 1)

    mean_duty = float(np.mean(duty))
    duty_excess = max(0.0, mean_duty - _TWT_EFF_DUTY_HINGE)

    # Per-STA drop rate, clip to [0, 1], skip STAs with no enqueued.
    rates = np.where(enq > 0, np.minimum(1.0, drops / np.maximum(enq, 1)), 0.0)
    mean_drop_rate = float(np.mean(rates))

    w_thr = _TWT_EFF_W_THR * thr_ratio  # positive
    w_eng = -_TWT_EFF_W_ENG * duty_excess  # negative (penalty)
    w_drp = -_TWT_EFF_W_DRP * mean_drop_rate  # negative (penalty)
    total = w_thr + w_eng + w_drp

    return {
        "total": total,
        "components": {
            "w_throughput": w_thr,
            "w_energy": w_eng,
            "w_drop": w_drp,
            "w_queue": 0.0,
            "w_channel": 0.0,
            "w_airtime": 0.0,
            "raw_throughput": thr_ratio,
            "raw_duty_excess": duty_excess,
            "raw_drop_rate": mean_drop_rate,
            "raw_mean_duty": mean_duty,
        },
        "details": {},
        "metrics": {
            "total_bytes": total_bytes,
            "total_drops": float(np.sum(drops)),
            "mean_duty_cycle": mean_duty,
            "mean_drop_rate": mean_drop_rate,
        },
        "weights": {
            "throughput": _TWT_EFF_W_THR,
            "energy": _TWT_EFF_W_ENG,
            "drops": _TWT_EFF_W_DRP,
            "queue": 0.0,
            "channel": 0.0,
            "airtime": 0.0,
        },
        "n_sta": len(sta_deltas),
    }


class TwtEfficientRewardFunction:
    """Wrapper matching RewardFunction's interface."""

    def __init__(self):
        self.preset = "twt_efficient"
        self.weights = {
            "throughput": _TWT_EFF_W_THR,
            "energy": _TWT_EFF_W_ENG,
            "drops": _TWT_EFF_W_DRP,
            "queue": 0.0,
            "channel": 0.0,
            "airtime": 0.0,
        }

    def __call__(self, sta_deltas, action=None):
        return compute_twt_efficient_reward(sta_deltas, action=action)

    def get_weights(self):
        return self.weights.copy()


# --- TWT_EFFICIENT_V2 — full 6-term design with airtime cost and EDA-calibrated guardrails ---
# Designed after careful calibration against the 1.4M-row EDA dataset.
# See /tmp/calibrate_twt_efficient_v2.py for the validation details.
#
# Key insight over v1: there are TWO physical costs of wake time —
#   1. PER-STA energy (mean_duty_cycle):   how long each STA is awake
#                                          → reduced by multi-group split
#   2. CHANNEL airtime (total_TWT/BI):     how much channel time TWT reserves
#                                          → reduced by shorter total wake
# Multi-group schedules reduce (1) but NOT (2).
# In unsaturated networks, the agent should also pick shorter total TWT to free channel for non-TWT traffic / coexistence.
# v1 missed this; v2 adds the airtime term.
#
# Additional changes from v1:
#   - LINEAR energy cost always paid (in addition to the hinge above 0.20)
#   - Drop hinge raised from 0.10 to 0.30 (v1's 0.10 fired on 76% of EDA transitions, was a constant cost not a guardrail)
#   - BSR fill guardrail added (penalize queue-near-full only above 0.50)
#   - No starvation term (per-STA bytes < 10% mean is the "default" in heterogeneous-traffic env; 98.6% of EDA transitions had STA starving under the old definition)
#
# Validated reward distribution under random actions across the 40-schedule EDA dataset:
#   p10=-0.087  p50=+0.095  p90=+0.300  mean=+0.101  std=0.148
# Per-policy reward estimates (per step):
#   fixed_s00                : -0.72  (energy hinge dominates)
#   smart multi-group (4g x 22ms): +0.22 (best constant)
#   short TWT (light traffic):   +0.13
#   over-sleep degenerate:      -0.03 (drop+BSR guardrails catch it)
# Hinge fire rates under random actions:
#   duty_hinge:    10.9%   drop_hinge: 27.8%   bsr_hinge: 30.8%

_TWT_V2_PHY_PEAK_BYTES_PER_STEP = int(
    143.4 * 1e6 / 8 * (102.4 / 1000.0) * 25
)  # ≈ 45_888_000
_TWT_V2_BI_MS = 102.4
_TWT_V2_DUTY_HINGE = 0.20
_TWT_V2_DROP_HINGE = 0.30  # raised from v1's 0.10 (was firing 76% of EDA)
_TWT_V2_BSR_HINGE = 0.50

# Weights (post-calibration)
_TWT_V2_W_THR = 2.0  # primary positive signal — must dominate penalties
_TWT_V2_W_ENG_LIN = 0.1  # small always-paid linear energy
_TWT_V2_W_ENG_HNG = 1.5  # heavy hinge above 0.20 duty (kills fixed_s00)
_TWT_V2_W_AIRTIME = 0.2  # channel airtime cost (per-step linear)
_TWT_V2_W_DRP = 0.3  # drop guardrail above 30% drop rate
_TWT_V2_W_BSR = 0.3  # queue overflow guardrail above 50% BSR

# Schedule total_TWT_duration_ms lookup.
# Computed once at module import.
# Uses SCHEDULE_DURATIONS already defined at the top of this file.


def compute_twt_efficient_v2_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Full 6-term TWT-efficient reward (linear bytes + airtime + duty
    hinge + drop hinge + BSR hinge). Calibrated against the 1.4M-row
    EDA. The `action` argument provides schedule_idx for the airtime
    term — if not provided, airtime defaults to 0 (neutral)."""
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}

    bytes_tx = get_sta_values(sta_deltas, "delta_bytes_transmitted")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")
    enq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    duty = get_sta_values(sta_deltas, "duty_cycle")
    bsr = get_sta_values(sta_deltas, "bsr_queue_index")

    total_bytes = float(np.sum(bytes_tx))
    thr_ratio = total_bytes / max(_TWT_V2_PHY_PEAK_BYTES_PER_STEP, 1)

    mean_duty = float(np.mean(duty))
    duty_excess = max(0.0, mean_duty - _TWT_V2_DUTY_HINGE)

    # Airtime (channel reservation) — from chosen schedule's total wake duration.
    schedule_idx = int(action[0]) if action is not None else None
    if schedule_idx is not None:
        total_TWT_ms = SCHEDULE_DURATIONS.get(schedule_idx, 0)
    else:
        total_TWT_ms = 0
    airtime_frac = total_TWT_ms / _TWT_V2_BI_MS

    rates = np.where(enq > 0, np.minimum(1.0, drops / np.maximum(enq, 1)), 0.0)
    mean_drop_rate = float(np.mean(rates))
    drop_excess = max(0.0, mean_drop_rate - _TWT_V2_DROP_HINGE)

    bsr_fills = np.minimum(1.0, bsr / 254.0)
    mean_bsr_fill = float(np.mean(bsr_fills))
    bsr_excess = max(0.0, mean_bsr_fill - _TWT_V2_BSR_HINGE)

    w_thr_term = _TWT_V2_W_THR * thr_ratio
    w_eng_lin_term = -_TWT_V2_W_ENG_LIN * mean_duty
    w_eng_hng_term = -_TWT_V2_W_ENG_HNG * duty_excess
    w_airtime_term = -_TWT_V2_W_AIRTIME * airtime_frac
    w_drp_term = -_TWT_V2_W_DRP * drop_excess
    w_bsr_term = -_TWT_V2_W_BSR * bsr_excess

    total = (
        w_thr_term
        + w_eng_lin_term
        + w_eng_hng_term
        + w_airtime_term
        + w_drp_term
        + w_bsr_term
    )

    return {
        "total": total,
        "components": {
            "w_throughput": w_thr_term,
            "w_energy_lin": w_eng_lin_term,
            "w_energy_hinge": w_eng_hng_term,
            "w_airtime": w_airtime_term,
            "w_drop": w_drp_term,
            "w_bsr": w_bsr_term,
            "raw_thr_ratio": thr_ratio,
            "raw_mean_duty": mean_duty,
            "raw_duty_excess": duty_excess,
            "raw_airtime_frac": airtime_frac,
            "raw_drop_rate": mean_drop_rate,
            "raw_drop_excess": drop_excess,
            "raw_bsr_fill": mean_bsr_fill,
            "raw_bsr_excess": bsr_excess,
        },
        "details": {},
        "metrics": {
            "total_bytes": total_bytes,
            "total_drops": float(np.sum(drops)),
            "mean_duty_cycle": mean_duty,
            "airtime_frac": airtime_frac,
            "mean_drop_rate": mean_drop_rate,
            "mean_bsr_fill": mean_bsr_fill,
        },
        "weights": {
            "throughput": _TWT_V2_W_THR,
            "energy_lin": _TWT_V2_W_ENG_LIN,
            "energy_hinge": _TWT_V2_W_ENG_HNG,
            "airtime": _TWT_V2_W_AIRTIME,
            "drop": _TWT_V2_W_DRP,
            "bsr": _TWT_V2_W_BSR,
        },
        "n_sta": len(sta_deltas),
    }


class TwtEfficientV2RewardFunction:
    """Wrapper for the 6-term TWT-efficient reward."""

    def __init__(self):
        self.preset = "twt_efficient_v2"
        self.weights = {
            "throughput": _TWT_V2_W_THR,
            "energy_lin": _TWT_V2_W_ENG_LIN,
            "energy_hinge": _TWT_V2_W_ENG_HNG,
            "airtime": _TWT_V2_W_AIRTIME,
            "drop": _TWT_V2_W_DRP,
            "bsr": _TWT_V2_W_BSR,
        }

    def __call__(self, sta_deltas, action=None):
        return compute_twt_efficient_v2_reward(sta_deltas, action=action)

    def get_weights(self):
        return self.weights.copy()


# --- TWT_EFFICIENT_V3 — adds Jain's fairness term, swaps mean(duty)→max(duty), tightens drop/BSR hinges to EDA p60 (real guardrails, not p90-silent) ---
# Root cause it fixes (vs v2): under v2 the trained transformer converged to a state-INVARIANT degenerate action (sched=18, sta_groups=[1,0,...,0]) — one STA alone, 15 in one group — because:
#   - v2 used mean(duty) and mean(throughput). Means hide unfairness: a policy that starves 15 STAs while one is happy averages out OK.
#   - v2 drop hinge at 0.30 fired on <10% of EDA → silent guardrail.
#   - v2 BSR hinge at 0.50 fired on <40% of EDA → silent guardrail.
#   - Net: 5 of 6 v2 terms gave zero gradient on assignment choice.
#
# v3 changes:
#   - NEW: w_fair * Jain(per-STA delta_bytes). Direct fairness gradient. Degenerate (1, 15-zero) → Jain ≈ 1/N=0.0625; even sharing → 1.0. This is the term that breaks the assignment-invariance symmetry.
#   - mean(duty) → max(duty): one over-awake STA is no longer amortized.
#   - drop_hinge 0.30 → 0.226 (= EDA drop_mean p60)
#   - bsr_hinge  0.50 → 0.481 (= EDA bsr_mean p60)
#
# Calibrated against the same 1.4M-row EDA so each term contributes ~0.3–0.5 at the median operating point.
# See /tmp/calibrate_twt_efficient_v3.py.

_TWT_V3_BI_MS = 102.4
_TWT_V3_DUTY_HINGE = 0.176  # duty_max EDA p50 (catches all over-awake STAs)
_TWT_V3_DROP_HINGE = 0.226  # drop_mean EDA p60 (fires on ~40% of states)
_TWT_V3_BSR_HINGE = 0.481  # bsr_mean  EDA p60 (fires on ~40% of states)

_TWT_V3_W_THR_SUM = 0.15  # linear in network bytes / peak
_TWT_V3_W_FAIR = 1.25  # linear in Jain index — strongest commit signal
_TWT_V3_W_ENG_LIN = 1.0  # linear in MAX duty (over-awake STA penalty)
_TWT_V3_W_ENG_HNG = 1.5  # hinge above duty p50 (kills fixed_s00 over-wake)
_TWT_V3_W_AIRTIME = 0.6  # channel airtime cost (per-step linear)
_TWT_V3_W_DRP = 1.0  # drop guardrail above p60
_TWT_V3_W_BSR = 0.5  # BSR guardrail above p60

_TWT_V3_PHY_PEAK_BYTES_PER_STEP = int(143.4 * 1e6 / 8 * (_TWT_V3_BI_MS / 1000.0) * 25)


def compute_twt_efficient_v3_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, int]] = None,
) -> Dict:
    """7-term reward: Jain fairness + max-duty energy + EDA-anchored hinges.
    Replaces v2's mean-only signals that allowed degenerate assignments to
    score nearly as well as fair ones. `action[0]` provides schedule_idx for
    the airtime term."""
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}

    bytes_tx = get_sta_values(sta_deltas, "delta_bytes_transmitted")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")
    enq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    duty = get_sta_values(sta_deltas, "duty_cycle")
    bsr = get_sta_values(sta_deltas, "bsr_queue_index")

    total_bytes = float(np.sum(bytes_tx))
    thr_sum = total_bytes / max(_TWT_V3_PHY_PEAK_BYTES_PER_STEP, 1)

    jain_thr = float(jains_fairness(bytes_tx))

    max_duty = float(np.max(duty)) if len(duty) else 0.0
    duty_excess = max(0.0, max_duty - _TWT_V3_DUTY_HINGE)

    schedule_idx = int(action[0]) if action is not None else None
    if schedule_idx is not None:
        total_TWT_ms = SCHEDULE_DURATIONS.get(schedule_idx, 0)
    else:
        total_TWT_ms = 0
    airtime_frac = total_TWT_ms / _TWT_V3_BI_MS

    rates = np.where(enq > 0, np.minimum(1.0, drops / np.maximum(enq, 1)), 0.0)
    mean_drop_rate = float(np.mean(rates))
    drop_excess = max(0.0, mean_drop_rate - _TWT_V3_DROP_HINGE)

    bsr_fills = np.minimum(1.0, bsr / 254.0)
    mean_bsr_fill = float(np.mean(bsr_fills))
    bsr_excess = max(0.0, mean_bsr_fill - _TWT_V3_BSR_HINGE)

    w_thr_term = _TWT_V3_W_THR_SUM * thr_sum
    w_fair_term = _TWT_V3_W_FAIR * jain_thr
    w_eng_lin_term = -_TWT_V3_W_ENG_LIN * max_duty
    w_eng_hng_term = -_TWT_V3_W_ENG_HNG * duty_excess
    w_airtime_term = -_TWT_V3_W_AIRTIME * airtime_frac
    w_drp_term = -_TWT_V3_W_DRP * drop_excess
    w_bsr_term = -_TWT_V3_W_BSR * bsr_excess

    total = (
        w_thr_term
        + w_fair_term
        + w_eng_lin_term
        + w_eng_hng_term
        + w_airtime_term
        + w_drp_term
        + w_bsr_term
    )

    return {
        "total": total,
        "components": {
            "w_throughput": w_thr_term,
            "w_fairness": w_fair_term,
            "w_energy_lin": w_eng_lin_term,
            "w_energy_hinge": w_eng_hng_term,
            "w_airtime": w_airtime_term,
            "w_drop": w_drp_term,
            "w_bsr": w_bsr_term,
            "raw_thr_sum": thr_sum,
            "raw_jain": jain_thr,
            "raw_max_duty": max_duty,
            "raw_duty_excess": duty_excess,
            "raw_airtime_frac": airtime_frac,
            "raw_drop_rate": mean_drop_rate,
            "raw_drop_excess": drop_excess,
            "raw_bsr_fill": mean_bsr_fill,
            "raw_bsr_excess": bsr_excess,
        },
        "details": {},
        "metrics": {
            "total_bytes": total_bytes,
            "total_drops": float(np.sum(drops)),
            "max_duty_cycle": max_duty,
            "jain_thr": jain_thr,
            "airtime_frac": airtime_frac,
            "mean_drop_rate": mean_drop_rate,
            "mean_bsr_fill": mean_bsr_fill,
        },
        "weights": {
            "throughput": _TWT_V3_W_THR_SUM,
            "fairness": _TWT_V3_W_FAIR,
            "energy_lin": _TWT_V3_W_ENG_LIN,
            "energy_hinge": _TWT_V3_W_ENG_HNG,
            "airtime": _TWT_V3_W_AIRTIME,
            "drop": _TWT_V3_W_DRP,
            "bsr": _TWT_V3_W_BSR,
        },
        "n_sta": len(sta_deltas),
    }


class TwtEfficientV3RewardFunction:
    """v3 wrapper: adds Jain fairness, max-duty energy, EDA-anchored hinges."""

    def __init__(self):
        self.preset = "twt_efficient_v3"
        self.weights = {
            "throughput": _TWT_V3_W_THR_SUM,
            "fairness": _TWT_V3_W_FAIR,
            "energy_lin": _TWT_V3_W_ENG_LIN,
            "energy_hinge": _TWT_V3_W_ENG_HNG,
            "airtime": _TWT_V3_W_AIRTIME,
            "drop": _TWT_V3_W_DRP,
            "bsr": _TWT_V3_W_BSR,
        }

    def __call__(self, sta_deltas, action=None):
        return compute_twt_efficient_v3_reward(sta_deltas, action=action)

    def get_weights(self):
        return self.weights.copy()


# --- PROPORTIONAL-FAIRNESS / SCALE-FREE REWARD FAMILY (`*_pf` presets) ---
# Deployment-agnostic reward built from intrinsic ratios that need NO offline calibration.
# Designed from the distribution analysis in `reward-exploration/runs/exp_20260512_184226/analysis/summary.json` over 1.4M (STA, step) rows on the current dynamic env.
# Key design choices:
#
# - **No `soft_clip` per component** — components are linear in their inputs, so PPO sees a meaningful gradient at every operating point. The original `throughput` preset saturated each component at +/- 1.0; that's exactly how the trained policy got trapped in a 20ms-wake corner.
# - **Proportional fairness via `log(1 + N · ratio_i)`** — concave per-STA shape gives diminishing returns. Raising one starving STA is worth more than piling more bytes onto a saturated STA, so fairness is implicit; no separate Jain's term needed.
# - **Penalty terms anchored at p90, not p50** — `drop_rate`, `bsr_overflow`, and `excess_duty_cycle` are silent at the median operating point and bite when things go bad. Anchoring at p50 would make them constantly activated and create a pathological gradient (the same problem the `airtime` term had in the original `throughput` preset).
# - **Inputs are intrinsic ratios (peak PHY rate, drops/enqueued, BSR/254, duty cycle)** — no `NORM` dependency. Should work across deployments without re-running an EDA. See `reward-exploration/spawn_worker.py` for the ratio definitions.

# --- Intrinsic protocol-level constants (NOT deployment-tuned). ---
_PF_PHY_PEAK_RATE_MBPS = 143.4
_PF_BEACON_INTERVAL_MS = 102.4
_PF_BYTES_PER_BI_PEAK = int(
    _PF_PHY_PEAK_RATE_MBPS * 1e6 / 8 * (_PF_BEACON_INTERVAL_MS / 1000.0)
)  # ≈ 1_835_520
_PF_BSR_MAX = 254  # BSR index range (802.11ax)
_PF_DUTY_MEDIAN = 0.16  # empirical p50 of duty_cycle on current env
_PF_DUTY_HINGE_SCALE = 0.05  # empirical (p90 - p50) of duty_cycle
_PF_DUTY_HINGE_CAP = 2.0  # cap excess-duty hinge at p99 - p50 (≈ 0.10/0.05).
# Without this, an agent that lands on full-wake actions gets penalty O(20+) per STA, drowning all other gradient signal at init.

# --- Term coefficients (sized so each weighted term ≈ 0.5 at its calibration percentile; PPO advantage normalization handles the rest) ---
#     pf:           coef * mean_i log(1 + N · ratio_thr_i)           → 0.5 @ p50
#     drop:         coef * mean_i drop_rate_i                          → -0.5 @ p90
#     bsr_overflow: coef * mean_i ratio_bsr_i                          → -0.5 @ p90
#     excess_duty:  coef * mean_i max(0, (duty - 0.16)/0.05)           → -0.5 @ p90
#
# Mixing different physical objectives is done by varying these weights; the functional shape of each term is identical across presets.

PF_PRESETS = {
    # Weights are 10× SMALLER than the original calibration.
    # Reason: the training agent's action space (raw payload over the 20-schedule exploration-scripts/ table) produces higher-duty-cycle states than the 40x40 reward-exploration sampling on which the original coefs were calibrated.
    # With original weights, per-episode rewards were ~-200 and PPO's value head couldn't track the magnitude — policy gradient got drowned.
    # Scaling to ~-20/episode keeps the relative shape identical while putting reward magnitudes in the range PPO's advantage normalization is happy with (similar to throughput_linear).
    # Maximize throughput, weak energy/queue regularization.
    "throughput_pf": {
        "pf": 0.139,
        "drop": 0.061,
        "bsr_overflow": 0.050,
        "excess_duty": 0.050,
    },
    # Equal-weight all four objectives.
    "balanced_pf": {
        "pf": 0.080,
        "drop": 0.080,
        "bsr_overflow": 0.080,
        "excess_duty": 0.080,
    },
    # Heavy energy emphasis — reward saving wake time strongly.
    "energy_pf": {
        "pf": 0.040,
        "drop": 0.050,
        "bsr_overflow": 0.030,
        "excess_duty": 0.200,
    },
    # Latency-first — punish queue buildup hard.
    "latency_pf": {
        "pf": 0.060,
        "drop": 0.100,
        "bsr_overflow": 0.150,
        "excess_duty": 0.030,
    },
}


def _pf_throughput(sta_deltas: List[Dict[str, float]], num_sta: int) -> float:
    """Proportional fairness over per-STA bytes_tx, with intrinsic
    fair-share normalization. mean_i log(1 + N * delta_bytes_i / PEAK)."""
    if not sta_deltas:
        return 0.0
    bytes_tx = get_sta_values(sta_deltas, "delta_bytes_transmitted")
    ratios = bytes_tx / max(_PF_BYTES_PER_BI_PEAK, 1)
    return float(np.mean(np.log1p(num_sta * ratios)))


def _pf_drop_rate(sta_deltas: List[Dict[str, float]]) -> float:
    """Mean per-STA drop fraction (drops / enqueued), clipped to [0, 1].
    Zero when no packets were enqueued; we don't punish silence."""
    if not sta_deltas:
        return 0.0
    enq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")
    rates = np.where(enq > 0, np.minimum(1.0, drops / np.maximum(enq, 1)), 0.0)
    return float(np.mean(rates))


def _pf_bsr_fill(sta_deltas: List[Dict[str, float]]) -> float:
    """Mean BSR index normalized to [0, 1]. The BSR field is a MAC-layer
    self-reported queue-fill estimate that saturates at 254 ('over half-full')."""
    if not sta_deltas:
        return 0.0
    bsr = get_sta_values(sta_deltas, "bsr_queue_index")
    return float(np.mean(np.minimum(1.0, bsr / _PF_BSR_MAX)))


def _pf_excess_duty(sta_deltas: List[Dict[str, float]]) -> float:
    """Hinge penalty for duty cycle ABOVE the empirical median, capped at
    `_PF_DUTY_HINGE_CAP`. Zero below the median, grows linearly above,
    flat above the cap. The median (0.16) and scale (0.05) are p50 and
    (p90 - p50) of duty_cycle on the current env."""
    if not sta_deltas:
        return 0.0
    duty = get_sta_values(sta_deltas, "duty_cycle")
    excess = np.maximum(0.0, (duty - _PF_DUTY_MEDIAN) / _PF_DUTY_HINGE_SCALE)
    excess = np.minimum(excess, _PF_DUTY_HINGE_CAP)
    return float(np.mean(excess))


def compute_reward_pf(
    sta_deltas: List[Dict[str, float]],
    preset: str = "balanced_pf",
    action: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Compute a proportional-fairness scale-free reward.

    Args:
        sta_deltas: per-STA delta dicts (same format as compute_reward).
        preset: one of `PF_PRESETS`.
        action: ignored — this reward family does not condition on actions.

    Returns:
        dict with 'total', 'components', 'metrics'.
    """
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}
    if preset not in PF_PRESETS:
        raise ValueError(f"Unknown PF preset: {preset}. Use: {list(PF_PRESETS.keys())}")
    w = PF_PRESETS[preset]
    n_sta = len(sta_deltas)

    r_pf = _pf_throughput(sta_deltas, n_sta)
    r_drop = _pf_drop_rate(sta_deltas)  # in [0, 1]
    r_overflow = _pf_bsr_fill(sta_deltas)  # in [0, 1]
    r_duty = _pf_excess_duty(sta_deltas)  # >= 0, unbounded above

    w_pf = w["pf"] * r_pf  # positive contribution
    w_drop = -w["drop"] * r_drop
    w_overflow = -w["bsr_overflow"] * r_overflow
    w_duty = -w["excess_duty"] * r_duty

    total = w_pf + w_drop + w_overflow + w_duty

    # Aggregate raw metrics for logging
    total_bytes = float(np.sum(get_sta_values(sta_deltas, "delta_bytes_transmitted")))
    total_drops = float(np.sum(get_sta_values(sta_deltas, "delta_drops_expired")))
    total_energy = float(np.sum(get_sta_values(sta_deltas, "delta_energy_mj")))
    avg_queue = float(np.mean(get_sta_values(sta_deltas, "queue_size_bytes")))

    return {
        "total": total,
        "components": {
            "w_pf_throughput": w_pf,
            "w_drop": w_drop,
            "w_bsr_overflow": w_overflow,
            "w_excess_duty": w_duty,
            "raw_pf_throughput": r_pf,
            "raw_drop_rate": r_drop,
            "raw_bsr_fill": r_overflow,
            "raw_excess_duty": r_duty,
        },
        "details": {},
        "metrics": {
            "total_bytes": total_bytes,
            "total_drops": total_drops,
            "total_energy_mj": total_energy,
            "avg_queue_bytes": avg_queue,
        },
        "weights": w,
        "n_sta": n_sta,
    }


class PFRewardFunction:
    """Wrapper for the PF reward family, matching RewardFunction's interface."""

    def __init__(self, preset: str = "balanced_pf"):
        if preset not in PF_PRESETS:
            raise ValueError(
                f"Unknown PF preset: {preset}. Use: {list(PF_PRESETS.keys())}"
            )
        self.preset = preset
        self.weights = PF_PRESETS[preset]

    def __call__(self, sta_deltas, action=None):
        return compute_reward_pf(sta_deltas, preset=self.preset, action=action)

    def get_weights(self):
        return self.weights.copy()


# --- TWT_PF_DEMAND_V1 ("v4") — class-blind per-STA SLO reward ---
# Design intent (from the in-session crisis catalog):
#   - All quality dimensions normalized by each STA's OWN observed demand (served_ratio = pkts_tx / pkts_eq), so heterogeneous traffic cannot confound the reward (v3's Jain-on-bytes failure mode).
#   - PF (alpha=1, log) aggregation across STAs gives smooth gradient everywhere with implicit anti-starvation curvature.
#   - max-violation term `max_i (1 - served_ratio_i)^2` provides SHARP anti-starvation that the mean-PF aggregation alone is too gentle for (1/N attenuation problem with N=16).
#   - Energy and airtime kept as LINEAR costs (they are true costs, not satisfaction axes); airtime is global (channel reservation), duty is per-STA (energy).
#   - All inputs are intrinsic protocol ratios; zero NORM dependency.
#
# Mitigations baked into the implementation:
#   A. mean-PF too weak alone           -> add `w_mv * max(1-served)^2`
#   B. action=None silently zeros air   -> raise ValueError if action missing
#   C. always-negative reward           -> offset PF terms by -log(eps)
#                                          so each is in [0, log((1+eps)/eps)]
#   D. low-demand STA noise dominates   -> exclude STAs with pkts_eq < FLOOR
#                                          from the PF sum (treat as served)
#   E. steep gradient near served=0     -> per-step reward clipped to +/- R_CLIP
#
# See conversation history for the full audit and v3 NS-3 numbers that motivated each choice.

_PF_V1_EPS = 0.01  # log offset; log(0.01) = -4.6 (sharp but bounded)
_PF_V1_DEMAND_FLOOR = 3  # STAs with pkts_eq below this -> treated as served
_PF_V1_DUTY_MEDIAN = 0.16  # empirical p50 of duty_cycle on current env
_PF_V1_DUTY_HINGE_SCALE = 0.05  # empirical (p90 - p50)
_PF_V1_DUTY_HINGE_CAP = 10.0  # cap raised from 2.0 — first NS-3 validation showed cap=2 hid 80% of fixed_s00's energy cost (duty_max=0.85 gives raw excess=13.8, capped to 2.0).
# With cap=10, fixed_s00's over-wake is properly penalized while max per-step energy cost stays bounded at w_eng * cap = 0.2 * 10 = 2.0 (under the PF satisfaction sum of ~1.2).
# Still no gradient drowning.
_PF_V1_BI_MS = 102.4
# One agent step = TWT_UPDATE_INTERVAL_BI(25) × BEACON_INTERVAL_MS(102.4) = 2560 ms.
# Used to normalize the MEASURED airtime occupancy (sum of per-STA Δairtime_used_us over the step) into a channel-busy fraction in [0,1]. Mirrors twt_spawn_worker._STEP_US.
_PF_V1_STEP_US = 25 * 102.4 * 1000.0  # 2.56e6 us
_PF_V1_R_CLIP = 5.0  # per-step reward clip for stability

# Term weights.
# Chosen so each PF term tops out near +0.4 at full service and max-violation gives ~ -1.0 for one fully-starved STA out of 16.
_PF_V1_W_PF_THR = 0.10
_PF_V1_W_PF_DROP = 0.08
_PF_V1_W_PF_BSR = 0.05
_PF_V1_W_MV = 1.00  # sharp anti-starvation on worst-served STA
_PF_V1_W_ENG = 0.20
_PF_V1_W_AIR = 0.50

# NOTE (2026-05-31): the QoEH SoC reward bucket (twt_pf_demand_qoeh_v1) was REMOVED.
# REHD state-of-charge is uncontrollable by TWT scheduling under realistic harvest (EDA + the Phase-1 PDW sweep: SoC pinned ~0.75 regardless of action), so a SoC reward has no usable gradient.
# REHD energy-failure is instead captured by v1's existing class-blind QoS terms (served_ratio / drop_rate / max_starv — a starved REHD shows low served + high expiry drops).
# The controllable energy lever is the Power Delivery Window (an AP action), which raises REHD *throughput* (which v1's served term already rewards), not stored SoC.


def _pf_log_offset(x: np.ndarray, eps: float) -> np.ndarray:
    """log(eps + x) shifted to [0, log((1+eps)/eps)]. Always non-negative."""
    return np.log(eps + x) - np.log(eps)


def compute_twt_pf_demand_v1_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, ...]] = None,
) -> Dict:
    """v4 PF reward: demand-served per-STA, class-blind, with max-violation
    anti-starvation and linear energy/airtime costs.

    Args:
        sta_deltas: per-STA delta dicts (same format as other compute_* fns).
                    Must carry `delta_airtime_used_us` for the measured airtime
                    cost (compute_sta_deltas provides it).
        action: OPTIONAL. Retained for API compatibility / logging only — the
                airtime cost is now MEASURED from per-STA Δairtime_used_us
                (2026-06-04), so the schedule index is no longer needed (and
                baselines pay their actual occupancy automatically). Previously
                REQUIRED when airtime was index-based.
    """
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}

    pkts_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")
    pkts_eq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")
    duty = get_sta_values(sta_deltas, "duty_cycle")
    bsr = get_sta_values(sta_deltas, "bsr_queue_index")

    # --- Per-STA demand-served ratio ---
    # served = pkts_tx / pkts_eq, clipped to [0, 1].
    # When pkts_eq=0 (STA had nothing to send), there is no fault: treat as fully served.
    served_raw = np.where(
        pkts_eq > 0,
        np.minimum(1.0, pkts_tx / np.maximum(pkts_eq, 1)),
        1.0,
    )
    # Demand floor (issue D): low-demand STAs would otherwise dominate noise.
    include_mask = pkts_eq >= _PF_V1_DEMAND_FLOOR
    served_for_pf = np.where(include_mask, served_raw, 1.0)

    # --- Per-STA drop and BSR fill (already demand-normalized) ---
    drop_rate = np.where(
        pkts_eq > 0,
        np.minimum(1.0, drops / np.maximum(pkts_eq, 1)),
        0.0,
    )
    bsr_fill = np.minimum(1.0, bsr / 254.0)

    # --- PF satisfaction terms (offset to non-negative; issue C) ---
    pf_thr_terms = _pf_log_offset(served_for_pf, _PF_V1_EPS)
    pf_drop_terms = _pf_log_offset(1.0 - drop_rate, _PF_V1_EPS)
    pf_bsr_terms = _pf_log_offset(1.0 - bsr_fill, _PF_V1_EPS)

    pf_thr = float(np.mean(pf_thr_terms))
    pf_drop = float(np.mean(pf_drop_terms))
    pf_bsr = float(np.mean(pf_bsr_terms))

    # --- Max-violation (issue A): sharp anti-starvation on worst-served STA ---
    max_starv = float(np.max((1.0 - served_for_pf) ** 2))

    # --- Energy: per-STA excess-duty hinge, capped (reused from PF family) ---
    excess = np.maximum(0.0, (duty - _PF_V1_DUTY_MEDIAN) / _PF_V1_DUTY_HINGE_SCALE)
    excess = np.minimum(excess, _PF_V1_DUTY_HINGE_CAP)
    excess_duty = float(np.mean(excess))

    # --- Airtime: MEASURED channel occupancy (spend, not reservation) ---
    # Switched 2026-06-04 from index-based SCHEDULE_DURATIONS[sched]/BI (the raw reserved window) to the actual per-step channel time STAs occupied.
    # The index version billed *reservation*, not *spend*: it over-charged every PDW-compressed multi-group schedule (EDA: reservation 0.80 vs measured occupancy 0.27 on a long schedule — a 3x over-charge) while charging s0 fairly (ratio ~1.0), biasing the agent into s0.
    # Measured occupancy aligns with the efficiency objective ("pay for what you spend"), self-accounts for PDW compression (occupancy falls as D grows — no (95-D)/95 factor needed), and needs no action arg.
    # Σ per-STA TX airtime ≈ channel-busy fraction (single-channel CSMA: one TXer at a time).
    # Clipped to [0,1].
    airtime_us = get_sta_values(sta_deltas, "delta_airtime_used_us")
    airtime_frac = float(np.clip(np.sum(airtime_us) / _PF_V1_STEP_US, 0.0, 1.0))

    # --- Weighted terms ---
    w_pf_thr_term = _PF_V1_W_PF_THR * pf_thr
    w_pf_drop_term = _PF_V1_W_PF_DROP * pf_drop
    w_pf_bsr_term = _PF_V1_W_PF_BSR * pf_bsr
    w_mv_term = -_PF_V1_W_MV * max_starv
    w_eng_term = -_PF_V1_W_ENG * excess_duty
    w_air_term = -_PF_V1_W_AIR * airtime_frac

    total = (
        w_pf_thr_term
        + w_pf_drop_term
        + w_pf_bsr_term
        + w_mv_term
        + w_eng_term
        + w_air_term
    )
    # Issue E: clip per-step reward for PPO value-fn stability.
    total = float(np.clip(total, -_PF_V1_R_CLIP, _PF_V1_R_CLIP))

    return {
        "total": total,
        "components": {
            "w_pf_thr": w_pf_thr_term,
            "w_pf_drop": w_pf_drop_term,
            "w_pf_bsr": w_pf_bsr_term,
            "w_max_starv": w_mv_term,
            "w_energy": w_eng_term,
            "w_airtime": w_air_term,
            "raw_pf_thr": pf_thr,
            "raw_pf_drop": pf_drop,
            "raw_pf_bsr": pf_bsr,
            "raw_max_starv": max_starv,
            "raw_mean_served": float(np.mean(served_for_pf)),
            "raw_min_served": float(np.min(served_for_pf)),
            "raw_mean_drop": float(np.mean(drop_rate)),
            "raw_max_drop": float(np.max(drop_rate)),
            "raw_mean_bsr": float(np.mean(bsr_fill)),
            "raw_max_bsr": float(np.max(bsr_fill)),
            "raw_excess_duty": excess_duty,
            "raw_max_duty": float(np.max(duty)) if len(duty) else 0.0,
            "raw_airtime_frac": airtime_frac,
            "n_included": int(np.sum(include_mask)),
        },
        "details": {},
        "metrics": {
            "total_bytes": float(
                np.sum(get_sta_values(sta_deltas, "delta_bytes_transmitted"))
            ),
            "total_drops": float(np.sum(drops)),
            "total_pkts_tx": float(np.sum(pkts_tx)),
            "total_pkts_eq": float(np.sum(pkts_eq)),
            "airtime_frac": airtime_frac,
        },
        "weights": {
            "pf_thr": _PF_V1_W_PF_THR,
            "pf_drop": _PF_V1_W_PF_DROP,
            "pf_bsr": _PF_V1_W_PF_BSR,
            "max_starv": _PF_V1_W_MV,
            "energy": _PF_V1_W_ENG,
            "airtime": _PF_V1_W_AIR,
            "eps": _PF_V1_EPS,
            "demand_floor": _PF_V1_DEMAND_FLOOR,
        },
        "n_sta": len(sta_deltas),
    }


class TwtPfDemandV1RewardFunction:
    """v4 wrapper: PF demand-served, max-violation anti-starvation, linear costs."""

    def __init__(self):
        self.preset = "twt_pf_demand_v1"
        self.weights = {
            "pf_thr": _PF_V1_W_PF_THR,
            "pf_drop": _PF_V1_W_PF_DROP,
            "pf_bsr": _PF_V1_W_PF_BSR,
            "max_starv": _PF_V1_W_MV,
            "energy": _PF_V1_W_ENG,
            "airtime": _PF_V1_W_AIR,
        }

    def __call__(self, sta_deltas, action=None):
        return compute_twt_pf_demand_v1_reward(sta_deltas, action=action)

    def get_weights(self):
        return self.weights.copy()


# --- twt_pf_demand_v2 — SERVED/EXPIRY-dominant reward for the 5-segment over/under regime ---
# WHY (2026-06-05 EDA): v1 is pathological in oversaturation — its excess_duty term (cap 10 * 0.20 = -2.0) and airtime cost dominate the PF-LOG served term (which compresses served 0.4 vs 1.0 into a ~0.09 reward gap).
# v1 ranked a STARVING action (serves 40%, drops 64%) ABOVE one serving ~all.
# v2:
#   - served is LINEAR and DOMINANT (serving more always pays clearly).
#   - explicit expiry (drop_rate) penalty = the user's "minimize expiry" objective (captures deadline misses specifically, distinct from still-queued packets).
#   - sharp worst-STA anti-starvation max(1-served)^2 (kept from v1).
#   - SMALL linear airtime-occupancy efficiency cost: in UNDER segments (where a medium schedule fully serves) it makes the agent pick the cheapest-sufficient schedule; in OVER segments serving more still wins (served weight >> airtime).
#   - excess_duty term REMOVED (occupancy already prices resource use; duty double-counted it and dominated).
# Intrinsic [0,1] ratios, linear costs (rule #17).
# Class-blind.
# `action` accepted but UNUSED (airtime = MEASURED occupancy, like v1's 2026-06-04 fix).
_PF_V2_STEP_US = (
    25 * 102.4 * 1000.0
)  # one agent step in us (TWT_UPDATE_INTERVAL_BI * BI)
_PF_V2_W_SRV = 1.00  # linear served (dominant gain)
_PF_V2_W_DROP = 0.50  # explicit expiry penalty (deadline misses)
_PF_V2_W_MV = 0.50  # sharp worst-STA anti-starvation
_PF_V2_W_AIR = 0.50  # airtime-occupancy efficiency cost (subordinate to served)
_PF_V2_DEMAND_FLOOR = (
    3  # STAs with pkts_eq below this -> treated as served (noise floor)
)
_PF_V2_R_CLIP = 5.0


def compute_twt_pf_demand_v2_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, ...]] = None,
) -> Dict:
    """v2 reward: linear served (dominant) + explicit expiry penalty + sharp
    anti-starvation + small measured-airtime efficiency cost. No duty term."""
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}
    pkts_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")
    pkts_eq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")

    served = np.where(
        pkts_eq > 0, np.minimum(1.0, pkts_tx / np.maximum(pkts_eq, 1)), 1.0
    )
    include = pkts_eq >= _PF_V2_DEMAND_FLOOR
    served_for = np.where(include, served, 1.0)
    drop_rate = np.where(
        pkts_eq > 0, np.minimum(1.0, drops / np.maximum(pkts_eq, 1)), 0.0
    )
    drop_for = np.where(include, drop_rate, 0.0)

    mean_served = float(np.mean(served_for))
    mean_drop = float(np.mean(drop_for))
    max_starv = float(np.max((1.0 - served_for) ** 2))

    airtime_us = get_sta_values(sta_deltas, "delta_airtime_used_us")
    occ = float(np.clip(np.sum(airtime_us) / _PF_V2_STEP_US, 0.0, 1.0))

    w_srv = _PF_V2_W_SRV * mean_served
    w_drop = -_PF_V2_W_DROP * mean_drop
    w_mv = -_PF_V2_W_MV * max_starv
    w_air = -_PF_V2_W_AIR * occ
    total = float(np.clip(w_srv + w_drop + w_mv + w_air, -_PF_V2_R_CLIP, _PF_V2_R_CLIP))

    return {
        "total": total,
        "components": {
            "w_served": w_srv,
            "w_drop": w_drop,
            "w_max_starv": w_mv,
            "w_airtime": w_air,
            "raw_mean_served": mean_served,
            "raw_mean_drop": mean_drop,
            "raw_max_starv": max_starv,
            "raw_occupancy": occ,
            "raw_min_served": float(np.min(served_for)),
            "n_included": int(np.sum(include)),
        },
        "details": {},
        "metrics": {
            "total_pkts_tx": float(np.sum(pkts_tx)),
            "total_pkts_eq": float(np.sum(pkts_eq)),
            "total_drops": float(np.sum(drops)),
            "occupancy": occ,
        },
        "weights": {
            "served": _PF_V2_W_SRV,
            "drop": _PF_V2_W_DROP,
            "max_starv": _PF_V2_W_MV,
            "airtime": _PF_V2_W_AIR,
        },
        "n_sta": len(sta_deltas),
    }


class TwtPfDemandV2RewardFunction:
    """v2 wrapper: served-dominant, expiry-penalized, small airtime efficiency cost."""

    def __init__(self):
        self.preset = "twt_pf_demand_v2"
        self.weights = {
            "served": _PF_V2_W_SRV,
            "drop": _PF_V2_W_DROP,
            "max_starv": _PF_V2_W_MV,
            "airtime": _PF_V2_W_AIR,
        }

    def __call__(self, sta_deltas, action=None):
        return compute_twt_pf_demand_v2_reward(sta_deltas, action=action)

    def get_weights(self):
        return self.weights.copy()


# --- twt_pf_demand_v3 — MULTI-OBJECTIVE TRADEOFF reward (served + latency + expiry + REHD-energy sustainability + fairness + efficiency) ---
# WHY (2026-06-05, user-directed): the project objective is a *learnable, state-dependent TRADEOFF* across ALL aspects, not throughput alone.
# v1/v2 captured only served/expiry/airtime.
# v3 adds the two dimensions the user named:
#   (1) LATENCY — clear every STA's buffer with the lowest sojourn possible. Uses the REAL MAC queue-sojourn time (wired 2026-06-05: `step_latency_ms`), uniform across ALL STAs (no priority classes — every buffer matters equally).
#   (2) REHD ENERGY sustainability — REHDs are fragile RF-harvesters; reward keeping harvest >= consumption (H/C -> 1, the *controllable* energy lever: grouping lets REHDs sleep->harvest, ~2x H/C in tests). REHD-only by nature (vcap_max>0); SELF-ZEROS for non-REHDs (no harvester => no consumption) => stays class-blind.
# Uniform importance: REHD protection EMERGES from (a) uniform expiry/served on the worst contributors (REHDs starve most under a big shared SP) + (b) the REHD-only energy dimension — NOT from a class weight.
# Efficiency / anti-contention ("energy lost in wasteful contention") is priced by the measured-airtime occupancy cost.
#
# ALL terms are intrinsic [0,1] ratios, linear (rule #17), class-blind.
# EVERY weight (incl. LAT_DEADLINE_MS, DEMAND_FLOOR, MV_WORST_FRAC) is overridable via the `weights` arg so the EDA / fixed-policy sweeps can re-tune the tradeoff.
# Defaults below are STARTING points, not final.
#
# REV 2026-06-06 (post dense-grid EDA): the anti-starvation term was a SINGLE max((1-served)^2) — at 20 STAs/saturation it saturated to ~0.94 in 69% of steps (a near-CONSTANT penalty, weighted 3.7x the expiry term), giving no gradient AND compressing the finer expiry/latency/energy signals.
# Replaced with the WORST-K MEAN (mean of the k most-starved shortfalls, k = ceil(mv_worst_frac * n)); keeps the fairness/anti-starvation intent (emphasize the most-starved, protects REHDs) but is gradient-rich and not pinned by one noisy STA.
# Weight lowered 0.50 -> 0.30.
# The EDA also confirmed: served signs all correct, REHD H/C is the controllable energy lever via PDW (corr +0.32), occupancy ~uninformative at saturation (corr -0.05, small term).
_PF_V3_STEP_US = 25 * 102.4 * 1000.0  # one agent step in us
_PF_V3_DEFAULTS = {
    "w_srv": 1.00,  # served (throughput) — dominant linear gain
    "w_exp": 0.60,  # expiry (data survival / deadline miss) — heavy
    "w_lat": 0.40,  # latency (queue-sojourn, ALL STAs, uniform)
    "w_mv": 0.30,  # anti-starvation on the worst-k served STAs (fairness)
    "w_eng": 0.30,  # REHD energy sustainability (H/C), REHD-only
    "w_air": 0.30,  # measured-airtime occupancy (efficiency / anti-contention)
    "lat_deadline_ms": 2000.0,  # latency normalization scale (REHD~1900ms, mice~200ms)
    "demand_floor": 3,  # STAs below this pkts_eq treated as served (noise floor)
    "mv_worst_frac": 0.25,  # anti-starvation aggregates the worst 25% of served STAs
    "r_clip": 5.0,  # per-step reward clip (PPO value-fn stability)
}


def compute_twt_pf_demand_v3_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, ...]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Multi-objective tradeoff reward. See module comment above.

    total = w_srv*served - w_exp*expiry - w_lat*latency - w_mv*max_starv
            + w_eng*rehd_hc - w_air*occupancy

    `weights` overrides any default key (used by the offline EDA weight sweep)."""
    W = dict(_PF_V3_DEFAULTS)
    if weights:
        W.update(weights)
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}

    pkts_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")
    pkts_eq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")

    # --- Served / expiry (demand-normalized, class-blind) ---
    served = np.where(
        pkts_eq > 0, np.minimum(1.0, pkts_tx / np.maximum(pkts_eq, 1)), 1.0
    )
    include = pkts_eq >= W["demand_floor"]
    served_for = np.where(include, served, 1.0)
    drop_rate = np.where(
        pkts_eq > 0, np.minimum(1.0, drops / np.maximum(pkts_eq, 1)), 0.0
    )
    drop_for = np.where(include, drop_rate, 0.0)
    mean_served = float(np.mean(served_for))
    mean_drop = float(np.mean(drop_for))
    # Anti-starvation: mean of the worst-k shortfalls (REV 2026-06-06; see header).
    # Single-max saturated to a near-constant ~0.94; worst-k mean is gradient-rich.
    shortfall_sq = (1.0 - served_for) ** 2
    k = max(1, int(np.ceil(W["mv_worst_frac"] * len(shortfall_sq))))
    worst_k = np.sort(shortfall_sq)[::-1][:k]
    max_starv = float(np.mean(worst_k))

    # --- Latency: REAL per-step MAC queue-sojourn, uniform over STAs that served ---
    lat_ms = get_sta_values(sta_deltas, "step_latency_ms")
    lat_cnt = get_sta_values(sta_deltas, "latency_count")
    lat_mask = lat_cnt > 0
    if np.any(lat_mask):
        lat_norm = np.minimum(1.0, lat_ms[lat_mask] / W["lat_deadline_ms"])
        mean_latency = float(np.mean(lat_norm))
    else:
        mean_latency = 0.0

    # --- REHD energy sustainability: H/C capped to [0,1], REHD-only (self-zeros) ---
    vcap_max = get_sta_values(sta_deltas, "vcap_max_v")
    d_harv = get_sta_values(sta_deltas, "delta_harvested_j")
    d_cons = get_sta_values(sta_deltas, "delta_consumed_j")
    rehd_mask = (vcap_max > 0) & (d_cons > 0)
    if np.any(rehd_mask):
        hc = np.minimum(1.0, d_harv[rehd_mask] / np.maximum(d_cons[rehd_mask], 1e-12))
        rehd_hc = float(np.mean(hc))
        n_rehd_active = int(np.sum(rehd_mask))
    else:
        rehd_hc = 0.0
        n_rehd_active = 0

    # --- Efficiency: measured channel occupancy (anti-contention) ---
    airtime_us = get_sta_values(sta_deltas, "delta_airtime_used_us")
    occ = float(np.clip(np.sum(airtime_us) / _PF_V3_STEP_US, 0.0, 1.0))

    # --- Weighted terms ---
    w_srv_t = W["w_srv"] * mean_served
    w_exp_t = -W["w_exp"] * mean_drop
    w_lat_t = -W["w_lat"] * mean_latency
    w_mv_t = -W["w_mv"] * max_starv
    w_eng_t = W["w_eng"] * rehd_hc
    w_air_t = -W["w_air"] * occ
    total = w_srv_t + w_exp_t + w_lat_t + w_mv_t + w_eng_t + w_air_t
    total = float(np.clip(total, -W["r_clip"], W["r_clip"]))

    return {
        "total": total,
        "components": {
            "w_served": w_srv_t,
            "w_expiry": w_exp_t,
            "w_latency": w_lat_t,
            "w_max_starv": w_mv_t,
            "w_rehd_energy": w_eng_t,
            "w_airtime": w_air_t,
            "raw_mean_served": mean_served,
            "raw_mean_drop": mean_drop,
            "raw_mean_latency": mean_latency,
            "raw_max_starv": max_starv,
            "raw_rehd_hc": rehd_hc,
            "raw_occupancy": occ,
            "raw_min_served": float(np.min(served_for)),
            "n_included": int(np.sum(include)),
            "n_rehd_active": n_rehd_active,
        },
        "details": {},
        "metrics": {
            "total_pkts_tx": float(np.sum(pkts_tx)),
            "total_pkts_eq": float(np.sum(pkts_eq)),
            "total_drops": float(np.sum(drops)),
            "occupancy": occ,
            "mean_latency_norm": mean_latency,
            "rehd_hc": rehd_hc,
        },
        "weights": {
            k: W[k]
            for k in (
                "w_srv",
                "w_exp",
                "w_lat",
                "w_mv",
                "w_eng",
                "w_air",
                "lat_deadline_ms",
                "demand_floor",
                "mv_worst_frac",
            )
        },
        "n_sta": len(sta_deltas),
    }


class TwtPfDemandV3RewardFunction:
    """v3 multi-objective tradeoff: served + latency + expiry + REHD-energy +
    fairness + efficiency. Weights overridable (for the EDA sweep / training)."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.preset = "twt_pf_demand_v3"
        self._override = dict(weights) if weights else None
        self.weights = dict(_PF_V3_DEFAULTS)
        if weights:
            self.weights.update(weights)

    def __call__(self, sta_deltas, action=None):
        return compute_twt_pf_demand_v3_reward(
            sta_deltas, action=action, weights=self._override
        )

    def get_weights(self):
        return self.weights.copy()


# --- twt_pf_demand_v4 — α-FAIR per-STA NUM reward (the signal-rework TARGET, 2026-06-07) ---
# WHY: v1/v2/v3 MEAN-aggregate per-STA served/drop, and the controllability EDA (runs/eda_ctl_20260607_064813) proved the MEAN CANCELS opposing per-STA movement (REHD↑ / non-REHD↓): PF(α=1) reward range was ~2.4× the mean(α=0) range across EVERY action head (sched 0.67 vs 0.28).
# v4 replaces the mean with an α-FAIR (NUM / Nash) aggregation of a per-STA utility — the principled fairness objective (Kelly NUM 1998; Mo & Walrand α-fairness 2000).
# α=1 ⇒ proportional-fair / Nash social welfare.
#
# The CONCAVITY of U_α IS the user-requested "soft-but-steep" penalty: U_α(q) is gentle near q=1 and steep as q→0, so a starved / deadline-missing STA is sharply but SMOOTHLY penalized — no hard hinge, no class weighting.
#
# Per-STA utility (class-blind, intrinsic [0,1]):
#   q_i = served_i * (1 - loss_i)
#     served_i = min(1, Δtx/Δeq)        demand-normalized throughput  (=1 below demand_floor)
#     loss_i   = min(1, Δexpired/Δeq)   expiry = DEADLINE MISS.
#   The per-class queue MaxDelay (IoT200/REHD500 tight … Voice1000 … Camera2000/Video3000 elastic) makes expiry the class-aware latency proxy — a class-BLIND uniform penalty whose EFFECT is class-differentiated because tight classes expire first (latency-as-expiry, user insight 2026-06-06).
#   So NO explicit latency term by default (it would need a class scale and would reward speed beyond the SLO); w_lat is available (default 0) for EDA tuning.
#
# Energy: the EDA REVISED soc/hc to CONTROLLABLE under the lopsided scenario + PDW (soc η² sched .98/assign .91/pdw .95; old "SoC uncontrollable" overturned).
#   Small soft REHD term W_eng = mean_rehd(min(1,hc)-1) ≤ 0, hc=Δharv/Δcons.
#   REHD-only, SELF-ZEROS for non-REHD (no harvester ⇒ Δcons=0) ⇒ stays class-blind.
#
# total = W_qos + w_eng*W_eng - w_lat*lat - w_air*occ   (welfare-DEFICITS ≤0 + optional costs; maximized toward 0 = everyone served fairly within deadline + REHDs energy-sustainable).
# α, w_eng, w_lat, w_air, demand_floor, q_floor are all overridable for the round-2 EDA tuning.
_PF_V4_STEP_US = 25 * 102.4 * 1000.0
_PF_V4_DEFAULTS = {
    "alpha": 1.0,  # α-fairness: 1=PF/Nash (EDA-justified default); >1 = more egalitarian
    "w_eng": 0.30,  # REHD energy sustainability (H/C), REHD-only, self-zeros
    "w_lat": 0.0,  # OPTIONAL explicit latency term (OFF: expiry+PF is the deadline penalty)
    "w_air": 0.0,  # OPTIONAL measured-occupancy cost (OFF: served already prices airtime;
    #   avoids the s0-collapse double-count from the index-based v1 cost)
    "lat_deadline_ms": 2000.0,  # global latency norm IF w_lat>0 (class scale lives in expiry, not here)
    "demand_floor": 3,  # STAs below this Δeq treated as satisfied (noise floor)
    "q_floor": 0.02,  # utility floor so U_α (log / power) is bounded
    "r_clip": 5.0,  # per-step reward clip (PPO value-fn stability)
}


def _alpha_fair(q: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized α-fair utility U_α with U_α(1)=0 (perfect satisfaction ⇒ 0 deficit).
    α=1 ⇒ log (PF/Nash); α>1 ⇒ more egalitarian (steeper near 0); α=0 ⇒ linear (=mean).
    """
    q = np.clip(q, 1e-6, 1.0)
    if abs(alpha - 1.0) < 1e-9:
        return np.log(q)
    return (q ** (1.0 - alpha) - 1.0) / (1.0 - alpha)


def compute_twt_pf_demand_v4_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, ...]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """α-fair per-STA NUM reward. See module comment above. `weights` overrides any default."""
    W = dict(_PF_V4_DEFAULTS)
    if weights:
        W.update(weights)
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}

    pkts_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")
    pkts_eq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")

    # --- per-STA utility q_i = served*(1-loss), demand-normalized, class-blind ---
    served = np.where(
        pkts_eq > 0, np.minimum(1.0, pkts_tx / np.maximum(pkts_eq, 1)), 1.0
    )
    loss = np.where(pkts_eq > 0, np.minimum(1.0, drops / np.maximum(pkts_eq, 1)), 0.0)
    include = pkts_eq >= W["demand_floor"]  # noise floor: no real demand ⇒ satisfied
    q = np.where(include, served * (1.0 - loss), 1.0)
    q = np.clip(q, W["q_floor"], 1.0)
    u = _alpha_fair(q, W["alpha"])
    W_qos = float(np.mean(u))  # α-fair social welfare (≤0)

    # --- REHD energy sustainability (H/C), REHD-only, self-zeros for non-REHD ---
    vcap_max = get_sta_values(sta_deltas, "vcap_max_v")
    d_harv = get_sta_values(sta_deltas, "delta_harvested_j")
    d_cons = get_sta_values(sta_deltas, "delta_consumed_j")
    rehd_mask = (vcap_max > 0) & (d_cons > 0)
    if np.any(rehd_mask):
        hc = np.minimum(1.0, d_harv[rehd_mask] / np.maximum(d_cons[rehd_mask], 1e-12))
        W_eng = float(np.mean(hc - 1.0))  # ≤0 deficit
        n_rehd_active = int(np.sum(rehd_mask))
    else:
        W_eng = 0.0
        n_rehd_active = 0

    # --- OPTIONAL latency term (default off; deadline already priced by expiry+PF) ---
    lat_pen = 0.0
    if W["w_lat"] > 0:
        lat_ms = get_sta_values(sta_deltas, "step_latency_ms")
        lat_cnt = get_sta_values(sta_deltas, "latency_count")
        m = lat_cnt > 0
        if np.any(m):
            lat_pen = float(np.mean(np.minimum(1.0, lat_ms[m] / W["lat_deadline_ms"])))

    # --- OPTIONAL measured-occupancy cost (default off) ---
    occ = 0.0
    if W["w_air"] > 0:
        airtime_us = get_sta_values(sta_deltas, "delta_airtime_used_us")
        occ = float(np.clip(np.sum(airtime_us) / _PF_V4_STEP_US, 0.0, 1.0))

    total = W_qos + W["w_eng"] * W_eng - W["w_lat"] * lat_pen - W["w_air"] * occ
    total = float(np.clip(total, -W["r_clip"], W["r_clip"]))

    return {
        "total": total,
        "components": {
            "w_qos": W_qos,
            "w_rehd_energy": W["w_eng"] * W_eng,
            "w_latency": -W["w_lat"] * lat_pen,
            "w_airtime": -W["w_air"] * occ,
            "raw_mean_served": float(np.mean(np.where(include, served, 1.0))),
            "raw_mean_loss": float(np.mean(np.where(include, loss, 0.0))),
            "raw_mean_q": float(np.mean(q)),
            "raw_min_q": float(np.min(q)),
            "raw_eng_deficit": W_eng,
            "raw_latency": lat_pen,
            "raw_occupancy": occ,
            "n_included": int(np.sum(include)),
            "n_rehd_active": n_rehd_active,
            "alpha": W["alpha"],
        },
        "details": {},
        "metrics": {
            "total_pkts_tx": float(np.sum(pkts_tx)),
            "total_pkts_eq": float(np.sum(pkts_eq)),
            "total_drops": float(np.sum(drops)),
            "mean_q": float(np.mean(q)),
            "min_q": float(np.min(q)),
            "occupancy": occ,
        },
        "weights": {
            k: W[k]
            for k in (
                "alpha",
                "w_eng",
                "w_lat",
                "w_air",
                "lat_deadline_ms",
                "demand_floor",
                "q_floor",
            )
        },
        "n_sta": len(sta_deltas),
    }


class TwtPfDemandV4RewardFunction:
    """v4 α-fair per-STA NUM reward: U_α-aggregated per-STA utility q=served·(1−loss)
    + soft REHD-energy term. The principled replacement for the mean-aggregating v1/v2/v3.
    Weights overridable (for the round-2 EDA sweep / training)."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.preset = "twt_pf_demand_v4"
        self._override = dict(weights) if weights else None
        self.weights = dict(_PF_V4_DEFAULTS)
        if weights:
            self.weights.update(weights)

    def __call__(self, sta_deltas, action=None):
        return compute_twt_pf_demand_v4_reward(
            sta_deltas, action=action, weights=self._override
        )

    def get_weights(self):
        return self.weights.copy()


# --- twt_pf_demand_v5 — v4q α-fair QoS + SMOOTH demand-floor (Bayesian shrinkage) + ORACLE REHD-expiry penalty (2026-06-11) ---
# Fixes v4q UNDER-PRICING REHD starvation:
#   1. The hard demand_floor=3 step (44.6% of low-demand REHD steps -> q=1 "satisfied", masking their expiring backlog) is replaced by Laplace / Beta-Binomial SHRINKAGE (rule of succession; Efron-Morris empirical Bayes): served = (Δtx + m)/(Δeq + m), loss = Δexpired/(Δeq + m).
#      Δeq->0 => served->1 (satisfied), loss->0; Δeq>>m => the raw ratios.
#      Smooth, no cliff; m (=old demand_floor) is the pseudo-count.
#      The UNGATED loss already catches a backed-up REHD (Δexpired>>Δeq -> loss->1 -> q->0).
#   2. A dedicated ORACLE REHD-expiry penalty e_i = Δexpired/(Δexpired+Δtx) — normalized by queue THROUGHPUT (packets that LEFT the queue), so it is demand_floor-IMMUNE and well-defined at any demand.
#      REHD identified via energy oracle vcap_max>0; SELF-ZEROS for non-REHD => a REWARD-SIDE oracle term, NEVER in obs (obs class-blindness intact).
# Paired scenario change: REHD deadline 500->200 ms (now tightest class) so energy-locked REHDs expire sooner -> stronger, earlier starvation signal.
#   total = W_qos(shrunk) - w_rehd_exp*W_rehd_exp + w_eng*W_eng - w_air*occ
_PF_V5_DEFAULTS = dict(_PF_V4_DEFAULTS)
_PF_V5_DEFAULTS.update(
    {
        "w_eng": 0.0,  # keep v4q's QoS-only stance; REHD energy now priced by expiry
        "w_rehd_exp": 0.5,  # NEW: REHD expiry penalty weight (oracle, self-zeros non-REHD)
        "shrink_m": 3.0,  # Bayesian pseudo-count (was the hard demand_floor=3)
        "shrink_p0": 1.0,  # served prior at zero demand (1 = assume satisfied; loss prior = 0)
    }
)


def compute_twt_pf_demand_v5_reward(
    sta_deltas: List[Dict[str, float]],
    action: Optional[Tuple[int, ...]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """v5 α-fair NUM reward: smooth (shrinkage) demand floor + oracle REHD-expiry penalty.
    See module comment above. `weights` overrides any default."""
    W = dict(_PF_V5_DEFAULTS)
    if weights:
        W.update(weights)
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}, "metrics": {}}

    pkts_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")
    pkts_eq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    drops = get_sta_values(sta_deltas, "delta_drops_expired")
    m = W["shrink_m"]

    # --- SMOOTH demand floor: Laplace / Beta shrinkage toward a "satisfied" prior ---
    served = np.clip((pkts_tx + m * W["shrink_p0"]) / (pkts_eq + m), 0.0, 1.0)
    loss = np.clip(drops / (pkts_eq + m), 0.0, 1.0)  # loss prior = 0
    q = np.clip(served * (1.0 - loss), W["q_floor"], 1.0)
    u = _alpha_fair(q, W["alpha"])
    W_qos = float(np.mean(u))  # α-fair welfare (≤0)

    # --- ORACLE REHD-expiry penalty (throughput-normalized; demand_floor-immune) ---
    vcap_max = get_sta_values(sta_deltas, "vcap_max_v")
    throughput = drops + pkts_tx  # packets that LEFT the queue
    active_rehd = (vcap_max > 0) & (throughput > 0)
    if np.any(active_rehd):
        e = drops[active_rehd] / np.maximum(throughput[active_rehd], 1e-9)
        W_rehd_exp = float(np.mean(e))  # REHD drop-fraction ∈ [0,1]
        n_rehd_active = int(np.sum(active_rehd))
    else:
        W_rehd_exp = 0.0
        n_rehd_active = 0

    # --- optional v4 energy term (default OFF, w_eng=0) ---
    W_eng = 0.0
    if W["w_eng"] != 0.0:
        d_harv = get_sta_values(sta_deltas, "delta_harvested_j")
        d_cons = get_sta_values(sta_deltas, "delta_consumed_j")
        rehd_e = (vcap_max > 0) & (d_cons > 0)
        if np.any(rehd_e):
            hc = np.minimum(1.0, d_harv[rehd_e] / np.maximum(d_cons[rehd_e], 1e-12))
            W_eng = float(np.mean(hc - 1.0))

    # --- optional measured-occupancy cost (default off) ---
    occ = 0.0
    if W["w_air"] > 0:
        airtime_us = get_sta_values(sta_deltas, "delta_airtime_used_us")
        occ = float(np.clip(np.sum(airtime_us) / _PF_V4_STEP_US, 0.0, 1.0))

    total = W_qos - W["w_rehd_exp"] * W_rehd_exp + W["w_eng"] * W_eng - W["w_air"] * occ
    total = float(np.clip(total, -W["r_clip"], W["r_clip"]))

    return {
        "total": total,
        "components": {
            "w_qos": W_qos,
            "w_rehd_exp": -W["w_rehd_exp"] * W_rehd_exp,
            "w_rehd_energy": W["w_eng"] * W_eng,
            "w_airtime": -W["w_air"] * occ,
            "raw_mean_served": float(np.mean(served)),
            "raw_mean_loss": float(np.mean(loss)),
            "raw_mean_q": float(np.mean(q)),
            "raw_min_q": float(np.min(q)),
            "raw_rehd_expiry_frac": W_rehd_exp,
            "n_rehd_active": n_rehd_active,
            "alpha": W["alpha"],
        },
        "details": {},
        "metrics": {
            "total_pkts_tx": float(np.sum(pkts_tx)),
            "total_pkts_eq": float(np.sum(pkts_eq)),
            "total_drops": float(np.sum(drops)),
            "mean_q": float(np.mean(q)),
            "min_q": float(np.min(q)),
            "rehd_expiry_frac": W_rehd_exp,
        },
        "weights": {
            k: W[k]
            for k in (
                "alpha",
                "w_eng",
                "w_rehd_exp",
                "shrink_m",
                "shrink_p0",
                "q_floor",
            )
        },
        "n_sta": len(sta_deltas),
    }


class TwtPfDemandV5RewardFunction:
    """v5: α-fair QoS with a SMOOTH (Bayesian-shrinkage) demand floor + a dedicated oracle
    REHD-expiry penalty. Fixes v4q's masking of low-demand REHD starvation. Weights overridable.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.preset = "twt_pf_demand_v5"
        self._override = dict(weights) if weights else None
        self.weights = dict(_PF_V5_DEFAULTS)
        if weights:
            self.weights.update(weights)

    def __call__(self, sta_deltas, action=None):
        return compute_twt_pf_demand_v5_reward(
            sta_deltas, action=action, weights=self._override
        )

    def get_weights(self):
        return self.weights.copy()


# --- REGISTRY ---

REWARD_REGISTRY = {
    # Primary presets (match PRESET_WEIGHTS keys) — LEGACY (saturating)
    "throughput": lambda: RewardFunction("throughput"),
    "energy": lambda: RewardFunction("energy"),
    "queue": lambda: RewardFunction("queue"),
    # Diagnostic: linear, single-component, no soft_clip saturation.
    "throughput_linear": lambda: LinearThroughputRewardFunction(),
    # Diagnostic + drop penalty (Option B from offline reward-design study).
    "throughput_linear_with_drop": lambda: LinearThroughputWithDropRewardFunction(),
    # The reward TWT actually needs: bytes - energy_hinge - drop_rate.
    # Designed so fixed_s00 (duty ~0.88) is NOT optimal; moderate-wake + state-conditional scheduling is.
    "twt_efficient": lambda: TwtEfficientRewardFunction(),
    # Full 6-term TWT-efficient reward with airtime cost and EDA-calibrated guardrails.
    # Recommended for new training (post-2026-05-19).
    "twt_efficient_v2": lambda: TwtEfficientV2RewardFunction(),
    # v3: adds Jain fairness, swaps mean(duty)→max(duty), tightens drop/BSR hinges to EDA p60.
    # Fixes v2's degenerate-assignment failure mode.
    "twt_efficient_v3": lambda: TwtEfficientV3RewardFunction(),
    # v4: class-blind per-STA SLO reward.
    # PF on served_ratio + max-violation anti-starvation + linear energy/airtime costs.
    # Replaces v3 after Jain was proven inadequate under heterogeneous traffic demand.
    "twt_pf_demand_v1": lambda: TwtPfDemandV1RewardFunction(),
    # v2: served/expiry-dominant for the 5-segment over/under regime.
    # Linear served (un-compressed) + explicit expiry penalty + small measured-airtime efficiency cost; excess_duty REMOVED.
    # Fixes v1's oversaturation pathology (2026-06-05).
    "twt_pf_demand_v2": lambda: TwtPfDemandV2RewardFunction(),
    # v3: MULTI-OBJECTIVE tradeoff — served + expiry + REAL latency + REHD-energy (H/C) + worst-STA fairness + measured-airtime efficiency.
    # All weights overridable for the offline EDA weight-sweep.
    # The learnable, state-dependent tradeoff (2026-06-05).
    "twt_pf_demand_v3": lambda: TwtPfDemandV3RewardFunction(),
    # v4: α-FAIR per-STA NUM reward (2026-06-07 signal-rework TARGET).
    # Replaces the mean-aggregation of v1/v2/v3 (EDA-proven to cancel per-STA movement) with α-fair (PF/Nash) aggregation of per-STA utility q=served·(1−loss); + soft REHD-energy term.
    "twt_pf_demand_v4": lambda: TwtPfDemandV4RewardFunction(),
    # v4q: v4 with the REHD-energy term ZEROED (w_eng=0).
    # The 2026-06-08 squeeze audit showed w_eng creates a PDW-farming shortcut (corr(pdw,REHD-starve)≈0) that halves throughput, while REHD energy success is already rewarded via served REHD packets in W_qos.
    # Offline-validated: w_eng=0 strengthens anti-high-D ranking, leaves schedule SNR (~2.2) and QoS-discrimination (corr 0.97) unchanged.
    # With w_eng=0 the PDW head collapses to D=0, shrinking the effective action space to schedule×assignment.
    # α=1 (PF) retained.
    "twt_pf_demand_v4q": lambda: TwtPfDemandV4RewardFunction(weights={"w_eng": 0.0}),
    # v5: v4q + SMOOTH demand-floor (Bayesian/Laplace shrinkage, no hard q=1 cliff) + a dedicated ORACLE REHD-expiry penalty e=Δexpired/(Δexpired+Δtx) (throughput-normalized, demand_floor-immune, self-zeros for non-REHD).
    # Fixes v4q under-pricing REHD starvation (2026-06-11).
    # Paired with REHD deadline 500->200ms (now tightest).
    # Obs class-blind.
    "twt_pf_demand_v5": lambda: TwtPfDemandV5RewardFunction(),
    # Proportional-fairness scale-free family (recommended for new training).
    "throughput_pf": lambda: PFRewardFunction("throughput_pf"),
    "balanced_pf": lambda: PFRewardFunction("balanced_pf"),
    "energy_pf": lambda: PFRewardFunction("energy_pf"),
    "latency_pf": lambda: PFRewardFunction("latency_pf"),
    # Aliases for argparse compatibility
    "balanced": lambda: RewardFunction("throughput"),
    "latency": lambda: RewardFunction("queue"),
    "throughput_focused": lambda: RewardFunction("throughput"),
    "energy_focused": lambda: RewardFunction("energy"),
    "latency_focused": lambda: RewardFunction("queue"),
    # Additional aliases
    "fairness_focused": lambda: RewardFunction(
        "throughput"
    ),  # throughput has fairness bonus
}


def get_reward_function(name: str, **kwargs) -> RewardFunction:
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown: {name}. Available: {list(REWARD_REGISTRY.keys())}")
    return REWARD_REGISTRY[name]()


# --- REWARD LOGGER - Captures full reward breakdown for plotting ---

import csv
import os
from datetime import datetime


class RewardLogger:
    """
    Logs detailed reward breakdown per step for later plotting/analysis.

    Supports two modes:
    1. Immediate: Creates CSV file on init, writes each step immediately
    2. Buffered: Collects data in memory, saves on demand (useful for env integration)
    """

    # CSV columns
    COLUMNS = [
        # Episode/step info
        "episode",
        "step",
        "timestamp",
        # Total and weighted components
        "total_reward",
        "w_throughput",
        "w_queue",
        "w_energy",
        "w_drops",
        "w_channel",
        "w_airtime",
        # Raw component rewards
        "raw_throughput",
        "raw_queue",
        "raw_energy",
        "raw_drops",
        "raw_channel",
        "raw_airtime",
        # Throughput sub-rewards
        "throughput_bytes_reward",
        "throughput_packets_reward",
        "throughput_eff_reward",
        "throughput_fairness_bonus",
        # Queue sub-rewards
        "queue_reward",
        "queue_packets_reward",
        "queue_bsr_reward",
        "queue_drain_reward",
        "queue_max_penalty",
        # Energy sub-rewards
        "energy_sleep_reward",
        "energy_consumption_reward",
        "energy_duty_reward",
        "energy_worst_penalty",
        # Drops sub-rewards
        "drops_reward",
        "drops_worst_penalty",
        "drops_fairness_penalty",
        # Channel sub-rewards
        "channel_fcs_reward",
        "channel_rx_reward",
        "channel_recency_reward",
        # Airtime sub-rewards
        "airtime_duration_reward",
        # Raw metrics
        "total_bytes",
        "total_packets",
        "total_airtime",
        "throughput_efficiency",
        "throughput_fairness",
        "starving_stas",
        "avg_queue_bytes",
        "avg_queue_packets",
        "max_queue_bytes",
        "avg_bsr_index",
        "drain_ratio",
        "high_queue_stas",
        "total_energy_mj",
        "avg_sleep_ratio",
        "min_sleep_ratio",
        "avg_duty_cycle",
        "sleep_fairness",
        "total_drops",
        "max_drops",
        "high_drop_stas",
        "avg_fcs_errors",
        "avg_rx_fragments",
        "inactive_stas",
        "schedule_duration_ms",
        "beacon_utilization",
        # Z-scores (for detecting out-of-bounds metrics, |z| > 2 is unusual)
        "total_bytes_z",
        "total_packets_z",
        "total_airtime_z",
        "avg_queue_bytes_z",
        "avg_queue_packets_z",
        "avg_bsr_index_z",
        "total_energy_z",
        "avg_duty_cycle_z",
        "total_drops_z",
        "avg_fcs_errors_z",
        "avg_rx_fragments_z",
        # Action (optional)
        "schedule_idx",
        "assignment_idx",
    ]

    def __init__(
        self,
        preset: str = "throughput",
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Args:
            preset: Reward preset name (for reference only)
            log_dir: Directory to save CSV files (if None, uses buffered mode)
            run_name: Optional run name for filename (default: timestamp)
        """
        self.preset = preset
        self.episode = 0
        self.step = 0
        self.data = []  # Buffered rows

        # If log_dir provided, write to file immediately
        self.immediate_mode = log_dir is not None
        self.log_path = None

        if self.immediate_mode:
            os.makedirs(log_dir, exist_ok=True)

            if run_name is None:
                run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.log_path = os.path.join(log_dir, f"reward_log_{run_name}.csv")

            # Create file with header
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

            print(f"RewardLogger initialized (immediate mode): {self.log_path}")
        else:
            print(f"RewardLogger initialized (buffered mode) for preset: {preset}")

    def _build_row(self, reward_result: Dict, action: Optional[tuple] = None) -> List:
        """Build a row from reward_result dict."""
        components = reward_result.get("components", {})
        details = reward_result.get("details", {})
        metrics = reward_result.get("metrics", {})
        zscores = reward_result.get("zscores", {})

        row = [
            self.episode,
            self.step,
            datetime.now().isoformat(),
            reward_result.get("total", 0),
            # Weighted components
            components.get("w_throughput", 0),
            components.get("w_queue", 0),
            components.get("w_energy", 0),
            components.get("w_drops", 0),
            components.get("w_channel", 0),
            # Raw components
            components.get("raw_throughput", 0),
            components.get("raw_queue", 0),
            components.get("raw_energy", 0),
            components.get("raw_drops", 0),
            components.get("raw_channel", 0),
            # Throughput sub-rewards
            details.get("throughput_bytes_reward", 0),
            details.get("throughput_packets_reward", 0),
            details.get("throughput_eff_reward", 0),
            details.get("throughput_fairness_bonus", 0),
            # Queue sub-rewards
            details.get("queue_reward", 0),
            details.get("queue_packets_reward", 0),
            details.get("queue_bsr_reward", 0),
            details.get("queue_drain_reward", 0),
            details.get("queue_max_penalty", 0),
            # Energy sub-rewards
            details.get("energy_sleep_reward", 0),
            details.get("energy_consumption_reward", 0),
            details.get("energy_duty_reward", 0),
            details.get("energy_worst_penalty", 0),
            # Drops sub-rewards
            details.get("drops_reward", 0),
            details.get("drops_worst_penalty", 0),
            details.get("drops_fairness_penalty", 0),
            # Channel sub-rewards
            details.get("channel_fcs_reward", 0),
            details.get("channel_rx_reward", 0),
            details.get("channel_recency_reward", 0),
            # Raw metrics
            metrics.get("total_bytes", 0),
            metrics.get("total_packets", 0),
            metrics.get("total_airtime", 0),
            metrics.get("throughput_efficiency", 0),
            metrics.get("throughput_fairness", 0),
            metrics.get("starving_stas", 0),
            metrics.get("avg_queue_bytes", 0),
            metrics.get("avg_queue_packets", 0),
            metrics.get("max_queue_bytes", 0),
            metrics.get("avg_bsr_index", 0),
            metrics.get("drain_ratio", 0),
            metrics.get("high_queue_stas", 0),
            metrics.get("total_energy_mj", 0),
            metrics.get("avg_sleep_ratio", 0),
            metrics.get("min_sleep_ratio", 0),
            metrics.get("avg_duty_cycle", 0),
            metrics.get("sleep_fairness", 0),
            metrics.get("total_drops", 0),
            metrics.get("max_drops", 0),
            metrics.get("high_drop_stas", 0),
            metrics.get("avg_fcs_errors", 0),
            metrics.get("avg_rx_fragments", 0),
            metrics.get("inactive_stas", 0),
            # Z-scores (|z| > 2 indicates unusual values)
            zscores.get("total_bytes_z", 0),
            zscores.get("total_packets_z", 0),
            zscores.get("total_airtime_z", 0),
            zscores.get("avg_queue_bytes_z", 0),
            zscores.get("avg_queue_packets_z", 0),
            zscores.get("avg_bsr_index_z", 0),
            zscores.get("total_energy_z", 0),
            zscores.get("avg_duty_cycle_z", 0),
            zscores.get("total_drops_z", 0),
            zscores.get("avg_fcs_errors_z", 0),
            zscores.get("avg_rx_fragments_z", 0),
            # Action
            action[0] if action else -1,
            action[1] if action else -1,
        ]
        return row

    def log_step(
        self,
        reward_result: Dict,
        episode: Optional[int] = None,
        step: Optional[int] = None,
        action: Optional[tuple] = None,
    ):
        """
        Log a single step's reward breakdown.

        Args:
            reward_result: Output from compute_reward()
            episode: Episode number (uses internal counter if None)
            step: Step number (uses internal counter if None)
            action: Optional (schedule_idx, assignment_idx) tuple
        """
        if episode is not None:
            self.episode = episode
        if step is not None:
            self.step = step
        else:
            self.step += 1

        row = self._build_row(reward_result, action)

        if self.immediate_mode:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        else:
            self.data.append(row)

    def new_episode(self):
        """Call at the start of a new episode."""
        self.episode += 1
        self.step = 0

    def save(self, filepath: str):
        """Save buffered data to CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            writer.writerows(self.data)
        print(f"RewardLogger: Saved {len(self.data)} entries to {filepath}")

    def clear(self):
        """Clear buffered data."""
        self.data = []

    def get_log_path(self) -> str:
        """Return the path to the log file (immediate mode only)."""
        return self.log_path if self.log_path else ""

    def __len__(self):
        """Return number of buffered entries."""
        return len(self.data)
