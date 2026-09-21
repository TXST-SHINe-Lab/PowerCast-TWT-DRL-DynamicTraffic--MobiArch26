#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_spawn_worker.py - Runs ONE TWT episode in an isolated process.

Literal port of wifi-simulation/spawn_worker.py for the TWT pipeline.

Each worker:
  1. Gets a unique seed (from worker_id + batch_idx) -> unique SHM segment names
     (the underlying TWTWrapper derives MySeg_{seed} etc. from the seed).
  2. Imports the binding INSIDE the worker process (after spawn).
  3. Runs the full episode using the provided policy state_dict for inference.
  4. Collects a rollout buffer and pushes it to the result queue.
  5. Calls os._exit(0) at the end to skip the Boost.Interprocess static
     destructor, which hangs on the still-mapped shared memory.

The orchestrator MUST drain the result queue BEFORE p.join(): a child blocked on a full queue pipe never exits.
"""

import os

# Cap thread oversubscription BEFORE numpy/torch import.
# With N parallel workers, each Torch/BLAS pool would otherwise grab num_cores threads (N x oversubscription) and starve the single-threaded NS-3 sims — the real per-batch bottleneck.
# Policy inference here is tiny, so 1 thread/worker is plenty and frees cores for NS-3.
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import sys
import json
import pickle
import time
import traceback
import numpy as np

# --- Helpers (copied from the legacy file_comm_env.py / episode_file_runner.py) ---
INVALID_SENTINEL = 65535
BEACON_INTERVAL_MS = 102.4


def _clean_value(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        if v == INVALID_SENTINEL or np.isnan(v) or np.isinf(v):
            return default
        return float(v)
    return default


def _clean_positive(v, max_val=1e9):
    return float(np.clip(_clean_value(v, 0.0), 0.0, max_val))


NUM_STA_FEATURES = 7  # time-local feed (2026-06-07 rework); see spec below
# Variance-stabilized (log1p) + bounded, then PER-FEATURE running-z-scored INSIDE the policy (BasePolicy obs_rms, warm-started from obs_warmstart_stats.json).
# Order MUST match obs_warmstart_stats.json:
#    [0] bsr_be     log1p(bsr_ac_be)/log1p(254)              (heavy-tailed demand level)
#    [1] dpkts_rx   log1p(Δpackets_received_at_ap)/log1p(1632)
#    [2] dairtime   log1p(Δairtime_used_us)/log1p(371329)
#    [3] dfcs       log1p(Δfcs_error_count)/log1p(49)
#    [4] snr        clip((snr_db-48)/(65.35-48), 0, 1)        (weak, geometry)
#    [5] silence    recency: clip((now-last_rx)/STEP_US, 0, 1), 1.0 = never seen
#    [6] starv_rate clip(Δsp_with_starvation / Δsp_with_demand, 0, 1)  (REHD/airtime-bound tell)
# REPLACED the old leaky 18-feat set: dropped ALL cumulative-level features (airtime/awake/sleep/duty/pkts_tx as lifetime AVERAGES violated the time-local rule), the dead per-AC BSR (uniform AC_BE), and the REDUNDANT dbytes_rx (|r|=0.965 with dairtime, EDA round-1).
# Norms re-fit from the controllability EDA (norms_fitted.json). When this changes, move every rule-#29 dependent default together.
DEFAULT_NUM_STA = 16  # fallback when env_dict has no num_sta and caller passes None

# --- Obs normalization constants ---
# One agent step = TWT_UPDATE_INTERVAL_BI=25 × BEACON_INTERVAL_MS=102.4 = 2560 ms.
_STEP_BI = 25
_STEP_MS = _STEP_BI * BEACON_INTERVAL_MS  # 2560 ms
_STEP_US = _STEP_MS * 1000.0  # 2.56e6 us
_BSR_MAX = 254.0  # 802.11ax BSR encoding ceiling per AC
_BSR_AC_COUNT = 4  # AC_BE + AC_BK + AC_VI + AC_VO
_SILENCE_NORM_US = _STEP_US  # silence ≥ 1 agent step saturates
_RX_NORM = 1.0e5  # rx_fragments log1p cap (≥ ~75k expected at 1000pkt/step × 75 steps)
_PKTS_TX_NORM = 1.0e5  # packets_tx log1p cap (same magnitude)
_LOG_RX_NORM = float(np.log1p(_RX_NORM))  # ≈ 11.51
_LOG_PKTS_TX_NORM = float(np.log1p(_PKTS_TX_NORM))  # ≈ 11.51

# --- FINAL 7-feature feed: re-fit log1p / linear norms (round-1 controllability EDA,
# norms_fitted.json). p99 of the per-step Δ (counts) / raw level (bsr) / [p1,p99] (snr). ---
_BSR_LOG_NORM = float(np.log1p(254.0))  # bsr_be (per-AC BSR ceiling, heavy-tailed)
_DPKTS_LOG_NORM = float(np.log1p(1632.0))  # Δpackets_received_at_ap
_DAIR_LOG_NORM = float(np.log1p(371329.0))  # Δairtime_used_us
_DFCS_LOG_NORM = float(np.log1p(49.0))  # Δfcs_error_count
_SNR_LO = 48.0
_SNR_HI = 65.35

# --- QoEH-realistic per-step-delta normalization (Phase B, option B) ---
# Each new realistic counter is CUMULATIVE; we expose its per-step Δ, bounded to
# [0, 1], and let the recurrent net form its own statistic. Counts (SPs) per step
# are bounded by the number of beacon intervals in a step (one wake window/BI);
# byte/packet deltas use log1p compression like the base byte features.
_SP_COUNT_NORM = float(_STEP_BI)  # ≤ ~1 SP per BI → ≤ 25 SPs/step
_BYTES_STEP_NORM = 1.0e7  # ~10 MB/step log1p cap (heavy UL ≈ 1–2 MB/step)
_LOG_BYTES_STEP_NORM = float(np.log1p(_BYTES_STEP_NORM))  # ≈ 16.12

# --- Power Delivery Window (PDW) — 3rd action head (Phase 2) ---
# BI layout (beacon TBTT = t=0): beacon [0,~1.8] (zone [0,PDW_BEACON_MARGIN_MS]) | PDW power window [PDW_BEACON_MARGIN_MS, pdw_end] | comms/SP [pdw_end, 95] | BI tail.
# NS-3 delivers RF power DIRECTLY over [margin, pdw_end] (DeliverPdwPower, beacon-anchored) — a per-BI energy credit to every asleep REHD, NOT packetized frames, so it has 100% window occupancy and CANNOT leak past pdw_end onto the channel or into a comms SP.
# No tail guard is needed (an energy credit can't collide with an SP; a REHD awake in its SP isn't credited).
# The agent's pdw_idx picks the PDW END TIME: pdw_end = margin + idx*step = 5 + idx*5 -> {5,10,...,50} for idx 0..9 (NUM_PDW_LEVELS=10, CARVED 2026-06-21; this block previously described a superseded 15-level design).
# idx 0 -> pdw_end=5 -> power window [5,5] is EMPTY (no PDW).
# idx 1..9 deliver {5,10,...,45} ms of power (idx9=50ms end=45ms window).
# Decoder scale-shifts the (untouched) schedule into comms [pdw_end, 95]: scale=(95-pdw_end)/95, new_offset=pdw_end+offset*scale, new_wake=max(2, wake*scale).
# `pdw_duration_ms` sent to NS-3 IS pdw_end.
# Cap: worst SP end = pdw_end + wake*scale; verified offline over all 10 PDW levels x the carved schedule table — worst case 91.68 ms, well under BI=102.4 ms, min wake 2.0 ms floor — no BI overflow, SP>=2ms OK.
NUM_PDW_LEVELS = (
    10  # pdw_end ∈ {5,10,...,50} ms (idx 0..9; CARVED 2026-06-21, matches EDA)
)
PDW_STEP_MS = 5.0
PDW_BEACON_MARGIN_MS = (
    5.0  # beacon guard + PDW start (must match C++ PDW_BEACON_LEAD_US)
)
PDW_SCALE_DENOM_MS = 95.0  # nominal comms region width
PDW_MIN_WAKE_MS = 2.0  # per-group min wake floor (802.11 SP)


def apply_pdw_scale_shift(twt_group_configs, pdw_idx):
    """Scale-shift group offsets/wakes into the comms region [pdw_end, 95] and return
    pdw_end (ms) — the PDW window END (= comms front edge) that NS-3 needs. Mutates configs
    in place. pdw_end = PDW_BEACON_MARGIN_MS + idx*PDW_STEP_MS = 5 + idx*5 (idx 0 -> 5 =
    no PDW; idx 14 -> 75 = 70 ms power window [5,75]). Power is delivered directly over
    [margin, pdw_end], so the comms region starts right at pdw_end (no tail guard)."""
    pdw_end = PDW_BEACON_MARGIN_MS + float(int(pdw_idx) * PDW_STEP_MS)
    scale = (PDW_SCALE_DENOM_MS - pdw_end) / PDW_SCALE_DENOM_MS
    for gc in twt_group_configs:
        gc["twt_sp_offset_ms"] = pdw_end + gc["twt_sp_offset_ms"] * scale
        gc["twt_wake_duration_ms"] = max(
            PDW_MIN_WAKE_MS, gc["twt_wake_duration_ms"] * scale
        )
    return pdw_end


def default_obs_kwargs(policy_arch, n_total):
    """Arch-aware obs-shape kwargs sizing a policy for `n_total` active STAs at the
    current NUM_STA_FEATURES. The flat obs is num_sta·F, so obs_dim MUST track the
    topology (n_stations + n_rehd) — the constructor defaults assume 16 STAs and
    silently mismatch otherwise. Callers merge as {**default_obs_kwargs(...),
    **user_kwargs} so explicit --policy-kwargs always win.

    Each arch accepts a different subset (passing obs_dim to transformer_pointer,
    which has no such arg, would TypeError) — hence the per-arch mapping.
    analytical_* baselines derive their topology from the observation length at
    each predict() call rather than from constructor kwargs, so they need no
    auto-sized obs_dim and are left at defaults (eval-only baselines).
    """
    F = NUM_STA_FEATURES
    flat = int(n_total) * F
    # Auto-derive the action-head sizes from the shipped action tables so the policy heads
    # always match the (carved) table — no manual --policy-kwargs to forget
    # (same idea as deriving obs_dim from the topology). user_kwargs still override via the {**default, **user} merge.
    ns, na, npd = action_head_sizes()
    if policy_arch in ("mlp_ppo", "lstm_ppo"):
        # features_per_sta sizes the per-feature obs-norm buffers (BasePolicy obs_rms).
        return {
            "obs_dim": flat,
            "features_per_sta": F,
            "num_schedules": ns,
            "num_assignments": na,
            "num_pdw_levels": npd,
        }
    if policy_arch in ("random_policy", "fixed_policy"):
        return {
            "obs_dim": flat,  # baselines: no obs normalization
            "num_schedules": ns,
            "num_assignments": na,
            "num_pdw_levels": npd,
        }
    if policy_arch == "pointer_ppo":
        return {
            "obs_dim": flat,
            "num_sta": int(n_total),
            "features_per_sta": F,
            "num_schedules": ns,
            "num_pdw_levels": npd,
        }  # raw: no num_assignments head
    if policy_arch == "transformer_pointer":
        return {
            "features_per_sta": F,  # num_sta is variable (set obs)
            "num_schedules": ns,
            "num_pdw_levels": npd,
        }
    return {}  # analytical_* / unknown: leave to constructor defaults


def _resolve_num_sta(num_sta, *envs):
    """Resolve num_sta with priority: explicit arg -> env_dict["num_sta"] -> default."""
    if num_sta is not None:
        return int(num_sta)
    for env in envs:
        if env and "num_sta" in env:
            return int(env["num_sta"])
    return DEFAULT_NUM_STA


def _build_per_sta_features(prev_env, curr_env, num_sta):
    """Per-STA FINAL time-local feed: (num_sta, 7). REALISTIC, AP-observable obs only.
    Every feature is a per-step DELTA, instantaneous LEVEL, RECENCY, or RATIO — NO
    cumulative counter reaches the agent (hard rule;
    the old 18-feat set leaked lifetime averages and is removed). Each is variance-
    stabilized (log1p for bursty counts) + bounded ~[0,1]; the PER-FEATURE running
    z-score is applied DOWNSTREAM inside the policy (BasePolicy obs_rms, warm-started
    from obs_warmstart_stats.json) — NOT here. Oracle (energy/queue/served/expiry) stays
    REWARD-ONLY; the agent must INFER energy/REHD state from these signals (the AP is class-blind).

      Feature spec (order MUST match obs_warmstart_stats.json):
        [0] bsr_be     = log1p(bsr_queue_ac_be) / log1p(254)          level, heavy-tailed
        [1] dpkts_rx   = log1p(Δpackets_received_at_ap) / log1p(1632) Δ, served-volume proxy
        [2] dairtime   = log1p(Δairtime_used_us) / log1p(371329)      Δ, occupancy/cost proxy
        [3] dfcs       = log1p(Δfcs_error_count) / log1p(49)          Δ, channel quality
        [4] snr        = clip((snr_db − 48) / (65.35 − 48), 0, 1)     level, weak (geometry)
        [5] silence    = clip((now − last_rx_ts) / _STEP_US, 0, 1)    recency (1.0 = never seen)
        [6] starv_rate = clip(Δsp_with_starvation / Δsp_with_demand, 0, 1)  ratio, REHD tell
      (Dropped vs the old 18-feat set: all cumulative levels (airtime/awake/sleep/duty/
      pkts_tx averages — time-local violation), dead per-AC BSR (uniform AC_BE), and the
      REDUNDANT dbytes_rx (|r|=0.965 with dairtime, EDA round-1).)
    """
    curr_list = curr_env.get("sta_observations", []) if curr_env else []
    prev_list = prev_env.get("sta_observations", []) if prev_env else []
    now_us = (
        float(curr_env.get("simulation_time_sec", 0.0)) * 1.0e6 if curr_env else 0.0
    )
    out = np.zeros((num_sta, NUM_STA_FEATURES), dtype=np.float32)
    for i in range(num_sta):
        if i >= len(curr_list):
            continue  # defensive — env should always size to num_sta (no padding by design)
        cr = curr_list[i].get("realistic", {})
        pr = prev_list[i].get("realistic", {}) if i < len(prev_list) else {}

        def _d(key):  # per-step Δ of a cumulative realistic counter (clamped ≥ 0)
            return max(
                0.0, _clean_positive(cr.get(key, 0)) - _clean_positive(pr.get(key, 0))
            )

        # [0] bsr_be — instantaneous BE queue level (heavy-tailed) → log1p
        bsr_be = _clean_positive(cr.get("bsr_queue_ac_be", 0), _BSR_MAX)
        out[i, 0] = min(1.0, float(np.log1p(bsr_be)) / _BSR_LOG_NORM)
        # [1-3] per-step Δ of cumulative counters → log1p
        out[i, 1] = min(
            1.0, float(np.log1p(_d("packets_received_at_ap"))) / _DPKTS_LOG_NORM
        )
        out[i, 2] = min(1.0, float(np.log1p(_d("airtime_used_us"))) / _DAIR_LOG_NORM)
        out[i, 3] = min(1.0, float(np.log1p(_d("fcs_error_count"))) / _DFCS_LOG_NORM)
        # [4] snr — instantaneous level, linear (weak; sentinel/no-signal → 0 after clip)
        snr_db = _clean_value(cr.get("snr_db", 0.0))
        out[i, 4] = float(np.clip((snr_db - _SNR_LO) / (_SNR_HI - _SNR_LO), 0.0, 1.0))
        # [5] silence — recency of last UL the AP heard (sentinel 0 → fully silent)
        last_rx_us = _clean_positive(cr.get("last_rx_timestamp_us", 0))
        out[i, 5] = (
            1.0
            if last_rx_us <= 0.0
            else float(np.clip((now_us - last_rx_us) / _SILENCE_NORM_US, 0.0, 1.0))
        )
        # [6] starv_rate — Δstarved-SPs / Δdemanded-SPs (REHD/airtime-bound separator)
        d_dem = _d("sp_with_demand_count")
        out[i, 6] = (
            float(np.clip(_d("sp_with_starvation_count") / d_dem, 0.0, 1.0))
            if d_dem > 1e-9
            else 0.0
        )
    return out


def build_ppo_observation(prev_env, curr_env, num_sta=None):
    """Legacy flat obs: returns shape (num_sta * NUM_STA_FEATURES,). Used by
    mlp_ppo, lstm_ppo, pointer_ppo, and the analytical / fixed / random baselines.
    num_sta defaults to env_dict["num_sta"] then DEFAULT_NUM_STA — not hardcoded."""
    n = _resolve_num_sta(num_sta, curr_env, prev_env)
    return _build_per_sta_features(prev_env, curr_env, n).reshape(-1)


def build_ppo_obs_2d(prev_env, curr_env, num_sta=None):
    """Set / transformer-friendly obs: returns shape (num_sta, NUM_STA_FEATURES).
    Used by any policy that sets `is_set_obs_policy = True`."""
    n = _resolve_num_sta(num_sta, curr_env, prev_env)
    return _build_per_sta_features(prev_env, curr_env, n)


# Asymmetric-critic privileged features (ORACLE; critic-only, NEVER the actor/obs).
# The delayed PDW reward flows through stored capacitor energy, so the critic must see the energy state to value the post-action future.
# The STAR feature is delta-harvested (the direct, immediate, controllable consequence of opening D), which lets V() credit "harvested now -> REHD serves later".
# SoC is pinned near Vmin for active REHDs (greedy drain) so it's weak alone; is_rehd disambiguates SoC=0-because-non-REHD from SoC=0-because-depleted.
# Pre-scaled to ~O(1) so the running critic-norm isn't dead in batch 1 (harvested ~1e-4 J/step, consumed ~1e-3 J/step); _normalize_critic refines after.
NUM_CRITIC_FEATURES = 4  # [soc, is_rehd, dHarvested, dConsumed]
_HARV_SCALE = 1.0e4
_CONS_SCALE = 1.0e3


def build_critic_features(prev_env, curr_env, num_sta=None):
    """Flat (num_sta * NUM_CRITIC_FEATURES,) privileged oracle ENERGY state for the
    asymmetric critic. Mirrors build_ppo_observation's flat layout (per-STA F-block)."""
    n = _resolve_num_sta(num_sta, curr_env, prev_env)
    curr = curr_env.get("sta_observations", []) if curr_env else []
    prev = prev_env.get("sta_observations", []) if prev_env else []
    out = np.zeros((n, NUM_CRITIC_FEATURES), dtype=np.float32)
    for i in range(n):
        if i >= len(curr):
            continue
        co = curr[i].get("oracle", {})
        po = prev[i].get("oracle", {}) if i < len(prev) else {}
        vmax = _clean_positive(co.get("vcap_max_v", 0))
        vcap = _clean_positive(co.get("vcap_v", 0))
        out[i, 0] = float(np.clip(vcap / vmax, 0.0, 1.0)) if vmax > 1e-9 else 0.0  # SoC
        out[i, 1] = 1.0 if vmax > 0.0 else 0.0  # is_rehd
        d_harv = max(
            0.0,
            _clean_positive(co.get("harvested_total_j", 0))
            - _clean_positive(po.get("harvested_total_j", 0)),
        )
        d_cons = max(
            0.0,
            _clean_positive(co.get("consumed_total_j", 0))
            - _clean_positive(po.get("consumed_total_j", 0)),
        )
        out[i, 2] = d_harv * _HARV_SCALE
        out[i, 3] = d_cons * _CONS_SCALE
    return out.reshape(-1)


def load_action_tables(parent_dir):
    expdir = os.path.join(parent_dir, "exploration-scripts")
    with open(os.path.join(expdir, "schedule_table.json")) as f:
        st = json.load(f)
    with open(os.path.join(expdir, "assignment_table.json")) as f:
        at = json.load(f)
    return st, at


def action_head_sizes():
    """(num_schedules, num_assignments, num_pdw_levels) read from the shipped action tables
    so policy head dims always track the (carved) table. Used by default_obs_kwargs."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    st, at = load_action_tables(parent_dir)
    return len(st["schedules"]), len(at["assignments"]), NUM_PDW_LEVELS


def build_action_dict(
    schedule_idx, assignment_idx, schedule_table, assignment_table, num_sta, pdw_idx=0
):
    """Identical to episode_file_runner.py.legacy::build_action_dict, plus the
    PDW 3rd head: pdw_idx selects pdw_end = 5 + pdw_idx*PDW_STEP_MS and scale-shifts
    the group offsets/wakes into [pdw_end,95]. pdw_idx=0 → pdw_end=5 → no PDW, comms
    [5,95] (≈ legacy behavior)."""
    from generate_action_tables import apply_assignment_pattern

    sched = schedule_table["schedules"][schedule_idx]
    assign = assignment_table["assignments"][assignment_idx]
    num_groups = sched["num_groups"]

    twt_group_configs = [
        {
            "group_id": g["group_id"],
            "twt_wake_interval_ms": BEACON_INTERVAL_MS,
            "twt_wake_duration_ms": g["wake_duration_ms"],
            "twt_sp_offset_ms": g["sp_offset_ms"],
            "num_stas_assigned": 0,
        }
        for g in sched["groups"]
    ]
    d_ms = apply_pdw_scale_shift(twt_group_configs, pdw_idx)
    mapping = apply_assignment_pattern(assign, num_sta, num_groups)
    sta_assigns = []
    counts = {g["group_id"]: 0 for g in sched["groups"]}
    for sid, gid in mapping:
        sta_assigns.append({"sta_id": sid, "assigned_twt_group": gid, "enable_twt": 1})
        if gid in counts:
            counts[gid] += 1
    for gc in twt_group_configs:
        gc["num_stas_assigned"] = counts.get(gc["group_id"], 0)
    return {
        "num_sta": num_sta,
        "num_active_twt_groups": num_groups,
        "action_timestamp_ms": 0,
        "pdw_duration_ms": d_ms,
        "twt_group_configs": twt_group_configs,
        "sta_group_assignments": sta_assigns,
    }


def build_action_dict_raw(payload, schedule_table, num_sta):
    """
    Raw-action assembler for policies with `is_raw_action_policy = True`
    (e.g. pointer_ppo). The payload comes straight from policy.act_raw():

        payload = {
            "sched":      int                 # in [0, len(schedule_table["schedules"]))
            "sta_groups": np.ndarray (num_sta,) int   # each in [0, max_num_groups)
            "pdw":        int                 # PDW level in [0, NUM_PDW_LEVELS) (Phase 2)
        }

    The schedule_idx still selects per-group wake_duration_ms / sp_offset_ms
    from schedule_table (so the templated wake parameters are reused). The
    per-STA group assignments are taken directly from the policy and wrapped
    via modulo into [0, num_groups) -- this lets the policy emit values in
    the global [0, max_num_groups) range without breaking schedules whose
    num_groups < max_num_groups. The "pdw" head scale-shifts the schedule into
    [D,95] exactly as in build_action_dict (missing/0 => PDW off).
    """
    sched_idx = int(payload["sched"])
    sta_groups = payload["sta_groups"]
    pdw_idx = max(0, min(int(payload.get("pdw", 0)), NUM_PDW_LEVELS - 1))
    sched = schedule_table["schedules"][sched_idx]
    num_groups = int(sched["num_groups"])

    twt_group_configs = [
        {
            "group_id": g["group_id"],
            "twt_wake_interval_ms": BEACON_INTERVAL_MS,
            "twt_wake_duration_ms": g["wake_duration_ms"],
            "twt_sp_offset_ms": g["sp_offset_ms"],
            "num_stas_assigned": 0,
        }
        for g in sched["groups"]
    ]
    d_ms = apply_pdw_scale_shift(twt_group_configs, pdw_idx)
    sta_assigns = []
    counts = {g["group_id"]: 0 for g in sched["groups"]}
    for sid in range(num_sta):
        gid_raw = int(sta_groups[sid]) if sid < len(sta_groups) else 0
        gid = gid_raw % max(1, num_groups)  # wrap into the valid range
        sta_assigns.append({"sta_id": sid, "assigned_twt_group": gid, "enable_twt": 1})
        counts[gid] = counts.get(gid, 0) + 1
    for gc in twt_group_configs:
        gc["num_stas_assigned"] = counts.get(gc["group_id"], 0)
    return {
        "num_sta": num_sta,
        "num_active_twt_groups": num_groups,
        "action_timestamp_ms": 0,
        "pdw_duration_ms": d_ms,
        "twt_group_configs": twt_group_configs,
        "sta_group_assignments": sta_assigns,
    }


# --- The worker entry point. Mirrors wifi-simulation/spawn_worker.py::run_episode. ---
def run_episode(
    worker_id: int,
    seed: int,
    policy_payload_bytes: bytes,
    result_queue,
    ns3_path: str,
    reward_type: str = "balanced",
    warmup_steps: int = 5,  # settle apps/queues/energy before recording (2->5, 2026-06-02)
    max_steps: int = 1000,
    deterministic: bool = False,
    quiet_ns3: bool = True,
    disable_ns3_traces: bool = True,
    progress_queue=None,
    capture_components: bool = False,
    n_stations: int = None,
    n_rehd: int = None,
    rand_seed: int = None,
):
    """
    Run ONE TWT episode in this (newly spawned) process.

    Args:
        worker_id: int, identifier for logging.
        seed: int, drives both NS-3 RNG and unique SHM segment names
              (TWTWrapper derives MySeg_{seed} etc. from this).
        policy_payload_bytes: pickle.dumps({"name": str, "kwargs": dict,
              "state_dict": dict}) — see twt_models/. If None, the worker
              uses the registry default ("mlp_ppo") with random init.
        result_queue: multiprocessing.Queue to push the final rollout to.
        ns3_path: path to the ns-3.44 root (the dir containing ./ns3).
        reward_type: reward preset name (forwarded to reward_functions).
        warmup_steps: NS-3 warmup steps with default action before recording.
        max_steps: hard cap on episode length (NS-3 should end first).
    """
    # Redirect FD 1/2 to /dev/null when quiet_ns3=True.
    # Must happen BEFORE the heavy imports below (torch/twt_models) and before TWTWrapper spawns NS-3, so the inherited child FDs go to /dev/null too.
    # Catches:
    #   - TF/CUDA stub noise from torch's transitive deps
    #   - PyTorch UserWarning (TransformerEncoder enable_nested_tensor)
    #   - ns3ai_utils prints (Experiment initialized/destroyed/Running ns-3...)
    #   - ./ns3 wrapper chatter (ninja status)
    #   - NS-3 child stdout (inherited because show_output=True)
    # Worker ends with os._exit(0) so no need to restore.
    # Errors still reach the orchestrator via result_queue (traceback is captured into the dict).
    if quiet_ns3:
        sys.stdout.flush()
        sys.stderr.flush()
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 1)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)

    rollout = []
    episode_reward = 0.0
    bootstrap_value = 0.0
    wrapper = None
    seg_name = f"MySeg_{seed}"

    # Timing breakdown — populated as we go, sent back in the result dict
    t_start = time.time()
    timing = {
        "imports_s": 0.0,
        "policy_load_s": 0.0,
        "wrapper_init_s": 0.0,
        "boot_s": 0.0,  # wrapper.reset() = NS-3 startup until first env
        "warmup_step_s": [],
        "real_step_s": [],
        "bootstrap_s": 0.0,
        "total_pre_close_s": 0.0,  # everything except wrapper.close()
    }

    try:
        # --- Path & import setup (must happen INSIDE the worker process) ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)  # twt/
        sys.path.insert(0, parent_dir)
        sys.path.insert(0, os.path.join(parent_dir, "exploration-scripts"))
        sys.path.insert(0, script_dir)

        # Pre-clean stale shm
        if os.path.exists(f"/dev/shm/{seg_name}"):
            try:
                os.remove(f"/dev/shm/{seg_name}")
            except OSError:
                pass

        os.chdir(ns3_path)

        # Heavy imports come AFTER path setup
        t_imports_start = time.time()
        import torch

        # Cap Torch intra-op threads in THIS worker (runtime call — reliable even if torch was already imported during spawn bootstrap, unlike the env-var setdefault at module top).
        # Child-only: the parent never calls this, so its PPO update keeps full threading.
        # Stops 6 workers' inference from starving NS-3.
        torch.set_num_threads(1)
        from twt_models import get_policy
        from pb_twt_wrapper_py import TWTWrapper
        from reward_functions import get_reward_function

        timing["imports_s"] = time.time() - t_imports_start

        # --- Build policy from the payload (registry-based, modular) ---
        t_policy_start = time.time()
        if policy_payload_bytes is not None:
            payload = pickle.loads(policy_payload_bytes)
            policy = get_policy(payload["name"], **payload.get("kwargs", {}))
            policy.load_state_dict(payload["state_dict"])
        else:
            # Fallback: default architecture with random init
            policy = get_policy("mlp_ppo")
        policy.eval()
        recurrent_state = policy.initial_recurrent_state()
        timing["policy_load_s"] = time.time() - t_policy_start

        # --- Action tables + reward fn ---
        schedule_table, assignment_table = load_action_tables(parent_dir)
        num_schedules = len(schedule_table["schedules"])
        num_assignments = len(assignment_table["assignments"])
        reward_fn = get_reward_function(reward_type)

        # Inlined from file_comm_env.py.legacy to keep the worker self-contained.
        def compute_sta_deltas(prev_env, curr_env):
            curr_list = curr_env.get("sta_observations", [])
            prev_list = prev_env.get("sta_observations", []) if prev_env else []
            out = []
            for i, c in enumerate(curr_list):
                p = prev_list[i] if i < len(prev_list) else {}
                co = c.get("oracle", {})
                po = p.get("oracle", {}) if p else {}
                cr = c.get("realistic", {})
                pr = p.get("realistic", {}) if p else {}
                cur_at = _clean_positive(cr.get("airtime_used_us", 0))
                prv_at = _clean_positive(pr.get("airtime_used_us", 0))
                cur_aw = _clean_positive(co.get("awake_time_ms", 0))
                prv_aw = _clean_positive(po.get("awake_time_ms", 0))
                cur_sl = _clean_positive(co.get("sleep_time_ms", 0))
                prv_sl = _clean_positive(po.get("sleep_time_ms", 0))
                cur_pk = _clean_positive(co.get("packets_transmitted", 0))
                prv_pk = _clean_positive(po.get("packets_transmitted", 0))
                cur_by = _clean_positive(co.get("bytes_transmitted", 0))
                prv_by = _clean_positive(po.get("bytes_transmitted", 0))
                cur_en = _clean_positive(co.get("total_energy_consumed_mj", 0))
                prv_en = _clean_positive(po.get("total_energy_consumed_mj", 0))
                cur_dr = _clean_positive(co.get("mpdu_drops_expired", 0))
                prv_dr = _clean_positive(po.get("mpdu_drops_expired", 0))
                cur_eq = _clean_positive(co.get("packets_enqueued", 0))
                prv_eq = _clean_positive(po.get("packets_enqueued", 0))
                # --- v3: real MAC queue-sojourn latency (per-step avg over pkts served this step) ---
                cur_qds = _clean_positive(co.get("queue_delay_sum_ms", 0))
                prv_qds = _clean_positive(po.get("queue_delay_sum_ms", 0))
                cur_qdc = _clean_positive(co.get("queue_delay_count", 0))
                prv_qdc = _clean_positive(po.get("queue_delay_count", 0))
                d_qds = max(0.0, cur_qds - prv_qds)
                d_qdc = max(0.0, cur_qdc - prv_qdc)
                # --- v3: REHD capacitor energy (vcap_max_v>0 is the REHD tell; self-zeros for non-REHD) ---
                cur_hv = _clean_positive(co.get("harvested_total_j", 0))
                prv_hv = _clean_positive(po.get("harvested_total_j", 0))
                cur_cs = _clean_positive(co.get("consumed_total_j", 0))
                prv_cs = _clean_positive(po.get("consumed_total_j", 0))
                out.append(
                    {
                        "delta_airtime_used_us": max(0.0, cur_at - prv_at),
                        "delta_awake_time_ms": max(0.0, cur_aw - prv_aw),
                        "delta_sleep_time_ms": max(0.0, cur_sl - prv_sl),
                        "delta_packets_transmitted": max(0.0, cur_pk - prv_pk),
                        "bsr_queue_index": _clean_positive(
                            cr.get("bsr_queue_ac_be", 0), 255
                        ),
                        "fcs_error_count": _clean_positive(
                            cr.get("fcs_error_count", 0)
                        ),
                        "rx_fragment_count": _clean_positive(
                            cr.get("rx_fragment_count", 0)
                        ),
                        "last_rx_timestamp_us": _clean_positive(
                            cr.get("last_rx_timestamp_us", 0)
                        ),
                        "duty_cycle": float(
                            np.clip(_clean_value(co.get("duty_cycle", 0)), 0, 1.0)
                        ),
                        "delta_bytes_transmitted": max(0.0, cur_by - prv_by),
                        "delta_energy_mj": max(0.0, cur_en - prv_en),
                        "delta_drops_expired": max(0.0, cur_dr - prv_dr),
                        "delta_packets_enqueued": max(0.0, cur_eq - prv_eq),
                        "queue_size_bytes": _clean_positive(
                            co.get("queue_size_bytes", 0), 1e8
                        ),
                        "queue_size_packets": _clean_positive(
                            co.get("queue_size_packets", 0)
                        ),
                        # v3 multi-objective inputs:
                        "step_latency_ms": (d_qds / d_qdc) if d_qdc > 0 else 0.0,
                        "latency_count": d_qdc,
                        "vcap_max_v": _clean_positive(co.get("vcap_max_v", 0)),
                        "vcap_v": _clean_positive(co.get("vcap_v", 0)),
                        "delta_harvested_j": max(0.0, cur_hv - prv_hv),
                        "delta_consumed_j": max(0.0, cur_cs - prv_cs),
                    }
                )
            return out

        # --- Start NS-3 via the wrapper (one episode, one wrapper) ---
        t_winit_start = time.time()
        wrapper = TWTWrapper(
            log_dir=os.path.join(parent_dir, "results", "data-log"),
            enable_logging=False,
            verbose=False,
            quiet_ns3=quiet_ns3,
            disable_ns3_traces=disable_ns3_traces,
            n_stations=n_stations,
            n_rehd=n_rehd,
        )
        timing["wrapper_init_s"] = time.time() - t_winit_start

        # boot = wrapper.reset() — this spawns NS-3 and waits for first env
        t_boot_start = time.time()
        env_dict = wrapper.reset(seed=seed, rand_seed=rand_seed)
        timing["boot_s"] = time.time() - t_boot_start
        num_sta = env_dict.get("num_sta", 16)

        # --- Warmup (no recording) ---
        # Track prev_env across warmup steps so that after the loop, prev_env points to the env from the SECOND-TO-LAST warmup step and env_dict to the env from the LAST.
        # This way, the first main-loop iteration's obs has REAL deltas (delta_airtime / delta_awake / delta_sleep / delta_packets), not zeros.
        # Without this, prev_env == env_dict on rollout step 0 -> all 4 delta features are zero -> agent's first trained decision is made on partial obs.
        warm_action = build_action_dict(0, 0, schedule_table, assignment_table, num_sta)
        prev_env = (
            env_dict  # = empty env from wrapper.reset(); only used if warmup_steps==0
        )
        for _ in range(warmup_steps):
            t_step_start = time.time()
            prev_env = env_dict  # env BEFORE this step
            env_dict, done = wrapper.step(warm_action)
            timing["warmup_step_s"].append(time.time() - t_step_start)
            if done:
                raise RuntimeError("episode ended during warmup")
        # Postcondition (warmup_steps >= 1): prev_env != env_dict, both reflect real network state (the empty reset env was consumed by warmup step 1).

        # --- Main rollout loop ---
        is_raw = bool(getattr(policy, "is_raw_action_policy", False))
        is_set_obs = bool(getattr(policy, "is_set_obs_policy", False))
        is_asym = bool(getattr(policy, "is_asymmetric_critic", False))
        obs_builder = build_ppo_obs_2d if is_set_obs else build_ppo_observation
        for step_idx in range(max_steps):
            # Build PPO observation. num_sta resolves from env_dict (no hardcode).
            obs = obs_builder(prev_env, env_dict)
            # Privileged oracle features for the asymmetric critic (critic-only).
            critic_obs = (
                build_critic_features(prev_env, env_dict, num_sta) if is_asym else None
            )

            if is_raw:
                # Raw-action policies (e.g. pointer_ppo): the policy emits a
                # rich payload that bypasses the (schedule, assignment) lookup.
                payload, log_prob, value, recurrent_state = policy.act_raw(
                    obs, recurrent_state, deterministic=deterministic
                )
                action_dict = build_action_dict_raw(payload, schedule_table, num_sta)
                action_for_reward = (int(payload["sched"]), payload["sta_groups"])
                step_record = {
                    "obs": obs,
                    "payload": payload,
                    "log_prob": log_prob,
                    "value": value,
                }
            else:
                # Indexed policies (mlp_ppo, lstm_ppo, random, analytical_*).
                # act() now returns a 3rd action head: pdw_idx (PDW level).
                # critic_obs kwarg only for asymmetric-critic policies (others' act()
                # signatures don't accept it — e.g. fixed/random/analytical/mlp).
                if is_asym:
                    sched_idx, assign_idx, pdw_idx, log_prob, value, recurrent_state = (
                        policy.act(
                            obs,
                            recurrent_state,
                            deterministic=deterministic,
                            critic_obs=critic_obs,
                        )
                    )
                else:
                    sched_idx, assign_idx, pdw_idx, log_prob, value, recurrent_state = (
                        policy.act(obs, recurrent_state, deterministic=deterministic)
                    )
                sched_idx = max(0, min(sched_idx, num_schedules - 1))
                assign_idx = max(0, min(assign_idx, num_assignments - 1))
                pdw_idx = max(0, min(int(pdw_idx), NUM_PDW_LEVELS - 1))
                action_dict = build_action_dict(
                    sched_idx,
                    assign_idx,
                    schedule_table,
                    assignment_table,
                    num_sta,
                    pdw_idx=pdw_idx,
                )
                action_for_reward = (sched_idx, assign_idx)
                step_record = {
                    "obs": obs,
                    "schedule": sched_idx,
                    "assignment": assign_idx,
                    "pdw": pdw_idx,
                    "log_prob": log_prob,
                    "value": value,
                }
                if is_asym:
                    step_record["critic_obs"] = critic_obs

            t_step_start = time.time()
            next_env, done = wrapper.step(action_dict)
            timing["real_step_s"].append(time.time() - t_step_start)

            # Reward = the effect of THIS action: the window env_dict (state at decision time, pre-action) -> next_env (post-action) that action_dict actually drove.
            # FIXED 2026-06-21: was compute_sta_deltas(prev_env, env_dict) — the PREVIOUS window, a one-step reward lag that mis-credited a_t with a_{t-1}'s outcome.
            # This now matches the eval (twt_eval_structured) and the EDA collector, which already score each action's own window.
            # The obs/critic_obs above stay on (prev_env, env_dict): those describe the decision-time state s_t, not a reward.
            # Terminal step: next_env is {} -> empty deltas -> reward 0, with done=True; GAE handles the boundary via the done flag (no bootstrap past a terminal state).
            sta_deltas = compute_sta_deltas(env_dict, next_env or {})
            reward_result = reward_fn(sta_deltas, action=action_for_reward)
            reward = float(reward_result.get("total", 0.0))
            episode_reward += reward

            step_record["reward"] = reward
            step_record["done"] = bool(done)
            if capture_components:
                step_record["reward_components"] = dict(
                    reward_result.get("components", {})
                )
            rollout.append(step_record)

            if done or not next_env:
                break
            prev_env = env_dict
            env_dict = next_env

        # Bootstrap value V(s_T) for GAE: 0 if last transition is terminal, otherwise one extra forward pass on the next observation.
        t_bs_start = time.time()
        if rollout and not rollout[-1]["done"] and next_env is not None:
            try:
                next_obs = obs_builder(env_dict, next_env)
                if is_raw:
                    _, _, bootstrap_value, _ = policy.act_raw(
                        next_obs, recurrent_state, deterministic=deterministic
                    )
                elif is_asym:
                    next_critic = build_critic_features(env_dict, next_env, num_sta)
                    _, _, _, _, bootstrap_value, _ = policy.act(
                        next_obs,
                        recurrent_state,
                        deterministic=deterministic,
                        critic_obs=next_critic,
                    )
                else:
                    # act() returns (sched, assign, pdw, log_prob, value, rstate)
                    _, _, _, _, bootstrap_value, _ = policy.act(
                        next_obs, recurrent_state, deterministic=deterministic
                    )
            except Exception:
                bootstrap_value = 0.0
        else:
            bootstrap_value = 0.0
        timing["bootstrap_s"] = time.time() - t_bs_start
        timing["total_pre_close_s"] = time.time() - t_start

        # Aggregate per-episode metrics from the final env dict (cumulative since NS-3 simulation start).
        # Used by the evaluator for reporting.
        final_env = env_dict if env_dict else {}
        sta_list = final_env.get("sta_observations", [])
        total_bytes_tx = 0
        total_energy_mj = 0.0
        total_drops = 0
        total_queue_bytes = 0
        for sta in sta_list:
            o = sta.get("oracle", {}) or {}
            total_bytes_tx += int(o.get("bytes_transmitted", 0) or 0)
            total_energy_mj += float(o.get("total_energy_consumed_mj", 0) or 0)
            total_drops += int(o.get("mpdu_drops_expired", 0) or 0)
            total_queue_bytes += int(o.get("queue_size_bytes", 0) or 0)
        avg_queue_bytes = total_queue_bytes / max(len(sta_list), 1)
        sim_time_sec = float(final_env.get("simulation_time_sec", 0) or 0)

        # Push success result
        result_queue.put(
            {
                "worker_id": worker_id,
                "seed": seed,
                "rollout": rollout,
                "bootstrap_value": float(bootstrap_value),
                "episode_reward": episode_reward,
                "episode_length": len(rollout),
                "metrics": {
                    "total_bytes_tx": total_bytes_tx,
                    "total_energy_mj": total_energy_mj,
                    "total_drops": total_drops,
                    "avg_queue_bytes": avg_queue_bytes,
                    "sim_time_sec": sim_time_sec,
                },
                "timing": timing,
                "success": True,
            }
        )

    except Exception as e:
        tb = traceback.format_exc()
        timing["total_pre_close_s"] = time.time() - t_start
        try:
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "seed": seed,
                    "rollout": rollout,
                    "bootstrap_value": 0.0,
                    "episode_reward": episode_reward,
                    "episode_length": len(rollout),
                    "timing": timing,
                    "success": False,
                    "error": f"{e}\n{tb}",
                }
            )
        except Exception:
            pass

    finally:
        # Kill NS-3 child + clean SHM (mirroring wifi-simulation/spawn_worker.py)
        try:
            if wrapper is not None:
                wrapper.close()
        except Exception:
            pass
        for nm in (
            f"/dev/shm/MySeg_{seed}",
            f"/dev/shm/MyCpp2PyMsg_{seed}",
            f"/dev/shm/MyPy2CppMsg_{seed}",
            f"/dev/shm/MyLockable_{seed}",
        ):
            if os.path.exists(nm):
                try:
                    os.remove(nm)
                except OSError:
                    pass
        try:
            # FLUSH the result before os._exit — mp.Queue.put() is async (a feeder thread streams the multi-MB rollout through a 64KB pipe).
            # The old cancel_join_thread() DISCARDED any unflushed result whenever the parent wasn't mid-get() (the rolling pool's parent spends seconds pickling refill spawns) -> rollouts silently lost, and a feeder killed while holding the queue write-lock bricks the queue for all later workers (the 2026-06-12 training stall: 17 zombies, 5/24 results).
            # close()+join_thread() blocks until the feeder drains; the watchdog guarantees we still exit if the parent died and nobody ever drains the pipe.
            import threading

            threading.Timer(120.0, lambda: os._exit(1)).start()
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        # Skip Python's normal shutdown (Boost static destructor would hang)
        os._exit(0)
