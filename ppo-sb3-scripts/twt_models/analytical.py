#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models/analytical.py - Self-contained analytical baselines + registry adapters.

Ports the analytical model-based policies (queuing/energy/throughput math) into
the twt_models/ registry so the new evaluator/orchestrator can use them via
`--policy-arch <name>`. Each analytical policy is wrapped as a `BasePolicy`
subclass; the underlying decision logic is unchanged.

Registered names (adapters wrap each analytical policy as a BasePolicy):
    "analytical_demand"      -> AnalyticalDemandPolicy
    "analytical_md1"         -> AnalyticalMD1Policy (the paper's M/D/1 baseline)
    "analytical_md1_k6"      -> AnalyticalMD1RestrictedPolicy (K<=6 ablation)

Usage at the CLI:
    python3.11 twt_evaluator.py --policy-arch analytical_md1 \
        --reward-preset twt_pf_demand_v5 --n-episodes 10 --num-workers 4

Or compared head-to-head:
    python3.11 twt_evaluator.py --compare \
        checkpoints/twt_lstm_ppo_twt_pf_demand_v5_<ts>/ckpt_final.pt \
        analytical_md1 random_policy \
        --reward-preset twt_pf_demand_v5 --n-episodes 10 --num-workers 4
"""

from typing import Optional, Tuple

import numpy as np
import torch

from .base import BasePolicy
from . import register_policy

# --- SYSTEM CONSTANTS (from NS-3 simulation) ---


class WiFi6Constants:
    """Physical and MAC layer constants for 802.11ax WiFi 6."""

    # Beacon Interval
    BEACON_INTERVAL_MS = 102.4  # Standard TWT beacon interval

    # MCS and Rate
    MCS_INDEX = 5  # Default MCS for 802.11ax
    PHY_RATE_MBPS = 86.7  # Approximate for MCS 5, 20MHz, 1SS

    # Power model (mW)
    P_TX_MW = 250.0  # Transmit power
    P_RX_MW = 150.0  # Receive power
    P_IDLE_MW = 100.0  # Idle awake power
    P_SLEEP_MW = 5.0  # Sleep power
    P_ACTIVE_MW = (P_TX_MW + P_RX_MW + P_IDLE_MW) / 3  # Average active

    # A-MPDU aggregation
    MAX_AMPDU_SIZE = 65535  # bytes
    TYPICAL_AMPDU_FRAMES = 32

    # Per-packet overhead
    MAC_OVERHEAD_US = 200  # MAC layer overhead per MPDU
    DIFS_US = 34  # DCF Interframe Space
    SIFS_US = 16  # Short Interframe Space
    SLOT_US = 9  # Slot time

    # Contention window
    CW_MIN = 15
    CW_MAX = 1023

    # Frame sizes
    PACKET_SIZE_BYTES = 1472  # Typical UDP payload


def _load_schedule_table_for_analytical():
    """Load schedule_table.json once at module import and return the
    {idx: {groups, durations, total, name}} dict that analytical baselines use.
    Post-2026-05-25: replaces the previously-hardcoded 20-entry dict so this
    file stays in sync with the live action space (was silently routing
    analytical_throughput's sched-17 pick to a different schedule after the
    40x40 migration)."""
    import os, json

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(
        os.path.join(here, "..", "..", "exploration-scripts", "schedule_table.json")
    )
    with open(path) as f:
        raw = json.load(f)
    table = {}
    for s in raw["schedules"]:
        idx = int(s["schedule_id"])
        durations = [float(g["wake_duration_ms"]) for g in s["groups"]]
        table[idx] = {
            "groups": int(s["num_groups"]),
            "durations": durations,
            "total": float(s.get("total_duration_ms", sum(durations))),
            "name": str(s.get("name", f"S{idx}")),
        }
    return table


class TWT_Schedules:
    """TWT Schedule mappings. SCHEDULE_TABLE is loaded at module import from
    exploration-scripts/schedule_table.json (single source of truth)."""

    SCHEDULE_TABLE = _load_schedule_table_for_analytical()

    @classmethod
    def get_total_duration(cls, schedule_idx: int) -> float:
        """Get total wake duration in ms."""
        return cls.SCHEDULE_TABLE.get(schedule_idx, cls.SCHEDULE_TABLE[0])["total"]

    @classmethod
    def get_num_groups(cls, schedule_idx: int) -> int:
        """Get number of TWT groups."""
        return cls.SCHEDULE_TABLE.get(schedule_idx, cls.SCHEDULE_TABLE[0])["groups"]

    @classmethod
    def get_schedules_by_duration(
        cls, num_groups: int = None, ascending: bool = True
    ) -> list:
        """
        Get schedule indices sorted by total duration.

        Args:
            num_groups: Filter by number of groups (None = all)
            ascending: If True, shortest first (prefer shorter for airtime efficiency)

        Returns:
            List of (schedule_idx, total_duration, spec) tuples, sorted by duration
        """
        schedules = []
        for idx, spec in cls.SCHEDULE_TABLE.items():
            if num_groups is None or spec["groups"] == num_groups:
                schedules.append((idx, spec["total"], spec))

        schedules.sort(key=lambda x: x[1], reverse=not ascending)
        return schedules

    @classmethod
    def get_duty_cycle(cls, schedule_idx: int) -> float:
        """Calculate duty cycle (wake_time / beacon_interval)."""
        total = cls.get_total_duration(schedule_idx)
        return total / WiFi6Constants.BEACON_INTERVAL_MS


# --- REGISTRY ADAPTERS (BasePolicy interface) ---


# Lightweight stub so the impl classes' constructors stay happy.
# The impl classes' `predict()` does NOT actually use action_space — they look up schedule/assignment indices internally — but they accept it in their __init__ for SB3 compat, so we hand them a stub.
class _ActionSpaceStub:
    nvec = (40, 40)
    shape = (2,)


# --- DEMAND- & ENERGY-AWARE ANALYTICAL SCHEDULER ---
# The principled, ONLINE, hindsight-free heuristic baseline — a fair competitor for the learned policy (unlike a brute-force best-in-hindsight static).


def _load_assignment_meta():
    """Round-robin assignment index per group count + a single all_to_one idx,
    parsed from the live assignment_table.json (so the baseline stays in sync with
    the action space). Falls back to the known layout if the file can't be read."""
    import os, json

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(
        os.path.join(here, "..", "..", "exploration-scripts", "assignment_table.json")
    )
    rr, all_to_one = {}, 0
    try:
        a = json.load(open(path))["assignments"]
        first_a2o = None
        for i, x in enumerate(a):
            pt = x.get("pattern_type", "")
            if pt == "round_robin":
                rr[int(x.get("max_groups_needed", 0))] = i
            elif pt == "all_to_one" and first_a2o is None:
                first_a2o = i
        all_to_one = first_a2o or 0
    except Exception:
        rr = {2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10}
    return rr, all_to_one


class AnalyticalDemandPolicy:
    """Online, hindsight-free model-based TWT + PDW scheduler — the fair heuristic
    competitor for the learned policy. Reads the CURRENT 7-feature realistic obs
    over the FULL topology and drives all THREE action heads (schedule, assignment,
    Power Delivery Window) from demand + starvation signals. Unlike the legacy
    analytical_* baselines it (a) parses the live obs layout, (b) sees every STA,
    and (c) actually uses the RF-harvest PDW.

    Logic (realistic, AP-observable signals only — no oracle, no hindsight):
      1. LOAD -> airtime. Aggregate backlog (bsr_be) + under-service (silence) set
         a target duty cycle; pick the schedule whose total wake time is closest to
         it (biased to the chosen group count).
      2. HETEROGENEITY -> groups. Demand dispersion (CV of bsr_be) sets the number
         of TWT groups: uniform demand -> few groups; skewed -> more, so heavy and
         light STAs don't share a service period.
      3. ASSIGNMENT. Balanced round-robin across the chosen groups (fair sharing);
         a single group collapses to all-to-one.
      4. ENERGY -> PDW. SP-starvation (starv_rate) is the realistic tell of an
         energy-bound STA; its level opens a proportional Power Delivery Window so
         sleeping harvesters receive RF energy — the lever the legacy baselines
         never pulled.

    Restricted to the same action subspace the shipped policy used (first NUM_SCHED
    schedules / NUM_ASSIGN assignments / NUM_PDW PDW levels) for a fair comparison.
    """

    NUM_FEATURES = 7
    F_BSR, F_DPKTS, F_DAIR, F_DFCS, F_SNR, F_SILENCE, F_STARV = range(7)
    NUM_SCHED, NUM_ASSIGN, NUM_PDW = (
        24,
        25,
        10,
    )  # match the CARVED policy heads (2026-06-21)

    _RR_BY_GROUPS, _ALL_TO_ONE = _load_assignment_meta()

    def __init__(self, action_space, num_sta: int = 20):
        self.action_space = action_space
        self.num_sta = num_sta

    def _col(self, obs, fidx, n):
        return np.array([float(obs[i * self.NUM_FEATURES + fidx]) for i in range(n)])

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=float)
        n = max(1, len(obs) // self.NUM_FEATURES)  # full topology, derived from obs
        bsr = self._col(obs, self.F_BSR, n)
        sil = self._col(obs, self.F_SILENCE, n)
        starv = self._col(obs, self.F_STARV, n)

        # Warmup / all-zero obs: minimal airtime, no PDW.
        if not np.any(bsr) and not np.any(starv):
            return np.array([0, 0, 0]), None

        # 1) LOAD -> target duty cycle in [0.15, 0.90].
        load = float(np.clip(0.75 * np.mean(bsr) + 0.25 * np.mean(sil), 0.0, 1.0))
        target_duty = 0.15 + 0.75 * load

        # 2) HETEROGENEITY -> number of groups via demand dispersion (CV of backlog).
        cv = float(np.std(bsr)) / (float(np.mean(bsr)) + 1e-6)
        g_target = 1 if cv < 0.25 else 2 if cv < 0.5 else 3 if cv < 0.9 else 4
        g_target = max(1, min(g_target, n))

        # Pick the schedule (idx < NUM_SCHED) whose duty is closest to target, with a light bias toward the desired group count.
        best_idx, best_score = 0, float("inf")
        for idx, spec in TWT_Schedules.SCHEDULE_TABLE.items():
            if idx >= self.NUM_SCHED:
                continue
            duty = spec["total"] / WiFi6Constants.BEACON_INTERVAL_MS
            score = abs(duty - target_duty) + 0.05 * abs(spec["groups"] - g_target)
            if score < best_score:
                best_score, best_idx = score, idx
        sched_idx = best_idx
        actual_groups = TWT_Schedules.SCHEDULE_TABLE[sched_idx]["groups"]

        # 3) ASSIGNMENT: round-robin matched to the chosen group count; else all-to-one.
        if actual_groups <= 1 or not self._RR_BY_GROUPS:
            assign_idx = self._ALL_TO_ONE
        else:
            g = max(2, min(actual_groups, max(self._RR_BY_GROUPS)))
            assign_idx = self._RR_BY_GROUPS.get(g, self._ALL_TO_ONE)
        assign_idx = min(int(assign_idx), self.NUM_ASSIGN - 1)

        # 4) ENERGY -> PDW from SP-starvation (the realistic energy-bound tell).
        energy_need = float(np.clip(np.mean(starv), 0.0, 1.0))
        pdw_idx = max(
            0, min(self.NUM_PDW - 1, int(round(energy_need * (self.NUM_PDW - 1))))
        )

        return np.array([sched_idx, assign_idx, pdw_idx]), None


class _AnalyticalAdapter(BasePolicy):
    """
    Common adapter: wraps an analytical impl that exposes `predict(obs)`.

    Analytical baselines are non-trainable. They have no parameters, so
    state_dict round-trips are no-ops. We register a single zero buffer
    so torch.save / load_state_dict still succeed (uniform with the rest
    of the registry).
    """

    impl_class = None  # subclass overrides

    def __init__(self, num_sta: int = 16):
        super().__init__()
        if self.impl_class is None:
            raise ValueError(f"{type(self).__name__}: impl_class not set")
        self.num_sta = num_sta
        self._impl = self.impl_class(_ActionSpaceStub(), num_sta=num_sta)
        self.register_buffer("_unused", torch.zeros(1))

    def act(
        self,
        obs_np,
        recurrent_state: Optional[object] = None,
        deterministic: bool = False,
    ) -> Tuple[int, int, int, float, float, Optional[object]]:
        # Analytical policies are inherently deterministic — `deterministic` is accepted only for interface uniformity and is otherwise ignored.
        action, _ = self._impl.predict(np.asarray(obs_np), deterministic=deterministic)
        s = int(action[0])
        a = int(action[1])
        # Legacy analytical baselines return only [sched, assign] (no RF-energy model -> never request a Power Delivery Window, pdw=0).
        # The demand/energy-aware baseline returns a 3rd element: the PDW level.
        # log_prob and value are placeholders (these run only in the evaluator; rollouts never reach ppo_update).
        p = int(action[2]) if len(action) > 2 else 0
        return s, a, p, 0.0, 0.0, None

    def evaluate(
        self,
        obs_t: torch.Tensor,
        sched_t: torch.Tensor,
        assign_t: torch.Tensor,
        pdw_t: torch.Tensor,
        recurrent_state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stateless / non-trainable — return zeros at the right shape.
        # If you accidentally try to PPO-update one of these, the gradients are all zero and the optimizer is a no-op (correct semantics).
        B = obs_t.shape[0]
        device = obs_t.device
        zeros = torch.zeros(B, dtype=torch.float32, device=device)
        return zeros, zeros, zeros


@register_policy
class AnalyticalDemandAdapter(_AnalyticalAdapter):
    name = "analytical_demand"
    impl_class = AnalyticalDemandPolicy


# --- CLEAN M/D/1 BASELINE (2026-06-22 overhaul) ---
# The legacy analytical_queue is stale (old 13/18-feat obs via the broken ObservationParser, old 40x40 action space, no PDW).
# THIS is the fair, principled, ADAPTIVE queueing baseline for the carved 7-feature obs + 24x25x10 action space, recalibrated from the new EDA, reading ALL active STAs, with an energy-aware PDW rule (option B).
class AnalyticalMD1Policy:
    """M/D/1 queueing baseline, recalibrated for the current pipeline.

    Realistic-only (same 7-feature obs as the agent's actor; no oracle/energy obs),
    adaptive (re-decides every step from the live observation):
      SCHEDULE : M/D/1 queue feedback. `bsr_be` is the backlog; more backlog -> more
                 service rate needed to drive rho<1 -> higher wake-budget fraction.
                 Calibrated lambda=9.16, mu=6.85 pkts/BI/STA (EDA eda_full_20260621).
      GROUPS K : M/D/1 fairness split — high queue variance across STAs (cv) -> more
                 groups to isolate the congested.
      ASSIGN   : isolate (contiguous split_n) when load is unbalanced, else round-robin.
      PDW (E)  : energy-aware rule (option B) — infer REHD energy stress from `silence`
                 (rises with REHD count: 0.39 lo -> 0.63 hi) + `starv_rate`, and open
                 the power-delivery window proportionally.
    """

    NF = 7
    F_BSR, F_DPKTS, F_DAIR, F_DFCS, F_SNR, F_SIL, F_STV = range(7)
    LAMBDA, MU = 9.16, 6.85  # per-BI per-STA (EDA-calibrated; rho~1.34, over-loaded)
    Q_LO, Q_HI = 0.125, 0.50  # bsr_be backlog -> budget fraction
    CV_LO, CV_HI = 0.80, 1.20  # cv(bsr) across STAs -> #groups
    SIL_LO, SIL_HI = 0.39, 0.63  # silence -> PDW (lo..hi REHD energy stress)
    STV_BOOST = 0.55  # mean starvation above this -> +1 PDW level

    def __init__(self, action_space=None, num_sta: int = 16):
        self.num_sta = num_sta
        self._load_tables()

    def _load_tables(self):
        import os
        import json
        from collections import OrderedDict

        exp = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "exploration-scripts",
            )
        )
        sched = json.load(open(os.path.join(exp, "schedule_table.json")))["schedules"]
        assign = json.load(open(os.path.join(exp, "assignment_table.json")))[
            "assignments"
        ]
        self.n_pdw = 10
        byK = (
            OrderedDict()
        )  # schedules grouped by K, sorted by wake duration (budget-frac rank)
        for i, s in enumerate(sched):
            byK.setdefault(int(s["num_groups"]), []).append(
                (float(s["total_duration_ms"]), i)
            )
        self.K_list = list(byK.keys())
        self.sched_by_K = {K: [i for _, i in sorted(v)] for K, v in byK.items()}
        self.split_by_n, self.rr_by_n = {}, {}
        for i, a in enumerate(assign):
            n = int(a.get("num_groups", a.get("max_groups_needed", 1)))
            if a["pattern_type"] == "split_n":
                self.split_by_n[n] = i
            elif a["pattern_type"] == "round_robin":
                self.rr_by_n[n] = i

    @staticmethod
    def _nearest(lst, x):
        return min(lst, key=lambda v: abs(v - x))

    def _select_K(self, cv):
        """Group count from backlog dispersion: a RANK MAP over the FULL K ladder.

        The baseline sees every schedule the agent does, so "the baseline acts in the
        same action space" is literally true.

        A previous version filtered this ladder to the EDA's performing band (K<=6,
        "hot K=2..6, K>=7 cold"), locking the baseline out of all four K=8 schedules.
        Measured cost over 160 matched-scenario episodes: alpha-fair reward -1.1095 ->
        -1.0038 (+0.106) and Jain 0.7541 -> 0.7712 -- i.e. the restriction accounted for
        ~24% of the reported reward margin and ~14% of the fairness margin. That is not
        defensible in a head-to-head, so it is gone; see AnalyticalMD1RestrictedPolicy.

        This is a rank map, not a menu: kf in [0,1] is spread over len(ladder) levels, so
        naively appending a level would RESCALE every existing boundary (K=5 would start
        at cv=1.00 instead of 1.05), silently re-tuning the EDA-fit CV_LO/CV_HI. We keep
        the fitted 0.10-cv level width and drop the top clip instead, so every original
        boundary is preserved bit-for-bit and K=8 occupies a NEW band above the old
        (unbounded) K=6 one:

            K=2 cv<=0.85 | K=3 0.85-0.95 | K=4 0.95-1.05 | K=5 1.05-1.15  (unchanged)
            K=6 1.15-1.25 (was cv>=1.15) | K=8 cv>=1.25                   (new)
        """
        kf_raw = (cv - self.CV_LO) / (self.CV_HI - self.CV_LO)  # unclipped at the top
        ladder = list(self.K_list)  # every K in the table
        idx = int(round(kf_raw * 4.0))  # fitted level width
        return ladder[max(0, min(len(ladder) - 1, idx))]

    def predict(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=float).ravel()
        n = max(1, len(obs) // self.NF)
        M = obs[: n * self.NF].reshape(n, self.NF)
        q, sil, stv = M[:, self.F_BSR], M[:, self.F_SIL], M[:, self.F_STV]
        mean_q = float(np.mean(q))
        cv = float(np.std(q) / (np.mean(q) + 1e-6))
        mean_sil, mean_stv = float(np.mean(sil)), float(np.mean(stv))

        # SCHEDULE: backlog -> budget-fraction rank ; queue variance -> #groups K (hot range 2..6)
        frac01 = float(
            np.clip((mean_q - self.Q_LO) / (self.Q_HI - self.Q_LO), 0.0, 1.0)
        )
        K = self._select_K(cv)
        idxs = self.sched_by_K[K]
        sched_idx = idxs[int(round(frac01 * (len(idxs) - 1)))]

        # ASSIGNMENT: isolate (split_n~K) when unbalanced, else round-robin
        if cv >= 1.0 and self.split_by_n:
            assign_idx = self.split_by_n[self._nearest(list(self.split_by_n), K)]
        elif self.rr_by_n:
            assign_idx = self.rr_by_n[
                self._nearest(list(self.rr_by_n), min(K, max(self.rr_by_n)))
            ]
        else:
            assign_idx = 0

        # PDW: energy-aware (silence -> REHD stress, + starvation boost)
        pdw = (mean_sil - self.SIL_LO) / (self.SIL_HI - self.SIL_LO) * (self.n_pdw - 1)
        if mean_stv > self.STV_BOOST:
            pdw += 1.0
        pdw_idx = int(np.clip(round(pdw), 0, self.n_pdw - 1))

        return np.array([sched_idx, assign_idx, pdw_idx]), None


class AnalyticalMD1RestrictedPolicy(AnalyticalMD1Policy):
    """Ablation: M/D/1 with the group count clamped to the EDA's K<=6 band.

    Kept only to quantify what that restriction costs the baseline. Not the paper's
    baseline -- see AnalyticalMD1Policy._select_K.
    """

    def _select_K(self, cv):
        kf = float(np.clip((cv - self.CV_LO) / (self.CV_HI - self.CV_LO), 0.0, 1.0))
        hotK = [k for k in self.K_list if k <= 6] or self.K_list
        return hotK[int(round(kf * (len(hotK) - 1)))]


@register_policy
class AnalyticalMD1Adapter(_AnalyticalAdapter):
    name = "analytical_md1"
    impl_class = AnalyticalMD1Policy


@register_policy
class AnalyticalMD1RestrictedAdapter(_AnalyticalAdapter):
    name = "analytical_md1_k6"
    impl_class = AnalyticalMD1RestrictedPolicy
