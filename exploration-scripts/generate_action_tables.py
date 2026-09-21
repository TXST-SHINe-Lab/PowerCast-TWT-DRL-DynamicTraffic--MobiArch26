#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
exploration-scripts/generate_action_tables.py
=============================================
CARVED action grid (2026-06-21), built in three steps:

  1. Lay down a SYSTEMATIC, UNIFORM grid over the whole action space — a
     deterministic lattice, no random sampling, nothing left unexplored.
  2. Run the EDA over this grid to map the per-action landscape under the
     current physics.
  3. CARVE a performance-shaped table: prune the dead zones, densify the hot
     spots (finer segments near consistently-good actions). The action space is
     RE-ALLOCATED, never grown.

Step 1 was a 30 x 25 lattice (K = 1..10 x 3 budgets; all_to_one, round_robin, split_n,
weighted front/back, interleave). This file emits the step-3 carve: 24 schedules x 25
assignments, 24 x 25 x 10 = 6,000 joint actions with the PDW head.

SCHEDULES (24) — K in SCHED_KS x BUDGET_FRACS of the per-K feasible maximum
  B_max(K) = MAX_SPAN - (K-1)*gap, split into K equal-duration service periods.

ASSIGNMENTS (25):
  round_robin N in RR_NS           x4   (spread comms, no isolation group)
  split_n     N in SPLIT_NS        x9   (contiguous blocks; the last isolates the high-index REHD tail)
  weighted    m in HARVEST_TAILS,  x12  (last m STAs -> 1 harvest group,
              c in HARVEST_COMMS          the other 20 - m -> c comms groups)

apply_assignment_pattern() (runtime helper, imported by
ppo-sb3-scripts/twt_spawn_worker.build_action_dict) expands a pattern into (sta_id, group_id) pairs.
"""

import os
import json
import argparse
from datetime import datetime

# --- grid parameters (single source of truth for the lattice) ---
BEACON_INTERVAL_MS = 102.4
MAX_SPAN_MS = 95.0  # last SP end (sum durations + (K-1)*gap) must fit here
MIN_GROUP_DURATION_MS = 2.0  # floor each group at >= 2 ms
MIN_OFFSET_GAP_MS = 2.0  # guard between consecutive SPs
MAX_K = 16  # C++ ceiling MAX_NUM_TWT_GROUPS=16
# CARVED grid (2026-06-21, EDA-shaped from runs/eda_full_20260621).
# The prior coarse 30x25 grid was run through the variable-split EDA; this carve keeps the proven winners and prunes the dominated/dead actions (a strict SUBSET of EDA-tested actions — no new mechanism, no re-validation needed):
#   schedules : K in SCHED_KS x budget BUDGET_FRACS  (K=1 dead, K>=7 cold, frac=0.30 dead)
#   assignment: split_n SPLIT_NS (contiguous REHD-tail isolator; EDA-best 0.76-0.81 REHD served across ALL REHD counts) + round_robin RR_NS (congestion-bound fallback: spread comms, no isolation group). all_to_one / interleave / front-back weighted DROPPED (scatter REHDs into busy SPs / dominated in welfare); weighted is reused below for explicit REHD-tail cuts.
#   PDW       : all 10 levels kept (best PDW shifts 0->9 with REHD count = the key lever).
SCHED_KS = [2, 3, 4, 5, 6, 8]  # hot K=2..6 (peak 3-4) + K=8 hi-REHD fine isolation
BUDGET_FRACS = [
    0.45,
    0.60,
    0.75,
    0.90,
]  # 4 budget levels in the good range (0.30 dead -> dropped)
SPLIT_NS = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # contiguous REHD-tail isolators (EDA winner)
RR_NS = [2, 3, 4, 5]  # congestion-bound spread fallback
# REHD-tail isolation CUTS via the (already tested) `weighted` mechanism: the last `m` STAs (the contiguous REHD tail) -> ONE dedicated harvest group; the first (20-m) comms STAs spread across `c` groups.
# SAME contiguous-isolation mechanism as split_n (the EDA winner) but with an EXPLICIT REHD-group size, so the agent can size the cut to ANY inferred REHD count M=4..16 (the variable-boundary coverage).
# No new pattern_type / no new C++/decoder.
HARVEST_TAILS = [4, 6, 8, 10, 12, 14]
HARVEST_COMMS = [2, 3]


# --- schedules ---
def _bmax(K):
    """Max total wake budget (sum of durations) so that wake+gaps fit in MAX_SPAN."""
    return MAX_SPAN_MS - (K - 1) * MIN_OFFSET_GAP_MS


def _equal_durations(K, frac):
    """K equal-duration groups summing to ~frac*B_max(K), each >= MIN_GROUP."""
    B = max(K * MIN_GROUP_DURATION_MS, frac * _bmax(K))
    d = max(MIN_GROUP_DURATION_MS, round(B / K))
    durs = [float(d)] * K
    # Guard: rounding can nudge the span over MAX_SPAN at the high-budget / high-K corner — shave the per-group duration until wake+gaps fit.
    while (
        sum(durs) + (K - 1) * MIN_OFFSET_GAP_MS > MAX_SPAN_MS
        and d > MIN_GROUP_DURATION_MS
    ):
        d -= 1
        durs = [float(d)] * K
    return durs


def _build_schedule_dict(sid, K, durations, frac):
    groups = []
    off = 0.0
    for gi, d in enumerate(durations):
        groups.append(
            {
                "group_id": gi,
                "wake_duration_ms": float(d),
                "sp_offset_ms": float(off),
            }
        )
        off += d + MIN_OFFSET_GAP_MS
    total = int(sum(durations))
    return {
        "schedule_id": sid,
        "name": f"S{sid}_G{K}_b{int(round(frac * 100))}",
        "description": f"{K}g equal, durations={[int(x) for x in durations]}, total={total}ms",
        "num_groups": K,
        "total_duration_ms": total,
        "groups": groups,
    }


def gen_schedules():
    schedules = []
    sid = 0
    for K in SCHED_KS:
        for frac in BUDGET_FRACS:
            durations = _equal_durations(K, frac)
            schedules.append(_build_schedule_dict(sid, K, durations, frac))
            sid += 1
    return schedules


# --- assignments ---
def gen_assignments():
    A = []

    def add(d):
        d["assignment_id"] = len(A)
        d["name"] = f"A{len(A)}_{d.pop('_tag')}"
        A.append(d)

    # round_robin RR_NS — congestion-bound fallback: cyclic spread of comms, NO dedicated REHD group (best non-REHD served; the right move when load is airtime-bound, not energy-bound).
    # Kept small (2..5); higher-n round_robin scatters the REHD tail.
    for n in RR_NS:
        add(
            {
                "_tag": f"rr_{n}",
                "description": f"Round-robin {n}-group (i mod {n})",
                "pattern_type": "round_robin",
                "num_groups": n,
                "max_groups_needed": n,
            }
        )

    # split_n contiguous blocks SPLIT_NS — the REHD-tail ISOLATOR and EDA welfare winner: since REHDs are the contiguous high-index tail, the last block(s) capture them so they sleep/harvest together instead of burning RX in a busy shared SP.
    # EDA: 0.76-0.81 REHD served across ALL REHD counts M=4..16 (vs ~0.64 scatter / ~0.58 all_to_one).
    for n in SPLIT_NS:
        add(
            {
                "_tag": f"seq_{n}",
                "description": f"Sequential block split into {n} (REHD-tail isolator)",
                "pattern_type": "split_n",
                "num_groups": n,
                "ordering": "sequential",
                "max_groups_needed": n,
            }
        )

    # REHD-tail isolation cuts (weighted mechanism, EXPLICIT tail size) — see HARVEST_* above.
    # Densifies the proven contiguous-isolation winner so the agent can pick the REHD-group size to match any inferred REHD count, decoupled from the equal-split granularity.
    for m in HARVEST_TAILS:
        for c in HARVEST_COMMS:
            comms = round((20 - m) / 20.0 / c, 3)
            ratios = [comms] * c + [round(m / 20.0, 3)]
            add(
                {
                    "_tag": f"cut{m}t_{c}c",
                    "description": f"REHD-tail cut: last {m} STAs -> 1 harvest group; first {20 - m} -> {c} comms groups",
                    "pattern_type": "weighted",
                    "ratios": ratios,
                    "max_groups_needed": c + 1,
                }
            )

    return A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    schedules = gen_schedules()
    assignments = gen_assignments()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "schedule_table.json"), "w") as f:
        json.dump(
            {
                "description": (
                    "CARVED grid: K in %s x %d budget levels of the per-K feasible "
                    "maximum, equal-duration SPs. Systematic lattice, EDA-swept, "
                    "dominated regions pruned." % (SCHED_KS, len(BUDGET_FRACS))
                ),
                "generated": ts,
                "beacon_interval_ms": BEACON_INTERVAL_MS,
                "max_total_duration_ms": MAX_SPAN_MS,
                "max_k": MAX_K,
                "budget_fracs": BUDGET_FRACS,
                "num_schedules": len(schedules),
                "schedules": schedules,
            },
            f,
            indent=2,
        )

    with open(os.path.join(args.output_dir, "assignment_table.json"), "w") as f:
        json.dump(
            {
                # This string previously advertised all_to_one + interleave, which the carve had already pruned -- it is the only machine-readable description of the table, so a stale list here propagates into anything that reads it.
                "description": (
                    "CARVED assignment grid: round_robin(%s) + split_n(%s) + "
                    "weighted REHD-tail cuts (tails %s x %s comms groups)."
                    % (RR_NS, SPLIT_NS, HARVEST_TAILS, HARVEST_COMMS)
                ),
                "generated": ts,
                "max_k": MAX_K,
                "num_assignments": len(assignments),
                "assignments": assignments,
            },
            f,
            indent=2,
        )

    # --- coverage diagnostics ---
    ng = [s["num_groups"] for s in schedules]
    tot = [s["total_duration_ms"] for s in schedules]
    print(f"Wrote {len(schedules)} schedules + {len(assignments)} assignments")
    print(
        f"Joint (x10 PDW): {len(schedules) * len(assignments) * 10} combos "
        f"({len(schedules)}x{len(assignments)}x10)"
    )
    print("\n=== Schedule coverage ===")
    by_g = {}
    for v in ng:
        by_g[v] = by_g.get(v, 0) + 1
    print(f"  by num_groups (K):   {dict(sorted(by_g.items()))}")
    print(
        f"  total_duration min/mean/max: {min(tot)} / {sum(tot)/len(tot):.1f} / {max(tot)} ms"
    )
    print("  per-K budgets (ms):")
    for K in SCHED_KS:
        ds = [s["total_duration_ms"] for s in schedules if s["num_groups"] == K]
        print(f"    K={K:2d}: {ds}")
    print("\n=== Assignment coverage ===")
    by_t = {}
    for a in assignments:
        by_t[a["pattern_type"]] = by_t.get(a["pattern_type"], 0) + 1
    print(f"  by pattern_type: {dict(sorted(by_t.items()))}")


# RUNTIME helper (imported by ppo-sb3-scripts/twt_spawn_worker.build_action_dict): expands an assignment-pattern dict into concrete (sta_id, group_id) tuples.
# Kept here alongside the action tables; independent of table generation above.
def apply_assignment_pattern(pattern, num_sta, num_available_groups):
    """
    Apply an assignment pattern to generate concrete STA-to-group mappings.

    Args:
        pattern: Assignment pattern dict
        num_sta: Number of STAs to assign
        num_available_groups: Number of groups available from schedule

    Returns:
        List of (sta_id, group_id) tuples
    """
    pattern_type = pattern["pattern_type"]
    assignments = []

    # Cap the groups needed by what's available
    max_groups = min(pattern.get("max_groups_needed", 1), num_available_groups)

    if pattern_type == "all_to_one":
        target = pattern.get("target_group", 0) % num_available_groups
        for sta_id in range(num_sta):
            assignments.append((sta_id, target))

    elif pattern_type == "round_robin":
        num_groups = min(pattern.get("num_groups", 2), num_available_groups)
        for sta_id in range(num_sta):
            assignments.append((sta_id, sta_id % num_groups))

    elif pattern_type == "split_half":
        ordering = pattern.get("ordering", "sequential")
        if ordering == "sequential":
            half = num_sta // 2
            for sta_id in range(num_sta):
                group = 0 if sta_id < half else min(1, num_available_groups - 1)
                assignments.append((sta_id, group))
        else:  # even_odd
            for sta_id in range(num_sta):
                group = 0 if sta_id % 2 == 0 else min(1, num_available_groups - 1)
                assignments.append((sta_id, group))

    elif pattern_type == "split_n":
        num_groups = min(pattern.get("num_groups", 2), num_available_groups)
        per_group = num_sta // num_groups
        for sta_id in range(num_sta):
            group = min(sta_id // max(1, per_group), num_groups - 1)
            assignments.append((sta_id, group))

    elif pattern_type == "modulo":
        num_groups = min(pattern.get("num_groups", 2), num_available_groups)
        for sta_id in range(num_sta):
            assignments.append((sta_id, sta_id % num_groups))

    elif pattern_type == "weighted":
        ratios = pattern.get("ratios", [1.0])
        # Adjust ratios if we have fewer groups available
        if len(ratios) > num_available_groups:
            # Redistribute excess to last available group
            new_ratios = ratios[: num_available_groups - 1]
            new_ratios.append(sum(ratios[num_available_groups - 1 :]))
            ratios = new_ratios

        # Normalize ratios
        total_ratio = sum(ratios)
        normalized = [r / total_ratio for r in ratios]

        # Assign STAs based on ratios
        sta_idx = 0
        for group_id, ratio in enumerate(normalized):
            count = int(round(ratio * num_sta))
            for _ in range(count):
                if sta_idx < num_sta:
                    assignments.append((sta_idx, group_id))
                    sta_idx += 1

        # Handle any remaining STAs
        while sta_idx < num_sta:
            assignments.append((sta_idx, len(normalized) - 1))
            sta_idx += 1

    elif pattern_type == "interleave":
        step = pattern.get("step", 2)
        num_groups = min(pattern.get("num_groups", 2), num_available_groups)
        for sta_id in range(num_sta):
            group = (sta_id // step) % num_groups
            assignments.append((sta_id, group))

    else:
        # Default: all to group 0
        for sta_id in range(num_sta):
            assignments.append((sta_id, 0))

    return assignments


if __name__ == "__main__":
    main()
