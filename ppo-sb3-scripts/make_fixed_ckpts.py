#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
make_fixed_ckpts.py — emit fixed-policy checkpoints for the schedule
sensitivity diagnostic.

Writes a self-describing .pt for each (sched, asgn) combo so that
twt_evaluator.py --compare can load them like any trained model.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twt_models import get_policy

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixed_ckpts")
os.makedirs(OUT_DIR, exist_ok=True)

# (label, sched_idx, asgn_idx, pdw_idx) — pick combos that span the action table.
# pdw_idx∈[0,15) → PDW end = 5+pdw_idx·5 ms (0 = PDW off).
# See exploration-scripts/{schedule,assignment}_table.json for full names.
#
# STALE INDICES: these combos were sized to the 40x40x15 table this diagnostic was written against.
# The shipped table has since been carved down (24 schedules x 25 assignments x 10 PDW levels); several sched/asgn indices below (e.g. sched=35, asgn=39) exceed that range.
# Construction succeeds regardless (FixedPolicy stores the ints unvalidated), so the failure is silent until a checkpoint using an out-of-range index is actually stepped in an episode.
# Re-derive against the current tables before relying on this script.
#
# Two blocks:
#   (A) LEGACY refs (PDW off) — schedule-sensitivity diagnostic, kept for continuity.
#   (B) TOUGH Tier-1 shortlist — diverse static actions auto-selected from the post-PDW EDA (best (assign,pdw) per schedule tier + PDW sweep on the strong 4-group + global top-3). Spans g1→g8, duty t5→t88, D∈{5,10,15,25,50,75} so best-in-hindsight-per-seed (computed at eval) has a winner for every scenario.
COMBOS = [
    # (A) legacy (PDW off)
    ("s2_g1_75ms_all0", 2, 0, 0),
    ("s0_g1_min_all0", 0, 0, 0),
    ("s5_g2_eskew_seq", 5, 11, 0),
    ("s10_g3_eq_rr3", 10, 5, 0),
    ("s15_g4_eq_rr4", 15, 6, 0),  # strong legacy baseline
    ("s15_g4_eq_all0", 15, 0, 0),  # waste-sensitivity ref
    ("s25_g6_eq_rr6", 25, 8, 0),
    ("s35_g8_eq_rr8", 35, 10, 0),
    # (B) tough Tier-1 shortlist (EDA-derived, PDW-aware)
    ("t1_s0_a19_D75", 0, 19, 14),
    ("t1_s0_a26_D50", 0, 26, 9),
    ("t1_s0_a39_D75", 0, 39, 14),
    ("t1_s5_a37_D15", 5, 37, 2),
    ("t1_s10_a26_D50", 10, 26, 9),
    ("t1_s11_a26_D50", 11, 26, 9),
    ("t1_s15_a6_D50", 15, 6, 9),
    ("t1_s15_a27_D5", 15, 27, 0),
    ("t1_s15_a27_D10", 15, 27, 1),
    ("t1_s15_a39_D75", 15, 39, 14),
    ("t1_s22_a33_D25", 22, 33, 4),
    ("t1_s25_a33_D25", 25, 33, 4),
    ("t1_s35_a33_D25", 35, 33, 4),
]

for label, sched, asgn, pdw in COMBOS:
    policy = get_policy("fixed_policy", sched=sched, asgn=asgn, pdw=pdw)
    payload = {
        "name": "fixed_policy",
        "kwargs": {"sched": sched, "asgn": asgn, "pdw": pdw},
        "state_dict": policy.state_dict(),
    }
    out = os.path.join(OUT_DIR, f"fixed_{label}.pt")
    torch.save(payload, out)
    print(f"wrote {out}  (sched={sched}, asgn={asgn}, pdw={pdw})")
