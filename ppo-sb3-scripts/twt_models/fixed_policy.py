#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models/fixed_policy.py — Constant-action baseline.

Always returns the same (schedule_idx, assignment_idx) regardless of
observation. Used for the schedule-sensitivity diagnostic: hand-pick a few
(sched, asgn) combos and measure the spread in network metrics under fixed
seeds. If the spread is small, the env doesn't reward thoughtful scheduling
and PPO has nothing to learn.

Constructor kwargs:
    sched: int — fixed schedule index (0..num_schedules-1)
    asgn:  int — fixed assignment index (0..num_assignments-1)
    pdw:   int — fixed PDW level (0..num_pdw_levels-1); 0 = PDW off (default).
                 Lets the diagnostic also probe a fixed Power Delivery Window.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import BasePolicy
from . import register_policy


@register_policy
class FixedPolicy(BasePolicy):
    name = "fixed_policy"

    def __init__(
        self,
        obs_dim: int = 288,
        num_schedules: int = 40,
        num_assignments: int = 40,
        num_pdw_levels: int = 15,
        sched: int = 0,
        asgn: int = 0,
        pdw: int = 0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_schedules = num_schedules
        self.num_assignments = num_assignments
        self.num_pdw_levels = num_pdw_levels
        self.sched = int(sched)
        self.asgn = int(asgn)
        self.pdw = int(pdw)
        self.register_buffer("_unused", torch.zeros(1))

    def act(
        self,
        obs_np,
        recurrent_state: Optional[object] = None,
        deterministic: bool = False,
    ):
        return self.sched, self.asgn, self.pdw, 0.0, 0.0, None

    def evaluate(self, obs_t, sched_t, assign_t, pdw_t, recurrent_state=None):
        B = obs_t.shape[0]
        z = torch.zeros(B, dtype=torch.float32, device=obs_t.device)
        return z, z, z
