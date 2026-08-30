#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models/random_policy.py - Uniform-random baseline policy.

Picks (schedule_idx, assignment_idx) uniformly at random, no learning.
Useful as a sanity-check baseline for the evaluator (and for verifying that
non-trivial policies actually beat random).

Has parameters (a no-op buffer) so torch.save / load_state_dict work the same
way as for trainable policies — keeps the registry / checkpoint flow uniform.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BasePolicy
from . import register_policy


@register_policy
class RandomPolicy(BasePolicy):
    name = "random_policy"

    def __init__(
        self,
        obs_dim: int = 288,
        num_schedules: int = 40,
        num_assignments: int = 40,
        num_pdw_levels: int = 15,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_schedules = num_schedules
        self.num_assignments = num_assignments
        self.num_pdw_levels = num_pdw_levels
        # log uniform prob over the joint (schedule, assignment, pdw) action.
        self._log_prob_const = (
            -math.log(num_schedules)
            - math.log(num_assignments)
            - math.log(num_pdw_levels)
        )
        self._max_entropy = (
            math.log(num_schedules)
            + math.log(num_assignments)
            + math.log(num_pdw_levels)
        )
        # A no-op buffer so the module isn't parameter-empty (some torch ops want at least one parameter; also makes state_dict round-trip uniformly).
        self.register_buffer("_unused", torch.zeros(1))

    def act(
        self,
        obs_np,
        recurrent_state: Optional[object] = None,
        deterministic: bool = False,
    ) -> Tuple[int, int, int, float, float, Optional[object]]:
        # `deterministic` is honored by always picking action (0, 0, 0); useful only as a degenerate floor — most callers use deterministic=False.
        if deterministic:
            s = 0
            a = 0
            p = 0
        else:
            s = int(np.random.randint(0, self.num_schedules))
            a = int(np.random.randint(0, self.num_assignments))
            p = int(np.random.randint(0, self.num_pdw_levels))
        return s, a, p, float(self._log_prob_const), 0.0, None

    def evaluate(
        self,
        obs_t: torch.Tensor,
        sched_t: torch.Tensor,
        assign_t: torch.Tensor,
        pdw_t: torch.Tensor,
        recurrent_state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stateless: this baseline isn't trainable, but evaluate() is required by BasePolicy.
        # Returns log_prob = log(1 / (S * A * P)), value = 0, entropy = log(S) + log(A) + log(P) (uniform max entropy).
        B = obs_t.shape[0]
        log_prob = torch.full(
            (B,), self._log_prob_const, dtype=torch.float32, device=obs_t.device
        )
        value = torch.zeros(B, dtype=torch.float32, device=obs_t.device)
        entropy = torch.full(
            (B,), self._max_entropy, dtype=torch.float32, device=obs_t.device
        )
        return log_prob, value, entropy
