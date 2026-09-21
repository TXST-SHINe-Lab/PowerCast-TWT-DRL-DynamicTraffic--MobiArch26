#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models/mlp_ppo.py - MLP PPO policy with three independent Categorical heads.

Action space: MultiDiscrete([num_schedules, num_assignments, num_pdw_levels])  (24 x 25 x 10 with the shipped tables)
Observation:  Box(obs_dim,)  (num_sta x 7; sized per topology by twt_spawn_worker.default_obs_kwargs)

Architecture (faithful port of legacy train_ppo_V1.py, which used SB3's
PPO with a custom EnhancedFeatureExtractor + net_arch=dict(pi=[256,256],
vf=[256,256]) + activation_fn=GELU + ortho_init=True):

    features (mirrors EnhancedFeatureExtractor):
        Linear(obs_dim, 256) -> LayerNorm -> GELU
     -> Linear(256, 256)     -> LayerNorm -> GELU
     -> Linear(256, features_dim) -> GELU

    pi_mlp / vf_mlp (separate trunks, mirrors SB3's net_arch):
        Linear(features_dim, 256) -> GELU
     -> Linear(256, 256)          -> GELU

    heads:
        schedule_head    = Linear(256, num_schedules)
        assignment_head  = Linear(256, num_assignments)
        pdw_head         = Linear(256, num_pdw_levels)
        value_head       = Linear(256, 1)

All Linear weights use orthogonal init with gain sqrt(2); biases are zero.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base import BasePolicy
from . import register_policy


@register_policy
class MlpPpoPolicy(BasePolicy):
    name = "mlp_ppo"

    def __init__(
        self,
        obs_dim: int = 112,  # 16 STA × 7 feat fallback; default_obs_kwargs overrides per topology
        num_schedules: int = 40,
        num_assignments: int = 40,
        num_pdw_levels: int = 15,
        hidden_dim: int = 256,
        features_dim: Optional[int] = None,
        net_arch: Optional[List[int]] = None,
        features_per_sta: int = 7,
        obs_norm: bool = True,
    ):
        """
        Args:
            obs_dim:        observation length (num_sta * 7 features).
            num_schedules:  size of the schedule action head.
            num_assignments size of the assignment action head.
            num_pdw_levels: size of the PDW action head (pdw_end in {5,10,...,50} -> 10 with the shipped decoder).
            hidden_dim:     convenience knob; if features_dim/net_arch are not
                            given, both default to (hidden_dim, [hidden_dim]*2).
            features_dim:   output dim of the feature extractor (defaults to hidden_dim).
            net_arch:       list of pi/vf MLP hidden sizes (defaults to [hidden_dim, hidden_dim]).
        """
        super().__init__()
        if features_dim is None:
            features_dim = hidden_dim
        if net_arch is None:
            net_arch = [hidden_dim, hidden_dim]
        net_arch = list(net_arch)

        self.obs_dim = obs_dim
        self.num_schedules = num_schedules
        self.num_assignments = num_assignments
        self.num_pdw_levels = num_pdw_levels
        self.hidden_dim = hidden_dim
        self.features_dim = features_dim
        self.net_arch = net_arch

        # ----- Feature extractor (3 Linear layers; LN on the first two) -----
        self.features = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, features_dim),
            nn.GELU(),
        )

        # ----- Separate pi / vf MLPs (mirrors SB3's default net_arch behavior) -----
        self.pi_mlp = self._build_mlp(features_dim, net_arch)
        self.vf_mlp = self._build_mlp(features_dim, net_arch)
        head_in = net_arch[-1] if net_arch else features_dim

        # ----- Heads -----
        self.schedule_head = nn.Linear(head_in, num_schedules)
        self.assignment_head = nn.Linear(head_in, num_assignments)
        self.pdw_head = nn.Linear(head_in, num_pdw_levels)
        self.value_head = nn.Linear(head_in, 1)

        # ----- Orthogonal init (gain sqrt(2)) -----
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)

        # ----- Per-feature running obs normalization (buffers ride in state_dict) -----
        self.features_per_sta = int(features_per_sta)
        self.init_obs_norm(features_per_sta, enabled=bool(obs_norm))

    @staticmethod
    def _build_mlp(in_dim: int, sizes: List[int]) -> nn.Module:
        layers = []
        prev = in_dim
        for h in sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        return nn.Sequential(*layers) if layers else nn.Identity()

    def _forward(self, obs: torch.Tensor):
        obs = self._normalize_obs(
            obs
        )  # per-feature z-score (no-op if obs_norm disabled)
        feats = self.features(obs)
        pi_h = self.pi_mlp(feats)
        vf_h = self.vf_mlp(feats)
        return (
            self.schedule_head(pi_h),
            self.assignment_head(pi_h),
            self.pdw_head(pi_h),
            self.value_head(vf_h).squeeze(-1),
        )

    @torch.no_grad()
    def act(
        self,
        obs_np,
        recurrent_state: Optional[object] = None,
        deterministic: bool = False,
    ) -> Tuple[int, int, int, float, float, Optional[object]]:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        sl, al, pl, v = self._forward(obs_t)
        sd = Categorical(logits=sl)
        ad = Categorical(logits=al)
        pd = Categorical(logits=pl)
        if deterministic:
            s = sl.argmax(dim=-1)
            a = al.argmax(dim=-1)
            p = pl.argmax(dim=-1)
        else:
            s = sd.sample()
            a = ad.sample()
            p = pd.sample()
        log_prob = sd.log_prob(s) + ad.log_prob(a) + pd.log_prob(p)
        return (
            int(s.item()),
            int(a.item()),
            int(p.item()),
            float(log_prob.item()),
            float(v.item()),
            None,
        )

    def evaluate(
        self,
        obs_t: torch.Tensor,
        sched_t: torch.Tensor,
        assign_t: torch.Tensor,
        pdw_t: torch.Tensor,
        recurrent_state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sl, al, pl, v = self._forward(obs_t)
        sd = Categorical(logits=sl)
        ad = Categorical(logits=al)
        pd = Categorical(logits=pl)
        log_prob = sd.log_prob(sched_t) + ad.log_prob(assign_t) + pd.log_prob(pdw_t)
        entropy = sd.entropy() + ad.entropy() + pd.entropy()
        return log_prob, v, entropy
