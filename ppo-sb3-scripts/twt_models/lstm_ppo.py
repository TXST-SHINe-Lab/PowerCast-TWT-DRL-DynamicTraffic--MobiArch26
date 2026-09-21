#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models/lstm_ppo.py - LSTM-PPO policy with three independent Categorical heads.

Action space: MultiDiscrete([num_schedules, num_assignments, num_pdw_levels])  (24 x 25 x 10 with the shipped tables)
Observation:  Box(obs_dim,)  (num_sta x 7; sized per topology by twt_spawn_worker.default_obs_kwargs)

Architecture (port of legacy train_lstm_ppo_V1.py, which used sb3-contrib's
RecurrentPPO with policy="MlpLstmPolicy" and policy_kwargs:
    lstm_hidden_size=256, n_lstm_layers=1,
    shared_lstm=False, enable_critic_lstm=True,
    net_arch=dict(pi=[256,256], vf=[256,256])):

    pi_lstm: LSTM(obs_dim, lstm_hidden, num_layers=n_lstm_layers, batch_first=True)
    vf_lstm: LSTM(obs_dim, lstm_hidden, num_layers=n_lstm_layers, batch_first=True)

    pi_mlp / vf_mlp (separate trunks after each LSTM):
        Linear(lstm_hidden, 256) -> Tanh
     -> Linear(256, 256)         -> Tanh

    heads:
        schedule_head    = Linear(256, num_schedules)
        assignment_head  = Linear(256, num_assignments)
        pdw_head         = Linear(256, num_pdw_levels)
        value_head       = Linear(256, 1)

Note on activation: the legacy RecurrentPPO used the SB3 default
`activation_fn=nn.Tanh` (not GELU like train_ppo_V1.py), so this port uses
Tanh too.

================================================================
Recurrent training is now supported (sequence-level minibatching).
================================================================
This policy sets `is_temporal_recurrent_policy = True`, which routes the
PPO update through twt_batch_orchestrator._ppo_update_recurrent. That path
keeps each worker's episode intact and calls evaluate_sequence() to roll
the pi/vf LSTMs through the whole T-step trajectory in one shot. The
hidden states at update time are reproduced by re-rolling from h_0 = 0
because the policy weights are fixed during a rollout.

The legacy evaluate() (zero-state per single step) is kept only for the
flat-shuffle path; it is not used when this policy is selected. act()
threads recurrent_state correctly during rollout, as before.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base import BasePolicy
from . import register_policy

# Recurrent state convention: tuple (h_pi, c_pi, h_vf, c_vf), each of shape (n_lstm_layers, B=1, lstm_hidden).
# The B=1 batch dim is preserved across act() calls so we can also pass to the LSTM directly.


@register_policy
class LstmPpoPolicy(BasePolicy):
    name = "lstm_ppo"
    is_temporal_recurrent_policy = True

    def __init__(
        self,
        obs_dim: int = 112,  # 16 STA × 7 feat fallback; default_obs_kwargs overrides per topology
        num_schedules: int = 40,
        num_assignments: int = 40,
        num_pdw_levels: int = 15,
        lstm_hidden: int = 256,
        n_lstm_layers: int = 1,
        net_arch: Optional[List[int]] = None,
        features_per_sta: int = 7,
        obs_norm: bool = True,
        critic_extra_dim: int = 0,
        critic_features_per_sta: int = 0,
    ):
        super().__init__()
        if net_arch is None:
            net_arch = [256, 256]
        net_arch = list(net_arch)

        self.obs_dim = obs_dim
        self.num_schedules = num_schedules
        self.num_assignments = num_assignments
        self.num_pdw_levels = num_pdw_levels
        self.lstm_hidden = lstm_hidden
        self.n_lstm_layers = n_lstm_layers
        self.net_arch = net_arch

        # ----- Asymmetric critic: vf path also ingests privileged oracle features -----
        # The ACTOR (pi path) stays on realistic obs only; the CRITIC (vf path) gets obs ++ per-STA oracle energy state, so V() can value the state through which the delayed PDW reward flows.
        # critic_extra_dim = num_sta * critic_features_per_sta.
        self.critic_extra_dim = int(critic_extra_dim)
        self.critic_features_per_sta = int(critic_features_per_sta)
        self.is_asymmetric_critic = self.critic_extra_dim > 0
        vf_in_dim = obs_dim + self.critic_extra_dim

        # ----- Two separate LSTMs (pi / vf) -----
        # shared_lstm=False, enable_critic_lstm=True in the legacy.
        self.pi_lstm = nn.LSTM(
            obs_dim, lstm_hidden, num_layers=n_lstm_layers, batch_first=True
        )
        self.vf_lstm = nn.LSTM(
            vf_in_dim, lstm_hidden, num_layers=n_lstm_layers, batch_first=True
        )

        # ----- Post-LSTM MLPs with Tanh (matches SB3 RecurrentPPO default) -----
        self.pi_mlp = self._build_mlp(lstm_hidden, net_arch)
        self.vf_mlp = self._build_mlp(lstm_hidden, net_arch)
        head_in = net_arch[-1] if net_arch else lstm_hidden

        # ----- Heads -----
        self.schedule_head = nn.Linear(head_in, num_schedules)
        self.assignment_head = nn.Linear(head_in, num_assignments)
        self.pdw_head = nn.Linear(head_in, num_pdw_levels)
        self.value_head = nn.Linear(head_in, 1)

        # ----- Orthogonal init for Linear; default LSTM init -----
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                nn.init.zeros_(m.bias)

        # ----- Per-feature running obs normalization (buffers ride in state_dict) -----
        self.features_per_sta = int(features_per_sta)
        self.init_obs_norm(features_per_sta, enabled=bool(obs_norm))
        # Separate running normalizer for the critic's privileged oracle features.
        if self.is_asymmetric_critic:
            self.init_critic_norm(self.critic_features_per_sta, enabled=bool(obs_norm))

    @staticmethod
    def _build_mlp(in_dim: int, sizes: List[int]) -> nn.Module:
        layers = []
        prev = in_dim
        for h in sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        return nn.Sequential(*layers) if layers else nn.Identity()

    # --- Recurrent-state helpers ---
    def _zero_state(self, batch_size: int = 1, device=None):
        device = device or next(self.parameters()).device
        h_pi = torch.zeros(
            self.n_lstm_layers, batch_size, self.lstm_hidden, device=device
        )
        c_pi = torch.zeros(
            self.n_lstm_layers, batch_size, self.lstm_hidden, device=device
        )
        h_vf = torch.zeros(
            self.n_lstm_layers, batch_size, self.lstm_hidden, device=device
        )
        c_vf = torch.zeros(
            self.n_lstm_layers, batch_size, self.lstm_hidden, device=device
        )
        return (h_pi, c_pi, h_vf, c_vf)

    def initial_recurrent_state(self):
        """Override BasePolicy default; non-None for recurrent policies."""
        return self._zero_state(batch_size=1)

    # --- Single-step inference (rollout time) ---
    def _vf_input(self, obs_t_norm: torch.Tensor, critic_np) -> torch.Tensor:
        """Build the critic LSTM input: normalized obs, concatenated with the
        normalized privileged oracle features when the critic is asymmetric.
        obs_t_norm is already normalized and shaped (..., obs_dim); the returned
        tensor matches that leading shape with last dim = obs_dim (+ critic_extra_dim).
        """
        if not self.is_asymmetric_critic or critic_np is None:
            return obs_t_norm
        crit_t = (
            torch.as_tensor(critic_np, dtype=torch.float32)
            .reshape(*obs_t_norm.shape[:-1], self.critic_extra_dim)
            .to(obs_t_norm.device)
        )
        crit_t = self._normalize_critic(crit_t)
        return torch.cat([obs_t_norm, crit_t], dim=-1)

    @torch.no_grad()
    def act(
        self,
        obs_np,
        recurrent_state: Optional[object] = None,
        deterministic: bool = False,
        critic_obs=None,
    ) -> Tuple[int, int, int, float, float, Optional[object]]:
        if recurrent_state is None:
            recurrent_state = self._zero_state(batch_size=1)
        h_pi, c_pi, h_vf, c_vf = recurrent_state

        # Single time-step: (1, 1, obs_dim)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).view(1, 1, -1)
        obs_t = self._normalize_obs(obs_t)  # per-feature z-score (no-op if disabled)
        vf_in = self._vf_input(obs_t, critic_obs)  # obs (++ oracle) for the critic

        pi_out, (h_pi, c_pi) = self.pi_lstm(obs_t, (h_pi, c_pi))
        vf_out, (h_vf, c_vf) = self.vf_lstm(vf_in, (h_vf, c_vf))

        pi_h = self.pi_mlp(pi_out.squeeze(1))
        vf_h = self.vf_mlp(vf_out.squeeze(1))

        sl = self.schedule_head(pi_h)
        al = self.assignment_head(pi_h)
        pl = self.pdw_head(pi_h)
        v = self.value_head(vf_h).squeeze(-1)

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

        new_state = (h_pi, c_pi, h_vf, c_vf)
        return (
            int(s.item()),
            int(a.item()),
            int(p.item()),
            float(log_prob.item()),
            float(v.item()),
            new_state,
        )

    # --- Batched evaluation for the PPO update ---
    def evaluate(
        self,
        obs_t: torch.Tensor,
        sched_t: torch.Tensor,
        assign_t: torch.Tensor,
        pdw_t: torch.Tensor,
        recurrent_state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per-step evaluate with zero initial hidden state.

        See the file-level docstring for the orchestrator-level limitation:
        the current PPO update flat-shuffles minibatches across trajectories,
        so we cannot thread per-step recurrent states correctly. We therefore
        treat each step as standalone with zero initial state. The LSTM still
        backpropagates gradients through the one-step forward pass, but does
        not learn to use accumulated history during training.
        """
        B = obs_t.shape[0]
        device = obs_t.device
        h_pi = torch.zeros(self.n_lstm_layers, B, self.lstm_hidden, device=device)
        c_pi = torch.zeros(self.n_lstm_layers, B, self.lstm_hidden, device=device)
        h_vf = torch.zeros(self.n_lstm_layers, B, self.lstm_hidden, device=device)
        c_vf = torch.zeros(self.n_lstm_layers, B, self.lstm_hidden, device=device)

        obs_t = self._normalize_obs(obs_t)  # per-feature z-score (no-op if disabled)
        x = obs_t.unsqueeze(1)  # (B, 1, obs_dim)
        pi_out, _ = self.pi_lstm(x, (h_pi, c_pi))
        vf_out, _ = self.vf_lstm(x, (h_vf, c_vf))

        pi_h = self.pi_mlp(pi_out.squeeze(1))
        vf_h = self.vf_mlp(vf_out.squeeze(1))

        sl = self.schedule_head(pi_h)
        al = self.assignment_head(pi_h)
        pl = self.pdw_head(pi_h)
        v = self.value_head(vf_h).squeeze(-1)

        sd = Categorical(logits=sl)
        ad = Categorical(logits=al)
        pd = Categorical(logits=pl)
        log_prob = sd.log_prob(sched_t) + ad.log_prob(assign_t) + pd.log_prob(pdw_t)
        entropy = sd.entropy() + ad.entropy() + pd.entropy()
        return log_prob, v, entropy

    # --- Sequence evaluation (proper LSTM roll over a whole episode) ---
    # Used by the orchestrator's recurrent PPO update path.
    # Equivalent to threading act()'s recurrent state through the whole trajectory.
    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        sched_seq: torch.Tensor,
        assign_seq: torch.Tensor,
        pdw_seq: torch.Tensor,
        recurrent_state: Optional[object] = None,
        critic_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs_seq:    (T, obs_dim) — chronological observations from one episode.
        sched_seq:  (T,) int     — schedule indices taken during rollout.
        assign_seq: (T,) int     — assignment indices taken during rollout.
        pdw_seq:    (T,) int     — PDW levels taken during rollout (3rd head).
        critic_seq: (T, critic_extra_dim) — privileged oracle features for the
                    critic (asymmetric-critic only; None for symmetric).

        Returns (log_prob_seq, value_seq, entropy_seq), each (T,).

        Re-rolls pi_lstm and vf_lstm from h_0 = c_0 = 0 through the whole
        episode. Because the policy weights are fixed during rollout, this
        reproduces exactly the hidden states that act() saw — no need to
        save per-step states in the rollout buffer.
        """
        if recurrent_state is None:
            h_pi, c_pi, h_vf, c_vf = self._zero_state(
                batch_size=1, device=obs_seq.device
            )
        else:
            h_pi, c_pi, h_vf, c_vf = recurrent_state

        # batch=1, time=T
        obs_seq = self._normalize_obs(
            obs_seq
        )  # per-feature z-score (no-op if disabled)
        x = obs_seq.unsqueeze(0)  # (1, T, obs_dim)
        vf_x = self._vf_input(x, critic_seq)  # (1, T, obs_dim ++ critic_extra_dim)
        pi_out, _ = self.pi_lstm(x, (h_pi, c_pi))  # (1, T, lstm_hidden)
        vf_out, _ = self.vf_lstm(vf_x, (h_vf, c_vf))  # (1, T, lstm_hidden)

        pi_h = self.pi_mlp(pi_out.squeeze(0))  # (T, head_in)
        vf_h = self.vf_mlp(vf_out.squeeze(0))  # (T, head_in)

        sl = self.schedule_head(pi_h)  # (T, num_schedules)
        al = self.assignment_head(pi_h)  # (T, num_assignments)
        pl = self.pdw_head(pi_h)  # (T, num_pdw_levels)
        v = self.value_head(vf_h).squeeze(-1)  # (T,)

        sd = Categorical(logits=sl)
        ad = Categorical(logits=al)
        pd = Categorical(logits=pl)
        log_prob = (
            sd.log_prob(sched_seq) + ad.log_prob(assign_seq) + pd.log_prob(pdw_seq)
        )  # (T,)
        entropy = sd.entropy() + ad.entropy() + pd.entropy()  # (T,)
        return log_prob, v, entropy
