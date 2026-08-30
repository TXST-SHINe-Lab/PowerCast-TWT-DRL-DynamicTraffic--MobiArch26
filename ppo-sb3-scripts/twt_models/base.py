#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models/base.py - Common interface every TWT policy must implement.

Inspired by the reward_functions.py registry pattern.
A new architecture is added by creating a new file in twt_models/ that
subclasses BasePolicy and uses the @register_policy decorator. No edits
to the orchestrator, worker, or evaluator are required.

Two action interfaces are supported:

  (1) Indexed (legacy, default; used by mlp_ppo, lstm_ppo, random_policy,
      analytical_*). The policy emits two integers (schedule_idx,
      assignment_idx) which the worker passes to build_action_dict() to
      look up the C++ action via the schedule_table / assignment_table.
      Subclasses implement act() and evaluate().
      Combined action space = num_schedules * num_assignments (e.g. 20 x 19 = 380).

  (2) Raw (used by pointer_ppo). The policy emits a richer payload that
      escapes the lookup table and produces per-STA group assignments
      directly. Subclasses set `is_raw_action_policy = True` and implement
      act_raw() and evaluate_raw(). The worker calls build_action_dict_raw()
      instead of build_action_dict().
      Combined action space ~= num_schedules * max_num_groups ** num_sta
      (e.g. 20 * 8^16 ~= 5.6e15).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BasePolicy(nn.Module, ABC):
    """
    Abstract policy interface for the parallel TWT pipeline.

    Subclasses must:
      - set the class attribute `name` to a unique string (registry key)
      - implement EITHER (act + evaluate) for the indexed interface, OR
        (act_raw + evaluate_raw) for the raw interface (and set
        `is_raw_action_policy = True`)
      - override initial_recurrent_state() if the architecture is recurrent
    """

    name: str = "base"  # subclasses MUST override
    is_raw_action_policy: bool = False  # subclasses override to True for raw
    is_temporal_recurrent_policy: bool = (
        False  # True if hidden state propagates across timesteps
    )
    is_set_obs_policy: bool = False
    # ^ True -> the worker calls build_ppo_obs_2d() to feed (num_sta, F) tensors to act() / evaluate() instead of the legacy flat (num_sta * F,) vector.
    # Set this on transformer / set-attention policies.

    is_asymmetric_critic: bool = False
    # ^ True -> the VALUE path additionally consumes privileged ORACLE features (per-STA energy state) that the actor never sees.
    # The worker builds those critic features (twt_spawn_worker.build_critic_features) and passes them to act(..., critic_obs=) and evaluate_sequence(..., critic_seq=); the policy normalizes them with the separate _crit_* running stats below.
    # Keeps the actor realistic (deployable) while giving the critic the state through which the delayed reward (e.g. PDW->harvest->future REHD service) actually flows, so the existing GAE bootstrapping can propagate that credit correctly.

    # --- Per-feature running observation normalization (2026-06-07) ---
    # We are NOT on SB3, so VecNormalize does not apply (it's a VecEnv wrapper and its per-flat-dimension stats would break the set encoder's permutation invariance).
    # We port its MECHANISM as PER-FEATURE running mean/std (shared across the N STA tokens) — see twt_normalizer.py.
    # The stats are registered BUFFERS, so they (a) ride in state_dict -> auto-propagate to the spawn workers (consistent act/evaluate normalization, no separate snapshot plumbing), and (b) auto-persist in the self-describing .pt checkpoint (the classic forgot-to-save-VecNormalize-stats bug can't happen).
    # Subclasses that want normalization call init_obs_norm(F) in __init__ and self._normalize_obs(x) at the top of their obs forward; the orchestrator warm-starts + updates the stats.
    def init_obs_norm(
        self, features_per_sta: int, clip: float = 5.0, enabled: bool = True
    ):
        self._obs_F = int(features_per_sta)
        self._obs_clip = float(clip)
        self._obs_norm_enabled = bool(enabled)
        self.register_buffer("_obs_mean", torch.zeros(self._obs_F))
        self.register_buffer("_obs_var", torch.ones(self._obs_F))
        self.register_buffer("_obs_count", torch.tensor(1.0e-4))

    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        """z-score per feature + clip. No-op if init_obs_norm() was never called.
        Works for flat (B, N*F), 2-D (B, N, F), or unbatched obs — the feature axis
        is the innermost F-block, so reshape(-1, F) groups it correctly."""
        if not getattr(self, "_obs_norm_enabled", False):
            return x
        orig = x.shape
        xr = x.reshape(-1, self._obs_F)
        z = (xr - self._obs_mean) / torch.sqrt(self._obs_var + 1e-8)
        z = torch.clamp(z, -self._obs_clip, self._obs_clip)
        return z.reshape(orig)

    @torch.no_grad()
    def update_obs_rms(self, obs_np):
        """Welford/Chan parallel update of the per-feature buffers from a batch of RAW
        (variance-stabilized, pre-normalization) obs. Call from the orchestrator AFTER
        the PPO update so the stats used during the update match what the workers used.
        """
        if not getattr(self, "_obs_norm_enabled", False):
            return
        x = np.asarray(obs_np, dtype=np.float64).reshape(-1, self._obs_F)
        if x.shape[0] == 0:
            return
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        mean = self._obs_mean.detach().cpu().numpy().astype(np.float64)
        var = self._obs_var.detach().cpu().numpy().astype(np.float64)
        cnt = float(self._obs_count.item())
        delta = bm - mean
        tot = cnt + bc
        new_mean = mean + delta * bc / tot
        m2 = var * cnt + bv * bc + delta * delta * cnt * bc / tot
        new_var = m2 / tot
        self._obs_mean.copy_(torch.as_tensor(new_mean, dtype=self._obs_mean.dtype))
        self._obs_var.copy_(torch.as_tensor(new_var, dtype=self._obs_var.dtype))
        self._obs_count.copy_(torch.as_tensor(tot, dtype=self._obs_count.dtype))

    @torch.no_grad()
    def warmstart_obs_rms(self, mean, std, count: float = 1.0e6):
        """Initialize the running stats from precomputed mean/std (the EDA warm-start),
        so normalization is calibrated from step 1 instead of cold-start noisy."""
        if not getattr(self, "_obs_norm_enabled", False):
            return
        self._obs_mean.copy_(
            torch.as_tensor(np.asarray(mean), dtype=self._obs_mean.dtype)
        )
        self._obs_var.copy_(
            torch.as_tensor(np.asarray(std) ** 2, dtype=self._obs_var.dtype)
        )
        self._obs_count.copy_(
            torch.as_tensor(float(count), dtype=self._obs_count.dtype)
        )

    # --- Per-feature running normalization for the asymmetric CRITIC's privileged oracle features ---
    # Separate buffers from the actor's obs_rms; same Welford mechanism.
    # Buffers ride in state_dict -> propagate to workers + persist in the .pt, exactly like obs_rms.
    # A policy with an asymmetric critic calls init_critic_norm(F_crit) in __init__ and self._normalize_critic(x) on the critic-feature input; the orchestrator calls update_critic_rms() each batch.
    def init_critic_norm(
        self, features_per_sta: int, clip: float = 5.0, enabled: bool = True
    ):
        self._crit_F = int(features_per_sta)
        self._crit_clip = float(clip)
        self._crit_norm_enabled = bool(enabled)
        self.register_buffer("_crit_mean", torch.zeros(self._crit_F))
        self.register_buffer("_crit_var", torch.ones(self._crit_F))
        self.register_buffer("_crit_count", torch.tensor(1.0e-4))

    def _normalize_critic(self, x: torch.Tensor) -> torch.Tensor:
        """z-score per critic feature + clip. No-op if init_critic_norm() never called."""
        if not getattr(self, "_crit_norm_enabled", False):
            return x
        orig = x.shape
        xr = x.reshape(-1, self._crit_F)
        z = (xr - self._crit_mean) / torch.sqrt(self._crit_var + 1e-8)
        z = torch.clamp(z, -self._crit_clip, self._crit_clip)
        return z.reshape(orig)

    @torch.no_grad()
    def update_critic_rms(self, crit_np):
        """Welford/Chan parallel update of the per-feature critic buffers from a batch
        of RAW critic features. Mirrors update_obs_rms()."""
        if not getattr(self, "_crit_norm_enabled", False):
            return
        x = np.asarray(crit_np, dtype=np.float64).reshape(-1, self._crit_F)
        if x.shape[0] == 0:
            return
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        mean = self._crit_mean.detach().cpu().numpy().astype(np.float64)
        var = self._crit_var.detach().cpu().numpy().astype(np.float64)
        cnt = float(self._crit_count.item())
        delta = bm - mean
        tot = cnt + bc
        new_mean = mean + delta * bc / tot
        m2 = var * cnt + bv * bc + delta * delta * cnt * bc / tot
        new_var = m2 / tot
        self._crit_mean.copy_(torch.as_tensor(new_mean, dtype=self._crit_mean.dtype))
        self._crit_var.copy_(torch.as_tensor(new_var, dtype=self._crit_var.dtype))
        self._crit_count.copy_(torch.as_tensor(tot, dtype=self._crit_count.dtype))

    # --- Indexed interface (legacy default — mlp_ppo / lstm_ppo / random / ...) ---
    @abstractmethod
    def act(
        self, obs_np, recurrent_state: Optional[Any] = None, deterministic: bool = False
    ) -> Tuple[int, int, int, float, float, Optional[Any]]:
        """
        Indexed-action rollout step.

        Args:
            obs_np: numpy 1-D array of shape (obs_dim,)
            recurrent_state: opaque state for recurrent policies, or None.
            deterministic: if True, pick the most likely action (argmax) instead
                of sampling. Used at evaluation time. Stochastic policies should
                still return a valid log_prob (e.g. log p(argmax)).

        Returns:
            (schedule_idx, assignment_idx, pdw_idx, log_prob, value,
             next_recurrent_state)
            pdw_idx is the Power Delivery Window level (3rd action head, Phase 2);
            log_prob is the JOINT log-prob over all three heads. Non-recurrent
            policies return None for next_recurrent_state. Baselines that don't
            control the PDW return pdw_idx=0 with a 0 log-prob contribution.
        """

    @abstractmethod
    def evaluate(
        self,
        obs_t: torch.Tensor,
        sched_t: torch.Tensor,
        assign_t: torch.Tensor,
        pdw_t: torch.Tensor,
        recurrent_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Indexed-action batched evaluation for PPO.

        Args:
            obs_t:    tensor of shape (B, obs_dim)
            sched_t:  tensor of shape (B,) with int schedule indices
            assign_t: tensor of shape (B,) with int assignment indices
            pdw_t:    tensor of shape (B,) with int PDW levels (3rd head)
            recurrent_state: optional recurrent state (rolled along time inside).

        Returns:
            (log_prob, value, entropy) — each tensor of shape (B,). log_prob and
            entropy are summed over all three action heads.
        """

    def initial_recurrent_state(self) -> Optional[Any]:
        """Override in recurrent policies. Default: stateless (returns None)."""
        return None

    # --- Sequence-level evaluation (used by the orchestrator for recurrent policies; rolls hidden state through one whole episode in chronological order rather than treating each step as independent) ---
    # Default implementation just delegates to flat evaluate(); this is correct for non-recurrent policies.
    # Recurrent policies MUST override to do a proper LSTM/GRU roll over the time dimension.
    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        sched_seq: torch.Tensor,
        assign_seq: torch.Tensor,
        pdw_seq: torch.Tensor,
        recurrent_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Indexed-action sequence evaluation for recurrent PPO updates.

        Args:
            obs_seq:     (T, obs_dim) — one whole episode in chronological order.
            sched_seq:   (T,) int — schedule indices taken at rollout time.
            assign_seq:  (T,) int — assignment indices taken at rollout time.
            pdw_seq:     (T,) int — PDW levels taken at rollout time (3rd head).
            recurrent_state: optional initial state; default zeros.

        Returns:
            (log_prob_seq, value_seq, entropy_seq), each (T,).
        """
        return self.evaluate(obs_seq, sched_seq, assign_seq, pdw_seq, recurrent_state)

    def evaluate_sequence_raw(
        self,
        obs_seq: torch.Tensor,
        payload_seq: Dict[str, torch.Tensor],
        recurrent_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Raw-action counterpart to evaluate_sequence(). Same shape contract:
        obs_seq is (T, obs_dim), payload_seq's tensors are batched over T,
        returns three (T,) tensors.
        """
        return self.evaluate_raw(obs_seq, payload_seq, recurrent_state)

    # --- Raw-action interface (override when is_raw_action_policy = True) ---
    def act_raw(
        self, obs_np, recurrent_state: Optional[Any] = None, deterministic: bool = False
    ) -> Tuple[Dict[str, Any], float, float, Optional[Any]]:
        """
        Raw-action rollout step.

        Returns:
            (payload, log_prob, value, next_recurrent_state)

            payload: dict consumed by the worker's build_action_dict_raw().
                For pointer_ppo this is
                    {"sched": int, "sta_groups": np.ndarray (num_sta,) int64,
                     "pdw": int}.
                "pdw" is the Power Delivery Window level (3rd action head). The
                orchestrator stores the payload verbatim in the rollout buffer;
                the same payload is later re-batched into tensors and fed to
                evaluate_raw() during the PPO update.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.act_raw() not implemented. "
            "Set is_raw_action_policy=True only when overriding act_raw()."
        )

    def evaluate_raw(
        self,
        obs_t: torch.Tensor,
        payload_batch: Dict[str, torch.Tensor],
        recurrent_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Raw-action batched evaluation for PPO.

        Args:
            obs_t:         tensor of shape (B, obs_dim)
            payload_batch: dict of stacked tensors. For pointer_ppo:
                {"sched":      LongTensor (B,),
                 "sta_groups": LongTensor (B, num_sta),
                 "pdw":        LongTensor (B,)}.

        Returns:
            (log_prob, value, entropy) — each tensor of shape (B,)
        """
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate_raw() not implemented."
        )
