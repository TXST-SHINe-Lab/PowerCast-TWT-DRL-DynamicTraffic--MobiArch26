#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models/pointer_ppo.py - Autoregressive pointer/attention PPO policy
                            with optional temporal LSTM, group-action masking,
                            and per-STA value head.

Inspired by Kool et al. 2019 ("Attention, Learn to Solve Routing Problems")
and Bello et al.'s Pointer Networks.

================================================================
Why this exists
================================================================
The legacy mlp_ppo / lstm_ppo policies emit a (schedule_idx, assignment_idx)
pair from a precomputed lookup table (20 schedule templates x 19 assignment
patterns = 380 actions). This is a coarse compression of the real action
space. This policy keeps schedule_idx (a templated selector for per-group
wake parameters), but REPLACES assignment_idx with an autoregressive per-STA
group assignment vector of length num_sta. Combined raw action space is
    num_schedules * max_num_groups ** num_sta
    e.g. 20 * 8^16 ~= 5.6e15.

================================================================
Architecture
================================================================
    obs (288,) --reshape--> per-STA tokens (16, 18)

    Encoder (self-attention transformer, batch_first=True, norm_first=True):
        token_proj : Linear(features_per_sta=18, d_model)
        encoder    : TransformerEncoder(d_model, n_heads, ff_hidden, n_layers)
        outputs    : encoded STA tokens (B, num_sta, d_model)
                     pooled context = mean over STAs (B, d_model)

    Temporal LSTM (NEW; only if `temporal=True`):
        temporal_lstm : nn.LSTM(d_model -> temporal_hidden), batch_first=False
        carries (h, c) across PPO timesteps within an episode. Pooled context
        flows in; the LSTM output `action_ctx` feeds the schedule head and
        (optionally) the value head. The decoder is unchanged: it is still
        seeded by sched_embed(sched_idx), so temporal information reaches
        per-STA group decoding only through the schedule choice. This keeps
        the autoregressive decoder simple while giving the schedule + value
        heads memory across the real steps of an episode.

        The orchestrator's `_ppo_update_recurrent` path (the one that
        re-rolls hidden state from h_0=0 over a whole episode) is dispatched
        based on `is_temporal_recurrent_policy`, which we set in __init__
        from the `temporal` kwarg. Re-rolling from zero is correct because
        rollout-time and update-time weights are identical at the start of
        each PPO update.

    Schedule head (operates on action_ctx — pooled or temporal_h):
        Linear(head_in, head_in) -> GELU -> Linear(head_in, num_schedules)
        Categorical over num_schedules (e.g. 20).

    Autoregressive group decoder (LSTM, conditioned on schedule):
        sched_embed : Embedding(num_schedules, d_model)
        group_embed : Embedding(max_num_groups + 1, group_embed_dim)
                      (the "+1" slot is the <start> token used at i=0)
        decoder LSTM: nn.LSTM(d_model + group_embed_dim, d_model, num_layers=1)
        group_head  : Linear(d_model, max_num_groups)

        h_0 = sched_embed(sampled schedule).unsqueeze(0)
        c_0 = zeros
        prev_group_0 = <start>
        for i in 0..num_sta-1:
            input_i  = cat(encoded[:, i, :], group_embed(prev_group_i))
            (h, c)   = decoder(input_i, (h, c))
            logits_i = group_head(h)                          # (B, max_num_groups)
            logits_i = mask groups >= num_groups[sched]       # NEW (if mask_invalid_groups)
            sample group_i ~ Categorical(logits_i)
            prev_group_{i+1} = group_i
        Total group log-prob = sum_i log P(group_i | encoded, schedule, group_<i)

    Value head:
        per_sta_value=False: V = head(action_ctx).
        per_sta_value=True:  V = sum_i head(concat(encoded_i, action_ctx_bcast)).
                             Sum-of-per-STA-values gives the critic finer
                             credit assignment matched to the per-STA action.

================================================================
Action interface
================================================================
This policy uses the BasePolicy "raw" action interface:
    is_raw_action_policy = True
    act_raw(obs, recurrent_state)        -> (payload, log_prob, value, next_state)
    evaluate_raw(obs_t, payload_batch)   -> (log_prob, value, entropy)
    evaluate_sequence_raw(obs_seq, ...)  -> (log_prob, value, entropy)  # NEW override

The payload returned by act_raw() is unchanged:
    {
        "sched":      int                              # in [0, num_schedules)
        "sta_groups": np.ndarray (num_sta,) int64      # each in [0, max_num_groups)
    }
The worker passes this to build_action_dict_raw() in twt_spawn_worker.py,
which wraps each sta_groups[i] into [0, num_groups[sched]) via modulo. With
mask_invalid_groups=True the policy never proposes wrap-needed values in the
first place, so the modulo is a defensive no-op.

================================================================
Backward compatibility (constructor flags)
================================================================
    temporal=True                  -> enable the temporal LSTM cell
    temporal_hidden=None           -> defaults to d_model
    mask_invalid_groups=True       -> mask group logits at indices
                                       >= num_groups_per_schedule[sched]
    per_sta_value=True             -> sum-of-per-STA value function
    num_groups_per_schedule=None   -> auto-load from
                                       twt/exploration-scripts/schedule_table.json

To reproduce the original architecture exactly, instantiate with
    temporal=False, mask_invalid_groups=False, per_sta_value=False.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base import BasePolicy
from . import register_policy


@register_policy
class PointerPpoPolicy(BasePolicy):
    name = "pointer_ppo"
    is_raw_action_policy = True

    def __init__(
        self,
        obs_dim: int = 112,
        num_sta: int = 16,
        features_per_sta: int = 7,
        num_schedules: int = 40,
        num_pdw_levels: int = 15,
        max_num_groups: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        ff_hidden: int = 256,
        group_embed_dim: int = 32,
        dropout: float = 0.0,
        # New options: defaults give the upgraded architecture; flip all three off to reproduce the original pointer_ppo behavior.
        temporal: bool = True,
        temporal_hidden: Optional[int] = None,
        mask_invalid_groups: bool = True,
        per_sta_value: bool = True,
        num_groups_per_schedule: Optional[List[int]] = None,
    ):
        super().__init__()
        if obs_dim != num_sta * features_per_sta:
            raise ValueError(
                f"obs_dim ({obs_dim}) must equal num_sta ({num_sta}) * "
                f"features_per_sta ({features_per_sta})"
            )
        self.obs_dim = obs_dim
        self.num_sta = num_sta
        self.features_per_sta = features_per_sta
        self.num_schedules = num_schedules
        self.num_pdw_levels = num_pdw_levels
        self.max_num_groups = max_num_groups
        self.d_model = d_model
        self.group_embed_dim = group_embed_dim
        self.temporal = bool(temporal)
        self.mask_invalid_groups = bool(mask_invalid_groups)
        self.per_sta_value = bool(per_sta_value)
        self.temporal_hidden = (
            int(temporal_hidden) if temporal_hidden is not None else d_model
        )
        # The orchestrator dispatches to its sequence-aware PPO update path based on this attribute.
        # Setting it on `self` shadows the class-level default from BasePolicy.
        self.is_temporal_recurrent_policy = self.temporal

        # ----- Per-STA token projection + transformer encoder -----
        self.token_proj = nn.Linear(features_per_sta, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # ----- Temporal LSTM over pooled context (only if temporal=True) -----
        if self.temporal:
            self.temporal_lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=self.temporal_hidden,
                num_layers=1,
                batch_first=False,  # we use (T, B, d_model)
            )
            head_in_dim = self.temporal_hidden
        else:
            head_in_dim = d_model

        # ----- Schedule head -----
        self.schedule_head = nn.Sequential(
            nn.Linear(head_in_dim, head_in_dim),
            nn.GELU(),
            nn.Linear(head_in_dim, num_schedules),
        )

        # ----- PDW head (3rd action: Power Delivery Window level) -----
        # Categorical over D ∈ {0,5,...,50}.
        # Operates on the same action context as the schedule head (pooled / temporal_h).
        self.pdw_head = nn.Sequential(
            nn.Linear(head_in_dim, head_in_dim),
            nn.GELU(),
            nn.Linear(head_in_dim, num_pdw_levels),
        )

        # ----- Autoregressive group decoder -----
        self.START_TOKEN = max_num_groups
        self.sched_embed = nn.Embedding(num_schedules, d_model)
        self.group_embed = nn.Embedding(max_num_groups + 1, group_embed_dim)
        self.decoder = nn.LSTM(
            input_size=d_model + group_embed_dim,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.group_head = nn.Linear(d_model, max_num_groups)

        # ----- Value head -----
        if self.per_sta_value:
            value_in_dim = d_model + (self.temporal_hidden if self.temporal else 0)
        else:
            value_in_dim = head_in_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_in_dim, value_in_dim),
            nn.GELU(),
            nn.Linear(value_in_dim, 1),
        )

        # ----- num_groups_per_schedule buffer for action masking -----
        if self.mask_invalid_groups:
            if num_groups_per_schedule is None:
                num_groups_per_schedule = self._load_num_groups_per_schedule(
                    num_schedules
                )
            ng = torch.as_tensor(num_groups_per_schedule, dtype=torch.long)
            if ng.shape != (num_schedules,):
                raise ValueError(
                    f"num_groups_per_schedule must have len={num_schedules}; "
                    f"got shape {tuple(ng.shape)}"
                )
            if (ng < 1).any() or (ng > max_num_groups).any():
                raise ValueError(
                    f"num_groups_per_schedule entries must be in [1, {max_num_groups}]; "
                    f"got {ng.tolist()}"
                )
            self.register_buffer("num_groups_per_schedule", ng, persistent=True)
        else:
            # No masking — set to None as a non-buffer attribute so attribute lookups still work but the masking branches stay no-ops.
            self.num_groups_per_schedule = None

        # ----- Init -----
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)

        # Action heads start near-uniform (Schulman PPO convention).
        # With the default gain=sqrt(2) above, the initial decoder already imposes structure that PPO has to undo before it can learn — see https://arxiv.org/abs/2005.12729 (Engstrom et al.) for the empirical case for small final-layer init in policy networks.
        nn.init.orthogonal_(self.schedule_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.schedule_head[-1].bias)
        nn.init.orthogonal_(self.pdw_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.pdw_head[-1].bias)
        nn.init.orthogonal_(self.group_head.weight, gain=0.01)
        nn.init.zeros_(self.group_head.bias)

    # --- Action-table loader (for masking) ---
    @staticmethod
    def _load_num_groups_per_schedule(num_schedules: int) -> List[int]:
        # twt_models/pointer_ppo.py -> ppo-sb3-scripts/twt_models/ -> ppo-sb3-scripts/
        # parent of ppo-sb3-scripts/ is the project root (rl-twt-powercast/), so twt_dir/exploration-scripts/schedule_table.json.
        here = os.path.dirname(os.path.abspath(__file__))  # twt_models/
        twt_dir = os.path.dirname(os.path.dirname(here))  # rl-twt-powercast/
        path = os.path.join(twt_dir, "exploration-scripts", "schedule_table.json")
        with open(path) as f:
            st = json.load(f)
        schedules = st.get("schedules", [])
        if len(schedules) < num_schedules:
            raise ValueError(
                f"schedule_table.json has {len(schedules)} schedules; "
                f"PointerPpoPolicy requires at least {num_schedules}"
            )
        return [int(s["num_groups"]) for s in schedules[:num_schedules]]

    # --- Encoder pass ---
    def _encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, obs_dim) -> (encoded (B, num_sta, d_model), pooled (B, d_model))."""
        B = obs.shape[0]
        tokens = obs.view(B, self.num_sta, self.features_per_sta)
        tokens = self.token_proj(tokens)
        encoded = self.encoder(tokens)
        encoded = self.encoder_norm(encoded)
        pooled = encoded.mean(dim=1)
        return encoded, pooled

    # --- Temporal pass: project pooled context through the LSTM cell (single step) ---
    def _apply_temporal(
        self,
        pooled: torch.Tensor,
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        pooled: (B, d_model). recurrent_state: ((1, B, h), (1, B, h)) or None.
        Returns (action_ctx (B, h), new_state). Treats time as length-1.
        """
        x = pooled.unsqueeze(0)  # (1, B, d_model)
        if recurrent_state is None:
            h0 = torch.zeros(
                1, pooled.shape[0], self.temporal_hidden, device=pooled.device
            )
            c0 = torch.zeros_like(h0)
            recurrent_state = (h0, c0)
        out, new_state = self.temporal_lstm(x, recurrent_state)
        return out.squeeze(0), new_state

    # --- Group-action masking ---
    def _apply_group_mask(
        self, group_logits: torch.Tensor, sched_idx_per_token: torch.Tensor
    ) -> torch.Tensor:
        """
        group_logits: (..., max_num_groups). Last dim gets masked.
        sched_idx_per_token: shape == group_logits.shape[:-1].
        Caller is responsible for any broadcasting (e.g. expanding (B,) -> (B, num_sta)).
        """
        if self.num_groups_per_schedule is None:
            return group_logits
        ng = self.num_groups_per_schedule[sched_idx_per_token]  # (...)
        group_ids = torch.arange(self.max_num_groups, device=group_logits.device)
        # mask: True where group_id >= num_groups[sched] -> set logit to -inf
        mask = group_ids >= ng.unsqueeze(-1)
        return group_logits.masked_fill(mask, float("-inf"))

    # --- Decoder: teacher forcing (training / evaluate_raw / evaluate_sequence_raw) ---
    def _decode_teacher(
        self,
        encoded: torch.Tensor,  # (B, num_sta, d_model)
        sched_idx: torch.Tensor,  # (B,)
        sta_groups: torch.Tensor,  # (B, num_sta) ground-truth groups
    ) -> torch.Tensor:
        """Returns group_logits of shape (B, num_sta, max_num_groups). No masking."""
        B, T, _ = encoded.shape
        device = encoded.device
        prev_groups = torch.full(
            (B, T), self.START_TOKEN, dtype=torch.long, device=device
        )
        if T > 1:
            prev_groups[:, 1:] = sta_groups[:, :-1]
        prev_emb = self.group_embed(prev_groups)  # (B, T, group_embed_dim)
        dec_in = torch.cat([encoded, prev_emb], dim=-1)  # (B, T, d_model+gE)

        h0 = self.sched_embed(sched_idx).unsqueeze(0).contiguous()  # (1, B, d_model)
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder(dec_in, (h0, c0))  # (B, T, d_model)
        return self.group_head(dec_out)  # (B, T, max_num_groups)

    # --- Decoder: stepwise sampling (rollout / act_raw) ---
    def _decode_sample(
        self,
        encoded: torch.Tensor,  # (B, num_sta, d_model)
        sched_idx: torch.Tensor,  # (B,)
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (sta_groups (B, num_sta), log_prob_sum (B,)). Applies masking."""
        B, T, _ = encoded.shape
        device = encoded.device
        h = self.sched_embed(sched_idx).unsqueeze(0).contiguous()
        c = torch.zeros_like(h)
        prev_group = torch.full((B,), self.START_TOKEN, dtype=torch.long, device=device)
        sampled, log_probs = [], []
        for i in range(T):
            inp = torch.cat(
                [encoded[:, i, :], self.group_embed(prev_group)], dim=-1
            ).unsqueeze(
                1
            )  # (B, 1, d_model+gE)
            out, (h, c) = self.decoder(inp, (h, c))  # out: (B, 1, d_model)
            logits = self.group_head(out.squeeze(1))  # (B, max_num_groups)
            if self.mask_invalid_groups:
                logits = self._apply_group_mask(logits, sched_idx)
            dist = Categorical(logits=logits)
            g = logits.argmax(dim=-1) if deterministic else dist.sample()
            sampled.append(g)
            log_probs.append(dist.log_prob(g))
            prev_group = g
        sta_groups = torch.stack(sampled, dim=1)  # (B, num_sta)
        log_prob_sum = torch.stack(log_probs, dim=1).sum(dim=1)  # (B,)
        return sta_groups, log_prob_sum

    # --- Value computation: pooled / per-STA, with optional temporal context ---
    def _compute_value(
        self,
        encoded: torch.Tensor,  # (B, num_sta, d_model)
        action_ctx: torch.Tensor,  # (B, head_in_dim)
    ) -> torch.Tensor:
        """Returns V (B,)."""
        if self.per_sta_value:
            if self.temporal:
                ctx = action_ctx.unsqueeze(1).expand(-1, self.num_sta, -1)
                inp = torch.cat([encoded, ctx], dim=-1)
            else:
                inp = encoded
            per_sta = self.value_head(inp).squeeze(-1)  # (B, num_sta)
            return per_sta.sum(dim=1)
        else:
            return self.value_head(action_ctx).squeeze(-1)  # (B,)

    # --- BasePolicy hooks ---
    def initial_recurrent_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.temporal:
            return None
        # Lazy: returning None lets _apply_temporal create zeros on the right device when act_raw is first invoked in the worker.
        return None

    # --- Indexed-interface stubs (this policy is raw-only) ---
    def act(self, *args, **kwargs):
        raise NotImplementedError(
            "PointerPpoPolicy is a raw-action policy. "
            "Workers must call act_raw() instead of act()."
        )

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "PointerPpoPolicy is a raw-action policy. "
            "The orchestrator must call evaluate_raw() instead of evaluate()."
        )

    # --- Raw-action interface ---
    @torch.no_grad()
    def act_raw(
        self,
        obs_np,
        recurrent_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], float, float, Optional[Any]]:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(
            0
        )  # (1, obs_dim)
        encoded, pooled = self._encode(obs_t)

        if self.temporal:
            action_ctx, next_state = self._apply_temporal(pooled, recurrent_state)
        else:
            action_ctx = pooled
            next_state = None

        # Schedule
        sched_logits = self.schedule_head(action_ctx)  # (1, num_schedules)
        sched_dist = Categorical(logits=sched_logits)
        sched_idx = (
            sched_logits.argmax(dim=-1) if deterministic else sched_dist.sample()
        )  # (1,)
        sched_lp = sched_dist.log_prob(sched_idx)  # (1,)

        # PDW level (3rd head, on the same action context as the schedule)
        pdw_logits = self.pdw_head(action_ctx)  # (1, num_pdw_levels)
        pdw_dist = Categorical(logits=pdw_logits)
        pdw_idx = (
            pdw_logits.argmax(dim=-1) if deterministic else pdw_dist.sample()
        )  # (1,)
        pdw_lp = pdw_dist.log_prob(pdw_idx)  # (1,)

        # Per-STA groups (autoregressive, conditioned on schedule, w/ masking)
        sta_groups, group_lp = self._decode_sample(
            encoded, sched_idx, deterministic=deterministic
        )

        # Value
        value_t = self._compute_value(encoded, action_ctx)  # (1,)

        log_prob = float((sched_lp + group_lp + pdw_lp).item())
        value = float(value_t.item())
        sta_groups_np = sta_groups.squeeze(0).cpu().numpy().astype(np.int64)
        payload = {
            "sched": int(sched_idx.item()),
            "sta_groups": sta_groups_np,
            "pdw": int(pdw_idx.item()),
        }
        return payload, log_prob, value, next_state

    def evaluate_raw(
        self,
        obs_t: torch.Tensor,
        payload_batch: Dict[str, torch.Tensor],
        recurrent_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-step evaluation. The orchestrator's flat-shuffle PPO path uses
        this when is_temporal_recurrent_policy=False. For temporal=True, the
        orchestrator dispatches to evaluate_sequence_raw() instead."""
        sched_t = payload_batch["sched"]  # (B,)        int64
        sta_groups_t = payload_batch["sta_groups"]  # (B, num_sta) int64
        pdw_t = payload_batch["pdw"]  # (B,)        int64

        encoded, pooled = self._encode(obs_t)
        if self.temporal:
            action_ctx, _ = self._apply_temporal(pooled, recurrent_state)
        else:
            action_ctx = pooled

        # Schedule
        sched_logits = self.schedule_head(action_ctx)
        sched_dist = Categorical(logits=sched_logits)
        sched_lp = sched_dist.log_prob(sched_t)
        sched_ent = sched_dist.entropy()

        # PDW (3rd head)
        pdw_dist = Categorical(logits=self.pdw_head(action_ctx))
        pdw_lp = pdw_dist.log_prob(pdw_t)
        pdw_ent = pdw_dist.entropy()

        # Groups (teacher forcing on the buffered actions, with optional masking)
        group_logits = self._decode_teacher(encoded, sched_t, sta_groups_t)
        if self.mask_invalid_groups:
            sched_per_sta = sched_t.unsqueeze(-1).expand(-1, self.num_sta)
            group_logits = self._apply_group_mask(group_logits, sched_per_sta)
        group_dist = Categorical(logits=group_logits)
        group_lp = group_dist.log_prob(sta_groups_t).sum(dim=1)  # (B,)
        # Mean (not sum) over STAs so the entropy bonus has comparable magnitude to single-action policies (lstm_ppo / mlp_ppo).
        # With sum, ent_coef=0.01 effectively becomes ~0.17 vs the ~0.02 those policies see, which crushes decoder sharpening.
        # log_prob stays summed because the PPO ratio must be the joint.
        group_ent = group_dist.entropy().mean(dim=1)  # (B,)

        value = self._compute_value(encoded, action_ctx)  # (B,)
        return sched_lp + group_lp + pdw_lp, value, sched_ent + group_ent + pdw_ent

    def evaluate_sequence_raw(
        self,
        obs_seq: torch.Tensor,
        payload_seq: Dict[str, torch.Tensor],
        recurrent_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence-level evaluation that rolls the temporal LSTM through one
        full episode in chronological order. Called by the orchestrator's
        _ppo_update_recurrent path when is_temporal_recurrent_policy=True."""
        if not self.temporal:
            return self.evaluate_raw(obs_seq, payload_seq, recurrent_state)

        # obs_seq: (T, obs_dim). Encoder is stateless, so encode all T at once.
        T = obs_seq.shape[0]
        encoded_all, pooled_all = self._encode(obs_seq)
        # encoded_all: (T, num_sta, d_model), pooled_all: (T, d_model)

        # Roll temporal LSTM over T as the seq dim with batch=1.
        if recurrent_state is None:
            h0 = torch.zeros(1, 1, self.temporal_hidden, device=obs_seq.device)
            c0 = torch.zeros_like(h0)
            recurrent_state = (h0, c0)
        x = pooled_all.unsqueeze(1)  # (T, 1, d_model)
        action_ctx_seq, _ = self.temporal_lstm(x, recurrent_state)
        action_ctx_seq = action_ctx_seq.squeeze(1)  # (T, temporal_hidden)

        sched_seq = payload_seq["sched"]  # (T,)
        sta_groups_seq = payload_seq["sta_groups"]  # (T, num_sta)
        pdw_seq = payload_seq["pdw"]  # (T,)

        # Schedule (vectorized over T)
        sched_logits_seq = self.schedule_head(action_ctx_seq)  # (T, num_schedules)
        sched_dist = Categorical(logits=sched_logits_seq)
        sched_lp = sched_dist.log_prob(sched_seq)  # (T,)
        sched_ent = sched_dist.entropy()  # (T,)

        # PDW (3rd head, vectorized over T)
        pdw_dist = Categorical(
            logits=self.pdw_head(action_ctx_seq)
        )  # (T, num_pdw_levels)
        pdw_lp = pdw_dist.log_prob(pdw_seq)  # (T,)
        pdw_ent = pdw_dist.entropy()  # (T,)

        # Groups (teacher forcing on the buffered actions; treat T as the batch dim)
        group_logits = self._decode_teacher(encoded_all, sched_seq, sta_groups_seq)
        # group_logits: (T, num_sta, max_num_groups)
        if self.mask_invalid_groups:
            sched_per_sta = sched_seq.unsqueeze(-1).expand(-1, self.num_sta)
            group_logits = self._apply_group_mask(group_logits, sched_per_sta)
        group_dist = Categorical(logits=group_logits)
        group_lp = group_dist.log_prob(sta_groups_seq).sum(dim=1)  # (T,)
        group_ent = group_dist.entropy().mean(dim=1)  # (T,) — see evaluate_raw

        # Value (per-STA value head sees (T, num_sta, d_model) directly)
        value = self._compute_value(encoded_all, action_ctx_seq)  # (T,)
        return sched_lp + group_lp + pdw_lp, value, sched_ent + group_ent + pdw_ent
