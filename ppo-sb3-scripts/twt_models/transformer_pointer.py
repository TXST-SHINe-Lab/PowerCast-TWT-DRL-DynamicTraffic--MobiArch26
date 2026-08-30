#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_models/transformer_pointer.py — fully-transformer pointer policy.

Architecture (per timestep / over an episode):

    obs:  (T, B, N, F=7)        ; F per-STA features, N variable per session
        |
        |  Linear(F -> d_model)
        v
    sta_tokens:  (T, B, N, d_model)
        |
        |  Set Transformer encoder (n_set_layers, self-attention across N)
        v
    sta_enc:     (T, B, N, d_model)         ; each STA sees the global mix
        |
        |  mean-pool over N
        v
    global_t:    (T, B, d_model)
        |
        |  Causal self-attention over T (n_temporal_layers)
        v
    hist_global: (T, B, d_model)            ; history-aware
        |
        |  schedule_head -> (T, B, num_schedules)
        |
        |  pointer decoder (autoregressive over N) using
        |    init = sched_embed + hist_global
        |    sta context = sta_enc
        v
    per-STA group dist (T, B, N, max_num_groups)
        |
        |  per-STA value head: Linear(d_model -> 1), sum over N
        v
    value:       (T, B)

Variable-N: obs is (B, N, F) with N read at runtime from input shape. No
parameter is sized by N. Group masking optionally restricts logits to the
chosen schedule's num_groups.

Rollout (act_raw):
  Maintains a recurrent_state = (past_globals: (Tprev, d_model)) of all
  previous timesteps' global tokens. On each step we append the new global
  and re-run causal self-attention over the entire buffer. O(T^2) total
  but with T <= 73 this is negligible compared to NS-3 step time.

Training (evaluate_raw / evaluate_sequence_raw):
  Standard full-sequence forward with causal mask. Orchestrator dispatches
  to evaluate_sequence_raw via is_temporal_recurrent_policy = True.
"""

from typing import Any, Dict, List, Optional, Tuple

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base import BasePolicy
from . import register_policy


# --- Helper: causal self-attention block (pre-LN, post-norm-residual) ---
class _CausalSelfAttnBlock(nn.Module):
    """One pre-LN causal self-attention + FFN block. Operates on (T, B, d)."""

    def __init__(
        self, d_model: int, n_heads: int, ff_hidden: int, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=False,  # input shape (T, B, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """x: (T, B, d_model); attn_mask: (T, T) causal float (-inf above diag)."""
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Returns float (T, T) mask with -inf above diagonal (i.e. position t cannot
    attend to positions > t)."""
    m = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)


# --- Main policy ---
@register_policy
class TransformerPointerPolicy(BasePolicy):
    name = "transformer_pointer"
    is_raw_action_policy = True
    is_set_obs_policy = True
    is_temporal_recurrent_policy = True

    def __init__(
        self,
        features_per_sta: int = 7,
        num_schedules: int = 40,
        num_pdw_levels: int = 15,
        max_num_groups: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_set_layers: int = 3,
        n_temporal_layers: int = 2,
        ff_hidden: int = 256,
        group_embed_dim: int = 32,
        dropout: float = 0.0,
        mask_invalid_groups: bool = True,
        num_groups_per_schedule: Optional[List[int]] = None,
    ):
        super().__init__()
        self.features_per_sta = features_per_sta
        self.num_schedules = num_schedules
        self.num_pdw_levels = num_pdw_levels
        self.max_num_groups = max_num_groups
        self.d_model = d_model
        self.group_embed_dim = group_embed_dim
        self.mask_invalid_groups = bool(mask_invalid_groups)

        # ----- Per-STA token projection -----
        self.token_proj = nn.Linear(features_per_sta, d_model)

        # ----- Set Transformer encoder (self-attention across N STAs) -----
        # batch_first=True so input is (T*B, N, d_model). Pre-LN (norm_first).
        set_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(set_layer, num_layers=n_set_layers)
        self.set_norm = nn.LayerNorm(d_model)

        # ----- Causal self-attention over time (replaces pointer_ppo's LSTM) -----
        self.temporal_blocks = nn.ModuleList(
            [
                _CausalSelfAttnBlock(d_model, n_heads, ff_hidden, dropout)
                for _ in range(n_temporal_layers)
            ]
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        # ----- Schedule head -----
        self.schedule_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_schedules),
        )

        # ----- PDW head (3rd action: Power Delivery Window level) -----
        # Categorical over D ∈ {0,5,...,50}, on the history-aware global token.
        self.pdw_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_pdw_levels),
        )

        # ----- Autoregressive pointer decoder over STAs -----
        # Input to decoder LSTM at each step: concat(history-aware global, schedule embedding, prev_group_embed).
        # Output -> per-STA group logits.
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

        # ----- Per-STA value head (sum over N) -----
        self.value_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # ----- Schedule -> num_groups buffer for action masking -----
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
            self.num_groups_per_schedule = None

        # --- Init ---
        # Conservative for transformers under PPO: orthogonal Linear with reduced gain (transformers explode with default sqrt(2)).
        # Embeddings small.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

        # Action heads near-uniform (Schulman PPO convention).
        nn.init.orthogonal_(self.schedule_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.schedule_head[-1].bias)
        nn.init.orthogonal_(self.pdw_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.pdw_head[-1].bias)
        nn.init.orthogonal_(self.group_head.weight, gain=0.01)
        nn.init.zeros_(self.group_head.bias)

    # --- Action-table loader ---
    @staticmethod
    def _load_num_groups_per_schedule(num_schedules: int) -> List[int]:
        here = os.path.dirname(os.path.abspath(__file__))  # twt_models/
        twt_dir = os.path.dirname(
            os.path.dirname(here)
        )  # project root (rl-twt-powercast/)
        path = os.path.join(twt_dir, "exploration-scripts", "schedule_table.json")
        with open(path) as f:
            st = json.load(f)
        schedules = st.get("schedules", [])
        if len(schedules) < num_schedules:
            raise ValueError(
                f"schedule_table.json has {len(schedules)} schedules; "
                f"required {num_schedules}"
            )
        return [int(s["num_groups"]) for s in schedules[:num_schedules]]

    # --- Encoder helpers ---
    def _encode_set(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (T*B, N, F) -> sta_enc: (T*B, N, d_model)."""
        x = self.token_proj(obs)
        x = self.set_encoder(x)
        x = self.set_norm(x)
        return x

    def _temporal(self, global_t: torch.Tensor) -> torch.Tensor:
        """global_t: (T, B, d_model). Returns history-aware globals (T, B, d_model)."""
        T = global_t.shape[0]
        mask = _causal_mask(T, global_t.device)
        h = global_t
        for blk in self.temporal_blocks:
            h = blk(h, mask)
        return self.temporal_norm(h)

    # --- Group-mask helper (logits with -inf for invalid groups under sched) ---
    def _apply_group_mask(
        self, logits: torch.Tensor, sched_per_step: torch.Tensor
    ) -> torch.Tensor:
        """logits: (..., max_num_groups), sched_per_step: (...,) int.
        Sets logits[..., g] = -inf for g >= num_groups[sched]."""
        if self.num_groups_per_schedule is None:
            return logits
        ng = self.num_groups_per_schedule[sched_per_step]  # (...)
        # Build a mask of shape (..., max_num_groups)
        gids = torch.arange(self.max_num_groups, device=logits.device)
        mask = gids.expand(*ng.shape, self.max_num_groups) >= ng.unsqueeze(-1)
        return logits.masked_fill(mask, float("-inf"))

    # --- Autoregressive decoder (sample) — used at rollout time ---
    def _decode_sample(
        self,
        sta_enc: torch.Tensor,  # (B, N, d_model)
        hist_global: torch.Tensor,  # (B, d_model)
        sched_idx: torch.Tensor,  # (B,) int
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sta_groups: (B, N) int, group_log_prob: (B,), group_entropy: (B,))."""
        B, N, _ = sta_enc.shape
        device = sta_enc.device

        sched_emb = self.sched_embed(sched_idx)  # (B, d_model)
        # Initial decoder state: hist_global + sched_emb projected via the LSTM init (we feed it as the first input).
        # Use START_TOKEN as the initial group.
        prev_group = torch.full((B,), self.START_TOKEN, device=device, dtype=torch.long)
        h = None  # let LSTM init zeros
        sta_groups = []
        log_probs = []
        entropies = []
        for i in range(N):
            sta_ctx = sta_enc[:, i, :]  # (B, d_model)
            ge = self.group_embed(prev_group)  # (B, group_embed_dim)
            # Combine history-aware global + STA context as the decoder input.
            # (sched_emb is already baked into hist_global through the global token, but adding it directly improves signal early in training.)
            inp = torch.cat([sta_ctx + hist_global + sched_emb, ge], dim=-1)
            inp = inp.unsqueeze(1)  # (B, 1, *)
            out, h = self.decoder(inp, h)
            logits = self.group_head(out.squeeze(1))  # (B, max_num_groups)
            if self.mask_invalid_groups:
                logits = self._apply_group_mask(logits, sched_idx)
            dist = Categorical(logits=logits)
            if deterministic:
                g = logits.argmax(dim=-1)
            else:
                g = dist.sample()
            log_probs.append(dist.log_prob(g))
            entropies.append(dist.entropy())
            sta_groups.append(g)
            prev_group = g

        sta_groups = torch.stack(sta_groups, dim=1)  # (B, N)
        # Joint log-prob = sum over STAs (group dims are independent given hist).
        # Mean entropy over STAs (Schulman PPO convention — see pointer_ppo.py commit history for why mean and not sum).
        group_lp = torch.stack(log_probs, dim=1).sum(dim=1)
        group_ent = torch.stack(entropies, dim=1).mean(dim=1)
        return sta_groups, group_lp, group_ent

    # --- Autoregressive decoder (teacher) — used at PPO update time ---
    def _decode_teacher(
        self,
        sta_enc: torch.Tensor,  # (B, N, d_model)
        hist_global: torch.Tensor,  # (B, d_model)
        sched_idx: torch.Tensor,  # (B,)
        sta_groups: torch.Tensor,  # (B, N) int (the groups taken at rollout)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluates the joint group log-prob and entropy with teacher forcing
        on `sta_groups`. Returns (group_lp: (B,), group_ent: (B,))."""
        B, N, _ = sta_enc.shape
        sched_emb = self.sched_embed(sched_idx)  # (B, d_model)
        # prev_group sequence: [START, g_0, g_1, ..., g_{N-2}]
        start = torch.full(
            (B, 1), self.START_TOKEN, device=sta_enc.device, dtype=torch.long
        )
        prev_groups = torch.cat([start, sta_groups[:, :-1]], dim=1)  # (B, N)
        ge = self.group_embed(prev_groups)  # (B, N, group_embed_dim)
        # Decoder input per step = (sta_ctx + hist + sched_emb, group_embed)
        hist_b = (
            (hist_global + sched_emb).unsqueeze(1).expand(-1, N, -1)
        )  # (B, N, d_model)
        inp = torch.cat([sta_enc + hist_b, ge], dim=-1)  # (B, N, d+ge)
        out, _ = self.decoder(inp)  # (B, N, d_model)
        logits = self.group_head(out)  # (B, N, max_num_groups)
        if self.mask_invalid_groups:
            sched_per_sta = sched_idx.unsqueeze(1).expand(-1, N)  # (B, N)
            logits = self._apply_group_mask(logits, sched_per_sta)
        dist = Categorical(logits=logits)
        lp = dist.log_prob(sta_groups).sum(dim=1)  # (B,) sum over N
        ent = dist.entropy().mean(dim=1)  # (B,) mean over N
        return lp, ent

    # --- Value head ---
    def _compute_value(
        self, sta_enc: torch.Tensor, hist_global: torch.Tensor
    ) -> torch.Tensor:
        """sta_enc: (B, N, d), hist_global: (B, d) -> value: (B,) summed over N."""
        N = sta_enc.shape[1]
        hist_b = hist_global.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([sta_enc, hist_b], dim=-1)  # (B, N, 2d)
        per_sta = self.value_head(x).squeeze(-1)  # (B, N)
        return per_sta.sum(dim=1)

    # --- Recurrent state: a buffer of past global tokens for causal SA ---
    def initial_recurrent_state(self) -> Optional[torch.Tensor]:
        # Empty buffer at episode start. Stored as (T, d_model) tensor.
        return torch.zeros(0, self.d_model)

    # --- Indexed interface — not used (raw policy) ---
    def act(self, *args, **kwargs):
        raise NotImplementedError("TransformerPointerPolicy is raw; use act_raw().")

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            "TransformerPointerPolicy is raw; use evaluate_raw()."
        )

    # --- Raw rollout step ---
    @torch.no_grad()
    def act_raw(
        self,
        obs_np: np.ndarray,
        recurrent_state: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], float, float, torch.Tensor]:
        # obs_np: (N, F) — variable N
        device = next(self.parameters()).device
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        if obs.dim() != 2:
            raise ValueError(
                f"TransformerPointer expects 2-D obs (N, F), got shape {obs.shape}"
            )
        N, F_ = obs.shape
        # (1, N, F) -> set encode -> (1, N, d)
        sta_enc = self._encode_set(obs.unsqueeze(0))  # (1, N, d)
        # Pool to global (1, d)
        global_now = sta_enc.mean(dim=1)  # (1, d)

        # Append to past_globals buffer and run causal SA over the whole buffer.
        if recurrent_state is None or recurrent_state.numel() == 0:
            past = global_now  # (1, d)
        else:
            past = recurrent_state.to(device)
            if past.dim() == 2:  # (Tprev, d) without batch
                past = torch.cat([past, global_now], dim=0)  # (Tprev+1, d)
            else:
                raise ValueError(
                    f"unexpected recurrent_state shape {tuple(past.shape)}"
                )

        # Causal SA expects (T, B, d). Treat B=1.
        seq = past.unsqueeze(1)  # (T, 1, d)
        hist = self._temporal(seq)  # (T, 1, d)
        hist_global = hist[-1, 0, :]  # (d,)
        # Schedule head
        sched_logits = self.schedule_head(
            hist_global.unsqueeze(0)
        )  # (1, num_schedules)
        sched_dist = Categorical(logits=sched_logits)
        if deterministic:
            sched_idx = sched_logits.argmax(dim=-1)
        else:
            sched_idx = sched_dist.sample()
        sched_lp = sched_dist.log_prob(sched_idx)
        sched_ent = sched_dist.entropy()

        # PDW head (3rd action, on the history-aware global token)
        pdw_logits = self.pdw_head(hist_global.unsqueeze(0))  # (1, num_pdw_levels)
        pdw_dist = Categorical(logits=pdw_logits)
        if deterministic:
            pdw_idx = pdw_logits.argmax(dim=-1)
        else:
            pdw_idx = pdw_dist.sample()
        pdw_lp = pdw_dist.log_prob(pdw_idx)

        # Decode per-STA groups
        sta_groups, group_lp, group_ent = self._decode_sample(
            sta_enc, hist_global.unsqueeze(0), sched_idx, deterministic
        )
        # Value
        value = self._compute_value(sta_enc, hist_global.unsqueeze(0))  # (1,)

        payload = {
            "sched": int(sched_idx.item()),
            "sta_groups": sta_groups.squeeze(0).cpu().numpy().astype(np.int64),
            "pdw": int(pdw_idx.item()),
        }
        log_prob = float((sched_lp + group_lp + pdw_lp).item())
        v = float(value.item())
        # New recurrent state = past buffer (without batch dim).
        new_state = past.detach().cpu()  # (T, d)
        return payload, log_prob, v, new_state

    # --- Raw flat evaluation (single-step, used as fallback / non-recurrent path) ---
    def evaluate_raw(
        self,
        obs_t: torch.Tensor,  # (B, N, F)
        payload_batch: Dict[str, torch.Tensor],
        recurrent_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat (per-step, no temporal context) evaluation. Orchestrator should
        prefer evaluate_sequence_raw for recurrent updates; this is a fallback."""
        if obs_t.dim() == 3:
            B, N, _ = obs_t.shape
        else:
            raise ValueError(
                f"TransformerPointer expects (B, N, F) obs, got {tuple(obs_t.shape)}"
            )
        sta_enc = self._encode_set(obs_t)  # (B, N, d)
        global_t = sta_enc.mean(dim=1)  # (B, d)
        # Without temporal context, treat as T=1 sequence.
        hist = self._temporal(global_t.unsqueeze(0)).squeeze(0)  # (B, d)

        sched_idx = payload_batch["sched"].long()  # (B,)
        sta_groups = payload_batch["sta_groups"].long()  # (B, N)
        pdw_idx = payload_batch["pdw"].long()  # (B,)

        sched_logits = self.schedule_head(hist)
        sched_dist = Categorical(logits=sched_logits)
        sched_lp = sched_dist.log_prob(sched_idx)
        sched_ent = sched_dist.entropy()

        pdw_dist = Categorical(logits=self.pdw_head(hist))
        pdw_lp = pdw_dist.log_prob(pdw_idx)
        pdw_ent = pdw_dist.entropy()

        group_lp, group_ent = self._decode_teacher(sta_enc, hist, sched_idx, sta_groups)
        value = self._compute_value(sta_enc, hist)
        return sched_lp + group_lp + pdw_lp, value, sched_ent + group_ent + pdw_ent

    # --- Raw sequence evaluation (full episode roll, with causal SA over time) ---
    def evaluate_sequence_raw(
        self,
        obs_seq: torch.Tensor,  # (T, N, F)  — single-episode batch
        payload_seq: Dict[str, torch.Tensor],
        recurrent_state: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-episode forward used by the orchestrator's sequence-aware PPO
        update path. obs_seq is one episode (T, N, F); payload_seq has
        sched: (T,) and sta_groups: (T, N)."""
        if obs_seq.dim() != 3:
            raise ValueError(
                f"evaluate_sequence_raw expects (T, N, F), got {tuple(obs_seq.shape)}"
            )
        T, N, _ = obs_seq.shape
        # Treat as T mini-batches of B=1 for the set encoder by collapsing T into the batch dim: (T, N, F) -> set_encoder -> (T, N, d).
        sta_enc = self._encode_set(obs_seq)  # (T, N, d)
        global_t = sta_enc.mean(dim=1)  # (T, d)
        # Temporal: input shape (T, B=1, d).
        hist = self._temporal(global_t.unsqueeze(1)).squeeze(1)  # (T, d)

        sched_idx = payload_seq["sched"].long()  # (T,)
        sta_groups = payload_seq["sta_groups"].long()  # (T, N)
        pdw_idx = payload_seq["pdw"].long()  # (T,)

        sched_logits = self.schedule_head(hist)  # (T, num_schedules)
        sched_dist = Categorical(logits=sched_logits)
        sched_lp = sched_dist.log_prob(sched_idx)  # (T,)
        sched_ent = sched_dist.entropy()  # (T,)

        pdw_dist = Categorical(logits=self.pdw_head(hist))  # (T, num_pdw_levels)
        pdw_lp = pdw_dist.log_prob(pdw_idx)  # (T,)
        pdw_ent = pdw_dist.entropy()  # (T,)

        # Reuse teacher decoder per timestep (vectorized).
        group_lp, group_ent = self._decode_teacher(sta_enc, hist, sched_idx, sta_groups)
        value = self._compute_value(sta_enc, hist)  # (T,)
        return sched_lp + group_lp + pdw_lp, value, sched_ent + group_ent + pdw_ent
