#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_batch_orchestrator.py - Parallel PPO training for TWT, strict port of
                            wifi-simulation/batch_orchestrator.py.

Architecture (option A from design discussion):
  - One master TWT policy (BasePolicy subclass, picked from twt_models registry).
  - One Adam optimizer.
  - Per batch:
        1. Pickle {name, kwargs, state_dict} as the policy payload.
        2. Spawn NUM_WORKERS mp.Process workers (twt_spawn_worker.run_episode).
        3. Drain the result queue BEFORE joining (a child blocked on a full queue pipe never exits, so join() would hang).
        4. Join workers.
        5. Compute returns + GAE advantages from collected rollouts.
        6. Run K epochs of clipped-surrogate PPO over minibatches.
        7. Log to TensorBoard + JSONL; checkpoint every --save-freq batches.

Adding a new architecture:
    drop a new file in twt_models/ that subclasses BasePolicy and uses
    @register_policy. Then run:
        python twt_batch_orchestrator.py --policy-arch <new_name> ...
    No edits to this file or the worker are required.

Usage:
    python3.11 twt_batch_orchestrator.py \
        --policy-arch mlp_ppo \
        --num-workers 4 \
        --num-batches 50 \
        --max-steps-per-episode 100 \
        --reward-preset balanced
"""

import os
import warnings

# Silence TF/XLA C-level chatter from torch.utils.tensorboard's transitive deps.
# Must run BEFORE the torch import below, and also takes effect in spawn children because env vars are inherited via os.environ AND because spawn children re-execute this top-level block as __mp_main__ to unpickle the target function.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Silence the TransformerEncoder enable_nested_tensor UserWarning fired by
# pointer_ppo (and any other benign torch UserWarning at policy construction).
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Path setup so the spawn-method "spawn" can re-import everything in workers.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.join(_parent_dir, "exploration-scripts"))

from twt_models import get_policy, list_policies
from twt_normalizer import ReturnNormalizer
from twt_spawn_worker import run_episode, default_obs_kwargs, NUM_CRITIC_FEATURES


# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel PPO training for TWT (wifi-simulation pattern)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Architecture (registry-driven, modular)
    p.add_argument(
        "--policy-arch",
        type=str,
        default="mlp_ppo",
        help=f"Registered policy architecture. Available: see twt_models/",
    )
    p.add_argument(
        "--policy-kwargs",
        type=str,
        default="{}",
        help="JSON string of kwargs passed to the policy constructor",
    )
    # Parallelism + episode shape
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--max-steps-per-episode", type=int, default=100)
    p.add_argument(
        "--warmup-steps", type=int, default=5
    )  # settle apps/queues/energy before recording (2->5, 2026-06-02)
    p.add_argument("--reward-preset", type=str, default="balanced")
    p.add_argument("--base-seed", type=int, default=90000)
    # Topology: non-REHD STA count + REHD energy-harvesting sensor count.
    # total active = n_stations + n_rehd must be <= MAX_NUM_STA (32).
    # n_rehd>0 activates the energy/QoEH machinery (REHDs get harvesters).
    # Default mix keeps the established 16-STA scale with 1/4 REHDs.
    p.add_argument(
        "--n-stations",
        type=int,
        default=12,
        help="non-REHD STA count (IoT/Camera/Voice/Video)",
    )
    p.add_argument(
        "--n-rehd",
        type=int,
        default=8,
        help="REHD energy-harvesting sensor count (canonical 12+8, matches the post-PDW EDA)",
    )
    p.add_argument(
        "--crn-groups",
        type=int,
        default=0,
        help="Common-Random-Numbers: scenario groups per batch (0=off; each worker its own "
        "scenario). G>0 = G shared-scenario groups of num_workers/G workers — same NS-3 "
        "randSeed within a group (clean action-contrast advantage), unique SHM per worker.",
    )
    # Rolling work-pool (heterogeneous-core dispatch-on-completion) + variable split
    p.add_argument(
        "--pool-size",
        type=int,
        default=0,
        help="Rolling-pool concurrent slots (0 = --num-workers). Feeds the next episode to "
        "whichever slot frees first (heterogeneous cores), no per-batch barrier.",
    )
    p.add_argument(
        "--episodes-per-batch",
        type=int,
        default=0,
        help="Episodes (rollouts) per PPO batch (0 = --num-workers). Decoupled from pool-size; "
        "all collected under the FROZEN policy => on-policy with only a tail-drain.",
    )
    p.add_argument(
        "--variable-split",
        action="store_true",
        default=False,
        help="Vary n_rehd ~ U{4..16} per scenario at fixed total 20 (CRN-safe via rand_seed). "
        "Creates observable inter-scenario adaptivity; overrides --n-stations/--n-rehd per ep.",
    )
    # PPO hyperparameters
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument(
        "--seq-minibatch-size",
        type=int,
        default=4,
        help="Episodes per gradient step (used only for recurrent policies)",
    )
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--normalize-advantage", action="store_true", default=True)
    # Preprocessing (2026-06-07): per-feature running obs z-score (in-policy buffers) + return normalization.
    # Obs warm-start seeds the obs_rms from the EDA stats.
    p.add_argument(
        "--normalize-return",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale rewards by running return std (SB3 VecNormalize-style).",
    )
    p.add_argument(
        "--asymmetric-critic",
        action="store_true",
        default=False,
        help="lstm_ppo only: feed privileged per-STA ORACLE energy state "
        "(SoC, is_rehd, dHarvested, dConsumed) to the VALUE path only "
        "(actor stays realistic). Lets V() price the state through which "
        "the delayed PDW->harvest->future-service reward flows, fixing the "
        "backwards-PDW credit assignment. Invalidates symmetric checkpoints.",
    )
    p.add_argument(
        "--obs-warmstart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start the policy obs_rms from obs_warmstart_stats.json (EDA stats).",
    )
    # I/O
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--tensorboard-log", type=str, default="tb_logs")
    p.add_argument(
        "--save-freq", type=int, default=10, help="Save a checkpoint every N batches"
    )
    p.add_argument(
        "--worker-timeout",
        type=float,
        default=1200.0,
        help="Seconds to wait for each worker rollout via queue",
    )
    # Resume a long run from a saved checkpoint (policy + Adam state).
    # Lets multi-hour trainings run in OOM-safe chunks.
    # --num-batches is the TOTAL target; with --start-batch N the loop runs [N, num_batches) and saves ckpt_batch_{N+1..}.
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to ckpt_batch_*.pt / ckpt_final.pt to resume policy+optimizer from",
    )
    p.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="Batch index to resume from (e.g. 50 after ckpt_batch_00050); only used with --resume",
    )
    # Verbosity / trace controls (defaults: silent — only progress bar + errors)
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Master verbose gate: orchestrator info prints + NS-3 stdout + traces",
    )
    p.add_argument(
        "--verbose-ns3",
        action="store_true",
        dest="verbose",
        help=argparse.SUPPRESS,  # legacy alias, same effect as --verbose
    )
    p.add_argument(
        "--keep-ns3-stdout",
        action="store_true",
        help="Override: re-enable NS-3 stdout only (without orchestrator chatter)",
    )
    p.add_argument(
        "--keep-ns3-traces",
        action="store_true",
        help="Override: re-enable NS-3 CSV trace files only",
    )
    return p.parse_args()


# --- Cleanup ---
def cleanup_all_shm(seeds):
    for s in seeds:
        for nm in (
            f"/dev/shm/MySeg_{s}",
            f"/dev/shm/MyCpp2PyMsg_{s}",
            f"/dev/shm/MyPy2CppMsg_{s}",
            f"/dev/shm/MyLockable_{s}",
        ):
            if os.path.exists(nm):
                try:
                    os.remove(nm)
                except OSError:
                    pass


# --- GAE + returns ---
def compute_gae(rewards, values, dones, bootstrap_value, gamma, gae_lambda):
    """Standard GAE for one rollout. Returns numpy arrays (advantages, returns)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_v = 0.0 if dones[t] else bootstrap_value
            next_nonterminal = 0.0 if dones[t] else 1.0
        else:
            next_v = values[t + 1]
            next_nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
        advantages[t] = last_adv
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# --- Spawn N workers, drain queue, join (wifi-simulation discipline) ---
def collect_rollouts(args, batch_idx, policy_payload_bytes, ns3_root, pbar=None):
    """Collect K episodes (the PPO batch) via a ROLLING WORK-POOL: M slots, dispatch the next
    CRN-tagged work-item to whichever frees first (heterogeneous-core friendly, no per-batch
    barrier). All K run under the FROZEN policy payload => on-policy with only a tail-drain.
    Each work-item carries a UNIQUE SHM `seed`, its CRN-group `rand_seed` (the scenario), and a
    per-scenario `n_rehd` (variable split). Back-compat: pool=K=num_workers, no variable-split
    => same seeds/scenarios as the old batch loop."""
    from twt_pool import run_rolling_pool

    K = (
        args.episodes_per_batch
        if getattr(args, "episodes_per_batch", 0)
        else args.num_workers
    )
    M = args.pool_size if getattr(args, "pool_size", 0) else args.num_workers
    quiet_ns3 = not (args.verbose or args.keep_ns3_stdout)
    disable_ns3_traces = not (args.verbose or args.keep_ns3_traces)
    G = (
        args.crn_groups
        if (getattr(args, "crn_groups", 0) and args.crn_groups > 0)
        else 0
    )
    per = max(1, K // G) if G else 1
    var_split = getattr(args, "variable_split", False)
    if var_split:
        from twt_scenario import sample_split

    items = []
    for k in range(K):
        shm_seed = args.base_seed + batch_idx * K + k  # unique SHM per episode
        rand_seed = (
            (args.base_seed + 1_000_000 + batch_idx * G + (k // per)) if G else None
        )
        scenario = rand_seed if rand_seed is not None else shm_seed
        n_sta, n_rehd = (
            sample_split(scenario) if var_split else (args.n_stations, args.n_rehd)
        )
        items.append(
            {
                "worker_id": k,
                "seed": shm_seed,
                "policy_payload_bytes": policy_payload_bytes,
                "ns3_path": ns3_root,
                "reward_type": args.reward_preset,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps_per_episode,
                "quiet_ns3": quiet_ns3,
                "disable_ns3_traces": disable_ns3_traces,
                "n_stations": n_sta,
                "n_rehd": n_rehd,
                "rand_seed": rand_seed,
            }
        )
    cleanup_all_shm([it["seed"] for it in items])

    def _on_result(r, done, total):
        if pbar is not None:
            pbar.set_postfix(
                {
                    "batch": f"{batch_idx + 1}/{args.num_batches}",
                    "eps": f"{done}/{total}",
                },
                refresh=True,
            )

    results = run_rolling_pool(
        items,
        run_episode,
        pool_size=M,
        result_timeout=getattr(args, "worker_timeout", 900),
        on_result=_on_result,
    )
    cleanup_all_shm([it["seed"] for it in items])
    return results


# --- PPO update on collected rollouts (option A: works for any BasePolicy) ---
def ppo_update(policy, optimizer, rollouts_data, args):
    """
    rollouts_data: list of dicts, each with keys
        'obs'        : (T, obs_dim)  np.float32
        'log_prob'   : (T,)          np.float32
        'value'      : (T,)          np.float32
        'reward'     : (T,)          np.float32
        'done'       : (T,)          np.float32
        'bootstrap'  : float

    Plus, depending on policy.is_raw_action_policy:
        Indexed (legacy):
            'sched'  : (T,) int64
            'assign' : (T,) int64
        Raw (pointer_ppo):
            'sched'      : (T,)         int64
            'sta_groups' : (T, num_sta) int64
    """
    is_raw = bool(getattr(policy, "is_raw_action_policy", False))
    is_recurrent = bool(getattr(policy, "is_temporal_recurrent_policy", False))

    # Recurrent policies need sequence-level minibatching (LSTM rolls through a whole episode in one go).
    # Dispatch to the dedicated path; flat-shuffle below is incorrect for these because it discards temporal context.
    if is_recurrent:
        return _ppo_update_recurrent(policy, optimizer, rollouts_data, args, is_raw)

    # Compute advantages + returns per rollout, and stack per-key arrays
    all_obs, all_logp_old, all_adv, all_ret = [], [], [], []
    all_sched = []
    all_assign = []  # indexed only
    all_sta_groups = []  # raw only
    all_pdw = []  # 3rd action head (both interfaces)
    for r in rollouts_data:
        adv, ret = compute_gae(
            r["reward"],
            r["value"],
            r["done"],
            r["bootstrap"],
            args.gamma,
            args.gae_lambda,
        )
        all_obs.append(r["obs"])
        all_sched.append(r["sched"])
        all_pdw.append(r["pdw"])
        if is_raw:
            all_sta_groups.append(r["sta_groups"])
        else:
            all_assign.append(r["assign"])
        all_logp_old.append(r["log_prob"])
        all_adv.append(adv)
        all_ret.append(ret)

    obs = torch.from_numpy(np.concatenate(all_obs, axis=0).astype(np.float32))
    sched = torch.from_numpy(np.concatenate(all_sched, axis=0).astype(np.int64))
    pdw = torch.from_numpy(np.concatenate(all_pdw, axis=0).astype(np.int64))
    if is_raw:
        sta_groups = torch.from_numpy(
            np.concatenate(all_sta_groups, axis=0).astype(np.int64)
        )
        assign = None
    else:
        assign = torch.from_numpy(np.concatenate(all_assign, axis=0).astype(np.int64))
        sta_groups = None
    logp_old = torch.from_numpy(np.concatenate(all_logp_old, axis=0).astype(np.float32))
    adv = torch.from_numpy(np.concatenate(all_adv, axis=0).astype(np.float32))
    ret = torch.from_numpy(np.concatenate(all_ret, axis=0).astype(np.float32))

    if args.normalize_advantage and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs.shape[0]
    mb = max(1, min(args.minibatch_size, N))
    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_frac": [],
    }

    policy.train()
    for _ in range(args.n_epochs):
        idx = torch.randperm(N)
        for start in range(0, N, mb):
            mb_idx = idx[start : start + mb]
            mb_obs = obs[mb_idx]
            mb_sched = sched[mb_idx]
            mb_pdw = pdw[mb_idx]
            mb_logp_old = logp_old[mb_idx]
            mb_adv = adv[mb_idx]
            mb_ret = ret[mb_idx]

            if is_raw:
                payload_batch = {
                    "sched": mb_sched,
                    "sta_groups": sta_groups[mb_idx],
                    "pdw": mb_pdw,
                }
                log_prob, value, entropy = policy.evaluate_raw(mb_obs, payload_batch)
            else:
                mb_assign = assign[mb_idx]
                log_prob, value, entropy = policy.evaluate(
                    mb_obs, mb_sched, mb_assign, mb_pdw
                )

            ratio = torch.exp(log_prob - mb_logp_old)
            surr1 = ratio * mb_adv
            surr2 = (
                torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                * mb_adv
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, mb_ret)
            entropy_mean = entropy.mean()

            loss = (
                policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_mean
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_mean.item())
                metrics["approx_kl"].append((mb_logp_old - log_prob).mean().item())
                metrics["clip_frac"].append(
                    ((ratio - 1.0).abs() > args.clip_range).float().mean().item()
                )
    policy.eval()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# --- PPO update for TEMPORAL RECURRENT policies (e.g. lstm_ppo) ---
# Differs from the flat-shuffle path above in two ways:
#   1. Episodes are NEVER concatenated. Each rollout in `rollouts_data`
#      stays as one chronological sequence (T_i, ...).
#   2. Each gradient step processes a whole minibatch of episodes via
#      policy.evaluate_sequence(...) (or evaluate_sequence_raw for raw),
#      which rolls the LSTM from h_0 = c_0 = 0 through the full T_i steps.
#      The per-step PPO loss is then computed on the per-step (log_prob,
#      value, entropy) tensors returned by the policy.
#
# This is the wifi-simulation-pattern equivalent of sb3-contrib's RecurrentRolloutBuffer: each worker returns a complete trajectory, so we never have to reconstruct partial-episode hidden states.
# Re-rolling from h_0 = 0 reproduces the rollout-time hidden states because the policy weights are unchanged during the rollout.
def _ppo_update_recurrent(policy, optimizer, rollouts_data, args, is_raw):
    # Compute returns + advantages per episode (kept as separate sequences).
    for r in rollouts_data:
        adv, ret = compute_gae(
            r["reward"],
            r["value"],
            r["done"],
            r["bootstrap"],
            args.gamma,
            args.gae_lambda,
        )
        r["adv"] = adv.astype(np.float32)
        r["ret"] = ret.astype(np.float32)

    # Global advantage normalization (across all timesteps in this batch).
    if args.normalize_advantage:
        all_adv = np.concatenate([r["adv"] for r in rollouts_data])
        if all_adv.size > 1:
            mean = float(all_adv.mean())
            std = float(all_adv.std()) + 1e-8
            for r in rollouts_data:
                r["adv"] = (r["adv"] - mean) / std

    n_eps = len(rollouts_data)
    seq_mb = max(1, args.seq_minibatch_size)

    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_frac": [],
    }

    policy.train()
    for _ in range(args.n_epochs):
        ep_indices = np.random.permutation(n_eps)
        for mb_start in range(0, n_eps, seq_mb):
            mb_eps = [
                rollouts_data[i] for i in ep_indices[mb_start : mb_start + seq_mb]
            ]

            # Run each sequence through the LSTM, accumulate per-step tensors.
            log_prob_chunks = []
            logp_old_chunks = []
            value_chunks = []
            ret_chunks = []
            adv_chunks = []
            entropy_chunks = []

            for r in mb_eps:
                obs_seq = torch.from_numpy(r["obs"].astype(np.float32))
                sched_seq = torch.from_numpy(r["sched"].astype(np.int64))
                pdw_seq = torch.from_numpy(r["pdw"].astype(np.int64))
                logp_old_seq = torch.from_numpy(r["log_prob"].astype(np.float32))
                adv_seq = torch.from_numpy(r["adv"])
                ret_seq = torch.from_numpy(r["ret"])

                if is_raw:
                    payload_seq = {
                        "sched": sched_seq,
                        "sta_groups": torch.from_numpy(
                            r["sta_groups"].astype(np.int64)
                        ),
                        "pdw": pdw_seq,
                    }
                    log_prob_seq, value_seq, entropy_seq = policy.evaluate_sequence_raw(
                        obs_seq, payload_seq
                    )
                else:
                    assign_seq = torch.from_numpy(r["assign"].astype(np.int64))
                    critic_seq = (
                        torch.from_numpy(r["critic_obs"].astype(np.float32))
                        if "critic_obs" in r
                        else None
                    )
                    log_prob_seq, value_seq, entropy_seq = policy.evaluate_sequence(
                        obs_seq, sched_seq, assign_seq, pdw_seq, critic_seq=critic_seq
                    )

                log_prob_chunks.append(log_prob_seq)
                logp_old_chunks.append(logp_old_seq)
                value_chunks.append(value_seq)
                ret_chunks.append(ret_seq)
                adv_chunks.append(adv_seq)
                entropy_chunks.append(entropy_seq)

            log_prob = torch.cat(log_prob_chunks)
            logp_old = torch.cat(logp_old_chunks)
            value = torch.cat(value_chunks)
            ret = torch.cat(ret_chunks)
            adv = torch.cat(adv_chunks)
            entropy = torch.cat(entropy_chunks)

            ratio = torch.exp(log_prob - logp_old)
            surr1 = ratio * adv
            surr2 = (
                torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * adv
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret)
            entropy_mean = entropy.mean()

            loss = (
                policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_mean
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_mean.item())
                metrics["approx_kl"].append((logp_old - log_prob).mean().item())
                metrics["clip_frac"].append(
                    ((ratio - 1.0).abs() > args.clip_range).float().mean().item()
                )

    policy.eval()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


# --- Convert worker rollout (list of dicts) -> per-array dict for ppo_update ---
def stack_rollout(worker_result, is_raw_action_policy: bool = False):
    rollout = worker_result["rollout"]
    if not rollout:
        return None
    base = {
        "obs": np.stack([t["obs"] for t in rollout], axis=0).astype(np.float32),
        "log_prob": np.array([t["log_prob"] for t in rollout], dtype=np.float32),
        "value": np.array([t["value"] for t in rollout], dtype=np.float32),
        "reward": np.array([t["reward"] for t in rollout], dtype=np.float32),
        "done": np.array(
            [1.0 if t["done"] else 0.0 for t in rollout], dtype=np.float32
        ),
        "bootstrap": float(worker_result.get("bootstrap_value", 0.0)),
    }
    # Asymmetric critic: carry the per-step privileged oracle features (critic-only).
    if rollout[0].get("critic_obs") is not None:
        base["critic_obs"] = np.stack(
            [t["critic_obs"] for t in rollout], axis=0
        ).astype(np.float32)
    if is_raw_action_policy:
        base["sched"] = np.array(
            [int(t["payload"]["sched"]) for t in rollout], dtype=np.int64
        )
        base["sta_groups"] = np.stack(
            [np.asarray(t["payload"]["sta_groups"], dtype=np.int64) for t in rollout],
            axis=0,
        )
        base["pdw"] = np.array(
            [int(t["payload"].get("pdw", 0)) for t in rollout], dtype=np.int64
        )
    else:
        base["sched"] = np.array([t["schedule"] for t in rollout], dtype=np.int64)
        base["assign"] = np.array([t["assignment"] for t in rollout], dtype=np.int64)
        base["pdw"] = np.array([int(t.get("pdw", 0)) for t in rollout], dtype=np.int64)
    return base


# --- Main ---
def main():
    args = parse_args()
    user_kwargs = json.loads(args.policy_kwargs) if args.policy_kwargs else {}
    # Derive obs shape from the topology (n_stations + n_rehd); flat obs = num_sta·F so obs_dim MUST track it.
    # User --policy-kwargs override the auto-sized values.
    n_total = int(args.n_stations or 0) + int(args.n_rehd or 0)
    policy_kwargs = {**default_obs_kwargs(args.policy_arch, n_total), **user_kwargs}
    # Asymmetric critic: size the critic's privileged oracle-feature block to the topology.
    # setdefault so an explicit --policy-kwargs still wins.
    if args.asymmetric_critic and args.policy_arch == "lstm_ppo":
        policy_kwargs.setdefault("critic_features_per_sta", NUM_CRITIC_FEATURES)
        policy_kwargs.setdefault("critic_extra_dim", n_total * NUM_CRITIC_FEATURES)

    print("=" * 70)
    print("  TWT Batch Orchestrator (parallel PPO, wifi-simulation pattern)")
    print("=" * 70)
    print(
        f"  topology         : {args.n_stations} STA + {args.n_rehd} REHD = {n_total} total"
    )
    print(f"  policy           : {args.policy_arch}  kwargs={policy_kwargs}")
    print(f"  registered       : {list_policies()}")
    print(f"  workers          : {args.num_workers}")
    print(f"  batches          : {args.num_batches}")
    print(f"  steps/episode    : {args.max_steps_per_episode}")
    print(f"  reward preset    : {args.reward_preset}")
    print(f"  base seed        : {args.base_seed}")
    print("=" * 70)

    # ns3 root: ppo-sb3-scripts/ -> twt/ -> examples/ -> ai/ -> contrib/ -> ns-3.44/
    ns3_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", "..", ".."))

    # Output dirs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"twt_{args.policy_arch}_{args.reward_preset}_{timestamp}"
    out_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(_script_dir, args.output_dir)
    )
    tb_dir = (
        args.tensorboard_log
        if os.path.isabs(args.tensorboard_log)
        else os.path.join(_script_dir, args.tensorboard_log)
    )
    ckpt_dir = os.path.join(out_dir, run_name)
    log_dir = os.path.join(tb_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print(f"  ckpt dir         : {ckpt_dir}")
    print(f"  tb log dir       : {log_dir}")

    # Save hyperparams
    with open(os.path.join(ckpt_dir, "hyperparams.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "policy_kwargs_parsed": policy_kwargs,
                "ns3_root": ns3_root,
                "run_name": run_name,
            },
            f,
            indent=2,
        )

    # Build policy + optimizer (the master copy lives only here)
    policy = get_policy(args.policy_arch, **policy_kwargs)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    ret_norm = ReturnNormalizer(gamma=args.gamma) if args.normalize_return else None
    n_params = sum(p.numel() for p in policy.parameters())
    is_raw_action_policy = bool(getattr(policy, "is_raw_action_policy", False))
    print(f"  policy params    : {n_params:,}")
    print(f"  action interface : {'raw' if is_raw_action_policy else 'indexed'}")

    # Resume: restore policy + optimizer + batch offset. Checkpoints save
    # name/kwargs/state_dict/optimizer (see save block below); load_state_dict
    # (strict by default) self-validates the arch/topology match.
    start_batch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if ckpt.get("name") != args.policy_arch:
            raise SystemExit(
                f"--resume arch mismatch: ckpt name={ckpt.get('name')!r} "
                f"vs --policy-arch={args.policy_arch!r}"
            )
        policy.load_state_dict(ckpt["state_dict"])  # restores obs_rms buffers too
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if ret_norm is not None and ckpt.get("ret_norm") is not None:
            ret_norm.load_state_dict(ckpt["ret_norm"])
        start_batch = int(args.start_batch)
        print(f"  RESUMED          : {args.resume}")
        print(
            f"  resuming         : running batches [{start_batch + 1}, {args.num_batches}]"
        )

    # Warm-start the per-feature obs normalizer from the EDA stats (only on a FRESH run --
    # a resume already restored the trained obs_rms buffers via load_state_dict).
    if (
        (not args.resume)
        and args.obs_warmstart
        and getattr(policy, "_obs_norm_enabled", False)
    ):
        _ws_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "obs_warmstart_stats.json"
        )
        if os.path.exists(_ws_path):
            _ws = json.load(open(_ws_path))
            policy.warmstart_obs_rms(_ws["mean"], _ws["std"])
            print(
                f"  obs warm-start   : {len(_ws['mean'])} feat from {os.path.basename(_ws_path)}"
            )
        else:
            print(f"  obs warm-start   : SKIPPED (no {_ws_path}); obs_rms cold-starts")

    writer = SummaryWriter(log_dir=log_dir)
    jsonl = open(os.path.join(ckpt_dir, "training_log.jsonl"), "w", buffering=1)

    total_env_steps = 0
    train_t0 = time.time()
    pbar = tqdm(
        range(start_batch, args.num_batches),
        desc="train",
        unit="batch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    for batch_idx in pbar:
        if args.verbose:
            tqdm.write(f"\n--- BATCH {batch_idx + 1}/{args.num_batches} ---")
        batch_t0 = time.time()

        # Snapshot weights -> bytes (self-describing payload)
        payload = {
            "name": args.policy_arch,
            "kwargs": policy_kwargs,
            "state_dict": policy.state_dict(),
        }
        payload_bytes = pickle.dumps(payload)

        # Parallel rollout collection (live workers-done counter on the pbar)
        results = collect_rollouts(args, batch_idx, payload_bytes, ns3_root, pbar=pbar)

        rollouts_data = []
        ep_rewards = []
        ep_lengths = []
        n_failed = 0
        for r in sorted(results, key=lambda x: x.get("worker_id", -1)):
            wid = r.get("worker_id")
            if r.get("success"):
                stacked = stack_rollout(r, is_raw_action_policy=is_raw_action_policy)
                if stacked is not None:
                    rollouts_data.append(stacked)
                    ep_rewards.append(r["episode_reward"])
                    ep_lengths.append(r["episode_length"])
                    if args.verbose:
                        tqdm.write(
                            f"  worker {wid}: len={r['episode_length']}  "
                            f"reward={r['episode_reward']:.3f}"
                        )
            else:
                n_failed += 1
                err = r.get("error", "<no error msg>")
                tqdm.write(f"  worker {wid}: FAILED — {err[:300]}")  # error: always

        if not rollouts_data:
            tqdm.write("  no successful rollouts; skipping update")  # error: always
            continue

        # Return normalization (SB3 VecNormalize-style): scale rewards by the running return std BEFORE GAE so value targets are ~unit-scaled.
        # Per-episode (rollouts are in time order).
        # MUST precede ppo_update (which reads r["reward"]).
        if ret_norm is not None:
            for r in rollouts_data:
                r["reward"] = ret_norm.update_and_scale(r["reward"], r["done"])

        update_t0 = time.time()
        train_metrics = ppo_update(policy, optimizer, rollouts_data, args)
        update_dt = time.time() - update_t0

        # Refresh the per-feature obs running stats from this batch's RAW obs, AFTER the update --
        # so the stats used during the update == the stats the workers used to act (the next batch's pickled state_dict carries the refreshed buffers).
        if getattr(policy, "_obs_norm_enabled", False):
            policy.update_obs_rms(
                np.concatenate([r["obs"] for r in rollouts_data], axis=0)
            )
        # Same for the asymmetric critic's privileged-feature normalizer.
        if getattr(policy, "_crit_norm_enabled", False):
            _crit = [r["critic_obs"] for r in rollouts_data if "critic_obs" in r]
            if _crit:
                policy.update_critic_rms(np.concatenate(_crit, axis=0))

        batch_dt = time.time() - batch_t0
        batch_steps = sum(len(r["reward"]) for r in rollouts_data)
        total_env_steps += batch_steps

        avg_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        avg_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0

        # Log to TB
        writer.add_scalar("rollout/mean_episode_reward", avg_rew, batch_idx)
        writer.add_scalar("rollout/mean_episode_length", avg_len, batch_idx)
        writer.add_scalar("rollout/total_env_steps", total_env_steps, batch_idx)
        writer.add_scalar("rollout/n_failed", n_failed, batch_idx)
        writer.add_scalar("time/batch_seconds", batch_dt, batch_idx)
        writer.add_scalar("time/update_seconds", update_dt, batch_idx)
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, batch_idx)
        writer.flush()

        # JSONL
        jsonl.write(
            json.dumps(
                {
                    "batch": batch_idx,
                    "wall_time": time.time() - train_t0,
                    "batch_seconds": batch_dt,
                    "update_seconds": update_dt,
                    "total_env_steps": total_env_steps,
                    "ep_reward_mean": avg_rew,
                    "ep_length_mean": avg_len,
                    "n_workers_ok": len(rollouts_data),
                    "n_failed": n_failed,
                    **train_metrics,
                }
            )
            + "\n"
        )

        if args.verbose:
            tqdm.write(
                f"  batch {batch_idx + 1} done in {batch_dt:.1f}s "
                f"(update {update_dt:.1f}s)  avg_rew={avg_rew:.3f}  "
                f"failed={n_failed}"
            )

        # Live numbers in the progress bar's postfix (replaces the workers-done counter set during rollout collection).
        pbar.set_postfix(
            {
                "batch": f"{batch_idx + 1}/{args.num_batches}",
                "avg_rew": f"{avg_rew:.3f}",
                "ep_len": f"{avg_len:.1f}",
                "batch_s": f"{batch_dt:.0f}",
                "steps": total_env_steps,
                "fail": n_failed,
                "kl": f"{train_metrics.get('approx_kl', 0.0):.3f}",
            }
        )

        # Checkpoint
        if ((batch_idx + 1) % args.save_freq == 0) or (
            batch_idx + 1 == args.num_batches
        ):
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_batch_{batch_idx + 1:05d}.pt")
            torch.save(
                {
                    "name": args.policy_arch,
                    "kwargs": policy_kwargs,
                    "state_dict": policy.state_dict(),  # incl. obs_rms buffers
                    "optimizer": optimizer.state_dict(),
                    "ret_norm": ret_norm.state_dict() if ret_norm is not None else None,
                    "batch_idx": batch_idx + 1,
                },
                ckpt_path,
            )
            if args.verbose:
                tqdm.write(f"  checkpoint -> {ckpt_path}")

    pbar.close()

    # Final checkpoint (overwrites if same)
    final_path = os.path.join(ckpt_dir, "ckpt_final.pt")
    torch.save(
        {
            "name": args.policy_arch,
            "kwargs": policy_kwargs,
            "state_dict": policy.state_dict(),  # incl. obs_rms buffers
            "optimizer": optimizer.state_dict(),
            "ret_norm": ret_norm.state_dict() if ret_norm is not None else None,
        },
        final_path,
    )
    print(f"\nfinal checkpoint -> {final_path}")

    writer.close()
    jsonl.close()
    print(f"total wall time: {time.time() - train_t0:.1f}s")


if __name__ == "__main__":
    mp.set_start_method(
        "spawn"
    )  # spawn, not fork: each worker starts a clean interpreter and imports the ns3-ai binding itself
    main()
