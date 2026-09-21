#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_evaluator.py - Parallel evaluator for the TWT pipeline.

Same wifi-simulation pattern as twt_batch_orchestrator.py, but:
  - workers run with deterministic=True (argmax actions, no exploration)
  - no PPO update -- we just collect metrics
  - supports 1 or many policies in --compare mode (writes a side-by-side CSV)

Each --compare item is either:
  - a path ending in .pt   -> a checkpoint saved by twt_batch_orchestrator
  - a registered policy name (e.g. "random_policy", "mlp_ppo")

Examples:
  # Eval one trained checkpoint
  python3.11 twt_evaluator.py \
      --checkpoint checkpoints/twt_mlp_ppo_balanced_<ts>/ckpt_final.pt \
      --n-episodes 20 --num-workers 4

  # Eval just a baseline (no checkpoint)
  python3.11 twt_evaluator.py \
      --policy-arch random_policy \
      --n-episodes 20 --num-workers 4

  # Compare a checkpoint against baselines
  python3.11 twt_evaluator.py \
      --compare checkpoints/.../ckpt_final.pt random_policy \
      --n-episodes 20 --num-workers 4
"""

import os
import warnings

# Silence TF/XLA chatter from torch's transitive deps and benign UserWarnings (e.g. pointer_ppo's TransformerEncoder enable_nested_tensor).
# Must run BEFORE the torch import.
# Env var is inherited by spawn children; the warnings filter is re-applied because spawn children re-execute this top-level block.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import csv
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.join(_parent_dir, "exploration-scripts"))

from twt_models import get_policy, list_policies
from twt_spawn_worker import run_episode, default_obs_kwargs
from twt_scenario import sample_split


# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel evaluator for TWT (wifi-simulation pattern).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # What to evaluate (mutually-exclusive sources)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint saved by the orchestrator",
    )
    src.add_argument(
        "--policy-arch",
        type=str,
        default=None,
        help=f"Registered policy name (no checkpoint needed). "
        f"Available at runtime: see twt_models/",
    )
    src.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Two or more entries, each either a .pt path or a "
        "registered policy name; produces a side-by-side CSV",
    )

    # Episode shape
    p.add_argument(
        "--n-episodes", type=int, default=20, help="Total eval episodes per policy"
    )
    p.add_argument(
        "--num-workers", type=int, default=4, help="Parallel workers per batch"
    )
    p.add_argument("--max-steps-per-episode", type=int, default=20)
    p.add_argument(
        "--warmup-steps", type=int, default=5
    )  # match training (2->5, 2026-06-02)
    p.add_argument("--reward-preset", type=str, default="balanced")
    p.add_argument("--base-seed", type=int, default=200000)
    p.add_argument("--worker-timeout", type=float, default=900.0)
    # Topology MUST match the trained checkpoint's training topology (analogous to the train/eval horizon rule).
    # Defaults mirror the orchestrator (12 + 4 REHD).
    p.add_argument(
        "--n-stations",
        type=int,
        default=12,
        help="non-REHD STA count (must match training; ignored if --variable-split)",
    )
    p.add_argument(
        "--n-rehd",
        type=int,
        default=8,
        help="REHD energy-harvesting sensor count (must match training; canonical 12+8; ignored if --variable-split)",
    )
    p.add_argument(
        "--variable-split",
        action="store_true",
        default=False,
        help="Sample (n_stations, n_rehd) per-episode via twt_scenario.sample_split(seed) "
        "(total fixed at 20, n_rehd~U[4,16]) -- must match training",
    )
    p.add_argument(
        "--non-deterministic",
        action="store_true",
        default=False,
        help="Sample actions instead of argmax (debug / variance studies)",
    )
    # NS-3 verbosity / trace controls (defaults: silent + no CSV files)
    p.add_argument(
        "--verbose-ns3",
        action="store_true",
        help="Re-enable both NS-3 stdout AND CSV trace files (debug)",
    )
    p.add_argument(
        "--keep-ns3-stdout",
        action="store_true",
        help="Re-enable NS-3 stdout chatter only",
    )
    p.add_argument(
        "--keep-ns3-traces",
        action="store_true",
        help="Re-enable NS-3 CSV trace files only",
    )

    # I/O
    p.add_argument("--output-dir", type=str, default="eval_results")
    return p.parse_args()


# --- Helpers ---
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


def load_policy_payload(arg: str, n_total: int = 0) -> Tuple[bytes, str, dict, str]:
    """
    Resolve a CLI arg (either a .pt path or a registered policy name) to a
    pickled policy payload that workers can ingest.

    Args:
        arg: a .pt path or a registered policy name.
        n_total: active STA count (n_stations + n_rehd). For REGISTRY baselines
            (no embedded kwargs) the obs shape is auto-sized from this so they
            match the eval topology. Checkpoints carry their own kwargs (already
            sized to their training topology — eval must match train).

    Returns:
        (payload_bytes, display_name, kwargs, source_kind)
            source_kind is "checkpoint" or "registry".
    """
    if arg and arg.endswith(".pt"):
        if not os.path.exists(arg):
            raise FileNotFoundError(f"checkpoint not found: {arg}")
        ckpt = torch.load(arg, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "name" not in ckpt or "state_dict" not in ckpt:
            raise ValueError(
                f"checkpoint {arg} is not in the expected format "
                f"(must have 'name' and 'state_dict' keys)"
            )
        name = ckpt["name"]
        kwargs = ckpt.get("kwargs", {}) or {}
        # Sanity: rebuild the policy locally to make sure it loads cleanly.
        policy = get_policy(name, **kwargs)
        policy.load_state_dict(ckpt["state_dict"])
        payload = {"name": name, "kwargs": kwargs, "state_dict": policy.state_dict()}
        display = f"{name}@{os.path.basename(arg)}"
        return pickle.dumps(payload), display, kwargs, "checkpoint"

    elif arg in list_policies():
        kwargs = default_obs_kwargs(arg, n_total) if n_total > 0 else {}
        policy = get_policy(arg, **kwargs)
        payload = {"name": arg, "kwargs": kwargs, "state_dict": policy.state_dict()}
        return pickle.dumps(payload), arg, kwargs, "registry"

    else:
        raise ValueError(
            f"'{arg}' is neither a .pt file nor a registered policy "
            f"(known: {list_policies()})"
        )


# --- Run N episodes, in batches of num_workers, return per-episode metrics. ---
def evaluate_policy(
    payload_bytes: bytes, display_name: str, args, ns3_root: str
) -> List[dict]:
    """Returns a list of per-episode metric dicts (n_episodes long)."""
    n_eps = args.n_episodes
    nw = args.num_workers
    n_batches = math.ceil(n_eps / nw)
    print(f"\n========== Evaluating: {display_name} ==========")
    print(f"  episodes: {n_eps}  workers: {nw}  batches: {n_batches}", flush=True)

    eps_done = 0
    per_episode = []

    for b in range(n_batches):
        this_batch = min(nw, n_eps - eps_done)
        seeds = [args.base_seed + b * nw + wid for wid in range(this_batch)]
        cleanup_all_shm(seeds)

        # Resolve NS-3 verbosity from CLI flags (defaults to fully silent).
        quiet_ns3 = not (args.verbose_ns3 or args.keep_ns3_stdout)
        disable_ns3_traces = not (args.verbose_ns3 or args.keep_ns3_traces)

        result_queue = mp.Queue()
        workers = []
        for wid, seed in enumerate(seeds):
            if args.variable_split:
                n_sta, n_rehd = sample_split(seed)
            else:
                n_sta, n_rehd = args.n_stations, args.n_rehd
            p = mp.Process(
                target=run_episode,
                args=(wid, seed, payload_bytes, result_queue, ns3_root),
                kwargs={
                    "reward_type": args.reward_preset,
                    "warmup_steps": args.warmup_steps,
                    "max_steps": args.max_steps_per_episode,
                    "deterministic": (not args.non_deterministic),
                    "quiet_ns3": quiet_ns3,
                    "disable_ns3_traces": disable_ns3_traces,
                    "n_stations": n_sta,
                    "n_rehd": n_rehd,
                },
            )
            p.start()
            workers.append(p)

        # Drain BEFORE join: a child blocked on a full queue pipe never exits
        results = []
        for _ in range(this_batch):
            try:
                results.append(result_queue.get(timeout=args.worker_timeout))
            except Exception as e:
                print(f"  queue.get FAILED: {e}", flush=True)
        try:
            result_queue.close()
        except Exception:
            pass

        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        cleanup_all_shm(seeds)

        for r in sorted(results, key=lambda x: x.get("worker_id", -1)):
            success = r.get("success", False)
            metrics = r.get("metrics", {}) or {}
            per_episode.append(
                {
                    "policy": display_name,
                    "worker_id": r.get("worker_id"),
                    "seed": r.get("seed"),
                    "n_stations": (
                        sample_split(r.get("seed"))[0]
                        if args.variable_split
                        else args.n_stations
                    ),
                    "n_rehd": (
                        sample_split(r.get("seed"))[1]
                        if args.variable_split
                        else args.n_rehd
                    ),
                    "success": bool(success),
                    "ep_length": int(r.get("episode_length", 0)),
                    "ep_reward": float(r.get("episode_reward", 0.0)),
                    "total_bytes_tx": int(metrics.get("total_bytes_tx", 0)),
                    "total_energy_mj": float(metrics.get("total_energy_mj", 0.0)),
                    "total_drops": int(metrics.get("total_drops", 0)),
                    "avg_queue_bytes": float(metrics.get("avg_queue_bytes", 0.0)),
                    "sim_time_sec": float(metrics.get("sim_time_sec", 0.0)),
                    "error": (r.get("error") or "")[:300] if not success else "",
                }
            )

        eps_done += this_batch
        print(
            f"  batch {b + 1}/{n_batches} done -- {eps_done}/{n_eps} episodes",
            flush=True,
        )

    return per_episode


# --- Aggregate per-episode -> mean / std / min / max per metric ---
def aggregate(per_episode: List[dict]) -> dict:
    fields = [
        "ep_reward",
        "ep_length",
        "total_bytes_tx",
        "total_energy_mj",
        "total_drops",
        "avg_queue_bytes",
        "sim_time_sec",
    ]
    ok = [e for e in per_episode if e["success"]]
    n_total = len(per_episode)
    n_ok = len(ok)
    out = {
        "policy": per_episode[0]["policy"] if per_episode else "?",
        "n_episodes": n_total,
        "n_successful": n_ok,
        "n_failed": n_total - n_ok,
    }
    for f in fields:
        vals = [float(e[f]) for e in ok]
        if vals:
            v = np.array(vals, dtype=np.float64)
            out[f"{f}_mean"] = float(v.mean())
            out[f"{f}_std"] = float(v.std())
            out[f"{f}_min"] = float(v.min())
            out[f"{f}_max"] = float(v.max())
        else:
            out[f"{f}_mean"] = out[f"{f}_std"] = out[f"{f}_min"] = out[f"{f}_max"] = 0.0
    return out


# --- CSV writers ---
def write_per_episode_csv(path: str, rows: List[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_csv(path: str, summaries: List[dict]):
    if not summaries:
        return
    # Union of keys across summaries (in case some policies failed harder)
    keys = []
    seen = set()
    for s in summaries:
        for k in s.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow(s)


# --- Main ---
def main():
    args = parse_args()

    # Resolve which policies to evaluate
    if args.compare:
        items = list(args.compare)
    elif args.checkpoint:
        items = [args.checkpoint]
    elif args.policy_arch:
        items = [args.policy_arch]
    else:
        print(
            "ERROR: pass --checkpoint <path>, --policy-arch <name>, or --compare a b ...",
            file=sys.stderr,
        )
        sys.exit(2)

    print("=" * 70)
    print("  TWT Evaluator (parallel, wifi-simulation pattern)")
    print("=" * 70)
    print(f"  registered policies : {list_policies()}")
    print(f"  to eval             : {items}")
    print(f"  episodes/policy     : {args.n_episodes}")
    print(f"  workers             : {args.num_workers}")
    print(f"  steps/episode       : {args.max_steps_per_episode}")
    print(f"  reward preset       : {args.reward_preset}")
    print(f"  deterministic       : {not args.non_deterministic}")
    print("=" * 70)

    # Resolve ns3 root
    ns3_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", "..", ".."))

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(_script_dir, args.output_dir)
    )
    run_dir = os.path.join(out_dir, f"eval_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"  output dir          : {run_dir}")

    # Save eval config
    with open(os.path.join(run_dir, "eval_config.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                "items": items,
                "ns3_root": ns3_root,
                "registered_policies": list_policies(),
            },
            f,
            indent=2,
        )

    # Run each item, collect per-episode rows + per-policy summary
    all_rows = []
    summaries = []
    for arg in items:
        try:
            n_total = (
                20
                if args.variable_split
                else int(args.n_stations or 0) + int(args.n_rehd or 0)
            )
            payload_bytes, display, _kw, kind = load_policy_payload(
                arg, n_total=n_total
            )
        except Exception as e:
            print(f"  skipping '{arg}': {e}")
            continue
        print(
            f"  loaded {kind}: '{arg}' -> display='{display}'  "
            f"(payload {len(payload_bytes)} bytes)"
        )

        rows = evaluate_policy(payload_bytes, display, args, ns3_root)
        all_rows.extend(rows)

        # Per-policy CSV (raw episodes)
        per_pol = os.path.join(run_dir, f"episodes__{display.replace('/', '_')}.csv")
        write_per_episode_csv(per_pol, rows)

        # Aggregate
        summary = aggregate(rows)
        summaries.append(summary)
        print(
            f"  summary [{display}]: "
            f"reward {summary['ep_reward_mean']:.3f} +/- {summary['ep_reward_std']:.3f}, "
            f"bytes_tx {summary['total_bytes_tx_mean']:.0f}, "
            f"energy_mj {summary['total_energy_mj_mean']:.2f}, "
            f"drops {summary['total_drops_mean']:.1f}, "
            f"ok {summary['n_successful']}/{summary['n_episodes']}",
            flush=True,
        )

    # Combined CSVs
    write_per_episode_csv(os.path.join(run_dir, "episodes_all.csv"), all_rows)
    write_summary_csv(os.path.join(run_dir, "summary.csv"), summaries)

    print("\n" + "=" * 70)
    print(f"  DONE. Results in: {run_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.exit(main())
