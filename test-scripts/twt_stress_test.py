#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_stress_test.py - Strict port of wifi-simulation/stress_test.py for TWT.

4 workers x 5 batches = 20 episodes. Confirms the parallelization actually
works under sustained load:
  - All workers spawn cleanly with disjoint SHM segments.
  - The orchestrator-side queue drain + join order survives back-to-back batches.
  - No /dev/shm/My* leftovers between or after batches.
  - No orphan twt-powercast-main-simulation processes.
  - Per-batch wall time stays roughly stable (no slowdown trend).

We use a randomly-initialized mlp_ppo policy here -- the goal is the
parallelization plumbing, not training. NOTE: max_steps_per_episode is small
(default 5) so a stress run finishes in a few minutes rather than hours.

Run from ns-3.44 root:
  cd /home/ahmak/NS3-project/ns-allinone-3.44/ns-3.44
  /home/ahmak/NS3-project/EHRL/bin/python3.11 \
      contrib/ai/examples/rl-twt-powercast/ppo-sb3-scripts/twt_stress_test.py
"""

import os
import warnings

# Silence TF/XLA chatter and benign torch UserWarnings; inherited by spawn children via os.environ + re-execution of the top-level block.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import glob
import time
import pickle
import argparse
import subprocess
import multiprocessing as mp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="NS-3 steps per episode (5 = quick stress; raise for realism)",
    )
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--reward-preset", type=str, default="balanced")
    p.add_argument("--base-seed", type=int, default=95000)
    p.add_argument("--worker-timeout", type=float, default=600.0)
    return p.parse_args()


def cleanup_all_shm(seeds):
    cleaned = []
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
                    cleaned.append(nm)
                except OSError:
                    pass
    return cleaned


def all_my_shm():
    return sorted(glob.glob("/dev/shm/My*"))


def orphan_twt_pids():
    r = subprocess.run(
        ["pgrep", "-f", "twt-powercast-main-simulation"], capture_output=True, text=True
    )
    pids = [x for x in r.stdout.strip().split() if x]
    return pids


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    ppo_dir = os.path.join(
        parent_dir, "ppo-sb3-scripts"
    )  # twt_models / twt_spawn_worker live there
    ns3_root = os.path.abspath(os.path.join(parent_dir, "..", "..", "..", ".."))

    sys.path.insert(0, ppo_dir)
    sys.path.insert(0, parent_dir)

    from twt_models import get_policy, list_policies
    from twt_spawn_worker import run_episode

    print("=" * 70)
    print(
        f"  TWT STRESS TEST: {args.num_workers} workers x {args.num_batches} batches"
        f" = {args.num_workers * args.num_batches} episodes"
    )
    print(f"  NS3 root         : {ns3_root}")
    print(f"  Python           : {sys.version.split()[0]}")
    print(f"  Registered models: {list_policies()}")
    print(f"  Reward preset    : {args.reward_preset}")
    print(f"  Steps/episode    : {args.max_steps}")
    print("=" * 70, flush=True)

    # Pre-clean any leftover shm from prior runs
    pre = all_my_shm()
    if pre:
        print(f"  Pre-existing shm: {pre} -- cleaning")
        for f in pre:
            try:
                os.remove(f)
            except OSError:
                pass

    # Build a (random) policy and freeze it as the payload for all workers.
    policy = get_policy("mlp_ppo")
    payload = {"name": "mlp_ppo", "kwargs": {}, "state_dict": policy.state_dict()}
    payload_bytes = pickle.dumps(payload)
    print(f"  payload size     : {len(payload_bytes)} bytes")

    total_t0 = time.time()
    batch_times = []
    n_failed_total = 0
    n_episodes_ok = 0

    for batch_idx in range(args.num_batches):
        print(f"\n--- BATCH {batch_idx + 1}/{args.num_batches} ---", flush=True)
        seeds = [
            args.base_seed + batch_idx * args.num_workers + wid
            for wid in range(args.num_workers)
        ]
        cleanup_all_shm(seeds)

        batch_t0 = time.time()
        result_queue = mp.Queue()
        workers = []
        for wid, seed in enumerate(seeds):
            p = mp.Process(
                target=run_episode,
                args=(wid, seed, payload_bytes, result_queue, ns3_root),
                kwargs={
                    "reward_type": args.reward_preset,
                    "warmup_steps": args.warmup_steps,
                    "max_steps": args.max_steps,
                    "quiet_ns3": True,
                    "disable_ns3_traces": True,
                },
            )
            p.start()
            workers.append(p)

        # Drain the queue BEFORE join: a child blocked on a full queue pipe never exits
        results = []
        for _ in range(args.num_workers):
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

        batch_dt = time.time() - batch_t0
        batch_times.append(batch_dt)

        # Per-worker summary
        ok = 0
        for r in sorted(results, key=lambda x: x.get("worker_id", -1)):
            if r.get("success"):
                ok += 1
                print(
                    f"  worker {r['worker_id']} (seed={r['seed']}): "
                    f"len={r['episode_length']}  reward={r['episode_reward']:.3f}",
                    flush=True,
                )
            else:
                err = (r.get("error") or "")[:200]
                print(f"  worker {r.get('worker_id')}: FAILED -- {err}", flush=True)
        n_episodes_ok += ok
        n_failed_total += args.num_workers - ok

        # Post-batch invariants
        leftover = all_my_shm()
        orphans = orphan_twt_pids()
        print(
            f"  batch {batch_idx + 1} done in {batch_dt:.1f}s   "
            f"shm_leftover={leftover or '[]'}   orphans={orphans or '[]'}",
            flush=True,
        )

        if leftover:
            print(f"  WARNING: SHM leak after batch {batch_idx + 1}: {leftover}")
            cleanup_all_shm(seeds)  # try to recover
        if orphans:
            print(f"  WARNING: orphan NS-3 procs: {orphans}")

    total_dt = time.time() - total_t0

    # Final invariants
    final_leftover = all_my_shm()
    final_orphans = orphan_twt_pids()

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Total wall time     : {total_dt:.1f}s")
    print(f"  Total episodes      : {args.num_workers * args.num_batches}")
    print(f"  Successful episodes : {n_episodes_ok}")
    print(f"  Failed episodes     : {n_failed_total}")
    print(f"  Per-batch wall time : " + ", ".join(f"{t:.1f}s" for t in batch_times))
    print(f"  Final shm leftovers : {final_leftover or '[]'}")
    print(f"  Final NS-3 orphans  : {final_orphans or '[]'}")

    pass_episodes = n_failed_total == 0
    pass_shm = len(final_leftover) == 0
    pass_orphans = len(final_orphans) == 0

    # Stable batch time = no big slowdown (last batch <= 1.5x first batch)
    stable_time = len(batch_times) < 2 or batch_times[-1] <= 1.5 * batch_times[0]

    verdict = pass_episodes and pass_shm and pass_orphans and stable_time
    print("=" * 70)
    print(f"  VERDICT: {'PASS' if verdict else 'FAIL'}")
    print(f"    episodes_all_ok = {pass_episodes}")
    print(f"    no_shm_leftover = {pass_shm}")
    print(f"    no_ns3_orphans  = {pass_orphans}")
    print(f"    stable_batch_t  = {stable_time}")
    print("=" * 70)

    return 0 if verdict else 1


if __name__ == "__main__":
    mp.set_start_method(
        "spawn"
    )  # spawn, not fork: each worker starts a clean interpreter and imports the ns3-ai binding itself
    sys.exit(main())
