#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
twt_smoke_one_worker.py - Smoke test: spawn ONE TWT worker, verify a rollout returns.

This mirrors how wifi-simulation/batch_orchestrator.py spawns workers, but with
N=1 and a randomly-initialized policy. Purpose: verify twt_spawn_worker.run_episode
runs end-to-end before we build the full orchestrator.

Pass criteria:
  - Worker process exits.
  - A success=True result with non-empty rollout reaches the parent queue.
  - No /dev/shm/My* leftovers afterward.
  - No orphan twt-powercast-main-simulation processes.

Run from ns-3.44 root:
  cd /home/ahmak/NS3-project/ns-allinone-3.44/ns-3.44
  /home/ahmak/NS3-project/EHRL/bin/python3.11 \
      contrib/ai/examples/rl-twt-powercast/ppo-sb3-scripts/twt_smoke_one_worker.py
"""

import os
import warnings

# Silence TF/XLA chatter and benign torch UserWarnings; inherited by spawn children via os.environ + re-execution of the top-level block.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import time
import glob
import pickle
import multiprocessing as mp


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stations", type=int, default=12, help="non-REHD STA count")
    ap.add_argument("--n-rehd", type=int, default=4, help="REHD count")
    ap.add_argument("--reward-preset", type=str, default="twt_pf_demand_v1")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    ns3_root = os.path.abspath(os.path.join(parent_dir, "..", "..", "..", ".."))

    sys.path.insert(0, script_dir)
    sys.path.insert(0, parent_dir)

    # Build a fresh, untrained policy via the registry.
    # The payload is self-describing: workers reconstruct the same architecture from `name`+`kwargs`.
    import torch  # noqa
    from twt_models import get_policy, list_policies
    from twt_spawn_worker import run_episode, default_obs_kwargs

    print(f"[parent] Registered policies: {list_policies()}")

    # obs_dim derives from the topology (num_sta·F); auto-size so any n_stations/n_rehd combo works without manual obs_dim juggling.
    n_total = int(args.n_stations) + int(args.n_rehd)
    policy_name = "mlp_ppo"
    policy_kwargs = {
        **default_obs_kwargs(
            policy_name, n_total
        ),  # auto-sizes heads from the carved tables
        "hidden_dim": 256,
    }
    print(
        f"[parent] topology: {args.n_stations} STA + {args.n_rehd} REHD = {n_total} "
        f"-> policy kwargs={policy_kwargs}"
    )
    policy = get_policy(policy_name, **policy_kwargs)
    payload = {
        "name": policy_name,
        "kwargs": policy_kwargs,
        "state_dict": policy.state_dict(),
    }
    policy_bytes = pickle.dumps(payload)
    print(f"[parent] Pickled payload ({policy_name}): {len(policy_bytes)} bytes")

    # Pre-clean any stale shm
    seed = 80001
    for nm in (
        f"/dev/shm/MySeg_{seed}",
        f"/dev/shm/MyCpp2PyMsg_{seed}",
        f"/dev/shm/MyPy2CppMsg_{seed}",
        f"/dev/shm/MyLockable_{seed}",
    ):
        if os.path.exists(nm):
            try:
                os.remove(nm)
            except OSError:
                pass

    # Spawn ONE worker — same shape as wifi-simulation/batch_orchestrator.py
    q = mp.Queue()
    print(f"[parent] Spawning worker with seed={seed}, ns3_path={ns3_root}")
    p = mp.Process(
        target=run_episode,
        args=(0, seed, policy_bytes, q, ns3_root),
        kwargs={
            "reward_type": args.reward_preset,
            "warmup_steps": 2,
            "max_steps": 10,  # 10 real steps gives a clean step-time average
            "quiet_ns3": True,  # silent NS-3 by default; flip for debug
            "disable_ns3_traces": True,
            "n_stations": args.n_stations,
            "n_rehd": args.n_rehd,
        },
    )
    t0 = time.time()
    p.start()

    # Drain queue BEFORE join (wifi-simulation lesson #3)
    print(f"[parent] Waiting for rollout (timeout 300s)...")
    try:
        result = q.get(timeout=300)
    except Exception as e:
        print(f"[parent] FAIL: queue.get timed out / errored: {e}")
        p.terminate()
        p.join(timeout=10)
        sys.exit(1)
    finally:
        try:
            q.close()
        except Exception:
            pass

    p.join(timeout=30)
    if p.is_alive():
        print(f"[parent] Worker did not exit after join, terminating")
        p.terminate()
        p.join(timeout=5)

    elapsed = time.time() - t0
    print(f"[parent] Worker done in {elapsed:.1f}s (parent wall-clock)")
    print(f"[parent] success     : {result.get('success')}")
    print(f"[parent] worker_id   : {result.get('worker_id')}")
    print(f"[parent] seed        : {result.get('seed')}")
    print(f"[parent] ep_length   : {result.get('episode_length')}")
    print(f"[parent] ep_reward   : {result.get('episode_reward'):.3f}")
    if not result.get("success"):
        print(f"[parent] error      :\n{result.get('error', '<no error msg>')[:1500]}")

    # --- Phase-by-phase timing breakdown ---
    timing = result.get("timing", {}) or {}
    if timing:
        warmup_steps = timing.get("warmup_step_s", []) or []
        real_steps = timing.get("real_step_s", []) or []
        sum_warm = sum(warmup_steps)
        sum_real = sum(real_steps)
        avg_warm = sum_warm / len(warmup_steps) if warmup_steps else 0.0
        avg_real = sum_real / len(real_steps) if real_steps else 0.0
        pre_close = timing.get("total_pre_close_s", 0.0)
        accounted = (
            timing.get("imports_s", 0.0)
            + timing.get("policy_load_s", 0.0)
            + timing.get("wrapper_init_s", 0.0)
            + timing.get("boot_s", 0.0)
            + sum_warm
            + sum_real
            + timing.get("bootstrap_s", 0.0)
        )
        unaccounted = pre_close - accounted
        print()
        print(f"[parent] === phase timings (worker side) ===")
        print(f"[parent]   imports        : {timing.get('imports_s', 0):6.2f}s")
        print(f"[parent]   policy load    : {timing.get('policy_load_s', 0):6.2f}s")
        print(f"[parent]   wrapper __init__: {timing.get('wrapper_init_s', 0):6.2f}s")
        print(
            f"[parent]   boot (reset)   : {timing.get('boot_s', 0):6.2f}s   "
            f"<-- NS-3 startup until first env arrives"
        )
        print(
            f"[parent]   warmup steps   : {sum_warm:6.2f}s "
            f"({len(warmup_steps)} steps, avg {avg_warm:.2f}s/step)"
        )
        print(
            f"[parent]   real steps     : {sum_real:6.2f}s "
            f"({len(real_steps)} steps, avg {avg_real:.2f}s/step)"
        )
        print(f"[parent]   bootstrap pass : {timing.get('bootstrap_s', 0):6.2f}s")
        print(
            f"[parent]   unaccounted    : {unaccounted:6.2f}s   (rollout assembly + queue.put)"
        )
        print(f"[parent]   --- pre-close  : {pre_close:6.2f}s")
        print(
            f"[parent]   close + os._exit (parent-side): "
            f"{elapsed - pre_close:6.2f}s "
            f"(~kill + cleanup + 1s sleep in TWTWrapper.close)"
        )

    # Validate post-conditions
    leftover_shm = sorted(glob.glob("/dev/shm/My*"))
    print(f"[parent] /dev/shm leftovers: {leftover_shm}")

    import subprocess

    orphan = subprocess.run(
        ["pgrep", "-f", "twt-powercast-main-simulation"], capture_output=True, text=True
    )
    print(f"[parent] orphan twt-main pids: {orphan.stdout.strip() or '(none)'}")

    ok = (
        result.get("success") is True
        and result.get("episode_length", 0) > 0
        and not leftover_shm
        and not orphan.stdout.strip()
    )
    print(f"[parent] VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    mp.set_start_method("spawn")  # wifi-simulation lesson #4
    sys.exit(main())
