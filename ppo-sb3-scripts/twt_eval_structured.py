#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Structured, per-device evaluation: trained model vs the analytical baseline.

Runs TWO policies head-to-head over a STRUCTURED scenario grid that spans the
energy-stress edge cases, with MATCHED scenarios (both policies see the identical
NS-3 scenario per (split, seed) via a shared rand_seed — exact paired comparison),
and logs DETAILED per-(step, STA) metrics so the companion analyzer
(twt_eval_report.py) can compare model vs baseline by device class on many axes,
not just reward.

Structural grid (total STAs fixed at 20 — the locked topology):
  * REHD-split axis: n_rehd in {4, 8, 12, 16}  (REHD-light -> balanced -> REHD-heavy)
  * Load axis:       the built-in 5-segment over/under-saturation coin (labeled
                     per row via the exact C++ splitmix64 coin)
  * K seeds per split for statistics.

Per-(policy, split, seed, step, STA) row: device_class, is_rehd, action, and the
per-step deltas + levels needed for served-ratio / drop-rate / throughput / airtime /
latency / fairness and (REHD) harvested / consumed / SoC / unpowered-TX.

Usage:
  python3.11 twt_eval_structured.py \
      --model-ckpt <abs>/results/runs/<run>/pipeline/train/<ts>/ckpt_final.pt \
      --baseline analytical_demand \
      --reward-preset twt_pf_demand_v5 --seeds 10 --max-steps 100 \
      --num-workers 20 --out <abs>/runs/eval_struct_<ts>
"""

import argparse
import os
import pickle
import sys
import time
import multiprocessing as mp
from datetime import datetime

_here = os.path.dirname(os.path.abspath(__file__))
_proj = os.path.dirname(_here)

# Total STAs is fixed (locked topology); the split varies how many are REHD.
TOTAL_STA = 20
DEFAULT_SPLITS = [4, 8, 12, 16]  # n_rehd values; n_stations = TOTAL_STA - n_rehd
N_TRAFFIC_SEGMENTS = 5  # must match twt-constants.h
DURATION_IN_UPDATE = 100  # must match twt-constants.h
_SEG_UPDATES = DURATION_IN_UPDATE // N_TRAFFIC_SEGMENTS
_DEVICE_NAMES = {0: "IoT", 1: "Camera", 2: "Voice", 3: "Video", 4: "REHD"}

# --- exact C++ over/under coin (twt-simulation-config.cc::UpdateTrafficSegment) ---
_M64 = 0xFFFFFFFFFFFFFFFF


def _splitmix64(x):
    x &= _M64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _M64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def _is_over(rand_seed, seg_idx):
    h = (
        (int(rand_seed) * 0x9E3779B97F4A7C15) + ((seg_idx + 1) * 0xBF58476D1CE4E5B9)
    ) & _M64
    return (_splitmix64(h) & 1) == 0  # over when low bit == 0 (matches C++)


def _g(d, k):
    return float(d.get(k, 0) or 0)


def _extract(prev_env, curr_env, num_sta):
    """Per-STA dict of the deltas + levels needed for both the reward (v5 keys)
    and the per-device metrics. Cumulative counters -> per-step delta; levels as-is."""
    cl = curr_env.get("sta_observations", [])
    pl = prev_env.get("sta_observations", []) if prev_env else []
    out = []
    for i in range(num_sta):
        c = cl[i] if i < len(cl) else {}
        p = pl[i] if i < len(pl) else {}
        co, cr = c.get("oracle", {}), c.get("realistic", {})
        po = p.get("oracle", {}) if p else {}
        pr = p.get("realistic", {}) if p else {}
        d_qds = max(0.0, _g(co, "queue_delay_sum_ms") - _g(po, "queue_delay_sum_ms"))
        d_qdc = max(0.0, _g(co, "queue_delay_count") - _g(po, "queue_delay_count"))
        out.append(
            {
                # reward (v5) inputs:
                "delta_packets_transmitted": max(
                    0.0, _g(co, "packets_transmitted") - _g(po, "packets_transmitted")
                ),
                "delta_packets_enqueued": max(
                    0.0, _g(co, "packets_enqueued") - _g(po, "packets_enqueued")
                ),
                # --- policy-INVARIANT demand + true delivery (added for the camera-ready) ---
                # packets_enqueued is a MAC-ADMISSION counter: when the uplink is blocked (energy gate) upstream back-pressure drops packets before it, so it shrinks under exactly the policies that starve a STA -- measured 1.58x between two policies on identical matched scenarios.
                # packets_generated is the app-layer offered load and does not move with the schedule.
                # packets_transmitted counts PHY TX STARTS (retries and control frames included, hence tx > enq); packets_received_at_ap counts frames the AP actually decoded.
                # (delta_generated, delta_rx_at_ap) is therefore the honest (demand, delivered) pair; the two legacy fields are kept so the reward and every published number stay reproducible.
                "delta_packets_generated": max(
                    0.0, _g(co, "packets_generated") - _g(po, "packets_generated")
                ),
                "delta_packets_rx_at_ap": max(
                    0.0,
                    _g(cr, "packets_received_at_ap") - _g(pr, "packets_received_at_ap"),
                ),
                "delta_drops_expired": max(
                    0.0, _g(co, "mpdu_drops_expired") - _g(po, "mpdu_drops_expired")
                ),
                "vcap_max_v": _g(co, "vcap_max_v"),
                "delta_harvested_j": max(
                    0.0, _g(co, "harvested_total_j") - _g(po, "harvested_total_j")
                ),
                "delta_consumed_j": max(
                    0.0, _g(co, "consumed_total_j") - _g(po, "consumed_total_j")
                ),
                "delta_airtime_used_us": max(
                    0.0, _g(cr, "airtime_used_us") - _g(pr, "airtime_used_us")
                ),
                # extra metric inputs:
                "delta_bytes_transmitted": max(
                    0.0, _g(co, "bytes_transmitted") - _g(po, "bytes_transmitted")
                ),
                "lat_sum_ms": d_qds,
                "lat_count": d_qdc,
                # levels:
                "device_class": int(co.get("device_class", cr.get("device_class", -1))),
                "vcap_v": _g(co, "vcap_v"),
                "bsr_be": _g(cr, "bsr_queue_ac_be"),
                "unpowered_tx": _g(co, "unpowered_tx_events"),
                "queue_bytes": _g(co, "queue_size_bytes"),
            }
        )
    return out


def eval_worker(task, out_dir, result_q):
    """One (policy, split, scenario) episode -> one parquet shard."""
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    sys.path.insert(0, _proj)
    sys.path.insert(0, os.path.join(_proj, "exploration-scripts"))
    sys.path.insert(0, _here)
    ns3_root = os.path.abspath(os.path.join(_here, "..", "..", "..", ".."))
    os.chdir(ns3_root)

    shm_seed = task["shm_seed"]
    status = {
        "tag": task["tag"],
        "shm_seed": shm_seed,
        "ok": False,
        "rows": 0,
        "err": None,
    }
    wrapper = None
    try:
        import numpy as np
        import pandas as pd
        from pb_twt_wrapper_py import TWTWrapper
        from twt_models import get_policy
        from twt_spawn_worker import (
            build_action_dict,
            build_ppo_observation,
            build_critic_features,
            load_action_tables,
            NUM_PDW_LEVELS,
        )
        from reward_functions import get_reward_function

        # --- policy ---
        if task["kind"] == "ckpt":
            payload = pickle.loads(task["payload"])
            policy = get_policy(payload["name"], **payload.get("kwargs", {}))
            policy.load_state_dict(payload["state_dict"])
        else:
            policy = get_policy(task["reg_name"])
        policy.eval()
        recurrent_state = policy.initial_recurrent_state()
        is_asym = bool(getattr(policy, "is_asymmetric_critic", False))

        st, at = load_action_tables(_proj)
        n_sched, n_assign = len(st["schedules"]), len(at["assignments"])
        reward_fn = get_reward_function(task["reward_preset"])

        ns, nr = task["n_stations"], task["n_rehd"]
        rand_seed = task["rand_seed"]
        warmup, max_steps = task["warmup"], task["max_steps"]

        wrapper = TWTWrapper(
            log_dir=os.path.join(_proj, "results", "data-log"),
            enable_logging=False,
            verbose=False,
            quiet_ns3=True,
            disable_ns3_traces=True,
            n_stations=ns,
            n_rehd=nr,
        )
        env = wrapper.reset(seed=shm_seed, rand_seed=rand_seed)
        num_sta = env.get("num_sta", ns + nr)

        warm = build_action_dict(0, 0, st, at, num_sta)
        prev = env
        for _ in range(warmup):
            prev = env
            env, done = wrapper.step(warm)
            if done:
                raise RuntimeError("episode ended during warmup")

        rows = []
        for step_idx in range(max_steps):
            obs = build_ppo_observation(prev, env)
            critic_obs = build_critic_features(prev, env, num_sta) if is_asym else None
            if is_asym:
                s, a, p, _, _, recurrent_state = policy.act(
                    obs, recurrent_state, deterministic=True, critic_obs=critic_obs
                )
            else:
                s, a, p, _, _, recurrent_state = policy.act(
                    obs, recurrent_state, deterministic=True
                )
            s = max(0, min(int(s), n_sched - 1))
            a = max(0, min(int(a), n_assign - 1))
            p = max(0, min(int(p), NUM_PDW_LEVELS - 1))
            act = build_action_dict(s, a, st, at, num_sta, pdw_idx=p)

            nxt, done = wrapper.step(act)
            if not nxt:
                break
            # effect of THIS action: env -> nxt deltas
            dl = _extract(env, nxt, num_sta)
            rw = float(reward_fn(dl, action=(s, a)).get("total", 0.0))
            seg = min(N_TRAFFIC_SEGMENTS - 1, (warmup + step_idx) // _SEG_UPDATES)
            over = _is_over(rand_seed, seg)

            for i in range(num_sta):
                d = dl[i]
                vmax = d["vcap_max_v"]
                rows.append(
                    {
                        "policy": task["policy_label"],
                        "n_rehd": nr,
                        "n_stations": ns,
                        "rand_seed": rand_seed,
                        "step": step_idx,
                        "segment": seg,
                        "over": bool(over),
                        "sta_id": i,
                        "device_class": d["device_class"],
                        "is_rehd": vmax > 0,
                        "sched": s,
                        "assign": a,
                        "pdw_idx": p,
                        "reward": rw,
                        "d_tx": d["delta_packets_transmitted"],
                        "d_eq": d["delta_packets_enqueued"],
                        "d_gen": d["delta_packets_generated"],
                        "d_rx_ap": d["delta_packets_rx_at_ap"],
                        "d_drops": d["delta_drops_expired"],
                        "d_bytes": d["delta_bytes_transmitted"],
                        "d_airtime_us": d["delta_airtime_used_us"],
                        "lat_sum_ms": d["lat_sum_ms"],
                        "lat_count": d["lat_count"],
                        "d_harv_j": d["delta_harvested_j"],
                        "d_cons_j": d["delta_consumed_j"],
                        "soc": (d["vcap_v"] / vmax if vmax > 0 else 0.0),
                        "bsr_be": d["bsr_be"],
                        "unpowered_tx": d["unpowered_tx"],
                        "queue_bytes": d["queue_bytes"],
                    }
                )
            prev = env
            env = nxt
            if done:
                break

        df = pd.DataFrame(rows)
        shard = os.path.join(out_dir, f"{task['tag']}.parquet")
        tmp = shard + ".tmp"
        df.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
        os.replace(tmp, shard)
        status.update(ok=True, rows=len(df))
    except Exception as e:
        import traceback

        status["err"] = f"{e}\n{traceback.format_exc()}"[:1500]
    finally:
        try:
            result_q.put(status)  # before close, so the feeder thread flushes it
        except Exception:
            pass
        try:
            if wrapper is not None:
                wrapper.close()
        except Exception:
            pass
        for nm in (
            f"/dev/shm/MySeg_{shm_seed}",
            f"/dev/shm/MyCpp2PyMsg_{shm_seed}",
            f"/dev/shm/MyPy2CppMsg_{shm_seed}",
            f"/dev/shm/MyLockable_{shm_seed}",
        ):
            try:
                os.remove(nm)
            except OSError:
                pass
        try:
            result_q.cancel_join_thread()
        except Exception:
            pass
        os._exit(0)


def _build_model_payload(ckpt_path):
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return pickle.dumps(
        {
            "name": ck["name"],
            "kwargs": ck.get("kwargs", {}),
            "state_dict": ck["state_dict"],
        }
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-ckpt", required=True, help="trained checkpoint (.pt)")
    ap.add_argument(
        "--baseline", default="analytical_demand", help="registry policy name"
    )
    ap.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=DEFAULT_SPLITS,
        help="n_rehd values to test (n_stations = 20 - n_rehd)",
    )
    ap.add_argument("--seeds", type=int, default=10, help="scenarios per split")
    ap.add_argument("--reward-preset", default="twt_pf_demand_v5")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=20)
    ap.add_argument("--base-scenario-seed", type=int, default=300000)
    ap.add_argument("--base-shm-seed", type=int, default=700000)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run twt_eval_report on the results when collection finishes (default on)",
    )
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Absolute: workers chdir to the ns-3 root, so a relative --out would break the save.
    out_dir = (
        os.path.abspath(args.out)
        if args.out
        else os.path.join(_here, "runs", f"eval_struct_{ts}")
    )
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    model_payload = _build_model_payload(args.model_ckpt)
    model_label = "model"
    base_label = args.baseline

    # Build the task list.
    # Matched scenarios: model & baseline share rand_seed per (split, k); each task gets a UNIQUE shm_seed (no SHM collision across workers).
    tasks, shm = [], args.base_shm_seed
    for si, n_rehd in enumerate(args.splits):
        n_sta = TOTAL_STA - n_rehd
        for k in range(args.seeds):
            rand_seed = args.base_scenario_seed + si * 1000 + k
            for pol in ("model", "baseline"):
                shm += 1
                common = dict(
                    n_stations=n_sta,
                    n_rehd=n_rehd,
                    rand_seed=rand_seed,
                    reward_preset=args.reward_preset,
                    max_steps=args.max_steps,
                    warmup=args.warmup_steps,
                    shm_seed=shm,
                )
                if pol == "model":
                    tasks.append(
                        dict(
                            kind="ckpt",
                            payload=model_payload,
                            policy_label=model_label,
                            tag=f"model_r{n_rehd}_s{k}",
                            **common,
                        )
                    )
                else:
                    tasks.append(
                        dict(
                            kind="registry",
                            reg_name=base_label,
                            policy_label=base_label,
                            tag=f"{base_label}_r{n_rehd}_s{k}",
                            **common,
                        )
                    )

    # resume: skip tasks whose shard already exists
    tasks = [
        t
        for t in tasks
        if not os.path.exists(os.path.join(shard_dir, f"{t['tag']}.parquet"))
    ]
    total = len(tasks)
    print(f"[eval] out={out_dir}")
    print(f"[eval] model={os.path.relpath(args.model_ckpt)}  baseline={base_label}")
    print(
        f"[eval] splits(n_rehd)={args.splits}  seeds/split={args.seeds}  "
        f"reward={args.reward_preset}  steps={args.max_steps}"
    )
    print(f"[eval] {total} episodes to run ({args.num_workers} workers)")

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    done = 0
    t0 = time.time()
    i = 0
    while i < total or done < total:
        # keep the worker pool filled
        running = []
        while i < total and len(running) < args.num_workers:
            p = ctx.Process(target=eval_worker, args=(tasks[i], shard_dir, result_q))
            p.start()
            running.append(p)
            i += 1
        # drain a wave
        for _ in range(len(running)):
            st = result_q.get()
            done += 1
            tag = st.get("tag", "?")
            if st["ok"]:
                print(f"  [{done}/{total}] {tag}: {st['rows']} rows")
            else:
                print(f"  [{done}/{total}] {tag}: FAIL {str(st['err'])[:200]}")
        for p in running:
            p.join()
    print(f"[eval] DONE: {done} episodes, {time.time()-t0:.1f}s -> {shard_dir}")
    if args.report:
        from twt_eval_report import generate

        generate(out_dir)
    else:
        print(f"[eval] analyze: python3.11 twt_eval_report.py {out_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
