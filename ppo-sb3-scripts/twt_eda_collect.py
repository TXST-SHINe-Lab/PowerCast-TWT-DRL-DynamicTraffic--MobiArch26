#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""EDA data collector for the QoEH TWT pipeline.

Drives the NS-3 wrapper DIRECTLY (no policy) through a SYSTEMATIC sweep of the
40x40x9 THREE-HEAD action space (schedule x assignment x PDW level) and logs
EVERYTHING that feeds the model (the 18 obs features) and the reward (raw
realistic+oracle metrics, per-step deltas, per-STA reward inputs, and the full
reward breakdown) — one tidy row per (episode, update, STA) — to parquet shards
for distribution analysis.

Systematic coverage: for global episode index g = round*num_workers + worker_id
and update u, the action index is (g*UPDATES + u) mod 24000, decoded D-fastest
(pdw = idx%15; assign = (idx//15)%40; sched = (idx//600)%40) so the PDW dimension
is densely sampled and Δharvested responds per-update. Tiles the 24000-combo grid
across the run (more episodes = denser coverage; ~3 visits/combo at 1000 eps).

Run from the NS-3.44 root, EHRL venv. Spawn pattern mirrors the orchestrator
(spawn workers per round, drain queue before join, workers os._exit(0)).

  python3.11 contrib/ai/examples/rl-twt-powercast/ppo-sb3-scripts/twt_eda_collect.py \
      --num-batches 100 --num-workers 6 --n-stations 12 --n-rehd 8
"""

import os
import sys
import json
import time
import argparse
import datetime
import glob
import queue as _queue
import multiprocessing as mp

_here = os.path.dirname(os.path.abspath(__file__))
_proj = os.path.dirname(_here)


# Grid dimensions are DERIVED from the shipped action tables, never hardcoded.
# They used to be pinned at N_SCHED=30 (the pre-carve grid).
# Once the tables were carved to 24 schedules, `sched = (global_idx + u) % 30` kept emitting indices 24-29, which index past the end of a 24-entry table -- every episode drawing one died with an IndexError.
# A short smoke never reaches those indices, so the failure only shows up at scale: 4 episodes reported "0 fails" while 2000 episodes failed immediately.
def _grid_dims():
    import json as _json
    import os as _os

    _exp = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "exploration-scripts",
    )
    _ns = len(_json.load(open(_os.path.join(_exp, "schedule_table.json")))["schedules"])
    _na = len(
        _json.load(open(_os.path.join(_exp, "assignment_table.json")))["assignments"]
    )
    return _ns, _na


N_SCHED, N_ASSIGN = _grid_dims()
N_PDW = 10  # PDW levels used in training: pdw_end ∈ {5,...,50} ms => T_pdw ∈ {0,...,45} (idx0=no PDW)
GRID = N_SCHED * N_ASSIGN * N_PDW
UPDATES = (
    100  # DURATION_IN_UPDATE (must match twt-constants.h; bumped 75->100 2026-06-01)
)

# SPAN mode (2026-06-07): for a SHORT EDA that still covers the whole action space, cycle each action dimension with a coprime stride so EVERY episode marginally sweeps all sched (period 30), all assign (stride 13, coprime to 25 -> all 25), all pdw (period 10), with a per-episode offset so workers decorrelate.
# (Sequential tiling only spans sched after ~240 eps.)
EDA_SPAN = os.environ.get("EDA_SPAN", "0") == "1"
EDA_SCALE = float(
    os.environ.get("EDA_SCALE", "0") or 0
)  # >0 overrides C++ trafficScale default

OBS_NAMES = ["bsr_be", "dpkts_rx", "dairtime", "dfcs", "snr", "silence", "starv_rate"]
# 7-feat FINAL time-local feed (2026-06-07 signal rework).
# Order MUST match twt_spawn_worker._build_per_sta_features / obs_warmstart_stats.json.
# (Was the stale 18-name list — it index-overran the (num_sta,7) obs and failed every worker; the analyzer rebuilds features from the raw real_/orc_ columns, not these obs_ cols.)


def _clean_shm(seed):
    for nm in (
        f"/dev/shm/MySeg_{seed}",
        f"/dev/shm/MyCpp2PyMsg_{seed}",
        f"/dev/shm/MyPy2CppMsg_{seed}",
        f"/dev/shm/MyLockable_{seed}",
    ):
        try:
            os.remove(nm)
        except OSError:
            pass


def _deltas(prev, curr):
    """Per-STA reward inputs — mirrors twt_spawn_worker.run_episode's nested
    compute_sta_deltas (the keys v1 reads). Also computes vcap/harvested/consumed
    deltas — NOT used by v1, kept only as energy-analysis columns in the dataset."""
    cl = curr.get("sta_observations", [])
    pl = prev.get("sta_observations", []) if prev else []
    g = lambda d, k: float(d.get(k, 0) or 0)
    out = []
    for i, c in enumerate(cl):
        p = pl[i] if i < len(pl) else {}
        co, cr = c.get("oracle", {}), c.get("realistic", {})
        po = p.get("oracle", {}) if p else {}
        out.append(
            {
                "delta_packets_transmitted": max(
                    0.0, g(co, "packets_transmitted") - g(po, "packets_transmitted")
                ),
                "delta_packets_enqueued": max(
                    0.0, g(co, "packets_enqueued") - g(po, "packets_enqueued")
                ),
                "delta_drops_expired": max(
                    0.0, g(co, "mpdu_drops_expired") - g(po, "mpdu_drops_expired")
                ),
                "delta_bytes_transmitted": max(
                    0.0, g(co, "bytes_transmitted") - g(po, "bytes_transmitted")
                ),
                "duty_cycle": g(co, "duty_cycle"),
                "bsr_queue_index": g(cr, "bsr_queue_ac_be"),
                "vcap_v": g(co, "vcap_v"),
                "vcap_max_v": g(co, "vcap_max_v"),
                "delta_harvested_j": max(
                    0.0, g(co, "harvested_total_j") - g(po, "harvested_total_j")
                ),
                "delta_consumed_j": max(
                    0.0, g(co, "consumed_total_j") - g(po, "consumed_total_j")
                ),
            }
        )
    return out


def worker(
    global_idx,
    seed,
    out_dir,
    n_stations,
    n_rehd,
    reward_preset,
    max_updates,
    result_queue,
):
    # Quiet NS-3 / ninja chatter (inherited by the spawned child).
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

    wrapper = None
    status = {
        "global_idx": global_idx,
        "seed": seed,
        "ok": False,
        "rows": 0,
        "err": None,
    }
    try:
        import numpy as np
        import pandas as pd
        from pb_twt_wrapper_py import TWTWrapper
        from twt_spawn_worker import (
            build_action_dict,
            load_action_tables,
            _build_per_sta_features,
        )
        from reward_functions import compute_twt_pf_demand_v1_reward

        sched_table, assign_table = load_action_tables(_proj)
        scheds = sched_table["schedules"]
        real_fields = list(TWTWrapper.REALISTIC_FIELDS)
        orac_fields = list(TWTWrapper.ORACLE_FIELDS)

        _wkw = dict(
            enable_logging=False,
            verbose=False,
            quiet_ns3=True,
            disable_ns3_traces=True,
            n_stations=n_stations,
            n_rehd=n_rehd,
        )
        if EDA_SCALE > 0:
            _wkw["traffic_scale"] = EDA_SCALE
        wrapper = TWTWrapper(**_wkw)
        env = wrapper.reset(seed=seed)
        num_sta = env.get("num_sta", n_stations + n_rehd)

        rows = []
        prev = env
        for u in range(min(UPDATES, max_updates)):
            if EDA_SPAN:
                # coprime-cycle: each episode marginally sweeps ALL of each dimension
                sched = (global_idx + u) % N_SCHED
                assign = (global_idx * 7 + u * 13) % N_ASSIGN  # 13 coprime to 40
                pdw_idx = (global_idx * 3 + u) % N_PDW
            else:
                idx = (global_idx * UPDATES + u) % GRID
                # 3-head decode (D fastest so the PDW dimension is densely sampled and Δharvested responds per-update; assign next; sched slowest).
                pdw_idx = idx % N_PDW
                rem = idx // N_PDW
                assign = rem % N_ASSIGN
                sched = (rem // N_ASSIGN) % N_SCHED
            act = build_action_dict(
                sched, assign, sched_table, assign_table, num_sta, pdw_idx=pdw_idx
            )
            # per-STA group + wake duration from the action
            grp_wake = {
                gc["group_id"]: gc["twt_wake_duration_ms"]
                for gc in act["twt_group_configs"]
            }
            sta_grp = {
                a["sta_id"]: a["assigned_twt_group"]
                for a in act["sta_group_assignments"]
            }
            num_groups = scheds[sched]["num_groups"]
            nominal_duty = scheds[sched].get("total_duration_ms", 0) / 102.4

            curr, done = wrapper.step(act)
            obs = _build_per_sta_features(prev, curr, num_sta)  # (num_sta, 7)
            dl = _deltas(prev, curr)
            rw = compute_twt_pf_demand_v1_reward(dl, action=(sched, assign))
            comp = rw["components"]
            met = rw["metrics"]
            sim_t = float(curr.get("simulation_time_sec", 0.0))

            sta_list = curr.get("sta_observations", [])
            for i in range(num_sta):
                if i >= len(sta_list):
                    break
                s = sta_list[i]
                r = s.get("realistic", {})
                o = s.get("oracle", {})
                d = dl[i]
                sid = r.get("sta_id", i)
                vmax = float(o.get("vcap_max_v", 0) or 0)
                is_rehd = vmax > 0
                pkts_eq = d["delta_packets_enqueued"]
                pkts_tx = d["delta_packets_transmitted"]
                row = {
                    # context
                    "seed": seed,
                    "global_idx": global_idx,
                    "update_idx": u,
                    "sim_time_s": sim_t,
                    "sta_id": int(sid),
                    "device_class": int(r.get("device_class", -1)),
                    "is_rehd": bool(is_rehd),
                    # action
                    "sched": sched,
                    "assign": assign,
                    "num_groups": int(num_groups),
                    "pdw_idx": int(pdw_idx),
                    "pdw_ms": float(act["pdw_duration_ms"]),
                    "assigned_group": int(sta_grp.get(sid, 0)),
                    "wake_duration_ms": float(
                        grp_wake.get(sta_grp.get(sid, 0), 0)
                    ),  # decoder-scaled
                    "nominal_duty": float(
                        nominal_duty
                    ),  # RAW schedule duty (pre-D-scaling)
                    # per-STA reward inputs (derived)
                    "served_ratio": (
                        min(1.0, pkts_tx / pkts_eq) if pkts_eq > 0 else 1.0
                    ),
                    "drop_rate": (
                        min(1.0, d["delta_drops_expired"] / pkts_eq)
                        if pkts_eq > 0
                        else 0.0
                    ),
                    "bsr_fill": min(1.0, d["bsr_queue_index"] / 254.0),
                    "soc": (float(o.get("vcap_v", 0) or 0) / vmax if vmax > 0 else 0.0),
                    # reward output (per-update, denormalized onto each STA row)
                    "reward_total": rw["total"],
                    "rwd_w_pf_thr": comp.get("w_pf_thr", 0),
                    "rwd_w_pf_drop": comp.get("w_pf_drop", 0),
                    "rwd_w_pf_bsr": comp.get("w_pf_bsr", 0),
                    "rwd_w_max_starv": comp.get("w_max_starv", 0),
                    "rwd_w_energy": comp.get("w_energy", 0),
                    "rwd_w_airtime": comp.get("w_airtime", 0),
                    "rwd_airtime_frac": met.get("airtime_frac", 0),
                }
                # model input: 7 obs features (order matches _build_per_sta_features)
                for j, nm in enumerate(OBS_NAMES):
                    row[f"obs_{nm}"] = float(obs[i, j])
                # per-step deltas
                for k, v in d.items():
                    row[f"d_{k}"] = float(v)
                # raw realistic + oracle
                for f in real_fields:
                    if f != "sta_id":
                        row[f"real_{f}"] = float(r.get(f, 0) or 0)
                for f in orac_fields:
                    if f != "sta_id":
                        row[f"orc_{f}"] = float(o.get(f, 0) or 0)
                rows.append(row)
            prev = curr
            if done:
                break

        df = pd.DataFrame(rows)
        shard = os.path.join(out_dir, f"ep_{global_idx:04d}_seed{seed}.parquet")
        tmp = shard + ".tmp"
        df.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
        os.replace(
            tmp, shard
        )  # atomic: a terminated worker never leaves a partial shard
        status.update(ok=True, rows=len(df), shard=os.path.basename(shard))
    except Exception as e:
        import traceback

        status["err"] = f"{e}\n{traceback.format_exc()}"[:1500]
    finally:
        try:
            result_queue.put(status)  # BEFORE close (rule #18)
        except Exception:
            pass
        try:
            if wrapper is not None:
                wrapper.close()
        except Exception:
            pass
        for nm in (
            f"/dev/shm/MySeg_{seed}",
            f"/dev/shm/MyCpp2PyMsg_{seed}",
            f"/dev/shm/MyPy2CppMsg_{seed}",
            f"/dev/shm/MyLockable_{seed}",
        ):
            try:
                os.remove(nm)
            except OSError:
                pass
        try:
            result_queue.cancel_join_thread()
        except Exception:
            pass
        os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-batches", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument(
        "--pool-size",
        type=int,
        default=0,
        help="rolling-pool concurrent slots (0 = --num-workers). DGX big.LITTLE: ~20-24.",
    )
    ap.add_argument(
        "--wall-deadline-epoch",
        type=float,
        default=0.0,
        help="absolute epoch-seconds wall-clock cap; stop dispatching new episodes past "
        "it, drain in-flight, exit (0 = no cap). Resume-safe across restarts.",
    )
    ap.add_argument("--n-stations", type=int, default=12)
    ap.add_argument("--n-rehd", type=int, default=8)
    ap.add_argument(
        "--variable-split",
        action="store_true",
        default=False,
        help="per-episode REHD split ~ U{split_lo..split_hi}, total 20 (matches training)",
    )
    ap.add_argument("--split-lo", type=int, default=4)
    ap.add_argument("--split-hi", type=int, default=16)
    ap.add_argument("--base-seed", type=int, default=50000)
    ap.add_argument("--reward-preset", type=str, default="twt_pf_demand_v1")
    ap.add_argument(
        "--max-updates",
        type=int,
        default=UPDATES,
        help="cap updates/episode (default 75 = full; small for a fast dry-run)",
    )
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(_here, "runs", f"eda_{ts}")
    shard_dir = os.path.join(
        out_dir, "shards"
    )  # parquet shards isolated from manifest.json
    os.makedirs(shard_dir, exist_ok=True)
    n_episodes = args.num_batches * args.num_workers
    upd = min(UPDATES, args.max_updates)
    GET_TIMEOUT = (
        900  # per-result wait; a full 75-step episode is ~3-6 min, so >15 min = hung
    )
    print(f"[eda] out={out_dir}")
    print(
        f"[eda] {n_episodes} episodes, "
        f"{'variable 4-16 REHD' if args.variable_split else f'{args.n_stations}STA+{args.n_rehd}REHD'}, "
        f"30x25x10 grid, {upd} updates/ep"
    )
    print(
        f"[eda] grid coverage ~= {n_episodes * upd / GRID:.1f} visits/combo  "
        f"(resumable: existing shards in {shard_dir} are skipped)"
    )

    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "-C", _proj, "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        sha = "unknown"

    if args.variable_split:
        from twt_scenario import sample_split
    from twt_pool import run_rolling_pool

    # Build the work-item list (one per episode), skipping already-collected shards (resume).
    # Each item's REHD split is deterministic in its seed -> resume-safe + CRN-safe.
    # The ROLLING POOL then keeps `pool_size` slots busy and dispatches the next item to whichever core frees first -> no per-batch barrier stalling on the DGX's slow A725 cores.
    items = []
    skipped = 0
    for g in range(n_episodes):
        seed = args.base_seed + g
        shard = os.path.join(shard_dir, f"ep_{g:04d}_seed{seed}.parquet")
        if os.path.exists(shard):
            skipped += 1
            continue  # resume: already collected
        ep_ns, ep_nr = (
            sample_split(seed, total=20, lo=args.split_lo, hi=args.split_hi)
            if args.variable_split
            else (args.n_stations, args.n_rehd)
        )
        items.append(
            {
                "global_idx": g,
                "seed": seed,
                "out_dir": shard_dir,
                "n_stations": ep_ns,
                "n_rehd": ep_nr,
                "reward_preset": args.reward_preset,
                "max_updates": args.max_updates,
            }
        )
    for it in items:
        _clean_shm(it["seed"])  # clear any stale segment before dispatch

    M = args.pool_size if args.pool_size else args.num_workers
    t0 = time.time()
    fails = []
    print(
        f"[eda] ROLLING POOL: {len(items)} episodes to run ({skipped} already done), "
        f"pool_size={M} on a 20-core big.LITTLE box, dispatch-on-completion",
        flush=True,
    )

    def _on_result(r, done, total):
        if not r.get("ok"):
            fails.append(
                (r.get("global_idx"), r.get("seed"), (r.get("err") or "")[:200])
            )
        if done % M == 0 or done == total:
            n_shards = len(glob.glob(os.path.join(shard_dir, "*.parquet")))
            elapsed = time.time() - t0
            eta = (elapsed / max(1, done)) * (total - done)
            print(
                f"[eda] {done}/{total} done  shards={n_shards}  skipped={skipped}  "
                f"fails={len(fails)}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                flush=True,
            )

    results = run_rolling_pool(
        items,
        worker,
        pool_size=M,
        result_timeout=GET_TIMEOUT,
        on_result=_on_result,
        deadline=(args.wall_deadline_epoch or None),
    )
    got = {r.get("global_idx") for r in results}
    # Only flag "no result + no shard" as a FAIL on a FULL run.
    # If the pool was truncated (wall-deadline cap or a hung slot), len(results) < len(items) and the un-dispatched episodes are simply NOT-RUN-YET (picked up on the next resume), not failures.
    completed = len(results) >= len(items)
    for it in items:
        if (
            completed
            and it["global_idx"] not in got
            and not os.path.exists(
                os.path.join(
                    shard_dir, f"ep_{it['global_idx']:04d}_seed{it['seed']}.parquet"
                )
            )
        ):
            fails.append((it["global_idx"], it["seed"], "no result + no shard"))
        _clean_shm(it["seed"])

    # accurate total rows from shard metadata (cheap; no full load)
    import pyarrow.parquet as pq

    shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    total_rows = 0
    for f in shard_files:
        try:
            total_rows += pq.ParquetFile(f).metadata.num_rows
        except Exception:
            pass

    manifest = {
        "timestamp": ts,
        "git_sha": sha,
        "n_episodes": n_episodes,
        "num_batches": args.num_batches,
        "num_workers": args.num_workers,
        "n_stations": args.n_stations,
        "n_rehd": args.n_rehd,
        "reward_preset": args.reward_preset,
        "base_seed": args.base_seed,
        "updates_per_episode": upd,
        "grid": GRID,
        "sweep": "schedule-major (g*75+u)%1600",
        "shards_written": len(shard_files),
        "total_rows": total_rows,
        "skipped": skipped,
        "failures": fails,
        "wall_sec": time.time() - t0,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[eda] DONE: {len(shard_files)}/{n_episodes} shards on disk, {total_rows} rows, "
        f"{len(fails)} fails, {(time.time()-t0)/60:.1f} min -> {out_dir}"
    )
    print(f"[eda] load: import pandas as pd; df = pd.read_parquet('{shard_dir}')")
    if fails:
        print(f"[eda] {len(fails)} failures, e.g.: {fails[0]}")
    if fails:
        print(f"[eda] {len(fails)} FAILURES, e.g.: {fails[0]}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
