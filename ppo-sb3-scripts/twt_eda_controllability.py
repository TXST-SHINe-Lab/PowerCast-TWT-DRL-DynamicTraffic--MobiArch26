#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""CONTROLLABILITY EDA — the non-negotiable gate before wiring obs/v4 + training.

Question: do the agent's THREE action heads (schedule, assignment, PDW) actually
MOVE the obs and reward signals on the NEW lopsided scenario? If a signal doesn't
respond to any head, it's an uncontrollable input (the SoC lesson) and must not be
in the reward; if no obs responds, the POMDP has no observable lever and learning
is hopeless.

Design — matched-seed (CRN) constant-action FACTORIAL sweep. NS-3 is deterministic
per (scenario_seed, action), so for a FIXED scenario seed the ONLY thing that moves
a signal is the action => any spread across action levels is PURE action effect
(zero scenario confound). We sweep ONE head at a time around a center action:

  sched axis : sched in SCHED_LEVELS, (assign, pdw) = CENTER
  assign axis: assign in ASSIGN_LEVELS, (sched, pdw) = CENTER
  pdw axis   : pdw in PDW_LEVELS, (sched, assign) = CENTER

Each (scenario_seed, action) runs a FULL episode (energy needs the full horizon to
equilibrate — a short run pins SoC ~1.0 and hides controllability, per the PDW plan).
Per-(update, STA) RAW realistic+oracle fields are logged; the analyzer computes the
final time-local feed offline and the per-head controllability decomposition.

SHM seed is unique per worker (base+g); the SCENARIO seed is passed via reset(rand_seed=)
so the same scenario is reused across actions (CRN) without SHM collisions.

  python3.11 .../ppo-sb3-scripts/twt_eda_controllability.py --n-stations 12 --n-rehd 8
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

UPDATES = (
    100  # full horizon (DURATION_IN_UPDATE); analyzer uses the last 30% as steady state
)

# Center action: a mid multi-group schedule, mid assignment, mid PDW (D=50).
CENTER_SCHED, CENTER_ASSIGN, CENTER_PDW = 15, 6, 9
SCHED_LEVELS = [0, 5, 11, 15, 22, 30, 35]  # num_groups 1..8 span
ASSIGN_LEVELS = [0, 5, 6, 11, 15, 25, 35]  # assignment-pattern span
PDW_LEVELS = [0, 2, 4, 6, 9, 12, 14]  # pdw_end 5,15,25,35,50,65,75 ms

# DEFAULT scenario seeds (CRN). Disjoint from train/eval seed bands.
SEEDS = [300000 + i for i in range(6)]


def build_actions():
    """Factorial-around-center set of distinct (sched, assign, pdw) actions, tagged by axis."""
    seen = {}

    def add(s, a, p, axis):
        key = (s, a, p)
        seen.setdefault(key, set()).add(axis)

    for s in SCHED_LEVELS:
        add(s, CENTER_ASSIGN, CENTER_PDW, "sched")
    for a in ASSIGN_LEVELS:
        add(CENTER_SCHED, a, CENTER_PDW, "assign")
    for p in PDW_LEVELS:
        add(CENTER_SCHED, CENTER_ASSIGN, p, "pdw")
    return [(s, a, p, sorted(ax)) for (s, a, p), ax in seen.items()]


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


def worker(
    g,
    shm_seed,
    scenario_seed,
    sched,
    assign,
    pdw,
    axes,
    out_dir,
    n_stations,
    n_rehd,
    traffic_scale,
    result_q,
):
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    for _v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_v] = "1"
    sys.path.insert(0, _proj)
    sys.path.insert(0, os.path.join(_proj, "exploration-scripts"))
    sys.path.insert(0, _here)
    os.chdir(os.path.abspath(os.path.join(_here, "..", "..", "..", "..")))

    status = {
        "g": g,
        "scenario_seed": scenario_seed,
        "ok": False,
        "rows": 0,
        "err": None,
    }
    wrapper = None
    try:
        import pandas as pd
        from pb_twt_wrapper_py import TWTWrapper
        from twt_spawn_worker import build_action_dict, load_action_tables

        st, at = load_action_tables(_proj)
        scheds = st["schedules"]
        real_fields = [f for f in TWTWrapper.REALISTIC_FIELDS if f != "sta_id"]
        orac_fields = [f for f in TWTWrapper.ORACLE_FIELDS if f != "sta_id"]

        wkw = dict(
            enable_logging=False,
            verbose=False,
            quiet_ns3=True,
            disable_ns3_traces=True,
            n_stations=n_stations,
            n_rehd=n_rehd,
        )
        if traffic_scale and traffic_scale > 0:
            wkw["traffic_scale"] = traffic_scale
        wrapper = TWTWrapper(**wkw)
        env = wrapper.reset(
            seed=shm_seed, rand_seed=scenario_seed
        )  # CRN scenario, unique SHM
        num_sta = env.get("num_sta", n_stations + n_rehd)

        act = build_action_dict(sched, assign, st, at, num_sta, pdw_idx=pdw)
        grp_wake = {
            gc["group_id"]: gc["twt_wake_duration_ms"]
            for gc in act["twt_group_configs"]
        }
        sta_grp = {
            a_["sta_id"]: a_["assigned_twt_group"]
            for a_ in act["sta_group_assignments"]
        }
        num_groups = scheds[sched]["num_groups"]
        nominal_duty = scheds[sched].get("total_duration_ms", 0) / 102.4
        axis_tag = "+".join(axes)

        rows = []
        for u in range(UPDATES):
            curr, done = wrapper.step(act)
            sim_t = float(curr.get("simulation_time_sec", 0.0))
            for i, s in enumerate(curr.get("sta_observations", [])):
                r = s.get("realistic", {})
                o = s.get("oracle", {})
                vmax = float(o.get("vcap_max_v", 0) or 0)
                sid = int(r.get("sta_id", i))
                row = {
                    "scenario_seed": scenario_seed,
                    "g": g,
                    "update_idx": u,
                    "sim_time_s": sim_t,
                    "sta_id": sid,
                    "device_class": int(o.get("device_class", -1)),
                    "is_rehd": vmax > 0,
                    "sched": sched,
                    "assign": assign,
                    "pdw_idx": pdw,
                    "axis": axis_tag,
                    "pdw_ms": float(act["pdw_duration_ms"]),
                    "num_groups": int(num_groups),
                    "assigned_group": int(sta_grp.get(sid, 0)),
                    "wake_duration_ms": float(grp_wake.get(sta_grp.get(sid, 0), 0)),
                    "nominal_duty": float(nominal_duty),
                }
                for f in real_fields:
                    row[f"real_{f}"] = float(r.get(f, 0) or 0)
                for f in orac_fields:
                    row[f"orc_{f}"] = float(o.get(f, 0) or 0)
                rows.append(row)
            if done:
                break

        df = pd.DataFrame(rows)
        shard = os.path.join(
            out_dir, f"ctl_{g:04d}_s{scenario_seed}_a{sched}-{assign}-{pdw}.parquet"
        )
        tmp = shard + ".tmp"
        df.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
        os.replace(tmp, shard)
        status.update(ok=True, rows=len(df), shard=os.path.basename(shard))
    except Exception as e:
        import traceback

        status["err"] = f"{e}\n{traceback.format_exc()}"[:1500]
    finally:
        try:
            result_q.put(status)
        except Exception:
            pass
        try:
            if wrapper is not None:
                wrapper.close()
        except Exception:
            pass
        _clean_shm(shm_seed)
        try:
            result_q.cancel_join_thread()
        except Exception:
            pass
        os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stations", type=int, default=12)
    ap.add_argument("--n-rehd", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument(
        "--base-seed", type=int, default=60000, help="SHM seed base (unique per worker)"
    )
    ap.add_argument(
        "--traffic-scale", type=float, default=0.0, help=">0 overrides C++ default"
    )
    ap.add_argument(
        "--seeds", type=int, nargs="*", default=None, help="override CRN scenario seeds"
    )
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else SEEDS
    actions = build_actions()
    tasks = [(scn, s, a, p, ax) for scn in seeds for (s, a, p, ax) in actions]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(_here, "runs", f"eda_ctl_{ts}")
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    print(f"[ctl] out={out_dir}")
    print(
        f"[ctl] {len(actions)} actions x {len(seeds)} CRN seeds = {len(tasks)} episodes "
        f"@ {args.n_stations}STA+{args.n_rehd}REHD, {UPDATES} updates/ep"
    )
    print(
        f"[ctl] axes: sched={SCHED_LEVELS} assign={ASSIGN_LEVELS} pdw={PDW_LEVELS} "
        f"center=({CENTER_SCHED},{CENTER_ASSIGN},{CENTER_PDW})"
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

    t0 = time.time()
    fails = []
    skipped = 0
    n_batches = (len(tasks) + args.num_workers - 1) // args.num_workers
    for b in range(n_batches):
        chunk = tasks[b * args.num_workers : (b + 1) * args.num_workers]
        q = mp.Queue()
        spawned = []
        for j, (scn, s, a, p, ax) in enumerate(chunk):
            g = b * args.num_workers + j
            shm_seed = args.base_seed + g
            shard = os.path.join(shard_dir, f"ctl_{g:04d}_s{scn}_a{s}-{a}-{p}.parquet")
            if os.path.exists(shard):
                skipped += 1
                continue
            _clean_shm(shm_seed)
            pr = mp.Process(
                target=worker,
                args=(
                    g,
                    shm_seed,
                    scn,
                    s,
                    a,
                    p,
                    ax,
                    shard_dir,
                    args.n_stations,
                    args.n_rehd,
                    args.traffic_scale,
                    q,
                ),
            )
            pr.start()
            spawned.append((g, shm_seed, pr))
        results = []
        for _ in spawned:
            try:
                results.append(q.get(timeout=900))
            except _queue.Empty:
                break
        got = {r.get("g") for r in results}
        for g, shm_seed, pr in spawned:
            pr.join(timeout=15)
            if pr.is_alive():
                pr.terminate()
                pr.join(timeout=5)
            if g not in got:
                _clean_shm(shm_seed)
        try:
            q.close()
            q.cancel_join_thread()
        except Exception:
            pass
        for r in results:
            if not r.get("ok"):
                fails.append(
                    (r.get("g"), r.get("scenario_seed"), (r.get("err") or "")[:200])
                )
        n_shards = len(glob.glob(os.path.join(shard_dir, "*.parquet")))
        elapsed = time.time() - t0
        eta = (elapsed / (b + 1)) * (n_batches - b - 1)
        print(
            f"[ctl] batch {b+1}/{n_batches}  shards={n_shards}/{len(tasks)}  "
            f"skipped={skipped}  fails={len(fails)}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
            flush=True,
        )

    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    manifest = {
        "timestamp": ts,
        "git_sha": sha,
        "mode": "controllability_crn_factorial",
        "n_stations": args.n_stations,
        "n_rehd": args.n_rehd,
        "updates_per_episode": UPDATES,
        "seeds": seeds,
        "center": [CENTER_SCHED, CENTER_ASSIGN, CENTER_PDW],
        "sched_levels": SCHED_LEVELS,
        "assign_levels": ASSIGN_LEVELS,
        "pdw_levels": PDW_LEVELS,
        "n_actions": len(actions),
        "n_episodes": len(tasks),
        "traffic_scale": args.traffic_scale,
        "shards_written": len(files),
        "total_rows": total_rows,
        "skipped": skipped,
        "failures": fails,
        "wall_sec": time.time() - t0,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[ctl] DONE: {len(files)}/{len(tasks)} shards, {total_rows} rows, {len(fails)} fails, "
        f"{(time.time()-t0)/60:.1f} min -> {out_dir}"
    )
    if fails:
        print(f"[ctl] FAILURES e.g.: {fails[0]}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
