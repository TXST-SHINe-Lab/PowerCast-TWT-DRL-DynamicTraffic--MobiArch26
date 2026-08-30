#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Fixed-policy confirmation sweep (sustained action) for v3 reward tuning.

The dense-grid EDA changes the action every step, washing out sustained-policy dynamics
(esp. REHD capacitor draining under a sustained big SP). This sweep HOLDS ONE ACTION per
episode for the full horizon so those dynamics equilibrate, and contrasts:
  - granularity g in {1,2,3,4,6,8}  (number of TWT groups)
  - assignment  in {rr, smart}      (round-robin+equal windows vs class/demand-aware+sized)
  - pdw         in {5,25,50} ms      (harvest window; comms SPs packed into [D,93])
across N seeds. Each fixed-action episode spans all 5 traffic segments, so every episode
yields BOTH over- and under-saturated steps (segment-tagged via the same splitmix coin the
C++ uses). Per step it builds the real v3 sta_deltas and computes the REAL reward.

Outputs per (episode, step): the action, regime, the v3 total + raw per-axis values
(served/drop/latency/starv/rehd_hc/occupancy) and per-class metrics. Because the reward is
a linear combo of those raw axes (max_starv saved raw too), WEIGHTS CAN BE RE-TUNED OFFLINE
from this parquet with no re-simulation — that's the "tune further" step.

Answers: (1) true REHD-protection magnitude under sustained policy, (2) does smart assignment
Pareto-beat round-robin, (3) is the v3-optimal action REGIME-DEPENDENT (adaptivity headroom)?
"""

import os, sys, time, queue as _queue, multiprocessing as mp
from collections import defaultdict
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.dirname(_here)


def _envint(k, d):
    return int(os.environ.get(k, d))


def _envlist(k, d):
    return [float(x) for x in os.environ[k].split(",")] if k in os.environ else d


N_STA, N_REHD = 12, 8
NUM = N_STA + N_REHD
STEPS, WARMUP, SEG_UPD = _envint("SW_STEPS", 90), _envint("SW_WARMUP", 5), 20
BI = 102.4
COMMS_HI = 93.0
SCALE = float(
    os.environ.get("SW_SCALE", "1.9")
)  # trafficScale anchor (re-calibrate after heterogeneity change)
_NSEED = _envint("SW_NSEED", 8)
SEEDS = [6100 + i for i in range(_NSEED)]
GRAN = [int(x) for x in _envlist("SW_GRAN", [1, 2, 3, 4, 6, 8])]
ASSIGN = os.environ.get("SW_ASSIGN", "rr,smart").split(",")
PDW = _envlist("SW_PDW", [5.0, 25.0, 50.0])
PARALLEL = _envint("SW_PARALLEL", 6)

M = (1 << 64) - 1


def _sm(x):
    x &= M
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & M
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & M
    return x ^ (x >> 31)


def is_over(seed, seg):
    seg = int(seg)
    h = ((int(seed) * 0x9E3779B97F4A7C15) + ((seg + 1) * 0xBF58476D1CE4E5B9)) & M
    return (_sm(h) & 1) == 0


def mk(groups, pdw_end):
    """groups = list of (sta_ids, duration_ms); comms packed into [pdw_end, 93]."""
    cfgs, assigns = [], []
    off = float(pdw_end)
    for gid, (sids, dur) in enumerate(groups):
        dur = float(max(2.0, dur))
        cfgs.append(
            {
                "group_id": gid,
                "twt_wake_interval_ms": BI,
                "twt_wake_duration_ms": dur,
                "twt_sp_offset_ms": off,
                "num_stas_assigned": len(sids),
            }
        )
        for s in sids:
            assigns.append(
                {"sta_id": int(s), "assigned_twt_group": gid, "enable_twt": 1}
            )
        off += dur
    return {
        "num_sta": NUM,
        "num_active_twt_groups": len(groups),
        "action_timestamp_ms": 0,
        "pdw_duration_ms": float(pdw_end),
        "twt_group_configs": cfgs,
        "sta_group_assignments": assigns,
    }


def build_action(g, mode, pdw_end, demand):
    """demand: np array len NUM of per-STA measured enqueue demand. REHDs = ids [N_STA, NUM)."""
    budget = COMMS_HI - pdw_end
    rehd = list(range(N_STA, NUM))
    nonr = list(range(N_STA))
    if g == 1:
        return mk([(list(range(NUM)), budget)], pdw_end)
    if mode == "rr":
        b = [[] for _ in range(g)]
        for i in range(NUM):
            b[i % g].append(i)
        groups = [(grp, budget / g) for grp in b if grp]
        return mk(groups, pdw_end)
    # smart: isolate REHDs in dedicated group(s); non-REHDs binned by demand; windows demand-sized
    g_rehd = max(1, round(g * N_REHD / NUM))
    g_rehd = min(g_rehd, g - 1)  # leave >=1 group for non-REHD
    g_non = g - g_rehd
    # REHDs split across g_rehd groups (contiguous)
    rehd_sorted = sorted(rehd, key=lambda i: -demand[i])
    rgroups = [rehd_sorted[j::g_rehd] for j in range(g_rehd)]
    # non-REHDs sorted by demand, binned contiguously (similar-demand together -> anti perf-anomaly)
    nonr_sorted = sorted(nonr, key=lambda i: -demand[i])
    splits = np.array_split(nonr_sorted, g_non)
    ngroups = [list(s) for s in splits if len(s)]
    all_groups = [grp for grp in (rgroups + ngroups) if grp]
    # demand-size the windows (min 2 ms), proportional to each group's total demand
    dem = np.array([max(1e-6, sum(demand[i] for i in grp)) for grp in all_groups])
    win = np.maximum(2.0, budget * dem / dem.sum())
    # renormalize so total <= budget
    if win.sum() > budget:
        win = win * (budget / win.sum())
        win = np.maximum(2.0, win)
    return mk(list(zip(all_groups, win)), pdw_end)


def build_deltas(prev_list, curr_list):
    """Per-STA delta dicts in the format reward_functions v3 reads."""
    g = lambda d, k: float((d or {}).get(k, 0) or 0)
    out = []
    for i, c in enumerate(curr_list):
        p = prev_list[i] if i < len(prev_list) else {}
        co, po = c.get("oracle", {}), (p.get("oracle", {}) if p else {})
        cr = c.get("realistic", {})
        d_qds = max(0.0, g(co, "queue_delay_sum_ms") - g(po, "queue_delay_sum_ms"))
        d_qdc = max(0.0, g(co, "queue_delay_count") - g(po, "queue_delay_count"))
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
                "delta_airtime_used_us": max(
                    0.0,
                    g(cr, "airtime_used_us")
                    - g(p.get("realistic", {}) if p else {}, "airtime_used_us"),
                ),
                "duty_cycle": float(np.clip(g(co, "duty_cycle"), 0, 1)),
                "bsr_queue_index": g(cr, "bsr_queue_ac_be"),
                "step_latency_ms": (d_qds / d_qdc) if d_qdc > 0 else 0.0,
                "latency_count": d_qdc,
                "vcap_max_v": g(co, "vcap_max_v"),
                "vcap_v": g(co, "vcap_v"),
                "delta_harvested_j": max(
                    0.0, g(co, "harvested_total_j") - g(po, "harvested_total_j")
                ),
                "delta_consumed_j": max(
                    0.0, g(co, "consumed_total_j") - g(po, "consumed_total_j")
                ),
            }
        )
    return out


def worker(g, mode, pdw_end, seed, shm, out_q):
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    for v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[v] = "1"
    sys.path.insert(0, _here)
    sys.path.insert(0, _pkg)
    os.chdir(os.path.abspath(os.path.join(_pkg, "..", "..", "..", "..")))
    res = {
        "g": g,
        "mode": mode,
        "pdw": pdw_end,
        "seed": seed,
        "ok": False,
        "err": None,
        "rows": [],
    }
    w = None
    try:
        from pb_twt_wrapper_py import TWTWrapper
        import reward_functions as rf

        gg = lambda d, k: float((d or {}).get(k, 0) or 0)
        w = TWTWrapper(
            enable_logging=False,
            verbose=False,
            quiet_ns3=True,
            disable_ns3_traces=True,
            n_stations=N_STA,
            n_rehd=N_REHD,
            traffic_scale=SCALE,
        )
        env = w.reset(seed=shm, rand_seed=seed)
        # warmup with a permissive 1-group action; measure per-STA demand
        warm = mk([(list(range(NUM)), COMMS_HI - 5.0)], 5.0)
        demand = np.zeros(NUM)
        prev = env
        for _ in range(WARMUP):
            env, done = w.step(warm)
            cl, pl = env.get("sta_observations", []), prev.get("sta_observations", [])
            for i in range(min(len(cl), len(pl))):
                demand[i] += max(
                    0.0,
                    gg(cl[i].get("oracle", {}), "packets_enqueued")
                    - gg(pl[i].get("oracle", {}), "packets_enqueued"),
                )
            prev = env
        act = build_action(g, mode, pdw_end, demand)
        prev = env
        for k in range(STEPS):
            env, done = w.step(act)
            pl = prev.get("sta_observations", [])
            cl = env.get("sta_observations", [])
            if not cl:
                break
            sd = build_deltas(pl, cl)
            r = rf.compute_twt_pf_demand_v3_reward(sd)
            c = r["components"]
            seg = k // SEG_UPD
            reg = "OVER" if is_over(seed, seg) else "UNDER"
            # per-class raw served/expiry/latency
            tx = np.array([d["delta_packets_transmitted"] for d in sd])
            eq = np.array([d["delta_packets_enqueued"] for d in sd])
            dr = np.array([d["delta_drops_expired"] for d in sd])
            isre = np.array([d["vcap_max_v"] > 0 for d in sd])
            srv = np.where(eq > 0, np.minimum(1, tx / np.maximum(eq, 1)), 1.0)
            exp = np.where(eq > 0, np.minimum(1, dr / np.maximum(eq, 1)), 0.0)
            res["rows"].append(
                dict(
                    g=g,
                    mode=mode,
                    pdw=pdw_end,
                    seed=seed,
                    step=k,
                    regime=reg,
                    v3=r["total"],
                    raw_served=c["raw_mean_served"],
                    raw_drop=c["raw_mean_drop"],
                    raw_lat=c["raw_mean_latency"],
                    raw_starv=c["raw_max_starv"],
                    raw_hc=c["raw_rehd_hc"],
                    raw_occ=c["raw_occupancy"],
                    srv_rehd=float(srv[isre].mean()) if isre.any() else np.nan,
                    exp_rehd=float(exp[isre].mean()) if isre.any() else np.nan,
                    srv_non=float(srv[~isre].mean()) if (~isre).any() else np.nan,
                    n_groups=act["num_active_twt_groups"],
                )
            )
            prev = env
            if done:
                break
        res["ok"] = True
    except Exception as e:
        import traceback

        res["err"] = f"{type(e).__name__}: {e} | {traceback.format_exc()[-300:]}"
    finally:
        try:
            out_q.put(res)
        except Exception:
            pass
        try:
            if w is not None:
                w.close()
        except Exception:
            pass
        for nm in (
            f"/dev/shm/MySeg_{shm}",
            f"/dev/shm/MyCpp2PyMsg_{shm}",
            f"/dev/shm/MyPy2CppMsg_{shm}",
            f"/dev/shm/MyLockable_{shm}",
        ):
            try:
                os.remove(nm)
            except OSError:
                pass
        try:
            out_q.cancel_join_thread()
        except Exception:
            pass
        os._exit(0)


def main():
    import pandas as pd

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_here, "runs", f"fixed_sweep_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    with open("/dev/shm/fixed_sweep_outdir.txt", "w") as f:
        f.write(out_dir)
    tasks = []
    shm = 793000
    for g in GRAN:
        for mode in ASSIGN:
            if g == 1 and mode == "smart":
                continue  # identical to rr at g1
            for pdw in PDW:
                for seed in SEEDS:
                    tasks.append((g, mode, pdw, seed, shm))
                    shm += 1
    print(
        f"=== FIXED-POLICY SWEEP: {len(tasks)} episodes, scale={SCALE}, {STEPS}+{WARMUP} steps, -> {out_dir}",
        flush=True,
    )
    rows = []
    t0 = time.time()
    for i in range(0, len(tasks), PARALLEL):
        chunk = tasks[i : i + PARALLEL]
        q = mp.Queue()
        ps = []
        for g, mode, pdw, seed, sh in chunk:
            pr = mp.Process(target=worker, args=(g, mode, pdw, seed, sh, q))
            pr.start()
            ps.append(pr)
        got = 0
        for _ in chunk:
            try:
                r = q.get(timeout=1800)
                if r.get("ok"):
                    rows.extend(r["rows"])
                elif r.get("err"):
                    print(
                        f"  [FAIL] g{r['g']} {r['mode']} D{r['pdw']} s{r['seed']}: {r['err'][:120]}",
                        flush=True,
                    )
                got += 1
            except _queue.Empty:
                print("  [TIMEOUT] chunk", flush=True)
                break
        for pr in ps:
            pr.join(timeout=20)
            if pr.is_alive():
                pr.terminate()
        try:
            q.close()
            q.cancel_join_thread()
        except Exception:
            pass
        el = (time.time() - t0) / 60
        done = i + len(chunk)
        eta = el / max(done, 1) * (len(tasks) - done)
        print(
            f"  batch {i//PARALLEL+1}/{(len(tasks)+PARALLEL-1)//PARALLEL}  eps={done}/{len(tasks)}  rows={len(rows)}  {el:.1f}m  eta={eta:.1f}m",
            flush=True,
        )
    df = pd.DataFrame(rows)
    pq = os.path.join(out_dir, "sweep.parquet")
    df.to_parquet(pq, index=False)
    print(f"\n[sweep] wrote {pq}  ({len(df)} step-rows)", flush=True)
    analyze(df)
    sys.stdout.flush()
    os._exit(0)


def analyze(df):
    import pandas as pd

    print(
        "\n=== REHD-protection + reward by (assignment, granularity, regime), averaged over PDW+seed ==="
    )
    print(
        f"{'mode':>5} {'g':>2} {'reg':>5} | {'v3':>6} {'srvNon':>6} {'srvREHD':>7} {'expREHD':>7} {'lat':>5} {'H/C':>5} {'occ':>5}"
    )
    for mode in ["rr", "smart"]:
        for g in GRAN:
            for reg in ["UNDER", "OVER"]:
                s = df[(df["mode"] == mode) & (df["g"] == g) & (df["regime"] == reg)]
                if len(s) < 10:
                    continue
                print(
                    f"{mode:>5} {g:>2} {reg:>5} | {s.v3.mean():6.3f} {s.srv_non.mean():6.2f} {s.srv_rehd.mean():7.2f} "
                    f"{s.exp_rehd.mean():7.3f} {s.raw_lat.mean():5.2f} {s.raw_hc.mean():5.2f} {s.raw_occ.mean():5.2f}"
                )
        print()
    print("=== v3-OPTIMAL (g, mode, pdw) per regime — is it state-dependent? ===")
    for reg in ["UNDER", "OVER"]:
        gg = df[df["regime"] == reg].groupby(["g", "mode", "pdw"])["v3"].mean()
        if len(gg):
            best = gg.idxmax()
            print(f"  {reg}: best (g,mode,pdw)={best}  v3={gg.max():.3f}")
    print("\n=== smart vs rr (Δv3, averaged over g>1/pdw/seed) by regime ===")
    for reg in ["UNDER", "OVER"]:
        a = df[
            (df["regime"] == reg) & (df["mode"] == "smart") & (df["g"] > 1)
        ].v3.mean()
        b = df[(df["regime"] == reg) & (df["mode"] == "rr") & (df["g"] > 1)].v3.mean()
        print(f"  {reg}: smart={a:.3f}  rr={b:.3f}  Δ={a-b:+.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        import pandas as pd

        d = (
            sys.argv[2]
            if len(sys.argv) > 2
            else open("/dev/shm/fixed_sweep_outdir.txt").read().strip()
        )
        analyze(pd.read_parquet(os.path.join(d, "sweep.parquet")))
    else:
        mp.set_start_method("spawn")
        main()
