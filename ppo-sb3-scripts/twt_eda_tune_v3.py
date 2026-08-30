#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Offline v3 reward weight-tuning on the EDA parquet.

Reads the dense action-grid EDA (runs/eda_v3weights_*/shards/*.parquet), reconstructs
per-step per-STA deltas (served / expiry / REAL latency / REHD energy / occupancy) by
diffing the cumulative oracle counters within each (seed, sta) timeline, derives the
over/under regime from (seed, segment) via the same splitmix64 coin the C++ uses, and
evaluates the multi-objective v3 reward.

Answers three questions:
  (A) COMMENSURABILITY — is each weighted term contributing, none dominating/vanishing?
  (B) CORRECTNESS — does total reward move the right way with served/expiry/latency/H-C?
  (C) STATE-DEPENDENCE / TRADEOFF — does the v3-reward-maximizing grouping DIFFER by
      regime (the learnable adaptivity headroom)? Does it protect REHDs in OVER?
Then sweeps weight variants and reports which maximizes the over/under optimum divergence
while keeping all terms live.
"""

import os, sys, glob, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- regime coin (matches twt-simulation-config.cc UpdateTrafficSegment) ----
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


SEG_UPD = 20  # 100 updates / 5 segments

CUM = [
    "orc_packets_transmitted",
    "orc_packets_enqueued",
    "orc_mpdu_drops_expired",
    "orc_queue_delay_sum_ms",
    "orc_queue_delay_count",
    "orc_consumed_total_j",
    "orc_harvested_total_j",
    "real_airtime_used_us",
    "orc_bytes_transmitted",
]
KEEP = [
    "seed",
    "update_idx",
    "sta_id",
    "sched",
    "pdw_idx",
    "orc_vcap_max_v",
    "orc_vcap_v",
] + CUM


def load(run_dir, max_shards=None):
    shards = sorted(glob.glob(os.path.join(run_dir, "shards", "*.parquet")))
    if max_shards:
        shards = shards[:max_shards]
    parts = []
    for i, s in enumerate(shards):
        d = pd.read_parquet(s, columns=KEEP)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    return df


def build_steps(df):
    """Diff cumulatives within (seed,sta) -> per-step deltas. Row u carries action(u)."""
    df = df.sort_values(["seed", "sta_id", "update_idx"]).reset_index(drop=True)
    g = df.groupby(["seed", "sta_id"], sort=False)
    d = {}
    for c in CUM:
        d["d_" + c] = g[c].diff()
    out = df.copy()
    for k, v in d.items():
        out[k] = v
    out = out.dropna(subset=["d_orc_packets_enqueued"]).reset_index(
        drop=True
    )  # drops u=0
    # clip negatives (counter resets shouldn't happen, but be safe)
    for c in CUM:
        out["d_" + c] = out["d_" + c].clip(lower=0.0)
    # per-step metrics
    eq = out["d_orc_packets_enqueued"].values
    tx = out["d_orc_packets_transmitted"].values
    dr = out["d_orc_mpdu_drops_expired"].values
    out["served"] = np.where(eq > 0, np.minimum(1.0, tx / np.maximum(eq, 1)), 1.0)
    out["droprate"] = np.where(eq > 0, np.minimum(1.0, dr / np.maximum(eq, 1)), 0.0)
    qdc = out["d_orc_queue_delay_count"].values
    qds = out["d_orc_queue_delay_sum_ms"].values
    out["lat_ms"] = np.where(qdc > 0, qds / np.maximum(qdc, 1), np.nan)
    cons = out["d_orc_consumed_total_j"].values
    harv = out["d_orc_harvested_total_j"].values
    out["is_rehd"] = out["orc_vcap_max_v"].values > 0
    out["hc"] = np.where(
        (out["is_rehd"].values) & (cons > 0),
        np.minimum(1.0, harv / np.maximum(cons, 1e-12)),
        np.nan,
    )
    out["n_groups"] = (out["sched"].values // 5) + 1
    # regime
    seg = (out["update_idx"].values // SEG_UPD).astype(int)
    out["seg"] = seg
    over = np.array([is_over(s, g) for s, g in zip(out["seed"].values, seg)])
    out["regime"] = np.where(over, "OVER", "UNDER")
    return out


# ---- v3 reward on a per-(seed,update) STEP aggregate (matches reward_functions.v3) ----
def step_reward(grp, W, demand_floor=3):
    eq = grp["d_orc_packets_enqueued"].values
    inc = eq >= demand_floor
    served = grp["served"].values
    served_for = np.where(inc, served, 1.0)
    drop_for = np.where(inc, grp["droprate"].values, 0.0)
    mean_served = served_for.mean()
    mean_drop = drop_for.mean()
    # worst-k mean shortfall (matches reward_functions v3 REV 2026-06-06)
    if len(served_for):
        ssq = np.sort((1.0 - served_for) ** 2)[::-1]
        k = max(1, int(np.ceil(W.get("mv_worst_frac", 0.25) * len(ssq))))
        max_starv = float(ssq[:k].mean())
    else:
        max_starv = 0.0
    lat = grp["lat_ms"].values
    latm = lat[~np.isnan(lat)]
    mean_lat = (
        float(np.minimum(1.0, latm / W["lat_deadline_ms"]).mean()) if len(latm) else 0.0
    )
    hc = grp["hc"].values
    hcm = hc[~np.isnan(hc)]
    rehd_hc = float(hcm.mean()) if len(hcm) else 0.0
    occ = float(
        np.clip(grp["d_real_airtime_used_us"].sum() / (25 * 102.4 * 1000.0), 0, 1)
    )
    terms = dict(
        w_served=W["w_srv"] * mean_served,
        w_expiry=-W["w_exp"] * mean_drop,
        w_latency=-W["w_lat"] * mean_lat,
        w_starv=-W["w_mv"] * max_starv,
        w_energy=W["w_eng"] * rehd_hc,
        w_air=-W["w_air"] * occ,
    )
    total = float(np.clip(sum(terms.values()), -W["r_clip"], W["r_clip"]))
    return (
        total,
        terms,
        dict(
            served=mean_served,
            drop=mean_drop,
            lat=mean_lat,
            starv=max_starv,
            hc=rehd_hc,
            occ=occ,
        ),
    )


DEFAULT_W = dict(
    w_srv=1.0,
    w_exp=0.6,
    w_lat=0.4,
    w_mv=0.5,
    w_eng=0.3,
    w_air=0.3,
    lat_deadline_ms=2000.0,
    r_clip=5.0,
)


def aggregate_steps(steps, W):
    """Compute per-step v3 reward + components, return a step-level dataframe."""
    rows = []
    for (seed, u), grp in steps.groupby(["seed", "update_idx"], sort=False):
        total, terms, raw = step_reward(grp, W)
        ng = int(grp["n_groups"].iloc[0])
        reg = grp["regime"].iloc[0]
        pdw = int(grp["pdw_idx"].iloc[0])
        rows.append(
            dict(
                seed=seed,
                u=u,
                n_groups=ng,
                pdw=pdw,
                regime=reg,
                total=total,
                **terms,
                **raw,
            )
        )
    return pd.DataFrame(rows)


def main():
    run = (
        sys.argv[1]
        if len(sys.argv) > 1
        else open("/dev/shm/eda_outdir.txt").read().strip()
    )
    if not os.path.isabs(run):
        run = (
            os.path.join(HERE, "..", "..", "..", "..", "..", run)
            if not os.path.exists(run)
            else run
        )
    run = os.path.abspath(run)
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(f"[tune] loading {run}")
    df = load(run, mx)
    print(
        f"[tune] rows={len(df):,} seeds={df.seed.nunique()} stas={df.sta_id.nunique()}"
    )
    steps = build_steps(df)
    print(f"[tune] step-rows={len(steps):,}")

    # --- validate regime mapping: OVER should have higher demand than UNDER ---
    eqO = steps.loc[steps.regime == "OVER", "d_orc_packets_enqueued"].mean()
    eqU = steps.loc[steps.regime == "UNDER", "d_orc_packets_enqueued"].mean()
    print(
        f"\n[VALIDATE regime] mean Δenqueue/step  OVER={eqO:.1f}  UNDER={eqU:.1f}  ratio={eqO/max(eqU,1e-9):.2f}x"
    )
    print(f"  REHD frac of STAs: {steps.is_rehd.mean():.2f} (expect ~0.40 = 8/20)")

    sr = aggregate_steps(steps, DEFAULT_W)
    print(
        f"\n=== (A) TERM COMMENSURABILITY (default weights { {k:DEFAULT_W[k] for k in ['w_srv','w_exp','w_lat','w_mv','w_eng','w_air']} }) ==="
    )
    print(f"{'term':>12} {'mean':>8} {'p10':>8} {'p50':>8} {'p90':>8} {'std':>8}")
    for t in [
        "w_served",
        "w_expiry",
        "w_latency",
        "w_starv",
        "w_energy",
        "w_air",
        "total",
    ]:
        v = sr[t].values
        print(
            f"{t:>12} {v.mean():8.3f} {np.percentile(v,10):8.3f} {np.percentile(v,50):8.3f} {np.percentile(v,90):8.3f} {v.std():8.3f}"
        )

    print(
        f"\n=== (B) CORRECTNESS: corr(total reward, raw axis) — signs should be +,-,-,+ ==="
    )
    for ax, want in [
        ("served", "+"),
        ("drop", "-"),
        ("lat", "-"),
        ("hc", "+"),
        ("starv", "-"),
        ("occ", "-"),
    ]:
        c = np.corrcoef(sr["total"], sr[ax])[0, 1]
        print(f"  corr(total,{ax:7}) = {c:+.3f}  (want {want})")

    print(
        f"\n=== (C) STATE-DEPENDENCE: per (regime x n_groups) means; is the v3-optimal grouping different? ==="
    )
    print(
        f"{'reg':>5} {'g':>2} {'n':>5} | {'srvALL':>6} {'srvREHD':>7} {'expREHD':>7} {'lat_ms':>7} {'H/C':>5} {'occ':>5} | {'v3_rew':>7}"
    )
    opt = {}
    for reg in ["UNDER", "OVER"]:
        best = (None, -1e9)
        for g in range(1, 9):
            sub = sr[(sr.regime == reg) & (sr.n_groups == g)]
            if len(sub) < 20:
                continue
            st = steps[(steps.regime == reg) & (steps.n_groups == g)]
            srvALL = st["served"].mean()
            rh = st[st.is_rehd]
            srvR = rh["served"].mean()
            expR = rh["droprate"].mean()
            latR = st["lat_ms"].mean()
            hc = st["hc"].mean()
            occ = sub["occ"].mean()
            rew = sub["total"].mean()
            print(
                f"{reg:>5} {g:>2} {len(sub):>5} | {srvALL:6.2f} {srvR:7.2f} {expR:7.2f} {latR:7.0f} {hc:5.2f} {occ:5.2f} | {rew:7.3f}"
            )
            if rew > best[1]:
                best = (g, rew)
        opt[reg] = best
        print()
    print(
        f"  v3-OPTIMAL grouping: UNDER -> g{opt['UNDER'][0]} ({opt['UNDER'][1]:.3f}) ; OVER -> g{opt['OVER'][0]} ({opt['OVER'][1]:.3f})"
    )
    print(
        f"  => {'STATE-DEPENDENT (adaptive lever exists)' if opt['UNDER'][0]!=opt['OVER'][0] else 'SAME optimum both regimes (weak adaptivity at this weighting)'}"
    )

    # --- weight sweep: which weighting maximizes regime-optimum divergence + REHD protection in OVER ---
    print(
        f"\n=== (D) WEIGHT SWEEP: regime-optimal grouping + OVER REHD served under each weighting ==="
    )
    variants = {
        "default": DEFAULT_W,
        "heavy_expiry": {**DEFAULT_W, "w_exp": 1.2},
        "heavy_latency": {**DEFAULT_W, "w_lat": 1.0},
        "heavy_energy": {**DEFAULT_W, "w_eng": 0.8},
        "rehd_focus": {**DEFAULT_W, "w_exp": 1.0, "w_eng": 0.6, "w_lat": 0.6},
        "throughput_only": {**DEFAULT_W, "w_exp": 0.2, "w_lat": 0.1, "w_eng": 0.0},
    }
    print(
        f"{'variant':>16} | {'UNDERopt':>8} {'OVERopt':>8} {'adaptive?':>9} | {'OVER g1 rew':>11} {'OVER g4 rew':>11}"
    )
    for name, W in variants.items():
        srw = aggregate_steps(steps, W)
        o = {}
        for reg in ["UNDER", "OVER"]:
            b = (None, -1e9)
            for g in range(1, 9):
                sub = srw[(srw.regime == reg) & (srw.n_groups == g)]
                if len(sub) < 20:
                    continue
                if sub["total"].mean() > b[1]:
                    b = (g, sub["total"].mean())
            o[reg] = b
        g1 = srw[(srw.regime == "OVER") & (srw.n_groups == 1)]["total"].mean()
        g4 = srw[(srw.regime == "OVER") & (srw.n_groups == 4)]["total"].mean()
        adap = "YES" if o["UNDER"][0] != o["OVER"][0] else "no"
        print(
            f"{name:>16} | g{o['UNDER'][0]:<7} g{o['OVER'][0]:<7} {adap:>9} | {g1:>11.3f} {g4:>11.3f}"
        )

    # save the step-level reward table for any further offline work
    outcsv = os.path.join(run, "v3_step_rewards_default.csv")
    sr.to_csv(outcsv, index=False)
    print(f"\n[tune] wrote {outcsv}")


if __name__ == "__main__":
    main()
