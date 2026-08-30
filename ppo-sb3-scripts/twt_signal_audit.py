#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Characterize ALL ~89 per-STA signals in the EDA parquet to decide which are usable
for the OBS (realistic only) and the REWARD (oracle allowed).

Per signal (cumulative counters -> per-step delta within (seed,sta); levels -> value):
  ALIVE        : std>0 and >1 distinct value (dead/constant signals are useless)
  CV           : std/|mean| — relative variability
  SCHED-eta2   : fraction of the signal's variance explained by the action (sched index)
                 = how much the SCHEDULE moves it (key for reward/learning usability)
  CLASS-d      : |mean_REHD - mean_nonREHD| / pooled_std (Cohen's d) — class separation
  REDUNDANT    : highest |pearson| with any OTHER signal (>=0.97 => drop one)
"""

import os, sys, glob
import numpy as np
import pandas as pd

OUT = (
    os.path.abspath(open("/dev/shm/eda_outdir.txt").read().strip())
    if os.path.exists("/dev/shm/eda_outdir.txt")
    else None
)
RUN = sys.argv[1] if len(sys.argv) > 1 else OUT


def main():
    shards = sorted(glob.glob(os.path.join(RUN, "shards", "*.parquet")))
    cols0 = pd.read_parquet(shards[0]).columns
    sig = [c for c in cols0 if c.startswith("real_") or c.startswith("orc_")]
    keys = ["seed", "sta_id", "sched", "pdw_idx"]
    df = pd.concat(
        [pd.read_parquet(s, columns=keys + sig) for s in shards], ignore_index=True
    )
    df = df.sort_values(["seed", "sta_id", "pdw_idx"]).reset_index(
        drop=True
    )  # ordering for diff
    print(f"rows={len(df):,} signals={len(sig)}")
    is_rehd = df["orc_vcap_max_v"].values > 0

    # classify cumulative (non-decreasing within (seed,sta)) vs level; build analysis value
    g = df.groupby(["seed", "sta_id"], sort=False)
    sched = df["sched"].values
    rows = []
    av = (
        {}
    )  # analysis values per signal (delta for cumulative-dynamic, value otherwise)
    for c in sig:
        v = df[c].values.astype(float)
        gc = g[c]
        within = float(gc.std().mean())  # avg variation WITHIN a (seed,sta) timeline
        between = float(gc.mean().std())  # variation ACROSS STAs/scenarios
        total_std = float(np.nanstd(v))
        # classify
        if total_std < 1e-12:
            cls = "dead"
        elif within < 1e-9 and between > 1e-12:
            cls = "static"  # constant per STA, varies across (a descriptor)
        else:
            cls = "dynamic"
        # analysis value: cumulative counter -> per-step delta; else the value
        d = gc.diff().values
        frac_nonneg = np.nanmean(d >= -1e-9)
        cumulative = (cls == "dynamic") and (frac_nonneg > 0.95) and (np.nanmin(v) >= 0)
        x = np.where(np.isnan(d), 0.0, np.clip(d, 0, None)) if cumulative else v
        av[c] = x
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        cv = sd / abs(mu) if abs(mu) > 1e-12 else np.inf
        # schedule eta^2 (only meaningful for dynamic signals)
        if cls == "dynamic" and sd > 1e-12:
            tmp = pd.DataFrame({"x": x, "s": sched})
            gm = tmp.groupby("s")["x"].transform("mean")
            eta2 = float(((gm - x.mean()) ** 2).mean() / (x.var() + 1e-12))
        else:
            eta2 = 0.0
        xr, xn = v[is_rehd], v[~is_rehd]  # class sep always on the VALUE
        pooled = np.sqrt((np.nanvar(xr) + np.nanvar(xn)) / 2 + 1e-12)
        d_cls = abs(np.nanmean(xr) - np.nanmean(xn)) / pooled if pooled > 0 else 0.0
        rows.append(
            dict(
                signal=c,
                cls=cls,
                mean=mu,
                std=sd,
                cv=cv,
                sched_eta2=eta2,
                class_d=float(d_cls),
            )
        )
    R = pd.DataFrame(rows)
    R["alive"] = R.cls != "dead"

    # redundancy: pairwise |corr| among ALIVE signals on a 60k-row sample of analysis values
    alive_sigs = R[R.alive].signal.tolist()
    n = len(df)
    idx = np.random.RandomState(0).choice(n, min(60000, n), replace=False)
    A = np.column_stack([av[c][idx] for c in alive_sigs])
    A = np.nan_to_num(A)
    C = np.corrcoef(A, rowvar=False)
    np.fill_diagonal(C, 0.0)
    red = {}
    for i, c in enumerate(alive_sigs):
        j = np.argmax(np.abs(C[i]))
        red[c] = (alive_sigs[j], float(C[i, j]))
    R["redund_with"] = R.signal.map(lambda c: red.get(c, ("", 0.0))[0])
    R["redund_corr"] = R.signal.map(lambda c: round(red.get(c, ("", 0.0))[1], 2))

    R["realistic"] = R.signal.str.startswith("real_")
    R = R.sort_values(["realistic", "sched_eta2"], ascending=[False, False])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    R = R.sort_values(
        ["realistic", "cls", "sched_eta2"], ascending=[False, True, False]
    )

    def show(sub, title):
        print(f"\n===== {title} =====")
        print(
            f"{'signal':36} {'cls':8} {'sched_eta2':>10} {'class_d':>8} {'cv':>7} {'redund(|r|>.97)':>22}"
        )
        for _, r in sub.iterrows():
            rd = (
                f"{r.redund_with[:18]}={r.redund_corr}"
                if abs(r.redund_corr) >= 0.97
                else ""
            )
            print(
                f"{r.signal:36} {r.cls:8} {r.sched_eta2:10.3f} {r.class_d:8.2f} {r.cv:7.2f} {rd:>22}"
            )

    show(R[R.realistic], "REALISTIC signals (obs-eligible)")
    show(R[~R.realistic], "ORACLE signals (reward-only)")

    print("\n===== SUMMARY =====")
    for k in ["dead", "static", "dynamic"]:
        s = R[R.cls == k].signal.tolist()
        print(
            f"{k.upper()} ({len(s)}): {[x.replace('real_','').replace('orc_','') for x in s]}"
        )
    rr = R[R.realistic & (R.cls == "dynamic")]
    good_obs = rr[(rr.sched_eta2 > 0.02) & (rr.redund_corr.abs() < 0.97)]
    print(
        f"\nGOOD OBS candidates (realistic, DYNAMIC, sched_eta2>0.02, non-redundant): {len(good_obs)}"
    )
    for _, r in good_obs.sort_values("sched_eta2", ascending=False).iterrows():
        print(
            f"   {r.signal.replace('real_',''):32} eta2={r.sched_eta2:.3f} class_d={r.class_d:.2f}"
        )


if __name__ == "__main__":
    main()
