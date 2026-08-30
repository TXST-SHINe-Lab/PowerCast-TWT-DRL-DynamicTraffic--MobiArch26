#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Per-device comparison report for a twt_eval_structured.py run: model vs baseline.

Reads the per-(step, STA) parquet shards and compares the two policies by DEVICE
CLASS across many axes (not just reward): served ratio, drop/expiry rate, throughput,
latency, fairness, and — for REHD — harvested/consumed energy, sustainability, SoC,
and unpowered-TX events. Comparisons are PAIRED by scenario (rand_seed), so the
model-vs-baseline delta carries a win-rate (and a Wilcoxon p-value if scipy is present).

Usage:  python3.11 twt_eval_report.py <eval_struct_run_dir>
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

_DEVICE = {0: "IoT", 1: "Camera", 2: "Voice", 3: "Video", 4: "REHD"}
EPS = 1e-9
try:
    from scipy.stats import wilcoxon

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _load(run_dir):
    sd = (
        run_dir
        if os.path.basename(run_dir.rstrip("/")) == "shards"
        else os.path.join(run_dir, "shards")
    )
    paths = sorted(glob.glob(os.path.join(sd, "*.parquet")))
    if not paths:
        sys.exit(f"[report] no shards under {sd}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["cls"] = df["device_class"].map(_DEVICE).fillna("?")
    return df


# --- metric reducers over a row-group (sums of per-step deltas) ---
# served/drop clipped to [0,1] (demand-satisfaction fraction; matches the v5 reward, where tx/eq can momentarily exceed 1 while draining prior backlog).
def _served(g):
    return min(1.0, g.d_tx.sum() / max(g.d_eq.sum(), EPS))


def _drop(g):
    return min(1.0, g.d_drops.sum() / max(g.d_eq.sum(), EPS))


# --- policy-invariant counterparts (see twt_eval_structured._extract) ---
# _served divides PHY TX STARTS by MAC ADMISSIONS: the numerator counts retries and the denominator shrinks when a blocked queue back-pressures the app, so a policy that admits less traffic scores HIGHER.
# _delivered divides frames the AP actually decoded by the app's offered load -- neither term moves with the schedule.
# Report both.
#
# WHICH ONE THE PAPER QUOTES, AND WHY.
# The paper's served/drop/fairness figures are the _served/_drop (MAC-admission) family, deliberately.
# A packet suppressed by a harvester's energy gate is never transmitted, so it is invisible to the AP and cannot enter any AP-side accounting: admitted traffic is the only demand a deployed AP could measure.
# d_gen is an app-layer oracle counted on the STA node (TxTraceAtApp), available here only because this is a simulator.
#
# The cost of that choice, on the shipped 160-episode eval, so nobody has to re-derive it: the energy gate makes 89.4% (DRL) / 81.9% (M/D/1) of REHD-generated packets never reach the enqueued counter, and the agent's lower admission is part of why its REHD served ratio reads 0.535 vs 0.358.
# Re-normalized on d_gen the picture is a tie -- REHD delivered 0.0469 vs 0.0458 (p=0.90), and Jain 0.488 vs 0.510 -- i.e. the QoS/fairness margins are admission-sensitive.
# The ENERGY results are not: harvested/consumed/soc are read straight off the capacitor with no demand term, and hold under either normalization (consumed 1.9x pooled / 2.3x paired, p=1.7e-07).
# Invariant one-line summary of the eval: same delivered load, roughly half the energy.
def _delivered(g):
    return (
        min(1.0, g.d_rx_ap.sum() / max(g.d_gen.sum(), EPS))
        if "d_gen" in g
        else float("nan")
    )


def _admitted(g):
    return (
        min(1.0, g.d_eq.sum() / max(g.d_gen.sum(), EPS))
        if "d_gen" in g
        else float("nan")
    )


def _thru(g):
    return g.d_bytes.sum() / max(len(g), 1)  # mean bytes / STA-step


def _lat(g):
    return g.lat_sum_ms.sum() / max(g.lat_count.sum(), EPS)


def _air(g):
    return g.d_airtime_us.sum() / max(len(g), 1)  # mean airtime us / STA-step


def _harv(g):
    return g.d_harv_j.sum()


def _cons(g):
    return g.d_cons_j.sum()


def _sustain(g):
    return g.d_harv_j.sum() / max(g.d_cons_j.sum(), EPS)


def _soc(g):
    return g.soc.mean()


def _unpow(g):
    return g.unpowered_tx.max()  # cumulative -> final


_QOS_METRICS = [
    ("served_ratio", _served, "+"),
    ("delivered_ratio", _delivered, "+"),
    ("admitted_frac", _admitted, "+"),
    ("drop_rate", _drop, "-"),
    ("throughput_B", _thru, "+"),
    ("latency_ms", _lat, "-"),
]
_ENERGY_METRICS = [
    ("harvested_J", _harv, "+"),
    ("consumed_J", _cons, "."),
    ("sustain_H/C", _sustain, "+"),
    ("soc", _soc, "+"),
    ("unpowered_tx", _unpow, "-"),
]


def _jain(x):
    x = np.asarray(x, dtype=float)
    return (x.sum() ** 2) / (len(x) * (x**2).sum() + EPS) if len(x) else float("nan")


def _paired(df, reducer, cls=None, better="+"):
    """Per-(rand_seed) metric for model & baseline -> (model_mean, base_mean, win_rate, p).
    win = fraction of scenarios the model is BETTER (direction-aware: '+' higher-is-better,
    '-' lower-is-better, '.' neutral -> raw model>base)."""
    d = df if cls is None else df[df.cls == cls]
    rows = []
    for (pol, seed), g in d.groupby(["policy", "rand_seed"]):
        rows.append((pol, seed, reducer(g)))
    m = pd.DataFrame(rows, columns=["policy", "rand_seed", "val"])
    piv = m.pivot_table(index="rand_seed", columns="policy", values="val")
    pols = [c for c in piv.columns if c == "model"] + [
        c for c in piv.columns if c != "model"
    ]
    if "model" not in piv.columns or len(pols) < 2:
        return None
    base = [c for c in piv.columns if c != "model"][0]
    piv = piv.dropna(subset=["model", base])
    if len(piv) == 0:
        return None
    mv, bv = piv["model"].values, piv[base].values
    win = float(np.mean(mv < bv)) if better == "-" else float(np.mean(mv > bv))
    p = float("nan")
    if _HAS_SCIPY and len(mv) >= 2 and np.any(mv != bv):
        try:
            p = float(wilcoxon(mv, bv).pvalue)
        except Exception:
            p = float("nan")
    return dict(
        model=float(np.mean(mv)),
        base=float(np.mean(bv)),
        base_name=base,
        win=win,
        p=p,
        n=len(piv),
    )


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a"
    ax = abs(x)
    if ax >= 1e6 or (0 < ax < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def generate(run):
    """Build the per-device comparison report for an eval-structured run dir.
    Writes report.txt + per_class_comparison.csv and returns the report text."""
    run = os.path.abspath(run)
    df = _load(run)
    base_name = [p for p in df.policy.unique() if p != "model"]
    base_name = base_name[0] if base_name else "?"
    splits = sorted(df.n_rehd.unique())
    n_scen = df.rand_seed.nunique()

    L = []

    def rep(s=""):
        L.append(s)
        print(s)

    rep("=" * 84)
    rep(f"STRUCTURED PER-DEVICE EVAL  —  model  vs  {base_name}")
    rep(f"run={run}")
    rep(
        f"scenarios={n_scen}  splits(n_rehd)={splits}  rows={len(df):,}  "
        f"paired-stats={'Wilcoxon' if _HAS_SCIPY else 'win-rate only'}"
    )
    rep("=" * 84)

    # ---- 1. Overall (all classes) ----
    rep("\n[1] OVERALL (all device classes, paired by scenario)")
    rep(
        f"  {'metric':14} {'model':>10} {base_name[:10]:>10} {'delta':>10} {'bettr%':>6} {'p':>8}"
    )
    rep("  " + "-" * 62)
    for name, red, _dir in _QOS_METRICS + [("reward", lambda g: g.reward.mean(), "+")]:
        r = _paired(df, red, better=_dir)
        if r:
            rep(
                f"  {name:14} {_fmt(r['model']):>10} {_fmt(r['base']):>10} "
                f"{_fmt(r['model']-r['base']):>10} {100*r['win']:>5.0f} {_fmt(r['p'],3):>8}"
            )
    # fairness (Jain over per-STA served_ratio, per scenario)
    fr = _paired(
        df, lambda g: _jain(g.groupby("sta_id").apply(_served).values), better="+"
    )
    if fr:
        rep(
            f"  {'fairness(Jain)':14} {_fmt(fr['model']):>10} {_fmt(fr['base']):>10} "
            f"{_fmt(fr['model']-fr['base']):>10} {100*fr['win']:>5.0f} {_fmt(fr['p'],3):>8}"
        )

    # ---- 2. Per device class (QoS) ----
    rep(
        "\n[2] PER DEVICE CLASS  (model vs baseline; delta>0 better for served/thru, worse for drop/lat)"
    )
    for cls in ["IoT", "Camera", "Voice", "Video", "REHD"]:
        if cls not in df.cls.unique():
            continue
        rep(f"  --- {cls} ---")
        rep(
            f"     {'metric':12} {'model':>10} {base_name[:10]:>10} {'delta':>10} {'bettr%':>6} {'p':>8}"
        )
        for name, red, _dir in _QOS_METRICS:
            r = _paired(df, red, cls=cls, better=_dir)
            if r:
                rep(
                    f"     {name:12} {_fmt(r['model']):>10} {_fmt(r['base']):>10} "
                    f"{_fmt(r['model']-r['base']):>10} {100*r['win']:>5.0f} {_fmt(r['p'],3):>8}"
                )

    # ---- 3. REHD energy ----
    if "REHD" in df.cls.unique():
        rep("\n[3] REHD ENERGY  (oracle; the harvesting story)")
        rep(
            f"     {'metric':12} {'model':>10} {base_name[:10]:>10} {'delta':>10} {'bettr%':>6} {'p':>8}"
        )
        for name, red, _dir in _ENERGY_METRICS:
            r = _paired(df, red, cls="REHD", better=_dir)
            if r:
                rep(
                    f"     {name:12} {_fmt(r['model']):>10} {_fmt(r['base']):>10} "
                    f"{_fmt(r['model']-r['base']):>10} {100*r['win']:>5.0f} {_fmt(r['p'],3):>8}"
                )

    # ---- 4. Per-split (edge cases: REHD-light -> REHD-heavy) ----
    rep("\n[4] BY REHD-SPLIT  (served_ratio / reward, model vs baseline)")
    rep(
        f"  {'n_rehd':>7} {'served(m)':>10} {'served(b)':>10} {'rew(m)':>9} {'rew(b)':>9}"
    )
    for nr in splits:
        sub = df[df.n_rehd == nr]
        sm = _served(sub[sub.policy == "model"])
        sb = _served(sub[sub.policy != "model"])
        rm = sub[sub.policy == "model"].reward.mean()
        rb = sub[sub.policy != "model"].reward.mean()
        rep(f"  {nr:>7} {_fmt(sm):>10} {_fmt(sb):>10} {_fmt(rm,2):>9} {_fmt(rb,2):>9}")

    # ---- 5. Over/under load ----
    if df.over.nunique() > 1:
        rep("\n[5] BY LOAD SEGMENT  (served_ratio / reward)")
        rep(
            f"  {'segment':>9} {'served(m)':>10} {'served(b)':>10} {'rew(m)':>9} {'rew(b)':>9}"
        )
        for ov, lbl in [(True, "over"), (False, "under")]:
            sub = df[df.over == ov]
            sm = _served(sub[sub.policy == "model"])
            sb = _served(sub[sub.policy != "model"])
            rm = sub[sub.policy == "model"].reward.mean()
            rb = sub[sub.policy != "model"].reward.mean()
            rep(
                f"  {lbl:>9} {_fmt(sm):>10} {_fmt(sb):>10} {_fmt(rm,2):>9} {_fmt(rb,2):>9}"
            )

    # ---- write report + a tidy per-class CSV ----
    with open(os.path.join(run, "report.txt"), "w") as f:
        f.write("\n".join(L))
    recs = []
    for cls in df.cls.unique():
        for name, red, _dir in _QOS_METRICS + _ENERGY_METRICS:
            if name in [m[0] for m in _ENERGY_METRICS] and cls != "REHD":
                continue
            r = _paired(df, red, cls=cls, better=_dir)
            if r:
                recs.append(
                    dict(
                        device_class=cls,
                        metric=name,
                        model=r["model"],
                        baseline=r["base"],
                        delta=r["model"] - r["base"],
                        win_rate=r["win"],
                        p=r["p"],
                        n=r["n"],
                    )
                )
    pd.DataFrame(recs).to_csv(
        os.path.join(run, "per_class_comparison.csv"), index=False
    )
    rep(f"\n[report] wrote report.txt + per_class_comparison.csv -> {run}")
    return "\n".join(L)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: twt_eval_report.py <eval_struct_run_dir>")
    generate(sys.argv[1])


if __name__ == "__main__":
    main()
