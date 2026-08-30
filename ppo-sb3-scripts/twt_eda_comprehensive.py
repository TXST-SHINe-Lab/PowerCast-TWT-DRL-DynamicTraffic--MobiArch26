#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""COMPREHENSIVE EDA — one analyzer over the controllability shards that captures
EVERY aspect of the signal feed before wiring obs/v4 + training.

Reads RAW realistic+oracle per-(seed,action,update,STA) rows (from
twt_eda_controllability.py, or any collector with the same schema) and reports:

  A. INVENTORY      — each raw signal: realistic/oracle, alive/dead/static/dynamic.
  B. DISTRIBUTION   — the FINAL time-local feed (obs + reward), per signal:
                      p50/p99/%zero/skew + the re-FIT transform/norm + a verdict.
                      (these fitted norms are what gets wired into the obs builder/reward.)
  C. CONTROLLABILITY— per ACTION HEAD (sched/assign/pdw), eta^2 of each feed signal,
                      split NETWORK-MEAN vs PER-STA. NS-3 is deterministic per
                      (scenario_seed, action), so within a seed all spread across a
                      head's levels is PURE action effect (zero scenario confound).
                      The split exposes the "mean cancels per-STA movement" trap.
  D. CLASS SEP      — per-device-class mean of each feed signal + REHD-vs-non Cohen's d.
  E. REDUNDANCY     — pairwise |corr| among obs feed signals (>=.95 => drop one).
  F. REWARD SWING   — per-STA utility q=served*(1-drop); aggregate by mean(a=0) /
                      PF(a=1) / a=2; RANGE of each aggregate across each head's levels.
                      Proves alpha-fair "swings with the schedule" where mean cancels.
  G. OBSERVABILITY  — corr(realistic proxy, oracle truth): is the POMDP observable?

Writes analysis_report.txt + EDA_FINDINGS.md + PNGs into the run dir.

  python3.11 .../ppo-sb3-scripts/twt_eda_comprehensive.py <run_dir>
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

EPS = 1e-9
STEP_US = 25 * 102.4 * 1000.0  # one update window in us (DURATION-update span proxy)
DEV_NAMES = {0: "IoT", 1: "Camera", 2: "Voice", 3: "Video", 4: "REHD", -1: "?"}


def _skew(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    m, s = x.mean(), x.std()
    return float(((x - m) ** 3).mean() / (s**3 + 1e-12)) if s > 0 else 0.0


def _log1p_norm(x, p99):
    return np.clip(np.log1p(np.clip(x, 0, None)) / np.log1p(max(p99, 1.0)), 0, 1)


def load_all(run):
    shards = sorted(glob.glob(os.path.join(run, "shards", "*.parquet")))
    if not shards:
        sys.exit(f"no shards in {run}/shards")
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    # twt_eda_collect.py names the scenario seed column "seed"; the controllability collector uses "scenario_seed".
    # Alias so this analyzer accepts either schema.
    if "scenario_seed" not in df.columns and "seed" in df.columns:
        df["scenario_seed"] = df["seed"]
    # Per-STA trajectory key, used below for per-step diffs.
    # This MUST identify one (episode, STA) trace.
    #
    # BUGFIX: the key used to be ["scenario_seed","sched","assign","pdw_idx","sta_id"], i.e. it included the ACTION.
    # That is only valid for a constant-action collector.
    # twt_eda_collect.py varies the action on EVERY step (both its grid mode and EDA_SPAN), so each group held exactly ONE row, .diff() was all-NaN, and every per-step delta collapsed to 0 -- which is why a full EDA report can come out reading "DEAD/sparse" and "FLAT/uncontrollable" for every signal with eta^2=0.00 and NaN correlations.
    # Both collectors emit a per-episode id ("g" from twt_eda_controllability, "global_idx" from twt_eda_collect), so key on that instead and treat the action as a factor, never as part of the identity.
    ep_col = next((c for c in ("g", "global_idx", "episode") if c in df.columns), None)
    if ep_col is None:  # last resort: assume one action per (seed, episode)
        ep_col = "scenario_seed"
        print(
            "[eda] WARNING: no per-episode id column; falling back to scenario_seed. "
            "If this run varied the action within an episode, deltas will be wrong."
        )
    key = [ep_col, "sta_id"]
    df = df.sort_values(key + ["update_idx"]).reset_index(drop=True)
    print(
        f"[eda] {len(shards)} shards, {len(df):,} rows, "
        f"{df['scenario_seed'].nunique()} seeds, "
        f"{df[['sched','assign','pdw_idx']].drop_duplicates().shape[0]} distinct actions"
    )
    return df, key


def add_deltas(df, key):
    g = df.groupby(key, sort=False)
    cum = [
        "real_packets_received_at_ap",
        "real_bytes_received_at_ap",
        "real_airtime_used_us",
        "real_fcs_error_count",
        "real_rx_fragment_count",
        "real_sp_with_demand_count",
        "real_sp_with_starvation_count",
        "real_sp_completed_count",
        "orc_packets_transmitted",
        "orc_packets_enqueued",
        "orc_mpdu_drops_expired",
        "orc_bytes_transmitted",
        "orc_awake_time_ms",
        "orc_sleep_time_ms",
        "orc_queue_delay_sum_ms",
        "orc_queue_delay_count",
        "orc_harvested_total_j",
        "orc_consumed_total_j",
    ]
    for c in cum:
        if c in df.columns:
            df["d_" + c] = np.clip(np.nan_to_num(g[c].diff().values), 0, None)
    return df


def build_feed(df, fit=True, norms=None):
    """Return (feed_df_cols_added, norms, meta). meta[name]=(realistic, rehd_only, transform)."""
    n = norms or {}
    d = lambda c: df["d_" + c].values if "d_" + c in df.columns else np.zeros(len(df))
    v = lambda c: (
        np.nan_to_num(df[c].values.astype(float))
        if c in df.columns
        else np.zeros(len(df))
    )
    meta = {}

    def fitp99(name, x):
        if fit:
            n[name] = float(np.nanpercentile(x[x > 0], 99)) if (x > 0).any() else 1.0
        return n.get(name, 1.0)

    # ---------- OBS (realistic, time-local) ----------
    bsr = v("real_bsr_queue_ac_be")
    p = fitp99("bsr_be", bsr)
    df["f_bsr_be"] = _log1p_norm(bsr, p)
    meta["bsr_be"] = (True, False, f"log1p/{p:.0f}")
    for fld, nm in [
        ("real_packets_received_at_ap", "dpkts_rx"),
        ("real_bytes_received_at_ap", "dbytes_rx"),
        ("real_airtime_used_us", "dairtime"),
        ("real_fcs_error_count", "dfcs"),
    ]:
        x = d(fld)
        p = fitp99(nm, x)
        df["f_" + nm] = _log1p_norm(x, p)
        meta[nm] = (True, False, f"d+log1p/{p:.0f}")
    snr = v("real_snr_db")
    if fit:
        nz = snr[snr > 0]
        n["snr_lo"] = float(np.nanpercentile(nz, 1)) if nz.size else 49.0
        n["snr_hi"] = float(np.nanpercentile(nz, 99)) if nz.size else 69.0
    lo, hi = n.get("snr_lo", 49.0), n.get("snr_hi", 69.0)
    df["f_snr"] = np.clip((snr - lo) / max(hi - lo, 1e-6), 0, 1)
    meta["snr"] = (True, False, f"lin[{lo:.0f},{hi:.0f}]")
    now = v("real_observation_time_ms") * 1000.0
    lr = v("real_last_rx_timestamp_us")
    df["f_silence"] = np.where(lr <= 0, 1.0, np.clip((now - lr) / STEP_US, 0, 1))
    meta["silence"] = (True, False, "recency/step")
    dem = d("real_sp_with_demand_count")
    stv = d("real_sp_with_starvation_count")
    df["f_starv_rate"] = np.clip(stv / np.maximum(dem, EPS), 0, 1)
    meta["starv_rate"] = (True, False, "ratio dStv/dDem")

    # ---------- REWARD (oracle ok, intrinsic ratios) ----------
    eq = d("orc_packets_enqueued")
    tx = d("orc_packets_transmitted")
    dr = d("orc_mpdu_drops_expired")
    df["f_served"] = np.where(eq > 0, np.clip(tx / np.maximum(eq, EPS), 0, 1), 1.0)
    meta["served"] = (False, False, "dTx/dEq")
    df["f_drop"] = np.where(eq > 0, np.clip(dr / np.maximum(eq, EPS), 0, 1), 0.0)
    meta["drop_rate"] = (False, False, "dDrop/dEq")
    qc = d("orc_queue_delay_count")
    qs = d("orc_queue_delay_sum_ms")
    lat = np.where(qc > 0, qs / np.maximum(qc, EPS), 0.0)
    p = fitp99("latency", lat)
    df["f_latency"] = np.clip(lat / max(p, 1.0), 0, 1)
    meta["latency"] = (False, False, f"dSum/dCnt /{p:.0f}ms")
    bl = v("orc_queue_size_packets")
    p = fitp99("backlog", bl)
    df["f_backlog"] = _log1p_norm(bl, p)
    meta["backlog"] = (False, False, f"log1p/{p:.0f}")
    aw = d("orc_awake_time_ms")
    sl = d("orc_sleep_time_ms")
    df["f_duty"] = np.where((aw + sl) > 0, aw / np.maximum(aw + sl, EPS), 0.0)
    meta["duty_step"] = (False, False, "dAw/(dAw+dSl)")
    hv = d("orc_harvested_total_j")
    cs = d("orc_consumed_total_j")
    isr = v("orc_vcap_max_v") > 0
    df["f_hc"] = np.where(isr & (cs > 0), np.clip(hv / np.maximum(cs, EPS), 0, 2), 0.0)
    meta["hc(REHD)"] = (False, True, "dHrv/dCons")
    vm = v("orc_vcap_max_v")
    df["f_soc"] = np.where(
        vm > 0, np.clip(v("orc_vcap_v") / np.maximum(vm, EPS), 0, 1), 0.0
    )
    meta["soc(REHD)"] = (False, True, "vcap/vmax")
    # per-STA utility (for reward swing): q = served*(1-drop)
    df["f_util"] = df["f_served"] * (1.0 - df["f_drop"])
    return df, n, meta


FEED_OBS = [
    "bsr_be",
    "dpkts_rx",
    "dbytes_rx",
    "dairtime",
    "dfcs",
    "snr",
    "silence",
    "starv_rate",
]
FEED_RWD = [
    "served",
    "drop_rate",
    "latency",
    "backlog",
    "duty_step",
    "hc(REHD)",
    "soc(REHD)",
]
FCOL = {
    "bsr_be": "f_bsr_be",
    "dpkts_rx": "f_dpkts_rx",
    "dbytes_rx": "f_dbytes_rx",
    "dairtime": "f_dairtime",
    "dfcs": "f_dfcs",
    "snr": "f_snr",
    "silence": "f_silence",
    "starv_rate": "f_starv_rate",
    "served": "f_served",
    "drop_rate": "f_drop",
    "latency": "f_latency",
    "backlog": "f_backlog",
    "duty_step": "f_duty",
    "hc(REHD)": "f_hc",
    "soc(REHD)": "f_soc",
}


def eta2_by_level(sub, valcol, levelcol):
    """eta^2 of valcol across the levels of levelcol, computed WITHIN each scenario_seed
    (NS-3 deterministic per (seed,action) => pure action effect), averaged over seeds.
    """
    vals = []
    for _, grp in sub.groupby("scenario_seed"):
        x = grp[valcol].values
        if x.std() < 1e-12 or grp[levelcol].nunique() < 2:
            vals.append(0.0)
            continue
        gm = grp.groupby(levelcol)[valcol].transform("mean")
        vals.append(float(((gm - x.mean()) ** 2).mean() / (x.var() + 1e-12)))
    return float(np.mean(vals)) if vals else 0.0


def section_controllability(sdf, meta, rep):
    """For each head, eta^2 (network-mean and per-STA) of each feed signal across that
    head's levels (others held at center). Uses the per-UPDATE steady-state samples so
    within-level variance = the across-update jitter at a FIXED action (the proper noise
    floor). Collapsing to one mean/episode would make eta^2 trivially 1.0."""
    center = {
        "sched": (CENTER["assign"], CENTER["pdw"]),
        "assign": (CENTER["sched"], CENTER["pdw"]),
        "pdw": (CENTER["sched"], CENTER["assign"]),
    }
    headcol = {"sched": "sched", "assign": "assign", "pdw": "pdw_idx"}
    other = {
        "sched": ("assign", "pdw_idx"),
        "assign": ("sched", "pdw_idx"),
        "pdw": ("sched", "assign"),
    }
    rep("\n" + "=" * 78)
    rep(
        "C. CONTROLLABILITY  (eta^2 across each head's levels, within-seed = pure action)"
    )
    rep(
        "   net = network-mean signal moved | sta = per-STA signal moved (the granular lever)"
    )
    rep(
        "   within-level noise = across-update jitter at a fixed action (steady-state window)"
    )
    rep("=" * 78)
    results = {}
    for grpname, feats in [
        ("OBS (realistic)", FEED_OBS),
        ("REWARD (oracle)", FEED_RWD),
    ]:
        rep(f"\n  -- {grpname} --")
        rep(
            f"  {'signal':12} {'transform':16} | "
            f"{'sched':>12} {'assign':>12} {'pdw':>12}  verdict"
        )
        rep(
            f"  {'':12} {'':16} | {'net  /  sta':>12} {'net  /  sta':>12} {'net  /  sta':>12}"
        )
        for nm in feats:
            col = FCOL[nm]
            real, rehd_only, tf = meta.get(nm, (False, False, "?"))
            base = sdf[sdf.is_rehd] if rehd_only else sdf
            line = {}
            best = 0.0
            for head in ["sched", "assign", "pdw"]:
                a_other, p_other = other[head]
                co = center[head]
                sub = base[(base[a_other] == co[0]) & (base[p_other] == co[1])]
                # network-mean per (seed, level, update): collapse STAs, keep update samples
                netmean = (
                    sub.groupby(["scenario_seed", headcol[head], "update_idx"])[col]
                    .mean()
                    .reset_index()
                )
                net_e = eta2_by_level(netmean, col, headcol[head])
                # per-STA: eta^2 per sta_id (across-update samples at each level), then mean
                ste = [
                    eta2_by_level(g, col, headcol[head])
                    for _, g in sub.groupby("sta_id")
                ]
                sta_e = float(np.mean(ste)) if ste else 0.0
                line[head] = (net_e, sta_e)
                best = max(best, net_e, sta_e)
            results[nm] = line
            vd = (
                "CONTROLLABLE"
                if best > 0.05
                else ("weak" if best > 0.01 else "FLAT/uncontrollable")
            )
            rep(
                f"  {nm:12} {tf:16} | "
                + " ".join(
                    f"{line[h][0]:5.2f}/{line[h][1]:5.2f}"
                    for h in ["sched", "assign", "pdw"]
                )
                + f"  {vd}"
            )
    return results


def section_distribution(df, meta, rep):
    rep("\n" + "=" * 78)
    rep(
        "B. DISTRIBUTION + TRANSFORM  (the FINAL time-local feed; norms re-fit on this scenario)"
    )
    rep("=" * 78)
    rep(
        f"  {'signal':12} {'kind':6} {'transform':16} {'p50':>6} {'p99':>6} {'%zero':>6} {'skew':>7}  verdict"
    )
    for grpname, feats in [("OBS", FEED_OBS), ("REWARD", FEED_RWD)]:
        rep(f"  -- {grpname} --")
        for nm in feats:
            col = FCOL[nm]
            real, rehd_only, tf = meta[nm]
            x = df[df.orc_vcap_max_v > 0][col].values if rehd_only else df[col].values
            x = x[np.isfinite(x)]
            p50, p99 = np.percentile(x, [50, 99])
            pz = float((x == 0).mean())
            sk = _skew(x)
            good = (
                (p99 <= 1.001) and (abs(sk) < 1.5) and (pz < 0.9) and (x.std() > 1e-6)
            )
            vd = (
                "GOOD"
                if good
                else ("DEAD/sparse" if (pz >= 0.9 or x.std() < 1e-6) else "skewed")
            )
            kind = "real" if real else "orac"
            rep(
                f"  {nm:12} {kind:6} {tf:16} {p50:6.2f} {p99:6.2f} {pz:6.2f} {sk:7.2f}  {vd}"
            )


def section_classsep(steady, meta, rep):
    rep("\n" + "=" * 78)
    rep(
        "D. CLASS SEPARATION  (per-device-class steady-state mean + REHD-vs-non Cohen's d)"
    )
    rep("=" * 78)
    classes = sorted(steady.device_class.unique())
    hdr = (
        "  "
        + f"{'signal':12} "
        + " ".join(f"{DEV_NAMES.get(c,c):>8}" for c in classes)
        + f"{'|d|REHD':>9}"
    )
    rep(hdr)
    for nm in FEED_OBS + FEED_RWD:
        col = FCOL[nm]
        means = [steady[steady.device_class == c][col].mean() for c in classes]
        xr = steady[steady.is_rehd][col].values
        xn = steady[~steady.is_rehd][col].values
        pooled = np.sqrt((np.nanvar(xr) + np.nanvar(xn)) / 2 + 1e-12)
        dco = abs(np.nanmean(xr) - np.nanmean(xn)) / pooled if pooled > 0 else 0.0
        rep("  " + f"{nm:12} " + " ".join(f"{m:8.3f}" for m in means) + f"{dco:9.2f}")


def section_redundancy(df, rep):
    rep("\n" + "=" * 78)
    rep("E. REDUNDANCY  (pairwise |corr| among OBS feed signals; >=.95 => drop one)")
    rep("=" * 78)
    cols = [FCOL[n] for n in FEED_OBS]
    n = len(df)
    idx = np.random.RandomState(0).choice(n, min(60000, n), replace=False)
    A = np.nan_to_num(np.column_stack([df[c].values[idx] for c in cols]))
    C = np.corrcoef(A, rowvar=False)
    np.fill_diagonal(C, 0.0)
    for i, nm in enumerate(FEED_OBS):
        j = int(np.argmax(np.abs(C[i])))
        flag = "  <-- REDUNDANT" if abs(C[i, j]) >= 0.95 else ""
        rep(f"  {nm:12} max|r|={abs(C[i,j]):.2f} with {FEED_OBS[j]}{flag}")


def section_reward_swing(steady, rep):
    """Per-STA utility q=served*(1-drop). Aggregate by mean(a=0)/PF(a=1)/a=2. For each head,
    RANGE of the aggregate across that head's levels (within seed, mean over seeds)."""
    rep("\n" + "=" * 78)
    rep("F. REWARD SWING  (does the aggregate MOVE with the schedule, or cancel?)")
    rep("   per-STA utility q=served*(1-drop); mean(a=0) cancels, PF/a=2 should swing")
    rep("=" * 78)
    headcol = {"sched": "sched", "assign": "assign", "pdw": "pdw_idx"}
    center = {
        "sched": ("assign", "pdw_idx", CENTER["assign"], CENTER["pdw"]),
        "assign": ("sched", "pdw_idx", CENTER["sched"], CENTER["pdw"]),
        "pdw": ("sched", "assign", CENTER["sched"], CENTER["assign"]),
    }

    def agg(qvals, alpha):
        q = np.clip(
            qvals, 1e-2, 1.0
        )  # floor avoids log/inverse blow-up on full starvation
        if alpha == 0:
            return float(np.mean(q))
        if abs(alpha - 1.0) < 1e-9:
            return float(np.mean(np.log(q)))  # PF / Nash
        return float(np.mean(q ** (1 - alpha) / (1 - alpha)))  # alpha-fair utility

    rep(f"  {'head':8} {'mean(a=0) range':>18} {'PF(a=1) range':>16} {'a=2 range':>14}")
    for head in ["sched", "assign", "pdw"]:
        oc, pc, ov, pv = center[head]
        sub = steady[(steady[oc] == ov) & (steady[pc] == pv)]
        ranges = {0: [], 1: [], 2: []}
        for _, gseed in sub.groupby("scenario_seed"):
            per_level = {0: [], 1: [], 2: []}
            for _, gl in gseed.groupby(headcol[head]):
                qv = gl["f_util"].values
                for a in (0, 1, 2):
                    per_level[a].append(agg(qv, a))
            for a in (0, 1, 2):
                if len(per_level[a]) >= 2:
                    ranges[a].append(max(per_level[a]) - min(per_level[a]))
        rep(
            f"  {head:8} {np.mean(ranges[0]):18.4f} {np.mean(ranges[1]):16.4f} {np.mean(ranges[2]):14.4f}"
        )
    rep(
        "  (PF/a=2 range >> mean range  =>  alpha-fair reward 'sees' the schedule; mean is blind.)"
    )

    # who gets served: per-class served vs each head (does scheduling reallocate service?)
    rep(
        "\n  per-class served_ratio swing across each head (max-min over levels, mean/seed):"
    )
    for head in ["sched", "assign", "pdw"]:
        oc, pc, ov, pv = center[head]
        sub = steady[(steady[oc] == ov) & (steady[pc] == pv)]
        parts = []
        for c in sorted(sub.device_class.unique()):
            cs = sub[sub.device_class == c]
            rr = []
            for _, gs in cs.groupby("scenario_seed"):
                lm = gs.groupby(headcol[head])["f_served"].mean()
                if len(lm) >= 2:
                    rr.append(lm.max() - lm.min())
            parts.append(
                f"{DEV_NAMES.get(c,c)}={np.mean(rr):.3f}"
                if rr
                else f"{DEV_NAMES.get(c,c)}=na"
            )
        rep(f"    {head:8} " + "  ".join(parts))


def section_observability(df, rep):
    rep("\n" + "=" * 78)
    rep(
        "G. OBSERVABILITY  (corr realistic-obs proxy vs oracle truth => POMDP observable?)"
    )
    rep("=" * 78)

    def corr(a, b):
        a = np.nan_to_num(df[a].values)
        b = np.nan_to_num(df[b].values)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    pairs = [
        ("f_dpkts_rx", "f_served", "Δpkts_rx (real) ~ served (oracle)"),
        ("f_bsr_be", "f_backlog", "bsr_be (real) ~ backlog (oracle)"),
        ("f_starv_rate", "f_drop", "starv_rate (real) ~ drop_rate (oracle)"),
        ("f_silence", "f_served", "silence (real) ~ served (oracle, expect -)"),
        ("f_dairtime", "f_duty", "Δairtime (real) ~ duty_step (oracle)"),
    ]
    for a, b, desc in pairs:
        rep(f"  r={corr(a,b):+.2f}   {desc}")


def section_inventory(df, rep):
    rep("=" * 78)
    rep("A. INVENTORY  (raw signals: realistic/oracle, alive/dead/static/dynamic)")
    rep("=" * 78)
    sig = [c for c in df.columns if c.startswith("real_") or c.startswith("orc_")]
    g = df.groupby(
        ["scenario_seed", "sta_id", "sched", "assign", "pdw_idx"], sort=False
    )
    dead = static = dyn = 0
    deadlist = []
    for c in sig:
        within = float(g[c].std().mean())
        total = float(np.nanstd(df[c].values))
        if total < 1e-12:
            dead += 1
            deadlist.append(c.replace("real_", "").replace("orc_", ""))
        elif within < 1e-9:
            static += 1
        else:
            dyn += 1
    rep(
        f"  {len(sig)} raw signals: {dyn} dynamic, {static} static(descriptor), {dead} dead"
    )
    if deadlist:
        rep(f"  DEAD: {deadlist}")


def make_plots(steady, ctl, run):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    # controllability heatmap (max of net/sta per head)
    feats = FEED_OBS + FEED_RWD
    M = np.array([[max(ctl[f][h]) for h in ["sched", "assign", "pdw"]] for f in feats])
    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.3, M.max()))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["sched", "assign", "pdw"])
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    for i in range(len(feats)):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{M[i,j]:.2f}",
                ha="center",
                va="center",
                color="w" if M[i, j] < 0.5 * M.max() else "k",
                fontsize=7,
            )
    ax.set_title("controllability eta^2 (max net/sta)")
    fig.colorbar(im, ax=ax, shrink=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(run, "controllability.png"), dpi=110)
    plt.close(fig)


CENTER = {"sched": 15, "assign": 6, "pdw": 9}


def main():
    run = (
        os.path.abspath(sys.argv[1])
        if len(sys.argv) > 1
        else os.path.abspath(open("/dev/shm/eda_ctl_outdir.txt").read().strip())
    )
    mpath = os.path.join(run, "manifest.json")
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        CENTER["sched"], CENTER["assign"], CENTER["pdw"] = m.get("center", [15, 6, 9])

    df, key = load_all(run)
    df = add_deltas(df, key)
    df, norms, meta = build_feed(df, fit=True)

    # steady state = last 30% of updates per episode
    umax = df.update_idx.max()
    thr = int(0.7 * umax)
    sdf = df[df.update_idx >= thr].copy()
    gcols = [
        "scenario_seed",
        "sched",
        "assign",
        "pdw_idx",
        "sta_id",
        "is_rehd",
        "device_class",
    ]
    fcols = [c for c in df.columns if c.startswith("f_")]
    steady = sdf.groupby(gcols)[fcols].mean().reset_index()
    print(
        f"[eda] steady-state: updates>={thr} ({umax+1} total), {len(steady)} episode-STA points"
    )

    lines = []
    rep = lambda s="": (lines.append(s), print(s))[0]
    rep(f"COMPREHENSIVE SIGNAL EDA  —  {run}")
    rep(
        f"rows={len(df):,}  steady-points={len(steady)}  center action=(s{CENTER['sched']},"
        f"a{CENTER['assign']},p{CENTER['pdw']})"
    )
    section_inventory(df, rep)
    section_distribution(df, meta, rep)
    ctl = section_controllability(sdf, meta, rep)
    section_classsep(steady, meta, rep)
    section_redundancy(df, rep)
    section_reward_swing(steady, rep)
    section_observability(df, rep)

    rep("\n" + "=" * 78)
    rep("RE-FIT NORM CONSTANTS (wire these into the obs builder / reward)")
    rep("=" * 78)
    for k, vv in norms.items():
        rep(f"  {k:14} = {vv:.4g}")

    with open(os.path.join(run, "analysis_report.txt"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(run, "norms_fitted.json"), "w") as f:
        json.dump(norms, f, indent=2)
    make_plots(steady, ctl, run)
    print(
        f"\n[eda] wrote analysis_report.txt + norms_fitted.json + controllability.png -> {run}"
    )


if __name__ == "__main__":
    main()
