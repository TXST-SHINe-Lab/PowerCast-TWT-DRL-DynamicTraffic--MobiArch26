#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""EDA distribution analysis for a twt_eda_collect.py dataset.

Reads the parquet shards and reports the signal health that matters before
training: (A) obs-feature health (saturation / dead / range — model inputs),
(B) reward-component dynamic range (does each term actually move? esp. the QoEH
SoC bucket), (C) REHD energy distributions (SoC stressed or always-high?),
(D) REHD vs non-REHD separation on key signals, (E) energy<->QoS tradeoff vs
schedule duty, (F) dead/constant raw columns. Then COMPLIANCE (metric-level,
full sweep): (G) PDW sweep — harvest scales/dozes with D + D-vs-traffic pricing,
(H) duty compliance (observed awake-fraction vs scheduled wake/BI), (I) offline
schedule validity for every (sched x pdw) action. Saves an overview PNG if
matplotlib is present.

  python3.11 twt_eda_analyze.py --data runs/eda_50b_12s8r/shards
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

OBS = [
    "bsr",
    "airtime",
    "fcs",
    "rxfrag",
    "silence",
    "awake",
    "sleep",
    "duty",
    "pktstx",
    "dAir",
    "dAwk",
    "dSlp",
    "dPkt",
    "dBytSP",
    "dPktSP",
    "dSPdone",
    "dSPdem",
    "dSPstarv",
]  # 18 feat (DL deltas dropped 2026-05-31)
RWD = [
    "rwd_w_pf_thr",
    "rwd_w_pf_drop",
    "rwd_w_pf_bsr",
    "rwd_w_max_starv",
    "rwd_w_energy",
    "rwd_w_airtime",
]  # SoC bucket (w_soc/w_efloor) removed


def load(data_dir):
    fs = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not fs:
        sys.exit(f"no parquet shards in {data_dir}")
    print(f"[eda] loading {len(fs)} shards ...", flush=True)
    df = pd.concat((pd.read_parquet(f) for f in fs), ignore_index=True)
    print(f"[eda] {len(df):,} rows x {df.shape[1]} cols", flush=True)
    return df


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir of parquet shards")
    ap.add_argument(
        "--png", default=None, help="overview plot path (default alongside data)"
    )
    args = ap.parse_args()
    df = load(args.data)
    rehd = df[df.is_rehd]
    nonr = df[~df.is_rehd]

    sec("DATASET")
    print(
        f"episodes={df.seed.nunique()}  updates/ep={df.update_idx.nunique()}  "
        f"STAs/update={df.groupby(['seed','update_idx']).size().iloc[0]}"
    )
    print(f"rows: total={len(df):,}  REHD={len(rehd):,}  non-REHD={len(nonr):,}")
    print(f"device_class mix: {df.groupby('device_class').size().to_dict()}  (4=REHD)")
    print(
        f"(sched,assign) combos covered: {len(df[['sched','assign']].drop_duplicates())}/1600  "
        f"sched {df.sched.min()}-{df.sched.max()}  assign {df['assign'].min()}-{df['assign'].max()}"
    )

    # (A) obs-feature health -------------------------------------------------
    sec("(A) OBS FEATURE HEALTH (model inputs; all should be in [0,1])")
    rows = []
    for nm in OBS:
        c = df[f"obs_{nm}"]
        rows.append(
            {
                "feature": nm,
                "mean": c.mean(),
                "std": c.std(),
                "p50": c.median(),
                "p99": c.quantile(0.99),
                "max": c.max(),
                "%=0": 100 * (c.abs() < 1e-9).mean(),
                "%sat(>0.99)": 100 * (c > 0.99).mean(),
            }
        )
    h = pd.DataFrame(rows).set_index("feature")
    print(h.round(4).to_string())
    dead = h.index[(h["std"] < 1e-6)].tolist()
    sat = h.index[(h["%sat(>0.99)"] > 50)].tolist()
    nearzero = h.index[(h["mean"] < 0.01) & (h["std"] < 0.01)].tolist()
    print(f"\n  DEAD/constant: {dead or 'none'}")
    print(f"  >50% saturated at 1.0: {sat or 'none'}")
    print(f"  near-zero (mean<.01,std<.01): {nearzero or 'none'}")

    # (B) reward component dynamic range -------------------------------------
    sec("(B) REWARD COMPONENT DYNAMIC RANGE (does each term move / teach?)")
    rows = []
    for c in RWD + ["reward_total"]:
        s = df[c]
        rows.append(
            {
                "component": c,
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "max": s.max(),
                "range": s.max() - s.min(),
                "%==0": 100 * (s.abs() < 1e-9).mean(),
            }
        )
    print(pd.DataFrame(rows).set_index("component").round(4).to_string())
    print(
        "  (the QoEH SoC bucket was removed — SoC uncontrollable; v1 QoS terms only.)"
    )

    # (C) REHD energy distribution -------------------------------------------
    sec("(C) REHD ENERGY (SoC stressed, or always-high?)")
    soc = rehd.soc
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print("SoC percentiles:", {f"p{q}": round(np.percentile(soc, q), 3) for q in qs})
    print(
        f"SoC mean={soc.mean():.3f} std={soc.std():.3f}  "
        f"frac<0.5={100*(soc<0.5).mean():.1f}%  frac<0.85(=Vmin-ish)={100*(soc<0.85).mean():.1f}%  "
        f"frac>=1.0={100*(soc>=1.0).mean():.1f}%"
    )
    print(
        f"per-update Δharvested_j: mean={rehd.d_delta_harvested_j.mean():.3e}  "
        f"Δconsumed_j: mean={rehd.d_delta_consumed_j.mean():.3e}  "
        f"harvest/consume ratio={rehd.d_delta_harvested_j.sum()/max(rehd.d_delta_consumed_j.sum(),1e-12):.3f}"
    )
    print(
        f"unpowered_tx_events (cum, final/ep mean): "
        f"{rehd.groupby('seed').orc_unpowered_tx_events.max().mean():.1f}"
    )

    # (D) REHD vs non-REHD separation ----------------------------------------
    sec("(D) REHD vs NON-REHD on key signals (mean [p50])")
    keys = [
        "soc",
        "served_ratio",
        "drop_rate",
        "bsr_fill",
        "obs_dSPstarv",
        "obs_silence",
        "obs_duty",
    ]
    cmp = pd.DataFrame(
        {
            "REHD": {k: f"{rehd[k].mean():.3f} [{rehd[k].median():.3f}]" for k in keys},
            "nonREHD": {
                k: f"{nonr[k].mean():.3f} [{nonr[k].median():.3f}]" for k in keys
            },
        }
    )
    print(cmp.to_string())

    # (E) energy<->QoS tradeoff vs schedule duty -----------------------------
    sec("(E) ENERGY<->QoS TRADEOFF vs nominal duty (binned)")
    df["_duty_bin"] = pd.cut(
        df.nominal_duty,
        [0, 0.15, 0.35, 0.6, 1.01],
        labels=["<0.15", "0.15-0.35", "0.35-0.6", ">0.6"],
    )
    g = df.groupby("_duty_bin", observed=True).agg(
        n=("soc", "size"),
        served=("served_ratio", "mean"),
        drop=("drop_rate", "mean"),
        bsr=("bsr_fill", "mean"),
        starv=("obs_dSPstarv", "mean"),
        rehd_soc=(
            "soc",
            lambda s: df.loc[s.index][df.is_rehd].soc.mean() if False else np.nan,
        ),
    )
    # rehd_soc per bin computed cleanly:
    g["rehd_soc"] = df[df.is_rehd].groupby("_duty_bin", observed=True).soc.mean()
    g["reward"] = df.groupby("_duty_bin", observed=True).reward_total.mean()
    print(g.round(3).to_string())

    # (F) dead/constant raw columns ------------------------------------------
    sec("(F) DEAD / CONSTANT raw columns (real_* / orc_*)")
    raws = [c for c in df.columns if c.startswith(("real_", "orc_"))]
    deadraw = [c for c in raws if df[c].abs().max() < 1e-12]
    constraw = [c for c in raws if df[c].std() < 1e-12 and df[c].abs().max() >= 1e-12]
    print(f"all-zero ({len(deadraw)}): {deadraw}")
    print(f"constant-nonzero ({len(constraw)}): {constraw}")

    # --- COMPLIANCE CHECKS (metric-level, full sweep) ---
    # PHY-sleep-edge TWT compliance is proven separately at D=45/50/75 via _pdw_compliance_run.
    # These confirm, across EVERY swept config, that (G) REHDs doze + harvest through the PDW and D is priced by traffic, (H) STAs don't wake beyond their scheduled SP, and (I) every (sched x pdw) action is timing-valid.

    # (G) PDW sweep: does D control harvest, and is it well-priced by traffic?
    sec(
        "(G) PDW SWEEP — harvest & traffic vs D (D controls harvest? priced by traffic?)"
    )
    lv = sorted(df.pdw_ms.unique())
    piv = pd.DataFrame(index=pd.Index(lv, name="pdw_ms"))
    piv["n"] = df.groupby("pdw_ms").size()
    piv["rehd_harv_j"] = rehd.groupby("pdw_ms").d_delta_harvested_j.mean()
    piv["rehd_served"] = rehd.groupby("pdw_ms").served_ratio.mean()
    piv["rehd_drop"] = rehd.groupby("pdw_ms").drop_rate.mean()
    piv["rehd_soc"] = rehd.groupby("pdw_ms").soc.mean()
    piv["nonr_served"] = nonr.groupby("pdw_ms").served_ratio.mean()
    piv["nonr_drop"] = nonr.groupby("pdw_ms").drop_rate.mean()
    piv["nonr_bsr"] = nonr.groupby("pdw_ms").bsr_fill.mean()
    print(piv.round(4).to_string())
    hv = piv["rehd_harv_j"].values
    ups = sum(hv[i + 1] >= hv[i] - 1e-12 for i in range(len(hv) - 1))
    h0, h1 = float(hv[0]), float(hv[-1])
    harv_pos = (
        (rehd[rehd.pdw_ms > 5].d_delta_harvested_j > 0).mean()
        if (rehd.pdw_ms > 5).any()
        else 0.0
    )
    print(
        f"\n  [harvest doze] REHD Δharvest monotonic↑ in D: {ups}/{len(hv) - 1} steps up; "
        f"D={lv[0]:.0f}->{lv[-1]:.0f}ms = {h0:.2e}->{h1:.2e} J ({h1 / max(h0, 1e-12):.1f}x). "
        f"REHD harvest>0 for D>5: {100 * harv_pos:.1f}%  "
        f"=> {'OK (doze+harvest across whole sweep)' if (h1 > h0 and harv_pos > 0.99) else 'CHECK'}"
    )
    print(
        f"  [pricing] REHD served vs D: {piv.rehd_served.iloc[0]:.3f} -> {piv.rehd_served.iloc[-1]:.3f}  "
        f"(does the lever lift REHD throughput?)"
    )
    print(
        f"  [pricing] nonREHD served vs D: {piv.nonr_served.iloc[0]:.3f} -> {piv.nonr_served.iloc[-1]:.3f}  "
        f"(should DROP at high D — big D steals comms airtime; this is how D 'pays' in v1)"
    )

    # (H) Duty compliance: observed awake-fraction vs the scheduled SP wake/BI.
    sec("(H) TWT DUTY COMPLIANCE — observed awake-fraction <= scheduled wake/BI?")
    sched_duty = df["wake_duration_ms"] / 102.4
    excess = df["obs_duty"] - sched_duty
    print(
        f"excess (obs_duty - scheduled wake/BI): mean={excess.mean():.3f} p50={excess.median():.3f} "
        f"p99={excess.quantile(0.99):.3f} max={excess.max():.3f}"
    )
    print(
        f"  rows awake >0.10 beyond schedule (over-wake): {100 * (excess > 0.10).mean():.2f}%  "
        f"(0% => no STA stays awake past its SP)"
    )
    byd = (
        df.assign(_ex=excess)
        .groupby("pdw_ms")
        ._ex.apply(lambda s: 100 * (s > 0.10).mean())
    )
    print("  %over-wake by pdw_ms:", {int(k): round(v, 2) for k, v in byd.items()})
    print(
        "  (proxy: obs_duty = AP-derived awake fraction; PHY-sleep-edge compliance proven via "
        "_pdw_compliance_run at D=45/50/75.)"
    )

    # (I) Schedule validity for the WHOLE action space (offline, decoder-only).
    sec(
        "(I) SCHEDULE VALIDITY — every (sched x pdw) config fits the BI? (offline, full action space)"
    )
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        from twt_spawn_worker import (
            apply_pdw_scale_shift,
            load_action_tables,
            NUM_PDW_LEVELS,
            PDW_MIN_WAKE_MS,
        )

        st, _ = load_action_tables(_proj)
        BI = 102.4
        worst_end = 0.0
        worst = None
        min_wake = 1e9
        bad = 0
        for idx in range(NUM_PDW_LEVELS):
            for si, sc in enumerate(st["schedules"]):
                gc = [
                    {
                        "group_id": g["group_id"],
                        "twt_wake_duration_ms": g["wake_duration_ms"],
                        "twt_sp_offset_ms": g["sp_offset_ms"],
                    }
                    for g in sc["groups"]
                ]
                apply_pdw_scale_shift(gc, idx)
                for c in gc:
                    e = c["twt_sp_offset_ms"] + c["twt_wake_duration_ms"]
                    if e > worst_end:
                        worst_end, worst = e, (idx, si)
                    min_wake = min(min_wake, c["twt_wake_duration_ms"])
                    if e >= BI or c["twt_wake_duration_ms"] < PDW_MIN_WAKE_MS - 1e-9:
                        bad += 1
        ncfg = NUM_PDW_LEVELS * len(st["schedules"])
        ok = worst_end < BI and min_wake >= PDW_MIN_WAKE_MS - 1e-9 and bad == 0
        print(
            f"worst SP end={worst_end:.2f} ms (BI cap {BI}; at idx{worst[0]}/sched{worst[1]})  "
            f"min wake={min_wake:.2f} ms (floor {PDW_MIN_WAKE_MS})  overflow/under-floor configs={bad}"
        )
        print(
            f"  -> all {ncfg} (sched x pdw) configs valid: {'YES' if ok else 'NO — INVESTIGATE'}"
        )
    except Exception as e:
        print(f"  (skipped offline schedule check: {e})")

    # overview PNG ------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        ax[0, 0].hist(rehd.soc, bins=60)
        ax[0, 0].set_title("REHD SoC")
        ax[0, 1].hist(df.reward_total, bins=80)
        ax[0, 1].set_title("reward_total")
        ax[0, 2].hist(rehd.obs_dSPstarv, bins=60, alpha=0.6, label="REHD")
        ax[0, 2].hist(nonr.obs_dSPstarv, bins=60, alpha=0.6, label="nonREHD")
        ax[0, 2].legend()
        ax[0, 2].set_title("obs_dSPstarv")
        ax[1, 0].hist(df.served_ratio, bins=60)
        ax[1, 0].set_title("served_ratio")
        ax[1, 1].scatter(df.nominal_duty, df.reward_total, s=1, alpha=0.02)
        ax[1, 1].set_title("reward vs duty")
        ax[1, 2].boxplot([h["std"].values])
        ax[1, 2].set_title("obs feature std spread")
        plt.tight_layout()
        png = args.png or os.path.join(
            os.path.dirname(args.data.rstrip("/")), "eda_overview.png"
        )
        plt.savefig(png, dpi=90)
        print(f"\n[eda] overview plot -> {png}")
    except Exception as e:
        print(f"\n[eda] (plot skipped: {e})")


if __name__ == "__main__":
    main()
