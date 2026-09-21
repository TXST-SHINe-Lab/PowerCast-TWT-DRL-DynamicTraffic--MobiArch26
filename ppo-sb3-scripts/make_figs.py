#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""make_figs.py — generate ALL data-driven paper figures in one place.

Run:  python3.11 make_figs.py                       (latest finished run -> <run>/figs/)
      RUN=repro_20260825 STAGES="4" bash reproduce.sh (same, for a named run)

Figures produced:
  harvest_curve.png  — PowerCast datasheet RF-to-DC efficiency + harvested power vs distance (analytical)
  vcap.png           — REHD capacitor V_cap vs time by distance, 0.7 V class (from vcap_data.csv)
  eval_perclass.png  — DRL vs M/D/1: per-class served/drop + alpha-fair reward/fairness vs #harvesters
  training_curve.png — LSTM-PPO + asymmetric-critic mean episode reward over training

Also prints (no figure) REHD sustainability numbers (harvested/consumed/SoC/served-per-joule,
DRL vs M/D/1) used as the paper's in-text T1/T2 figures.

NOT generated here (hand-drawn vector art, kept as-is): system_model.jpg, beacon_interval.jpg.

Data sources (each overridable by environment variable):
  PAPER_RUN_DIR    : run folder results/runs/<run>/; default = the run with the newest pipeline/eval/report.txt
  PAPER_FIGS_DIR   : output directory; default <run>/figs/
  PAPER_EVAL_DIR   : eval shards (parquet) of the DRL-vs-M/D/1 structured eval; default <run>/pipeline/eval/shards/
  PAPER_TRAIN_GLOB : training_log.jsonl file(s), stitched in order (handles resume); default <run>/pipeline/train/*/training_log.jsonl
  PAPER_VCAP_DATA  : extracted REHD V_cap traces for vcap.png; default <run>/figs/vcap_data.csv, else the committed results/figs/vcap_data.csv
Each figure is independent: a missing data source skips only that figure.
With the committed run the four PNGs are byte-identical to the paper's.
"""

import os
import glob
import json
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RESULTS = os.path.join(PROJ, "results")


def _latest_run():
    """Run folder under results/runs/ whose eval report was written last, or "" if none."""
    reports = glob.glob(
        os.path.join(RESULTS, "runs", "*", "pipeline", "eval", "report.txt")
    )
    if not reports:
        return ""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(max(reports, key=os.path.getmtime)))
    )


# --- data sources ---
# Default to the latest finished run; reproduce.sh stage 4 sets every variable explicitly for the run it just produced.
_RUN = os.environ.get("PAPER_RUN_DIR") or _latest_run()
FIGS = os.environ.get(
    "PAPER_FIGS_DIR",
    os.path.join(_RUN, "figs") if _RUN else os.path.join(RESULTS, "figs"),
)
os.makedirs(FIGS, exist_ok=True)
EVAL_DIR = os.environ.get(
    "PAPER_EVAL_DIR", os.path.join(_RUN, "pipeline", "eval", "shards")
)
TRAIN_DIRS = sorted(
    glob.glob(
        os.environ.get(
            "PAPER_TRAIN_GLOB",
            os.path.join(_RUN, "pipeline", "train", "*", "training_log.jsonl"),
        )
    )
)
# A run without its own stage-3 traces (e.g. the committed repro_20260825) falls back to the committed CSV.
_VCAP_RUN = os.path.join(FIGS, "vcap_data.csv")
VCAP_DATA = os.environ.get(
    "PAPER_VCAP_DATA",
    (
        _VCAP_RUN
        if os.path.exists(_VCAP_RUN)
        else os.path.join(RESULTS, "figs", "vcap_data.csv")
    ),
)
# Episodes per batch for the training-curve x-axis.
# This is `--episodes-per-batch`, NOT `--num-workers`: a pool of W workers can be driven through more than W episodes per batch (the shipped run is 16 workers x 24 episodes).
# It used to be hardcoded to 16, which plotted the 200x24 = 4800-episode run as if it were 3200.
# Read it per run from the hyperparams.json written beside each training_log.jsonl; EPB_FALLBACK only applies to a log with no hyperparams.json next to it.
EPB_FALLBACK = int(os.environ.get("PAPER_EPB", 24))


def _episodes_per_batch(log_path):
    """`episodes_per_batch` from the hyperparams.json beside this training log."""
    hp = os.path.join(os.path.dirname(log_path), "hyperparams.json")
    try:
        with open(hp) as f:
            v = int(json.load(f).get("episodes_per_batch") or 0)
        if v > 0:
            return v
    except Exception:
        pass
    return EPB_FALLBACK


# --- shared style ---
C_MODEL, C_BASE = "#1f77b4", "#d62728"  # DRL / M/D/1
DEV = {0: "IoT", 1: "Camera", 2: "Voice", 3: "Video", 4: "REHD"}
ORDER = ["IoT", "Camera", "Voice", "Video", "REHD"]
EPS = 1e-9


# --- 1. harvest_curve.png — PowerCast P21XXCSR-EVB Band-6 (2.45 GHz) datasheet model ---
def harvest_curve():
    BOOST, FSPL_1M, RXGAIN, FLOOR = 0.85, 40.05, 6.0, -12.0
    eta_dbm = np.array(
        [
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            16,
        ]
    )
    eta_pct = np.array(
        [
            0,
            2,
            10,
            21,
            31,
            37,
            39.5,
            40,
            40.2,
            41,
            42,
            42.3,
            42.6,
            42.9,
            43.3,
            43.8,
            44.3,
            44.9,
            45.6,
            46,
            46.3,
            46.3,
        ]
    )

    def eta(p):
        p = np.asarray(p, float)
        return np.where(p < FLOOR, 0.0, np.interp(p, eta_dbm, eta_pct) / 100.0)

    def p_rx(d, eirp=36.0):
        return eirp - (FSPL_1M + 20.0 * np.log10(d)) + RXGAIN

    def p_dc_mw(prx):
        return 10.0 ** (prx / 10.0) * eta(prx) * BOOST

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    p = np.linspace(-16, 16, 2000)
    ax1.plot(p, 100 * eta(p), color="tab:blue", lw=2.6, label="2450 MHz (datasheet)")
    ax1.axvline(FLOOR, color="red", ls="--", lw=1.2, label="$-12$ dBm floor")
    ax1.axvspan(
        p_rx(4.75),
        p_rx(1.5),
        color="tab:orange",
        alpha=0.18,
        label="in-room [1.5,4.75] m",
    )
    ax1.set_xlabel("received power $P_{rx}$ (dBm)")
    ax1.set_ylabel("RF-to-DC efficiency η (%)")
    ax1.set_title("(a) Band-6 efficiency")
    ax1.set_xlim(-16, 5)
    ax1.set_ylim(0, 50)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc="lower right")

    d = np.linspace(0.3, 12, 2000)
    pw = p_dc_mw(p_rx(d)) * 1000.0
    pw = np.where(pw > 0, pw, np.nan)
    ax2.plot(d, pw, color="tab:orange", lw=2.6, label="36 EIRP (30+6 dBi)")
    ax2.axvspan(1.5, 4.75, color="gray", alpha=0.12, label="annulus [1.5,4.75] m")
    d_floor = 10 ** ((36.0 + RXGAIN - FSPL_1M - FLOOR) / 20.0)
    ax2.axvline(
        d_floor, color="red", ls="--", lw=1.0, label=f"$-12$ floor @ {d_floor:.0f} m"
    )
    ax2.set_xlabel("distance from AP (m)")
    ax2.set_ylabel("harvested DC power (µW)")
    ax2.set_title("(b) Harvested power")
    ax2.set_yscale("log")
    ax2.set_xlim(0.3, 7)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=12, loc="lower left")
    fig.tight_layout(pad=0.6, w_pad=1.2)
    fig.savefig(os.path.join(FIGS, "harvest_curve.png"), dpi=150)
    plt.close(fig)
    print("wrote harvest_curve.png")


# --- 2. vcap.png — REHD capacitor V_cap vs time by distance (0.7 V class) ---
# Source: figs/vcap_data.csv (dist_m,time_s,vcap_v) — 4 REHDs, closest 2 / farthest 2.
def vcap():
    if not os.path.exists(VCAP_DATA):
        print(f"skip vcap.png (no {VCAP_DATA})")
        return
    VMIN, VMAX, T0, T1 = 0.64, 0.738, 9.0, 13.0
    series = {}
    for r in csv.DictReader(open(VCAP_DATA)):
        d = float(r["dist_m"])
        series.setdefault(d, []).append((float(r["time_s"]), float(r["vcap_v"])))
    dists = sorted(series)
    cols = [
        "#6a3d9a",
        "#33a02c",
        "#e31a1c",
        "#1f78b4",
    ]  # near purple, mid green, far red/blue
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for d, c in zip(dists, cols):
        tv = [(t, v) for t, v in series[d] if T0 <= t <= T1]
        ax.plot(
            [t for t, _ in tv],
            [v for _, v in tv],
            lw=1.8,
            color=c,
            marker=".",
            ms=3,
            label=f"$d={d:.1f}$ m",
        )
    ax.axhline(VMIN, color="k", ls=":", lw=1.1)
    ax.axhline(VMAX, color="k", ls=":", lw=1.1)
    ax.text(T1, VMIN, " $V_{\\min}$", va="center", fontsize=13)
    ax.text(T1, VMAX, " $V_{\\max}$", va="center", fontsize=13)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("capacitor voltage $V_{cap}$ (V)")
    ax.set_title("REHD harvest via PDW, 0.7 V class (closest 2 / farthest 2)")
    ax.set_xlim(T0, T1)
    ax.set_ylim(VMIN - 0.006, VMAX + 0.008)
    ax.grid(True, alpha=0.3)
    ax.legend(title="distance from AP", ncol=2, loc="center left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "vcap.png"), dpi=150)
    plt.close(fig)
    print("wrote vcap.png")


# --- 3. eval_perclass.png — DRL vs M/D/1, 4-panel: per-class served/drop + reward/fairness vs #harvesters ---
def eval_perclass():
    shards = sorted(glob.glob(os.path.join(EVAL_DIR, "*.parquet")))
    if not shards:
        print(f"skip eval_perclass.png (no shards in {EVAL_DIR})")
        return
    import pandas as pd

    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df["cls"] = df["device_class"].map(DEV)
    m, b = df[df.policy == "model"], df[df.policy != "model"]
    DRL, MD1 = "DRL (ours)", "M/D/1"

    def served(g):
        return min(1.0, g.d_tx.sum() / max(g.d_eq.sum(), EPS))

    def drop(g):
        return min(1.0, g.d_drops.sum() / max(g.d_eq.sum(), EPS))

    def jain(dd, nr):
        js = []
        for _, sc in dd[dd.n_rehd == nr].groupby("rand_seed"):
            per = (
                sc.groupby("sta_id")
                .apply(lambda s: s.d_tx.sum() / max(s.d_eq.sum(), EPS))
                .values
            )
            if len(per) and (per**2).sum() > 0:
                js.append(per.sum() ** 2 / (len(per) * (per**2).sum()))
        return float(np.mean(js)) if js else np.nan

    sp = sorted(df.n_rehd.unique())
    # The sizes the paper figure was rendered with (they used to leak in from harvest_curve's rcParams).
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
    x = np.arange(len(ORDER))
    w = 0.38
    ax[0, 0].bar(
        x - w / 2, [served(m[m.cls == c]) for c in ORDER], w, label=DRL, color=C_MODEL
    )
    ax[0, 0].bar(
        x + w / 2, [served(b[b.cls == c]) for c in ORDER], w, label=MD1, color=C_BASE
    )
    ax[0, 0].set_title("(a) Served ratio by class")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(ORDER)
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].legend(loc="lower left", fontsize=8)
    ax[0, 1].bar(x - w / 2, [drop(m[m.cls == c]) for c in ORDER], w, color=C_MODEL)
    ax[0, 1].bar(x + w / 2, [drop(b[b.cls == c]) for c in ORDER], w, color=C_BASE)
    ax[0, 1].set_title("(b) Drop/expiry by class")
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(ORDER)
    xs = np.arange(len(sp))
    ax[1, 0].bar(
        xs - w / 2,
        [m[m.n_rehd == nr].reward.mean() for nr in sp],
        w,
        color=C_MODEL,
        label=DRL,
    )
    ax[1, 0].bar(
        xs + w / 2,
        [b[b.n_rehd == nr].reward.mean() for nr in sp],
        w,
        color=C_BASE,
        label=MD1,
    )
    ax[1, 0].set_title(r"(c) $\alpha$-fair reward")
    ax[1, 0].set_xlabel("# harvesters (of 20)")
    ax[1, 0].set_ylabel("Mean step reward")
    ax[1, 0].set_xticks(xs)
    ax[1, 0].set_xticklabels(sp)
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].bar(xs - w / 2, [jain(m, nr) for nr in sp], w, color=C_MODEL, label=DRL)
    ax[1, 1].bar(xs + w / 2, [jain(b, nr) for nr in sp], w, color=C_BASE, label=MD1)
    ax[1, 1].set_title("(d) Fairness (Jain)")
    ax[1, 1].set_xlabel("# harvesters (of 20)")
    ax[1, 1].set_ylabel("Jain index")
    ax[1, 1].set_xticks(xs)
    ax[1, 1].set_xticklabels(sp)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "eval_perclass.png"))
    plt.close(fig)
    print("wrote eval_perclass.png")


# --- 3b. REHD sustainability numbers (T1/T2 in the paper text) — console only, no figure. ---
# Ported from the retired _gen_md1_figs.py so this number set has one home.
def print_rehd_sustainability():
    shards = sorted(glob.glob(os.path.join(EVAL_DIR, "*.parquet")))
    if not shards:
        print("skip REHD sustainability numbers (no eval shards)")
        return
    import pandas as pd

    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df["cls"] = df["device_class"].map(DEV)
    m, b = df[df.policy == "model"], df[df.policy != "model"]
    mr, br = m[m.cls == "REHD"], b[b.cls == "REHD"]

    # Per-REHD-episode normalizer.
    # The divisor used to be
    #   sta_id.nunique() * rand_seed.nunique()
    # i.e. the UNION of REHD indices over every split (16, since ids 4..19 each appear somewhere) times the scenario count -- but a scenario with n_rehd=4 contributes 4 harvesters, not 16.
    # Pooled over splits {4,8,12,16} that over-divides by 1280/800 = 1.6x, so every absolute mJ/STA figure came out at 0.625x its true value (ratios were unaffected).
    # Count the actual (scenario, REHD) pairs instead.
    def _rehd_episodes(g):
        return max(int(g.groupby("rand_seed").sta_id.nunique().sum()), 1)

    def served(g):
        return min(1.0, g.d_tx.sum() / max(g.d_eq.sum(), EPS))

    def cons_j(g):
        return g.d_cons_j.sum() / _rehd_episodes(g)

    def harv_mj(g):
        return g.d_harv_j.sum() / _rehd_episodes(g) * 1e3

    def cons_mj(g):
        return cons_j(g) * 1e3

    def soc(g):
        return g.soc.mean()

    def served_per_j(g):
        return served(g) / max(cons_j(g), EPS)  # demand satisfied per joule

    def autonomy(g):
        return g.d_harv_j.sum() / max(g.d_cons_j.sum(), EPS)

    # Policy-INVARIANT counterparts: d_eq is a MAC ADMISSION counter that shrinks when a blocked queue back-pressures the app, and d_tx counts PHY TX starts incl. retries, so served() flatters a policy that admits less.
    # d_gen/d_rx_ap move with neither.
    def delivered(g):
        return (
            min(1.0, g.d_rx_ap.sum() / max(g.d_gen.sum(), EPS))
            if "d_gen" in g
            else float("nan")
        )

    def delivered_per_j(g):
        return delivered(g) / max(cons_j(g), EPS)

    print("\n===== REHD sustainability: DRL vs M/D/1 (paper tab numbers) =====")
    print(
        f"  harvested (mJ/STA)        : DRL {harv_mj(mr):6.3f}  M/D/1 {harv_mj(br):6.3f}   "
        f"({harv_mj(mr) / max(harv_mj(br), EPS):.2f}x)"
    )
    print(
        f"  consumed  (mJ/STA)        : DRL {cons_mj(mr):6.1f}  M/D/1 {cons_mj(br):6.1f}   "
        f"({cons_mj(br) / cons_mj(mr):.1f}x less)"
    )
    print(
        f"  REHD served               : DRL {served(mr):6.3f}  M/D/1 {served(br):6.3f}"
    )
    print(f"  SoC                       : DRL {soc(mr):6.3f}  M/D/1 {soc(br):6.3f}")
    print(
        f"  autonomy H/C              : DRL {autonomy(mr):6.3f}  M/D/1 {autonomy(br):6.3f}   "
        f"({autonomy(mr) / max(autonomy(br), EPS):.1f}x)"
    )
    print(
        f"  >> demand served / joule  : DRL {served_per_j(mr):6.2f}  M/D/1 {served_per_j(br):6.2f}   "
        f"({served_per_j(mr) / max(served_per_j(br), EPS):.1f}x)"
    )
    print(
        f"  delivered ratio (invariant): DRL {delivered(mr):6.3f}  M/D/1 {delivered(br):6.3f}"
    )
    print(
        f"  >> delivered / joule      : DRL {delivered_per_j(mr):6.2f}  M/D/1 {delivered_per_j(br):6.2f}   "
        f"({delivered_per_j(mr) / max(delivered_per_j(br), EPS):.1f}x)"
    )
    print(
        f"  REHD bytes / STA-step     : DRL {mr.d_bytes.sum()/max(len(mr),1):8.1f}  "
        f"M/D/1 {br.d_bytes.sum()/max(len(br),1):8.1f}"
    )


# --- 4. training_curve.png — LSTM-PPO + asymmetric critic, reward over training ---
# Stitches TRAIN_DIRS in order (resume boundary continuous, unmarked).
def training_curve():
    if not TRAIN_DIRS:
        print("skip training_curve.png (no training_log.jsonl)")
        return
    rew, xe, done_eps = [], [], 0
    for p in TRAIN_DIRS:
        epb = _episodes_per_batch(p)  # per-run, so stitched resumes stay honest
        for l in open(p):
            if not l.strip():
                continue
            rew.append(json.loads(l)["ep_reward_mean"])
            xe.append(done_eps)  # episodes completed BEFORE this batch
            done_eps += epb
    if not rew:
        print("skip training_curve.png (training log has no ep_reward_mean rows)")
        return
    xe = np.asarray(xe)
    # Moving-average window must fit the series: a short run (smoke, or a job that died early) otherwise produced an empty x against a non-empty y and crashed the LAST stage of the pipeline.
    k = max(1, min(5, len(rew)))
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.plot(xe, rew, color=C_MODEL, alpha=0.25, lw=1, label="per-batch")
    sm = np.convolve(rew, np.ones(k) / k, mode="valid")
    if k > 1:
        ax.plot(xe[k - 1 :], sm, color=C_MODEL, lw=2, label=f"{k}-batch moving avg")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean episode reward")
    ax.set_title(r"LSTM-PPO + asymmetric critic ($\alpha$-fair objective)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "training_curve.png"))
    plt.close(fig)
    print(
        f"wrote training_curve.png ({len(rew)} batches, reward {rew[0]:.1f} -> {rew[-1]:.1f})"
    )


if __name__ == "__main__":
    for fn in (
        harvest_curve,
        vcap,
        eval_perclass,
        print_rehd_sustainability,
        training_curve,
    ):
        # Every figure starts from matplotlib defaults; rcParams set by one figure must not restyle the next.
        plt.rcdefaults()
        try:
            fn()
        except Exception as e:
            print(f"FAILED {fn.__name__}: {e}")
    print("run     ->", _RUN or "(none found)")
    print("figures ->", FIGS)
