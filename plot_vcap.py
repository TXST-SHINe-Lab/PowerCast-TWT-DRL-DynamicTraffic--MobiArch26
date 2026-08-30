#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
plot_vcap.py — plot per-REHD capacitor voltage Vcap(t) AND MAC queue depth from
the diagnostic CSV written by `twt-powercast-main-simulation --logVcap=true`.

CSV schema: time_ms,sta_id,vcap_v,vmin_v,vmax_v,phase[,queue_pkts]
  vmin = deep-discharge / TX-disable threshold; vmax = nominal full charge.
  phase: 0=charging, 1=active, 2=discharging, 3=protection.
  queue_pkts = MAC backlog (all ACs) — packets waiting to TX.

Outputs:
  vcap_all.png  — two stacked panels (Vcap top, queue bottom), all REHDs overlaid.
  vcap_grid.png — per-REHD panel: Vcap (left axis) + vmin/vmax + queue (right axis).
"""

import argparse
import os
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path):
    series = defaultdict(lambda: {"t": [], "vcap": [], "q": []})
    meta = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sid = int(row["sta_id"])
            series[sid]["t"].append(float(row["time_ms"]) / 1000.0)
            series[sid]["vcap"].append(float(row["vcap_v"]))
            series[sid]["q"].append(int(row.get("queue_pkts", 0) or 0))
            meta[sid] = (float(row["vmin_v"]), float(row["vmax_v"]))
    return series, meta


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "data-log/vcap_trace_7001.csv"))
    ap.add_argument("--out", default=os.path.join(here, "results", "data-log"))
    args = ap.parse_args()

    series, meta = load(args.csv)
    sids = sorted(series)
    os.makedirs(args.out, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    # --- Figure 1: stacked Vcap (top) + queue (bottom), shared time axis ---
    fig, (axv, axq) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for k, sid in enumerate(sids):
        vmin, vmax = meta[sid]
        c = cmap(k)
        axv.plot(
            series[sid]["t"],
            series[sid]["vcap"],
            color=c,
            lw=1.2,
            label=f"STA{sid} (vmin={vmin:.2f})",
        )
        axv.axhline(vmin, color=c, ls=":", lw=0.7, alpha=0.5)
        axq.plot(series[sid]["t"], series[sid]["q"], color=c, lw=1.0)
    axv.set_ylabel("Vcap (V)")
    axv.set_title(
        f"Per-REHD Vcap(t) and MAC queue — {len(sids)} REHDs "
        f"(dotted = each REHD's vmin)"
    )
    axv.grid(True, alpha=0.3)
    axv.legend(fontsize=8, ncol=3, loc="best")
    axq.set_ylabel("queue (pkts)")
    axq.set_xlabel("time (s)")
    axq.grid(True, alpha=0.3)
    f1 = os.path.join(args.out, "vcap_all.png")
    fig.tight_layout()
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # --- Figure 2: per-REHD twin-axis (Vcap left, queue right) ---
    n = len(sids)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
    for k, sid in enumerate(sids):
        ax = axes[k // ncol][k % ncol]
        vmin, vmax = meta[sid]
        t, v, q = series[sid]["t"], series[sid]["vcap"], series[sid]["q"]
        ax.plot(t, v, color=cmap(k), lw=1.2, label="Vcap")
        ax.axhline(vmin, color="red", ls="--", lw=1, label=f"vmin={vmin:.2f}")
        ax.axhline(vmax, color="green", ls="--", lw=1, label=f"vmax={vmax:.2f}")
        ax.set_ylabel("Vcap (V)")
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
        axr = ax.twinx()
        axr.plot(t, q, color="gray", lw=0.9, alpha=0.6)
        axr.set_ylabel("queue (pkts)", color="gray")
        axr.tick_params(axis="y", labelcolor="gray")
        vmn = min(v)
        ax.set_title(
            f"STA{sid}  (minV={vmn:.3f}, maxQ={max(q) if q else 0})", fontsize=9
        )
        ax.legend(fontsize=7, loc="upper left")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    f2 = os.path.join(args.out, "vcap_grid.png")
    fig.tight_layout()
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    # --- console summary ---
    print(f"Loaded {sum(len(series[s]['t']) for s in sids)} samples for {n} REHDs")
    print(
        f"{'STA':>4} {'vmin':>6} {'Vcap_min':>9} {'Vcap_end':>9} {'maxQ':>6} {'endQ':>6}"
    )
    for sid in sids:
        vmin, _ = meta[sid]
        v, q = series[sid]["vcap"], series[sid]["q"]
        print(
            f"{sid:>4} {vmin:>6.2f} {min(v):>9.3f} {v[-1]:>9.3f} {max(q):>6d} {q[-1]:>6d}"
        )
    print(f"\nSaved:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
