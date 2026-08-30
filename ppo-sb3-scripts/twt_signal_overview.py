#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Short signal overview on a fresh EDA run: for each realistic + oracle signal, show
ALIVE% and the per-DEVICE-CLASS mean (IoT/Camera/Voice/Video/REHD) — using orc_device_class
(now oracle) as the label. Cumulative counters shown as per-step delta (time-local).
Answers: do per-AC BSR + MCS revive in the new mix? how do signals separate device types?
"""

import os, sys, glob
import numpy as np, pandas as pd

RUN = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.abspath(open("/dev/shm/eda_signals_outdir.txt").read().strip())
)
CLS = {0: "IoT", 1: "Camera", 2: "Voice", 3: "Video", 4: "REHD"}


def main():
    sh = sorted(glob.glob(os.path.join(RUN, "shards", "*.parquet")))
    if not sh:
        print("no shards yet at", RUN)
        return
    sig = [c for c in pd.read_parquet(sh[0]).columns if c.startswith(("real_", "orc_"))]
    df = pd.concat(
        [
            pd.read_parquet(s, columns=["seed", "sta_id", "update_idx"] + sig)
            for s in sh
        ],
        ignore_index=True,
    )
    df = df.sort_values(["seed", "sta_id", "update_idx"]).reset_index(drop=True)
    g = df.groupby(["seed", "sta_id"], sort=False)
    dc = np.asarray(df["orc_device_class"].values).astype(int).ravel()
    classes = sorted(int(c) for c in np.unique(dc) if int(c) in CLS)
    print(
        f"rows={len(df):,}  device mix: "
        + ", ".join(f"{CLS[c]}={int((dc==c).mean()*100)}%" for c in classes)
    )

    def line(name, x):
        alive = (np.abs(x) > 0).mean()
        per = "  ".join(f"{CLS[c]:>6}={np.nanmean(x[dc==c]):8.2f}" for c in classes)
        print(f"  {name:26} alive={alive:4.2f} | {per}")

    print("\n=== REALISTIC (per-step delta for counters, value for levels) ===")
    for c in sig:
        if not c.startswith("real_"):
            continue
        v = df[c].values.astype(float)
        d = np.clip(np.nan_to_num(g[c].diff().values), 0, None)
        cumlike = np.nanmean((np.nan_to_num(g[c].diff().values)) >= -1e-9) > 0.98 and (
            np.nanmax(v) > np.nanmin(v)
        )
        line(c.replace("real_", "") + ("(Δ)" if cumlike else ""), d if cumlike else v)

    print("\n=== ORACLE (per-step delta for counters, value for levels) ===")
    for c in sig:
        if not c.startswith("orc_"):
            continue
        v = df[c].values.astype(float)
        d = np.clip(np.nan_to_num(g[c].diff().values), 0, None)
        cumlike = np.nanmean((np.nan_to_num(g[c].diff().values)) >= -1e-9) > 0.98 and (
            np.nanmax(v) > np.nanmin(v)
        )
        line(c.replace("orc_", "") + ("(Δ)" if cumlike else ""), d if cumlike else v)


if __name__ == "__main__":
    main()
