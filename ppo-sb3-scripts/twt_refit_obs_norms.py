#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Refit the 7-feature obs-normalization warm-start stats from an EDA run.

The policy z-scores each obs feature with a running mean/std (BasePolicy obs_rms),
warm-started from obs_warmstart_stats.json. Those warm-start stats are simply the
per-feature mean/std of the POST-transform obs values produced by
_build_per_sta_features — which twt_eda_collect.py logs as the obs_<feature> columns.

This tool reads those obs_<feature> columns from an EDA run's parquet shards, computes
the per-feature mean/std, prints a side-by-side comparison against the current
obs_warmstart_stats.json, and writes a PROPOSED json. It does NOT overwrite the live
file unless --apply is given — the unified pipeline driver prints this comparison and
prompts the user before applying.

The transform_norms (the obs-builder constants in twt_spawn_worker._build_per_sta_features)
are copied from the current file unchanged: refitting the z-score does not change the
upstream transforms.

Usage:
  twt_refit_obs_norms.py --shards <eda_run>/shards [--current obs_warmstart_stats.json]
                         [--out proposed.json] [--apply]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Must match _build_per_sta_features / obs_warmstart_stats.json "feature_order".
OBS_NAMES = ["bsr_be", "dpkts_rx", "dairtime", "dfcs", "snr", "silence", "starv_rate"]
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CURRENT = os.path.join(_HERE, "obs_warmstart_stats.json")


def _load_obs(shard_dir):
    paths = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not paths:  # allow passing the run root instead of its shards/ subdir
        paths = sorted(glob.glob(os.path.join(shard_dir, "shards", "*.parquet")))
    if not paths:
        sys.exit(f"[refit] no parquet shards found under {shard_dir}")
    cols = [f"obs_{n}" for n in OBS_NAMES]
    frames = [pd.read_parquet(p, columns=cols) for p in paths]
    return pd.concat(frames, ignore_index=True), len(paths)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--shards",
        required=True,
        help="EDA run dir (with shards/*.parquet) or the shards/ dir itself",
    )
    ap.add_argument(
        "--current",
        default=_DEFAULT_CURRENT,
        help="current obs_warmstart_stats.json (default: alongside this script)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="proposed json path (default: <shards>/../obs_warmstart_proposed.json)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="overwrite --current in place (skip the proposal file)",
    )
    args = ap.parse_args()

    df, n_shards = _load_obs(args.shards)
    new_mean = [float(df[f"obs_{n}"].mean()) for n in OBS_NAMES]
    new_std = [float(df[f"obs_{n}"].std()) for n in OBS_NAMES]
    n_rows = len(df)

    cur = json.load(open(args.current)) if os.path.exists(args.current) else {}
    cur_mean = cur.get("mean", [float("nan")] * len(OBS_NAMES))
    cur_std = cur.get("std", [float("nan")] * len(OBS_NAMES))

    def _pct(new, old):
        return 100.0 * (new - old) / old if old else float("nan")

    print(f"\n[refit] {n_shards} shards, {n_rows:,} rows from {args.shards}")
    print(f"[refit] comparing against {args.current}\n")
    print(
        f"  {'feature':12} {'cur_mean':>9} {'new_mean':>9} {'d%':>7}   "
        f"{'cur_std':>9} {'new_std':>9} {'d%':>7}"
    )
    print("  " + "-" * 72)
    for i, n in enumerate(OBS_NAMES):
        print(
            f"  {n:12} {cur_mean[i]:9.4f} {new_mean[i]:9.4f} {_pct(new_mean[i], cur_mean[i]):7.1f}   "
            f"{cur_std[i]:9.4f} {new_std[i]:9.4f} {_pct(new_std[i], cur_std[i]):7.1f}"
        )

    proposed = {
        "feature_order": OBS_NAMES,
        "mean": new_mean,
        "std": new_std,
        "count": n_rows,
        "transform_norms": cur.get("transform_norms", {}),
        "source": f"refit from {os.path.basename(os.path.abspath(args.shards.rstrip('/')))} "
        f"({n_shards} shards / {n_rows} rows)",
        "note": "post-transform pre-zscore stats for the 7-feat time-local obs; "
        "transform_norms copied from the previous file (obs-builder constants unchanged).",
    }

    if args.apply:
        json.dump(proposed, open(args.current, "w"), indent=1)
        print(f"\n[refit] APPLIED -> {args.current}")
        return

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.shards.rstrip("/"))),
        "obs_warmstart_proposed.json",
    )
    json.dump(proposed, open(out, "w"), indent=1)
    print(f"\n[refit] wrote proposal -> {out}")
    print("[refit] (not applied; copy it over obs_warmstart_stats.json to adopt)")


if __name__ == "__main__":
    main()
