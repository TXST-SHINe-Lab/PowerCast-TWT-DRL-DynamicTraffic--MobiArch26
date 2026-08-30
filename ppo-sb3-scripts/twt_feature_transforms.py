#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Design + verify the obs feature transforms: delta every cumulative counter, then
map each signal to a good (bounded, well-spread, low-skew) distribution for the NN.

For each candidate obs signal:
  - cumulative -> per-step delta within (seed,sta); level -> value
  - report RAW delta/value distribution (p50/p90/p99/max, %zero, skew)
  - apply a candidate transform and report the POST distribution (target: ~[0,1],
    p99<=~1, std reasonable, skew shrunk)
Picks the normalizer constant from p99 so the bulk lands in [0,1] (clip the tail).
Outputs a transform spec dict to bake into _build_per_sta_features.
NOTE: fit on the OLD-scenario parquet -> transform TYPES are reliable; the constants
get RE-FIT on the new per-STA sweep (magnitudes shift with the new rates).
"""

import os, sys, glob, json
import numpy as np
import pandas as pd


def _skew(x):
    x = np.asarray(x, float)
    m = x.mean()
    s = x.std()
    return float(((x - m) ** 3).mean() / (s**3 + 1e-12)) if s > 0 else 0.0


RUN = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.abspath(open("/dev/shm/eda_outdir.txt").read().strip())
)
STEP_US = 25 * 102.4 * 1000.0  # one agent step (us)
SP_PER_STEP = 25.0

# obs signal -> (is_cumulative, transform). transforms applied AFTER delta(if cum).
SPEC = {
    "real_bsr_queue_ac_be": (False, "div255"),
    "real_bsr_queue_ac_bk": (False, "div255"),
    "real_bsr_queue_ac_vi": (False, "div255"),
    "real_bsr_queue_ac_vo": (False, "div255"),
    "real_packets_received_at_ap": (True, "log1p"),
    "real_airtime_used_us": (True, "log1p"),
    "real_fcs_error_count": (True, "log1p"),
    "real_snr_db": (False, "snr_lin"),
    "real_last_rx_mcs": (False, "div11"),
    "real_sp_with_starvation_count": (True, "div_sp"),
    "real_sp_with_demand_count": (True, "div_sp"),
}


def apply_tf(x, tf, p99):
    if tf == "div255":
        return np.clip(x / 255.0, 0, 1)
    if tf == "div11":
        return np.clip(x / 11.0, 0, 1)
    if tf == "snr_lin":
        return np.clip((x - 49.0) / 20.0, 0, 1)  # SNR ~49..69 dB -> [0,1]
    if tf == "frac_step":
        return np.clip(x / STEP_US, 0, 1)  # airtime fraction
    if tf == "div_sp":
        return np.clip(x / SP_PER_STEP, 0, 1)  # SPs/step
    if tf == "log1p":
        return np.clip(np.log1p(x) / np.log1p(max(p99, 1.0)), 0, 1)  # p99 -> ~1
    return x


def main():
    shards = sorted(glob.glob(os.path.join(RUN, "shards", "*.parquet")))
    cols = ["seed", "sta_id", "update_idx"] + list(SPEC.keys())
    df = pd.concat(
        [pd.read_parquet(s, columns=cols) for s in shards], ignore_index=True
    )
    df = df.sort_values(["seed", "sta_id", "update_idx"]).reset_index(
        drop=True
    )  # TIME order
    g = df.groupby(["seed", "sta_id"], sort=False)
    print(f"rows={len(df):,}\n")
    print(
        f"{'signal':30} {'kind':4} {'p50':>10} {'p90':>10} {'p99':>10} {'%zero':>6} {'skew':>7} | {'TF':>9} {'norm':>10} -> {'post p50':>8} {'post p99':>8} {'post skew':>9}"
    )
    spec_out = {}
    for c, (cum, tf) in SPEC.items():
        v = df[c].values.astype(float)
        x = np.clip(g[c].diff().values, 0, None) if cum else v
        x = np.nan_to_num(x)
        p50, p90, p99 = np.percentile(x, [50, 90, 99])
        pz = float(np.mean(x == 0))
        sk = float(_skew(x))
        y = apply_tf(x, tf, p99)
        yp50, yp99 = np.percentile(y, [50, 99])
        ysk = float(_skew(y))
        norm = (
            round(float(max(p99, 1.0)), 1)
            if tf == "log1p"
            else {
                "div255": 255,
                "div11": 11,
                "snr_lin": 20,
                "frac_step": STEP_US,
                "div_sp": SP_PER_STEP,
            }[tf]
        )
        spec_out[c] = {"delta": cum, "transform": tf, "norm": norm}
        print(
            f"{c.replace('real_',''):30} {'cum' if cum else 'lvl':4} {p50:10.2f} {p90:10.2f} {p99:10.2f} {pz:6.2f} {sk:7.1f} | {tf:>9} {norm:10.1f} -> {yp50:8.3f} {yp99:8.3f} {ysk:9.2f}"
        )
    out = os.path.join(RUN, "obs_transform_spec.json")
    json.dump(spec_out, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print(
        "\nRead: post p99 should be ~1, post skew should be SMALLER than raw skew (good distribution)."
    )
    print(
        "'%zero' high (>0.8) => signal is sparse/near-dead in THIS scenario (recheck on new sweep)."
    )


if __name__ == "__main__":
    main()
