#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Verify the FULL used signal set (obs + reward) is time-local + well-distributed.
Each signal: delta (cumulative) / value (level) / recency (timestamp) / ratio, then a
transform; report post-distribution (p50, p99, %zero, skew). 'good' = bounded ~[0,1],
|skew|<~1.5, not degenerate. NOTE: old-scenario constants; types are what matter."""

import os, sys, glob
import numpy as np, pandas as pd

RUN = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.abspath(open("/dev/shm/eda_signals_outdir.txt").read().strip())
)
STEP_US = 25 * 102.4 * 1000.0
LAT_NORM = 2000.0


def skew(x):
    x = np.asarray(x, float)
    m, s = x.mean(), x.std()
    return float(((x - m) ** 3).mean() / (s**3 + 1e-12)) if s > 0 else 0.0


def L(x, p99):
    return np.clip(np.log1p(x) / np.log1p(max(p99, 1.0)), 0, 1)


def main():
    sh = sorted(glob.glob(f"{RUN}/shards/*.parquet"))
    need = [
        "seed",
        "sta_id",
        "update_idx",
        "real_bsr_queue_ac_be",
        "real_bsr_queue_ac_bk",
        "real_bsr_queue_ac_vi",
        "real_bsr_queue_ac_vo",
        "real_packets_received_at_ap",
        "real_bytes_received_at_ap",
        "real_airtime_used_us",
        "real_fcs_error_count",
        "real_rx_fragment_count",
        "real_snr_db",
        "real_last_rx_mcs",
        "real_last_rx_timestamp_us",
        "real_sp_completed_count",
        "real_sp_with_demand_count",
        "real_sp_with_starvation_count",
        "real_bytes_rx_at_ap_in_sp",
        "real_packets_rx_at_ap_in_sp",
        "real_observation_time_ms",
        "orc_packets_transmitted",
        "orc_packets_enqueued",
        "orc_mpdu_drops_expired",
        "orc_queue_delay_sum_ms",
        "orc_queue_delay_count",
        "orc_queue_size_packets",
        "orc_awake_time_ms",
        "orc_sleep_time_ms",
        "orc_harvested_total_j",
        "orc_consumed_total_j",
        "orc_vcap_v",
        "orc_vcap_max_v",
        "orc_unpowered_tx_events",
    ]
    df = pd.concat([pd.read_parquet(s, columns=need) for s in sh], ignore_index=True)
    df = df.sort_values(["seed", "sta_id", "update_idx"]).reset_index(drop=True)
    g = df.groupby(["seed", "sta_id"], sort=False)
    d = lambda c: np.clip(np.nan_to_num(g[c].diff().values), 0, None)  # per-step delta
    v = lambda c: np.nan_to_num(df[c].values.astype(float))  # level value
    now = v("real_observation_time_ms") * 1000.0
    eps = 1e-9
    feats = {}  # name -> (transformed array, kind)
    # ---- OBS (realistic, time-local) ----
    for ac in ["be", "bk", "vi", "vo"]:
        feats[f"bsr_{ac}"] = (
            np.clip(v(f"real_bsr_queue_ac_{ac}") / 255, 0, 1),
            "level/255",
        )
    for c, nm in [
        ("real_packets_received_at_ap", "Δpkts_rx_ap"),
        ("real_bytes_received_at_ap", "Δbytes_rx_ap"),
        ("real_airtime_used_us", "Δairtime"),
        ("real_fcs_error_count", "Δfcs"),
        ("real_rx_fragment_count", "Δrx_frag"),
        ("real_bytes_rx_at_ap_in_sp", "Δbytes_sp"),
        ("real_packets_rx_at_ap_in_sp", "Δpkts_sp"),
    ]:
        x = d(c)
        feats[nm] = (L(x, np.percentile(x, 99)), "Δ+log1p")
    feats["snr"] = (np.clip((v("real_snr_db") - 49) / 20, 0, 1), "level lin")
    feats["mcs"] = (np.clip(v("real_last_rx_mcs") / 11, 0, 1), "level/11")
    lr = v("real_last_rx_timestamp_us")
    sil = np.where(lr <= 0, 1.0, np.clip((now - lr) / STEP_US, 0, 1))
    feats["silence"] = (sil, "recency")
    feats["Δsp_completed"] = (np.clip(d("real_sp_completed_count") / 25, 0, 1), "Δ/25")
    dem = d("real_sp_with_demand_count")
    stv = d("real_sp_with_starvation_count")
    feats["starvation_rate"] = (
        np.clip(stv / np.maximum(dem, eps), 0, 1),
        "ratio Δstv/Δdem",
    )
    # ---- REWARD (oracle, time-local) ----
    eq = d("orc_packets_enqueued")
    tx = d("orc_packets_transmitted")
    dr = d("orc_mpdu_drops_expired")
    feats["served"] = (
        np.where(eq > 0, np.clip(tx / np.maximum(eq, eps), 0, 1), 1.0),
        "ratio Δtx/Δeq",
    )
    feats["drop_rate"] = (
        np.where(eq > 0, np.clip(dr / np.maximum(eq, eps), 0, 1), 0.0),
        "ratio Δdrop/Δeq",
    )
    qc = d("orc_queue_delay_count")
    qs = d("orc_queue_delay_sum_ms")
    feats["latency_norm"] = (
        np.where(qc > 0, np.clip((qs / np.maximum(qc, eps)) / LAT_NORM, 0, 1), 0.0),
        "Δsum/Δcnt /2000",
    )
    bl = v("orc_queue_size_packets")
    feats["backlog"] = (L(bl, np.percentile(bl, 99)), "level+log1p")
    aw = d("orc_awake_time_ms")
    sl = d("orc_sleep_time_ms")
    feats["duty_step"] = (
        np.where((aw + sl) > 0, aw / np.maximum(aw + sl, eps), 0.0),
        "Δaw/(Δaw+Δsl)",
    )
    hv = d("orc_harvested_total_j")
    cs = d("orc_consumed_total_j")
    isr = v("orc_vcap_max_v") > 0
    hc = np.where((isr) & (cs > 0), np.clip(hv / np.maximum(cs, eps), 0, 1), 0.0)
    feats["hc(REHD)"] = (hc, "ratio Δhrv/Δcons")
    vm = v("orc_vcap_max_v")
    soc = np.where(vm > 0, np.clip(v("orc_vcap_v") / np.maximum(vm, eps), 0, 1), 0.0)
    feats["soc(REHD)"] = (soc, "ratio vcap/vmax")
    feats["Δunpowered_tx(REHD)"] = (
        np.clip(d("orc_unpowered_tx_events") / 10, 0, 1),
        "Δ/10",
    )

    print(
        f"rows={len(df):,}\n{'feature':22} {'transform':16} {'p50':>6} {'p99':>6} {'%zero':>6} {'skew':>7}  verdict"
    )
    for nm, (x, tf) in feats.items():
        p50, p99 = np.percentile(x, [50, 99])
        pz = float((x == 0).mean())
        sk = skew(x)
        good = (p99 <= 1.001) and (abs(sk) < 1.5) and (pz < 0.9) and (x.std() > 1e-6)
        vd = (
            "GOOD"
            if good
            else ("DEAD/sparse" if (pz >= 0.9 or x.std() < 1e-6) else "skewed")
        )
        print(f"{nm:22} {tf:16} {p50:6.2f} {p99:6.2f} {pz:6.2f} {sk:7.2f}  {vd}")


if __name__ == "__main__":
    main()
