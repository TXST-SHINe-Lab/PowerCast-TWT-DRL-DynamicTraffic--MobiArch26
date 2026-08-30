#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
plot_harvest_curve.py — PowerCast P21XXCSR-EVB Band-6 (2.4 GHz) harvesting model,
digitized directly from the device DATASHEET efficiency curve (Powerharvester
Efficiency vs. RFIN, Band 6, 0.7 V setting, 2450 MHz trace).

Panel 1: RF-to-DC efficiency η vs received power P_rx (datasheet 2450 MHz curve):
         zero below the −12 dBm sensitivity floor, steep rise to a ~46% peak at
         +8 dBm, held flat above +8 dBm.
Panel 2: harvested DC power vs distance at the AP's 36 dBm EIRP (30 dBm + 6 dBi),
         with the [1.5, 4.75] m REHD annulus shaded.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOOST = 0.85  # PCC210 boost-converter efficiency (datasheet)
FSPL_1M = 40.05  # Friis free-space loss @ 1 m, 2.4 GHz (dB)
RECTENNA_GAIN = 6.0  # REHD_RX_GAIN_DBI
SENS_FLOOR = -12.0  # datasheet 2.45 GHz RF input minimum (η = 0 below this)
CLAMP_HI = 15.0  # +15 dBm max rated input

# --- datasheet Band-6 / 0.7 V efficiency curve, 2450 MHz trace (digitized) ---
# (received power dBm, RF-to-DC efficiency %).
# The datasheet curve peaks ≈46% at +8 dBm and then rolls off; we HOLD it flat at the peak for >+8 dBm, since any higher input can be attenuated down to the optimal point (but not boosted up).
PEAK_DBM = 8.0
_ETA_DBM = np.array(
    [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 16]
)
_ETA_PCT = np.array(
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
    """RF-to-DC efficiency (fraction): datasheet 2450 MHz curve, 0 below −12 dBm,
    held flat at the ~46% peak above +8 dBm."""
    p = np.asarray(p, dtype=float)
    e = np.interp(p, _ETA_DBM, _ETA_PCT) / 100.0
    return np.where(p < SENS_FLOOR, 0.0, e)


def p_rx_dbm(d, eirp):
    """Received power at REHD: EIRP - Friis(d) + rectenna gain."""
    return eirp - (FSPL_1M + 20.0 * np.log10(d)) + RECTENNA_GAIN


def p_dc_mw(p_rx):
    """Harvested DC power (mW) = P_rx_mW * eta * boost."""
    p_mw = 10.0 ** (p_rx / 10.0)
    return p_mw * eta(p_rx) * BOOST


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results", "data-log")
    os.makedirs(out, exist_ok=True)

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

    # --- Panel 1: datasheet efficiency curve ---
    p = np.linspace(-16, 16, 2000)
    ax1.plot(p, 100 * eta(p), color="tab:blue", lw=2.6, label="2450 MHz (datasheet)")
    ax1.axvline(SENS_FLOOR, color="red", ls="--", lw=1.2, label="−12 dBm floor")
    nm_far, nm_near = p_rx_dbm(4.75, 36.0), p_rx_dbm(1.5, 36.0)
    ax1.axvspan(
        nm_far, nm_near, color="tab:orange", alpha=0.18, label="in-room [1.5,4.75] m"
    )
    ax1.set_xlabel("received power $P_{rx}$ (dBm)")
    ax1.set_ylabel("RF-to-DC efficiency η (%)")
    ax1.set_title("(a) Band-6 efficiency")
    ax1.set_xlim(-16, 5)
    ax1.set_ylim(0, 50)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc="lower right")

    # --- Panel 2: harvested DC power vs distance ---
    d = np.linspace(0.3, 12, 2000)
    pw = p_dc_mw(p_rx_dbm(d, 36.0)) * 1000.0  # µW
    pw = np.where(pw > 0, pw, np.nan)  # drop the η=0 (P_rx<−12) region on log axis
    ax2.plot(d, pw, color="tab:orange", lw=2.6, label="36 EIRP (30+6 dBi)")
    ax2.axvspan(1.5, 4.75, color="gray", alpha=0.12, label="annulus [1.5,4.75] m")
    # distance at which P_rx hits the −12 dBm floor
    d_floor = 10 ** ((36.0 + RECTENNA_GAIN - FSPL_1M - SENS_FLOOR) / 20.0)
    ax2.axvline(
        d_floor, color="red", ls="--", lw=1.0, label=f"−12 floor @ {d_floor:.0f} m"
    )
    ax2.set_xlabel("distance from AP (m)")
    ax2.set_ylabel("harvested DC power (µW)")
    ax2.set_title("(b) Harvested power")
    ax2.set_yscale("log")
    ax2.set_xlim(0.3, 7)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=12, loc="lower left")

    f = os.path.join(out, "harvest_curve.png")
    fig.tight_layout(pad=0.6, w_pad=1.2)
    fig.savefig(f, dpi=150)
    plt.close(fig)

    # --- console reference ---
    print(
        "Datasheet Band-6 2450 MHz η curve, boost=%.2f, Friis 2.4 GHz, +6 dBi rectenna"
        % BOOST
    )
    print(f"  sensitivity floor: {SENS_FLOOR} dBm  →  36-EIRP reaches {d_floor:.2f} m")
    print(f"\n{'dist(m)':>7} | {'P_rx_36':>8} {'η_36':>5} {'µW_36':>8}")
    for dd in [1.0, 1.5, 2, 3, 4, 4.5, d_floor, 5, 6]:
        pn = p_rx_dbm(dd, 36.0)
        print(f"{dd:>7.2f} | {pn:>8.1f} {100*eta(pn):>4.0f}% {1000*p_dc_mw(pn):>7.2f}")
    print(f"\nSaved: {f}")


if __name__ == "__main__":
    main()
