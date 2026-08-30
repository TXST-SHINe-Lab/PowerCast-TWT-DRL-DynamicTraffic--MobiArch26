// Copyright (c) 2025 Texas State University
//
// SPDX-License-Identifier: GPL-2.0-only
//
// Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
// PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

#ifndef TWT_CONSTANTS_H
#define TWT_CONSTANTS_H

#include <filesystem>
#include <string>
#include <system_error>

// --- EXAMPLE-DIRECTORY RESOLUTION ---
// CANONICAL LAYOUT: this example must live at
//     <ns-3.44>/contrib/ai/examples/rl-twt-powercast/
// i.e. clone it with that explicit target name (see README "Quick start"):
//     git clone <repo-url> rl-twt-powercast
// The simulation is always launched from the NS-3 root (./ns3 run, and the Python wrapper chdir()s there), so the canonical path below resolves as-is.
//
// The literal is the contract.
// The __FILE__ fallback exists only so the binary still finds its own source directory if it is invoked with a different working directory; it is NOT a license to rename the folder.
// Rename it and the hardcoded literal stops matching, which is exactly the breakage the README warns about.
#define TWT_EXAMPLE_DIR_NAME "rl-twt-powercast"
#define TWT_EXAMPLE_DIR_CANONICAL "contrib/ai/examples/" TWT_EXAMPLE_DIR_NAME

inline const std::string&
TwtExampleDir()
{
    static const std::string dir = []
    {
        std::error_code ec;
        if (std::filesystem::is_directory(TWT_EXAMPLE_DIR_CANONICAL, ec))
        {
            return std::string(TWT_EXAMPLE_DIR_CANONICAL);
        }
        // Fallback: launched from some other cwd. Recover this file's own directory.
        const std::string f(__FILE__);
        const auto pos = f.find_last_of('/');
        return (pos == std::string::npos) ? std::string(".") : f.substr(0, pos);
    }();
    return dir;
}

/// Absolute path to `rel` inside this example's directory (for SOURCE files, e.g.
/// ph-harvester-config.txt).
inline std::string
TwtExamplePath(const std::string& rel)
{
    return TwtExampleDir() + "/" + rel;
}

/// Absolute path to `rel` inside this example's single GENERATED-OUTPUT root,
/// <example>/results/. Everything the run produces — ns-3 trace CSVs, Vcap logs,
/// pcap, EDA/training/eval artifacts, figures — lands under here and nowhere else,
/// so a clean slate is one `rm -rf results/` and nothing generated is ever mixed in
/// with source. Parent directories are created on demand: the C++ writers open
/// ofstreams directly and would otherwise fail silently into a missing directory.
inline std::string
TwtResultsPath(const std::string& rel)
{
    const std::string full = TwtExampleDir() + "/results/" + rel;
    const auto slash = full.find_last_of('/');
    if (slash != std::string::npos)
    {
        std::error_code ec; // non-throwing: a pre-existing directory is not an error
        std::filesystem::create_directories(full.substr(0, slash), ec);
    }
    return full;
}

// --- CORE CONFIGURATION CONSTANTS ---
// MAX_NUM_STA is the compile-time CEILING — sizes the C++ EnvStruct/ActionStruct arrays in shared memory.
// Bumped to 32 to leave headroom for variable-N experiments without a rebuild every time.
// ACTIVE_NUM_STA is the per-config runtime count (must be <= MAX_NUM_STA).
#define MAX_NUM_STA 32    // Maximum number of STAs (compile-time ceiling)
#define ACTIVE_NUM_STA 16 // Default number of active STAs in simulation
#define MAX_NUM_TWT_GROUPS \
    16 // Maximum number of TWT groups (raised 8->16 2026-06-05 for the per-STA airtime-allocation action space: up to 12 SPs/BI so hot STAs get dedicated (anomaly-immune) SPs + rate-homogeneous idle pools).
#define MAC_ADDR_LEN 6 // MAC address length in bytes

// --- NETWORK TOPOLOGY CONSTANTS ---
// Room is a small office / conference scale (14 m × 14 m, AP at origin → walls at ±7 m).
// REHDs are constrained to a [REHD_SPAWN_MIN_RADIUS_M, REHD_SPAWN_RADIUS_M] = [1.5 m, 4.75 m] harvest annulus around the AP (see SetupMobility) — fits inside the room with a 1 m margin.
// The inner 1.5 m exclusion stops any REHD sitting point-blank on the AP; the 4.75 m outer edge puts the farthest REHDs in the Friis-loss regime.
// Non-REHD STAs spawn anywhere.
#define DEFAULT_ROOM_LENGTH 14.0 // Default room length in meters (was 10)

// --- RF DEPLOYMENT (Phase-2 / QoEH realism) ---
// AP: power-beacon-class AP. 30 dBm conducted + 6 dBi antenna gain → 36 dBm EIRP.
//   (Was 30 dBm / 36 EIRP "power-beacon" — lowered 2026-05-28 so harvest is no longer ~10-65x consumption.
//   At 26 EIRP, in-room harvest is the same order as REHD TX drain, so capacitors actually cycle and the deep-discharge gate fires.)
// Non-REHD STAs: typical mobile-device class (NS-3 default, 16 dBm, 0 dB gain).
// REHDs: ultra-low-power sensors — 10 dBm TX, 6 dBi rectenna gain at RX.
// Channel: Friis (free-space, n=2.0, ref 40 dB @ 1 m) — the most-lossless defensible indoor model.
//   Combined with the powers above and the [1.5, 4.75] m REHD annulus, the near edge stays energy-positive while the far edge drains.
#define AP_TX_POWER_DBM \
    30.0 // 30 dBm conducted + 6 dBi = 36 dBm EIRP (FCC Part 15.247 point-to-multipoint maximum). At 36 dBm EIRP the datasheet -12 dBm sensitivity floor is reached at ~5 m, so the REHD annulus is capped at 4.75 m to keep every harvester above the floor (datasheet eta curve + 0.85 boost).
#define AP_TX_GAIN_DBI 6.0
#define REHD_TX_POWER_DBM 10.0
#define REHD_RX_GAIN_DBI 6.0
#define REHD_SPAWN_RADIUS_M \
    4.75 // REHD harvest-annulus OUTER radius (=> P_rx > -12 dBm floor at 36 EIRP)
#define REHD_SPAWN_MIN_RADIUS_M 1.5 // REHD harvest-annulus INNER radius (point-blank exclusion)
#define WIFI_CENTER_FREQ_HZ 2.4e9   // 2.4 GHz ISM band (PowerCast P21XXCSR Band 6)

// REHD power budget while the radio is AWAKE (single-antenna time-switching → no harvest in these states).
// Anchored to a real ultra-low-power 2.4 GHz radio (TI CC2652P-class coin-cell IoT MCU): RX/decode ~7.5 mW, IDLE/CCA/deep-sleep leakage ~1 µW.
// (The ns-3 PHY_STATE_*_MA standard-STA model below is for CONVENTIONAL STAs only — it never drives the REHD capacitor.)
// TX drain is separate: radiated RF / PA_EFFICIENCY (=0.75) debited in SetConsumedEnergy.
// SLEEP harvests via the rectenna AND leaks REHD_SLEEP_POWER_W in parallel.
#define REHD_IDLE_POWER_W 1.0e-6 // 1 µW — IDLE / CCA-busy listening (LP-radio class)
#define REHD_RX_POWER_W \
    7.5e-3 // 7.5 mW — actively decoding a frame (low-power-RX mode; eased from 10 mW)
#define REHD_SLEEP_POWER_W 1.0e-6 // 1 µW — deep-sleep leakage (parallel to rectenna harvest)

// --- BEACON INTERVAL ---
#define BEACON_INTERVAL_MS 102.4 // Standard beacon interval (102.4ms)

// --- ENERGY MODEL CONSTANTS (mA - milliamps) ---
// WiFi PHY state power consumption model based on typical 802.11ax devices
#define PHY_STATE_IDLE_MA 50.0     // IDLE state: 50 mA
#define PHY_STATE_CCA_BUSY_MA 50.0 // CCA_BUSY state: 50 mA
#define PHY_STATE_RX_MA 66.0       // RX state: 66 mA
#define PHY_STATE_TX_MA 232.0      // TX state: 232 mA
#define PHY_STATE_SLEEP_MA 0.12    // SLEEP state: 0.12 mA
#define BATTERY_VOLTAGE_V 3.0      // Standard Li-ion battery voltage

// --- TRAFFIC & QUEUE CONSTANTS ---
#define MAX_QUEUE_SIZE_BYTES 65536 // Maximum BSR reportable, matches 802.11ax BSR encoding
#define PAYLOAD_SIZE_BYTES 1400    // Default payload size for traffic generation

// --- TWT UPDATE TIMING CONSTANTS (in Beacon Intervals) ---
#define TWT_UPDATE_INTERVAL_BI 25 // TWT update interval in BI
#define DURATION_IN_UPDATE \
    100 // Number of update cycles (was 75; bumped 2026-06-01 for the post-PDW EDA — longer episodes => more grid samples + steadier energy state. Global: training/eval episodes are now 100 updates too. SIMULATION_DURATION_BI auto-derives. Python EDA UPDATES must match.)
#define TWT_UPDATE_START_BI 90 // When to start TWT updates in BI
#define TWT_SETUP_TIME_BI 75   // Initial TWT setup time in BI

// --- POWER DELIVERY WINDOW (PDW) BURST CONSTANTS ---
// The AP delivers RF power as a dedicated power beacon over [PDW_BEACON_LEAD, pdw_end] of each BI (pdw_end = the agent's 3rd action head, in ms).
// We do NOT transmit packetized WiFi frames for this — going through UDP/IP/WiFi-MAC drags in CSMA contention, which (a) leaves IFS/backoff gaps between frames (wasted airtime) and (b) lets a momentarily-busy channel defer the whole back-to-back chain late, spilling power frames into the comms region.
// Instead the harvest is delivered DIRECTLY (DeliverPdwPower): at PDW_BEACON_LEAD after each beacon, every REHD asleep at that instant is credited RF energy for the FULL window duration, computed with the SAME analytical Friis path as the comms harvest (OnPhyTxEnd).
// This is the continuous-RF-waveform ideal a real Powercast power transmitter emits (it does not packetize) — 100% window occupancy, zero gaps, and bounded to the window by construction (an energy credit cannot spill onto the channel or collide with the next SP).
// Net: harvest = AP_RxPower x (pdw_end - PDW_BEACON_LEAD).
#define PDW_BCAST_WIFI_MODE \
    "HeMcs0" // pinned NonUnicastMode so the AP BEACON goes at HeMcs0 (~1.8 ms) instead of the lowest basic rate (~1 Mbps -> ~12 ms). (No PDW data frames are sent any more; this now only governs the beacon/group-frame rate.)
#define PDW_BEACON_LEAD_US \
    5000 // start power delivery 5 ms into each BI, so the AP's beacon (TBTT = phase 0, airtime ~1.8 ms) has finished and any REHD that briefly woke to RX the beacon is back asleep (harvest gate = IsStateSleep).
// MUST match Python PDW_BEACON_MARGIN_MS: the power window is [PDW_BEACON_LEAD, pdw_end], pdw_end = 5 + idx*5 ms.
// Beacon jitter is disabled for predictable TBTT.

// --- SIMULATION TIMING CONSTANTS (in Beacon Intervals) ---
#define SIMULATION_DURATION_BI \
    (TWT_UPDATE_START_BI + (DURATION_IN_UPDATE * TWT_UPDATE_INTERVAL_BI) + 5)
#define APP_START_TIME_MIN_BI 25 // Minimum app start time in BI
#define APP_START_TIME_MAX_BI 45 // Maximum app start time in BI
#define METRICS_START_TIME_BI 75 // When to start tracking metrics in BI

// --- WiFi PHY CONFIGURATION CONSTANTS ---
#define DEFAULT_MCS 4                 // Default MCS
#define DEFAULT_GUARD_INTERVAL_NS 800 // Guard Interval in nanoseconds (0.8us = standard GI)
#define DEFAULT_CHANNEL_WIDTH_MHZ 20  // Channel BW in MHz
#define RTS_CTS_THRESHOLD 2347        // Standard WiFi RTS/CTS threshold in bytes
// Controls when RTS/CTS handshake is triggered: frames > 2347 bytes use RTS/CTS for collision avoidance.
// In this setup with max packet sizes of 1500B, this threshold is rarely triggered.

// --- WiFi MAC CONFIGURATION CONSTANTS ---
#define MAX_MISSED_BEACONS 0xFFFFFFFF // STA never disconnects (max beacons missed)
#define BLOCK_ACK_THRESHOLD 1         // Enable Block Ack after 1 packet (WiFi 6 feature)
#define WIFI_STARTUP_WINDOW_BI 20     // WiFi PHY startup window in Beacon Intervals

// --- RANDOM NUMBER GENERATION CONSTANTS ---
// RNG Stream Management for Reproducible Multi-Run Studies
//
// NS3 uses independent RNG streams to ensure:
//   1. Deterministic behavior: Same seed → identical random sequences (reproducible)
//   2. Independence: Each device has separate stream to avoid correlation
//   3. Parameter sweeps: Multiple runs with different seeds for statistical analysis
//
// Usage for multi-run studies:
//   - Set RNG_INITIAL_SEED to base seed (e.g., 42)
//   - Create loop: for(int run=1; run<=NUM_RUNS; run++) { seed = RNG_INITIAL_SEED + run; }
//   - Each run generates different topology/traffic patterns while maintaining independence
//
// Example multi-run execution:
//   ./ns3 run "twt-powercast-main-simulation --randSeed=42 --simulationTime=60"    // Run 1
//   ./ns3 run "twt-powercast-main-simulation --randSeed=43 --simulationTime=60"    // Run 2
//   ./ns3 run "twt-powercast-main-simulation --randSeed=44 --simulationTime=60"    // Run 3
//   → Statistical analysis across runs provides confidence intervals
//
#define RNG_INITIAL_STREAM_ID 100 // Starting stream ID for WiFi devices
#define RNG_DEFAULT_RUN_NUMBER 1  // Default run number in parameter study
#define RNG_INITIAL_SEED 10       // Base seed for reproducibility (override via cmdline)

// --- DYNAMIC ENV: MOBILITY (intra-episode STA movement) ---
// Set ENABLE_DYNAMIC_MOBILITY = 0 to recover the legacy ConstantPositionMobilityModel behavior (positions still randomized at episode start, but never change after).
#define ENABLE_DYNAMIC_MOBILITY 1   // 1 = STAs random-walk; 0 = stationary (legacy)
#define MOBILITY_SPEED_MIN_MPS 0.05 // Drift-speed lower bound (m/s) — small drift in 10m room
#define MOBILITY_SPEED_MAX_MPS 0.2 // Drift-speed upper bound (m/s) — was 1.0; too fast for 10m room
#define MOBILITY_DIR_CHANGE_S 2.0  // Per-STA direction-change interval (s)
// Collision-avoidance buffers (asymmetric by class):
//   - Two non-REHD STAs must stay MOBILITY_BUFFER_M apart.
//   - Any pair involving a REHD only needs REHD_BUFFER_M (REHDs are tiny sensors with a smaller personal bubble; others may sit closer to them).
//     Applied both at spawn (rejection sampling) and at runtime (EnforceBuffer).
#define MOBILITY_BUFFER_M 0.5                // Min STA-vs-STA separation (m), was 1.5
#define REHD_BUFFER_M 0.25                   // Min separation for any pair involving a REHD (m)
#define MOBILITY_CORRECTION_INTERVAL_MS 1000 // Period of buffer-violation correction sweep (ms)

// --- DYNAMIC ENV: TRAFFIC (intra-episode rate / burstiness changes per STA) ---
// Per-STA app params get redrawn uniformly within the class range on every traffic-update event.
// Class is fixed for the episode; only the parameters drift.
// Set ENABLE_DYNAMIC_TRAFFIC = 0 to keep the constant per-class params from twt-simulation-config.h.
#define ENABLE_DYNAMIC_TRAFFIC \
    1   // 1 = per-STA traffic params drift; 0 = static (5-class topology).
        // ON 2026-06-04 for the dynamic-regime training.
// SEGMENTED-COIN regime (2026-06-04): the episode is split into N_TRAFFIC_SEGMENTS equal segments.
// At each segment boundary an independent 50/50 coin sets that segment OVER or UNDER saturation, and EVERY non-REHD STA re-draws its traffic params (rate/burstiness) from its class range scaled by trafficScale * (over? SEG_OVER_MULT : SEG_UNDER_MULT).
//   - The per-STA re-draw shifts WHICH STAs are hot each segment -> heterogeneity (the optimal grouping/assignment changes per segment).
//   - The coin multiplier puts the aggregate just over / just under saturation so the agent must SWITCH STRATEGY (under -> spend minimal airtime/energy; over -> minimize expiry).
//   - trafficScale is the SATURATION anchor S_sat (calibrated). REHDs hit `default:` in the UpdateTrafficForSta switch, so their pattern is untouched.
//     Coin uses an ns3 RNG stream seeded by randSeed -> reproducible per scenario (CRN-compatible).
#define N_TRAFFIC_SEGMENTS 5 // 5 segments x 20 agent-updates = 100 updates/episode
// Swing WIDENED to +/-35% (2026-06-05).
// The +/-15% EDA showed both regimes need near-max airtime (a medium schedule serves only ~0.80 even UNDER), so max-airtime won BOTH -> no real adaptivity.
// A medium schedule's capacity ~1.29 load-units (EDA), so UNDER must be <=~1.25 for it to fully serve (efficiency lever).
// +/-35% around S_sat=1.9 -> UNDER 1.235 (medium serves all -> use the cheapest sufficient schedule), OVER 2.565 (even max airtime drops ~17% -> minimize expiry).
// This makes the OVER vs UNDER optima genuinely differ under the served-dominant v2 reward.
// WIDENED again 2026-06-07 (heterogeneity push): 1.35/0.65 still gave a near-constant optimum (fixed-policy sweep: D=50+moderate-g won BOTH regimes).
// 1.6/0.4 = a 4x load swing so the under-segment is genuinely light (light load -> serve everyone with minimal grouping/low PDW) vs heavy over (-> isolate elephants + protect REHDs) => the optimal schedule should flip intra-episode.
// Paired with the lopsided elephant/mouse mix.
#define SEG_OVER_MULT 1.6  // over-saturated segment  = 1.6 * S_sat
#define SEG_UNDER_MULT 0.4 // under-saturated segment = 0.4 * S_sat
// The over/under coin is HASH-based per (randSeed, segIdx) in UpdateTrafficSegment (splitmix64), NOT an ns3 RNG draw: varying SEED with fixed RUN gives ns3 substreams persistent cross-seed correlation at fixed draw positions (measured: a warm-up discard just moved the biased segment), whereas the hash is ~0.50 over-rate per segment and decorrelated across both seed and segment.
#define TRAFFIC_UPDATE_MEAN_S \
    10.0 // (legacy Poisson mode only) mean inter-arrival of update events (s)

// IoT Sensor: OnOff with constant ON/OFF (very low-rate, periodic)
#define IOT_RATE_MIN_BPS 50000  // 50 kbps
#define IOT_RATE_MAX_BPS 500000 // 500 kbps
#define IOT_ON_MIN_MS 10
#define IOT_ON_MAX_MS 50
#define IOT_OFF_MIN_MS 50
#define IOT_OFF_MAX_MS 200

// Video Camera: UDP CBR. Only rate varies (via packet interval).
#define CAMERA_RATE_MIN_BPS 1000000 // 1 Mbps
#define CAMERA_RATE_MAX_BPS 5000000 // 5 Mbps

// Voice Assistant: OnOff with exponential ON/OFF (turn-taking)
#define VOICE_RATE_MIN_BPS 32000  // 32 kbps
#define VOICE_RATE_MAX_BPS 128000 // 128 kbps
#define VOICE_ON_MEAN_MIN_MS 1000
#define VOICE_ON_MEAN_MAX_MS 4000
#define VOICE_OFF_MEAN_MIN_MS 1000
#define VOICE_OFF_MEAN_MAX_MS 5000

// Video Streaming: OnOff with exponential ON/OFF (bursty VBR)
#define VIDEO_STREAM_RATE_MIN_BPS 2000000  // 2 Mbps
#define VIDEO_STREAM_RATE_MAX_BPS 15000000 // 15 Mbps
#define VIDEO_STREAM_ON_MEAN_MIN_MS 2000
#define VIDEO_STREAM_ON_MEAN_MAX_MS 8000
#define VIDEO_STREAM_OFF_MEAN_MIN_MS 500
#define VIDEO_STREAM_OFF_MEAN_MAX_MS 3000

// --- QoEH per-Service-Period (SP) UL-accounting metric definitions (Phase 2) ---
// An SP = a STA's TWT wake window, delimited by PHY sleep intervals.
// At each SP start the AP snapshots the STA's last-reported BSR (= "demand") and counts the UL data bytes it serves within that window; at SP end an SP is flagged "starved" if it had demand yet served < SP_STARVATION_RATIO of that demand.
// These feed the REALISTIC sp_with_starvation_count / sp_utilization obs + reward signals — the AP-observable fingerprint of an energy-starved STA (gets its SP, can't drain it).
#define SP_DEMAND_MIN_BYTES 1   // BSR total >= this at SP start => the STA had demand (D2)
#define SP_STARVATION_RATIO 0.5 // served < 0.5*demand within an SP => that SP was starved (D3)
#define SP_MIN_AWAKE_US 1000    // ignore awake windows < 1 ms (beacon-RX blips, not real SPs)

#endif // TWT_CONSTANTS_H