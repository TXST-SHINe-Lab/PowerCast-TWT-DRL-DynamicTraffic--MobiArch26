# TWT-PowerCast: Deep RL for 802.11ax TWT Scheduling with RF Energy Harvesting

<p align="center">
  <img src="https://img.shields.io/badge/NS--3-3.44-blue" alt="NS-3 Version">
  <img src="https://img.shields.io/badge/Python-3.11-green" alt="Python Version">
  <img src="https://img.shields.io/badge/WiFi-802.11ax-orange" alt="WiFi Standard">
  <img src="https://img.shields.io/badge/RL-Recurrent%20PPO-red" alt="RL Algorithm">
</p>

A reinforcement-learning framework that co-designs **802.11ax Target Wake Time (TWT)
scheduling** with **PowerCast RF energy harvesting**. An NS-3 simulation runs the full
WiFi MAC/PHY for a heterogeneous cell of traditional stations (IoT / camera / voice /
video) and **REHD** nodes — battery-free sensors that harvest the access point's RF
emissions while asleep and spend the stored charge on uplink. A recurrent PPO agent
decides, on every scheduling call, the TWT group configuration, the station→group
assignment, and a dedicated **Power Delivery Window (PDW)** during which the AP beams
energy to the sleeping harvesters. All NS-3 ↔ Python communication is over `ns3-ai`
shared memory (no sockets / files on the hot path).

The control problem is a partially-observed multi-agent scheduling task: the AP must keep
throughput-hungry STAs served *and* keep the energy-harvesting sensors alive, while it is
**blind to which station is which** — it must infer the hidden REHD/traffic layout from the
realistic protocol signals it can actually observe.

**Author**: Ahmed Maksud — SHINE Lab, Texas State University
**PI**: Prof. Marcelo Menezes De Carvalho

---

## Quick start

> **Directory name is fixed.** This project must live at
> `contrib/ai/examples/rl-twt-powercast/`. The name is hardcoded — in
> `twt-constants.h` (config + output paths), in `CMakeLists.txt` (where the pybind
> `.so` is deployed), and in `setup_fresh_ns3.sh` (the `add_subdirectory()` line) —
> so clone it with that explicit target name, exactly as shown below. All three
> places *check* the name and fail loudly on a mismatch rather than silently
> building nothing.

On a fresh NS-3.44 + ns3-ai checkout, set up once (applies the required NS-3 patches and
registers the build subdir), build, then run the whole experiment with one driver:

```bash
cd $NS3_ROOT/contrib/ai/examples/
git clone <repo-url> rl-twt-powercast     # <-- the target name is REQUIRED
cd $NS3_ROOT

bash contrib/ai/examples/rl-twt-powercast/setup_fresh_ns3.sh   # one-time: patches + subdir
# ... configure + ./ns3 build (see "Build" below) ...
bash contrib/ai/examples/rl-twt-powercast/run_pipeline.sh      # all 4 stages, end-to-end
```

`run_pipeline.sh` runs the full experiment in order: **(1)** harvest/Vcap plots (shows RF
harvesting is happening), **(2)** full EDA + obs-normalization fit (prompts before changing
the live norms), **(3)** full training (lstm_ppo + asymmetric critic, v5 reward), **(4)**
full evaluation vs baselines. Each stage is independently toggleable and tunable via env
vars — e.g. `RUN_TRAIN=0 RUN_EVAL=0 bash run_pipeline.sh` for just plots + EDA, or
`TRAIN_BATCHES=10 EDA_BATCHES=3 bash run_pipeline.sh` for a quick smoke of all four.

For an exact, paper-pinned reproduction (the exact parameters that produced the committed
figures/checkpoint) including action-table carving and figure regeneration, use
`reproduce.sh` instead — it wraps `run_pipeline.sh` with paper-exact arguments and adds the
build/carve/figs bookends.

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. System architecture](#2-system-architecture)
- [3. The physical scenario](#3-the-physical-scenario)
- [4. NS-3 simulation source files](#4-ns-3-simulation-source-files)
  - [`twt-constants.h`](#twt-constantsh)
  - [`pb-twt-core.h`](#pb-twt-coreh)
  - [`ph-harvester-hardware.h` / `.cc`](#ph-harvester-hardwareh--cc)
  - [`ph-deployment-helper.h` / `.cc`](#ph-deployment-helperh--cc)
  - [`twt-simulation-config.h` / `.cc`](#twt-simulation-configh--cc)
  - [`twt-trace-callbacks.h` / `.cc`](#twt-trace-callbacksh--cc)
  - [`twt-metrics.h` / `.cc`](#twt-metricsh--cc)
  - [`pb-twt-wrapper-ns3.h` / `.cc`](#pb-twt-wrapper-ns3h--cc)
  - [`pb-twt-interface.cc`](#pb-twt-interfacecc)
  - [`pb_twt_wrapper_py.py`](#pb_twt_wrapper_pypy)
  - [`twt-powercast-main-simulation.cc`](#twt-powercast-main-simulationcc)
- [5. How the trace callbacks work](#5-how-the-trace-callbacks-work)
- [6. Metrics collection: two-level design](#6-metrics-collection-two-level-design)
- [7. C++–Python bridge: NS3-AI shared memory](#7-cpython-bridge-ns3-ai-shared-memory)
- [8. RL formulation](#8-rl-formulation)
- [9. Policy & the asymmetric critic](#9-policy--the-asymmetric-critic)
- [10. Exploratory data analysis (the EDA stage)](#10-exploratory-data-analysis-the-eda-stage)
- [11. Build](#11-build)
- [12. Train](#12-train)
- [13. Evaluate](#13-evaluate)
- [14. Results](#14-results)
- [Quick reference](#quick-reference)
- [Notes](#notes)

---

## 1. Overview

This project implements a **closed-loop reinforcement-learning controller** for IEEE
802.11ax (WiFi 6) Target Wake Time scheduling, extended with a **PowerCast RF energy
harvesting** model for a subset of the stations (REHDs). An NS-3 simulation runs the
heterogeneous WiFi cell; every `TWT_UPDATE_INTERVAL_BI` beacon intervals it pauses, ships
per-STA realistic + oracle metrics to a Python controller over shared memory, blocks for a
new schedule, applies it (TWT group timing *and* the PDW power-delivery window), and
resumes. The Python side is normally the trained recurrent-PPO agent, but the same
`TWTWrapper` API also drives the analytical baselines, fixed-action sweeps, and the EDA
random-action collector.

**Key numbers (canonical / shipped configuration):**

- 20 STAs total — canonical mix **12 non-REHD + 8 REHD**; training varies the split
  4–16 REHD (total fixed at 20) so the energy dimension differs per scenario
- 5 device classes: IoT, Camera, Voice, Video (non-REHD) + REHD (4 hardware sub-types)
- Beacon interval: **102.4 ms**; agent decision cadence: every `TWT_UPDATE_INTERVAL_BI` =
  **25 BIs** (~2.56 s)
- Episode length: `DURATION_IN_UPDATE` = **100** agent-update steps (~4.3 min simulated);
  warm-up before metrics/TWT start at `TWT_UPDATE_START_BI` = 90 BI
- Action space: **24 schedules × 25 assignments × 10 PDW levels** (EDA-carved grid, see
  §8)
- Compile-time ceilings: `MAX_NUM_STA = 32`, `MAX_NUM_TWT_GROUPS = 16`

---

## 2. System architecture

```
 ┌──────────────────────────── Python (PPO) ────────────────────────────┐
 │  twt_batch_orchestrator.py   master policy + Adam + asymmetric critic  │
 │        │  pickle({name, kwargs, state_dict})                           │
 │        ├── worker 0 ─ spawn ─► TWTWrapper ─┐                           │
 │        ├── worker 1 ─ spawn ─► TWTWrapper ─┤  ns3-ai shared memory     │
 │        └── worker N ─ spawn ─► TWTWrapper ─┘  (MySeg_<seed>, …)        │
 │  drain rollouts → join → PPO update on merged rollouts → checkpoint    │
 └───────────────────────────────────┬───────────────────────────────────┘
                                      ▼
 ┌──────────────────────────── NS-3 (C++) ──────────────────────────────┐
 │  twt-powercast-main-simulation.cc                                      │
 │   every 25 beacons: build EnvStruct from trace counters + harvester    │
 │   state → send over SHM → receive ActionStruct → reprogram each STA's  │
 │   TWT timing (WifiMac::SetTwtSchedule) and the PDW                      │
 │  PowerCast energy model: time-switching harvest + energy-gated uplink   │
 └───────────────────────────────────────────────────────────────────────┘
```

Each worker runs one episode in a fresh `spawn`ed process, pushes its rollout to the
parent, and exits with `os._exit(0)` (skipping Boost.Interprocess static destructors that
would otherwise hang on mapped SHM). NS-3 is deterministic per `(seed, action)`, so
matched-seed comparisons are exact and the Common-Random-Numbers groups (`--crn-groups`)
give PPO clean per-scenario advantages.

**Per-step data flow**, one agent decision:

```
PHY/MAC events (TX/RX, PHY-state transitions, BSR reports, drops, ...)
   │
   └─► trace callbacks (twt-trace-callbacks.cc) ──► per-STA global C arrays
                                                            │
                          every 25 BI (TWT_UPDATE_INTERVAL_BI)
                                                            │
                     TwtMetrics::LogAndSendCallLevelMetrics()
                             │ PopulateStaObservationRaw() fills EnvStruct
                             │  (per-STA StaRealisticMetrics + StaOracleMetrics
                             │   + harvester Vcap/harvested/consumed state)
                             ▼
                     TWTWrapper::RequestTWTSchedule(EnvStruct)
                             │ ns3-ai shared memory (blocks NS-3 thread)
                             ▼
                     Python: TWTWrapper.step() unblocks with env_dict
                     twt_spawn_worker builds the 7-feature/STA obs, calls the
                     policy, decodes (sched, assign, pdw) → ActionStruct
                             │ shared memory
                             ▼
                     TwtNetworkSetup::ApplyTWTSchedule(ActionStruct)
                       → per-group WifiMac::SetTwtSchedule(...)
                       → SetupPowerDeliveryWindow's m_pdwDurationMs updated
```

Because every value NS-3 reports is a **cumulative** counter (never reset mid-run), the
Python side is responsible for turning them into per-step deltas — see §6.

---

## 3. The physical scenario

**Topology (fixed 20-station token set).** A 14 m × 14 m room, AP at the origin on the
**2.4 GHz ISM band** (the only band the PowerCast rectenna harvests). The canonical mix is
**12 non-REHD + 8 REHD**; during training the split is varied per scenario (4–16 REHD,
total fixed at 20) so the energy dimension differs across runs in a way the agent can
observe through aggregate signals.

**Device classes (5).** Each non-REHD STA draws a class — **IoT**, **Camera**, **Voice**,
or **Video** — with class-specific traffic models and packet deadlines (latency-as-expiry).
**REHD** sensors draw one of four hardware sub-types (T1–T4, differing capacitor/voltage
class) and sit in a `[1.5, 4.75] m` harvest annulus around the AP (the 4.75 m outer edge
is where the datasheet's -12 dBm sensitivity floor is reached at 36 dBm EIRP). The AP is **blind to
device class** — class is in neither the observation nor the reward.

**RF & energy harvesting (PowerCast P21XXCSR rectenna).**

- AP: 30 dBm conducted + 6 dBi → **36 dBm EIRP** (the FCC Part 15.247 point-to-multipoint
  maximum); non-REHD STAs 16 dBm; REHDs 10 dBm
  conducted + 6 dBi rectenna gain. Shared `FriisPropagationLossModel` @ 2.4 GHz.
- **Time-switching harvest:** a REHD harvests RF energy *only while its PHY is asleep*
  (TWT doze); the received power from every concurrent transmitter is computed analytically
  through the same Friis path as the comms link.
- **Energy-gated uplink:** a REHD transmits as many queued packets as its capacitor can
  afford while staying above `Vmin` (strict two-threshold capacitor, `DEEP_RATIO = 1.0`);
  when depleted, its uplink queue is *blocked*
  at the MAC (never by forcing PHY sleep, which would desync the TWT controller) and
  unblocked the instant it can afford a packet. Packets that age out past the per-class
  deadline expire.

**Power Delivery Window (PDW).** The AP barely transmits in an uplink-dominated cell, so
left alone the harvesters starve. The PDW is the controllable energy lever: after each
beacon the AP delivers a continuous RF energy beam over `[5, pdw_end]` ms; every REHD
asleep in that window harvests it. Energy is credited directly (linear in window duration)
rather than as packets, so it cannot leak onto the channel or collide with comms SPs.
Opening the PDW costs comms airtime (the schedule is compressed into `[pdw_end, 95]`), so
the agent must trade harvester throughput against everyone else's.

**Dynamics.** Stations random-walk (REHDs stationary). Traffic is non-stationary: each
episode is split into **5 segments**; at every boundary an independent 50/50 coin sets the
segment **over-** (×1.6) or **under-saturated** (×0.4) around the calibrated load anchor
`trafficScale = 1.9`, and every non-REHD STA re-draws its traffic parameters. Over vs under
have genuinely different optimal schedules, so tracking the regime is a real lever.

---

## 4. NS-3 simulation source files

All files below live in the project root alongside `CMakeLists.txt`. `twt-constants.h` is
the single source of truth for every numerical constant — nothing timing/energy/sizing-
related is hardcoded elsewhere.

### `twt-constants.h`

**Role**: every compile-time constant in the simulation, grouped by concern. Editing scale
or timing means editing only this file (and rebuilding).

| Group                     | Examples                                                                                                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Directory resolution      | `TWT_EXAMPLE_DIR_NAME = "rl-twt-powercast"` — the hardcoded contract behind the fixed clone-name rule above                                                                                                                          |
| Network sizing            | `MAX_NUM_STA = 32` (ceiling), `ACTIVE_NUM_STA = 16` (default), `MAX_NUM_TWT_GROUPS = 16`, `MAC_ADDR_LEN = 6`                                                                                                                         |
| Room / RF deployment      | `DEFAULT_ROOM_LENGTH = 14.0` m, `AP_TX_POWER_DBM = 30.0`, `AP_TX_GAIN_DBI = 6.0`, `REHD_TX_POWER_DBM = 10.0`, `REHD_RX_GAIN_DBI = 6.0`, `REHD_SPAWN_RADIUS_M = 4.75`, `REHD_SPAWN_MIN_RADIUS_M = 1.5`, `WIFI_CENTER_FREQ_HZ = 2.4e9` |
| REHD power budget         | `REHD_IDLE_POWER_W = 1 µW`, `REHD_RX_POWER_W = 7.5 mW`, `REHD_SLEEP_POWER_W = 1 µW`                                                                                                                                                  |
| Beacon / timing           | `BEACON_INTERVAL_MS = 102.4`, `TWT_UPDATE_INTERVAL_BI = 25`, `DURATION_IN_UPDATE = 100`, `TWT_UPDATE_START_BI = 90`, `TWT_SETUP_TIME_BI = 75`                                                                                        |
| PDW burst                 | `PDW_BCAST_WIFI_MODE = "HeMcs0"`, `PDW_BEACON_LEAD_US = 5000`                                                                                                                                                                        |
| Metrics window            | `METRICS_START_TIME_BI = 75`, `APP_START_TIME_MIN_BI = 25`, `APP_START_TIME_MAX_BI = 45`                                                                                                                                             |
| WiFi PHY/MAC              | `DEFAULT_MCS = 4`, `DEFAULT_GUARD_INTERVAL_NS = 800`, `DEFAULT_CHANNEL_WIDTH_MHZ = 20`, `RTS_CTS_THRESHOLD = 2347`, `MAX_MISSED_BEACONS = 0xFFFFFFFF`, `BLOCK_ACK_THRESHOLD = 1`                                                     |
| Energy model (mA)         | `PHY_STATE_IDLE_MA = 50`, `PHY_STATE_CCA_BUSY_MA = 50`, `PHY_STATE_RX_MA = 66`, `PHY_STATE_TX_MA = 232`, `PHY_STATE_SLEEP_MA = 0.12`, `BATTERY_VOLTAGE_V = 3.0`                                                                      |
| Traffic                   | `MAX_QUEUE_SIZE_BYTES = 65536`, `PAYLOAD_SIZE_BYTES = 1400`                                                                                                                                                                          |
| RNG streams               | `RNG_INITIAL_STREAM_ID = 100`, `RNG_DEFAULT_RUN_NUMBER = 1`, `RNG_INITIAL_SEED = 10`                                                                                                                                                 |
| Dynamic mobility          | `ENABLE_DYNAMIC_MOBILITY`, `MOBILITY_SPEED_MIN/MAX_MPS`, `MOBILITY_DIR_CHANGE_S`, `MOBILITY_BUFFER_M = 0.5`, `REHD_BUFFER_M = 0.25`                                                                                                  |
| Dynamic traffic           | `ENABLE_DYNAMIC_TRAFFIC`, `N_TRAFFIC_SEGMENTS = 5`, `SEG_OVER_MULT = 1.6`, `SEG_UNDER_MULT = 0.4`                                                                                                                                    |
| Per-class traffic ranges  | `IOT_/CAMERA_/VOICE_/VIDEO_STREAM_RATE_MIN/MAX_BPS` + on/off timing                                                                                                                                                                  |
| Service-Period accounting | `SP_DEMAND_MIN_BYTES = 1`, `SP_STARVATION_RATIO = 0.5`, `SP_MIN_AWAKE_US = 1000`                                                                                                                                                     |

Also defines `TwtExampleDir()` / `TwtExamplePath(rel)` / `TwtResultsPath(rel)` — resolve
paths relative to this example's directory (source inputs) or its single generated-output
root `results/` (everything a run produces, so `rm -rf results/` is always a clean slate).

---

### `pb-twt-core.h`

**Role**: every C++ struct that crosses the NS-3 ↔ Python boundary, plus the `sta_id`-keyed
split between what a **real** AP could observe and what only the **simulator** knows.

**`StaRealisticMetrics`** — everything obtainable through standard 802.11ax/k mechanisms,
**time-local or cumulative-with-Python-computed-deltas**, never containing device class or
energy state:

- Identification: `sta_id`, `sta_mac[6]`, `is_active`
- 802.11ax BSR per Access Category: `bsr_queue_ac_{be,bk,vi,vo}` (quantized 0–255) +
  `bsr_scaling_factor`
- AP-observable RX counters: `rx_fragment_count`, `fcs_error_count`
- 802.11k link measurement: raw `rcpi`/`rsni` + converted `rssi_dbm`/`snr_db`/
  `link_margin_db` (`INT8_MIN` sentinel = no measurement yet, since `int8_t` can't hold NaN)
- MAC-header observations: last RX frame type/subtype, MCS, NSS, channel width, guard
  interval, Power-Management bit, last-RX TSF timestamp
- AP-derived: `bytes_received_at_ap`, `packets_received_at_ap`, per-AC byte/packet counts,
  `airtime_used_us`
- `user_priority` — the QoS UP the AP reads directly off received frames
- **QoEH per-Service-Period accounting** (added for the energy work): `bytes_rx_at_ap_in_sp`
  / `packets_rx_at_ap_in_sp` (τ served within the STA's current SP), `sp_completed_count`,
  `sp_with_demand_count`, `sp_with_starvation_count` — the AP-observable fingerprint of an
  energy-starved STA (gets its SP, can't drain it)
- AP DL accounting: `dl_bytes_to_sta`, `dl_duration_us_to_sta`, `dl_unicast_count_to_sta`
- Metadata: `observation_time_ms`, `observation_sequence_num`

**`StaOracleMetrics`** — simulation-only ground truth; **never** reaches the actor, used
only for reward computation and offline analysis:

- STA-local TX counters: `tx_fragment_count`, `tx_failed_count`, `tx_retry_count`,
  `ack_failure_count`
- Legacy energy bookkeeping: `total_energy_consumed_mj`, `awake_time_ms`, `sleep_time_ms`,
  `duty_cycle`
- Application layer: `packets_generated`, `packets_enqueued`, `bytes_generated`
- Queue drops: `mpdu_drops_expired`, `mpdu_drops_queue_full`, `psdu_response_timeouts`
- TX side: `packets_transmitted`, `bytes_transmitted`, A-MPDU stats
- Queue state, latency (`avg_latency_ms` etc.), simulation-only position
- **PowerCast harvester state** (the QoEH oracle block): `vcap_v`, `vcap_max_v`,
  `e_nominal_j`, `e_sunk_j`, `e_avail_j`, `harvested_total_j`, `consumed_total_j`,
  `t_active_us`, `unpowered_tx_events`, `output_enabled`, `harvester_phase`, plus an
  AP-side RF-delivery estimate `est_rf_energy_delivered_j`
- Device characteristics moved here from realistic in the 2026-06-07 signal audit:
  `device_class`, `nominal_msdu_size`, `mean_data_rate_kbps`, `delay_bound_ms`,
  `tx_power_dbm` — all realistic-*eligible* in principle (802.11e ADDTS / 802.11h-k TPC)
  but this sim reads them from config rather than modeling the signaling, so they're kept
  oracle for honesty

**`StaEnvStruct`** pairs one `StaRealisticMetrics` + one `StaOracleMetrics` per STA.
**`EnvStruct`** is the top-level C++→Python message: metadata (`num_sta`,
`simulation_time_sec`, `observation_timestamp_ms`, `observation_count`,
`beacon_interval_ms`), global AP TX-state diagnostics (`ap_tx_power_dbm`,
`dl_broadcast_*`), and `sta_observations[MAX_NUM_STA]`. Aggregate TWT/channel stats are
deliberately **not** included — the Python controller already knows the group assignments
from the action it sent, and every aggregation is computable from the per-STA data.

**`TwtGroupConfig`** — one TWT group's timing: `group_id`, `twt_wake_interval_ms`,
`twt_wake_duration_ms`, `twt_sp_offset_ms`, `num_stas_assigned`.
**`StaGroupAssignment`** — `sta_id` → `assigned_twt_group` + `enable_twt`.
**`ActionStruct`** — the Python→C++ message: metadata, the **PDW** 3rd action head
(`pdw_duration_ms` = `pdw_end`), `twt_group_configs[MAX_NUM_TWT_GROUPS]`, and
`sta_group_assignments[MAX_NUM_STA]`.

---

### `ph-harvester-hardware.h` / `.cc`

**Role**: `PowercastEnergyHarvester`, an `ns3::energy::EnergyHarvester` subclass modeling
one REHD's P21XXCSR-EVB rectenna + boost converter + storage capacitor. No defaults — a
harvester is invalid until `Configure(capClass, voltClass)` is called.

**Hardware classes** (`CapacitorClass` × `VoltageClass`, cross product = 9 combinations):

| Capacitor class | Value     | Voltage class | Vmax / Vmin |
| --------------- | --------- | ------------- | ----------- |
| `CLASS_A`       | 500 µF    | `CLASS_1`     | 1.2 V       |
| `CLASS_B`       | 2200 µF   | `CLASS_2`     | 0.9 V       |
| `CLASS_C`       | 20 000 µF | `CLASS_3`     | 0.7 V       |

**Energy state**: `m_vcap` (current V), `m_nominalEnergy = ½·C·V²`, `m_sunkEnergy` (unusable
reserve below `DEEP_RATIO·Vmin`), `m_availableEnergy = nominal − sunk`. `m_totalActiveTime_us`
tracks cumulative time above `Vmin` for the `S_r` sustainability metric.

**Key methods**:

- `SetHarvestedEnergy(rxPowerDbm, duration)` — integrates the P21XXCSR Band-6 (2450 MHz)
  RF-to-DC efficiency curve (digitized from the datasheet: 0 below −12 dBm, rising to
  a ~46% peak at +8 dBm, held flat above) over `duration`, credits the capacitor.
- `SetConsumedEnergy(txPowerDbm, duration)` / `ConsumeDcEnergy(energyJ)` — debit radiated
  TX energy (`/ PA_EFFICIENCY`) or small awake-state (IDLE/RX/CCA) housekeeping drain.
- `CanSustainTransmission(txPowerDbm, duration) const` — the energy-audit gate: can the
  capacitor pay for this TX without dropping below the 0.8·Vmin hard floor? No hysteresis —
  a REHD unblocks the instant it can afford one more packet.
- `GetVcap()`, `ComputePhase()` (0=charging / 1=active / 2=discharging / 3=protection),
  `IsOutputEnabled()`, `GetTotalHarvestedEnergy()`, `GetTotalConsumedEnergy()`.

This class is deliberately decoupled from *where* the RF comes from — `twt-simulation- config.cc`'s time-switching harvest loop (§5) is what decides when to call
`SetHarvestedEnergy`.

---

### `ph-deployment-helper.h` / `.cc`

**Role**: `PowercastEnergyHarvesterHelper`, an `EnergyHarvesterHelper` subclass for
installing harvesters at scale. Not the code path this repo's REHD templates actually use
(that's `REHD_TYPE_TEMPLATES` in `twt-simulation-config.h`, applied directly via
`Configure()`), but available for file-based / templated deployment:

- `LoadConfigurationFromFile(filename)` — parses `NodeID CapacitorClass VoltageClass` lines
  (see `ph-harvester-config.txt`)
- `SetDefaultTemplate(DeploymentTemplate)` — `UNIFORM_A1` / `UNIFORM_B2` /
  `MIXED_DEPLOYMENT` / `HIGH_CAPACITY` / `LOW_POWER`
- `SetAssignmentMethod(AssignmentMethod)` — `SEQUENTIAL` / `RANDOM` / `FILE_BASED` /
  `MANUAL`, used as the fallback for nodes with no explicit `ConfigureNode()` call
- `ConfigureNode(nodeId, capClass, voltClass)`, `Install(EnergySourceContainer)` →
  `EnergyHarvesterContainer`
- `ExportConfigurationToFile`, `GetConfigurationStatistics`, `ValidateConfiguration`

---

### `twt-simulation-config.h` / `.cc`

**Role**: the largest file in the project — network topology, per-device-class traffic
profiles, REHD hardware/traffic templates, the dynamic-mobility and dynamic-traffic
schedulers, the time-switching harvest model, and `ApplyTWTSchedule()` /
`SetupPowerDeliveryWindow()` / `DeliverPdwPower()`, which turn an `ActionStruct` into live
NS-3 TWT + PDW state.

**`DeviceClass` enum** — `DEVICE_IOT_SENSOR` (0), `DEVICE_VIDEO_CAMERA` (1),
`DEVICE_VOICE_ASSISTANT` (2), `DEVICE_VIDEO_STREAMING` (3), `DEVICE_REHD` (4).

**Non-REHD traffic profiles** (`StaApplicationConfig`, drawn with a lopsided class mix —
IoT 40% / Voice 25% / Camera 20% / Video 15% — so the network has genuine mice-vs-elephants
heterogeneity):

| Class           | Rate    | Pattern                   | Deadline | Notes                           |
| --------------- | ------- | ------------------------- | -------- | ------------------------------- |
| IoT Sensor      | 40 kbps | near-CBR                  | 500 ms   | tiny telemetry mouse            |
| Video Camera    | 4 Mbps  | low-variation CBR         | 2 s      | mains-powered, elastic          |
| Voice Assistant | 64 kbps | bursty (2 s on / 3 s off) | 1 s      | latency-critical                |
| Video Streaming | 12 Mbps | bursty, high variance     | 3 s      | the elephant — hogs a shared SP |

**REHD hardware/traffic sub-types** (`RehdType` T1–T4, one drawn uniformly per REHD at
episode start; hardware is fixed per sub-type, traffic parameters jitter within range):

| Sub-type         | Capacitor / Voltage      | Rate range | Burst period | Deadline                        |
| ---------------- | ------------------------ | ---------- | ------------ | ------------------------------- |
| T1 Telemetry     | A / 1 (500 µF, 1.2 V)    | 12–18 kbps | 0.4–0.6 s    | 30 s (tolerant)                 |
| T2 Motion        | B / 2 (2200 µF, 0.9 V)   | 24–34 kbps | 0.25–0.4 s   | 10 s                            |
| T3 Asset Tracker | C / 1 (20 000 µF, 1.2 V) | 60–80 kbps | 0.3–0.5 s    | 30 s                            |
| T4 Vibration     | B / 3 (2200 µF, 0.7 V)   | 32–46 kbps | 0.2–0.3 s    | 5 s (tightest — most demanding) |

Each burst is a fixed 50 ms OnOff pulse sized so ~one packet goes out per burst — the
REHD's uplink queue builds a small backlog between bursts that the energy-gated MAC then
has to drain.

**`TwtSimulationConfig`** — the top-level config struct: `simId`, `randSeed`,
`nStations` (non-REHD count) / `nRehd`, `trafficScale` (the α in the saturation anchor,
default 1.9), `beaconInterval_s`, TWT setup timing, `twtNominalWakeDuration` (95 ms — the
pre-agent default is deliberately permissive, i.e. "no TWT"), `pdwDurationMs`, and the
`TI_currentModel_mA` energy-model map. Its constructor derives `simulationTime_ms`,
`firstTwtSpStart`, and `keepTrackOfMetricsFrom_ms` from the BI-denominated constants above.

**`TwtNetworkSetup`** — orchestrates the whole build:

1. `CreateNodes()` — AP + STA + P2P-server nodes; `InitializeHeterogeneousNetwork()` draws
   each STA's class (and REHD sub-type)
1. `ConfigureWifi()` — 802.11ax HE PHY/MAC, per-class TX power/gain overrides for REHDs,
   beacon jitter disabled (TWT/PDW timing must be exact), RNG stream assignment
1. `SetupMobility()` / `ScheduleDynamicMobility()` — rejection-sampled initial placement +
   optional per-STA random walk with wall/buffer reflection
1. `SetupEnergyHarvesting(configFile)` — installs `BasicEnergySource` +
   `PowercastEnergyHarvester` on every REHD node from the hardware-class catalog, wires the
   network-wide `PhyTxBegin`/`PhyTxEnd` callbacks that drive the time-switching harvest
   (see §5), and the per-AC `MaxDelay`/`Expired` binding for the REHD uplink queues
1. `SetupApplications()` / `ScheduleDynamicTraffic()` — installs the per-class OnOff/UDP
   apps and the segmented-coin traffic re-draw scheduler
1. `SetupTwtSchedule()` / `SetupPowerDeliveryWindow()` — the pre-agent default TWT (or the
   diagnostic grouped-sleep test schedule) and arms the beacon-anchored PDW hook
1. `ApplyTWTSchedule(const ActionStruct&)` — per agent step: for each active TWT group,
   converts `twt_wake_interval_ms`/`twt_wake_duration_ms`/`twt_sp_offset_ms` to `Time` via
   `Seconds(ms/1000.0)` (never `MilliSeconds()`, which truncates `102.4` to `102` and drifts
   the SP ~0.4 ms/BI), then calls the NS-3 TWT MLME (`WifiMac::SetTwtSchedule`) for every
   STA in the group. Also updates `m_pdwDurationMs` from the action's PDW head.

**Time-switching harvest model.** A REHD is single-antenna: transceiver *xor* rectenna, so
it can only harvest while its PHY is asleep. `OnPhyTxBegin`/`OnPhyTxEnd` are hooked network-
wide (every node is a potential ambient RF source); on each TX's end, every REHD currently
`IsStateSleep()` is credited `SetHarvestedEnergy(CalcRxPower(...), duration)` using the
*same* `FriisPropagationLossModel` the channel uses, and a transmitting REHD debits its own
`SetConsumedEnergy` for the frame it just sent. `ReconcileRehdLock()` re-evaluates the
depleted/blocked state after every capacitor change and blocks/unblocks the REHD's uplink
MAC queue accordingly (never by forcing PHY sleep, which would fight the TWT controller).

---

### `twt-trace-callbacks.h` / `.cc`

**Role**: every NS-3 trace-sink function in the project, plus the global per-STA C arrays
they accumulate into. `TwtMetrics` reads these arrays on demand; nothing here talks to
Python directly.

**How NS-3 trace sinks work**: objects like `WifiPhy`, `WifiMac`, and `Application` expose
named `TraceSource`s. A sink is a plain C++ function connected to one at simulation setup
time (`Config::Connect(<path>, MakeCallback(&Fn))`); NS-3 invokes it with the event's data
whenever that trace source fires.

**Representative global arrays** (all `[MAX_NUM_STA]`-indexed unless noted):

| Array                                                                            | Fired by                                                          | Tracks                                               |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------- |
| `awakeTimeElapsedForSta_ms_TI` / `sleepTimeElapsedForSta_ms_TI`                  | `PhyStateTrace_inPlace`                                           | time in each PHY macro-state                         |
| `packetsEnqueuedAtMacForSta`                                                     | `MacTxTrace`                                                      | oracle `packets_enqueued` (v4/v5 reward denominator) |
| `bsrQueueBytesAc{Be,Bk,Vi,Vo}ForSta`                                             | `BsrReceivedCallback`                                             | live 802.11ax BSR per AC                             |
| `bytesReceivedAc*ForSta` / `packetsReceivedAc*ForSta`                            | `ApPhyRxEndTrace`                                                 | AP-side per-AC UL traffic                            |
| `bytesRxAtApInSpForSta`, `spCompletedCountForSta`, `spWithStarvationCountForSta` | `PhyStateTrace_inPlace` (SLEEP edges) + `ApMonitorSnifferRxTrace` | per-Service-Period served/demand accounting          |
| `dlBytesToStaForSta`, `dlDurationUsToStaForSta`                                  | `ApPhyTxPsduBeginTrace`                                           | AP-known DL delivery per STA                         |
| `lastRssiDbmForSta`, `lastSnrDbForSta`                                           | `ApPhyRxEndTrace` / `ApMonitorSnifferRxTrace`                     | 802.11k-style link quality                           |
| `ampduCountForSta`, `ampduMpdusTotalForSta`                                      | `AmpduAggregationTrace`                                           | A-MPDU aggregation stats                             |
| `g_dlBroadcastBytes` / `*DurationUs` / `*Count` (global, not per-STA)            | `ApPhyTxPsduBeginTrace`                                           | beacon + PDW broadcast diagnostics                   |

`g_onApBeaconTx` is a `std::function<void()>` hook fired once per real AP beacon TX
(from `ApPhyTxPsduBeginTrace`); `TwtNetworkSetup::SetupPowerDeliveryWindow()` registers a
lambda there so the PDW burst is anchored to the *actual* beacon, not a computed grid that
could drift from the true TBTT.

---

### `twt-metrics.h` / `.cc`

**Role**: `TwtMetrics` packages the raw accumulator arrays into the two logging levels and
the `EnvStruct` sent over IPC — the bridge between the trace-callback layer and everything
downstream.

- `InitializeArrays()` — allocates every global array to `nTotal()` STAs
- `LogBiLevelMetrics()` — called every beacon interval starting at `METRICS_START_TIME_BI`;
  appends one CSV row per STA with every raw counter (fine-grained series for the EDA
  pipeline)
- `LogAndSendCallLevelMetrics() → EnvStruct` — called every `TWT_UPDATE_INTERVAL_BI` BIs;
  calls `PopulateStaObservationRaw()` per active STA (fills both `StaRealisticMetrics` and
  `StaOracleMetrics`, including a `GetHarvester(staId)` query for the REHD oracle block),
  writes the call-level CSV row, and returns the struct handed to
  `TWTWrapper::RequestTWTSchedule()`
- All values logged are **cumulative snapshots** — see §6 for why deltas live in Python

---

### `pb-twt-wrapper-ns3.h` / `.cc`

**Role**: `TWTWrapper`, an `ns3::Object` wrapping the `ns3-ai`
`Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>`. NS-3 is the **consumer** of shared memory
here (`SetIsMemoryCreator(false)`) — the Python side creates the segments in `reset()`.

- `Initialize()` — resolves the typed interface, disables vector mode (fixed-size struct),
  enables finish handling for clean teardown
- `RequestTWTSchedule(const EnvStruct&) → ActionStruct` — `CppSendBegin()` → copy `env`
  into the Cpp→Python segment → `CppSendEnd()` (unblocks Python) → `CppRecvBegin()` (blocks
  until Python replies) → read the `ActionStruct` → `CppRecvEnd()`. This call blocks the
  NS-3 simulation thread until Python responds; `run_pipeline.sh` / the spawn workers clean
  up stale `/dev/shm/My*` segments between runs so a crashed prior episode can't deadlock
  the next one.
- `EnableLogging(true, path)` — optional per-step CSV of the interaction (timestamp,
  success, `num_sta`, sim time, active-group count)

---

### `pb-twt-interface.cc`

**Role**: the `pybind11` module (`pb_twt_powercast_interface_py`) exposing every struct in
`pb-twt-core.h` as an importable Python class. Built by `./ns3 build` and imported by
`pb_twt_wrapper_py.py`.

Exposes `StaRealisticMetrics`, `StaOracleMetrics` (all fields read/write attributes),
`StaEnvStruct` (`.realistic` / `.oracle`), `EnvStruct` (metadata fields +
`.sta_observations` as a Python list), `TwtGroupConfig`, `StaGroupAssignment`,
`ActionStruct` (`.twt_group_configs` / `.sta_group_assignments` as lists), plus module-
level constants (`MAX_NUM_STA`, `MAX_NUM_TWT_GROUPS`, `MAC_ADDR_LEN`). `sta_mac` is exposed
as `py::bytes` with a getter/setter that does a proper `memcpy`; `sta_observations` uses the
`reference` return policy so Python-side mutation writes straight into the shared-memory
struct.

---

### `pb_twt_wrapper_py.py`

**Role**: the Python-side counterpart to `pb-twt-wrapper-ns3.cc`. Hides every `ns3-ai`
detail behind `reset()` / `step()` / `close()`; every controller in this repo (the PPO
worker, the analytical baselines, the EDA collector, the smoke tests) talks only to this
class.

```python
wrapper = TWTWrapper(verbose=True)
env = wrapper.reset(seed=9000)
while True:
    action = create_default_action(env["num_sta"], num_groups=2)
    env, done = wrapper.step(action)
    if done:
        break
wrapper.close()
```

**`reset(seed, rand_seed=None)`**: imports the pybind module lazily
(`_import_bindings()`), creates the `Ns3AiMsgInterface` as the memory **creator** (Python
side), spawns the `twt-powercast-main-simulation` binary as a subprocess with unique
`--segmentName`/`--cpp2pyMsgName`/`--py2cppMsgName`/`--lockableName` values, waits for the
first `EnvStruct`, and converts it to a plain dict via `_env_struct_to_dict()`.
`rand_seed` (distinct from the SHM-naming `seed`) is the scenario seed forwarded to
`--randSeed`, letting Common-Random-Numbers groups share a scenario across workers with
disjoint shared-memory segments.

**`step(action_dict)`**: `_dict_to_action_struct()` fills an `ActionStruct`, writes it and
signals NS-3, blocks in `_receive_env()` for the next `EnvStruct`, converts it back to a
dict, and checks the `done` flag.

**`env_dict`** keys: `num_sta`, `simulation_time_sec`, `observation_count`,
`beacon_interval_ms`, `sta_observations` (list of `{realistic: {...}, oracle: {...}}`
matching the C++ struct fields 1:1). **`action_dict`** keys: `num_active_twt_groups`,
`pdw_duration_ms`, `group_configs` (list of `{group_id, twt_wake_interval_ms, twt_wake_duration_ms, twt_sp_offset_ms}`), `sta_assignments` (list of `{sta_id, assigned_twt_group, enable_twt}`).

`get_num_sta()` / `get_num_twt_groups()` are convenience accessors; `close()` kills the
NS-3 subprocess and removes the four `/dev/shm/My*` segments for this worker's seed.

---

### `twt-powercast-main-simulation.cc`

**Role**: the NS-3 `main()`. Wires every module above together and drives the event loop.

**Selected CLI arguments** (all overridable; see `--PrintHelp` for the full list):

| Argument                                                           | Default              | Description                                                                                                     |
| ------------------------------------------------------------------ | -------------------- | --------------------------------------------------------------------------------------------------------------- |
| `randSeed`                                                         | 9000                 | scenario seed                                                                                                   |
| `nStations` / `nRehd`                                              | `ACTIVE_NUM_STA` / 0 | non-REHD / REHD counts; total must be ≤ `MAX_NUM_STA`                                                           |
| `trafficScale`                                                     | 1.9                  | saturation-anchor multiplier on non-REHD offered load                                                           |
| `pdwDurationMs`                                                    | 0.0                  | static PDW window (ignored once the agent starts sending its own PDW head) — must be in `[0, 94)`               |
| `enableDynamicTWT`                                                 | true                 | whether the periodic Python round-trip runs at all                                                              |
| `segmentName` / `cpp2pyMsgName` / `py2cppMsgName` / `lockableName` | `"My *"`             | shared-memory segment names — the spawn workers set these uniquely per episode                                  |
| `quietMode`                                                        | false                | suppress non-error `std::cout` in the update loop (workers set this `true`)                                     |
| `disableTraces`                                                    | false                | skip opening the CSV trace files (internal callbacks still fire and populate the `EnvStruct` arrays regardless) |
| `logVcap` / `vcapIntervalMs`                                       | false / 50           | diagnostic per-REHD `Vcap(t)` CSV, sampled independently of the agent cadence                                   |

**Startup sequence**: parse args → seed the RNG (before *any* `RandomVariable` is created)
→ `TwtNetworkSetup`: `CreateNodes → ConfigureWifi → SetupMobility → ScheduleDynamicMobility → ConfigureInternet → SetupEnergyHarvesting → SetupApplications → ScheduleDynamicTraffic → SetupTwtSchedule → EnablePcap` →
`TwtMetrics::InitializeArrays()` + BI/call-level logging init → `TWTWrapper::Initialize()`
→ connect every trace source (`ConnectSummaryTraces`, `ConnectE2ETraces`,
`ConnectQosMetricTraces`, `ConnectPhyStateTraces`, `ConnectTimeoutAndDropTraces`,
`Connect802dot11kTraces`) → `BsrManager` callback → schedule `PeriodicTWTUpdate` at
`twtUpdateStart_s` and `PeriodicBiLevelLogging` at `TWT_SETUP_TIME_BI` → `Simulator::Run()`.

`PeriodicTWTUpdate()` is the per-step loop: `LogAndSendCallLevelMetrics()` →
`RequestTWTSchedule(env)` (blocks for Python) → `ApplyTWTSchedule(action)` → reschedule
itself unless `DURATION_IN_UPDATE` steps have run. At the end of the run,
`CppSetFinished()` unblocks any pending Python `PyRecvBegin()`, harvester `Ptr<>`s are
released *before* `Simulator::Destroy()` (releasing after would UAF — the file-scope
harvester statics are torn down after NS-3 globals at process exit), and the sim exits.

---

## 5. How the trace callbacks work

The trace layer is the nervous system connecting hardware-level PHY/MAC events to the
Python-visible observation. Three representative chains:

**PHY state → energy + Service-Period accounting** (`PhyStateTrace_inPlace`, connected via
`ConnectPhyStateTraces`): fires with the state that just *ended* — a `SLEEP` record means
the STA was asleep from `start` until now. This single callback (a) accumulates
`awakeTimeElapsedForSta_ms_TI` / `sleepTimeElapsedForSta_ms_TI` for the energy-mA model,
and (b) on a SLEEP transition, finalizes the just-ended Service Period (compares
`spCurServedBytesForSta` against the BSR-demand snapshot taken at SP start, against
`SP_STARVATION_RATIO`) and opens the next one.

**Ambient TX → REHD harvest** (`OnPhyTxBegin` / `OnPhyTxEnd`, connected network-wide, not
just to REHDs — every node is a potential RF source for a sleeping REHD):
`OnPhyTxBegin` records the transmitter's conducted power and start time.
`OnPhyTxEnd` computes the elapsed duration, then (1) if the transmitter *was* a REHD,
debits its own capacitor via `SetConsumedEnergy` (gated by `CanSustainTransmission`), and
(2) for **every** REHD currently `IsStateSleep()`, computes the analytical RX power via the
shared `FriisPropagationLossModel` and credits `SetHarvestedEnergy`. `ReconcileRehdLock()`
runs after every capacitor change to keep the uplink-queue block/unblock state in sync.

**BSR → per-AC demand** (`BsrReceivedCallback`, hooked into `BsrManager`, a
patched-in singleton — see `mod-files/` below): fires whenever the AP receives a Buffer
Status Report (Trigger Frame response or QoS Data frame); maps the frame's TID to an
Access Category and updates `bsrQueueBytesAc{Be,Bk,Vi,Vo}ForSta[staId]`. This is the
realistic `bsr_be` feature's only data source — the agent's demand signal.

At every controller-update step, `PopulateStaObservationRaw()` reads all of these arrays
(NS-3 is single-threaded, so this is atomic with respect to the callbacks) into the
`EnvStruct`. Because every array is cumulative, the Python side always subtracts the
previous step's snapshot to get the per-step deltas actually fed to the observation/reward
(see `twt_spawn_worker.py`'s `_build_per_sta_features` and `reward_functions.py`).

---

## 6. Metrics collection: two-level design

|              | BI-level                                            | Call-level                                                                       |
| ------------ | --------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Trigger**  | every beacon interval, from `METRICS_START_TIME_BI` | every `TWT_UPDATE_INTERVAL_BI` BIs (each agent step)                             |
| **Method**   | `TwtMetrics::LogBiLevelMetrics()`                   | `TwtMetrics::LogAndSendCallLevelMetrics()`                                       |
| **Output**   | `data-log/` BI-level CSV                            | `data-log/` call-level CSV **+** `EnvStruct` over IPC                            |
| **Consumer** | `twt_eda_collect.py` / the EDA pipeline             | the live Python controller                                                       |
| **Content**  | one row per STA per BI, all raw cumulative counters | one row per STA per step, same counters, aligned to the agent's decision cadence |

**Why cumulative?** The simulation never resets a counter mid-run, so both levels are
monotonically increasing. Delta computation — throughput/step, energy/step, drop-rate/step
— is kept entirely in Python, where it's simpler to vectorize, easier to debug, and
trivially reproducible from a raw CSV. `twt_normalizer.py`'s running z-score operates on
those deltas, not on the raw cumulative values.

---

## 7. C++–Python bridge: NS3-AI shared memory

`ns3-ai`'s `Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>` creates three POSIX shared-
memory segments per simulation instance, named from the worker's `--segmentName` etc.:

| Segment              | Direction    | Content                   |
| -------------------- | ------------ | ------------------------- |
| `MyLockable_<seed>`  | both         | synchronization primitive |
| `MyCpp2PyMsg_<seed>` | C++ → Python | `EnvStruct`               |
| `MyPy2CppMsg_<seed>` | Python → C++ | `ActionStruct`            |

**Protocol** (strictly alternating): C++ writes `EnvStruct` → signals Python → C++ blocks
on `CppRecvBegin()` → Python reads `EnvStruct`, computes `ActionStruct` → Python writes it,
signals C++ → C++ unblocks, reads the action, simulation resumes.

**Bridge stack, top to bottom:**

```
RL agent / analytical baseline / EDA collector
    │  Python dict (env_dict / action_dict)
    ▼
TWTWrapper  (pb_twt_wrapper_py.py)
    │  _env_struct_to_dict() / _dict_to_action_struct()
    ▼
pb_twt_powercast_interface_py  (pb-twt-interface.cc, built by ./ns3 build)
    │  identical C-level struct layout on both sides
    ▼
POSIX shared memory  (/dev/shm/My*)
    ▼
TWTWrapper  (pb-twt-wrapper-ns3.cc)  — CppSendBegin/End, CppRecvBegin/End
    ▼
Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>
    ▼
twt-powercast-main-simulation.cc  (RequestTWTSchedule → ApplyTWTSchedule)
```

**Stale-segment cleanup.** A crashed prior run leaves `/dev/shm/My*` behind and the next
`reset()` on the same seed will deadlock. Each spawn worker removes its own four segments
in `finally:` (before *and* after the episode); `run_pipeline.sh` sweeps leftovers between
stages.

**NS-3 source patches (`mod-files/`).** Stock NS-3 3.44's WiFi module exposes neither
real-time BSR data nor a unilateral-TWT MLME call, so `mod-files/` carries the patches this
project needs applied before the first build: a new `BsrManager` singleton
(`bsr-manager.{h,cc}`) hooked into `qos-frame-exchange-manager.cc` so `RecordBsr()` fires
on every received Buffer Status Report; a patched `sta-wifi-mac.{h,cc}` adding
`GetTimeTillNextBeacon()` and unilateral TWT MLME calls; and `wifi-twt-agreement.{h,cc}`
(from the [gtgnan/wifiTwt](https://github.com/gtgnan/wifiTwt) project, adapted for the
NS-3.44 API). `setup_fresh_ns3.sh` applies all of it in one shot; `mod-files/` is excluded
from the formatting/comment conventions applied to the rest of the repo, so its diff
against upstream NS-3 stays minimal and reviewable.

---

## 8. RL formulation

### Observation — realistic, time-local, AP-observable (7 features / STA)

Everything the **actor** sees is obtainable through standard 802.11 mechanisms (BSR,
802.11k stats, AP-side PHY, the AP's own TWT schedule and RX) and is **time-local** — a
per-step delta, an instantaneous level, a recency, or a ratio. No cumulative counter, no
device class, no energy state ever reaches the actor; the recurrence supplies memory.

| #   | feature      | type          | meaning                                             |
| --- | ------------ | ------------- | --------------------------------------------------- |
| 0   | `bsr_be`     | level (log1p) | buffer-status report, BE queue (demand magnitude)   |
| 1   | `dpkts_rx`   | Δ (log1p)     | packets the AP received this step (served volume)   |
| 2   | `dairtime`   | Δ (log1p)     | airtime consumed this step (occupancy / cost)       |
| 3   | `dfcs`       | Δ (log1p)     | FCS errors this step (channel quality)              |
| 4   | `snr`        | level         | uplink SNR (weak; geometry)                         |
| 5   | `silence`    | recency       | time since the AP last heard this STA               |
| 6   | `starv_rate` | ratio         | starved-SPs / demanded-SPs (energy-vs-airtime tell) |

The flat observation is `num_sta · 7` (= 140 at 20 STAs). A per-feature running z-score
(`obs_rms`, warm-started from `obs_warmstart_stats.json`) is applied inside the policy and
persisted in the checkpoint.

### Action — three heads

| head       | meaning                                                           | space         |
| ---------- | ----------------------------------------------------------------- | ------------- |
| schedule   | TWT group configuration (group count, wake durations, SP offsets) | indexed table |
| assignment | STA → TWT-group mapping pattern                                   | indexed table |
| PDW        | power-window end time `pdw_end = 5 + idx·5` ms (idx 0 = no PDW)   | 10 levels     |

The schedule/assignment tables in [`exploration-scripts/`](exploration-scripts/) are a
deterministic, EDA-carved grid, not a random sample: **24 schedules** (group count K ∈
{2,3,4,5,6,8} × 4 wake-budget fractions) × **25 assignments** (9 contiguous `split_n`
REHD-tail isolators + 4 `round_robin` congestion fallbacks + 12 explicit REHD-tail cuts) —
pruned down from an earlier coarse 30 × 25 sweep to the actions the EDA showed actually
matter. The PDW is a decoder that scale-shifts the chosen schedule into the comms region
`[pdw_end, 95]`. The shipped policy uses the full **24 × 25 × 10** table (`--policy-kwargs`
auto-derives the head sizes from it).

### Reward — `twt_pf_demand_v5` (α-fair NUM, QoS-only)

A Network-Utility-Maximization objective. Per STA, utility `q_i = served·(1 − loss)`,
aggregated by **α-fair welfare** `Uα` (α = 1 → proportional-fair / Nash log-utility — *not*
a mean, so a starved station can't be averaged away). The reward may use **oracle** signals
(it is computed offline), but the *actor* never sees them. v5 adds two pieces over the
plain α-fair QoS:

- a **smooth (Bayesian-shrinkage) demand floor** — `served = (Δtx + m)/(Δeq + m)` — so a
  low-demand REHD is treated as satisfied without a hard cliff that would mask its backlog;
- a dedicated **oracle REHD-expiry penalty** `Δexpired/(Δexpired + Δtx)`, throughput-
  normalized (demand-floor-immune) and **self-zeroing for non-REHDs** (identified by the
  energy oracle), so REHD energy-starvation is priced directly without putting energy in
  the observation.

`reward_functions.py` keeps the full iterative history (`twt_efficient` →
`twt_efficient_v2`/`v3` → `twt_pf_demand_v1`…`v5`) in one registry, each version's block
comment recording exactly what changed and why — v5 is what the shipped run trains against.

---

## 9. Policy & the asymmetric critic

The shipped policy is **`lstm_ppo`**, a recurrent actor-critic. Recurrence is required: a
struggling REHD and a struggling congested STA look identical in the realistic features but
need opposite responses (more sleep to harvest vs more airtime), so the agent must
disambiguate from action→response history — a belief state a memoryless MLP cannot form.
`twt_models/` also ships `mlp_ppo`, `pointer_ppo`/`transformer_pointer` (raw per-STA-group
action heads instead of an indexed table), `random_policy`, `fixed_policy`, and the online
analytical baselines (`analytical_demand`, `analytical_md1`, `analytical_md1_k6`) behind
one `@register_policy` registry.

The **asymmetric (privileged) critic** is the key architectural piece of the latest run.
The PDW reward is *delayed* — open the window now, the REHD harvests, and the benefit shows
up several steps later as served uplink. A critic that only sees the realistic observation
cannot value that future. So the **value function** (and only the value function — this is
centralized-training / decentralized-execution) additionally consumes a small block of
privileged **oracle energy features** per STA: `[SoC, is_rehd, ΔHarvested, ΔConsumed]`. The
star feature is Δharvested — the direct, immediate, controllable consequence of opening the
PDW — letting `V()` credit "harvested now → REHD serves later." At evaluation the critic is
unused, so the actor remains fully realistic and deployable.

---

## 10. Exploratory data analysis (the EDA stage)

The design choices above (which signals feed the agent, how each is transformed, the reward
weights, the action-space pricing) are not guesses — they came from a systematic EDA pass run
*before* training, using a wider toolbox of signal-audit/controllability/reward-tuning scripts
than this snapshot ships (kept in a separate research fork, not needed to reproduce the paper's
numbers). This is what justified, e.g., dropping the redundant byte-count feature, choosing the
7-feature time-local set, and keeping energy out of the reward except as the expiry penalty.

What ships here is the part of the EDA stage the locked pipeline still runs on every
`run_pipeline.sh`/`reproduce.sh` invocation — **`twt_eda_collect.py`** drives the NS-3 wrapper
directly (no policy) through the action grid, logging every obs feature and reward input to
parquet shards, and **`twt_refit_obs_norms.py`** refits `obs_warmstart_stats.json` from those
shards (§12, stage 2).

---

## 11. Build

Run from the NS-3.44 root (the Python wrapper `chdir`s there). On a fresh checkout first
apply the patches and register the build subdir:

```bash
bash contrib/ai/examples/rl-twt-powercast/setup_fresh_ns3.sh
```

Then configure + build. **Pin an absolute `Python3_EXECUTABLE`** to the project venv — a
relative path can silently relink the pybind `.so` against the system Python and break the
whole pipeline:

```bash
cd $NS3_ROOT
source <venv>/bin/activate
./ns3 configure --enable-examples --enable-tests -- \
    -DNS3_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="<venv>/bin/python3.11"
./ns3 build
python3.11 -c "import pb_twt_powercast_interface_py"   # verify the binding imports
```

`contrib/ai/examples/CMakeLists.txt` must contain `add_subdirectory(rl-twt-powercast)` and the
`mod-files/` patches must be applied — without them `WifiMac::SetTwtSchedule` and the BSR
signal don't exist. (`setup_fresh_ns3.sh` handles both.)

---

## 12. Train

Exact command for the shipped `results/runs/<run>/pipeline/train/<ts>/ckpt_final.pt` (see `results/runs/<run>/pipeline/train/<ts>/hyperparams.json`
for every flag/seed): `lstm_ppo` + asymmetric critic, v5 reward, variable REHD split.

```bash
# NOTE: do NOT pass --policy-kwargs to size the action heads. They are auto-derived
# from the shipped action tables (twt_spawn_worker.default_obs_kwargs), so a hardcoded
# override silently mis-sizes the policy the moment the tables change -- an earlier
# version of this command pinned num_schedules:25 against a 24-entry table.
python3.11 contrib/ai/examples/rl-twt-powercast/ppo-sb3-scripts/twt_batch_orchestrator.py \
    --policy-arch lstm_ppo --asymmetric-critic \
    --reward-preset twt_pf_demand_v5 --n-stations 12 --n-rehd 8 --variable-split \
    --num-workers 16 --pool-size 16 --episodes-per-batch 24 --crn-groups 4 \
    --num-batches 200 --max-steps-per-episode 100 --warmup-steps 5 \
    --n-epochs 10 --ent-coef 0.01 --learning-rate 3e-4 --base-seed 100000 \
    --obs-warmstart --normalize-return --save-freq 10 \
    --output-dir <ABS path>/ppo-sb3-scripts/runs/<label>
```

---

## 13. Evaluate

The headline evaluation is a **structured, per-device** comparison of the trained model
against a fair **online analytical baseline** (`analytical_demand` — a demand- and
energy-aware heuristic that drives schedule/groups/assignment/PDW from the realistic obs,
with no hindsight). It runs both policies head-to-head over the REHD-split
edge-case grid with **matched scenarios** (identical NS-3 scenario per seed → exact paired
comparison) and reports per-device-class metrics — served ratio, drop/expiry, throughput,
latency, fairness, and REHD harvested/consumed/SoC/unpowered-TX — not just reward.

One command (collection + per-device report in one go) — `run_pipeline.sh`'s STAGE 4,
run standalone by disabling the earlier stages; `CKPT` selects the checkpoint (defaults to
the newest fresh checkpoint under `RUN_ROOT`, then falls back to `results/runs/<run>/pipeline/train/<ts>/ckpt_final.pt`):

```bash
RUN_PLOT=0 RUN_EDA=0 RUN_TRAIN=0 bash run_pipeline.sh                 # evaluates newest results/ ckpt_final.pt
RUN_PLOT=0 RUN_EDA=0 RUN_TRAIN=0 CKPT=path/to.pt bash run_pipeline.sh  # or a specific checkpoint
```

Under the hood (STAGE 4 of `run_pipeline.sh`; tune via env vars or by calling directly):

```bash
python3.11 ppo-sb3-scripts/twt_eval_structured.py \
    --model-ckpt results/runs/<run>/pipeline/train/<ts>/ckpt_final.pt --baseline analytical_demand \
    --splits 4 8 12 16 --seeds 10 --reward-preset twt_pf_demand_v5 \
    --max-steps 100 --num-workers 20 --out ppo-sb3-scripts/runs/eval_<ts>
# -> writes report.txt + per_class_comparison.csv (auto-run; --no-report to skip)
```

The checkpoint is self-describing (`{name, kwargs, state_dict}` + embedded
obs/return/critic normalizers), so no `--policy-arch` flag is needed.

> A brute-force, best-in-hindsight static schedule is intentionally **not** used as a
> competitor: searching all actions with oracle knowledge of outcomes is not a fair peer for
> an online learner. The comparison is model vs the principled online heuristic.

---

## 14. Results

Deterministic evaluation over 20 scenarios (variable split, 100 steps, `twt_pf_demand_v5`;
mean episode reward, higher is better):

| policy                                     | reward (mean ± std) | note                      |
| ------------------------------------------ | ------------------- | ------------------------- |
| **lstm_ppo + asymmetric critic** (shipped) | **−44.18 ± 19.3**   | the learned agent         |
| fixed `t1_s15_a6_D50`                      | −29.06 ± 11.9       | strongest static (PDW-on) |
| fixed `s5_g2_eskew_seq`                    | −44.34 ± 18.2       | strong static             |
| fixed `s15_g4_eq_rr4`                      | −68.11 ± 24.4       | static                    |
| random_policy                              | −77.09 ± 17.4       | floor                     |

The learned policy comfortably beats the random, analytical, and weaker static baselines
and **matches a strong hand-tuned static** (−44.18 vs −44.34). The asymmetric critic
improved the result over an otherwise-identical shared-critic run (−50.30 → −44.18),
confirming that exposing the privileged energy state to the value function fixes the
delayed-PDW credit-assignment problem.

The remaining gap to the single best static (−29.06) is the expected consequence of
**partial observability**: which schedule is optimal depends on the hidden REHD/traffic
layout, which is only weakly inferable from the realistic signals (the AP is class-blind by
design). At a near-static load the optimum is close to constant, so a strong static is a
high bar. The next research lever is a **constraint-rotation regime** in which the binding
constraint flips between segments, making adaptivity mandatory and observable rather than
merely available.

---

## Quick reference

### Project structure

```
rl-twt-powercast/
├── twt-powercast-main-simulation.cc   ← NS-3 main() entry point
├── twt-constants.h                    ← every simulation constant
├── pb-twt-core.h                      ← EnvStruct / ActionStruct / Realistic / Oracle metrics
├── ph-harvester-hardware.h/.cc        ← PowercastEnergyHarvester (per-REHD capacitor model)
├── ph-deployment-helper.h/.cc         ← PowercastEnergyHarvesterHelper (templated install)
├── ph-harvester-config.txt            ← hardware-class catalog (CAP_A/B/C, committed default)
├── twt-trace-callbacks.h/.cc          ← PHY/MAC trace sinks + accumulator arrays
├── twt-metrics.h/.cc                  ← BI-level and call-level metric packaging
├── twt-simulation-config.h/.cc        ← topology, device/REHD classes, ApplyTWTSchedule(), PDW
├── pb-twt-wrapper-ns3.h/.cc           ← C++ ns3-ai shared-memory IPC (TWTWrapper)
├── pb-twt-interface.cc                ← pybind11 bindings for every struct
├── pb_twt_wrapper_py.py               ← Python ns3-ai wrapper (reset/step/close)
├── CMakeLists.txt                     ← NS-3 CMake build integration
├── mod-files/                         ← NS-3 WiFi source patches (apply before build)
│   ├── bsr-manager.h/.cc
│   ├── sta-wifi-mac.h/.cc
│   ├── wifi-twt-agreement.h/.cc
│   ├── install_bsr_manager.sh
│   └── twt-complete-setup.sh
├── exploration-scripts/               ← action tables (schedule/assignment JSON + generator)
├── ppo-sb3-scripts/                   ← PPO pipeline + locked EDA stage
│   ├── twt_batch_orchestrator.py        parent: master policy, Adam, PPO update
│   ├── twt_spawn_worker.py              per-episode worker (obs/critic features, action decode)
│   ├── twt_eda_collect.py               EDA stage: signal collection for obs-norm refit
│   ├── twt_refit_obs_norms.py           refit obs-normalization stats from an EDA run
│   ├── twt_eval_structured.py           structured per-device eval: model vs analytical baseline
│   ├── twt_eval_report.py               per-device-class comparison report from the above
│   ├── twt_pool.py, twt_scenario.py     work-pool dispatch + variable-split sampler
│   ├── twt_normalizer.py                running obs/return/critic normalizers
│   ├── reward_functions.py              reward registry (canonical: twt_pf_demand_v5)
│   ├── twt_models/                      policy registry: lstm_ppo + analytical.py baselines
│   └── vcap_episode.py                  one constant-action episode w/ --logVcap (paper Vcap figure)
├── results/runs/<run>/pipeline/       ← ckpt_final.pt + hyperparams + eval shards (provenance)
├── plot_harvest_curve.py, plot_vcap.py← energy-dynamics diagnostics (run_pipeline.sh stage 1)
├── run_pipeline.sh                    ← one driver for all 4 stages (plot → EDA → train → eval)
├── reproduce.sh                       ← wraps run_pipeline.sh with paper-exact args + figures
└── setup_fresh_ns3.sh                 ← one-shot patch + build-subdir registration
```

### Key configuration points

| What to change                                          | Where                                                  |
| ------------------------------------------------------- | ------------------------------------------------------ |
| STA/REHD counts, TWT groups, episode length, PDW timing | `twt-constants.h`                                      |
| Energy model (mA per PHY state, REHD power budget)      | `twt-constants.h` — `PHY_STATE_*_MA`, `REHD_*_POWER_W` |
| REHD hardware/traffic sub-types                         | `twt-simulation-config.h` — `REHD_TYPE_TEMPLATES`      |
| Non-REHD per-class traffic profiles                     | `twt-simulation-config.h` — `*_CONFIG` structs         |
| Action-space tables (schedules/assignments)             | `exploration-scripts/generate_action_tables.py`        |
| Reward function / weights                               | `ppo-sb3-scripts/reward_functions.py`                  |
| Observation feature builder                             | `ppo-sb3-scripts/twt_spawn_worker.py`                  |
| RL hyperparameters                                      | `ppo-sb3-scripts/twt_batch_orchestrator.py` CLI flags  |
| Policy architectures                                    | `ppo-sb3-scripts/twt_models/`                          |

### Build & run

```bash
# 1. Apply NS-3 patches (once)
bash contrib/ai/examples/rl-twt-powercast/setup_fresh_ns3.sh

# 2. Build (see §11 for the full pinned-Python3_EXECUTABLE form)
./ns3 configure --enable-examples --enable-tests \
    -DNS3_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python3.11)"
./ns3 build

# 3. Full pipeline
bash contrib/ai/examples/rl-twt-powercast/run_pipeline.sh
```

### License

GNU General Public License v2-only — see the SPDX header in each source file.

**Author**: Ahmed Maksud — [ahmed.maksud@email.ucr.edu](mailto:ahmed.maksud@email.ucr.edu)
**PI**: Marcelo Menezes De Carvalho — SHINE Lab, Texas State University

---

## Notes

- `MAX_NUM_STA = 32` and `MAX_NUM_TWT_GROUPS = 16` are compile-time ceilings in
  `twt-constants.h` (the single source of truth for episode shape, RF deployment, and
  dynamics toggles). Editing it requires a rebuild.
- All oracle metrics logged by NS-3 are **cumulative**; the Python side computes per-step
  deltas explicitly.
- Train and eval horizons must match — a policy trained at 100 steps collapses if evaluated
  at a shorter horizon; always pass `--max-steps-per-episode` explicitly.
