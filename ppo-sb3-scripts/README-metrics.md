# Observation, Critic and Reward Signals

What the agent sees, what only the critic sees, what only the reward sees, and where each number comes from.
The actor has to stay deployable on a real AP, so its inputs are limited to what 802.11ax/k signalling exposes; the critic and the reward run offline in the simulator and may use oracle state.

## Table of Contents

1. [Overview](#overview)
1. [Raw Signals from NS-3](#raw-signals-from-ns-3)
1. [Actor Observation (7 features per STA)](#actor-observation-7-features-per-sta)
1. [Critic Features (4 per STA, oracle)](#critic-features-4-per-sta-oracle)
1. [Reward Inputs (`sta_deltas`)](#reward-inputs-sta_deltas)
1. [Reward `twt_pf_demand_v5`](#reward-twt_pf_demand_v5)
1. [Normalization](#normalization)
1. [Data Flow](#data-flow)
1. [Excluded Signals](#excluded-signals)

---

## Overview

| Consumer | Sees                                          | Built by                                                   | May use oracle?     |
| -------- | --------------------------------------------- | ---------------------------------------------------------- | ------------------- |
| actor    | 7 realistic, time-local features/STA          | `twt_spawn_worker._build_per_sta_features`                 | no                  |
| critic   | actor obs + 4 energy features/STA             | `twt_spawn_worker.build_critic_features`                   | yes (training only) |
| reward   | per-STA deltas of oracle + realistic counters | `twt_spawn_worker.compute_sta_deltas` → `reward_functions` | yes (offline)       |

Rules the whole pipeline is built around:

- Time-local. Every actor feature is a per-step delta, an instantaneous level, a recency, or a ratio. No lifetime counter or lifetime average reaches the policy — memory is the LSTM's job.
- Class-blind. Device class and REHD-ness are in neither the observation nor the reward's inputs except where the reward identifies REHDs through the energy oracle (`vcap_max_v > 0`) for its expiry penalty.
- Cumulative in C++, delta in Python. NS-3 never resets a counter; the worker keeps the previous `EnvStruct` and differences.

---

## Raw Signals from NS-3

`pb-twt-core.h` ships two structs per STA in every `EnvStruct` (see the root README §4 for the full field list).

**`StaRealisticMetrics`** — AP-observable: per-AC BSR (`bsr_queue_ac_{be,bk,vi,vo}`), `rx_fragment_count`, `fcs_error_count`, 802.11k `rcpi/rsni` → `rssi_dbm/snr_db/link_margin_db`, last-RX MAC-header fields (type/subtype, MCS, NSS, width, GI, PM bit, TSF), `bytes/packets_received_at_ap` (+ per AC), `airtime_used_us`, `user_priority`, the per-Service-Period block (`bytes/packets_rx_at_ap_in_sp`, `sp_completed_count`, `sp_with_demand_count`, `sp_with_starvation_count`), AP DL accounting, `observation_time_ms`, `observation_sequence_num`.

**`StaOracleMetrics`** — simulator-only: TX/retry/ACK-failure counters, legacy energy bookkeeping (`total_energy_consumed_mj`, `awake/sleep_time_ms`, `duty_cycle`), app-layer `packets/bytes_generated`, `packets_enqueued`, queue drops (`mpdu_drops_expired`, `mpdu_drops_queue_full`), `packets/bytes_transmitted`, A-MPDU stats, queue size/latency (`queue_delay_sum_ms`, `queue_delay_count`, `avg_latency_ms`), position, the PowerCast block (`vcap_v`, `vcap_max_v`, `e_nominal/sunk/avail_j`, `harvested_total_j`, `consumed_total_j`, `t_active_us`, `unpowered_tx_events`, `output_enabled`, `harvester_phase`, `est_rf_energy_delivered_j`), and the per-class config values (`device_class`, `nominal_msdu_size`, `mean_data_rate_kbps`, `delay_bound_ms`, `tx_power_dbm`).

`pb_twt_wrapper_py.py` turns each into a plain dict: `env["sta_observations"][i]["realistic"]` / `["oracle"]`.

---

## Actor Observation (7 features per STA)

Flat `num_sta × 7` (140 at 20 STAs), no padding, order fixed by `obs_warmstart_stats.json`.
`Δ` is the difference between consecutive `EnvStruct`s (one agent step = 25 BI = 2.56 s), clamped at 0.

| #   | Feature      | Raw source (realistic)                                | Transform                                         | Reads as                       |
| --- | ------------ | ----------------------------------------------------- | ------------------------------------------------- | ------------------------------ |
| 0   | `bsr_be`     | `bsr_queue_ac_be` (0–254)                             | `log1p(x) / log1p(254)`                           | demand level, heavy-tailed     |
| 1   | `dpkts_rx`   | Δ`packets_received_at_ap`                             | `log1p(Δ) / log1p(1632)`                          | served volume this step        |
| 2   | `dairtime`   | Δ`airtime_used_us`                                    | `log1p(Δ) / log1p(371329)`                        | occupancy / cost this step     |
| 3   | `dfcs`       | Δ`fcs_error_count`                                    | `log1p(Δ) / log1p(49)`                            | channel quality                |
| 4   | `snr`        | `snr_db` (sentinel → 0)                               | `clip((x − 48) / 17.35, 0, 1)`                    | geometry, weak                 |
| 5   | `silence`    | `last_rx_timestamp_us`, sim time                      | `clip((now − last_rx) / 2.56 s, 0, 1)`, 1 = never | how long since the AP heard it |
| 6   | `starv_rate` | Δ`sp_with_starvation_count` / Δ`sp_with_demand_count` | `clip(ratio, 0, 1)`, 0 if no demanded SPs         | got its SP, could not drain it |

The `log1p` caps are the EDA p99 values; everything lands in ~[0, 1] before the policy's running z-score.

Why these seven: the signal audit over the controllability EDA found every cumulative-level feature of the earlier 18-feature set (lifetime airtime / awake / sleep / duty / packets-TX averages) leaked history, the per-AC BSR was uniform AC_BE (only `bsr_be` moves), and `dbytes_rx` duplicated `dairtime` (|r| = 0.965).
`starv_rate` was added because an energy-starved REHD and an airtime-starved STA look the same in demand and service; only the REHD keeps getting service periods it cannot fill.

---

## Critic Features (4 per STA, oracle)

With `--asymmetric-critic`, the value function also receives `num_sta × 4` from `build_critic_features()`:

| #   | Feature      | Source (oracle)                          | Scale |
| --- | ------------ | ---------------------------------------- | ----- |
| 0   | `soc`        | `vcap_v / vcap_max_v`, clipped to [0, 1] | —     |
| 1   | `is_rehd`    | `vcap_max_v > 0`                         | 0/1   |
| 2   | `dHarvested` | Δ`harvested_total_j`                     | × 1e4 |
| 3   | `dConsumed`  | Δ`consumed_total_j`                      | × 1e3 |

Non-REHD STAs read `[0, 0, 0, 0]`.
`lstm_ppo` concatenates the block onto the value-LSTM input only (`vf_in_dim = obs_dim + critic_extra_dim`); the actor path is unchanged and the block is dropped at evaluation.
Δharvested responds immediately to the PDW head, so the critic can credit a window opened several steps before the served uplink shows up.

---

## Reward Inputs (`sta_deltas`)

`compute_sta_deltas(prev_env, curr_env)` in the worker returns one dict per STA; the reward functions index these keys:

| Key                                                                                                                                                                                                    | Source                            | Kind    | Used by v5                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- | ------- | ---------------------------- |
| `delta_packets_transmitted`                                                                                                                                                                            | oracle `packets_transmitted`      | Δ       | yes (served numerator)       |
| `delta_packets_enqueued`                                                                                                                                                                               | oracle `packets_enqueued`         | Δ       | yes (demand)                 |
| `delta_drops_expired`                                                                                                                                                                                  | oracle `mpdu_drops_expired`       | Δ       | yes (loss, REHD expiry)      |
| `vcap_max_v`                                                                                                                                                                                           | oracle                            | level   | yes (REHD identification)    |
| `vcap_v`                                                                                                                                                                                               | oracle                            | level   | v4 energy term (off)         |
| `delta_harvested_j`, `delta_consumed_j`                                                                                                                                                                | oracle                            | Δ       | v4 energy term (off)         |
| `delta_airtime_used_us`                                                                                                                                                                                | realistic                         | Δ       | `w_air` occupancy cost (off) |
| `step_latency_ms`, `latency_count`                                                                                                                                                                     | oracle `queue_delay_sum_ms/count` | Δ ratio | v3 latency term              |
| `delta_bytes_transmitted`, `delta_energy_mj`, `delta_awake/sleep_time_ms`, `duty_cycle`, `queue_size_bytes/packets`, `bsr_queue_index`, `fcs_error_count`, `rx_fragment_count`, `last_rx_timestamp_us` | mixed                             | —       | earlier presets              |

The same dict is what `twt_eda_collect.py` logs per row (`d_*` columns), so any preset can be re-scored offline from the shards.

---

## Reward `twt_pf_demand_v5`

Per step, over all `n` STAs, with defaults `alpha = 1`, `shrink_m = 3`, `shrink_p0 = 1`, `w_rehd_exp = 0.5`, `q_floor = 0.02`, `r_clip = 5`, `w_eng = 0`, `w_air = 0`:

```
served_i = clip((Δtx_i + m·p0) / (Δeq_i + m), 0, 1)        # Laplace/Beta shrinkage toward "satisfied"
loss_i   = clip(Δexpired_i / (Δeq_i + m), 0, 1)             # loss prior = 0
q_i      = clip(served_i · (1 − loss_i), q_floor, 1)        # per-STA QoS utility
U_α(q)   = log q                       (α = 1; PF / Nash)   # (q^(1−α) − 1)/(1−α) otherwise
W_qos    = mean_i U_α(q_i)                                  # ≤ 0, 0 = everyone satisfied

REHD set R = { i : vcap_max_i > 0 and Δexpired_i + Δtx_i > 0 }
e_i      = Δexpired_i / (Δexpired_i + Δtx_i)   for i in R     # fraction of the REHD's queue exits that expired
W_exp    = mean_{i∈R} e_i          (0 if R is empty)

total    = clip(W_qos − w_rehd_exp · W_exp, −r_clip, r_clip)
```

Reading it:

- `W_qos` is an α-fair welfare rather than a mean of ratios: with α = 1 one starved STA (`q → 0.02`) pulls the log-sum down by `log 0.02 / n ≈ −3.9/n`, and over-service elsewhere cannot offset it. This is why the REHDs count.
- The shrinkage replaces v4's hard `demand_floor = 3` cliff (44.6 % of low-demand REHD steps were scored "satisfied" with an expiring backlog). With `Δeq → 0` the ratio tends to the prior `p0 = 1`; with `Δeq ≫ m` it is the raw ratio.
- `W_exp` prices REHD starvation directly and is normalized by queue *throughput* (packets that left the queue), so it is defined at any demand level and immune to the floor. It self-zeros for non-REHDs, which keeps the reward oracle-side only.
- The v4 energy term (`w_eng`, H/C sustainability) is kept in the code but off: the 2026-06-08 audit showed it rewarded opening the PDW for its own sake (PDW farming) at a large throughput cost, while REHD energy success already shows up as served REHD packets.

The function returns `components` (`w_qos`, `w_rehd_exp`, `raw_mean_served`, `raw_mean_loss`, `raw_mean_q`, `raw_min_q`, `raw_rehd_expiry_frac`, `n_rehd_active`) alongside `total`; `twt_eda_collect.py` and the evaluators log them.

Per-episode returns are the sum over the ~95 recorded steps (the 100-update episode minus the 5 warm-up steps), which is why the training curve sits between −140 and −60 while the eval report quotes the per-step mean (−0.68 for the shipped model, −1.00 for the baseline).

---

## Normalization

`twt_normalizer.py` ports the SB3 `VecNormalize` mechanism (Welford/Chan running mean + var, z-score, clip, persisted stats) with two changes for this problem:

- Observation stats are per feature, shared across STAs: shape `(7,)`, not `(140,)`. The observation is a set of STA tokens, so per-flat-dimension statistics would break permutation symmetry and waste 20× the samples. The stats live inside the policy (`BasePolicy.obs_rms`), are warm-started from `obs_warmstart_stats.json` (`--obs-warmstart`) and saved in the checkpoint.
- Returns, not rewards, are normalized: a scalar `RunningMeanStd` over discounted returns per stream; rewards are scaled by `1/√var` (not centred) so value targets are unit-scale (`--normalize-return`).
- The critic block gets its own `(4,)` running stats (`init_critic_norm`).

`obs_warmstart_stats.json` is the per-feature mean/std of the post-transform `obs_*` columns over an EDA run; `run_pipeline.sh` stage 2 refits it (`twt_refit_obs_norms.py`) so the warm start always matches the scenario the collector saw.

---

## Data Flow

```
NS-3 trace callbacks → per-STA cumulative arrays
        │  every 25 BI: PopulateStaObservationRaw() → EnvStruct (realistic + oracle per STA)
        ▼
pb_twt_wrapper_py.TWTWrapper.step() → env_dict
        │
        ├─ _build_per_sta_features(prev, curr) ──► obs (n×7) ──► policy.obs_rms ──► actor & critic
        ├─ build_critic_features(prev, curr)  ──► critic block (n×4) ──► critic only (training)
        └─ compute_sta_deltas(prev, curr)     ──► sta_deltas ──► reward_fn(preset) ──► r_t
                                                       │
                                                       └──► twt_eda_collect.py parquet rows (obs_*, d_*, real_*, orc_*, reward_total)
```

---

## Excluded Signals

From the actor:

| Signal                                                                                       | Why not                                                                                                                            |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| any cumulative counter or lifetime average                                                   | time-locality; the earlier 18-feature set leaked the whole history                                                                 |
| `bsr_queue_ac_{bk,vi,vo}`, `user_priority`, `power_mgmt_bit`, `mpdu_drops_queue_full`        | dead on this scenario (all traffic AC_BE, queues never overflow), per the EDA signal audit                                         |
| `dbytes_rx`                                                                                  |                                                                                                                                    |
| `device_class`, `mean_data_rate_kbps`, `delay_bound_ms`, `nominal_msdu_size`, `tx_power_dbm` | would be obtainable via ADDTS / TPC signalling in a real deployment, but this sim reads them from config — kept oracle for honesty |
| every energy field (`vcap_*`, `harvested/consumed_total_j`, `harvester_phase`, …)            | the AP must infer harvester state from protocol signals; that is the problem being solved                                          |

From the reward: `awake/sleep_time_ms` and `duty_cycle` (the schedule already fixes them), `est_rf_energy_delivered_j` (dead — the PDW credits energy directly, not through this estimate), and the index-based airtime cost of v1 (double-counted the served term and collapsed the schedule to the minimum).
