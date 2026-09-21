# Analytical Baselines for TWT-PowerCast Scheduling

`twt_models/analytical.py` holds the online, hindsight-free heuristics the learned policy is compared against.
They read the same 7-feature realistic observation as the actor, re-decide every step, and drive all three action heads (schedule, assignment, PDW), so the paper compares the model against a controller with the same information and the same action space rather than against a best-in-hindsight static schedule picked with oracle knowledge.

## Table of Contents

1. [Registered Names](#registered-names)
1. [Why These Baselines](#why-these-baselines)
1. [Shared Machinery](#shared-machinery)
1. [`analytical_md1` — the paper's baseline](#analytical_md1--the-papers-baseline)
1. [`analytical_md1_k6` — ablation](#analytical_md1_k6--ablation)
1. [`analytical_demand`](#analytical_demand)
1. [Running Them](#running-them)
1. [Calibration Notes](#calibration-notes)

---

## Registered Names

| Name                | Class                           | Role                                                                                                                        |
| ------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `analytical_md1`    | `AnalyticalMD1Policy`           | The paper's baseline (Sec. V-A, Eq. 9): M/D/1 queue feedback + energy-aware PDW                                             |
| `analytical_md1_k6` | `AnalyticalMD1RestrictedPolicy` | Same, group count clamped to K ≤ 6 — kept only to quantify what that restriction cost                                       |
| `analytical_demand` | `AnalyticalDemandPolicy`        | Earlier demand/starvation heuristic; the default `--baseline` of `twt_eval_structured.py` but not the one the pipeline uses |

Each is wrapped by an `_AnalyticalAdapter(BasePolicy)` so the orchestrator, evaluator and workers treat it like any other registry entry: `act()` returns the three indices, `evaluate()` returns zeros (no value function, no gradients).

---

## Why These Baselines

The baseline has to match the learned policy on three things:

- Same information. The heuristics see only the realistic observation — `bsr_be`, `silence`, `starv_rate`, … — never device class, capacitor state or the oracle counters.
- Same action space. They pick from the same 24 schedules, 25 assignments and 10 PDW levels, loaded from `../exploration-scripts/*.json` at import time, so a table change moves both sides together.
- Online. They adapt every step from the live observation; nothing is fitted per scenario after the fact.

A brute-force best static schedule per seed would score higher but is not used: it needs the outcome to choose the action.

---

## Shared Machinery

- `WiFi6Constants` — PHY/MAC numbers (102.4 ms BI, MCS/rate, power model, IFS/slot, CW, A-MPDU) used by the legacy queueing/energy/throughput helpers.
- `TWT_Schedules` — `SCHEDULE_TABLE` loaded from `schedule_table.json`, with `get_total_duration(idx)`, `get_num_groups(idx)`, `get_duty_cycle(idx)`, `get_schedules_by_duration()`.
- `_load_assignment_meta()` — index of the round-robin pattern per group count (and `all_to_one` if present) from `assignment_table.json`.
- Feature columns: `F_BSR=0, F_DPKTS=1, F_DAIR=2, F_DFCS=3, F_SNR=4, F_SILENCE=5, F_STARV=6`; the flat observation is reshaped to `(num_sta, 7)` and `num_sta` is derived from its length, so the baselines follow the variable split automatically.

---

## `analytical_md1` — the paper's baseline

`AnalyticalMD1Policy` (the "2026-06-22 overhaul" block).
Realistic-only, adaptive, recalibrated from the carved-grid EDA (`eda_full_20260621`).

Per step, from the `(num_sta, 7)` observation it computes `mean_q` and `cv` of `bsr_be` across STAs, and the means of `silence` and `starv_rate`, then:

| Head       | Rule                                                                                                                                                                                                                    |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| schedule   | M/D/1 feedback. Backlog → wake budget: `frac = clip((mean_q − Q_LO)/(Q_HI − Q_LO), 0, 1)` picks the budget rank within the chosen K; more backlog → more service time so ρ = λ/μ drops below 1                          |
| groups K   | Fairness split. Backlog dispersion → group count: `cv ≤ 0.85 → K=2`, `0.85–0.95 → 3`, `0.95–1.05 → 4`, `1.05–1.15 → 5`, `1.15–1.25 → 6`, `≥ 1.25 → 8` (the full K ladder of the table, 0.10-cv bands)                   |
| assignment | `cv ≥ 1.0` → the `split_n` pattern nearest K (isolate the congested/tail block); else the `round_robin` pattern nearest K                                                                                               |
| PDW        | Energy-aware. `pdw = (mean_sil − SIL_LO)/(SIL_HI − SIL_LO) · 9`, `+1` level if `mean_starv > STV_BOOST`; silence rises with the REHD count (0.39 at few → 0.63 at many), so it is the realistic proxy for energy stress |

Calibrated constants (class attributes):

```python
LAMBDA, MU  = 9.16, 6.85    # pkts / BI / STA from the EDA (rho ~ 1.34: the cell is over-loaded on average)
Q_LO, Q_HI  = 0.125, 0.50   # bsr_be backlog -> budget-fraction rank
CV_LO, CV_HI = 0.80, 1.20   # cv(bsr_be) across STAs -> group count
SIL_LO, SIL_HI = 0.39, 0.63 # mean silence -> PDW level (lo..hi REHD energy stress)
STV_BOOST   = 0.55          # mean starv_rate above this -> +1 PDW level
```

`_select_K` is a rank map over the full K ladder with a fixed 0.10-cv band width.
An earlier version clipped the ladder at K ≤ 6 ("the EDA's hot band"); over 160 matched episodes that restriction cost the baseline 0.106 in α-fair reward (−1.1095 → −1.0038) and 0.017 in Jain fairness — about 24 % of the reported reward margin — so it was removed and lives on only as the ablation below.

---

## `analytical_md1_k6` — ablation

`AnalyticalMD1RestrictedPolicy` overrides `_select_K` to clamp to `K ≤ 6` (`kf` clipped to `[0, 1]`, spread over the hot Ks).
Not the paper's baseline; run it to reproduce the number above.

---

## `analytical_demand`

`AnalyticalDemandPolicy` — the first heuristic written against the 7-feature observation, before the M/D/1 recalibration.

1. Load → airtime: `load = 0.75·mean(bsr_be) + 0.25·mean(silence)`, `target_duty = 0.15 + 0.75·load`; pick the schedule whose `total/102.4` is closest, with a `0.05·|K − K_target|` bias.
1. Heterogeneity → groups: `cv(bsr_be) < 0.25 → 1`, `< 0.5 → 2`, `< 0.9 → 3`, else `4`.
1. Assignment: round-robin matched to the schedule's group count (all-to-one if a single group).
1. PDW: `round(mean(starv_rate) · 9)`.

It stays registered (and is the `--baseline` default of `twt_eval_structured.py`) because earlier eval runs used it; `run_pipeline.sh` passes `EVAL_BASELINE=analytical_md1`.

---

## Running Them

Any registry name works wherever a policy is accepted:

```bash
# Score a baseline alone
python3.11 ppo-sb3-scripts/twt_evaluator.py --policy-arch analytical_md1 \
    --reward-preset twt_pf_demand_v5 --n-episodes 20 --num-workers 10 --max-steps-per-episode 100

# Head-to-head with a checkpoint and the random floor
python3.11 ppo-sb3-scripts/twt_evaluator.py --compare \
    results/runs/<run>/pipeline/train/<run_name>/ckpt_final.pt analytical_md1 random_policy \
    --reward-preset twt_pf_demand_v5 --n-episodes 20 --num-workers 10 --max-steps-per-episode 100

# The paper's structured comparison (what run_pipeline.sh stage 4 runs)
python3.11 ppo-sb3-scripts/twt_eval_structured.py --model-ckpt <ckpt> --baseline analytical_md1 \
    --splits 4 8 12 16 --seeds 20 --reward-preset twt_pf_demand_v5 --max-steps 100 --num-workers 20 --out <dir>
```

---

## Calibration Notes

- `LAMBDA`/`MU` are per-BI per-STA arrival and service rates read off the carved-grid EDA at the max-airtime action; they set the *scale* of the backlog→budget map, not a closed-form schedule.
- `Q_LO/Q_HI`, `CV_LO/CV_HI` and `SIL_LO/SIL_HI` are the low/high bounds of the corresponding observation statistics over the EDA shards, so each rank map spans the range actually seen.
- Changing the action tables changes what the K ladder and budget ranks resolve to; the constants above do not need touching unless the scenario (traffic scale, REHD templates, annulus) changes.
- The legacy `analytical_queue`/`analytical_throughput` family from the ICCCN'26 code is not registered here: it parsed the old 13/18-feature observation and predates the PDW head.
