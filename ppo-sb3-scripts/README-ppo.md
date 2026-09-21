# PPO Scripts for TWT-PowerCast Scheduling

This directory holds the RL pipeline: a custom multi-process PPO (no Stable-Baselines3 at run time — the SB3 mechanisms we need, `RunningMeanStd`/`VecNormalize`-style normalization and `RecurrentPPO`'s sequence minibatching, are ported into plain PyTorch), the per-episode worker that talks to NS-3 over `ns3-ai` shared memory, the policy registry, the reward registry, the locked EDA stage, and the evaluation tools.

## Table of Contents

1. [Files](#files)
1. [Quick Start](#quick-start)
1. [Pipeline Architecture](#pipeline-architecture)
1. [Action Space](#action-space)
1. [Observation Space](#observation-space)
1. [Models](#models)
1. [Reward Function](#reward-function)
1. [Training](#training)
1. [Evaluation](#evaluation)
1. [EDA Stage](#eda-stage)
1. [Figures](#figures)
1. [Output Structure](#output-structure)

---

## Files

| File                        | Description                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `twt_batch_orchestrator.py` | Training driver: master policy + Adam, rolling worker pool, GAE, clipped-surrogate PPO, checkpoints                                      |
| `twt_spawn_worker.py`       | Runs ONE episode in a spawned process: obs/critic features, action decode, per-STA deltas, reward                                        |
| `twt_pool.py`               | Rolling work-pool (dispatch-on-completion) used by the orchestrator, evaluator and EDA collector                                         |
| `twt_scenario.py`           | Variable REHD/non-REHD split sampler (`n_rehd ~ U{4..16}`, total fixed at 20, CRN-safe)                                                  |
| `twt_normalizer.py`         | `RunningMeanStd`, `ReturnNormalizer`, warm-start loading for obs / return / critic normalization                                         |
| `reward_functions.py`       | Reward registry (`REWARD_PRESETS`); the shipped preset is `twt_pf_demand_v5`                                                             |
| `obs_warmstart_stats.json`  | Per-feature mean/std the policy's obs z-score starts from (refit by the EDA stage)                                                       |
| `twt_models/`               | Policy registry — `lstm_ppo` (shipped), `mlp_ppo`, `pointer_ppo`, `transformer_pointer`, `fixed_policy`, `random_policy`, `analytical_*` |
| `twt_eda_collect.py`        | EDA stage 2: drives NS-3 through the action grid, writes parquet shards                                                                  |
| `twt_refit_obs_norms.py`    | Refits `obs_warmstart_stats.json` from an EDA run                                                                                        |
| `twt_eval_structured.py`    | Headline evaluation: model vs analytical baseline over the REHD-split grid, matched scenarios                                            |
| `twt_eval_report.py`        | Per-device-class report (`report.txt`, `per_class_comparison.csv`) from a structured-eval run                                            |
| `twt_evaluator.py`          | General evaluator: one checkpoint, or `--compare` several policies, reward-only                                                          |
| `vcap_episode.py`           | One constant-action episode with `--logVcap` for the capacitor-voltage figure (`reproduce.sh` stage 3)                                   |
| `make_figs.py`              | The paper's four figures + Table 2 numbers from the latest run (`reproduce.sh` stage 4)                                                  |

The smoke/stress tests live in [`../test-scripts/`](../test-scripts/) and import from here; the action tables live in [`../exploration-scripts/`](../exploration-scripts/).

---

## Quick Start

All commands run from the NS-3 root with the venv active (the worker `chdir`s there to find the `ns3` launcher).

```bash
source ~/NS3-project/EHRL/bin/activate
cd ~/NS3-project/ns-allinone-3.44/ns-3.44
P=contrib/ai/examples/rl-twt-powercast

# Sanity: one worker, one short episode
python3.11 $P/test-scripts/twt_smoke_one_worker.py

# Train (paper configuration)
python3.11 $P/ppo-sb3-scripts/twt_batch_orchestrator.py \
    --policy-arch lstm_ppo --asymmetric-critic --reward-preset twt_pf_demand_v5 \
    --n-stations 12 --n-rehd 8 --variable-split \
    --num-workers 16 --pool-size 16 --episodes-per-batch 24 --crn-groups 4 \
    --num-batches 200 --max-steps-per-episode 100 --warmup-steps 5 \
    --n-epochs 10 --ent-coef 0.01 --learning-rate 3e-4 --base-seed 100000 \
    --obs-warmstart --normalize-return --save-freq 10 \
    --output-dir $P/results/runs/my_run/pipeline/train

# Evaluate vs the M/D/1 baseline (80 matched scenarios) + per-device report
python3.11 $P/ppo-sb3-scripts/twt_eval_structured.py \
    --model-ckpt $P/results/runs/my_run/pipeline/train/<run_name>/ckpt_final.pt \
    --baseline analytical_md1 --splits 4 8 12 16 --seeds 20 \
    --reward-preset twt_pf_demand_v5 --max-steps 100 --warmup-steps 5 \
    --num-workers 20 --out $P/results/runs/my_run/pipeline/eval

# Quick reward-only comparison of any registered policies
python3.11 $P/ppo-sb3-scripts/twt_evaluator.py --compare \
    $P/results/runs/my_run/pipeline/train/<run_name>/ckpt_final.pt analytical_md1 random_policy \
    --reward-preset twt_pf_demand_v5 --n-episodes 10 --num-workers 10 --max-steps-per-episode 100
```

`run_pipeline.sh` at the repo root wires these together with the right defaults; the commands above are what it runs.

---

## Pipeline Architecture

```
┌──────────────────────────────── twt_batch_orchestrator.py (parent) ───────────────────────────────────┐
│  policy = get_policy(arch)(**default_obs_kwargs)      ← twt_models registry                           │
│  optimizer = Adam                                                                                     │
│  per batch:                                                                                           │
│    payload = pickle({name, kwargs, state_dict})       ← frozen for the whole batch (on-policy)        │
│    items   = K work-items: (shm seed, CRN rand_seed, n_rehd from twt_scenario.sample_split)           │
│    run_rolling_pool(items, run_episode, pool_size=M)  ← twt_pool: M slots, dispatch-on-completion     │
│          │  each item → mp.Process(spawn)                                                             │
│          ▼                                                                                            │
│    ┌──── twt_spawn_worker.run_episode ───────────────────────────────────────────────────────────┐    │
│    │  policy = unpickle(payload)                                                                 │    │
│    │  wrapper = TWTWrapper(); env = wrapper.reset(seed, rand_seed)     ← NS-3 boots              │    │
│    │  for t in range(warmup + max_steps):                                                        │    │
│    │      obs    = build_ppo_observation(prev_env, env)      (num_sta × 7, realistic only)       │    │
│    │      critic = build_critic_features(prev_env, env)      (num_sta × 4, oracle energy)        │    │
│    │      a, logp, v, h = policy.act(obs, critic, h)                                             │    │
│    │      action_dict = build_action_dict(a, tables, num_sta)  + apply_pdw_scale_shift           │    │
│    │      env, done = wrapper.step(action_dict)                                                  │    │
│    │      r = reward_fn(compute_sta_deltas(prev_env, env))                                       │    │
│    │  result_queue.put(rollout); wrapper.close(); os._exit(0)                                    │    │
│    └─────────────────────────────────────────────────────────────────────────────────────────────┘    │
│    drain queue → join → compute_gae → PPO update (flat minibatches, or per-episode sequences for      │
│    recurrent policies) → TensorBoard + training_log.jsonl → ckpt every --save-freq                    │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Things to know before modifying it:

- One process per episode. Boost.Interprocess static destructors hang on mapped shared memory at normal exit, so every worker ends with `os._exit(0)` after its `result_queue.put()` has flushed. `twt_pool.py`'s docstring has the flush rule.
- Rolling pool, not batch barrier. `--pool-size M` slots stay busy and the next work-item goes to whichever frees first. On the DGX's big.LITTLE cores a synchronous batch would wait on the slowest A725 core every time.
- CRN groups. `--crn-groups G` splits each batch into G groups that share an NS-3 `randSeed` (same scenario, unique shared-memory segment per worker). Workers in a group differ only in the actions they sampled, which gives PPO a clean per-scenario advantage contrast.
- Variable split. `--variable-split` draws `n_rehd ~ U{4..16}` per scenario (`twt_scenario.py`, splitmix64 of the scenario seed so a CRN group shares its split). Total stays 20 because the observation is `num_sta × 7` with no padding.
- Recurrent update. `lstm_ppo` sets `is_temporal_recurrent_policy`, which routes the update through `_ppo_update_recurrent`: episodes stay intact, `evaluate_sequence()` re-rolls the LSTMs from `h0 = 0` over each T-step trajectory, `--seq-minibatch-size` episodes per minibatch.
- Normalization lives in the policy. `BasePolicy.obs_rms` z-scores the observation inside `act()`/`evaluate()` and is saved in the `state_dict`, so a checkpoint carries its own statistics. `--obs-warmstart` initializes it from `obs_warmstart_stats.json`; `--normalize-return` adds a scalar `ReturnNormalizer` on discounted returns (rewards are scaled, not centered).

---

## Action Space

`MultiDiscrete([24, 25, 10])` — schedule × assignment × PDW level = 6,000 combinations.

| Head       | Index → meaning                                                                             | Source                                         |
| ---------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| schedule   | one of 24 `(K groups, wake budget)` rows, equal-duration SPs                                | `../exploration-scripts/schedule_table.json`   |
| assignment | one of 25 STA→group patterns (`round_robin`, `split_n`, weighted REHD-tail cuts)            | `../exploration-scripts/assignment_table.json` |
| PDW        | `pdw_end = 5 + idx·5` ms, idx 0 = no window; schedule is scale-shifted into `[pdw_end, 95]` | `twt_spawn_worker.apply_pdw_scale_shift`       |

Head sizes are read from the tables at import time (`default_obs_kwargs`), so `--policy-kwargs` must never pin `num_schedules`/`num_assignments`/`num_pdw_levels` by hand.

---

## Observation Space

`Box(num_sta × 7,)` — 140 at 20 STAs, built by `_build_per_sta_features()`.
Every feature is realistic (AP-observable) and time-local; the order must match `obs_warmstart_stats.json`:

| #   | Feature      | Transform                                                  | Type    |
| --- | ------------ | ---------------------------------------------------------- | ------- |
| 0   | `bsr_be`     | `log1p(bsr_queue_ac_be) / log1p(254)`                      | level   |
| 1   | `dpkts_rx`   | `log1p(Δpackets_received_at_ap) / log1p(1632)`             | Δ       |
| 2   | `dairtime`   | `log1p(Δairtime_used_us) / log1p(371329)`                  | Δ       |
| 3   | `dfcs`       | `log1p(Δfcs_error_count) / log1p(49)`                      | Δ       |
| 4   | `snr`        | `clip((snr_db − 48) / (65.35 − 48), 0, 1)`                 | level   |
| 5   | `silence`    | `clip((now − last_rx_ts) / 2.56 s, 0, 1)`, 1 = never heard | recency |
| 6   | `starv_rate` | `clip(Δsp_with_starvation / Δsp_with_demand, 0, 1)`        | ratio   |

The asymmetric critic additionally receives `num_sta × 4` oracle energy features from `build_critic_features()`: `[SoC = vcap/vmax, is_rehd, Δharvested_J × 1e4, Δconsumed_J × 1e3]`.
They never reach the actor.
[`README-metrics.md`](README-metrics.md) has the full signal inventory.

---

## Models

All policies subclass `twt_models/base.py:BasePolicy` and register with `@register_policy`; the orchestrator, evaluator and workers only ever see the registry name.
Adding an architecture is one new file in `twt_models/`.

| Name                                                       | File                     | Notes                                                                                                                                                                |
| ---------------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lstm_ppo`                                                 | `lstm_ppo.py`            | Shipped. Separate `pi`/`vf` LSTMs (256 hidden, 1 layer) → 2×256 Tanh MLP trunks → three Categorical heads + value. Port of SB3-Contrib `RecurrentPPO(MlpLstmPolicy)` |
| `mlp_ppo`                                                  | `mlp_ppo.py`             | Memoryless MLP actor-critic; used by the smoke test and as the ablation                                                                                              |
| `pointer_ppo`                                              | `pointer_ppo.py`         | Per-STA pointer heads over raw group assignment instead of the indexed table                                                                                         |
| `transformer_pointer`                                      | `transformer_pointer.py` | Transformer encoder over STAs + pointer heads                                                                                                                        |
| `fixed_policy`                                             | `fixed_policy.py`        | Returns one constant `(sched, assign, pdw)` — a static schedule scored like any other policy                                                                         |
| `random_policy`                                            | `random_policy.py`       | Uniform over the three heads — the floor                                                                                                                             |
| `analytical_demand`, `analytical_md1`, `analytical_md1_k6` | `analytical.py`          | Online heuristics, see [`README-analytical.md`](README-analytical.md)                                                                                                |

With `--asymmetric-critic` the orchestrator passes `critic_features_per_sta=4` / `critic_extra_dim=num_sta×4` to the policy; `lstm_ppo` concatenates the critic block onto the value LSTM's input only.

---

## Reward Function

`reward_functions.py` is a registry (`REWARD_PRESETS`, `get_reward_function(name)`) that keeps every version we trained against, each with a block comment on what changed and why:

| Preset                                                                 | Idea                                                                                                       |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `throughput` / `energy` / `queue` (+ aliases `balanced`, `latency`, …) | The original 6-component weighted reward from the ICCCN'26 work (z-scored components)                      |
| `throughput_pf`, `balanced_pf`, `energy_pf`, `latency_pf`              | Scale-free proportional-fair family                                                                        |
| `twt_efficient`, `_v2`, `_v3`                                          | Efficiency-per-airtime rewards from the schedule-index era                                                 |
| `twt_pf_demand_v1`–`v3`                                                | Demand-aware served/expiry rewards, mean-aggregated                                                        |
| `twt_pf_demand_v4`, `v4q`                                              | α-fair (PF/Nash) aggregation of per-STA utility `q = served·(1 − loss)`; `v4q` zeroes the REHD-energy term |
| `twt_pf_demand_v5`                                                     | v4q + Bayesian-shrinkage demand floor + oracle REHD-expiry penalty. The shipped preset.                    |

v5 per step, with `m = 3` (pseudo-count), `α = 1`, `w_rehd_exp = 0.5`, `q_floor = 0.02`, clipped to ±5:

```
served_i = (Δtx_i + m) / (Δeq_i + m)          loss_i = Δexpired_i / (Δeq_i + m)
q_i      = clip(served_i · (1 − loss_i), 0.02, 1)
W_qos    = mean_i log(q_i)                                        # α-fair welfare, ≤ 0
e_i      = Δexpired_i / (Δexpired_i + Δtx_i)   for REHDs (vcap_max > 0) with traffic
total    = W_qos − 0.5 · mean_i e_i
```

Inputs come from `compute_sta_deltas()` in the worker: per-STA `Δpackets_transmitted`, `Δpackets_enqueued`, `Δdrops_expired`, `Δharvested_j`, `Δconsumed_j`, `vcap_max_v`, … computed by differencing consecutive `EnvStruct`s.

---

## Training

### Flags that define the shipped run

| Flag                                                            | Value                   | Notes                                                             |
| --------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------- |
| `--policy-arch`                                                 | `lstm_ppo`              |                                                                   |
| `--asymmetric-critic`                                           | on                      | privileged energy features to the value function only             |
| `--reward-preset`                                               | `twt_pf_demand_v5`      |                                                                   |
| `--n-stations` / `--n-rehd`                                     | 12 / 8                  | canonical split; `--variable-split` overrides per scenario        |
| `--variable-split`                                              | on                      | `n_rehd ~ U{4..16}`                                               |
| `--num-workers` / `--pool-size`                                 | 16 / 16                 | concurrent NS-3 instances                                         |
| `--episodes-per-batch`                                          | 24                      | episodes per PPO batch (rolling pool drains the tail)             |
| `--crn-groups`                                                  | 4                       | 4 shared-scenario groups per batch                                |
| `--num-batches`                                                 | 200                     | 4,800 episodes total                                              |
| `--max-steps-per-episode`                                       | 100                     | = `DURATION_IN_UPDATE`; must equal the eval horizon               |
| `--warmup-steps`                                                | 5                       | steps before the rollout starts recording                         |
| `--n-epochs` / `--minibatch-size` / `--seq-minibatch-size`      | 10 / 64 / 4             | PPO epochs; flat vs sequence minibatches                          |
| `--learning-rate` / `--ent-coef` / `--vf-coef`                  | 3e-4 / 0.01 / 0.5       |                                                                   |
| `--gamma` / `--gae-lambda` / `--clip-range` / `--max-grad-norm` | 0.99 / 0.95 / 0.2 / 0.5 | defaults                                                          |
| `--normalize-advantage` / `--normalize-return`                  | on / on                 |                                                                   |
| `--obs-warmstart`                                               | on                      | initialize `obs_rms` from `obs_warmstart_stats.json`              |
| `--base-seed`                                                   | 100000                  | scenario seed base                                                |
| `--save-freq`                                                   | 10                      | checkpoint every 10 batches                                       |
| `--worker-timeout`                                              | 1200 s                  | a slot that has not returned by then is written off for the batch |

`--resume <ckpt>` / `--start-batch N` continue a run; `--verbose` keeps NS-3 stdout and the trace CSVs (`--keep-ns3-stdout`, `--keep-ns3-traces` individually).

### Checkpoint format

```python
torch.save({
    "name": "lstm_ppo",              # registry name
    "kwargs": {...},                 # obs_dim, head sizes, critic dims — enough to rebuild the module
    "state_dict": policy.state_dict(),   # weights + obs_rms buffers
    "optimizer": optimizer.state_dict(),
    "ret_norm": ret_norm.state_dict(),   # or None
}, "ckpt_final.pt")
```

Every consumer (`twt_eval_structured.py`, `twt_evaluator.py`) rebuilds the policy from `name` + `kwargs`, so no `--policy-arch` is needed at evaluation.

### Logs

`training_log.jsonl` has one row per batch: `batch`, `wall_time`, `batch_seconds`, `update_seconds`, `total_env_steps`, `ep_reward_mean`, `ep_length_mean`, `n_workers_ok`, `n_failed`, plus the PPO metrics (`policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_frac`, …).
The same scalars go to TensorBoard under `--tensorboard-log` (`tensorboard --logdir results/runs/<run>/pipeline/tb_logs`).

---

## Evaluation

### `twt_eval_structured.py` + `twt_eval_report.py` (headline)

Runs the checkpoint and `--baseline <registry name>` over `--splits` × `--seeds` matched scenarios (same `rand_seed` for both policies → identical NS-3 scenario, paired statistics), deterministic actions, and writes one parquet row per `(policy, split, seed, step, STA)` with device class, action and every per-step delta needed downstream.
The report runs automatically (`--no-report` to skip) and can be re-run alone:

```bash
python3.11 ppo-sb3-scripts/twt_eval_report.py results/runs/<run>/pipeline/eval
```

`report.txt` sections: overall (paired Wilcoxon), per device class, REHD energy, by REHD split, by load segment.
Two normalizations of "served" are reported side by side — `served_ratio` (PHY TX / MAC admissions, what an AP can measure) and `delivered_ratio` (AP RX / app-generated, an oracle) — because a REHD's energy gate suppresses packets before admission; the report header explains which one the paper quotes and why.
`--base-scenario-seed` (300000) and `--base-shm-seed` (700000) keep eval scenarios disjoint from the training seed band.

### `twt_evaluator.py` (general)

```bash
python3.11 ppo-sb3-scripts/twt_evaluator.py --checkpoint ckpt.pt --n-episodes 20 --num-workers 10 ...
python3.11 ppo-sb3-scripts/twt_evaluator.py --compare ckpt.pt analytical_md1 random_policy ...
```

Same worker/pool machinery as training with `deterministic=True` and no update; `--compare` takes any mix of checkpoint paths and registry names and writes a side-by-side CSV of per-episode rewards.
`--variable-split` and `--n-stations/--n-rehd` behave as in training.

---

## EDA Stage

`twt_eda_collect.py` is stage 2 of `run_pipeline.sh`.
It runs the wrapper with no policy through the 24 × 25 × 10 grid, one parquet shard per episode, resumable, and `twt_refit_obs_norms.py` fits the 7 per-feature means/stds from the `obs_*` columns.
`EDA_SPAN=1` (what the pipeline sets) cycles each head with a coprime stride so every episode sweeps all values of every head; without it the walk tiles the grid sequentially (PDW fastest) and only spans the schedules after ~240 episodes:

```bash
EDA_SPAN=1 python3.11 ppo-sb3-scripts/twt_eda_collect.py --num-batches 100 --num-workers 20 \
    --n-stations 12 --n-rehd 8 --reward-preset twt_pf_demand_v5 --max-updates 100 --out <run>/eda
python3.11 ppo-sb3-scripts/twt_refit_obs_norms.py --shards <run>/eda/shards --out <run>/eda/obs_warmstart_proposed.json
```

`run_pipeline.sh` then copies the proposal over `obs_warmstart_stats.json` (`NORMS_MODE=apply`), asks (`prompt`), or leaves it (`skip`).

---

## Figures

`make_figs.py` builds the paper's four figures and prints the Table 2 (REHD sustainability) numbers.
With no arguments it reads the latest finished run (the run folder under `results/runs/` whose `pipeline/eval/report.txt` was written last) and writes into that run's `figs/`:

```bash
python3.11 ppo-sb3-scripts/make_figs.py                  # latest run -> results/runs/<run>/figs/
RUN=repro_20260825 STAGES="4" bash reproduce.sh          # same, for a named run
```

On the committed run the four PNGs come out byte-identical to the paper's (and to `results/figs/`).
Each figure starts from matplotlib's default style, with its font sizes set inside its own function, so the output does not depend on which figures ran before it.
Every input can be overridden:

| Variable           | Default                                                       | Figure                               |
| ------------------ | ------------------------------------------------------------- | ------------------------------------ |
| `PAPER_RUN_DIR`    | latest run folder, as above                                   | all                                  |
| `PAPER_FIGS_DIR`   | `<run>/figs/`                                                 | all                                  |
| `PAPER_EVAL_DIR`   | `<run>/pipeline/eval/shards/`                                 | `eval_perclass.png`, Table 2 numbers |
| `PAPER_TRAIN_GLOB` | `<run>/pipeline/train/*/training_log.jsonl`                   | `training_curve.png`                 |
| `PAPER_VCAP_DATA`  | `<run>/figs/vcap_data.csv`, else `results/figs/vcap_data.csv` | `vcap.png`                           |
| —                  | datasheet model in the script                                 | `harvest_curve.png`                  |

`vcap_episode.py <seed> <simId>` produces the raw `Vcap(t)` traces for `vcap.png`: a 4 non-REHD + 16 REHD topology (many harvesters to sample the annulus), REHDs in isolated TWT groups, a fixed action with the widest PDW (`pdw_end = 50 ms`) — a harvest-physics probe, not the agent's behaviour.

---

## Output Structure

Everything a run produces lands under `results/` (`TwtResultsPath()` on the C++ side, `RUN_ROOT` in the drivers):

```
results/
├── data-log/                                   NS-3 per-run trace output, Vcap CSVs, stage-1 PNGs (gitignored)
├── figs/                                       the paper's figures + vcap_data.csv, as submitted
└── runs/<run>/                                 run_<timestamp>/ for new runs; repro_20260825/ is the paper's
    ├── inputs_before/                          action tables + obs norms before this run regenerated them (reproduce.sh)
    ├── figs/                                   make_figs.py output
    └── pipeline/
        ├── eda/
        │   ├── shards/ep_NNNN_seed<seed>.parquet   one per EDA episode (gitignored, regenerable)
        │   ├── manifest.json                       grid, seeds, failures
        │   └── obs_warmstart_proposed.json         refit norms
        ├── train/<run_name>/
        │   ├── ckpt_batch_00010.pt … ckpt_final.pt
        │   ├── hyperparams.json                    every flag + the derived policy kwargs
        │   └── training_log.jsonl
        ├── tb_logs/<run_name>/                     TensorBoard events
        └── eval/
            ├── shards/*.parquet                    per-(policy, split, seed, step, STA) rows
            ├── report.txt
            └── per_class_comparison.csv
```

No driver deletes an earlier run folder (`reproduce.sh` only does with the opt-in `CLEAN=1`).

The committed run is `results/runs/repro_20260825/pipeline/` (checkpoints, EDA manifest and norms, eval shards and report); the EDA shards were dropped from git as regenerable.
