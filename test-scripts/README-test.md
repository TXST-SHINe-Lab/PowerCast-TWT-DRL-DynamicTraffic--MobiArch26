# TWT-PowerCast Smoke-Test Scripts

Two self-contained checks of the NS-3 ↔ Python bridge and the worker/orchestrator process model.
Run them once after every build (and after any change to `pb_twt_wrapper_py.py`, `twt_spawn_worker.py` or the shared-memory plumbing) — before spending hours on EDA or training.

Both scripts import the pipeline package from `../ppo-sb3-scripts/` and resolve the NS-3 root from their own location, so they run from anywhere.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Files](#files)
  - [`twt_smoke_one_worker.py`](#twt_smoke_one_workerpy)
  - [`twt_stress_test.py`](#twt_stress_testpy)
- [What a Failure Looks Like](#what-a-failure-looks-like)

---

## Overview

Every training/eval driver in this repo works the same way: a parent process pickles a policy, spawns N worker processes, each worker boots one NS-3 simulation over `ns3-ai` shared memory, runs an episode, puts its rollout on a queue and dies with `os._exit(0)`.
The two failure modes that cost the most time are (a) a worker that never returns (shared-memory deadlock, wrong Python linked into the pybind `.so`) and (b) leaked state between episodes (`/dev/shm/My*` segments or orphaned `twt-powercast-main-simulation` processes) that deadlocks the *next* run.
These scripts exercise that path and check for those leaks.

| Script                    | Episodes | Verifies                                                                |
| ------------------------- | -------- | ----------------------------------------------------------------------- |
| `twt_smoke_one_worker.py` | 1        | end-to-end rollout, phase timings, clean teardown                       |
| `twt_stress_test.py`      | 20       | parallel workers with disjoint segments, back-to-back batches, no leaks |

---

## Quick Start

```bash
# 1. Activate the venv and go to the NS-3 root (the wrapper chdirs there anyway)
source ~/NS3-project/EHRL/bin/activate
cd ~/NS3-project/ns-allinone-3.44/ns-3.44

# 2. One worker, one short episode (2 warm-up + 10 real steps)
python3.11 contrib/ai/examples/rl-twt-powercast/test-scripts/twt_smoke_one_worker.py

# 3. Four workers x five batches of 5-step episodes
python3.11 contrib/ai/examples/rl-twt-powercast/test-scripts/twt_stress_test.py
```

Both print `VERDICT: PASS` and exit 0 on success.

---

## Files

### `twt_smoke_one_worker.py`

Spawns **one** worker with a freshly-initialised (untrained) `mlp_ppo` policy and runs a single 10-step episode after 2 warm-up steps.
It mirrors `twt_batch_orchestrator.py` with `N = 1`: builds the policy through the `twt_models` registry, pickles `{name, kwargs, state_dict}`, starts `twt_spawn_worker.run_episode` in a `spawn`ed process, and waits on the result queue.

```bash
python3.11 test-scripts/twt_smoke_one_worker.py \
    --n-stations 12 --n-rehd 4 \             # topology (default 12 + 4)
    --reward-preset twt_pf_demand_v1         # any name in reward_functions.REWARD_PRESETS
```

What it checks and prints:

- the registered policy names and the auto-derived policy kwargs (`obs_dim = num_sta × 7`, `num_schedules = 24`, `num_assignments = 25`, `num_pdw_levels = 10`) — a quick way to confirm the action tables loaded;
- `success`, `ep_length`, `ep_reward` from the rollout;
- **phase timings** measured inside the worker: imports, policy load, NS-3 boot until the first `EnvStruct`, warm-up steps, real steps, bootstrap value pass, and the parent-side `close() + os._exit`;
- `/dev/shm` leftovers and orphaned `twt-powercast-main-simulation` PIDs after teardown — both must be empty.

```
[parent] topology: 12 STA + 4 REHD = 16 -> policy kwargs={'obs_dim': 112, 'features_per_sta': 7, 'num_schedules': 24, 'num_assignments': 25, 'num_pdw_levels': 10, 'hidden_dim': 256}
[parent] Worker done in 18.4s (parent wall-clock)
[parent] success     : True
[parent] ep_length   : 10
[parent]   boot (reset)   :   8.89s   <-- NS-3 startup until first env arrives
[parent]   real steps     :   6.13s (10 steps, avg 0.61s/step)
[parent] /dev/shm leftovers: []
[parent] orphan twt-main pids: (none)
[parent] VERDICT: PASS
```

### `twt_stress_test.py`

Runs the orchestrator's batch loop under sustained load: `--num-workers` processes per batch, `--num-batches` batches back-to-back, each episode only `--max-steps` NS-3 steps so the test is about the *process model*, not the physics.

```bash
python3.11 test-scripts/twt_stress_test.py \
    --num-workers 4 --num-batches 5 \        # 20 episodes (defaults)
    --max-steps 5 --warmup-steps 1 \         # short episodes (raise for realism)
    --reward-preset balanced --base-seed 95000 --worker-timeout 600
```

Per batch it reports wall time, `/dev/shm` leftovers and orphan PIDs; at the end it prints a results block and a three-part verdict:

```
  Total episodes      : 20
  Successful episodes : 20
  Per-batch wall time : 10.9s, 13.5s, 12.2s, 11.3s, 13.1s
  VERDICT: PASS
    episodes_all_ok = True     # every worker returned a rollout
    no_shm_leftover = True     # no /dev/shm/My* after the last batch
    no_ns3_orphans  = True     # no twt-powercast-main-simulation left running
    stable_batch_t  = True     # last batch took <= 1.5x the first (no accumulating slowdown)
```

Seeds are `base_seed + batch·workers + worker_id`, so every episode gets its own `MySeg_<seed>` segment; `stable_batch_t` catches the slow-leak case where batches keep getting slower (last batch > 1.5× the first fails it).

---

## What a Failure Looks Like

| Symptom                                              | Usual cause                                                                                                 |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: pb_twt_powercast_interface_py` | pybind `.so` not built, or built against a different Python — re-run `./ns3 configure` with the venv pinned |
| worker times out, NS-3 process alive                 | stale `/dev/shm/My*` from a crashed run: `rm -f /dev/shm/My*` and retry                                     |
| `IndexError` inside the worker on the first step     | action tables changed size after a checkpoint / EDA was produced — re-run `generate_action_tables.py`       |
| `stable_batch_t = False` only                        | machine contention (another run active) rather than a leak — re-run when idle before investigating          |
