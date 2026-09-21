# TWT-PowerCast Action Tables

The agent's action is a triple of indices `(schedule, assignment, pdw)`.
The first two index the tables in this directory; the third is decoded in `ppo-sb3-scripts/twt_spawn_worker.py`.

These files are required at run time.
The PPO worker, the analytical baselines and the EDA collector all load the two JSON tables when they are imported, and the policy's head sizes are read from them (`twt_spawn_worker.default_obs_kwargs`), so a checkpoint trained against one table cannot be evaluated against another.

## Table of Contents

1. [Files](#files)
1. [Generating the Tables](#generating-the-tables)
1. [`schedule_table.json`](#schedule_tablejson)
1. [`assignment_table.json`](#assignment_tablejson)
1. [The PDW Head](#the-pdw-head)
1. [How the Grid Was Chosen](#how-the-grid-was-chosen)

---

## Files

| File                        | Description                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| `generate_action_tables.py` | Writes both tables; also defines `apply_assignment_pattern()`, the decoder the worker imports |
| `schedule_table.json`       | 24 TWT schedules: group count K × wake budget, equal-duration service periods                 |
| `assignment_table.json`     | 25 STA → group patterns: round-robin, contiguous splits, explicit REHD-tail cuts              |

---

## Generating the Tables

`run_pipeline.sh` stage 0 and `reproduce.sh` stage 1 run the generator.
The output is deterministic: a regenerated table differs from the committed one only in its `generated` timestamp.

```bash
python3.11 exploration-scripts/generate_action_tables.py                 # writes into this directory
python3.11 exploration-scripts/generate_action_tables.py --output-dir X  # elsewhere
```

```
Wrote 24 schedules + 25 assignments
Joint (x10 PDW): 6000 combos (24x25x10)
=== Schedule coverage ===
  by num_groups (K):   {2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 8: 4}
  total_duration min/mean/max: 36 / 59.4 / 84 ms
=== Assignment coverage ===
  by pattern_type: {'round_robin': 4, 'split_n': 9, 'weighted': 12}
```

The grid is set by the constants at the top of the script:

| Constant                | Value                      | Meaning                                                             |
| ----------------------- | -------------------------- | ------------------------------------------------------------------- |
| `MAX_SPAN_MS`           | 95.0                       | the last SP must end by 95 ms of the 102.4 ms beacon interval       |
| `MIN_GROUP_DURATION_MS` | 2.0                        | per-group wake floor                                                |
| `MIN_OFFSET_GAP_MS`     | 2.0                        | guard between consecutive SPs                                       |
| `SCHED_KS`              | `[2, 3, 4, 5, 6, 8]`       | group counts                                                        |
| `BUDGET_FRACS`          | `[0.45, 0.60, 0.75, 0.90]` | fractions of the per-K maximum budget `B_max(K) = 95 − (K − 1)·gap` |
| `RR_NS`                 | `[2, 3, 4, 5]`             | round-robin group counts                                            |
| `SPLIT_NS`              | `[2 … 10]`                 | contiguous-split group counts                                       |
| `HARVEST_TAILS`         | `[4, 6, 8, 10, 12, 14]`    | REHD-tail sizes for the weighted cuts                               |
| `HARVEST_COMMS`         | `[2, 3]`                   | comms groups in front of the tail                                   |

---

## `schedule_table.json`

24 entries, one per `(K, budget)` pair.
Each schedule has K equal service periods laid out back to back with a 2 ms gap, every wake interval one beacon interval long:

```json
{
  "schedule_id": 0,
  "name": "S0_G2_b45",
  "description": "2g equal, durations=[21, 21], total=42ms",
  "num_groups": 2,
  "total_duration_ms": 42,
  "groups": [
    {"group_id": 0, "wake_duration_ms": 21.0, "sp_offset_ms": 0.0},
    {"group_id": 1, "wake_duration_ms": 21.0, "sp_offset_ms": 23.0}
  ]
}
```

Header fields: `beacon_interval_ms`, `max_total_duration_ms` (95), `max_k` (16, the C++ ceiling `MAX_NUM_TWT_GROUPS`), `budget_fracs`, `num_schedules`.
Total wake time ranges from 36 to 84 ms per beacon interval.

---

## `assignment_table.json`

25 entries in three families.
The simulator gives the non-REHD stations the low indices and the REHDs the high ones (`InitializeHeterogeneousNetwork()`), so a pattern that groups a contiguous tail of indices is how a schedule can put the harvesters in their own service period without the AP knowing which stations they are.

| ids   | `pattern_type` | Count | Mapping                                                                                             |
| ----- | -------------- | ----- | --------------------------------------------------------------------------------------------------- |
| 0–3   | `round_robin`  | 4     | STA i → group i mod N, N ∈ {2, 3, 4, 5}: spreads load, no isolation                                 |
| 4–12  | `split_n`      | 9     | contiguous blocks, N ∈ {2 … 10}: the last block isolates the high-index tail                        |
| 13–24 | `weighted`     | 12    | last m ∈ {4, 6, 8, 10, 12, 14} STAs → one harvest group, the other 20 − m → c ∈ {2, 3} comms groups |

When the chosen schedule has fewer groups than a pattern asks for, `apply_assignment_pattern()` uses what is available: `round_robin` and `split_n` shrink N to the schedule's K, and `weighted` merges the surplus groups into its last available group.

---

## The PDW Head

The third head is not a table.
`twt_spawn_worker.apply_pdw_scale_shift()` maps `idx ∈ 0 … 9` to `pdw_end = 5 + 5·idx` ms (idx 0 = empty window, no PDW), then scales and shifts the chosen schedule into the remaining comms region `[pdw_end, 95]` ms: `scale = (95 − pdw_end)/95`, offsets moved past `pdw_end`, wakes floored at 2 ms.
A wide PDW therefore compresses everyone else's service periods.
Over all 10 levels × 24 schedules the latest SP ends at 91.68 ms, inside the 102.4 ms beacon interval.

---

## How the Grid Was Chosen

The first table was a stratified-random 40 × 40 sample.
It was replaced by a systematic 30 × 25 lattice, which the variable-split EDA swept under the current physics, and the present 24 × 25 table was carved from it: a subset of EDA-tested actions with the dead and dominated ones removed.

- Schedules: K = 1 and the 0.30 budget were dead, and K ≥ 7 was cold except for the four K = 8 schedules kept for fine isolation at high REHD counts.
- Assignments: contiguous `split_n` isolation of the REHD tail was the best family at every REHD count (0.76–0.81 REHD served, against ~0.64 for scattered and ~0.58 for all-to-one); `all_to_one`, `interleave` and front/back `weighted` were dropped, and the explicit tail cuts were added so the agent can size the harvest group to any REHD count from 4 to 16.
- PDW: all ten levels were kept, because the best level moves from 0 to 9 as the REHD count grows.
