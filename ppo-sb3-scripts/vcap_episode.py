#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Run ONE episode with --logVcap enabled, to tap each REHD's V_cap + position
for the paper's vcap figure. Invoked by reproduce.sh stage 3 as:

    vcap_episode.py <seed> <simId>          # seeds 42-45 -> simId 7342-7345

writing results/data-log/vcap_trace_<simId>.csv. logVcap/vcapIntervalMs/simId are
injected into the ns-3 run via the Experiment.run setting dict (no wrapper edit).

NOT the paper's headline scenario, deliberately. This is a HARVEST-PHYSICS probe,
so the topology is inverted to 4 non-REHD + 16 REHD (train/eval run 12 + 8): many
harvesters to sample the spawn annulus densely, few competing STAs. The action is
a fixed constant (sched 13, assign 7, pdw 9 -> pdw_end = 50 ms, the widest window
of the 10 levels) with REHDs in isolated TWT groups -- NOT a trained policy. So
the figure shows what the harvester does under a known-favorable schedule, not
what the agent chose. Describe it that way.

The mechanics ARE faithful: real TWT schedule, real PDW window where STAs sleep
and harvest, realistic traffic (C++ default trafficScale).
"""

import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # rl-twt-powercast example dir
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)  # pb_twt_wrapper_py
sys.path.insert(
    0, os.path.join(PARENT, "exploration-scripts")
)  # generate_action_tables
import twt_spawn_worker as W  # sets up sys.path
from twt_spawn_worker import build_action_dict, load_action_tables
from pb_twt_wrapper_py import TWTWrapper

# --- inject logVcap/vcapIntervalMs/simId into the ns-3 run (setting dict -> CLI) ---
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SIMID = int(sys.argv[2]) if len(sys.argv) > 2 else 7400

import ns3ai_utils

_orig_run = ns3ai_utils.Experiment.run


def _run_with_vcap(self, setting=None, **kw):
    setting = dict(setting or {})
    setting.setdefault("logVcap", "true")
    setting.setdefault("vcapIntervalMs", 100)
    setting.setdefault("simId", SIMID)
    return _orig_run(self, setting=setting, **kw)


ns3ai_utils.Experiment.run = _run_with_vcap


def main():
    st, at = load_action_tables(PARENT)
    SCHED, ASGN, PDW = (
        13,
        7,
        9,
    )  # K=5 sched, seq_5 assign (STAs g0-2, REHDs g3-4), PDW end=50 ms
    wrapper = TWTWrapper(
        log_dir=os.path.join(PARENT, "results", "data-log"),
        enable_logging=False,
        verbose=False,
        quiet_ns3=True,
        disable_ns3_traces=True,
        n_stations=4,
        n_rehd=16,
    )
    k = -1
    try:
        env = wrapper.reset(seed=SEED, rand_seed=SEED)
        num_sta = env.get("num_sta", 20)
        action = build_action_dict(SCHED, ASGN, st, at, num_sta, pdw_idx=PDW)
        for k in range(45):
            env, done = wrapper.step(action)
            if done:
                break
        print(f"EPISODE DONE: {k + 1} steps, num_sta={num_sta}, pdw_idx={PDW}")
    finally:
        wrapper.close()


if __name__ == "__main__":
    main()
    os._exit(0)
