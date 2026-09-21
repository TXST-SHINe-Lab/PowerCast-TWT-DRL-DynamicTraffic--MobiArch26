# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Rolling work-pool: dispatch-on-completion process pool for heterogeneous cores.

The collector/orchestrator are batch-synchronous (spawn N, wait for ALL N), so on the
DGX's big.LITTLE cores the batch is gated by whichever worker landed on a slow A725 core.
This pool keeps `pool_size` slots busy and feeds
the next work item to whichever slot frees first — fast X925 slots naturally pull more
items, no per-batch barrier (only the tail when < pool_size items remain). No pinning.

KEY for PPO on-policy correctness + CRN: the *caller* pre-builds the work items (each
carries its own unique SHM `seed` and its CRN-group `rand_seed`), so dispatch is dynamic
but scenario identity travels with the item regardless of which slot runs it. Collect K
items under a frozen policy payload, THEN update => on-policy with a small tail-drain.

Usage:
    items = [{"wid": i, "seed": s, "rand_seed": rs, ...}, ...]   # kwargs per worker
    results = run_rolling_pool(items, run_episode_target, pool_size=24)

`worker_target(result_queue=<Queue>, **item)` MUST put exactly one result dict on
result_queue and end with os._exit(0) (ns3-ai SHM teardown discipline).

CRITICAL — flush before os._exit: mp.Queue.put() is async (a feeder thread writes to the
pipe). os._exit(0) kills the process *immediately*, so a result put() right before it is
LOST unless the feeder has flushed. Either (a) `result_queue.close(); result_queue.join_thread()`
right after put() (deterministic), or (b) rely on a >~0.2 s teardown between put() and exit
(the real run_episode flushes during its ~1 s wrapper.close()). Verified: without a flush,
a fast worker's result never arrives and the pool times out. Put the result FIRST, then do
SHM/wrapper teardown, then os._exit.
"""

import multiprocessing as mp
import queue as _queue
import time as _time


def run_rolling_pool(
    work_items,
    worker_target,
    pool_size,
    result_timeout=900,
    on_result=None,
    on_dispatch=None,
    deadline=None,
):
    """Run `work_items` through `pool_size` concurrent slots, dispatch-on-completion.

    Returns the list of result dicts (in completion order). Tolerant of a hung slot:
    if a result_queue.get times out, returns what was collected so far (caller decides).
    """
    items = list(work_items)
    n_total = len(items)
    if n_total == 0:
        return []
    pool_size = max(1, min(pool_size, n_total))

    result_queue = mp.Queue()
    live = {}  # pid -> Process (currently dispatched, maybe finished-not-reaped)
    next_idx = 0
    results = []

    def _spawn():
        nonlocal next_idx
        if next_idx >= n_total:
            return False
        item = items[next_idx]
        next_idx += 1
        p = mp.Process(
            target=worker_target, kwargs={**item, "result_queue": result_queue}
        )
        p.start()
        live[p.pid] = p
        if on_dispatch:
            on_dispatch(item, p.pid)
        return True

    def _reap():
        for pid in [pid for pid, p in list(live.items()) if not p.is_alive()]:
            live.pop(pid).join(timeout=1)

    # prime the pool
    for _ in range(pool_size):
        _spawn()

    # Collect + refill, keeping `pool_size` slots alive until the queue is exhausted.
    # If `deadline` (epoch seconds) is set, stop DISPATCHING new items once past it, let the in-flight workers finish, then return what was collected (hard wall-clock cap).
    while len(results) < n_total:
        if deadline is not None and _time.time() >= deadline and not live:
            break  # past the wall-clock deadline and nothing in flight -> stop
        try:
            r = result_queue.get(timeout=result_timeout)
        except _queue.Empty:
            break  # a slot hung; bail with partial results (caller handles)
        results.append(r)
        if on_result:
            on_result(r, len(results), n_total)
        _reap()
        if deadline is None or _time.time() < deadline:
            while len(live) < pool_size and next_idx < n_total:
                _spawn()
        # else: past the deadline -> dispatch no new items; just drain the in-flight ones

    # drain/join survivors
    try:
        result_queue.close()
    except Exception:
        pass
    for p in list(live.values()):
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    return results
