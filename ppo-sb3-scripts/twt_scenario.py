# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Variable REHD/non-REHD split sampler (2026-06-12).

Keeps the TOTAL fixed at 20 (locked-topology / fixed-token-set constraint, memory
feedback-fixed-topology-no-padding) but varies HOW MANY of the 20 are REHD per scenario,
so the REHD-energy dimension differs across runs — and, unlike the hidden harvest geometry,
the split shows up in AGGREGATE signals (total bytes/airtime/queue scale with the non-REHD
count) → observable adaptivity. Combined with the already-active 5-segment over/under
traffic drift, this gives the agent both inter-scenario (split) and intra-episode (segments)
adaptivity, both observable.

Deterministic per scenario_seed (splitmix64, matching the C++ over/under coin style) so a
CRN group (shared rand_seed) shares its split — CRN-safe. n_rehd ~ uniform integer in
[lo, hi]; n_stations = total - n_rehd.
"""

_MASK = 0xFFFFFFFFFFFFFFFF


def _splitmix64(x):
    x &= _MASK
    x = (x * 0x9E3779B97F4A7C15) & _MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK
    x ^= x >> 31
    return x


def sample_split(scenario_seed, total=20, lo=4, hi=16):
    """Return (n_stations, n_rehd) for a scenario. n_rehd ~ U{lo..hi}, n_stations = total-n_rehd.
    Deterministic in scenario_seed (CRN-safe). Default lo=4,hi=16 keeps both classes >= 4.
    """
    n_rehd = lo + (_splitmix64(int(scenario_seed) + 1) % (hi - lo + 1))
    n_rehd = int(min(hi, max(lo, n_rehd)))
    return int(total - n_rehd), n_rehd


if __name__ == "__main__":
    # sanity: distribution over a range of seeds + CRN-determinism
    import collections

    c = collections.Counter()
    for s in range(200000, 200000 + 5000):
        ns, nr = sample_split(s)
        assert ns + nr == 20 and 4 <= nr <= 16 and 4 <= ns <= 16
        c[nr] += 1
    assert sample_split(200000) == sample_split(200000)  # deterministic
    print("n_rehd distribution over 5000 seeds (expect ~roughly uniform 4..16):")
    for nr in sorted(c):
        print(f"  n_rehd={nr:2d} n_sta={20-nr:2d}: {c[nr]:4d} ({100*c[nr]/5000:.1f}%)")
    print("determinism OK; examples:", [sample_split(200000 + i) for i in range(6)])
