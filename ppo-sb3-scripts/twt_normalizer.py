#!/usr/bin/env python3.11
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Running normalization for the custom (non-SB3) PPO pipeline.

We are NOT on SB3, so `VecNormalize` (a VecEnv wrapper) does not apply. This ports its
MECHANISM — Welford/Chan parallel running mean+var (RunningMeanStd) + on-the-fly z-score
+ clip + persisted stats — but adapted to our setting:

  * OBS is a SET: (num_sta, F) per step. We keep PER-FEATURE stats (shape (F,)) SHARED
    across the 20 STA tokens — NOT per-flat-dimension like vanilla VecNormalize (which
    would break the set encoder's permutation invariance and waste 20x the samples).
  * RETURN normalization is a scalar RunningMeanStd over discounted returns; rewards are
    scaled by 1/sqrt(return_var) (SB3 VecNormalize `norm_reward` equivalent) so value
    targets are unit-scaled.

Preprocessing theory satisfied: variance-stabilizing transform (log1p, in the obs builder)
-> running standardization (here: zero-mean / unit-var, ADAPTIVE) -> clip outliers (±clip σ).
Fixes the all-positive/uncentered inputs (LeCun, Efficient BackProp 1998) + distribution
shift (the EDA-fit norms were on the action-SWEEP distribution) + return scaling (PopArt-lite).

Stats are plain python/numpy so they serialize into the self-describing `.pt` checkpoint
alongside the policy state_dict (the classic VecNormalize gotcha = forgetting to persist
them -> eval without normalization -> broken policy).
"""

import numpy as np


class RunningMeanStd:
    """Welford/Chan parallel running mean & variance. `shape=()` => scalar (returns);
    `shape=(F,)` => per-feature vector (obs). `update(x)` reduces over ALL leading axes,
    so an obs batch of shape (B, num_sta, F) updates the (F,) stats from B*num_sta samples.
    """

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.shape = tuple(shape)
        self.mean = np.zeros(self.shape, np.float64)
        self.var = np.ones(self.shape, np.float64)
        self.count = float(epsilon)

    def update(self, x):
        x = np.asarray(x, np.float64)
        if self.shape:
            x = x.reshape(-1, self.shape[-1])  # (N, F) — last axis is the feature
            bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        else:
            x = x.reshape(-1)
            bm, bv, bc = float(x.mean()), float(x.var()), x.shape[0]
        if bc == 0:
            return
        self._update_from_moments(bm, bv, bc)

    def _update_from_moments(self, bm, bv, bc):
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        M2 = m_a + m_b + delta * delta * self.count * bc / tot
        self.var = M2 / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x, clip: float = 5.0):
        z = (np.asarray(x, np.float64) - self.mean) / self.std
        if clip is not None:
            z = np.clip(z, -clip, clip)
        return z.astype(np.float32)

    def state_dict(self):
        return {
            "shape": list(self.shape),
            "mean": np.asarray(self.mean).tolist(),
            "var": np.asarray(self.var).tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, d):
        self.shape = tuple(d["shape"])
        self.mean = np.asarray(d["mean"], np.float64)
        self.var = np.asarray(d["var"], np.float64)
        self.count = float(d["count"])
        return self

    @classmethod
    def from_stats(cls, mean, std, count: float = 1.0e6):
        """Warm-start from precomputed mean/std (e.g. the round-2 EDA per-feature stats),
        so normalization is calibrated from step 1 instead of cold-start noisy."""
        mean = np.asarray(mean, np.float64)
        r = cls(shape=mean.shape)
        r.mean = mean
        r.var = np.asarray(std, np.float64) ** 2
        r.count = float(count)
        return r


class ReturnNormalizer:
    """Scales rewards by 1/std(discounted return) — SB3 VecNormalize `norm_reward` style.
    Tracks a running discounted return per parallel stream, updates a scalar RunningMeanStd
    with it, and divides rewards by its std. Does NOT subtract the mean (keeps reward sign).
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8, clip: float = 10.0):
        self.ret_rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip

    def update_and_scale(self, rewards, dones):
        """rewards, dones: 1-D arrays for ONE trajectory (in time order). Returns scaled
        rewards. Maintains the discounted-return accumulator, resetting at episode bounds.
        """
        rewards = np.asarray(rewards, np.float64)
        dones = np.asarray(dones)
        ret = 0.0
        returns = np.empty_like(rewards)
        for t in range(len(rewards)):
            ret = rewards[t] + self.gamma * ret * (1.0 - float(dones[t]))
            returns[t] = ret
        self.ret_rms.update(returns)
        scaled = rewards / (self.ret_rms.std + self.epsilon)
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {
            "ret_rms": self.ret_rms.state_dict(),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    def load_state_dict(self, d):
        self.ret_rms.load_state_dict(d["ret_rms"])
        self.gamma = d.get("gamma", self.gamma)
        self.epsilon = d.get("epsilon", self.epsilon)
        self.clip = d.get("clip", self.clip)
        return self


# --- Self-test (offline, no NS-3) ---
# Verify the running stats match a one-shot numpy estimate, the normalize+clip is correct, and save/load round-trips.
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    F = 7
    # heavy-tailed-ish per-feature data, different scales per feature
    true_mean = rng.uniform(0.1, 0.8, F)
    true_std = rng.uniform(0.1, 0.5, F)
    data = rng.randn(50, 20, F) * true_std + true_mean  # (steps, sta, F)

    rms = RunningMeanStd(shape=(F,))
    for t in range(data.shape[0]):  # stream batches of (20, F)
        rms.update(data[t])
    flat = data.reshape(-1, F)
    err_mean = np.abs(rms.mean - flat.mean(0)).max()
    err_var = np.abs(rms.var - flat.var(0)).max()
    print(
        f"[test] running vs one-shot:  max|Δmean|={err_mean:.2e}  max|Δvar|={err_var:.2e}"
    )
    assert err_mean < 1e-6 and err_var < 1e-6, "running stats diverge from one-shot"

    z = rms.normalize(flat, clip=5.0)
    print(
        f"[test] normalized: mean≈{z.mean():+.3f} (→0)  std≈{z.std():.3f} (→1)  "
        f"range[{z.min():.2f},{z.max():.2f}] (clip ±5)"
    )
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05

    # save/load round-trip
    rms2 = RunningMeanStd(shape=(F,)).load_state_dict(rms.state_dict())
    assert np.allclose(rms2.mean, rms.mean) and np.allclose(rms2.var, rms.var)

    # warm-start
    ws = RunningMeanStd.from_stats(true_mean, true_std, count=1e6)
    assert np.allclose(ws.mean, true_mean) and np.allclose(
        ws.std, np.sqrt(true_std**2 + 1e-8), atol=1e-3
    )

    # return normalizer: rewards ~N(-1, 0.7), should come out ~unit-scaled
    rew = rng.randn(95) * 0.7 - 1.0
    dn = np.zeros(95)
    dn[-1] = 1.0
    rn = ReturnNormalizer(gamma=0.99)
    sc = rn.update_and_scale(rew, dn)
    print(
        f"[test] return-norm: raw std={rew.std():.3f} -> scaled reward std={sc.std():.3f}  "
        f"(return std est={rn.ret_rms.std:.3f})"
    )
    print("[test] ALL PASS")
