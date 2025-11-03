from __future__ import annotations

import numpy as np
from typing import Tuple

from rl.track_env import TrackEnv


def plan_cem(env: TrackEnv, horizon: int = 20, iters: int = 3,
             pop: int = 128, elites: int = 16,
             init_std: Tuple[float, float, float] = (0.3, 0.2, 0.1)) -> np.ndarray:
    """Cross‑Entropy Method planner on action sequences.

    Returns the first action of the optimized sequence.
    Actions are (steer[-1..1], throttle[0..1], brake[0..1]).
    Cost = negative reward (env.step) with strong off‑track penalty.
    """
    assert elites > 0 and elites <= pop
    rng = np.random.default_rng(0)

    # Mean sequence (start: straight + throttle)
    mean = np.zeros((horizon, 3), dtype=np.float32)
    mean[:, 0] = 0.0  # steer
    mean[:, 1] = 0.8  # throttle
    mean[:, 2] = 0.0  # brake
    std = np.array(init_std, dtype=np.float32)

    def _roll(seq: np.ndarray) -> float:
        snap = env.snapshot()
        total_cost = 0.0
        for a in seq:
            # clamp
            aa = np.array([np.clip(a[0], -1.0, 1.0), np.clip(a[1], 0.0, 1.0), np.clip(a[2], 0.0, 1.0)], dtype=np.float32)
            _, r, done, info = env.step(aa)
            cost = -float(r)
            # privilégier le frein si sur‑vitesse
            v_ref = float(info.get("v_ref", 0.0))
            over = float(info.get("overspeed", 0.0))
            if over > 0.5:  # au‑delà d'~2 km/h
                thr = float(aa[1])
                brk = float(aa[2])
                # pénaliser le throttle en excès et le "coast" (thr haut, brk bas)
                cost += 0.8 * over * max(0.0, thr - 0.2)
                if brk < 0.1 and thr > 0.2:
                    cost += 0.6 * over
                # bonus léger pour freins modérés en approche (trail braking)
                cost -= 0.4 * over * min(0.8, brk)
            if not info.get("on", True):
                cost += 10.0  # big penalty off track
                total_cost += cost
                break
            total_cost += cost
            if info.get("lap_done", False):
                # rewarding finishing the lap quickly
                total_cost -= 5.0
                break
        env.restore(snap)
        return total_cost

    for _ in range(iters):
        seqs = rng.normal(0.0, 1.0, size=(pop, horizon, 3)).astype(np.float32)
        seqs = mean[None, :, :] + seqs * std[None, None, :]
        costs = np.zeros((pop,), dtype=np.float32)
        for i in range(pop):
            costs[i] = _roll(seqs[i])
        elite_idx = np.argsort(costs)[:elites]
        elite = seqs[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.reshape(-1, 3).std(axis=0) + 1e-3
        # keep some throttle bias
        mean[:, 1] = np.clip(mean[:, 1], 0.3, 1.0)
        mean[:, 2] = np.clip(mean[:, 2], 0.0, 0.7)

    return np.array([np.clip(mean[0, 0], -1.0, 1.0), np.clip(mean[0, 1], 0.0, 1.0), np.clip(mean[0, 2], 0.0, 1.0)], dtype=np.float32)
