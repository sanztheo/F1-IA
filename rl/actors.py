from __future__ import annotations

import numpy as np
from typing import Dict

try:
    # Local import to avoid circulars in some tools
    from .models.mlp_policy import forward as mlp_forward
except Exception:  # pragma: no cover
    mlp_forward = None  # type: ignore


def act(policy: Dict, obs: np.ndarray) -> np.ndarray:
    """Unified action function supporting both heuristic and NN policies.

    Heuristics accepted:
    - {"__fullsend__": True}
    - {"__pursuit__": True} with optional {"__noisy__": True}
    Otherwise, uses MLP forward.
    """
    if policy.get("__fullsend__"):
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    if policy.get("__pursuit__"):
        lat_n = float(obs[0])  # [-1..1] lateral error normalized
        head_n = float(obs[1])  # [-1..1] heading error normalized
        k_lat, k_head = 0.8, 1.2
        steer = float(np.clip(-(k_lat * lat_n + k_head * head_n), -1.0, 1.0))
        throttle = float(np.clip(0.9 - 0.4 * abs(head_n) - 0.2 * abs(lat_n), 0.0, 1.0))
        a = np.array([steer, throttle, 0.0], dtype=np.float32)
        if policy.get("__noisy__"):
            a += np.array([np.random.normal(0, 0.05), np.random.normal(0, 0.05), 0.0], dtype=np.float32)
            a[0] = float(np.clip(a[0], -1.0, 1.0))
            a[1] = float(np.clip(a[1], 0.0, 1.0))
        return a

    # NN policy
    if mlp_forward is None:
        raise RuntimeError("MLP forward not available")
    a = mlp_forward(policy, obs)
    if policy.get("__noisy__"):
        a += np.array([np.random.normal(0, 0.05), np.random.normal(0, 0.05), 0.0], dtype=np.float32)
        a[0] = float(np.clip(a[0], -1.0, 1.0))
        a[1] = float(np.clip(a[1], 0.0, 1.0))
    return a

