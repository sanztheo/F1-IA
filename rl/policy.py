from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any


def init_mlp(sizes: Tuple[int, ...], rng: np.random.Generator | None = None) -> Dict[str, np.ndarray]:
    """Init MLP avec biais de sortie favorables à l'exploration (gaz>frein).

    Dernière couche (3 sorties: steer, throttle, brake) reçoit des biais:
    - steer: 0.0 (neutre)
    - throttle: +1.5  → sigm(1.5) ≈ 0.82
    - brake:   -2.0  → sigm(-2.0) ≈ 0.12
    """
    rng = rng or np.random.default_rng(0)
    params: Dict[str, np.ndarray] = {}
    Lm1 = len(sizes) - 1
    for i in range(Lm1):
        w = rng.normal(0, 1/np.sqrt(sizes[i]), size=(sizes[i], sizes[i+1]))
        b = np.zeros((sizes[i+1],), dtype=np.float32)
        # biais de sortie utiles
        if i == Lm1 - 1 and sizes[i+1] >= 3:
            b[:3] = np.array([0.0, 1.5, -2.0], dtype=np.float32)
        params[f"W{i}"] = w.astype(np.float32)
        params[f"b{i}"] = b
    return params


def forward(params: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = x.astype(np.float32)
    L = len(params)//2
    for i in range(L):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < L - 1:
            h = np.tanh(h)
    # map outputs to actions
    steer = np.tanh(h[0])
    throttle = 1/(1+np.exp(-h[1]))
    brake = 1/(1+np.exp(-h[2]))
    return np.array([steer, throttle, brake], dtype=np.float32)


def mutate(params: Dict[str, np.ndarray], sigma: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    # Anomaly policies (e.g., {'__fullsend__': True}) are not mutated like MLP weights
    if params.get("__fullsend__"):
        return {"__fullsend__": True}
    out: Dict[str, np.ndarray] = {}
    for k, v in params.items():
        if hasattr(v, "shape"):
            out[k] = v + rng.normal(0, sigma, size=v.shape).astype(v.dtype)
        else:
            # non-array safety
            out[k] = v
    return out
