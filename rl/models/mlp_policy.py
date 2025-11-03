from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def init_mlp(sizes: Tuple[int, ...], rng: np.random.Generator | None = None) -> Dict[str, np.ndarray]:
    """Initialize a small MLP policy with exploration‑friendly output biases.

    Output dims (3): steer, throttle, brake.
    Biases push throttle high and brake low at start to avoid stall.
    """
    rng = rng or np.random.default_rng(0)
    params: Dict[str, np.ndarray] = {}
    Lm1 = len(sizes) - 1
    for i in range(Lm1):
        w = rng.normal(0, 1 / np.sqrt(sizes[i]), size=(sizes[i], sizes[i + 1]))
        b = np.zeros((sizes[i + 1],), dtype=np.float32)
        if i == Lm1 - 1 and sizes[i + 1] >= 3:
            b[:3] = np.array([0.0, 1.5, -2.0], dtype=np.float32)
        params[f"W{i}"] = w.astype(np.float32)
        params[f"b{i}"] = b
    return params


def forward(params: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = x.astype(np.float32)
    L = len(params) // 2
    for i in range(L):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if i < L - 1:
            h = np.tanh(h)
    steer = np.tanh(h[0])
    throttle = 1.0 / (1.0 + np.exp(-h[1]))
    brake = 1.0 / (1.0 + np.exp(-h[2]))
    return np.array([steer, throttle, brake], dtype=np.float32)


def mutate(params: Dict[str, np.ndarray], sigma: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    if params.get("__fullsend__") or params.get("__pursuit__"):
        # Heuristic pseudo‑policies are passed through
        return dict(params)
    out: Dict[str, np.ndarray] = {}
    for k, v in params.items():
        if hasattr(v, "shape"):
            out[k] = v + rng.normal(0, sigma, size=v.shape).astype(v.dtype)
        else:
            out[k] = v
    return out


def is_heuristic(params: Dict[str, np.ndarray]) -> bool:
    return bool(params.get("__fullsend__") or params.get("__pursuit__"))

