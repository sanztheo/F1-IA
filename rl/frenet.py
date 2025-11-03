from __future__ import annotations

import numpy as np
from typing import Tuple


def arc_length(xy: np.ndarray) -> np.ndarray:
    s = np.zeros(len(xy), dtype=float)
    if len(xy) <= 1:
        return s
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s[1:] = np.cumsum(ds)
    return s


def tangents_normals(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    t = np.stack([dx, dy], axis=1)
    n = np.stack([-dy, dx], axis=1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    ang = np.arctan2(t[:, 1], t[:, 0])
    return t, n, ang


def curvature(xy: np.ndarray) -> np.ndarray:
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
    return (dx * ddy - dy * ddx) / denom


def lookahead(values: np.ndarray, idx: int, count: int, step: int) -> np.ndarray:
    n = len(values)
    out = []
    j = idx
    for _ in range(count):
        j = (j + step) % n
        out.append(values[j])
    return np.array(out, dtype=float)

