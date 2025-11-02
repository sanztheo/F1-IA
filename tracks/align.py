from __future__ import annotations

import numpy as np


def procrustes_2d(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Trouve la meilleure transformation de similarité (R, s, t) telle que A ≈ s R B + t.

    Retourne (R 2x2, s scalaire, t 2,). A et B doivent avoir même nombre de points.
    """
    assert A.shape == B.shape and A.shape[1] == 2
    Ac = A - A.mean(axis=0)
    Bc = B - B.mean(axis=0)
    H = Bc.T @ Ac
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = np.trace(R.T @ H) / np.sum(Bc ** 2)
    t = A.mean(axis=0) - s * (R @ B.mean(axis=0))
    return R, float(s), t


def apply_similarity(R: np.ndarray, s: float, t: np.ndarray, P: np.ndarray) -> np.ndarray:
    return (s * (R @ P.T)).T + t

