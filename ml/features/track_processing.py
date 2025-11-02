from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def arc_length(xy: np.ndarray) -> np.ndarray:
    ds = np.sqrt(((np.diff(xy, axis=0)) ** 2).sum(axis=1))
    s = np.zeros(len(xy))
    s[1:] = np.cumsum(ds)
    return s


def curvature(xy: np.ndarray) -> np.ndarray:
    """Courbure discrète 2D (approx. par dérivées finies)."""
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa


def normals(xy: np.ndarray) -> np.ndarray:
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    tang = np.stack([dx, dy], axis=1)
    n = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return n / n_norm


def offset_line(xy_center: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    n = normals(xy_center)
    return xy_center + n * offsets[:, None]


def posdata_to_centerline(pos_df: pd.DataFrame) -> np.ndarray:
    """Construit un centerline approximatif à partir des positions FastF1.

    On regroupe par tour, on normalise par la distance, puis on moyenne.
    """
    if not {"X", "Y"}.issubset(set(pos_df.columns)):
        # Colonnes FastF1 typiques: 'X', 'Y'
        raise ValueError("Positions FastF1 attendues avec colonnes 'X', 'Y'")

    xy = pos_df[["X", "Y"]].to_numpy()
    # tri sommaire par temps croissant si présent
    if "Date" in pos_df.columns:
        pos_df = pos_df.sort_values("Date")
        xy = pos_df[["X", "Y"]].to_numpy()
    # Option minimale: lissage léger (médiane) pour réduire le bruit
    from scipy.signal import medfilt

    k = 7 if len(xy) >= 7 else (len(xy) // 2 * 2 + 1)
    xy_smooth = np.column_stack([medfilt(xy[:, 0], k), medfilt(xy[:, 1], k)])
    # Échantillonnage uniforme en arc-length
    s = arc_length(xy_smooth)
    if s[-1] <= 0:
        return xy_smooth
    n_pts = min(2000, len(xy_smooth))
    s_u = np.linspace(0, s[-1], n_pts)
    x_u = np.interp(s_u, s, xy_smooth[:, 0])
    y_u = np.interp(s_u, s, xy_smooth[:, 1])
    return np.column_stack([x_u, y_u])

