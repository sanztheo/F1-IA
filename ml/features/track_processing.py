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
    """Construit un centerline stable à partir de `positions` FastF1.

    Étapes robustes:
    - garder un seul pilote (celui avec le plus d'échantillons) pour éviter le « mélange » multi-voitures;
    - ignorer l'ordre temporel et reconstruire une boucle fermée via l'angle polaire autour du centroïde;
    - médiane par bin angulaire, puis lissage et rééchantillonnage.
    """
    if not {"X", "Y"}.issubset(set(pos_df.columns)):
        raise ValueError("Positions FastF1 attendues avec colonnes 'X', 'Y'")

    df = pos_df.copy()
    if "Driver" in df.columns and df["Driver"].notna().any():
        # choisir le pilote avec le plus de points
        vc = df["Driver"].dropna().value_counts()
        if len(vc):
            top = vc.idxmax()
            df = df[df["Driver"] == top]
    elif "DriverNumber" in df.columns and df["DriverNumber"].notna().any():
        vc = df["DriverNumber"].dropna().value_counts()
        if len(vc):
            top = vc.idxmax()
            df = df[df["DriverNumber"] == top]

    xy = df[["X", "Y"]].to_numpy()
    if len(xy) < 10:
        return xy

    # Centroïde et angles
    cx, cy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    ang = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)

    # Bins angulaires et médiane par bin
    nb = min(2000, max(200, len(xy) // 20))
    bins = np.linspace(-np.pi, np.pi, nb + 1)
    idx = np.digitize(ang, bins) - 1
    x_med = np.zeros(nb)
    y_med = np.zeros(nb)
    for b in range(nb):
        mask = idx == b
        if not np.any(mask):
            # interpolation des trous
            x_med[b] = np.nan
            y_med[b] = np.nan
        else:
            x_med[b] = np.median(xy[mask, 0])
            y_med[b] = np.median(xy[mask, 1])

    # combler les NaN par interpolation circulaire
    def _interp_circular(vec: np.ndarray) -> np.ndarray:
        v = vec.copy()
        n = len(v)
        isn = np.isnan(v)
        if np.any(isn):
            x = np.arange(n)
            v[isn] = np.interp(x[isn], x[~isn], v[~isn])
        return v

    x_med = _interp_circular(x_med)
    y_med = _interp_circular(y_med)

    outline = np.column_stack([x_med, y_med])
    # lissage léger
    from scipy.signal import savgol_filter
    win = 21 if len(outline) >= 21 else (len(outline) // 2 * 2 + 1)
    xs = savgol_filter(outline[:, 0], win, 3, mode="wrap")
    ys = savgol_filter(outline[:, 1], win, 3, mode="wrap")
    outline = np.column_stack([xs, ys])

    # rééchantillonnage uniforme en arc-length
    s = arc_length(outline)
    n_pts = 2000
    s_u = np.linspace(0, s[-1], n_pts)
    x_u = np.interp(s_u, s, outline[:, 0])
    y_u = np.interp(s_u, s, outline[:, 1])
    return np.column_stack([x_u, y_u])
