from __future__ import annotations

import numpy as np
from typing import Dict, Any


def speed_profile_from_curvature(kappa: np.ndarray, mu: float, a_lat_max_base: float = 12.0) -> np.ndarray:
    grav = 9.80665
    a_lat = a_lat_max_base * mu
    v = np.sqrt(np.maximum(1e-3, (a_lat * grav) / (np.abs(kappa) + 1e-6)))
    return v


def simulate_lap(
    xy: np.ndarray,
    kappa: np.ndarray,
    v_limit: np.ndarray,
    a_long_max: float = 9.0,
) -> Dict[str, Any]:
    """Intègre une passe avant/arrière pour les contraintes longitudinales.

    Retourne temps total, profil vitesse.
    """
    # ds
    ds = np.zeros(len(xy))
    ds[1:] = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
    ds[0] = ds[1]

    v = np.minimum(v_limit.copy(), 120.0)  # borne haute raisonnable

    # Passe avant (accélération)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(max(1e-6, v[i - 1] ** 2 + 2 * a_long_max * ds[i])))
    # Passe arrière (freinage)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(max(1e-6, v[i + 1] ** 2 + 2 * a_long_max * ds[i + 1])))

    # Temps
    with np.errstate(divide="ignore"):
        dt = np.where(v > 1e-3, ds / v, 0.0)
    T = float(np.sum(dt))
    return {"time_s": T, "v": v, "ds": ds}

