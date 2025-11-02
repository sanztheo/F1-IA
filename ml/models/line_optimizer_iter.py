from __future__ import annotations

import numpy as np
from typing import Tuple

from ..features.track_processing import curvature, offset_line
from ..models.phys_params import LateralEnvelope
from simulation.lap_simulator import simulate_lap
import cvxpy as cp


def optimize_line_iterative(
    centerline_xy: np.ndarray,
    track_half_width: float,
    env_multiplier: np.ndarray,
    envelope: LateralEnvelope,
    a_long_max: float = 9.0,
    iterations: int = 3,
    smoothing_weight: float = 1.0,
    apex_weight: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimisation itérative type two-step line + speed.

    1) vitesse limite depuis enveloppe latérale (a_lat = c0 + c1*v^2, modulée par env_multiplier)
    2) simulation avant/arrière pour profil v(s)
    3) mise à jour de la ligne via convex optimisation d'offsets pondérée par 1/v(s)
    Répéter 2–3 fois.
    """
    xy = centerline_xy.copy()
    y_off = np.zeros(len(xy))
    for _ in range(max(1, iterations)):
        kap = curvature(xy)
        # v_lim à partir de a_lat_max = m*(c0 + c1*v^2) => v = sqrt( (a_lat_max / |kappa|) )
        a_lat_max = env_multiplier * envelope.a_lat(np.ones_like(kap))  # eval avec v^2 absorbé plus tard
        # pour l'inversion, on approxime en ramenant c0+c1*v^2 ~ c0 + c1*v_prev^2, ici v_prev inconnue: on linearise en utilisant v_lim précédente
        # fallback simple: utiliser la forme classique v = sqrt( (m*c0) / |kappa| ), puis augmenter avec c1 via facteur
        v_base = np.sqrt(np.maximum(1e-6, (env_multiplier * envelope.c0) / (np.abs(kap) + 1e-9)))
        # correction aéro: augmenter là où v_base est élevé
        v_lim = np.sqrt(np.maximum(1e-6, (env_multiplier * (envelope.c0 + envelope.c1 * (v_base ** 2))) / (np.abs(kap) + 1e-9)))

        sim = simulate_lap(xy, kap, v_lim, a_long_max=a_long_max)
        v = sim["v"]
        # étape ligne: offsets
        y_new = _offset_step(centerline_xy, v, kap, track_half_width, smoothing_weight, apex_weight)
        xy = offset_line(centerline_xy, y_new)
        y_off = y_new
    return xy, y_off


def _offset_step(centerline_xy: np.ndarray, v: np.ndarray, kappa_c: np.ndarray, track_half_width: float, smoothing_weight: float, apex_weight: float):
    n = len(centerline_xy)
    y = cp.Variable(n)
    D2 = _second_diff_matrix(n)
    w_time = 1.0 / np.maximum(1.0, v)
    apex_w = apex_weight * (np.abs(kappa_c) / (np.max(np.abs(kappa_c)) + 1e-9))
    obj = cp.sum(cp.multiply(w_time + apex_w, cp.abs(y))) + smoothing_weight * cp.sum_squares(D2 @ y)
    cons = [cp.abs(y) <= track_half_width]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, verbose=False)
    return np.asarray(y.value).reshape(-1) if y.value is not None else np.zeros(n)


def _second_diff_matrix(n: int):
    import scipy.sparse as sp
    e = np.ones(n)
    return sp.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))
