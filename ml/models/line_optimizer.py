from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Tuple

from ..features.track_processing import curvature, offset_line


def optimize_line(
    centerline_xy: np.ndarray,
    track_half_width: float,
    mu_profile: np.ndarray,
    a_lat_max_base: float = 12.0,
    smoothing_weight: float = 1.0,
    apex_weight: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimise un offset latéral y(s) le long du centerline.

    Approche convexifiée en 2 étapes:
      1) On estime une vitesse cible v_i depuis mu_profile et la courbure du centerline.
      2) On minimise un surrogate de temps (poids ~ 1/v_i) + lissage (||D2 y||^2),
         sous contrainte |y_i| <= track_half_width.

    Retourne (xy_racing_line, offsets)
    """
    n = len(centerline_xy)
    kappa_c = curvature(centerline_xy)
    # vitesse limite latérale sur le centerline
    grav = 9.80665
    a_lat_max = np.maximum(3.0, a_lat_max_base * (mu_profile / np.maximum(mu_profile.mean(), 1e-3)))
    v_lim = np.sqrt(np.maximum(1e-3, (a_lat_max * grav) / (np.abs(kappa_c) + 1e-6)))
    w_time = 1.0 / np.maximum(1.0, v_lim)  # poids temporel (plus de poids où on roule lentement)

    # Variable d'offset
    y = cp.Variable(n)

    # Matrices de différences finies (2ème dérivée)
    D = _second_diff_matrix(n)

    # Objectif: somme w_time*|y| (incite à couper virages) + smoothing*||D y||^2
    # On ajoute un poids d'apex basé sur |kappa_c|
    apex_w = apex_weight * (np.abs(kappa_c) / (np.max(np.abs(kappa_c)) + 1e-9))
    obj = cp.sum(cp.multiply(w_time + apex_w, cp.abs(y))) + smoothing_weight * cp.sum_squares(D @ y)

    cons = [cp.abs(y) <= track_half_width]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, verbose=False)

    if y.value is None:
        y_opt = np.zeros(n)
    else:
        y_opt = np.asarray(y.value).reshape(-1)

    xy = offset_line(centerline_xy, y_opt)
    return xy, y_opt


def _second_diff_matrix(n: int) -> np.ndarray:
    import scipy.sparse as sp
    e = np.ones(n)
    D2 = sp.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n))
    return D2

