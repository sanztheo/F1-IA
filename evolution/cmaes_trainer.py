from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class CMAESConfig:
    evaluations: int = 400
    n_ctrl: int = 25  # points de contrôle des offsets
    track_half_width: float = 6.0


def optimize_line_cmaes(centerline: np.ndarray, sim_fn, cfg: CMAESConfig) -> Dict[str, Any]:
    """Optimise une ligne en offsets via CMA-ES.

    - centerline: [N,2]
    - sim_fn: callable(xy)-> temps (s) + infos
    - Renvoie: dict(best_line, best_time, history)
    """
    try:
        from cmaes import CMA
    except Exception:
        raise RuntimeError("Installez la dépendance 'cmaes' pour l'apprentissage évolutionnaire.")

    N = len(centerline)
    s = np.linspace(0, 1, N)
    s_ctrl = np.linspace(0, 1, cfg.n_ctrl)

    def decode(u: np.ndarray) -> np.ndarray:
        # u in [-1,1]^n_ctrl → offsets en mètres dans [-half, +half]
        off = np.interp(s, s_ctrl, u)
        return off * cfg.track_half_width

    # Variables initiales ~0 offsets
    x0 = np.zeros(cfg.n_ctrl)
    sigma = 0.3  # écart-type initial
    opt = CMA(mean=x0, sigma=sigma)

    best_t = np.inf
    best_line = None
    hist = []

    iters = 0
    while iters < cfg.evaluations:
        solutions = []
        for _ in range(opt.population_size):
            z = opt.ask()
            off = decode(np.tanh(z))  # borne douce
            xy = _offset(centerline, off)
            t_s, extra = sim_fn(xy)
            solutions.append((z, t_s))
            hist.append(float(t_s))
            iters += 1
            if t_s < best_t:
                best_t = float(t_s)
                best_line = xy
            if iters >= cfg.evaluations:
                break
        opt.tell(solutions)

    return {"best_line": best_line, "best_time": best_t, "history": np.asarray(hist)}


def _offset(centerline: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    # utilise les normales discrètes
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    n = np.stack([-dy, dx], axis=1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return centerline + n * offsets[:, None]

