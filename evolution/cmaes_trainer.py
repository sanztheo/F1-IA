from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import pathlib as _pl
import numpy as np


@dataclass
class CMAESConfig:
    evaluations: int = 400
    n_ctrl: int = 25  # points de contrôle des offsets
    track_half_width: float = 6.0
    checkpoint_dir: str = "data/evolution"
    save_every: int = 1  # sauvegarde toutes les générations


def optimize_line_cmaes(centerline: np.ndarray, sim_fn, cfg: CMAESConfig, run_id: Optional[str] = None, resume: bool = True) -> Dict[str, Any]:
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

    def encode(line_xy: np.ndarray) -> np.ndarray:
        # approx offsets en mètres projetés sur les normales du centerline, puis re-échantillonnés sur s_ctrl
        offs = _project_offsets(centerline, line_xy)
        u = np.interp(s_ctrl, s, np.clip(offs / max(1e-6, cfg.track_half_width), -1.0, 1.0))
        return u

    # Préparer checkpoint
    run_dir = _pl.Path(cfg.checkpoint_dir)
    if run_id:
        run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialisation
    x0 = np.zeros(cfg.n_ctrl)
    sigma = 0.3

    # Reprise éventuelle
    state = None
    if resume:
        state = _load_checkpoint(run_dir)
        if state is not None and "u_best" in state:
            # warm start autour de la meilleure solution connue
            x0 = np.arctanh(np.clip(state["u_best"], -0.999, 0.999))
            sigma = float(state.get("sigma", 0.2))

    opt = CMA(mean=x0, sigma=sigma)

    best_t = np.inf if state is None else float(state.get("best_time", np.inf))
    best_line = None
    hist = [] if state is None else list(state.get("history", []))
    n_done = 0 if state is None else int(state.get("n_done", 0))
    target = cfg.evaluations

    gen = 0
    while n_done < target:
        solutions = []
        for _ in range(opt.population_size):
            z = opt.ask()
            off = decode(np.tanh(z))
            xy = _offset(centerline, off)
            t_s, _ = sim_fn(xy)
            solutions.append((z, t_s))
            hist.append(float(t_s))
            n_done += 1
            if t_s < best_t:
                best_t = float(t_s)
                best_line = xy
                u_best = encode(xy)
        opt.tell(solutions)
        gen += 1
        if gen % max(1, cfg.save_every) == 0:
            _save_checkpoint(run_dir, {
                "cfg": asdict(cfg),
                "n_done": n_done,
                "best_time": best_t,
                "history": hist,
                "sigma": sigma,
                "u_best": locals().get("u_best", np.zeros(cfg.n_ctrl)),
            })

    return {"best_line": best_line, "best_time": best_t, "history": np.asarray(hist), "n_done": n_done, "run_dir": str(run_dir)}


def _offset(centerline: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    # utilise les normales discrètes
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    n = np.stack([-dy, dx], axis=1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return centerline + n * offsets[:, None]


def _project_offsets(center: np.ndarray, line: np.ndarray) -> np.ndarray:
    dx = np.gradient(center[:, 0])
    dy = np.gradient(center[:, 1])
    n = np.stack([-dy, dx], axis=1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    d = line - center
    return np.sum(d * n, axis=1)


def _save_checkpoint(run_dir: _pl.Path, data: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "checkpoint.npz",
             n_done=np.array([data.get("n_done", 0)]),
             best_time=np.array([data.get("best_time", np.inf)]),
             history=np.asarray(data.get("history", []), dtype=float),
             sigma=np.array([data.get("sigma", 0.2)]),
             u_best=np.asarray(data.get("u_best", []), dtype=float))


def _load_checkpoint(run_dir: _pl.Path) -> Optional[Dict[str, Any]]:
    f = run_dir / "checkpoint.npz"
    if not f.exists():
        return None
    try:
        z = np.load(f, allow_pickle=True)
        return {
            "n_done": int(z.get("n_done", np.array([0]))[0]),
            "best_time": float(z.get("best_time", np.array([np.inf]))[0]),
            "history": z.get("history", np.array([], dtype=float)).tolist(),
            "sigma": float(z.get("sigma", np.array([0.2]))[0]),
            "u_best": z.get("u_best", np.array([], dtype=float)),
        }
    except Exception:
        return None
