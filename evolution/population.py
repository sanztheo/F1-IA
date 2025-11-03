from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import pathlib as _pl
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ml.features.track_processing import normals
from simulation.lap_simulator import simulate_lap


@dataclass
class EvoConfig:
    population_size: int = 400
    generations: int = 1
    n_ctrl: int = 25
    track_half_width: float = 6.0
    a_long_max: float = 9.0
    seed: int = 42
    checkpoint_dir: str = "data/evolution"


def run_evolution(centerline: np.ndarray, cfg: EvoConfig, run_id: str, resume: bool = True) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    N = len(centerline)
    s = np.linspace(0, 1, N)
    s_ctrl = np.linspace(0, 1, cfg.n_ctrl)

    run_dir = _pl.Path(cfg.checkpoint_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(run_dir) if resume else None

    if state is None:
        # init: population aléatoire uniforme [-0.5,0.5]
        pop_u = rng.uniform(-0.5, 0.5, size=(cfg.population_size, cfg.n_ctrl))
        gen0 = 0
        best = {"time": np.inf, "u": pop_u[0]}
        history: List[float] = []
    else:
        pop_u = state["pop"]
        gen0 = int(state["gen"])
        best = {"time": float(state["best_time"]), "u": state["best_u"]}
        history = list(state["history"])  # type: ignore

    for g in range(gen0, gen0 + cfg.generations):
        # Décoder et évaluer
        results = _evaluate_population(centerline, pop_u, s, s_ctrl, cfg)
        times = np.array([r["time"] for r in results])
        idx = np.argsort(times)
        if times[idx[0]] < best["time"]:
            best = {"time": float(times[idx[0]]), "u": pop_u[idx[0]]}
        history.append(float(times[idx[0]]))

        # Sélection + mutation (ES (mu, lambda)): garder top mu=20%, re‑échantillonner
        mu = max(2, int(0.2 * cfg.population_size))
        elites = pop_u[idx[:mu]]
        # Mutations gaussiennes décroissantes autour des élites
        sigma = max(0.02, 0.15 * (0.99 ** (g - gen0)))
        new_pop = []
        while len(new_pop) < cfg.population_size:
            e = elites[rng.integers(0, mu)]
            child = e + rng.normal(0.0, sigma, size=e.shape)
            child = np.clip(child, -1.0, 1.0)
            new_pop.append(child)
        pop_u = np.vstack(new_pop)

        # checkpoint
        _save_state(run_dir, gen=g + 1, pop=pop_u, best_time=best["time"], best_u=best["u"], history=np.asarray(history))

    # Préparer sortie
    best_xy = _decode_to_xy(centerline, best["u"], s, s_ctrl, cfg)
    return {"best_line": best_xy, "best_time": best["time"], "history": np.asarray(history), "gen": gen0 + cfg.generations, "run_dir": str(run_dir)}


def _evaluate_population(centerline: np.ndarray, pop_u: np.ndarray, s: np.ndarray, s_ctrl: np.ndarray, cfg: EvoConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def eval_one(u_row: np.ndarray) -> Dict[str, Any]:
        xy = _decode_to_xy(centerline, u_row, s, s_ctrl, cfg)
        from ml.features.track_processing import curvature
        k = curvature(xy)
        v_lim = np.sqrt(np.maximum(1e-3, (12.0 * 9.80665) / (np.abs(k) + 1e-6)))
        sim = simulate_lap(xy, k, v_lim, a_long_max=cfg.a_long_max)
        return {"time": sim["time_s"]}

    with ThreadPoolExecutor(max_workers=min(8, cfg.population_size)) as ex:
        futs = [ex.submit(eval_one, u) for u in pop_u]
        for f in as_completed(futs):
            out.append(f.result())
    return out


def _decode_to_xy(centerline: np.ndarray, u: np.ndarray, s: np.ndarray, s_ctrl: np.ndarray, cfg: EvoConfig) -> np.ndarray:
    # offsets en m
    off = np.interp(s, s_ctrl, u) * cfg.track_half_width
    # normales
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    n = np.stack([-dy, dx], axis=1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return centerline + n * off[:, None]


def _save_state(run_dir: _pl.Path, **data):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "pop_checkpoint.npz",
             gen=np.array([data.get("gen", 0)]),
             pop=data.get("pop"),
             best_time=np.array([data.get("best_time", np.inf)]),
             best_u=data.get("best_u"),
             history=np.asarray(data.get("history", []), dtype=float))


def _load_state(run_dir: _pl.Path) -> Optional[Dict[str, Any]]:
    f = run_dir / "pop_checkpoint.npz"
    if not f.exists():
        return None
    try:
        z = np.load(f, allow_pickle=True)
        return {
            "gen": int(z.get("gen", np.array([0]))[0]),
            "pop": z.get("pop"),
            "best_time": float(z.get("best_time", np.array([np.inf]))[0]),
            "best_u": z.get("best_u"),
            "history": z.get("history", np.array([], dtype=float)),
        }
    except Exception:
        return None

