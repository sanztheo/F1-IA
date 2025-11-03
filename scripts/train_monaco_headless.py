from __future__ import annotations

"""
Headless sans paramètres (Monaco SVG seulement).
Lance une évolution infinie (Ctrl‑C pour arrêter),
population fixe (400), workers=8, MLP « cerveau ».

Usage:
  python scripts/train_monaco_headless.py

Prérequis:
  - svg/monaco.svg présent
  - pip install -r requirements.txt
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Assurer l'import des modules du repo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.svg_loader import load_centerline_from_svg
from rl.track_env import TrackEnv
from rl.policy import init_mlp, mutate
from rl.actors import act
from rl.checkpoint import save_checkpoint, load_checkpoint


# Constantes simples
SVG_PATH = ROOT / "svg/monaco.svg"
TRACK_NAME = "Circuit de Monaco"
TARGET_LENGTH_M = 3337.0
POP = 400
HALF_WIDTH = 10.0  # largeur demi‑piste (m)
HORIZON = 3000
SIGMA = 0.10
WORKERS = 8
HIDDEN = (64, 64)  # MLP plus large pour de meilleures perfs




def evaluate_once(center: np.ndarray, half_w: float, pol: Dict[str, np.ndarray], max_steps: int, drs: list[tuple[float,float]], random_start: bool = True) -> Tuple[float, float, float]:
    env = TrackEnv(center, half_width=half_w, drs_zones=drs)
    obs = env.reset(0.0, random_start=random_start)
    total = 0.0
    best_lap = float('inf')
    max_progress = 0.0
    realistic_min_lap = 0.5 * (env.length / max(1e-6, env.params.max_speed))
    for _ in range(max_steps):
        a = act(pol, obs)
        obs, r, done, info = env.step(a)
        total += r
        max_progress = max(max_progress, float(info.get("progress_travel", 0.0)))
        if info.get("lap", 0) >= 1:
            lap_s = info.get("lap_time", info.get("t_lap", 0.0))
            if lap_s >= realistic_min_lap:
                best_lap = min(best_lap, lap_s)
                break
        if done:
            break
    if best_lap < float('inf'):
        fit = 10000.0 - best_lap
    else:
        fit = total + 50.0 * max_progress
    return fit, best_lap, max_progress


def _ensure_dir(p: Path | str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_ck(ck_dir: Path, gen: int, policies: List[Dict[str, np.ndarray]], best_time: float, best_policy: Dict[str, np.ndarray] | None = None) -> None:
    save_checkpoint(ck_dir / "rl_checkpoint.npz", gen, policies, best_time, best_policy)


def _load_ck(ck_dir: Path, pop: int) -> Tuple[List[Dict[str, np.ndarray]], int, float, Dict[str, np.ndarray] | None]:
    return load_checkpoint(ck_dir / "rl_checkpoint.npz", pop)


def load_centerline_monaco() -> np.ndarray:
    xy = load_centerline_from_svg(SVG_PATH, path_id=None, samples=4000)
    if xy.size == 0:
        raise SystemExit(f"SVG introuvable ou illisible: {SVG_PATH}")
    cur_len = float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum())
    if cur_len > 0:
        xy = xy * (TARGET_LENGTH_M / cur_len)
    return xy


def main():
    print("Headless Monaco – population=400, workers=8. Ctrl-C pour arrêter.")
    center = load_centerline_monaco()
    drs = [(0.00, 0.06)]  # zone DRS simple (ligne droite des stands)

    rng = np.random.default_rng(0)
    obs_dim = 3 + 5
    sizes = (obs_dim, *HIDDEN, 3)

    run_id = "Monaco_SVG"
    ck_dir = _ensure_dir(ROOT / "data/evolution" / run_id)

    policies, gen, best_overall, best_policy = _load_ck(ck_dir, POP)
    if not policies:
        policies = [init_mlp(sizes, rng) for _ in range(POP)]
        # anomalies + poursuite + bruit (10% chacune)
        k = max(1, POP // 10)
        for i in range(k):
            policies[i] = {"__fullsend__": True}
        for i in range(k, 2 * k):
            if i < len(policies):
                policies[i] = {"__pursuit__": True}
        for i in range(2 * k, 3 * k):
            if i < len(policies):
                policies[i] = {"__pursuit__": True, "__noisy__": True}
        best_overall = float('inf')
    else:
        # invalider un best irréaliste depuis ancien checkpoint
        min_lap_guard = 0.5 * (TARGET_LENGTH_M / 90.0)
        if best_overall < min_lap_guard:
            best_overall = float('inf')

    try:
        while True:
            print(f"Gen {gen} — evaluating {len(policies)} agents ...", flush=True)
            fits: List[Tuple[float, float, float]] = []
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(evaluate_once, center, HALF_WIDTH, p, HORIZON, drs, True) for p in policies]
                for i, f in enumerate(as_completed(futs), 1):
                    fit, lap, prog = f.result()
                    fits.append((fit, lap, prog))
                    if i % max(1, len(policies)//10) == 0:
                        print(f"  {i}/{len(policies)} done", flush=True)
            # élites et stats
            order = np.argsort([f for f, _, _ in fits])[::-1]
            elites_idx = order[: max(5, POP // 10)]
            elites = [policies[i] for i in elites_idx]
            if len(order):
                best_policy = policies[order[0]]
            laps = [fits[i][1] for i in elites_idx if fits[i][1] < float('inf')]
            progs = [fits[i][2] for i in elites_idx]
            if laps:
                best_overall = min(best_overall, float(min(laps)))
            best_lap_disp = f"{best_overall:.2f}s" if best_overall < float('inf') else "—"
            best_prog = float(max(progs)) if progs else float(max([p for _, _, p in fits]))
            mean_prog = float(np.mean([p for _, _, p in fits]))
            print(f"  Best lap: {best_lap_disp} | Best progress: {best_prog*100:.1f}% | Mean progress: {mean_prog*100:.1f}%", flush=True)

            # nouvelle population
            next_p: List[Dict[str, np.ndarray]] = []
            per = max(1, POP // len(elites))
            for e in elites:
                for _ in range(per):
                    if e.get("__fullsend__"):
                        next_p.append({"__fullsend__": True})
                    elif e.get("__pursuit__"):
                        next_p.append({"__pursuit__": True})
                    else:
                        next_p.append(mutate(e, SIGMA, rng))
            while len(next_p) < POP:
                e0 = elites[0]
                next_p.append({"__fullsend__": True} if e0.get("__fullsend__") else ({"__pursuit__": True} if e0.get("__pursuit__") else mutate(e0, SIGMA, rng)))
            # conserver 10% fullsend + 10% pursuit + 10% noisy pursuit en tête
            k = max(1, POP // 10)
            for i in range(k):
                next_p[i] = {"__fullsend__": True}
            for i in range(k, 2 * k):
                if i < len(next_p):
                    next_p[i] = {"__pursuit__": True}
            for i in range(2 * k, 3 * k):
                if i < len(next_p):
                    next_p[i] = {"__pursuit__": True, "__noisy__": True}

            policies = next_p[:POP]
            gen += 1
            _save_ck(ck_dir, gen, policies, best_overall, best_policy)
            print(f"Gen {gen} done. Best lap so far: {best_lap_disp} s", flush=True)
    except KeyboardInterrupt:
        print("\nArrêt demandé. Checkpoint sauvegardé.")


if __name__ == "__main__":
    main()
