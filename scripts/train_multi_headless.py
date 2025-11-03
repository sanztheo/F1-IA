from __future__ import annotations

"""
Headless multi‑pistes (généralisation). Par défaut, utilise Monaco (SVG) et
tout centerline présent dans data/tracks/*.npy. Boucle infinie de générations.

Usage simple:
  python scripts/train_multi_headless.py
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.svg_loader import load_centerline_from_svg
from tracks.fetch import get_centerline
from rl.track_env import TrackEnv
from rl.policy import init_mlp, mutate
from rl.actors import act
from rl.checkpoint import save_checkpoint, load_checkpoint


def _load_monaco() -> np.ndarray:
    svg = ROOT / "svg/monaco.svg"
    if not svg.exists():
        return np.zeros((0, 2))
    xy = load_centerline_from_svg(svg, None, 4000)
    if xy.size:
        cur = float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum())
        if cur > 0:
            xy = xy * (3337.0 / cur)
    return xy


def enumerate_tracks() -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    m = _load_monaco()
    if m.size:
        out.append(("Circuit de Monaco", m))
    # add cached *.npy tracks
    for p in (ROOT / "data/tracks").glob("*.npy"):
        try:
            arr = np.load(p)
            if arr.size:
                name = p.stem.replace("_", " ")
                if name.lower() != "monaco":
                    out.append((name, arr))
        except Exception:
            pass
    if not out:
        # fallback to a known fetch (Spa)
        spa = get_centerline("Circuit de Spa-Francorchamps", 2022)
        if spa.size:
            out.append(("Circuit de Spa-Francorchamps", spa))
    return out


def evaluate_once(center: np.ndarray, half_w: float, pol: Dict[str, np.ndarray], max_steps: int, drs: list[tuple[float,float]]) -> Tuple[float, float, float]:
    env = TrackEnv(center, half_width=half_w, drs_zones=drs, obs_mode="frenet", lookahead_k=10, lookahead_step=20)
    obs = env.reset(0.0, random_start=True)
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


def main():
    tracks = enumerate_tracks()
    if not tracks:
        raise SystemExit("Aucun circuit disponible (svg/monaco.svg manquant et aucun data/tracks/*.npy trouvé)")
    print(f"Multi‑pistes: {', '.join(n for n,_ in tracks)}")

    HALF_W = 10.0
    HORIZON = 2000
    POP = 400
    WORKERS = 8
    SIGMA = 0.10

    # probe obs_dim on first track
    probe_env = TrackEnv(tracks[0][1], half_width=HALF_W, obs_mode="frenet", lookahead_k=10, lookahead_step=20)
    probe_env.reset(0.0, random_start=True)
    obs_dim = int(probe_env.get_obs().size)

    run_dir = ROOT / "data/evolution/multi"
    ck_path = run_dir / "rl_checkpoint.npz"
    policies, gen, best_t, best_pol = load_checkpoint(ck_path, pop=POP)
    rng = np.random.default_rng(0)
    if not policies:
        policies = [init_mlp((obs_dim, 64, 64, 3), rng) for _ in range(POP)]
        k = max(1, POP // 10)
        for i in range(k):
            policies[i] = {"__fullsend__": True}
        for i in range(k, 2 * k):
            if i < len(policies):
                policies[i] = {"__pursuit__": True}
        for i in range(2 * k, 3 * k):
            if i < len(policies):
                policies[i] = {"__pursuit__": True, "__noisy__": True}
        best_t = float('inf')

    def drs_for(name: str) -> list[tuple[float, float]]:
        n = name.lower()
        if 'monaco' in n:
            return [(0.00, 0.06)]
        return []

    try:
        while True:
            print(f"Gen {gen} — eval {POP} agents sur {len(tracks)} pistes…", flush=True)
            # accumuler fitness sur toutes les pistes
            agg_fit = np.zeros((POP,), dtype=float)
            for tname, center in tracks:
                drs = drs_for(tname)
                # chunk per track
                chunk = max(1, POP // (WORKERS * 2))
                slabs: List[List[Dict]] = [policies[i:i+chunk] for i in range(0, POP, chunk)]
                with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                    futs = [ex.submit(lambda slab: [evaluate_once(center, HALF_W, p, HORIZON, drs) for p in slab], slab) for slab in slabs]
                    idx = 0
                    for f in as_completed(futs):
                        res = f.result()
                        for j, (fit, _, _) in enumerate(res):
                            agg_fit[idx + j] += fit
                        idx += len(res)
            # sélection
            order = np.argsort(agg_fit)[::-1]
            elites_idx = order[: max(5, POP // 10)]
            elites = [policies[i] for i in elites_idx]
            next_p: List[Dict] = []
            per = max(1, POP // len(elites))
            for e in elites:
                for _ in range(per):
                    if e.get("__fullsend__"):
                        next_p.append({"__fullsend__": True})
                    elif e.get("__pursuit__") and e.get("__noisy__"):
                        next_p.append({"__pursuit__": True, "__noisy__": True})
                    elif e.get("__pursuit__"):
                        next_p.append({"__pursuit__": True})
                    else:
                        next_p.append(mutate(e, SIGMA, rng))
            while len(next_p) < POP:
                e0 = elites[0]
                next_p.append({"__fullsend__": True} if e0.get("__fullsend__") else ({"__pursuit__": True} if e0.get("__pursuit__") else mutate(e0, SIGMA, rng)))
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
            save_checkpoint(ck_path, gen, policies, best_t, best_pol)
            print(f"Gen {gen} done (multi).", flush=True)
    except KeyboardInterrupt:
        print("\nArrêt demandé. Checkpoint multi sauvegardé.")


if __name__ == "__main__":
    main()

