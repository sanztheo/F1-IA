from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from tracks.fetch import get_centerline
from rl.track_env import TrackEnv
from rl.policy import init_mlp, mutate


def act(policy: Dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    if policy.get("__fullsend__"):
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    from rl.policy import forward
    return forward(policy, obs)


def evaluate_once(center: np.ndarray, half_w: float, pol: Dict[str, np.ndarray], max_steps: int) -> Tuple[float, float]:
    env = TrackEnv(center, half_width=half_w)
    obs = env.reset(0.0)
    total = 0.0
    best_lap = float('inf')
    for _ in range(max_steps):
        a = act(pol, obs)
        obs, r, done, info = env.step(a)
        total += r
        if info.get("lap", 0) >= 1:
            best_lap = min(best_lap, info.get("t_lap", 0.0))
            break
        if done:
            break
    if best_lap < float('inf'):
        fit = 10000.0 - best_lap
    else:
        fit = total
    return fit, best_lap


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_ck(ck_dir: Path, gen: int, policies: List[Dict[str, np.ndarray]], best_time: float) -> None:
    arr = np.array(policies, dtype=object)
    np.savez(ck_dir / "rl_checkpoint.npz", gen=np.array([gen]), best_time=np.array([best_time]), policies=arr, allow_pickle=True)


def _load_ck(ck_dir: Path, pop: int, obs_dim: int, rng: np.random.Generator) -> Tuple[List[Dict[str, np.ndarray]], int, float]:
    f = ck_dir / "rl_checkpoint.npz"
    if not f.exists():
        return [], 0, float('inf')
    try:
        z = np.load(f, allow_pickle=True)
        gen = int(z["gen"][0]) if "gen" in z else 0
        best_time = float(z.get("best_time", np.array([float('inf')]))[0])
        pols = z["policies"].tolist()
        if not isinstance(pols, list) or not pols:
            return [], gen, best_time
        return pols[:pop], gen, best_time
    except Exception:
        return [], 0, float('inf')


def main():
    ap = argparse.ArgumentParser(description="Headless population evolution (no rendering) with multiprocessing")
    ap.add_argument("--track", default="Circuit de Monaco")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--svg", type=str, default=None)
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--halfwidth", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--horizon", type=int, default=1200)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4)))
    args = ap.parse_args()

    # centerline
    if args.svg:
        from tracks.svg_loader import load_centerline_from_svg
        center = load_centerline_from_svg(args.svg, path_id=None, samples=4000)
        if center.size and 'monaco' in args.track.lower():
            cur = float(np.linalg.norm(center[1:] - center[:-1], axis=1).sum())
            if cur > 0:
                center = center * (3337.0 / cur)
    else:
        center = get_centerline(args.track, args.year)

    rng = np.random.default_rng(0)
    obs_dim = 3 + 5
    run_id = f"{args.track.replace(' ', '_')}_{args.year}"
    ck_dir = _ensure_dir(f"data/evolution/{run_id}")
    policies, gen, best_overall = _load_ck(ck_dir, args.pop, obs_dim, rng)
    if not policies:
        policies = [init_mlp((obs_dim, 32, 3), rng) for _ in range(args.pop)]
        # anomalies 10%
        for i in range(max(1, args.pop//10)):
            policies[i] = {"__fullsend__": True}
        best_overall = float('inf')

    for g in range(args.generations):
        print(f"Gen {gen} — evaluating {len(policies)} agents ...", flush=True)
        fits: List[Tuple[float, float]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(evaluate_once, center, args.halfwidth, p, args.horizon) for p in policies]
            for i, f in enumerate(as_completed(futs), 1):
                fit, lap = f.result()
                fits.append((fit, lap))
                if i % max(1, len(policies)//10) == 0:
                    print(f"  {i}/{len(policies)} done", flush=True)
        # select elites
        order = np.argsort([f for f, _ in fits])[::-1]
        elites_idx = order[: max(5, args.pop//10)]
        elites = [policies[i] for i in elites_idx]
        laps = [fits[i][1] for i in elites_idx if fits[i][1] < float('inf')]
        if laps:
            best_overall = min(best_overall, float(min(laps)))
        # create next population
        next_p: List[Dict[str, np.ndarray]] = []
        per = max(1, args.pop // len(elites))
        for e in elites:
            for _ in range(per):
                if e.get("__fullsend__"):
                    next_p.append({"__fullsend__": True})
                else:
                    next_p.append(mutate(e, args.sigma, rng))
        while len(next_p) < args.pop:
            e0 = elites[0]
            next_p.append({"__fullsend__": True} if e0.get("__fullsend__") else mutate(e0, args.sigma, rng))
        # anomalies 10%
        for i in range(max(1, args.pop//10)):
            next_p[i] = {"__fullsend__": True}
        policies = next_p[: args.pop]
        gen += 1
        _save_ck(ck_dir, gen, policies, best_overall)
        print(f"Gen {gen} done. Best lap so far: {best_overall if best_overall < float('inf') else '—'} s", flush=True)


if __name__ == "__main__":
    main()

