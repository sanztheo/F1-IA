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
from rl.actors import act
from rl.checkpoint import save_checkpoint, load_checkpoint




def evaluate_once(center: np.ndarray, half_w: float, pol: Dict[str, np.ndarray], max_steps: int, drs: list[tuple[float,float]], random_start: bool) -> Tuple[float, float, float]:
    env = TrackEnv(center, half_width=half_w, drs_zones=drs, obs_mode="frenet", lookahead_k=10, lookahead_step=20)
    obs = env.reset(0.0, random_start=random_start)
    total = 0.0
    best_lap = float('inf')
    max_progress = 0.0
    # borne réaliste minimale d'un tour (50% du temps théorique à v_max)
    realistic_min_lap = 0.5 * (env.length / max(1e-6, env.params.max_speed))
    for _ in range(max_steps):
        a = act(pol, obs)
        obs, r, done, info = env.step(a)
        total += r
        s_travel_frac = float(info.get("progress_travel", 0.0))
        max_progress = max(max_progress, s_travel_frac)
        if info.get("lap", 0) >= 1:
            lap_s = info.get("lap_time", info.get("t_lap", 0.0))
            # ignorer les tours irréalistes (glitch portique)
            if lap_s >= realistic_min_lap:
                best_lap = min(best_lap, lap_s)
                break
        if done:
            break
    if best_lap < float('inf'):
        fit = 10000.0 - best_lap
    else:
        # fitness sur la distance max atteinte en restant sur la piste
        fit = total + 50.0 * max_progress
    return fit, best_lap, max_progress


def evaluate_chunk(center: np.ndarray, half_w: float, policies: List[Dict[str, np.ndarray]], max_steps: int, drs: list[tuple[float,float]], random_start: bool) -> List[Tuple[float, float, float]]:
    # Réduit l'overhead: un seul process gère un lot complet séquentiellement
    out: List[Tuple[float, float, float]] = []
    for p in policies:
        out.append(evaluate_once(center, half_w, p, max_steps, drs, random_start))
    return out


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_ck(ck_dir: Path, gen: int, policies: List[Dict[str, np.ndarray]], best_time: float, best_policy: Dict[str, np.ndarray] | None = None) -> None:
    save_checkpoint(ck_dir / "rl_checkpoint.npz", gen, policies, best_time, best_policy)


def _load_ck(ck_dir: Path, pop: int, obs_dim: int, rng: np.random.Generator) -> Tuple[List[Dict[str, np.ndarray]], int, float, Dict[str, np.ndarray] | None]:
    return load_checkpoint(ck_dir / "rl_checkpoint.npz", pop)


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
    ap.add_argument("--random_start", action="store_true", help="Spawn agents at random track positions to diversify exploration")
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
    # probe env to infer observation dimension (frenet mode)
    _probe = TrackEnv(center, half_width=args.halfwidth, drs_zones=drs, obs_mode="frenet", lookahead_k=10, lookahead_step=20)
    _probe.reset(0.0, random_start=True)
    obs_dim = int(_probe.get_obs().size)
    run_id = f"{args.track.replace(' ', '_')}_{args.year}"
    ck_dir = _ensure_dir(f"data/evolution/{run_id}")
    policies, gen, best_overall, best_policy = _load_ck(ck_dir, args.pop, obs_dim, rng)
    if not policies:
        policies = [init_mlp((obs_dim, 64, 64, 3), rng) for _ in range(args.pop)]
        # anomalies + heuristiques + bruit d'action (10% chacune)
        k = max(1, args.pop//10)
        for i in range(k):
            policies[i] = {"__fullsend__": True}
        for i in range(k, 2*k):
            if i < len(policies):
                policies[i] = {"__pursuit__": True}
        for i in range(2*k, 3*k):
            if i < len(policies):
                policies[i] = {"__pursuit__": True, "__noisy__": True}
        best_overall = float('inf')
    else:
        # invalider un meilleur tour irréaliste provenant d'anciens checkpoints
        center_len = float(np.linalg.norm(center[1:] - center[:-1], axis=1).sum())
        min_lap_guard = 0.5 * (center_len / 90.0)  # 50% du temps à v_max~90 m/s
        if best_overall < min_lap_guard:
            best_overall = float('inf')

    # DRS zones défaut
    def default_drs(track_name: str) -> list[tuple[float,float]]:
        return [(0.00, 0.06)] if 'monaco' in track_name.lower() else []
    drs = default_drs(args.track)

    # log CSV
    hist_csv = ck_dir / "history.csv"

    for g in range(args.generations):
        print(f"Gen {gen} — evaluating {len(policies)} agents ...", flush=True)
        fits: List[Tuple[float, float, float]] = []
        # Evaluation par paquets pour réduire l'overhead IPC
        chunk = max(1, len(policies) // (args.workers * 2))
        slabs: List[List[Dict[str, np.ndarray]]] = [policies[i:i+chunk] for i in range(0, len(policies), chunk)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(evaluate_chunk, center, args.halfwidth, slab, args.horizon, drs, args.random_start) for slab in slabs]
            done_cnt = 0
            for f in as_completed(futs):
                res = f.result()
                fits.extend(res)
                done_cnt += len(res)
                if done_cnt % max(1, len(policies)//10) == 0:
                    print(f"  {done_cnt}/{len(policies)} done", flush=True)
        # select elites
        order = np.argsort([f for f,_,_ in fits])[::-1]
        elites_idx = order[: max(5, args.pop//10)]
        elites = [policies[i] for i in elites_idx]
        # best policy courante
        if len(order):
            best_policy = policies[order[0]]
        laps = [fits[i][1] for i in elites_idx if fits[i][1] < float('inf')]
        progs = [fits[i][2] for i in elites_idx]
        if laps:
            best_overall = min(best_overall, float(min(laps)))
        # stats log
        best_lap_disp = f"{best_overall:.2f}s" if best_overall < float('inf') else "—"
        best_prog = float(max(progs)) if progs else float(max([p for _,_,p in fits]))
        mean_prog = float(np.mean([p for _,_,p in fits]))
        print(f"  Best lap: {best_lap_disp} | Best progress: {best_prog*100:.1f}% | Mean progress: {mean_prog*100:.1f}%", flush=True)
        # CSV append
        try:
            import csv
            new = not hist_csv.exists()
            with open(hist_csv, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["gen","best_lap","best_progress","mean_progress"]) 
                w.writerow([gen, best_overall if best_overall < float('inf') else '', f"{best_prog:.4f}", f"{mean_prog:.4f}"])
        except Exception:
            pass
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
        # anomalies + poursuite conservées
        for i in range(max(1, args.pop//10)):
            next_p[i] = {"__fullsend__": True}
        offset = max(1, args.pop//10)
        for j in range(max(1, args.pop//10)):
            idx = offset + j
            if idx < len(next_p):
                next_p[idx] = {"__pursuit__": True}
        # ajouter 10% d'agents bruités pour exploration continue
        base = 2*max(1, args.pop//10)
        for j in range(max(1, args.pop//10)):
            idx = base + j
            if idx < len(next_p):
                next_p[idx] = {"__pursuit__": True, "__noisy__": True}
        policies = next_p[: args.pop]
        gen += 1
        _save_ck(ck_dir, gen, policies, best_overall, best_policy)
        print(f"Gen {gen} done. Best lap so far: {best_overall if best_overall < float('inf') else '—'} s", flush=True)


if __name__ == "__main__":
    main()
