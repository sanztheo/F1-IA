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
    if policy.get("__pursuit__"):
        # Heuristique: suivre la ligne (PD)
        lat_n = float(obs[0])  # [-1..1]
        head_n = float(obs[1])  # [-1..1]
        k_lat, k_head = 0.8, 1.2
        steer = np.clip(-(k_lat * lat_n + k_head * head_n), -1.0, 1.0)
        throttle = float(np.clip(0.9 - 0.4*abs(head_n) - 0.2*abs(lat_n), 0.0, 1.0))
        brake = 0.0
        a = np.array([steer, throttle, brake], dtype=np.float32)
        if policy.get("__noisy__"):
            a += np.array([np.random.normal(0, 0.05), np.random.normal(0, 0.05), 0.0], dtype=np.float32)
            a[0] = float(np.clip(a[0], -1.0, 1.0))
            a[1] = float(np.clip(a[1], 0.0, 1.0))
        return a
    from rl.policy import forward
    a = forward(policy, obs)
    if policy.get("__noisy__"):
        a += np.array([np.random.normal(0, 0.05), np.random.normal(0, 0.05), 0.0], dtype=np.float32)
        a[0] = float(np.clip(a[0], -1.0, 1.0))
        a[1] = float(np.clip(a[1], 0.0, 1.0))
    return a


def evaluate_once(center: np.ndarray, half_w: float, pol: Dict[str, np.ndarray], max_steps: int, drs: list[tuple[float,float]], random_start: bool) -> Tuple[float, float, float]:
    env = TrackEnv(center, half_width=half_w, drs_zones=drs)
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


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_ck(ck_dir: Path, gen: int, policies: List[Dict[str, np.ndarray]], best_time: float, best_policy: Dict[str, np.ndarray] | None = None) -> None:
    arr = np.array(policies, dtype=object)
    if best_policy is not None:
        np.savez(ck_dir / "rl_checkpoint.npz", gen=np.array([gen]), best_time=np.array([best_time]), policies=arr, best_policy=np.array([best_policy], dtype=object), allow_pickle=True)
    else:
        np.savez(ck_dir / "rl_checkpoint.npz", gen=np.array([gen]), best_time=np.array([best_time]), policies=arr, allow_pickle=True)


def _load_ck(ck_dir: Path, pop: int, obs_dim: int, rng: np.random.Generator) -> Tuple[List[Dict[str, np.ndarray]], int, float, Dict[str, np.ndarray] | None]:
    f = ck_dir / "rl_checkpoint.npz"
    if not f.exists():
        return [], 0, float('inf'), None
    try:
        z = np.load(f, allow_pickle=True)
        gen = int(z["gen"][0]) if "gen" in z else 0
        best_time = float(z.get("best_time", np.array([float('inf')]))[0])
        pols = z["policies"].tolist()
        best_pol = None
        if "best_policy" in z:
            bp = z["best_policy"].tolist()
            if isinstance(bp, list) and bp:
                best_pol = bp[0]
        if not isinstance(pols, list) or not pols:
            return [], gen, best_time, best_pol
        return pols[:pop], gen, best_time, best_pol
    except Exception:
        return [], 0, float('inf'), None


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
    obs_dim = 3 + 5
    run_id = f"{args.track.replace(' ', '_')}_{args.year}"
    ck_dir = _ensure_dir(f"data/evolution/{run_id}")
    policies, gen, best_overall, best_policy = _load_ck(ck_dir, args.pop, obs_dim, rng)
    if not policies:
        policies = [init_mlp((obs_dim, 32, 3), rng) for _ in range(args.pop)]
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
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(evaluate_once, center, args.halfwidth, p, args.horizon, drs, args.random_start) for p in policies]
            for i, f in enumerate(as_completed(futs), 1):
                fit, lap, prog = f.result()
                fits.append((fit, lap, prog))
                if i % max(1, len(policies)//10) == 0:
                    print(f"  {i}/{len(policies)} done", flush=True)
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
