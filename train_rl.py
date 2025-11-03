from __future__ import annotations

import argparse
import numpy as np
import pygame
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
import os

from tracks.fetch import get_centerline
from rl.track_env import TrackEnv
from rl.policy import init_mlp, forward, mutate
from viz.pygame_viewer import Viewer

# ---------- helpers (checkpointing + action policy) ----------
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


def act(policy: Dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    # anomalies: full-send policy (always throttle, no brake)
    if policy.get("__fullsend__"):
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return forward(policy, obs)


def evaluate(center: np.ndarray, half_w: float, params: Dict[str, np.ndarray], max_steps: int = 2000) -> float:
    env = TrackEnv(center, half_width=half_w)
    obs = env.reset(0.0)
    total = 0.0
    lap_time = float('inf')
    for _ in range(max_steps):
        a = act(params, obs)
        obs, r, done, info = env.step(a)
        total += r
        if info.get("lap", 0) >= 1:
            lap_time = min(lap_time, info.get("t_lap", 0.0))
            break
        if done:
            break
    # Fitness: finish fastest else farthest
    if lap_time < float('inf'):
        return 10000.0 - lap_time
    return float(total)


def main():
    ap = argparse.ArgumentParser(description="Trackmania-like RL with population evolution (no Streamlit)")
    ap.add_argument("--track", default="Circuit de Spa-Francorchamps")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--svg", type=str, default=None, help="Optional SVG centerline path (e.g., svg/monaco.svg)")
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--halfwidth", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=1200)
    # no --gens, resume is automatic via checkpoints
    args = ap.parse_args()

    if args.svg:
        from tracks.svg_loader import load_centerline_from_svg
        center = load_centerline_from_svg(args.svg, path_id=None, samples=4000)
        if center.size:
            # scale to Monaco length if provided track is Monaco, else keep scale
            if 'monaco' in args.track.lower():
                cur = float(np.linalg.norm(center[1:] - center[:-1], axis=1).sum())
                if cur > 0:
                    center = center * (3337.0 / cur)
        else:
            center = get_centerline(args.track, args.year)
    else:
        center = get_centerline(args.track, args.year)
    env = TrackEnv(center, half_width=args.halfwidth)
    
    rng = np.random.default_rng(0)
    obs_dim = 3 + 5
    viewer = Viewer(title=f"RL Evolution – {args.track} {args.year}")
    run_id = f"{args.track.replace(' ', '_')}_{args.year}"
    ck_dir = _ensure_dir(f"data/evolution/{run_id}")

    gen = 0
    follow = True

    # Build initial population policies (or resume from checkpoint)
    fleet_policies, gen, best_overall_time = _load_ck(ck_dir, args.pop, obs_dim, rng)
    if not fleet_policies:
        fleet_policies = [init_mlp((obs_dim, 32, 3), rng) for _ in range(args.pop)]
        best_overall_time = float('inf')
        # Inject anomalies (10%)
        k = max(1, args.pop // 10)
        for i in range(k):
            fleet_policies[i] = {"__fullsend__": True}
    # Spawn agents for current generation
    def spawn_fleet(policies: List[Dict[str, np.ndarray]]):
        fl = []
        colors = [(50 + (i*8)%200, 200 - (i*7)%150, 120 + (i*5)%120) for i in range(len(policies))]
        for i, pol in enumerate(policies):
            e = TrackEnv(center, half_width=args.halfwidth)
            obs = e.reset(0.0)
            fl.append({"env": e, "obs": obs, "pol": pol, "alive": True, "fit": 0.0, "lap_time": float('inf'), "color": colors[i]})
        return fl
    fleet = spawn_fleet(fleet_policies)
    step_in_gen = 0
    # best_overall_time is set above (in resume or init)

    while True:
        running, _, _ = viewer.handle_events()
        if not running:
            break
        dt = viewer.tick(60)
        viewer.clear()
        # Track with width (edges)
        left, right = TrackEnv(center, half_width=args.halfwidth).edges()
        viewer.draw_polyline_fast(left, color=(60, 140, 60), width=2, min_px=3.0)
        viewer.draw_polyline_fast(right, color=(60, 140, 60), width=2, min_px=3.0)
        viewer.draw_polyline_fast(center, color=(230, 230, 230), width=1, min_px=3.0)

        # Step fleet
        alive_cnt = 0
        best_idx = None
        best_rank = (-1, -1.0)
        for i, ag in enumerate(fleet):
            if not ag["alive"]:
                continue
            a = act(ag["pol"], ag["obs"])
            ag["obs"], r, done, info = ag["env"].step(a)
            ag["fit"] += r
            # draw every car
            viewer.draw_car_rect(ag["env"].state["x"], ag["env"].state["y"], ag["env"].state["th"], length=5.6, width=2.0, color=ag["color"])
            if info.get("lap", 0) >= 1 and ag["lap_time"] == float('inf'):
                ag["lap_time"] = info.get("t_lap", 0.0)
            if done or info.get("lap", 0) >= 1:
                ag["alive"] = False
            else:
                alive_cnt += 1
                rank = (int(info.get("lap", 0)), float(ag["env"].state.get("progress", 0.0)))
                if rank > best_rank:
                    best_rank = rank
                    best_idx = i
        # Rays/camera on best alive
        if best_idx is not None:
            ag = fleet[best_idx]
            pos = (ag["env"].state["x"], ag["env"].state["y"]) 
            rays = ag["env"].ray_endpoints(num=7, fov_deg=120.0, max_r=args.halfwidth*2.0)
            viewer.draw_rays(pos, rays, color=(255,210,90))
            if follow:
                viewer.center_on(pos)
        step_in_gen += 1

        # Status text
        viewer.draw_text(f"gen={gen} alive={alive_cnt}/{args.pop} step={step_in_gen}/{args.horizon} fps~{int(1/max(1e-3,dt))}", (10, 10))
        if best_overall_time < float('inf'):
            viewer.draw_text(f"best lap: {best_overall_time:.2f}s", (10, 30))
        # Controls cheat‑sheet (toujours visible, coin haut gauche)
        viewer.draw_text("Controls:", (10, 52))
        viewer.draw_text("  Mouse wheel: zoom (vers le curseur)", (10, 70))
        viewer.draw_text("  Drag gauche: déplacer la vue", (10, 88))
        viewer.draw_text("  C / V: caméra suivre / libre", (10, 106))
        viewer.draw_text("  G / H: afficher / cacher ghosts", (10, 124))
        viewer.draw_text("  T / P: démarrer / pause entraînement", (10, 142))
        viewer.draw_text("  ESC: quitter", (10, 160))

        # No extra ghosts: toute la population est déjà visible

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            break
        # no training toggles in this synchronous mode
        if keys[pygame.K_c] or keys[pygame.K_f]:
            follow = True
        if keys[pygame.K_v]:
            follow = False
        # End-of-generation conditions: all cars done or horizon reached
        if alive_cnt == 0 or step_in_gen >= args.horizon:
            # Score
            scored = []
            for ag in fleet:
                fit = ag["fit"]
                if ag["lap_time"] < float('inf'):
                    fit += 10000.0 - ag["lap_time"]
                    if ag["lap_time"] < best_overall_time:
                        best_overall_time = ag["lap_time"]
                else:
                    fit += 0.001 * ag["env"].state.get("progress", 0.0)
                scored.append((fit, ag))
            scored.sort(key=lambda x: x[0], reverse=True)
            elites = [ag["pol"] for _, ag in scored[: max(5, args.pop//10)]]
            # New generation
            fleet_policies = []
            per = max(1, args.pop // len(elites))
            for e in elites:
                for _ in range(per):
                    if e.get("__fullsend__"):
                        fleet_policies.append({"__fullsend__": True})
                    else:
                        fleet_policies.append(mutate(e, args.sigma, rng))
            while len(fleet_policies) < args.pop:
                e0 = elites[0]
                if e0.get("__fullsend__"):
                    fleet_policies.append({"__fullsend__": True})
                else:
                    fleet_policies.append(mutate(e0, args.sigma, rng))
            # keep anomalies each gen (10%)
            k = max(1, args.pop // 10)
            for i in range(k):
                fleet_policies[i] = {"__fullsend__": True}
            fleet = spawn_fleet(fleet_policies)
            gen += 1
            step_in_gen = 0
            _save_ck(ck_dir, gen, fleet_policies, best_overall_time)

        viewer.flip()


if __name__ == "__main__":
    main()
