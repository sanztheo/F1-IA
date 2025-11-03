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


def _save_ck(ck_dir: Path, gen: int, policies: List[Dict[str, np.ndarray]], best_time: float, best_policy: Dict[str, np.ndarray] | None = None) -> None:
    arr = np.array(policies, dtype=object)
    if best_policy is not None:
        np.savez(ck_dir / "rl_checkpoint.npz", gen=np.array([gen]), best_time=np.array([best_time]), policies=arr, best_policy=np.array([best_policy], dtype=object), allow_pickle=True)
    else:
        np.savez(ck_dir / "rl_checkpoint.npz", gen=np.array([gen]), best_time=np.array([best_time]), policies=arr, allow_pickle=True)


def _load_ck(ck_dir: Path, pop: int, obs_dim: np.random.Generator | None = None) -> Tuple[List[Dict[str, np.ndarray]], int, float, Dict[str, np.ndarray] | None]:
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
    ap = argparse.ArgumentParser(description="Viewer: rejoue la meilleure policy depuis le checkpoint (par défaut)")
    ap.add_argument("--track", default="Circuit de Spa-Francorchamps")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--svg", type=str, default=None, help="Optional SVG centerline path (e.g., svg/monaco.svg)")
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--halfwidth", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=1200)
    ap.add_argument("--evolve", action="store_true", help="(Optionnel) relancer l'évolution visuelle (ancien comportement)")
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
    # Simple DRS zone par défaut: ligne droite des stands (0.00–0.06)
    def default_drs(track_name: str, length: float) -> list[tuple[float,float]]:
        name = track_name.lower()
        if 'monaco' in name:
            return [(0.00, 0.06)]
        return []

    drs = default_drs(args.track, float(np.linalg.norm(center[1:] - center[:-1], axis=1).sum()))
    # Pré‑calcul des bords de piste pour le rendu (évite shapely à chaque frame)
    _edges_env = TrackEnv(center, half_width=args.halfwidth, drs_zones=drs)
    pre_left, pre_right = _edges_env.edges()
    
    rng = np.random.default_rng(0)
    obs_dim = 3 + 5
    viewer = Viewer(title=f"RL Evolution – {args.track} {args.year}")
    run_id = f"{args.track.replace(' ', '_')}_{args.year}"
    ck_dir = _ensure_dir(f"data/evolution/{run_id}")

    gen = 0
    follow = True

    # Mode VIEW (par défaut): charger la meilleure policy depuis checkpoint
    fleet_policies, gen, best_overall_time, best_policy = _load_ck(ck_dir, args.pop)
    if not args.evolve:
        viewer = Viewer(title=f"Replay – {args.track} {args.year}")
        # choisir la best_policy si dispo, sinon évaluer vite fait
        if best_policy is None:
            if fleet_policies:
                scores = [(evaluate(center, args.halfwidth, p, max_steps=args.horizon), p) for p in fleet_policies]
                scores.sort(key=lambda x: x[0], reverse=True)
                best_policy = scores[0][1]
        # si toujours rien, init une policy neutre
        if best_policy is None:
            best_policy = init_mlp((obs_dim, 64, 64, 3), rng)

        # Simulation d'une seule voiture (replay best)
        env = TrackEnv(center, half_width=args.halfwidth, drs_zones=drs)
        obs = env.reset(0.0)
        t = 0
        while True:
            running, _, _ = viewer.handle_events()
            if not running:
                break
            dt = viewer.tick(60)
            viewer.clear()
            viewer.draw_polyline_fast(pre_left, color=(60, 140, 60), width=2, min_px=3.0)
            viewer.draw_polyline_fast(pre_right, color=(60, 140, 60), width=2, min_px=3.0)
            viewer.draw_polyline_fast(center, color=(230, 230, 230), width=1, min_px=3.0)
            a = act(best_policy, obs)
            obs, r, done, info = env.step(a)
            viewer.draw_car_rect(env.state["x"], env.state["y"], env.state["th"], length=5.6, width=2.0, color=(80,170,250))
            rays = env.ray_endpoints(num=7, fov_deg=120.0, max_r=args.halfwidth*2.0)
            viewer.draw_rays((env.state["x"], env.state["y"]), rays, color=(255,210,90))
            viewer.center_on((env.state["x"], env.state["y"]))
            viewer.draw_text(f"Replay best | lap={info.get('lap',0)} t_lap={info.get('t_lap',0.0):.2f}s | fps~{int(1/max(1e-3,dt))}", (10,10))
            viewer.draw_text("ESC pour quitter", (10,30))
            if done and info.get("lap",0) < 1:
                # si sortie de piste: réinitialiser pour continuer la démo
                obs = env.reset(0.0, random_start=True)
            viewer.flip()
        return
    if not fleet_policies:
        fleet_policies = [init_mlp((obs_dim, 32, 3), rng) for _ in range(args.pop)]
        best_overall_time = float('inf')
        # Inject anomalies (10%)
        k = max(1, args.pop // 10)
        for i in range(k):
            fleet_policies[i] = {"__fullsend__": True}
    # Mode EVOLVE (optionnel avec --evolve): comportement précédent
    # Spawn agents for current generation
    def spawn_fleet(policies: List[Dict[str, np.ndarray]]):
        fl = []
        colors = [(50 + (i*8)%200, 200 - (i*7)%150, 120 + (i*5)%120) for i in range(len(policies))]
        for i, pol in enumerate(policies):
            e = TrackEnv(center, half_width=args.halfwidth, drs_zones=drs)
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
        viewer.draw_polyline_fast(pre_left, color=(60, 140, 60), width=2, min_px=3.0)
        viewer.draw_polyline_fast(pre_right, color=(60, 140, 60), width=2, min_px=3.0)
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
            # HUD barres throttle/brake
            thr = float(ag["env"].state.get("throttle", 0.0))
            brk = float(ag["env"].state.get("brake", 0.0))
            viewer.draw_text(f"Throttle: {thr:.2f}", (10, 180))
            viewer.draw_text(f"Brake:    {brk:.2f}", (10, 198))
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
