from __future__ import annotations

"""
Distillation du teacher CEM → MLP.

1) Crée un env Monaco (SVG), redémarre sur positions aléatoires.
2) À chaque pas, planifie une action avec CEM (horizon court) et enregistre (obs→action).
3) Après M échantillons, entraîne un MLP par SGD (numpy) et sauvegarde comme best_policy.

Usage:
  python scripts/distill_teacher.py --samples 50000 --epochs 5
"""

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.svg_loader import load_centerline_from_svg
from rl.track_env import TrackEnv
from rl.planning.cem import plan_cem
from rl.models.mlp_policy import init_mlp
from rl.models.mlp_train import sgd_train
from rl.checkpoint import save_checkpoint, load_checkpoint


def load_centerline_monaco() -> np.ndarray:
    svg = ROOT / "svg/monaco.svg"
    xy = load_centerline_from_svg(svg, path_id=None, samples=4000)
    if xy.size == 0:
        raise SystemExit(f"SVG introuvable: {svg}")
    target = 3337.0
    cur = float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum())
    if cur > 0:
        xy = xy * (target / cur)
    return xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--halfwidth", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=16)
    args = ap.parse_args()

    center = load_centerline_monaco()
    env = TrackEnv(center, half_width=args.halfwidth, drs_zones=[(0.00, 0.06)])
    obs_dim = 3 + 5

    X = np.zeros((args.samples, obs_dim), dtype=np.float32)
    Y = np.zeros((args.samples, 3), dtype=np.float32)
    i = 0
    while i < args.samples:
        env.reset(0.0, random_start=True)
        steps = 0
        while steps < 200 and i < args.samples:
            obs = env.get_obs()
            a = plan_cem(env, horizon=args.horizon, iters=2, pop=96, elites=12)
            X[i] = obs
            Y[i] = a
            _, _, done, info = env.step(a)
            i += 1
            steps += 1
            if done:
                break

    sizes = (obs_dim, 64, 64, 3)
    params = init_mlp(sizes)
    params = sgd_train(params, sizes, X, Y, epochs=args.epochs, batch=512, lr=3e-3)

    # Écrit dans le checkpoint Monaco
    run_dir = ROOT / "data/evolution/Monaco_SVG"
    run_dir.mkdir(parents=True, exist_ok=True)
    pols, gen, best_t, best_pol = load_checkpoint(run_dir / "rl_checkpoint.npz", pop=400)
    save_checkpoint(run_dir / "rl_checkpoint.npz", gen, pols or [params], best_t, best_policy=params)
    print(f"Distillation terminée: {i} échantillons, epochs={args.epochs}. Best policy mise à jour.")


if __name__ == "__main__":
    main()

