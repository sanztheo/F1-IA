from __future__ import annotations

import argparse
import numpy as np

from tracks.fetch import get_centerline
from evolution.population import run_evolution, EvoConfig
from viz.pygame_viewer import Viewer


def main():
    ap = argparse.ArgumentParser(description="Run population-based evolution with pygame viewer")
    ap.add_argument("--track", default="Circuit de Spa-Francorchamps")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--nctrl", type=int, default=25)
    ap.add_argument("--autoplay", action="store_true")
    args = ap.parse_args()

    center = get_centerline(args.track, args.year)
    if center.size == 0:
        raise SystemExit("No centerline available. Run scripts/fetch_maps.py or check track name.")

    run_id = f"{args.track.replace(' ','_')}_{args.year}"
    viewer = Viewer(title=f"Evolution – {args.track} {args.year}")

    playing = args.autoplay
    best_line = None
    best_time = None
    hist = []

    while True:
        running, _, _ = viewer.handle_events()
        if not running:
            break
        dt = viewer.tick(60)
        viewer.clear()

        # Draw track and current best
        viewer.draw_polyline(center, color=(100, 100, 120), width=2)
        if best_line is not None:
            viewer.draw_polyline(best_line, color=(233, 30, 99), width=3)

        viewer.draw_text("Controls: [SPACE]=toggle autoplay, [N]=next gens, [ESC]=quit", (10, 10))
        if best_time is not None:
            viewer.draw_text(f"Best time: {best_time:.3f}s | hist len: {len(hist)}", (10, 34))

        keys = __import__("pygame").pygame.key.get_pressed()
        if keys[32]:  # SPACE
            playing = True
        if keys[110]:  # N key
            playing = False
            best_line, best_time, hist = _step(center, args, run_id)
        if keys[27]:  # ESC
            break

        if playing:
            best_line, best_time, hist = _step(center, args, run_id)

        viewer.flip()


def _step(center: np.ndarray, args, run_id: str):
    res = run_evolution(center, EvoConfig(population_size=args.pop, generations=args.gens, n_ctrl=args.nctrl), run_id=run_id, resume=True)
    return res["best_line"], res["best_time"], res["history"]


if __name__ == "__main__":
    main()

