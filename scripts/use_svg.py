from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import sys

# Ensure repo root on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracks.svg_loader import load_centerline_from_svg


def main():
    ap = argparse.ArgumentParser(description="Cache a centerline from an SVG and scale to given length (meters)")
    ap.add_argument("--svg", required=True, help="Path to SVG (e.g., svg/monaco.svg)")
    ap.add_argument("--length", type=float, default=3337.0, help="Target lap length in meters")
    ap.add_argument("--out", default="data/tracks/monaco.npy", help="Output .npy path")
    args = ap.parse_args()

    xy = load_centerline_from_svg(args.svg, path_id=None, samples=4000)
    if xy.size == 0:
        print("FAILED: could not load SVG")
        raise SystemExit(1)
    cur_len = float(np.linalg.norm(xy[1:] - xy[:-1], axis=1).sum())
    if cur_len > 0 and args.length > 0:
        xy = xy * (args.length / cur_len)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, xy)
    print(f"Cached centerline: {out} ({len(xy)} pts, length≈{args.length:.1f} m)")


if __name__ == "__main__":
    main()
