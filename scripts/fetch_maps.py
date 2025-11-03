from __future__ import annotations

import argparse
from tracks.fetch import get_centerline


def main():
    ap = argparse.ArgumentParser(description="Fetch and cache track centerlines")
    ap.add_argument("--track", action="append", help="Track name (can be used multiple times)")
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()

    tracks = args.track or [
        "Circuit de Monaco",
        "Autodromo Nazionale Monza",
        "Circuit de Spa-Francorchamps",
    ]
    for t in tracks:
        cl = get_centerline(t, year=args.year)
        print(f"{t}: {'OK' if cl.size else 'FAILED'} ({len(cl)} points)")


if __name__ == "__main__":
    main()

