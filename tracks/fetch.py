from __future__ import annotations

import pathlib as _pl
from typing import Optional

import numpy as np

from .osm_tracks import fetch_track_outline, resample_linestring, outline_to_centerline
from ..replay.multicar_fastf1 import load_multicar


def slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def get_centerline(track_name: str, year: int = 2022, cache_dir: str | _pl.Path = "data/tracks") -> np.ndarray:
    """Retourne un centerline (N,2) pour un circuit, avec cache local.

    Ordre d'essai:
      1) charge depuis data/tracks/<slug>.npy si présent
      2) OSM (leisure=racetrack/highway=raceway)
      3) FastF1 (qualifs puis course) via meilleurs tours/positions
    """
    cache_dir = _pl.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(track_name)
    cache_file = cache_dir / f"{slug}.npy"
    if cache_file.exists():
        try:
            arr = np.load(cache_file)
            if arr.size:
                return arr
        except Exception:
            pass

    # OSM
    try:
        outline = fetch_track_outline(track_name)
        xy = resample_linestring(outline, 2000)
        if xy.size:
            cl = outline_to_centerline(xy)
            if cl is not None and len(cl):
                np.save(cache_file, cl)
                return cl
    except Exception:
        pass

    # FastF1 fallback (qualifs)
    event_name = _event_guess(track_name)
    for sess in ("Q", "R"):
        try:
            data = load_multicar(year, event_name, sess, n_drivers=5)
            cl = data.get("centerline")
            if cl is not None and len(cl):
                np.save(cache_file, cl)
                return cl
        except Exception:
            continue

    # Échec
    return np.zeros((0, 2))


def _event_guess(track_name: str) -> str:
    lower = track_name.lower()
    if "monaco" in lower:
        return "Monaco Grand Prix"
    if "monza" in lower or "autodromo nazionale" in lower:
        return "Italian Grand Prix"
    if "spa" in lower:
        return "Belgian Grand Prix"
    if "silverstone" in lower:
        return "British Grand Prix"
    return track_name

