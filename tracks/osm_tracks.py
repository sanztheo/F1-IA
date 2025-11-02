from __future__ import annotations

import pathlib as _pl
from typing import Optional, Tuple

import numpy as np
import shapely.geometry as sgeom
import shapely.ops as sops


def fetch_track_outline(place_name: str) -> sgeom.LineString:
    """Récupère une polyligne de piste depuis OpenStreetMap pour `place_name`.

    Cette fonction n'effectue aucun réseau ici; elle compte sur OSMnx si installé
    et connecté côté utilisateur. Si OSMnx n'est pas dispo, on renvoie une ligne vide.
    """
    try:
        import osmnx as ox  # type: ignore
    except Exception:
        return sgeom.LineString()

    # Tentatives robustes: leisure=racetrack puis highway=raceway
    tags_try = [
        {"leisure": "racetrack"},
        {"highway": "raceway"},
    ]
    gdf = None
    for tags in tags_try:
        try:
            gdf = ox.geometries_from_place(place_name, tags)
            if gdf is not None and not gdf.empty:
                break
        except Exception:
            continue
    if gdf is None or gdf.empty:
        return sgeom.LineString()

    # Extraire les polylignes/anneaux et choisir la plus grande par périmètre
    lines = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if isinstance(geom, (sgeom.LineString, sgeom.LinearRing)):
            lines.append(sgeom.LineString(geom))
        elif isinstance(geom, sgeom.Polygon):
            lines.append(sgeom.LineString(geom.exterior))
        elif isinstance(geom, sgeom.MultiPolygon):
            for p in geom.geoms:
                lines.append(sgeom.LineString(p.exterior))
        elif isinstance(geom, sgeom.MultiLineString):
            for l in geom.geoms:
                lines.append(sgeom.LineString(l))

    if not lines:
        return sgeom.LineString()

    lines.sort(key=lambda l: l.length, reverse=True)
    outline = lines[0]
    # Simplifier légèrement pour la performance
    outline = outline.simplify(1.0, preserve_topology=False)
    return outline


def resample_linestring(line: sgeom.LineString, n_points: int = 2000) -> np.ndarray:
    if line.is_empty:
        return np.zeros((0, 2))
    # Paramètre t in [0,1]
    d = np.linspace(0.0, 1.0, n_points)
    coords = [line.interpolate(frac, normalized=True).coords[0] for frac in d]
    return np.asarray(coords)


def outline_to_centerline(outline_xy: np.ndarray) -> np.ndarray:
    """Retourne un centerline lissé à partir d'un contour XY.

    Approche simple: lisser et rééchantillonner.
    """
    if len(outline_xy) == 0:
        return outline_xy
    from scipy.signal import savgol_filter

    win = 31 if len(outline_xy) >= 31 else (len(outline_xy) // 2 * 2 + 1)
    xs = savgol_filter(outline_xy[:, 0], win, 3, mode="wrap")
    ys = savgol_filter(outline_xy[:, 1], win, 3, mode="wrap")
    return np.column_stack([xs, ys])


