from __future__ import annotations

import pathlib as _pl
from typing import Optional
import numpy as np


def load_centerline_from_svg(svg_path: str | _pl.Path, path_id: Optional[str] = None, samples: int = 4000) -> np.ndarray:
    """Charge un SVG et extrait un path comme centerline (XY en mètres arbitraires).

    - svg_path: chemin vers le fichier SVG
    - path_id: id du <path> à utiliser (si None: premier path)
    - samples: nombre de points échantillonnés
    Retourne un np.ndarray (N,2) ou un tableau vide si échec.
    """
    try:
        from svgpathtools import svg2paths2, Path
    except Exception:
        return np.zeros((0, 2))

    try:
        paths, attrs, svg_attr = svg2paths2(str(svg_path))
    except Exception:
        return np.zeros((0, 2))

    target = None
    if path_id:
        for p, a in zip(paths, attrs):
            if a.get('id') == path_id:
                target = p
                break
    if target is None:
        target = paths[0] if paths else None
    if target is None:
        return np.zeros((0, 2))

    # Convertir en polyline uniforme
    L = target.length()
    if L <= 0:
        return np.zeros((0, 2))
    ts = np.linspace(0, 1, samples)
    pts = [target.point(t) for t in ts]
    xy = np.column_stack([np.real(pts), np.imag(pts)])
    return xy

