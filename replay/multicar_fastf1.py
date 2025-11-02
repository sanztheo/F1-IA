from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def load_multicar(year: int, event_name: str, session_code: str = "R", n_drivers: int = 10) -> Dict[str, Any]:
    """Charge une session FastF1 et renvoie une grille temps + positions XY pour `n_drivers`.

    Sortie:
      {
        't': np.ndarray [T], temps (s) normalisés à 0 pour le début de la séquence,
        'cars': List[Dict{name, color, x: [T], y: [T]}],
        'centerline': np.ndarray [N,2] (si dispo via positions agrégées)
      }
    """
    import fastf1

    sess = fastf1.get_session(year, event_name, session_code)
    sess.load()

    # Sélectionner les pilotes: top par meilleurs tours présents
    laps = sess.laps
    drivers = laps["Driver"].dropna().unique().tolist()
    drivers = drivers[: n_drivers]

    # Positions XY temporelles
    pos_df = None
    if hasattr(sess, "get_pos_data"):
        try:
            pos_df = sess.get_pos_data()
        except Exception:
            pos_df = None
    if pos_df is None and hasattr(sess, "get_position_data"):
        try:
            pos_df = sess.get_position_data()
        except Exception:
            pos_df = None
    if pos_df is None or pos_df.empty:
        raise RuntimeError("Position data not available for session")

    # Restreindre aux pilotes choisis et trier par Date
    pos_df = pos_df[pos_df["Driver"].isin(drivers)].copy()
    if "Date" in pos_df.columns:
        pos_df.sort_values("Date", inplace=True)

    # Normaliser le temps relatif et créer une grille uniforme
    t_rel = (pd.to_datetime(pos_df["Date"]).astype("int64") / 1e9).to_numpy() if "Date" in pos_df.columns else np.arange(len(pos_df))
    t_rel = t_rel - t_rel.min()
    # Grille à ~20 Hz
    T_max = t_rel.max()
    T = int(max(200, min(4000, T_max * 20)))
    t_grid = np.linspace(0, T_max, T)

    # Couleurs simples
    palette = ["#F44336", "#3F51B5", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#8BC34A", "#FF5722", "#795548", "#607D8B"]

    cars: List[Dict[str, Any]] = []
    for i, drv in enumerate(drivers):
        dfi = pos_df[pos_df["Driver"] == drv]
        if dfi.empty:
            continue
        t = (pd.to_datetime(dfi["Date"]).astype("int64") / 1e9).to_numpy() - t_rel.min() if "Date" in dfi.columns else np.arange(len(dfi))
        x = dfi["X"].to_numpy()
        y = dfi["Y"].to_numpy()
        # Interpolation sur la grille
        xi = np.interp(t_grid, t, x)
        yi = np.interp(t_grid, t, y)
        cars.append({"name": str(drv), "color": palette[i % len(palette)], "x": xi, "y": yi})

    # Centerline approximatif: médiane par angle (reprend util existant)
    try:
        from ml.features.track_processing import posdata_to_centerline
        centerline = posdata_to_centerline(pos_df)
    except Exception:
        centerline = None

    return {"t": t_grid, "cars": cars, "centerline": centerline}

