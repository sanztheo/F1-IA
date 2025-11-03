from __future__ import annotations

from typing import Dict, Any, List
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
        # Fallback: reconstruire un tour par pilote via son meilleur tour (télémétrie)
        return _build_from_best_laps(sess, drivers)

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


def _build_from_best_laps(sess, drivers: List[str]) -> Dict[str, Any]:
    """Fallback quand get_pos_data() est indisponible: utilise le meilleur tour de chaque pilote.

    - Extrait tel = lap.get_telemetry(); s'il contient X,Y on les prend;
    - sinon, si on peut construire un centerline depuis d'autres pilotes, on projette la distance sur ce centerline.
    """
    laps = sess.laps
    cars: List[Dict[str, Any]] = []
    center_candidates = []
    # Grille temporelle globale (sera définie après collecte des tours)
    T_max = 0.0
    series: List[Dict[str, Any]] = []
    for drv in drivers:
        try:
            lp = laps.pick_driver(drv).pick_fastest()
        except Exception:
            continue
        try:
            tel = lp.get_telemetry()
        except Exception:
            continue
        # temps relatif
        if "Date" in tel.columns:
            t = (pd.to_datetime(tel["Date"]).astype("int64") / 1e9)
            t = t - t.iloc[0]
        elif "Time" in tel.columns:
            t = tel["Time"].astype(float) - float(tel["Time"].iloc[0])
        else:
            t = pd.Series(range(len(tel)), dtype=float)
        x = tel["X"].to_numpy() if "X" in tel.columns else None
        y = tel["Y"].to_numpy() if "Y" in tel.columns else None
        if x is not None and y is not None:
            center_candidates.append(pd.DataFrame({"X": x, "Y": y}))
        series.append({"name": drv, "t": t.to_numpy(), "x": x, "y": y})
        T_max = max(T_max, float(t.iloc[-1]))

    if not series:
        raise RuntimeError("No telemetry available to build fallback replay")

    # Construire centerline si possible
    centerline = None
    if center_candidates:
        try:
            from ml.features.track_processing import posdata_to_centerline
            dfc = pd.concat(center_candidates, ignore_index=True)
            centerline = posdata_to_centerline(dfc)
        except Exception:
            centerline = None

    # Grille temporelle commune ~20 Hz
    T = int(max(200, min(4000, T_max * 20)))
    t_grid = np.linspace(0, T_max, T)

    # Si XY manquant pour un pilote, tenter projection sur le centerline via Distance si dispo
    for i, s in enumerate(series):
        t = s["t"]
        x = s["x"]
        y = s["y"]
        if x is None or y is None:
            # distances si dispo
            try:
                lp = laps.pick_driver(s["name"]).pick_fastest()
                car = lp.get_car_data().add_distance()
                dist = car["Distance"].to_numpy()
                dmax = float(dist[-1])
                if centerline is not None and len(centerline):
                    # approx: mapper dist [0,dmax] sur abscisse curviligne du centerline
                    from ml.features.track_processing import arc_length
                    s_center = arc_length(centerline)
                    x = np.interp(dist, np.linspace(0, dmax, len(s_center)), centerline[:, 0])
                    y = np.interp(dist, np.linspace(0, dmax, len(s_center)), centerline[:, 1])
                    # temps approx proportionnel à dist
                    t = np.linspace(0, T_max, len(dist))
                else:
                    continue
            except Exception:
                continue
        xi = np.interp(t_grid, t, x)
        yi = np.interp(t_grid, t, y)
        color = ["#F44336", "#3F51B5", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#8BC34A", "#FF5722", "#795548", "#607D8B"][i % 10]
        cars.append({"name": s["name"], "color": color, "x": xi, "y": yi})

    return {"t": t_grid, "cars": cars, "centerline": centerline}
