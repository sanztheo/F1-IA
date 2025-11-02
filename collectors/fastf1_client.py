from __future__ import annotations

import os
import pathlib as _pl
from typing import Dict, Any, Optional, Tuple

import pandas as pd


def _ensure_cache_dir():
    # FastF1 utilise un cache local; on laisse l'utilisateur gérer l'init.
    try:
        import fastf1
        cache_dir = _pl.Path.home() / ".fastf1"
        fastf1.Cache.enable_cache(str(cache_dir))
    except Exception:
        pass


def load_session(year: int, event: str, session_code: str):
    import fastf1

    _ensure_cache_dir()
    sess = fastf1.get_session(year, event, session_code)
    sess.load()
    return sess


def export_session_core(sess, out_dir: str | os.PathLike, fmt: str = "parquet") -> Dict[str, Any]:
    """Exporte infos essentielles d'une session FastF1.

    - tours (laps)
    - télémétrie best lap par pilote
    - positions XY (pos)
    - météo (weather)
    """
    out = _pl.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    laps = sess.laps
    weather = getattr(sess, "weather_data", None)
    pos = sess.get_pos_data()  # positions XY vs temps

    def _save(df: pd.DataFrame, name: str):
        p = out / f"{name}.{ 'csv' if fmt=='csv' else 'parquet'}"
        if df is not None and not df.empty:
            if fmt == "csv":
                df.to_csv(p, index=False)
            else:
                df.to_parquet(p, index=False)
        return str(p)

    manifest: Dict[str, Any] = {"tables": []}
    manifest["tables"].append({"name": "laps", "path": _save(laps, "laps")})
    if weather is not None:
        manifest["tables"].append({"name": "weather", "path": _save(weather, "weather")})
    manifest["tables"].append({"name": "positions", "path": _save(pos, "positions")})

    # Télémétrie du meilleur tour par pilote
    try:
        best_per_driver = laps.groupby("Driver").apply(lambda df: df.pick_fastest())
        tele_chunks = []
        for _, lap in best_per_driver.items():
            try:
                tel = lap.get_car_data().add_distance()
                tel["Driver"] = lap["Driver"]
                tele_chunks.append(tel)
            except Exception:
                continue
        if tele_chunks:
            telemetry = pd.concat(tele_chunks, ignore_index=True)
            manifest["tables"].append({
                "name": "telemetry_best_per_driver",
                "path": _save(telemetry, "telemetry_best_per_driver")
            })
    except Exception:
        pass

    return manifest


def session_identifier(year: int, event: str, session_code: str) -> str:
    return f"{year}_{event}_{session_code}"

