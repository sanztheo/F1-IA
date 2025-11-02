from __future__ import annotations

import argparse
import json
import pathlib as _pl
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

from .features.track_processing import posdata_to_centerline, curvature
from .models.vehicle_limits import VehicleLimitsModel


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(_pl.Path(config_path).read_text())
    data_dir = _pl.Path(cfg.get("data_dir", "data"))

    # Ingestion simple: on parcourt les exports FastF1 et on fabrique un set d'exemples
    raw = data_dir / "raw" / "fastf1"
    rows = []
    for sess_dir in raw.glob("*_*_*"):
        pos_path = sess_dir / "positions.parquet"
        laps_path = sess_dir / "laps.parquet"
        weather_path = sess_dir / "weather.parquet"
        if not pos_path.exists():
            continue
        try:
            pos = pd.read_parquet(pos_path)
        except Exception:
            try:
                pos = pd.read_csv(sess_dir / "positions.csv")
            except Exception:
                continue

        center = posdata_to_centerline(pos)
        kappa = curvature(center)
        # Proxy vitesse: si télémétrie dispo, mieux; ici minimal: derive distance/temps si Date existe
        if "Date" in pos.columns:
            t = pd.to_datetime(pos["Date"]).astype("int64") / 1e9
            dt = np.gradient(t.to_numpy())
            from .features.track_processing import arc_length

            s = arc_length(center)
            v = np.gradient(s) / (np.maximum(1e-3, np.gradient(np.interp(s, s, t.to_numpy()[: len(s)]))))
        else:
            v = np.full(len(kappa), 60.0)  # fallback 60 m/s

        # Météo si dispo
        weather = None
        if weather_path.exists():
            try:
                weather = pd.read_parquet(weather_path)
            except Exception:
                try:
                    weather = pd.read_csv(sess_dir / "weather.csv")
                except Exception:
                    pass
        if weather is not None and not weather.empty:
            ta = weather.get("AirTemp", pd.Series(np.nan)).median()
            tt = weather.get("TrackTemp", pd.Series(np.nan)).median()
            rain = weather.get("Rainfall", pd.Series(0.0)).median()
        else:
            ta = 20.0
            tt = 30.0
            rain = 0.0

        df = pd.DataFrame({
            "Speed": v,
            "Curvature": kappa,
            "AirTemp": ta,
            "TrackTemp": tt,
            "Rainfall": rain,
        })
        rows.append(df)

    if not rows:
        print("Aucune donnée brute trouvée. Exécutez d'abord collectors.fetch_resources.")
        return

    all_df = pd.concat(rows, ignore_index=True)
    model = VehicleLimitsModel()
    model.fit(all_df)

    out_dir = data_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "vehicle_limits.json"
    model.save(model_path)
    print(f"Modèle sauvegardé: {model_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)

