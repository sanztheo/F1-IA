from __future__ import annotations

import argparse
import pathlib as _pl
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

from .features.track_processing import posdata_to_centerline, curvature
from .models.vehicle_limits import VehicleLimitsModel
from .models.line_optimizer import optimize_line


def run_inference(config_path: str, session_id: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(_pl.Path(config_path).read_text())
    data_dir = _pl.Path(cfg.get("data_dir", "data"))
    half_width = float(cfg["optimization"]["track_half_width_m"]) \
        if cfg.get("optimization") else 6.0
    smooth_w = float(cfg["optimization"].get("smoothing_weight", 1.0))
    apex_w = float(cfg["optimization"].get("apex_weight", 2.0))
    a_lat_base = float(cfg["optimization"].get("a_lat_max", 12.0))

    # Charger positions FastF1
    sess_dir = data_dir / "raw" / "fastf1" / session_id
    pos_path = sess_dir / "positions.parquet"
    if not pos_path.exists():
        pos_path = sess_dir / "positions.csv"
    pos = pd.read_parquet(pos_path) if pos_path.suffix == ".parquet" else pd.read_csv(pos_path)

    center = posdata_to_centerline(pos)
    kappa = curvature(center)

    # Charger modèle de limites
    model_path = data_dir / "models" / "vehicle_limits.json"
    model = VehicleLimitsModel.load(model_path)

    # Construire DataFrame de features pour mu
    weather_path = sess_dir / "weather.parquet"
    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
    else:
        weather_csv = sess_dir / "weather.csv"
        weather = pd.read_csv(weather_csv) if weather_csv.exists() else pd.DataFrame()

    feats = pd.DataFrame({
        "Curvature": kappa,
        "AirTemp": weather.get("AirTemp", pd.Series(np.full(len(kappa), 20.0))),
        "TrackTemp": weather.get("TrackTemp", pd.Series(np.full(len(kappa), 30.0))),
        "Rainfall": weather.get("Rainfall", pd.Series(np.zeros(len(kappa)))),
        "Speed": pd.Series(np.full(len(kappa), 60.0)),  # placeholder si requis par extract
    })
    mu = model.predict_mu(feats)

    xy_line, offsets = optimize_line(
        center, track_half_width=half_width, mu_profile=mu,
        a_lat_max_base=a_lat_base, smoothing_weight=smooth_w, apex_weight=apex_w
    )

    return {
        "centerline": center,
        "racing_line": xy_line,
        "offsets": offsets,
        "mu": mu,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--session", required=True, help="Identifiant session ex: 2024_Monaco_R")
    args = ap.parse_args()
    out = run_inference(args.config, args.session)
    print("OK - Trajectoire calculée", {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in out.items()})

