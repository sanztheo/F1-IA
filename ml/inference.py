from __future__ import annotations

import argparse
import logging
import pathlib as _pl
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml

from .features.track_processing import posdata_to_centerline, curvature
from .models.vehicle_limits import VehicleLimitsModel
from .models.line_optimizer import optimize_line
from .models.phys_params import LateralEnvelope
from .models.vlim_regressor import VLimModel
from .models.line_optimizer_iter import optimize_line_iterative


logger = logging.getLogger("f1ia.ui")


def run_inference(config_path: str, session_id: str) -> Dict[str, Any]:
    logger.info("run_inference start: config=%s session=%s", config_path, session_id)
    cfg = yaml.safe_load(_pl.Path(config_path).read_text())
    data_dir = _pl.Path(cfg.get("data_dir", "data"))
    half_width = float(cfg["optimization"]["track_half_width_m"]) \
        if cfg.get("optimization") else 6.0
    smooth_w = float(cfg["optimization"].get("smoothing_weight", 1.0))
    apex_w = float(cfg["optimization"].get("apex_weight", 2.0))
    a_lat_base = float(cfg["optimization"].get("a_lat_max", 12.0))
    use_adv = bool(cfg.get("advanced", {}).get("use_advanced", True))
    iters = int(cfg.get("advanced", {}).get("iterations", 3))

    # Charger positions FastF1
    sess_dir = data_dir / "raw" / "fastf1" / session_id
    logger.info("session dir: %s", sess_dir)
    pos_path = sess_dir / "positions.parquet"
    if not pos_path.exists():
        pos_path = sess_dir / "positions.csv"
    if not pos_path.exists():
        logger.error("positions file missing for session: %s", sess_dir)
        raise FileNotFoundError(f"positions.* introuvable dans {sess_dir}")
    logger.info("positions file: %s", pos_path)
    try:
        pos = pd.read_parquet(pos_path) if pos_path.suffix == ".parquet" else pd.read_csv(pos_path)
    except Exception as e:
        logger.exception("failed to read positions: %s", e)
        raise

    center = posdata_to_centerline(pos)
    kappa = curvature(center)

    # Charger modèles
    model_path = data_dir / "models" / "vehicle_limits.json"
    adv_env_path = data_dir / "models" / "lateral_envelope.json"
    adv_vlim_path = data_dir / "models" / "vlim_model.joblib"
    model = None
    if model_path.exists():
        model = VehicleLimitsModel.load(model_path)

    env_model = LateralEnvelope.load(adv_env_path) if adv_env_path.exists() else None
    vlim_model = VLimModel.load(adv_vlim_path) if adv_vlim_path.exists() else None
    logger.info("models: env=%s vlim=%s old=%s", adv_env_path.exists(), adv_vlim_path.exists(), (model_path.exists()))

    # Construire DataFrame de features pour mu / multiplicateur
    weather_path = sess_dir / "weather.parquet"
    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
    else:
        weather_csv = sess_dir / "weather.csv"
        if weather_csv.exists():
            weather = pd.read_csv(weather_csv)
        else:
            weather = pd.DataFrame()
    logger.info("weather: parquet=%s csv=%s loaded_empty=%s", weather_path.exists(), (sess_dir / 'weather.csv').exists(), weather.empty)

    feats = pd.DataFrame({
        "Curvature": kappa,
        "AirTemp": weather.get("AirTemp", pd.Series(np.full(len(kappa), 20.0))),
        "TrackTemp": weather.get("TrackTemp", pd.Series(np.full(len(kappa), 30.0))),
        "Rainfall": weather.get("Rainfall", pd.Series(np.zeros(len(kappa)))),
        "Speed": pd.Series(np.full(len(kappa), 60.0)),  # placeholder si requis par extract
    })
    if use_adv and env_model is not None and vlim_model is not None:
        # multiplicateur environnemental
        m_env = vlim_model.predict_multiplier(feats[["Curvature", "AirTemp", "TrackTemp", "Rainfall"]])
        xy_line, offsets = optimize_line_iterative(
            center, track_half_width=half_width, env_multiplier=m_env, envelope=env_model,
            a_long_max=cfg["optimization"].get("a_long_max", 9.0), iterations=iters,
            smoothing_weight=smooth_w, apex_weight=apex_w
        )
        mu = None
    else:
        # fallback ancien modèle
        if model is None:
            logger.error("no models available: expected %s (advanced) or %s (legacy)", adv_env_path, model_path)
            raise FileNotFoundError("Aucun modèle de limites disponible. Entraînez ml.training_advanced ou ml.training.")
        mu = model.predict_mu(feats)
        xy_line, offsets = optimize_line(
            center, track_half_width=half_width, mu_profile=mu,
            a_lat_max_base=a_lat_base, smoothing_weight=smooth_w, apex_weight=apex_w
        )

    out = {
        "centerline": center,
        "racing_line": xy_line,
        "offsets": offsets,
        "mu": mu,
    }
    logger.info("run_inference done: center=%s line=%s", getattr(center, 'shape', None), getattr(xy_line, 'shape', None))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--session", required=True, help="Identifiant session ex: 2024_Monaco_R")
    args = ap.parse_args()
    out = run_inference(args.config, args.session)
    print("OK - Trajectoire calculée", {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in out.items()})
