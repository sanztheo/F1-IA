from __future__ import annotations

import argparse
import pathlib as _pl
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yaml
import time
from tqdm import tqdm

from .features.track_processing import posdata_to_centerline, curvature, arc_length
from .models.phys_params import fit_lateral_envelope, LateralEnvelope
from .models.vlim_regressor import VLimModel


def _load_positions(sess_dir: _pl.Path) -> pd.DataFrame | None:
    p = sess_dir / "positions.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    p = sess_dir / "positions.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None


def _load_telemetry(sess_dir: _pl.Path) -> pd.DataFrame | None:
    p = sess_dir / "telemetry_best_per_driver.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    p = sess_dir / "telemetry_best_per_driver.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None


def _load_weather(sess_dir: _pl.Path) -> pd.DataFrame:
    p = sess_dir / "weather.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    p = sess_dir / "weather.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame()


def build_training_table(raw_root: _pl.Path, show_progress: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[pd.DataFrame] = []
    stats = {"sessions_total": 0, "sessions_used": 0, "rows_total": 0}
    sess_list = sorted((raw_root / "fastf1").glob("*_*_*"))
    iterator = tqdm(sess_list, desc="Advanced training - sessions", smoothing=0.1) if show_progress else sess_list
    for sess_dir in iterator:
        stats["sessions_total"] += 1
        pos = _load_positions(sess_dir)
        tel = _load_telemetry(sess_dir)
        if pos is None or tel is None or pos.empty or tel.empty:
            continue
        center = posdata_to_centerline(pos)
        kappa = curvature(center)
        s = arc_length(center)

        # Choisir un pilote (le plus rapide) – on prend la médiane de tous les meilleurs tours
        v_list = []
        if "Distance" in tel.columns and "Speed" in tel.columns:
            # CarData Speed souvent en km/h => convertir en m/s si > 150
            sp = tel[["Distance", "Speed"]].dropna()
            v_vals = sp["Speed"].to_numpy()
            if np.nanmedian(v_vals) > 100:  # km/h probable
                v_vals = v_vals / 3.6
            # Remettre sur [0, L]
            dist = sp["Distance"].to_numpy()
            # Normaliser un tour à la longueur du centerline
            L = s[-1]
            # Interpolation sur s
            v_on_s = np.interp(s, np.clip(dist, 0, L), np.clip(v_vals, 0, None))
            v_list.append(v_on_s)
        if not v_list:
            continue
        v = np.median(np.stack(v_list, axis=0), axis=0)
        a_lat_obs = (v ** 2) * np.abs(kappa)

        w = _load_weather(sess_dir)
        ta = np.nanmedian(w.get("AirTemp", pd.Series([20.0]))) if not w.empty else 20.0
        tt = np.nanmedian(w.get("TrackTemp", pd.Series([30.0]))) if not w.empty else 30.0
        rain = np.nanmedian(w.get("Rainfall", pd.Series([0.0]))) if not w.empty else 0.0

        df = pd.DataFrame({
            "Speed": v,
            "Curvature": kappa,
            "a_lat_obs": a_lat_obs,
            "AirTemp": ta,
            "TrackTemp": tt,
            "Rainfall": rain,
        })
        rows.append(df)
        stats["sessions_used"] += 1
        stats["rows_total"] += len(df)

    tbl = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return tbl, stats


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(_pl.Path(config_path).read_text())
    data_dir = _pl.Path(cfg.get("data_dir", "data"))
    raw_root = data_dir / "raw"

    t0 = time.time()
    tbl, stats = build_training_table(raw_root, show_progress=True)
    if tbl.empty:
        print("Aucune donnée exploitable pour l'entraînement avancé.")
        print(f"Sessions: {stats['sessions_total']}, utilisées: {stats['sessions_used']}, lignes: {stats['rows_total']}")
        return

    v = tbl["Speed"].to_numpy()
    a_lat_obs = tbl["a_lat_obs"].to_numpy()

    # 1) Ajuster enveloppe latérale sur le 95e centile
    t_fit1 = time.time()
    env = fit_lateral_envelope(v, a_lat_obs, quantile=0.95)
    t_fit1 = time.time() - t_fit1

    # 2) Apprendre un multiplicateur environnemental m(s)
    #    cible: m_hat = a_lat_obs / (c0 + c1*v^2) (clamp)
    denom = env.c0 + env.c1 * (v ** 2)
    m_hat = np.clip(a_lat_obs / np.maximum(1e-6, denom), 0.5, 1.5)
    t_fit2 = time.time()
    vlim_model = VLimModel().fit(tbl[["Curvature", "AirTemp", "TrackTemp", "Rainfall"]], m_hat)
    t_fit2 = time.time() - t_fit2

    # Sauvegarde
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    env.save(models_dir / "lateral_envelope.json")
    vlim_model.save(models_dir / "vlim_model.joblib")
    elapsed = time.time() - t0
    print("Modèles avancés sauvegardés:")
    print(f"  - {models_dir / 'lateral_envelope.json'}")
    print(f"  - {models_dir / 'vlim_model.joblib'}")
    print("Résumé entraînement avancé:")
    print(f"  Sessions: {stats['sessions_total']} | utilisées: {stats['sessions_used']} | lignes: {stats['rows_total']}")
    print(f"  Fit enveloppe: {t_fit1:.2f}s | Fit multiplicateur: {t_fit2:.2f}s | Total: {elapsed:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
