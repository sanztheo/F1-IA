from __future__ import annotations

import json
import pathlib as _pl
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _features_df(df: pd.DataFrame) -> pd.DataFrame:
    # Caractéristiques simples: |kappa|, |dkappa|, AirTemp, TrackTemp, Rainfall
    k = np.abs(df.get("Curvature", pd.Series(np.zeros(len(df)))))
    dk = np.abs(np.gradient(k)) if len(k) else np.zeros(len(k))
    feats = pd.DataFrame({
        "abs_kappa": np.asarray(k),
        "abs_dkappa": np.asarray(dk),
        "AirTemp": df.get("AirTemp", pd.Series(np.zeros(len(k)))).to_numpy(),
        "TrackTemp": df.get("TrackTemp", pd.Series(np.zeros(len(k)))).to_numpy(),
        "Rainfall": df.get("Rainfall", pd.Series(np.zeros(len(k)))).to_numpy(),
    })
    return feats


@dataclass
class VLimModel:
    """Prédit un multiplicateur environnemental m(s)∈[0.5,1.5] appliqué à l'enveloppe latérale.

    v_lim(s) sera obtenu via inversion de a_lat_max(s) = m(s)*(c0 + c1*v^2).
    """

    model: Pipeline | None = None

    def fit(self, df: pd.DataFrame, target: np.ndarray) -> "VLimModel":
        X = _features_df(df)
        y = np.asarray(target).reshape(-1)

        # clamp et log pour stabiliser
        y = np.clip(y, 0.5, 1.5)
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("hgb", HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=300, l2_regularization=0.0)),
        ])
        pipe.fit(X, y)
        self.model = pipe
        return self

    def predict_multiplier(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.ones(len(df))
        X = _features_df(df)
        m = np.asarray(self.model.predict(X))
        return np.clip(m, 0.5, 1.5)

    def save(self, path: str | _pl.Path) -> None:
        import joblib
        _pl.Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | _pl.Path) -> "VLimModel":
        import joblib
        m = joblib.load(path)
        v = cls(model=m)
        return v
