from __future__ import annotations

import json
import pathlib as _pl
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


GRAV = 9.80665


@dataclass
class VehicleLimitsModel:
    """Petit modèle paramétrique pour estimer l'adhérence (mu) et a_long.

    - mu(features) via régression Ridge sur features météo/pneus.
    - a_long_max dépendant faiblement de mu (optionnel).
    """

    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    a_long_max: float = 9.0

    def fit(self, df: pd.DataFrame) -> "VehicleLimitsModel":
        feats, y_mu = self._extract_features_targets(df)
        if len(feats) == 0:
            # défaut raisonnable si pas de données suffisantes
            self.coef_ = np.zeros(3)
            self.intercept_ = 1.6  # mu ~ pneus slicks secs
            return self
        reg = Ridge(alpha=1.0)
        reg.fit(feats, y_mu)
        self.coef_ = reg.coef_.astype(float)
        self.intercept_ = float(reg.intercept_)
        return self

    def predict_mu(self, df: pd.DataFrame) -> np.ndarray:
        feats, _ = self._extract_features_targets(df, train=False)
        if self.coef_ is None:
            return np.full(len(feats), 1.6)
        return feats @ self.coef_ + self.intercept_

    def _extract_features_targets(self, df: pd.DataFrame, train: bool = True):
        # Features simples: température air, piste, pluie binaire
        ta = df.get("AirTemp", pd.Series(np.zeros(len(df))))
        ts = df.get("TrackTemp", pd.Series(np.zeros(len(df))))
        rain = df.get("Rainfall", pd.Series(np.zeros(len(df))))
        feats = np.column_stack([ta.to_numpy(), ts.to_numpy(), (rain > 0).astype(float)])
        if train:
            # cible mu approximée via v^2 * kappa / g (mu >= a_lat/g)
            v = df.get("Speed", pd.Series(np.nan)).to_numpy()
            kappa = df.get("Curvature", pd.Series(np.nan)).to_numpy()
            with np.errstate(invalid="ignore"):
                mu_hat = np.maximum(0.2, np.minimum(2.5, (v ** 2 * np.abs(kappa)) / GRAV))
            mu_hat = np.nan_to_num(mu_hat, nan=1.6)
            return feats, mu_hat
        else:
            return feats, None

    def save(self, path: str | _pl.Path) -> None:
        p = _pl.Path(path)
        p.write_text(json.dumps({
            "coef_": None if self.coef_ is None else self.coef_.tolist(),
            "intercept_": self.intercept_,
            "a_long_max": self.a_long_max,
        }, indent=2))

    @classmethod
    def load(cls, path: str | _pl.Path) -> "VehicleLimitsModel":
        d = json.loads(_pl.Path(path).read_text())
        m = cls()
        c = d.get("coef_")
        m.coef_ = None if c is None else np.array(c)
        m.intercept_ = float(d.get("intercept_", 1.6))
        m.a_long_max = float(d.get("a_long_max", 9.0))
        return m

