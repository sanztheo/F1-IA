from __future__ import annotations

import json
import pathlib as _pl
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


try:
    from sklearn.linear_model import QuantileRegressor
    _HAS_QR = True
except Exception:  # pragma: no cover
    from sklearn.linear_model import Ridge
    _HAS_QR = False


@dataclass
class LateralEnvelope:
    """Modèle simple de limite latérale: a_lat_max(s) = c0 + c1 * v(s)^2.

    - c0 agrège l'adhérence mécanique (≈ mu*g)
    - c1 agrège l'effet aéro (≈ mu*k)
    """

    c0: float = 9.0
    c1: float = 0.02

    def a_lat(self, v: np.ndarray) -> np.ndarray:
        return self.c0 + self.c1 * (v ** 2)

    def save(self, path: str | _pl.Path) -> None:
        _pl.Path(path).write_text(json.dumps({"c0": self.c0, "c1": self.c1}, indent=2))

    @classmethod
    def load(cls, path: str | _pl.Path) -> "LateralEnvelope":
        d = json.loads(_pl.Path(path).read_text())
        return cls(c0=float(d.get("c0", 9.0)), c1=float(d.get("c1", 0.02)))


def fit_lateral_envelope(v_ms: np.ndarray, a_lat_obs: np.ndarray, quantile: float = 0.95) -> LateralEnvelope:
    """Ajuste c0, c1 sur le quantile supérieur de a_lat ≈ c0 + c1 v^2.

    Utilise QuantileRegressor si dispo (robuste à l'enveloppe), sinon Ridge.
    """
    X = (v_ms ** 2).reshape(-1, 1)
    y = a_lat_obs.reshape(-1)

    mask = np.isfinite(X).ravel() & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    if len(y) < 100:
        return LateralEnvelope()

    if _HAS_QR:
        # Inclure constante via fit_intercept
        qr = QuantileRegressor(quantile=quantile, alpha=0.0, fit_intercept=True, solver="highs")
        qr.fit(X, y)
        c1 = float(qr.coef_[0])
        c0 = float(qr.intercept_)
    else:  # fallback
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(PolynomialFeatures(degree=1, include_bias=True), Ridge(alpha=1.0))
        model.fit(X, y)
        # coef_ pour deg=1: [bias, v2]
        coefs = getattr(model[-1], "coef_", np.array([0.0, 0.02]))
        c0 = float(coefs[0])
        c1 = float(coefs[1])

    # Bornes raisonnables
    c0 = float(np.clip(c0, 3.0, 25.0))
    c1 = float(np.clip(c1, 1e-4, 0.5))
    return LateralEnvelope(c0=c0, c1=c1)

