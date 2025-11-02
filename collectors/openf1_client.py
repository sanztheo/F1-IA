from __future__ import annotations

import os
from typing import Dict, Any, Iterable, Optional
import json
import pathlib as _pl

import requests
import pandas as pd


class OpenF1Client:
    """Client minimal pour OpenF1 (https://openf1.org).

    Note: Pour la trajectoire XY détaillée, on privilégie souvent FastF1.
    OpenF1 expose surtout l'avancement/position et les tours.
    """

    def __init__(self, base_url: str = "https://api.openf1.org/v1", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        r = requests.get(url, params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json()
        except json.JSONDecodeError:
            return r.text

    def locations(self, session_key: int | str, **kwargs) -> pd.DataFrame:
        data = self._get("location", {"session_key": session_key, **kwargs})
        return pd.DataFrame(data)

    def laps(self, session_key: int | str, driver_number: Optional[int] = None, **kwargs) -> pd.DataFrame:
        params = {"session_key": session_key, **kwargs}
        if driver_number is not None:
            params["driver_number"] = driver_number
        data = self._get("laps", params)
        return pd.DataFrame(data)

    def telemetry(self, session_key: int | str, driver_number: Optional[int] = None, **kwargs) -> pd.DataFrame:
        params = {"session_key": session_key, **kwargs}
        if driver_number is not None:
            params["driver_number"] = driver_number
        data = self._get("telemetry", params)
        return pd.DataFrame(data)


def save_df(df: pd.DataFrame, out: _pl.Path, fmt: str = "parquet") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)


def fetch_openf1_session_dump(
    session_key: int | str,
    out_dir: str | os.PathLike,
    fmt: str = "parquet",
    driver_number: Optional[int] = None,
) -> dict:
    """Télécharge quelques tables OpenF1 pour une session et les stocke.

    Retourne un manifeste minimal.
    """
    out_base = _pl.Path(out_dir)
    cli = OpenF1Client()

    manifest: Dict[str, Any] = {"session_key": session_key, "tables": []}

    tbls = {
        "location": cli.locations(session_key),
        "laps": cli.laps(session_key, driver_number=driver_number),
    }
    # telemetry peut être volumineux — optionnel
    try:
        tbls["telemetry"] = cli.telemetry(session_key, driver_number=driver_number)
    except Exception:
        pass

    for name, df in tbls.items():
        if df is not None and not df.empty:
            path = out_base / f"{name}.{ 'csv' if fmt=='csv' else 'parquet'}"
            save_df(df, path, fmt)
            manifest["tables"].append({"name": name, "path": str(path)})

    return manifest

