from __future__ import annotations

import pathlib as _pl
from typing import Dict, Any, Optional
import requests
import pandas as pd


class JolpicaErgast:
    """Client léger pour l'API Ergast-compatible de Jolpica.

    Base: https://api.jolpi.ca/ergast/f1
    """

    def __init__(self, base_url: str = "https://api.jolpi.ca/ergast/f1", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **params) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def circuits(self, season: int) -> pd.DataFrame:
        data = self._get(f"{season}/circuits.json")
        items = data.get("MRData", {}).get("CircuitTable", {}).get("Circuits", [])
        return pd.json_normalize(items)

    def results(self, season: int, round_: Optional[int] = None) -> pd.DataFrame:
        path = f"{season}.json" if round_ is None else f"{season}/{round_}/results.json"
        data = self._get(path)
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return pd.json_normalize(races, record_path=["Results"], meta=["raceName", "round", ["Circuit", "circuitName"]])

