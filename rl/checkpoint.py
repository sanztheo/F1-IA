from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def save_checkpoint(path: Path, gen: int, policies: List[Dict], best_time: float, best_policy: Dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(policies, dtype=object)
    if best_policy is not None:
        np.savez(path, gen=np.array([gen]), best_time=np.array([best_time]), policies=arr,
                 best_policy=np.array([best_policy], dtype=object), allow_pickle=True)
    else:
        np.savez(path, gen=np.array([gen]), best_time=np.array([best_time]), policies=arr, allow_pickle=True)


def load_checkpoint(path: Path, pop: int) -> Tuple[List[Dict], int, float, Dict | None]:
    if not path.exists():
        return [], 0, float('inf'), None
    try:
        z = np.load(path, allow_pickle=True)
        gen = int(z.get("gen", np.array([0]))[0])
        best_time = float(z.get("best_time", np.array([float('inf')]))[0])
        pols = z["policies"].tolist()
        best_pol = None
        if "best_policy" in z:
            bp = z["best_policy"].tolist()
            if isinstance(bp, list) and bp:
                best_pol = bp[0]
        if not isinstance(pols, list) or not pols:
            return [], gen, best_time, best_pol
        return pols[:pop], gen, best_time, best_pol
    except Exception:
        return [], 0, float('inf'), None

