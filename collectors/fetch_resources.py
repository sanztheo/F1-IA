from __future__ import annotations

"""Orchestrateur de collecte multi-sources.

Usage (après installation deps via votre venv):
  python -m collectors.fetch_resources --config config.yaml
"""

import argparse
import json
import pathlib as _pl
from dataclasses import dataclass
from typing import Dict, Any, List

import yaml
from tqdm import tqdm
import pandas as pd

from .fastf1_client import load_session, export_session_core, session_identifier


def _schedule_matches_for_track(year: int, track_query: str):
    """Retourne les noms d'événements (EventName) correspondant au track_query pour l'année.

    Évite les corrections hasardeuses de FastF1 en pré-filtrant via le calendrier.
    """
    import fastf1
    try:
        sched = fastf1.get_event_schedule(year, include_testing=False)
    except Exception:
        return []
    cols = [c for c in ["Location", "EventName", "OfficialEventName"] if c in sched.columns]
    if not cols:
        return []
    q = track_query.lower()
    mask = False
    for c in cols:
        mask = mask | sched[c].astype(str).str.lower().str.contains(q, na=False)
    matches = sched[mask]
    # Retirer éventuels événements annulés/inexistants (si colonne 'EventFormat' ou 'F1ApiSupport' indique absence)
    # FastF1 ne liste généralement pas Monaco 2020 car annulé, donc matches serait vide.
    return list(dict.fromkeys(matches.get("EventName", pd.Series(dtype=str)).tolist()))


@dataclass
class Config:
    data_dir: _pl.Path
    storage_format: str
    seasons: range
    tracks: List[str]
    sessions: List[str]


def read_config(path: str | _pl.Path) -> Config:
    cfg = yaml.safe_load(_pl.Path(path).read_text())
    data_dir = _pl.Path(cfg.get("data_dir", "data"))
    storage_format = cfg.get("storage_format", "parquet").lower()
    seasons = range(int(cfg["seasons"]["start"]), int(cfg["seasons"]["end"]) + 1)
    tracks = list(cfg.get("tracks", []))
    sessions = list(cfg.get("sessions", ["R"]))
    return Config(data_dir, storage_format, seasons, tracks, sessions)


def ensure_dirs(base: _pl.Path) -> Dict[str, _pl.Path]:
    raw = base / "raw"
    processed = base / "processed"
    models = base / "models"
    for d in (raw, processed, models):
        d.mkdir(parents=True, exist_ok=True)
    return {"raw": raw, "processed": processed, "models": models}


def main(config_path: str) -> None:
    cfg = read_config(config_path)
    dirs = ensure_dirs(cfg.data_dir)
    manifest: Dict[str, Any] = {"resources": []}

    for year in tqdm(cfg.seasons, desc="Seasons"):
        for event in tqdm(cfg.tracks, leave=False, desc=f"Year {year}"):
            # Résoudre 'event' vers 0..n noms d'événements concrets via le calendrier
            event_names = _schedule_matches_for_track(year, event)
            if not event_names:
                tqdm.write(f"Skip {year}-{event}: non présent au calendrier (annulé ou renommé)")
                continue
            for ev_name in event_names:
                for sess_code in tqdm(cfg.sessions, leave=False, desc="Sessions"):
                    try:
                        sess = load_session(year, ev_name, sess_code)
                    except Exception as e:
                        tqdm.write(f"Skip {year}-{ev_name}-{sess_code}: {e}")
                        continue

                    sid = session_identifier(year, ev_name, sess_code)
                    out_dir = dirs["raw"] / "fastf1" / sid
                    try:
                        m = export_session_core(sess, out_dir, fmt=cfg.storage_format)
                        manifest["resources"].append({
                            "provider": "fastf1",
                            "session": sid,
                            "path": str(out_dir),
                            "tables": m.get("tables", []),
                        })
                    except Exception as e:
                        tqdm.write(f"Export error {sid}: {e}")

    (cfg.data_dir / "resources.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifeste écrit: {cfg.data_dir / 'resources.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
