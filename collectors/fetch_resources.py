from __future__ import annotations

"""Orchestrateur de collecte multi-sources.

Usage (après installation deps via votre venv):
  python -m collectors.fetch_resources --config config.yaml
"""

import argparse
import json
import os
import logging
import pathlib as _pl
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import yaml
from tqdm import tqdm
import pandas as pd
import concurrent.futures as cf

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

    # Options d'exécution
    workers_env = int(os.environ.get("F1IA_FETCH_WORKERS", "0") or 0)
    default_workers = min(4, max(1, (os.cpu_count() or 4)))
    workers = workers_env if workers_env > 0 else default_workers
    quiet = os.environ.get("F1IA_FETCH_VERBOSE", "0") not in ("1", "true", "TRUE", "True")

    _configure_logging(quiet=quiet)

    # Construire la liste des tâches (année, event_name, session)
    tasks: List[Tuple[int, str, str]] = []
    for year in cfg.seasons:
        resolved: List[str] = []
        for event in cfg.tracks:
            event_names = _schedule_matches_for_track(year, event)
            if not event_names:
                if not quiet:
                    tqdm.write(f"Skip {year}-{event}: non présent au calendrier (annulé ou renommé)")
                continue
            resolved.extend(event_names)
        for ev_name in sorted(set(resolved)):
            for sess_code in cfg.sessions:
                tasks.append((year, ev_name, sess_code))

    if not tasks:
        print("Aucune session à traiter.")
        return

    ok = 0
    err = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_single, t[0], t[1], t[2], dirs["raw"], cfg.storage_format) for t in tasks]
        for fut in tqdm(cf.as_completed(futs), total=len(futs), desc=f"Fetching ({workers} workers)", smoothing=0.1):
            res = fut.result()
            if res and res.get("ok"):
                manifest["resources"].append(res["entry"]) 
                ok += 1
            else:
                err += 1
                if not quiet and res is not None:
                    tqdm.write(f"Error {res.get('session')}: {res.get('error')}")

    (cfg.data_dir / "resources.json").write_text(json.dumps(manifest, indent=2))
    print(f"Sessions ok: {ok}, erreurs: {err}. Manifeste écrit: {cfg.data_dir / 'resources.json'}")


def _fetch_single(year: int, ev_name: str, sess_code: str, raw_dir: _pl.Path, fmt: str) -> Dict[str, Any]:
    sid = session_identifier(year, ev_name, sess_code)
    try:
        sess = load_session(year, ev_name, sess_code)
    except Exception as e:
        return {"ok": False, "session": sid, "error": str(e)}
    out_dir = raw_dir / "fastf1" / sid
    try:
        m = export_session_core(sess, out_dir, fmt=fmt)
        entry = {
            "provider": "fastf1",
            "session": sid,
            "path": str(out_dir),
            "tables": m.get("tables", []),
        }
        return {"ok": True, "entry": entry}
    except Exception as e:
        return {"ok": False, "session": sid, "error": str(e)}


def _configure_logging(quiet: bool = True, level: int = logging.ERROR) -> None:
    # Réduit les logs FastF1/requests pour laisser la place aux barres de progression
    if quiet:
        base_level = level
    else:
        base_level = logging.INFO
    logging.getLogger().setLevel(base_level)
    for name in ("fastf1", "req", "core", "_api", "logger", "urllib3"):
        lg = logging.getLogger(name)
        lg.setLevel(base_level)
        lg.propagate = False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=None, help="Nombre de workers parallèles (défaut: env F1IA_FETCH_WORKERS ou auto)")
    ap.add_argument("--verbose", action="store_true", help="Afficher les logs FastF1 (par défaut silencieux)")
    args = ap.parse_args()

    # Propager options via env pour main()
    if args.workers is not None:
        os.environ["F1IA_FETCH_WORKERS"] = str(args.workers)
    if args.verbose:
        os.environ["F1IA_FETCH_VERBOSE"] = "1"
    main(args.config)
