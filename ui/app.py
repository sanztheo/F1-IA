from __future__ import annotations

import sys
from pathlib import Path
import pathlib as _pl
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml
import streamlit as st
import plotly.graph_objects as go

# Ajouter la racine du projet au PYTHONPATH quand lancé depuis ui/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replay.multicar_fastf1 import load_multicar
from tracks.osm_tracks import fetch_track_outline, resample_linestring, outline_to_centerline
from tracks.align import procrustes_2d, apply_similarity
from simulation.lap_simulator import simulate_lap


st.set_page_config(page_title="F1‑IA – Replay & IA Evolution", layout="wide")

log = logging.getLogger("f1ia.ui")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


@st.cache_data
def load_cfg(path: str = "config.yaml") -> Dict[str, Any]:
    return yaml.safe_load(_pl.Path(path).read_text())


EVENT_MAP = {
    "Circuit de Monaco": "Monaco Grand Prix",
    "Autodromo Nazionale Monza": "Italian Grand Prix",
    "Circuit de Spa-Francorchamps": "Belgian Grand Prix",
}


def make_animation(cars: List[Dict[str, Any]], track_xy: np.ndarray, fps: int = 20) -> go.Figure:
    T = len(cars[0]["x"]) if cars else 0
    fig = go.Figure()
    # Piste (centerline) en fond
    if len(track_xy):
        fig.add_trace(go.Scatter(x=track_xy[:, 0], y=track_xy[:, 1], name="Piste", mode="lines", line=dict(color="#666", width=2)))

    # Traces voitures (markers uniquement)
    for car in cars:
        fig.add_trace(go.Scatter(x=[car["x"][0]], y=[car["y"][0]], mode="markers", name=car["name"],
                                 marker=dict(size=10, color=car["color"])) )

    # Frames
    frames = []
    for i in range(T):
        frame_data = []
        # première trace est la piste -> on ne la met pas dans frames
        for car in cars:
            frame_data.append(go.Scatter(x=[car["x"][i]], y=[car["y"][i]], mode="markers",
                                         marker=dict(size=10, color=car["color"])) )
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.frames = frames
    # Boutons lecture
    frame_duration = int(1000 / max(1, fps))
    fig.update_layout(
        height=750,
        legend=dict(orientation="h"),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": frame_duration, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }
        ],
        sliders=[{
            "currentvalue": {"prefix": "t: ", "visible": True},
            "pad": {"t": 50},
            "steps": [{"args": [[str(i)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": str(i), "method": "animate"} for i in range(T)]
        }]
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def main():
    cfg = load_cfg()
    circuits = cfg.get("tracks", ["Circuit de Spa-Francorchamps"])
    year = int(cfg.get("reference_year", 2022))

    st.title("F1‑IA – Replay multi‑voitures + Trajectoire IA (CMA‑ES)")
    tab1, tab2 = st.tabs(["Replay réel", "IA Evolution"])

    with tab1:
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            circuit = st.selectbox("Circuit", circuits, index=0)
        with c2:
            year_sel = st.number_input("Année", value=year, min_value=2018, max_value=2025, step=1)
        with c3:
            sess_code = st.selectbox("Session", ["Q", "R"], index=1)

        n_cars = st.slider("Nombre de voitures", 3, 10, 10)
        fps = st.select_slider("Vitesse (fps)", options=[10, 15, 20, 30, 40], value=20)

        if st.button("Charger le replay", type="primary"):
            with st.spinner("Chargement des données pilotes (FastF1)…"):
                event = EVENT_MAP.get(circuit, circuit)
                data = load_multicar(int(year_sel), event, sess_code, n_cars)
                # Piste via OSM et alignement sur données XY
                outline = fetch_track_outline(circuit)
                track_xy = resample_linestring(outline, 2000)
                if track_xy.size == 0 or data.get("centerline") is None:
                    aligned = data.get("centerline") if data.get("centerline") is not None else np.zeros((0, 2))
                else:
                    try:
                        centerline = outline_to_centerline(track_xy)
                        R, s, t = procrustes_2d(data["centerline"], centerline)
                        aligned = apply_similarity(R, s, t, centerline)
                    except Exception:
                        aligned = data["centerline"]
                st.session_state["replay_payload"] = {"cars": data["cars"], "track": aligned}

        if "replay_payload" in st.session_state:
            cars = st.session_state["replay_payload"]["cars"]
            track = st.session_state["replay_payload"]["track"]
            fig = make_animation(cars, track, fps=fps)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("Expérimenter une trajectoire IA par évolution (CMA‑ES).")
        st.caption("Cette démo effectue des évaluations locales seulement quand vous cliquez. Budget défini dans config.yaml.")
        circuit2 = st.selectbox("Circuit pour l'IA", circuits, index=0, key="c2")
        event2 = EVENT_MAP.get(circuit2, circuit2)
        year2 = st.number_input("Année données (pour centerline)", value=year, min_value=2018, max_value=2025, step=1, key="y2")
        run_id = f"{event2.replace(' ', '_')}_{int(year2)}"
        step = st.select_slider("Évaluations à exécuter maintenant", options=[50,100,200,400], value=100)
        colA, colB = st.columns(2)
        if colA.button("Continuer l'entraînement (checkpoint)"):
            try:
                from evolution.cmaes_trainer import optimize_line_cmaes, CMAESConfig
            except Exception:
                st.error("Installez la dépendance 'cmaes' dans votre venv.")
                return
            with st.spinner("Prépare la piste et lance CMA‑ES (aperçu)…"):
                # Construire centerline depuis FastF1 (plus sûr) sinon OSM
                try:
                    data = load_multicar(int(year2), event2, "Q", 3)
                    center = data.get("centerline")
                except Exception:
                    center = None
                if center is None or len(center) == 0:
                    outline = fetch_track_outline(circuit2)
                    center = outline_to_centerline(resample_linestring(outline, 2000))
                if center is None or len(center) == 0:
                    st.error("Impossible d'obtenir la géométrie de piste.")
                    return

                evol = cfg.get("evolution", {})
                total = int(evol.get("evaluations_per_circuit", 400))
                conf = CMAESConfig(evaluations=step,
                                   n_ctrl=int(evol.get("n_ctrl_points", 25)),
                                   track_half_width=float(cfg["optimization"]["track_half_width_m"]))

                def sim_fn(xy):
                    from ml.features.track_processing import curvature
                    k = curvature(xy)
                    v_lim = np.sqrt(np.maximum(1e-3, (12.0 * 9.80665) / (np.abs(k) + 1e-6)))
                    out = simulate_lap(xy, k, v_lim, a_long_max=float(cfg["optimization"]["a_long_max"]))
                    return out["time_s"], out

                res = optimize_line_cmaes(center, sim_fn, conf, run_id=run_id, resume=True)
                st.session_state["evo_payload"] = {"center": center, "ia": res["best_line"], "time": res["best_time"], "hist": res["history"]}
        if colB.button("Réinitialiser le run"):
            import shutil
            import pathlib as _pl
            run_dir = _pl.Path("data/evolution") / run_id
            if run_dir.exists():
                shutil.rmtree(run_dir)
            st.success("Checkpoint supprimé. Vous pouvez relancer l'entraînement.")

        if "evo_payload" in st.session_state:
            payload = st.session_state["evo_payload"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=payload["center"][:, 0], y=payload["center"][:, 1], name="Centerline", mode="lines", line=dict(color="#777")))
            fig.add_trace(go.Scatter(x=payload["ia"][:, 0], y=payload["ia"][:, 1], name="IA", mode="lines", line=dict(color="#E91E63", width=4)))
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_layout(height=700, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Temps simulé IA (s)", f"{payload['time']:.3f}")
            st.line_chart(pd.DataFrame({"meilleur": payload["hist"]}).cummin())


if __name__ == "__main__":
    main()
