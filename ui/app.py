from __future__ import annotations

import pathlib as _pl
import sys
from pathlib import Path as _Path

# Assure l'import des paquets du projet quand Streamlit lance depuis `ui/`
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import numpy as np
import pandas as pd
import yaml
import streamlit as st
import plotly.graph_objects as go

from ml.inference import run_inference
from ml.features.track_processing import curvature
from simulation.lap_simulator import simulate_lap, speed_profile_from_curvature


st.set_page_config(page_title="F1-IA Trajectoire Optimale", layout="wide")


@st.cache_data
def load_cfg(path: str):
    return yaml.safe_load(_pl.Path(path).read_text())


def main():
    st.title("F1-IA – Trajectoire optimale")
    cfg = load_cfg("config.yaml")

    seasons = list(range(int(cfg["seasons"]["start"]), int(cfg["seasons"]["end"]) + 1))
    tracks = cfg.get("tracks", [])
    sessions = cfg.get("sessions", ["R"])

    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.selectbox("Saison", seasons, index=len(seasons) - 1)
    with c2:
        track = st.selectbox("Circuit", tracks, index=0)
    with c3:
        sess_code = st.selectbox("Session", sessions, index=sessions.index("R") if "R" in sessions else 0)

    session_id = f"{year}_{track}_{sess_code}"
    st.caption(f"Session: {session_id}")

    run_button = st.button("Lancer la simulation", type="primary")

    if run_button:
        with st.spinner("Calcul de la trajectoire IA..."):
            try:
                out = run_inference("config.yaml", session_id)
            except FileNotFoundError:
                st.error("Données manquantes. Exécutez d'abord la collecte et l'entraînement.")
                return
            center = out["centerline"]
            line = out["racing_line"]

            # Simulation temps au tour sur la ligne IA
            kappa = curvature(line)
            mu = np.median(out["mu"]) if isinstance(out.get("mu"), np.ndarray) else 1.6
            v_lim = speed_profile_from_curvature(kappa, mu)
            sim = simulate_lap(line, kappa, v_lim, a_long_max=cfg["optimization"].get("a_long_max", 9.0))

            # Placeholder pour la ligne de référence (centerline avec vitesses limites)
            kappa_ref = curvature(center)
            v_ref_lim = speed_profile_from_curvature(kappa_ref, mu)
            sim_ref = simulate_lap(center, kappa_ref, v_ref_lim, a_long_max=cfg["optimization"].get("a_long_max", 9.0))

        delta = sim_ref["time_s"] - sim["time_s"]
        thr = float(cfg["optimization"].get("delta_win_threshold_s", 0.15))
        verdict = "Victoire probable: OUI" if delta >= thr else "Victoire probable: NON"

        st.subheader("Verdict")
        st.metric("Delta IA vs Référence (s)", value=f"{delta:.3f}", delta=f"seuil {thr:.2f}s")
        st.success(verdict) if delta >= thr else st.warning(verdict)

        st.subheader("Trajectoire sur le circuit")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=center[:, 0], y=center[:, 1], name="Centerline", mode="lines", line=dict(color="#888", width=2)))
        fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], name="IA", mode="lines", line=dict(color="#E91E63", width=3)))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(height=700, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Profils vitesse")
        v1 = sim["v"]
        v2 = sim_ref["v"]
        s1 = np.cumsum(sim["ds"]) - sim["ds"][0]
        s2 = np.cumsum(sim_ref["ds"]) - sim_ref["ds"][0]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=s1, y=v1, name="IA", mode="lines"))
        fig2.add_trace(go.Scatter(x=s2, y=v2, name="Référence", mode="lines"))
        fig2.update_layout(height=400, xaxis_title="Distance (m)", yaxis_title="Vitesse (m/s)")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption("Note: la référence ici est le centerline; pour une comparaison pilote gagnant, branchez les meilleurs tours FastF1.")

    st.info("Étapes: 1) Collecte 2) Entraînement 3) Simulation. Utilisez votre venv.")


if __name__ == "__main__":
    main()
