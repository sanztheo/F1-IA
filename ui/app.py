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
import logging

from ml.inference import run_inference
from ml.features.track_processing import curvature
from simulation.lap_simulator import simulate_lap, speed_profile_from_curvature


st.set_page_config(page_title="F1-IA Trajectoire Optimale", layout="wide")

# Configure logger for terminal output
_logger = logging.getLogger("f1ia.ui")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


@st.cache_data
def load_cfg(path: str):
    return yaml.safe_load(_pl.Path(path).read_text())


def main():
    st.title("F1-IA – Trajectoire optimale")
    cfg = load_cfg("config.yaml")

    # Découverte des sessions disponibles directement depuis le disque
    data_dir = _pl.Path(cfg.get("data_dir", "data"))
    sessions_root = data_dir / "raw" / "fastf1"
    av_dirs_all = sorted([p.name for p in sessions_root.glob("*_*_*")])
    # Ne proposer que les sessions qui possèdent des positions.*
    av_dirs = []
    for name in av_dirs_all:
        p = sessions_root / name
        if (p / "positions.parquet").exists() or (p / "positions.csv").exists():
            av_dirs.append(name)
    if not av_dirs:
        st.error("Aucune session exploitable (positions.*) trouvée dans data/raw/fastf1.")
        st.caption("Exécutez la collecte ou sélectionnez une autre saison/circuit.")
        st.stop()

    # Parser en (year, event, code)
    parsed = []
    for name in av_dirs:
        # format attendu: YEAR_EventName_SessionCode (EventName peut contenir des espaces)
        parts = name.split("_")
        if len(parts) < 3:
            continue
        year = parts[0]
        sess_code = parts[-1]
        event = "_".join(parts[1:-1]).replace("_", " ")
        parsed.append((name, int(year), event, sess_code))

    if not parsed:
        st.error("Impossible d'interpréter les dossiers de sessions.")
        st.stop()

    years = sorted({y for _, y, _, _ in parsed})
    c1, c2, c3 = st.columns(3)
    with c1:
        year_sel = st.selectbox("Saison", years, index=len(years) - 1)
    events = sorted({ev for _, y, ev, _ in parsed if y == year_sel})
    with c2:
        event_sel = st.selectbox("Événement", events, index=0)
    sess_codes = sorted({sc for _, y, ev, sc in parsed if y == year_sel and ev == event_sel})
    with c3:
        sess_sel = st.selectbox("Session", sess_codes, index=sess_codes.index("R") if "R" in sess_codes else 0)

    # Reconstruire exactement le dossier
    # Retrouver le nom exact (avec espaces) tel que dans le dossier
    candidates = [name for name, y, ev, sc in parsed if y == year_sel and ev == event_sel and sc == sess_sel]
    session_id = candidates[0] if candidates else None
    st.caption(f"Session: {session_id}")

    run_button = st.button("Lancer la simulation", type="primary")

    if run_button and session_id:
        with st.spinner("Calcul de la trajectoire IA..."):
            try:
                out = run_inference("config.yaml", session_id)
            except FileNotFoundError as e:
                _logger.error("Data missing: %s", e)
                st.error("Données manquantes pour cette session. Vérifiez la collecte et l'entraînement.")
                st.caption(str(e))
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

        # Animation controls
        st.subheader("Simulation animée")
        colA, colB, colC, colD = st.columns([1,1,2,2])
        T_max = float(max(sim.get("t", [sim["time_s"]])[-1], sim_ref.get("t", [sim_ref["time_s"]])[-1]))
        if "sim_state" not in st.session_state:
            st.session_state.sim_state = {"t": 0.0, "playing": False, "speed": 1.0}
        with colA:
            if st.button("▶ Play" if not st.session_state.sim_state["playing"] else "⏸ Pause"):
                st.session_state.sim_state["playing"] = not st.session_state.sim_state["playing"]
        with colB:
            if st.button("⏹ Reset"):
                st.session_state.sim_state["t"] = 0.0
                st.session_state.sim_state["playing"] = False
        with colC:
            speed = st.select_slider("Vitesse", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=st.session_state.sim_state["speed"])
            st.session_state.sim_state["speed"] = speed
        with colD:
            t_val = st.slider("Temps (s)", min_value=0.0, max_value=T_max, value=float(st.session_state.sim_state["t"]), step=max(T_max/1000.0, 0.05))
            st.session_state.sim_state["t"] = float(t_val)

        # Compute positions for current time
        def _idx(tarr, t):
            import numpy as np
            return int(np.clip(np.searchsorted(tarr, t, side="right") - 1, 0, len(tarr)-1))

        t_ia = sim.get("t")
        t_rf = sim_ref.get("t")
        i_ia = _idx(t_ia, st.session_state.sim_state["t"]) if t_ia is not None else len(line)-1
        i_rf = _idx(t_rf, st.session_state.sim_state["t"]) if t_rf is not None else len(center)-1

        # Plot with moving markers
        st.subheader("Trajectoire sur le circuit")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=center[:, 0], y=center[:, 1], name="Centerline", mode="lines", line=dict(color="#888", width=2)))
        fig.add_trace(go.Scatter(x=line[:, 0], y=line[:, 1], name="IA", mode="lines", line=dict(color="#E91E63", width=3)))
        fig.add_trace(go.Scatter(x=[center[i_rf, 0]], y=[center[i_rf, 1]], mode="markers", name="Ref car", marker=dict(color="#00BCD4", size=10)))
        fig.add_trace(go.Scatter(x=[line[i_ia, 0]], y=[line[i_ia, 1]], mode="markers", name="IA car", marker=dict(color="#E91E63", size=10, symbol="triangle-up")))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(height=700, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

        # Auto-play loop (non bloquant): advance time and rerun
        if st.session_state.sim_state["playing"] and st.session_state.sim_state["t"] < T_max:
            import time as _time
            _time.sleep(1/30.0)
            st.session_state.sim_state["t"] = float(min(T_max, st.session_state.sim_state["t"] + (1/30.0)*st.session_state.sim_state["speed"]))
            st.experimental_rerun()

        delta = sim_ref["time_s"] - sim["time_s"]
        thr = float(cfg["optimization"].get("delta_win_threshold_s", 0.15))
        verdict = "Victoire probable: OUI" if delta >= thr else "Victoire probable: NON"

        st.subheader("Verdict")
        st.metric("Delta IA vs Référence (s)", value=f"{delta:.3f}", delta=f"seuil {thr:.2f}s")
        if delta >= thr:
            st.success(verdict)
        else:
            st.warning(verdict)

        st.subheader("Profils vitesse")
        v1 = sim["v"]
        v2 = sim_ref["v"]
        s1 = np.cumsum(sim["ds"]) - sim["ds"][0]
        s2 = np.cumsum(sim_ref["ds"]) - sim_ref["ds"][0]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=s1, y=v1, name="IA", mode="lines"))
        fig2.add_trace(go.Scatter(x=s2, y=v2, name="Référence", mode="lines"))
        fig2.update_layout(height=400, xaxis_title="Distance (m)", yaxis_title="Vitesse (m/s)")
        st.plotly_chart(fig2, width="stretch")

        st.caption("Note: la référence ici est le centerline; pour une comparaison pilote gagnant, branchez les meilleurs tours FastF1.")

    st.info("Étapes: 1) Collecte 2) Entraînement 3) Simulation. Utilisez votre venv.")


if __name__ == "__main__":
    main()
