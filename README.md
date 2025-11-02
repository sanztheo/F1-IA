# F1‑IA – Replay multi‑voitures + Trajectoire IA (CMA‑ES)

Projet simplifié et recentré sur 3 circuits F1 pour:
- rejouer précisément jusqu’à 10 voitures « réelles » avec FastF1 (positions XY/temps);
- afficher la piste extraite d’OpenStreetMap et alignée aux données;
- explorer une trajectoire IA via optimisation évolutionnaire (CMA‑ES), budget 400 évaluations par circuit.

Important: aucune installation automatique. Utilisez votre venv et installez `requirements.txt` vous‑même.

## Pré‑requis
- Python 3.10+
- venv activé: `source venv/bin/activate`
- `pip install -r requirements.txt`

## Configuration (`config.yaml`)
- `tracks`: 3 circuits (Monaco, Monza, Spa) – modifiables.
- `reference_year`: 2022 (saison stable et complète).
- `evolution`: `evaluations_per_circuit: 400`, `n_ctrl_points: 25`.
- `optimization.track_half_width_m`: largeur piste (pour contraintes IA).

## Lancer l’UI

```
streamlit run ui/app.py
```

- Onglet « Replay réel »:
  - Circuit, année (par défaut 2022), session (Q ou R), puis « Charger le replay ».
  - L’animation Plotly affiche la piste (OSM alignée) et jusqu’à 10 voitures réelles synchronisées.
- Onglet « IA Evolution »:
  - « Calculer une trajectoire IA (aperçu) » lance CMA‑ES et trace la ligne IA vs centerline.

## Détails techniques
- Piste OSM: `osmnx` + `shapely`; extraction `leisure=racetrack` (fallback `highway=raceway`) puis simplification et rééchantillonnage.
- Alignement piste↔données: Procrustes 2D (similarité) sur le centerline dérivé des XY FastF1.
- Replay: `fastf1.get_pos_data()` → interpolation à ~20 Hz → animation Plotly (frames Play/Pause + slider natifs).
- IA: offsets latéraux paramétrés par `n_ctrl_points`, optimisation `cmaes`, évaluation via `simulation/lap_simulator.py`.

## Dépannage rapide
- OSMnx nécessite internet au premier fetch; réessayez si la requête échoue ponctuellement.
- FastF1 crée un cache (macOS: `~/Library/Caches/fastf1`).
- Si l’extraction OSM ne renvoie rien pour un circuit, l’UI utilisera le centerline dérivé des données FastF1.

## Arborescence utile
- `ui/app.py` – UI Streamlit (replay + IA évolution).
- `replay/multicar_fastf1.py` – chargement et synchronisation de 10 voitures.
- `tracks/osm_tracks.py` – extraction/sampling OSM de la piste.
- `tracks/align.py` – alignement Procrustes 2D.
- `simulation/lap_simulator.py` – simulateur de tour (passes avant/arrière).

## Git / données
- `data/` est gardé vide (`.gitkeep`). Les caches/exports lourds ne sont pas versionnés.
