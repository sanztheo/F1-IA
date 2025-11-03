# F1‑IA – Replay multi‑voitures + Trajectoire IA (Évolution par population)

Projet simplifié et recentré sur 3 circuits F1 pour:
- rejouer précisément jusqu’à 10 voitures « réelles » avec FastF1 (positions XY/temps);
- afficher la piste extraite d’OpenStreetMap et alignée aux données;
- explorer une trajectoire IA via évolution par population (200–400 voitures/génération), avec reprise sur checkpoint.

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

## Workflow sans Streamlit (Pygame)

1) Préparer les maps (centerlines)

```
python scripts/fetch_maps.py --year 2022 --track "Circuit de Monaco" --track "Autodromo Nazionale Monza" --track "Circuit de Spa-Francorchamps"
```

2) Lancer l’évolution (fenêtre Pygame)

```
python run_evolution.py --track "Circuit de Spa-Francorchamps" --year 2022 --pop 400 --gens 2 --nctrl 25 --autoplay
```

- Contrôles: 
  - Molette = zoom, clic gauche + glisser = déplacer
  - Espace = auto‑play (enchaîne les générations), N = exécuter le prochain lot de générations, Échap = quitter
  - Les checkpoints sont sauvegardés automatiquement: vous pouvez fermer et relancer, la progression est reprise

## Détails techniques
- Piste OSM: `osmnx` + `shapely`; extraction `leisure=racetrack` (fallback `highway=raceway`) puis simplification et rééchantillonnage.
- Alignement piste↔données: Procrustes 2D (similarité) sur le centerline dérivé des XY FastF1.
- Replay (facultatif): `replay/multicar_fastf1.py` reste disponible pour générer des données multi‑voitures (utile pour vérifier la piste).
- IA: offsets latéraux paramétrés par `n_ctrl_points` (splines implicites), évolution par population (sélection + mutations), évaluation via `simulation/lap_simulator.py`.

## Dépannage rapide
- OSMnx nécessite internet au premier fetch; réessayez si la requête échoue ponctuellement.
- FastF1 crée un cache (macOS: `~/Library/Caches/fastf1`).
- Si l’extraction OSM ne renvoie rien pour un circuit, l’UI utilisera le centerline dérivé des données FastF1.
- Les checkpoints sont stockés dans `data/evolution/<run_id>/pop_checkpoint.npz`. Le run_id est basé sur `<Circuit>_<Année>`.

## Arborescence utile
- `run_evolution.py` – boucle Pygame pour lancer et visualiser l’évolution.
- `replay/multicar_fastf1.py` – chargement et synchronisation de 10 voitures (replay et fallback géométrie).
- `tracks/fetch.py` – fetch + cache (data/tracks/*.npy) du centerline.
- `tracks/osm_tracks.py` – extraction/sampling OSM de la piste.
- `tracks/align.py` – alignement Procrustes 2D.
- `evolution/population.py` – moteur d’évolution population (checkpoint/reprise).
- `simulation/lap_simulator.py` – simulateur de tour (passes avant/arrière).

## Git / données
- `data/` est gardé vide (`.gitkeep`). Les caches/exports lourds ne sont pas versionnés.
