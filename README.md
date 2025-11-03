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

## Workflow Monaco (simplifié)

1) Préparer les maps (centerlines)

```
python scripts/fetch_maps.py --year 2022 --track "Circuit de Monaco" --track "Autodromo Nazionale Monza" --track "Circuit de Spa-Francorchamps"
```

Ou utiliser ton SVG Monaco et l’échelle réelle 3 337 m:
```
python scripts/use_svg.py --svg svg/monaco.svg --length 3337 --out data/tracks/monaco.npy
```

2) Entraîner (headless infini, Monaco SVG uniquement)
```
python scripts/train_monaco_headless.py
```

3) Visualiser la meilleure policy (Pygame, voiture unique)
```
python train_rl.py --track "Circuit de Monaco" --svg svg/monaco.svg --halfwidth 10
```

Le viewer utilise la best_policy enregistrée; si aucune existe, il initialise une policy par défaut.

- Contrôles (affichés en haut‑gauche dans la fenêtre):
  - Molette: zoom vers le curseur  •  Drag: déplacer
  - C / V: caméra suivre / libre
  - G / H: afficher / cacher les ghosts (points bleus)
  - (Rejoue uniquement; pour entraîner, utiliser le script headless ci‑dessus)
  - ESC: quitter
- HUD: génération, meilleur score, FPS, vitesse, progression

Reprise automatique
- À la fin de chaque génération, un checkpoint est sauvegardé: `data/evolution/<Circuit>_<Année>/rl_checkpoint.npz`.
- Si tu quittes (même à gen 1), relance `python train_rl.py` et la simulation reprend à la dernière génération sauvegardée (même population).

## Physique / échelles (F1 + Monaco)
- Voiture (approximations réalistes): L=5.6 m, W=2.0 m, empattement=3.6 m.
- Vitesse max ~90 m/s (324 km/h). 0–100 km/h ≈ 2.6 s. Freinage jusqu’à ~6 g (clamp lat/long).
- Accélération latérale avec aéro: `a_lat_max(v) ≈ (1.8 + 0.00058 v²) g`, bornée à 6.5 g.
- Monaco: longueur 3 337 m; largeur typique ~10 m (option `--halfwidth`).

Contrôles viewer: molette=zoom, drag=déplacer, ESC=quitter

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
- `scripts/train_monaco_headless.py` – entraînement Monaco (unique, infini).
- `train_rl.py` – viewer de la meilleure policy.
- `scripts/distill_teacher.py` – teacher CEM + distillation (optionnel).

## Git / données
- `data/` est gardé vide (`.gitkeep`). Les caches/exports lourds ne sont pas versionnés.
- Optionnel – Teacher CEM + distillation (accélère l’apprentissage d’une bonne policy)
```
# génère un dataset (obs→actions teacher) et ajuste un MLP
python scripts/distill_teacher.py --samples 50000 --epochs 5

# ensuite, visualise la meilleure policy distillée
python train_rl.py --track "Circuit de Monaco" --svg svg/monaco.svg --halfwidth 10
```
