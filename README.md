# F1-IA – Trajectoire optimale

Ce dépôt propose une ossature de pipeline pour:
- collecter des données F1 gratuites (FastF1 + option OpenF1/Jolpica),
- entraîner un modèle léger d'adhérence/capacités,
- optimiser une ligne de course (cvxpy) et simuler un temps au tour,
- visualiser la trajectoire et lancer une simulation depuis une UI Streamlit.

Important: aucune installation automatique n'est faite ici. Utilisez votre venv et installez `requirements.txt` vous‑même.

## Pré-requis
- Python 3.10+
- venv activé: `source venv/bin/activate`
- Installation manuelle: `pip install -r requirements.txt`

## Configuration
Modifiez `config.yaml` (saisons, circuits, sessions, paramètres d'optimisation). Le format par défaut de stockage est Parquet (rapide, colonne, compressé). Vous pouvez le basculer en CSV si nécessaire.

### Choix Parquet vs CSV
- Par défaut: `storage_format: parquet` dans `config.yaml` (recommandé: plus rapide et compact, surtout pour de gros volumes 2020+).
- Pour CSV: mettez `storage_format: csv` puis relancez la collecte.

### Circuits et calendrier
- Les noms dans `tracks` sont des requêtes souples (ex: "Monza", "Silverstone"). Le collecteur interroge le calendrier FastF1 et ne conserve que les événements réellement présents l'année donnée. Les événements annulés (ex: Monaco 2020) sont automatiquement ignorés.

## 1) Collecte

```
python -m collectors.fetch_resources --config config.yaml --workers 4
# Options: --workers 4 (parallélisme), --verbose (afficher logs FastF1)
# Variables env: F1IA_FETCH_WORKERS, F1IA_FETCH_VERBOSE=1
```
Les données sont écrites sous `data/raw/fastf1/<year_event_session>/...`.

Notes d'exécution habituelles:
- `DEFAULT CACHE ENABLED`: FastF1 utilise un cache local (macOS: `~/Library/Caches/fastf1`). Normal.
- `Failed to load result data from Ergast`: attendu sur les saisons récentes (Ergast figé). Sans impact pour nous.
- `Car data is incomplete!`: fréquent sur certaines sessions; la collecte continue avec ce qui est disponible.

## 2) Entraînement

```
python -m ml.training --config config.yaml
```
Produit `data/models/vehicle_limits.json`.

### Entraînement avancé (recommandé)

Calibre une enveloppe latérale (quantile régression) + un multiplicateur environnemental et alimente l'optimiseur itératif.

```
python -m ml.training_advanced --config config.yaml
```
Produit `data/models/lateral_envelope.json` et `data/models/vlim_model.joblib`.

## 3) UI + Simulation

```
streamlit run ui/app.py
```
Choisissez la session et cliquez sur « Lancer la simulation ».

## Notes techniques
- L'optimisation de ligne est une convexification pragmatique: offsets latéraux sur le centerline, objectif lissage + proxy de temps. Pour un modèle plus physique (vitesses couplées), enrichir `ml/models/line_optimizer.py`.
- Un optimiseur itératif « two‑step » (ligne ↔ vitesse) est utilisé si `advanced.use_advanced: true` dans `config.yaml`.
- La simulation utilise des passes avant/arrière pour respecter l'accélération longitudinale et les limites latérales.
- Pour comparer à un vainqueur réel, chargez la télémétrie du meilleur tour via FastF1 et ajoutez le tracé « référence » dans `ui/app.py`.

## Dépannage
- Correction d'événement hasardeuse ("Correcting user input …"): évitée. Le collecteur filtre désormais via le calendrier officiel. Si un `track` n'est pas trouvé pour une année, il est indiqué comme "Skip … non présent au calendrier".
- Absence de `get_pos_data` dans votre version de FastF1: géré. Fallback vers `get_position_data`, sinon reconstruction XY depuis la télémétrie des meilleurs tours.
- Cache FastF1 saturé: vous pouvez le vider en supprimant `~/Library/Caches/fastf1` (ou équivalent sous Linux/Windows).

## Apple Silicon (M4)
- `cvxpy` utilise des solveurs (OSQP/ECOS/SCS). En cas d'erreur de solveur manquant, installez explicitement: `pip install osqp ecos scs` dans votre venv.
- Les roues précompilées récentes fonctionnent généralement sans toolchain supplémentaire sur Apple Silicon.

## Dossiers de données et Git
- `data/` est créé vide avec un `.gitkeep` pour conserver l'arborescence. Les artefacts volumineux `data/raw` et `data/processed` sont ignorés par Git.
