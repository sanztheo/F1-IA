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

## 1) Collecte

```
python -m collectors.fetch_resources --config config.yaml
```
Les données sont écrites sous `data/raw/fastf1/<year_event_session>/...`.

## 2) Entraînement

```
python -m ml.training --config config.yaml
```
Produit `data/models/vehicle_limits.json`.

## 3) UI + Simulation

```
streamlit run ui/app.py
```
Choisissez la session et cliquez sur « Lancer la simulation ».

## Notes techniques
- L'optimisation de ligne est une convexification pragmatique: offsets latéraux sur le centerline, objectif lissage + proxy de temps. Pour un modèle plus physique (vitesses couplées), enrichir `ml/models/line_optimizer.py`.
- La simulation utilise des passes avant/arrière pour respecter l'accélération longitudinale et les limites latérales.
- Pour comparer à un vainqueur réel, chargez la télémétrie du meilleur tour via FastF1 et ajoutez le tracé « référence » dans `ui/app.py`.

