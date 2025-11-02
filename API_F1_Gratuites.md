# API F1 Gratuites pour Algorithme IA de Trajectoire Optimale

## 1. OpenF1 API

**Description**: API open-source offrant des données F1 en temps réel et historiques, incluant les positions des voitures avec échantillonnage à ~3,7 Hz.

**Données disponibles**:
- Positions des voitures (X, Y, Z) sur le circuit
- Télémétrie complète (vitesse, accélération, freinage, DRS, régime moteur)
- Temps au tour et par secteur
- Données historiques gratuites sans authentification

**Format**: JSON ou CSV

**Documentation**: [https://openf1.org](https://openf1.org)

**API Endpoint**: `https://api.openf1.org/v1/`

**Exemple d'utilisation**:
```bash
# Récupérer les positions d'une session
curl "https://api.openf1.org/v1/location?session_key=9161"

# Récupérer les tours d'un pilote
curl "https://api.openf1.org/v1/laps?session_key=9161&driver_number=63"
```

---

## 2. FastF1 (Bibliothèque Python)

**Description**: Bibliothèque Python puissante donnant accès aux données de télémétrie, positions, chronométrage et résultats F1.

**Données disponibles**:
- Télémétrie détaillée (vitesse, throttle, brake, DRS, gear)
- Positions des voitures (coordonnées X, Y, Z)
- Temps au tour et secteurs
- Données météo
- Stratégies pneus
- Calendrier et résultats

**Format**: Pandas DataFrames (facile à exporter en CSV pour IA)

**Documentation**: [https://docs.fastf1.dev](https://docs.fastf1.dev)

**Installation**:
```bash
pip install fastf1
```

**Exemple d'utilisation**:
```python
import fastf1

# Charger une session
session = fastf1.get_session(2024, 'Monaco', 'R')
session.load()

# Récupérer les positions et télémétrie
laps = session.laps
telemetry = session.car_data
```

---

## 3. Ergast F1 API

**Description**: API historique fournissant des données F1 depuis 1950 (résultats, classements, circuits).

**Données disponibles**:
- Résultats de courses historiques
- Informations sur les circuits (coordonnées, longueur)
- Classements pilotes et constructeurs
- Temps de qualifications et tours rapides

**Format**: JSON ou XML

**Documentation**: [http://ergast.com/mrd/](http://ergast.com/mrd/)

**API Endpoint**: `http://ergast.com/api/f1/`

**Exemple d'utilisation**:
```bash
# Récupérer les circuits d'une saison
curl "http://ergast.com/api/f1/2024/circuits.json"

# Récupérer les résultats d'une course
curl "http://ergast.com/api/f1/2024/1/results.json"
```

**Note**: FastF1 intègre également l'accès à l'API Ergast

---

## Recommandation pour votre projet IA

Pour créer un algorithme d'optimisation de trajectoire, je recommande:

1. **FastF1** comme source principale:
   - Données de position précises (X, Y, Z)
   - Télémétrie complète pour analyser les trajectoires gagnantes
   - Export facile en CSV/Pandas pour entraîner votre IA
   - Cache intégré pour performance

2. **OpenF1** en complément:
   - Données en temps réel si besoin
   - Format CSV direct
   - Bonne granularité des positions

3. **Ergast** pour contexte:
   - Informations sur les circuits
   - Données historiques complémentaires

## Workflow suggéré

```python
import fastf1
import pandas as pd

# Charger plusieurs courses sur un même circuit
sessions = []
for year in range(2020, 2025):
    session = fastf1.get_session(year, 'Monaco', 'R')
    session.load()

    # Récupérer positions et télémétrie des voitures gagnantes
    winner = session.laps.pick_fastest()
    telemetry = winner.get_telemetry()

    sessions.append(telemetry)

# Combiner les données
all_data = pd.concat(sessions)

# Exporter pour entraînement IA
all_data.to_csv('f1_trajectories_training.csv', index=False)
```

Ce dataset contiendra les trajectoires optimales par circuit pour entraîner votre modèle.
