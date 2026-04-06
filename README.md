# Projet Data Scientist: Prédiction de la Qualité de l'Air

## Description

Ce projet vise à prédire la qualité de l'air dans une ville intelligente (Smart City) en utilisant des techniques avancées de machine learning. Le projet utilise des données météorologiques temporelles et des métadonnées des localisations pour entraîner des modèles de prédiction.

## Objectifs

- **Feature Engineering**: Création de variables dérivées à partir de séries temporelles météorologiques
- **Modélisation**: Entraînement de modèles avancés (LightGBM, XGBoost, CatBoost)
- **Dashboard**: Interface interactive avec Streamlit pour visualiser les prédictions
- **Analyse**: Compréhension des facteurs influençant la qualité de l'air

## Dataset

### Fichiers disponibles
- `Train.csv`: Données d'entraînement avec variables météorologiques et qualité de l'air
- `Test.csv`: Données de test pour les prédictions
- `airqo_metadata.csv`: Métadonnées des localisations (population, pratiques domestiques, etc.)
- `sample_sub.csv`: Format de soumission

### Variables météorologiques
- `temp`: Température
- `precip`: Précipitations
- `rel_humidity`: Humidité relative
- `wind_dir`: Direction du vent
- `wind_spd`: Vitesse du vent
- `atmos_press`: Pression atmosphérique

### Variable cible
- `target`: Qualité de l'air (valeur numérique à prédire)

## Technologies Utilisées

- **Package Manager**: UV (gestion des dépendances Python)
- **Notebook**: Marimo (alternative moderne à Jupyter)
- **Dashboard**: Streamlit (interface web interactive)
- **Machine Learning**: LightGBM, XGBoost, CatBoost
- **Data Science**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly

## Installation et Démarrage

### 1. Cloner le repository
```bash
git clone <repository-url>
cd AirQo
```

### 2. Créer l'environnement virtuel avec UV
```bash
# Initialisation du projet UV
uv init

# Installation des dépendances
uv add pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn plotly streamlit marimo tqdm
```

### 3. Lancer le dashboard Streamlit
```bash
uv run streamlit run streamlit_dashboard.py
```

## Structure du Projet

```
AirQo/
├── air_quality_prediction.ipynb     # Notebook pour l'entraînement des modèles
├── streamlit_dashboard.py           # Dashboard Streamlit interactif
├── Train.csv                        # Données d'entraînement
├── Test.csv                         # Données de test
├── airqo_metadata.csv               # Métadonnées des localisations
├── pipeline.pkl                     # Modèle final sauvegardé
├── README.md                        # Documentation du projet
├── pyproject.toml                   # Configuration UV
└── uv.lock                          # Lockfile des dépendances
```

## Méthodologie

### 1. Feature Engineering

#### Features temporels
- **Statistiques agrégées**: min, max, mean, std, median, range
- **Tendances**: différence entre premier et dernier point
- **Momentum**: changement moyen entre points consécutifs
- **Volatilité**: écart type des différences
- **Ratio de valeurs manquantes**: proportion de données manquantes

#### Features de localisation
- **Densité de population**: population / surface
- **Pratiques domestiques**: ratios de cuisson (charbon, bois, déchets)
- **Score de pollution domestique**: combinaison pondérée des pratiques

### 2. Modèles de Machine Learning

#### LightGBM
- Optimisé pour les données tabulaires
- Gestion efficace des valeurs manquantes
- Rapide et performant

#### XGBoost
- Robuste et polyvalent
- Régularisation intégrée
- Excellent sur données structurées

#### CatBoost
- Gère automatiquement les features catégorielles
- Moins sensible au surapprentissage
- Performances stables

### 3. Validation Croisée

- **Stratégie**: K-Fold avec K=5
- **Métrique principale**: RMSE (Root Mean Square Error)
- **Métriques secondaires**: MAE

## Résultats Attendus

### Performance des modèles
- **LightGBM**: RMSE 25.53, R2 0.64
- **XGBoost**: RMSE 25.53, R2 0.64
- **CatBoost**: RMSE 26.44, R2 0.61
- **Ensemble**: RMSE 25.53, R2 0.64

### Améliorations des modèles
- Log-transformation de la target pour stabiliser la distribution
- Feature engineering enrichi avec médiane, skewness, kurtosis, tendances et moyennes glissantes
- Fusion des métadonnées de localisation pour enrichir les prédictions
- Validation croisée K-Fold (K=5) pour une meilleure estimation de généralisation
- Ensembling LightGBM + XGBoost + CatBoost pour la prédiction finale

### Features les plus importantes
1. Température moyenne
2. Humidité relative moyenne
3. Densité de population
4. Score de pollution domestique
5. Précipitations moyennes

## Dashboard Streamlit

Le dashboard propose plusieurs fonctionnalités:

### Pages disponibles
1. **Accueil**: Vue d'ensemble du projet et métriques principales
2. **Analyse des Données**: Exploration visuelle des données
3. **Prédiction**: Interface pour faire des prédictions
4. **Résultats**: Performance des modèles et insights

### Fonctionnalités interactives
- Visualisation des distributions
- Analyse par localisation
- Prédictions manuelles avec sliders
- Comparaison des modèles
- Graphiques dynamiques avec Plotly
- Résultats de prédiction colorés : vert=bonne qualité, jaune=qualité moyenne, rouge=mauvaise qualité

## Analyse Exploratoire

### Distribution de la qualité de l'air
- Analyse de la distribution de la variable cible
- Détection des outliers et valeurs extrêmes
- Comparaison par localisation

### Analyse temporelle
- Étude des séries météorologiques
- Détection de patterns saisonniers
- Analyse des tendances

### Analyse géographique
- Impact des localisations sur la qualité de l'air
- Corrélation avec les métadonnées urbaines
- Cartographie des zones à risque

## Améliorations Possibles

### Feature Engineering avancé
- Moyennes glissantes sur différentes fenêtres
- Détection de saisons et cycles
- Features de Fourier pour les patterns périodiques
- Variables d'interaction entre features

### Optimisation des modèles
- Hyperparameter tuning avec Optuna
- Ensembling des modèles (stacking, blending)
- Cross-validation stratifiée
- Calibration des prédictions

### Deep Learning
- Modèles LSTM/GRU pour les séries temporelles
- Transformers pour les séquences
- Autoencoders pour feature learning

### Déploiement
- API REST avec FastAPI
- Monitoring en continu
- Mise à jour automatique des modèles
- Alertes en temps réel

## Impact Sociétal

### Applications pratiques
- **Surveillance environnementale**: Monitoring continu de la qualité de l'air
- **Santé publique**: Alertes pour les populations sensibles
- **Urbanisme**: Planification basée sur les prédictions de pollution
- **Politiques publiques**: Décisions basées sur les données

### Bénéfices attendus
- Réduction des maladies respiratoires
- Amélioration de la qualité de vie urbaine
- Optimisation des infrastructures
- Sensibilisation citoyenne



---
