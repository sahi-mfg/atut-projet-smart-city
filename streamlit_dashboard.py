"""
Dashboard Streamlit pour la prédiction de la qualité de l'air
Interface interactive pour visualiser les prédictions et explorer les données
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import xgboost as xgb

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="🌍 Prédiction Qualité de l'Air",
    page_icon="🌤️",
    layout="wide",
)

# Style CSS personnalisé
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Titre principal
st.markdown(
    '<h1 class="main-header">🌍 Prédiction de la Qualité de l\'Air</h1>',
    unsafe_allow_html=True,
)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_artifact():
    try:
        pipeline = joblib.load("pipeline.pkl")
        return pipeline  # dict: {"model": ..., "features": ...}
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        train = pd.read_csv("Train.csv")
        test = pd.read_csv("Test.csv")
        metadata = pd.read_csv("airqo_metadata.csv")
        return train, test, metadata
    except FileNotFoundError:
        st.error("Fichiers manquants")
        return None, None, None


# =========================
# PREPROCESSING (ALIGNÉ TRAINING)
# =========================
def preprocess_data(df, metadata_df):
    def replace_nan(x):
        if pd.isna(x) or str(x).strip() == "":
            return np.nan
        return float(str(x))

    features = ["temp", "precip", "rel_humidity", "wind_dir", "wind_spd", "atmos_press"]

    # Convert string -> list
    for col in features:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: [
                    replace_nan(v) for v in str(x).split(",") if str(v).strip() != ""
                ]
            )

    def clean(x):
        return [v for v in x if not pd.isna(v)] if isinstance(x, list) else []

    def stat(x, fn):
        return fn(x) if len(x) > 0 else 0

    # Feature engineering
    for col in features:
        if col in df.columns:
            df[col] = df[col].apply(clean)

            df[f"{col}_mean"] = df[col].apply(lambda x: stat(x, np.mean))
            df[f"{col}_std"] = df[col].apply(lambda x: stat(x, np.std))
            df[f"{col}_min"] = df[col].apply(lambda x: stat(x, np.min))
            df[f"{col}_max"] = df[col].apply(lambda x: stat(x, np.max))
            df[f"{col}_trend"] = df[col].apply(
                lambda x: x[-1] - x[0] if len(x) > 1 else 0
            )

    # Merge metadata
    if metadata_df is not None and "location" in df.columns:
        df = df.merge(metadata_df, on="location", how="left")
        df["pop_density"] = df["popn"] / (df["km2"] + 1e-8)

    # IMPORTANT : drop colonnes originales
    df = df.drop(columns=features, errors="ignore")

    return df


# =========================
# UI
# =========================
# st.title("🌍 Prédiction de la Qualité de l'Air")

page = st.sidebar.selectbox(
    "Navigation", ["Accueil", "Analyse des Données", "Prédiction"]
)

train, test, metadata = load_data()

# =========================
# ACCUEIL
# =========================
if page == "Accueil":
    st.markdown("""
    ## Objectif du Projet

    Ce dashboard permet de prédire la **qualité de l'air** dans différentes localisations
    en utilisant des données météorologiques et des caractéristiques urbaines.

    ### Fonctionnalités
    - **Analyse exploratoire** des données de qualité de l'air
    - **Prédictions interactives** avec le modèle gagnant (XGBoost)
    - **Visualisation** des tendances et patterns
    - **Comparaison** des performances des modèles

    ### Données Utilisées
    - Données météorologiques temporelles (température, précipitations, humidité, etc.)
    - Métadonnées des localisations (population, pratiques domestiques)
    - Mesures de qualité de l'air historiques

    ### Modèles Entraînés
    - LightGBM
    - XGBoost
    - CatBoost

    ### Meilleur modèle
    - XGBoost avec un RMSE de 26.9405 sur le test set
    """)

    # Métriques principales
    train, test, metadata = load_data()
    if train is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Données d'entraînement", f"{len(train):,}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🧪 Données de test", f"{len(test):,}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📍 Localisations", train["location"].nunique())
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_quality = train["target"].mean()
            st.metric("🌤️ Qualité moyenne", f"{avg_quality:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)


# Page d'analyse des données
elif page == "Analyse des Données":
    st.header("📈 Analyse Exploratoire des Données")

    train, test, metadata = load_data()
    if train is None:
        st.error("❌ Impossible de charger les données")
        st.stop()

    # Tabs pour différentes analyses
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📍 Localisations", "🌤️ Météo"])

    with tab1:
        st.subheader("Distribution de la Qualité de l'Air")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Histogramme
        sns.histplot(train["target"], bins=50, kde=True, ax=axes[0])
        axes[0].set_title("Distribution de la qualité de l'air")
        axes[0].set_xlabel("Valeur de la qualité de l'air")
        axes[0].set_ylabel("Fréquence")

        # Boxplot
        sns.boxplot(y=train["target"], ax=axes[1])
        axes[1].set_title("Boxplot de la qualité de l'air")
        axes[1].set_ylabel("Valeur de la qualité de l'air")

        plt.tight_layout()
        st.pyplot(fig)

        # Statistiques
        st.subheader("📋 Statistiques Descriptives")
        stats_df = train["target"].describe().reset_index()
        stats_df.columns = ["Métrique", "Valeur"]
        st.dataframe(stats_df, use_container_width=True)

    with tab2:
        st.subheader("Analyse par Localisation")

        # Distribution par localisation
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Countplot
        location_counts = train["location"].value_counts()
        axes[0].bar(location_counts.index, location_counts.values)
        axes[0].set_title("Nombre d'observations par localisation")
        axes[0].set_xlabel("Localisation")
        axes[0].set_ylabel("Nombre d'observations")
        axes[0].tick_params(axis="x", rotation=45)

        # Boxplot par localisation
        sns.boxplot(data=train, x="location", y="target", ax=axes[1])
        axes[1].set_title("Qualité de l'air par localisation")
        axes[1].set_xlabel("Localisation")
        axes[1].set_ylabel("Qualité de l'air")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        # Métadonnées des localisations
        if metadata is not None:
            st.subheader("🏙️ Métadonnées des Localisations")
            st.dataframe(metadata, use_container_width=True)

    with tab3:
        st.subheader("Analyse des Données Météorologiques")

        # Analyse des features météo
        weather_features = [
            "temp",
            "precip",
            "rel_humidity",
            "wind_dir",
            "wind_spd",
            "atmos_press",
        ]

        # Sélection d'un feature à analyser
        selected_feature = st.selectbox(
            "Sélectionnez une variable météorologique:", weather_features
        )

        if selected_feature in train.columns:
            st.write(f"Analyse de {selected_feature}")

            # Conversion pour l'analyse
            def replace_nan(x):
                if x == " ":
                    return np.nan
                else:
                    return float(x)

            feature_data = train[selected_feature].apply(
                lambda x: [replace_nan(X) for X in x.replace("nan", " ").split(",")]
            )

            # Statistiques sur les séries temporelles
            lengths = [len(x) for x in feature_data]
            means = [np.mean(x) for x in feature_data if len(x) > 0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Longueur des séries temporelles")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(lengths, bins=20, alpha=0.7)
                ax.set_xlabel("Longueur de la série")
                ax.set_ylabel("Fréquence")
                ax.set_title("Distribution des longueurs")
                st.pyplot(fig)

            with col2:
                st.subheader("Moyennes des valeurs")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(means, bins=20, alpha=0.7, color="orange")
                ax.set_xlabel("Moyenne")
                ax.set_ylabel("Fréquence")
                ax.set_title(f"Distribution des moyennes de {selected_feature}")
                st.pyplot(fig)

# =========================
# PRÉDICTION
# =========================
elif page == "Prédiction":
    st.header("Prédiction de la Qualité de l'Air")

    train, test, metadata = load_data()

    model_artifact = load_model_artifact()
    if model_artifact is None:
        st.stop()

    model = model_artifact["model"]
    features = model_artifact["features"]

    import xgboost as xgb

    def predict_model(model, X):
        if "xgboost" in str(type(model)).lower():
            return model.predict(xgb.DMatrix(X))
        return model.predict(X)

    prediction_mode = st.radio(
        "Mode de prédiction:",
        ["📊 Données de test", "🎯 Simulation utilisateur"],
    )

    # =========================
    # MODE TEST
    # =========================
    if prediction_mode == "📊 Données de test":
        sample_size = st.slider("Nombre d'échantillons:", 1, min(100, len(test)), 10)

        if st.button("🚀 Lancer les prédictions"):
            with st.spinner("Prédiction en cours..."):
                df = test.head(sample_size).copy()
                df_processed = preprocess_data(df, metadata)

                df_processed = df_processed.reindex(columns=features, fill_value=0)

                preds = predict_model(model, df_processed)

                results_df = pd.DataFrame(
                    {"Localisation": df["location"], "Prédiction": preds}
                )

                st.success("✅ Prédictions terminées")

                st.dataframe(results_df, use_container_width=True)

                fig = px.bar(
                    results_df,
                    x="Localisation",
                    y="Prédiction",
                    color="Prédiction",
                    color_continuous_scale="viridis",
                    title="Qualité de l'air prédite",
                )

                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # MODE MANUEL (UX PRO)
    # =========================
    else:
        st.subheader("🎯 Simulation personnalisée")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌤️ Météo")

            temp = st.text_input("Température (ex: 25,26,27)", "25,26,27")
            humidity = st.text_input("Humidité", "60,65,70")
            precip = st.text_input("Précipitations", "0,1,0")
            wind = st.text_input("Vent", "5,6,7")

        with col2:
            st.markdown("### Localisation")

            location = st.selectbox("Zone", train["location"].unique())

        if st.button("Prédire"):
            input_dict = {
                "location": location,
                "temp": temp,
                "precip": precip,
                "rel_humidity": humidity,
                "wind_dir": "10,20,30",
                "wind_spd": wind,
                "atmos_press": "1010,1012,1015",
            }

            df = pd.DataFrame([input_dict])

            df_processed = preprocess_data(df, metadata)
            df_processed = df_processed.reindex(columns=features, fill_value=0)

            pred = predict_model(model, df_processed)[0]
            pred = max(0, pred)

            # =========================
            # AFFICHAGE UX
            # =========================

            st.markdown("## Résultat")

            if pred < 50:
                st.success(f"Bonne qualité de l'air : {pred:.2f}")
            elif pred < 100:
                st.warning(f"Qualité modérée : {pred:.2f}")
            else:
                st.error(f"Mauvaise qualité : {pred:.2f}")

            # Gauge chart (visuel premium)
            fig = px.bar(
                x=["Qualité"],
                y=[pred],
                text=[f"{pred:.1f}"],
                title="Score de qualité de l'air",
            )
            st.plotly_chart(fig, use_container_width=True)
