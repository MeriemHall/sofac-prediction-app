import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="SOFAC - Prédiction Rendements 52-Semaines",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_historical_data():
    """Charger les données historiques Bank Al-Maghrib et HCP"""
    donnees_historiques = {
        '2020-03': {'taux_directeur': 2.00, 'inflation': 0.8, 'pib': -0.3, 'rendement_52s': 2.35},
        '2020-06': {'taux_directeur': 1.50, 'inflation': 0.7, 'pib': -15.8, 'rendement_52s': 2.00},
        '2020-09': {'taux_directeur': 1.50, 'inflation': 0.3, 'pib': -7.2, 'rendement_52s': 1.68},
        '2020-12': {'taux_directeur': 1.50, 'inflation': 0.3, 'pib': -4.8, 'rendement_52s': 1.93},
        '2021-03': {'taux_directeur': 1.50, 'inflation': 0.6, 'pib': 0.3, 'rendement_52s': 1.53},
        '2021-06': {'taux_directeur': 1.50, 'inflation': 1.1, 'pib': 13.9, 'rendement_52s': 1.52},
        '2021-12': {'taux_directeur': 1.50, 'inflation': 3.6, 'pib': 7.8, 'rendement_52s': 1.56},
        '2022-03': {'taux_directeur': 1.50, 'inflation': 4.8, 'pib': 2.1, 'rendement_52s': 1.61},
        '2022-06': {'taux_directeur': 1.50, 'inflation': 7.5, 'pib': 4.3, 'rendement_52s': 1.79},
        '2022-09': {'taux_directeur': 2.00, 'inflation': 7.4, 'pib': 3.7, 'rendement_52s': 2.18},
        '2023-03': {'taux_directeur': 3.00, 'inflation': 7.9, 'pib': 4.1, 'rendement_52s': 3.41},
        '2023-06': {'taux_directeur': 3.00, 'inflation': 5.3, 'pib': 2.6, 'rendement_52s': 3.34},
        '2023-09': {'taux_directeur': 3.00, 'inflation': 4.4, 'pib': 3.2, 'rendement_52s': 3.24},
        '2024-03': {'taux_directeur': 3.00, 'inflation': 2.1, 'pib': 3.5, 'rendement_52s': 2.94},
        '2024-09': {'taux_directeur': 2.75, 'inflation': 2.2, 'pib': 5.4, 'rendement_52s': 2.69},
        '2024-12': {'taux_directeur': 2.50, 'inflation': 2.3, 'pib': 4.6, 'rendement_52s': 2.53},
        '2025-03': {'taux_directeur': 2.25, 'inflation': 1.4, 'pib': 3.8, 'rendement_52s': 2.54},
        '2025-06': {'taux_directeur': 2.25, 'inflation': 1.3, 'pib': 3.7, 'rendement_52s': 1.75}
    }
    
    df_historique = []
    for date, valeurs in donnees_historiques.items():
        ligne = {'Date': date}
        ligne.update(valeurs)
        df_historique.append(ligne)
    
    return pd.DataFrame(df_historique)

@st.cache_data
def create_monthly_dataset(donnees_historiques):
    """Créer le jeu de données mensuelles par interpolation"""
    
    def interpolation_lineaire(date_debut, date_fin, valeur_debut, valeur_fin, date_cible):
        debut_num = date_debut.toordinal()
        fin_num = date_fin.toordinal()
        cible_num = date_cible.toordinal()
        
        if fin_num == debut_num:
            return valeur_debut
        
        progression = (cible_num - debut_num) / (fin_num - debut_num)
        return valeur_debut + progression * (valeur_fin - valeur_debut)
    
    date_debut = datetime(2020, 1, 1)
    date_fin = datetime(2025, 6, 30)
    
    donnees_mensuelles = []
    date_courante = date_debut
    
    # Conversion des données historiques
    dates_ancrage = {}
    for date_str, valeurs in donnees_historiques.items():
        date_obj = datetime.strptime(date_str + '-01', '%Y-%m-%d')
        dates_ancrage[date_obj] = valeurs
    
    while date_courante <= date_fin:
        date_str = date_courante.strftime('%Y-%m')
        est_ancrage = date_courante in dates_ancrage
        
        if est_ancrage:
            point_donnees = dates_ancrage[date_courante]
        else:
            # Interpolation
            dates_avant = [d for d in dates_ancrage.keys() if d <= date_courante]
            dates_apres = [d for d in dates_ancrage.keys() if d > date_courante]
            
            if dates_avant and dates_apres:
                date_avant = max(dates_avant)
                date_apres = min(dates_apres)
                donnees_avant = dates_ancrage[date_avant]
                donnees_apres = dates_ancrage[date_apres]
                
                point_donnees = {}
                for variable in ['taux_directeur', 'inflation', 'pib', 'rendement_52s']:
                    point_donnees[variable] = interpolation_lineaire(
                        date_avant, date_apres,
                        donnees_avant[variable], donnees_apres[variable],
                        date_courante
                    )
            elif dates_avant:
                date_avant = max(dates_avant)
                point_donnees = dates_ancrage[date_avant].copy()
            else:
                date_apres = min(dates_apres)
                point_donnees = dates_ancrage[date_apres].copy()
        
        donnees_mensuelles.append({
            'Date': date_str,
            'Taux_Directeur': point_donnees['taux_directeur'],
            'Inflation': point_donnees['inflation'],
            'Croissance_PIB': point_donnees['pib'],
            'Rendement_52s': point_donnees['rendement_52s'],
            'Est_Point_Ancrage': est_ancrage
        })
        
        # Passage au mois suivant
        if date_courante.month == 12:
            date_courante = date_courante.replace(year=date_courante.year + 1, month=1)
        else:
            date_courante = date_courante.replace(month=date_courante.month + 1)
    
    return pd.DataFrame(donnees_mensuelles)

def train_prediction_model(df_mensuel):
    """Entraîner le modèle de régression"""
    variables_explicatives = ['Taux_Directeur', 'Inflation', 'Croissance_PIB']
    X = df_mensuel[variables_explicatives]
    y = df_mensuel['Rendement_52s']
    
    modele = LinearRegression()
    modele.fit(X, y)
    
    y_pred = modele.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    scores_vc = cross_val_score(modele, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae_vc = -scores_vc.mean()
    
    return modele, r2, mae, rmse, mae_vc

@st.cache_data
def create_economic_scenarios():
    """Créer les scénarios économiques pour 2025-2026"""
    
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2026, 12, 31)
    
    dates_quotidiennes = []
    date_courante = date_debut
    
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
    # Décisions de politique monétaire
    decisions_politiques = {
        'Conservateur': {
            '2025-06': 2.25, '2025-09': 2.25, '2025-12': 2.00,
            '2026-03': 1.75, '2026-06': 1.75, '2026-09': 1.50, '2026-12': 1.50
        },
        'Cas_de_Base': {
            '2025-06': 2.25, '2025-09': 2.00, '2025-12': 1.75,
            '2026-03': 1.50, '2026-06': 1.50, '2026-09': 1.25, '2026-12': 1.25
        },
        'Optimiste': {
            '2025-06': 2.25, '2025-09': 1.75, '2025-12': 1.50,
            '2026-03': 1.25, '2026-06': 1.00, '2026-09': 1.00, '2026-12': 1.00
        }
    }
    
    scenarios = {}
    
    for nom_scenario in ['Conservateur', 'Cas_de_Base', 'Optimiste']:
        donnees_scenario = []
        taux_politiques = decisions_politiques[nom_scenario]
        
        for i, date in enumerate(dates_quotidiennes):
            jours_ahead = i + 1
            
            # Taux directeur selon calendrier
            date_str = date.strftime('%Y-%m')
            taux_directeur = 2.25
            for date_politique, taux in sorted(taux_politiques.items()):
                if date_str >= date_politique:
                    taux_directeur = taux
            
            # Variation de marché
            np.random.seed(hash(date.strftime('%Y-%m-%d')) % 2**32)
            variation_marche = np.random.normal(0, 0.02)
            
            # Scénarios d'inflation et PIB
            mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
            
            if nom_scenario == 'Conservateur':
                inflation_base = 1.4 + 0.5 * np.exp(-mois_depuis_debut / 18) + 0.2 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 - 0.5 * (mois_depuis_debut / 18) + 0.4 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            elif nom_scenario == 'Cas_de_Base':
                inflation_base = 1.4 + 0.3 * np.exp(-mois_depuis_debut / 12) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 - 0.2 * (mois_depuis_debut / 18) + 0.5 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            else:  # Optimiste
                inflation_base = 1.4 - 0.2 * (mois_depuis_debut / 18) + 0.1 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 + 0.1 * (mois_depuis_debut / 18) + 0.6 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            
            inflation = max(0.0, min(5.0, inflation_base + np.random.normal(0, 0.01)))
            pib = max(-2.0, min(6.0, pib_base + np.random.normal(0, 0.05)))
            
            donnees_scenario.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Taux_Directeur': taux_directeur,
                'Inflation': inflation,
                'Croissance_PIB': pib,
                'Jours_Ahead': jours_ahead,
                'Jour_Semaine': date.strftime('%A'),
                'Est_Weekend': date.weekday() >= 5
            })
        
        scenarios[nom_scenario] = pd.DataFrame(donnees_scenario)
    
    return scenarios

def generate_predictions(scenarios, modele, mae_historique):
    """Générer les prédictions avec correction de continuité"""
    
    rendement_juin_reel = 1.75
    predictions = {}
    
    for nom_scenario, scenario_df in scenarios.items():
        # Variables explicatives
        X_futur = scenario_df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
        
        # Prédictions brutes
        rendements_bruts = modele.predict(X_futur)
        
        # Correction de continuité
        juillet_1_brut = rendements_bruts[0]
        discontinuite = juillet_1_brut - rendement_juin_reel
        
        rendements_lisses = rendements_bruts.copy()
        for i in range(len(rendements_lisses)):
            jours_depuis_debut = i + 1
            if jours_depuis_debut <= 30:
                facteur_decroissance = np.exp(-jours_depuis_debut / 15)
            elif jours_depuis_debut <= 90:
                facteur_decroissance = np.exp(-30 / 15) * np.exp(-(jours_depuis_debut - 30) / 30)
            else:
                facteur_decroissance = 0
            
            ajustement = discontinuite * facteur_decroissance
            rendements_lisses[i] = rendements_bruts[i] - ajustement
        
        # Ajustements de scénario
        ajustements = []
        for i, ligne in scenario_df.iterrows():
            ajustement = 0
            
            if nom_scenario == 'Conservateur':
                ajustement += 0.1
            elif nom_scenario == 'Optimiste':
                ajustement -= 0.05
            
            # Incertitude temporelle
            jours_ahead = ligne['Jours_Ahead']
            incertitude = (jours_ahead / 365) * 0.1
            
            if nom_scenario == 'Conservateur':
                ajustement += incertitude
            elif nom_scenario == 'Optimiste':
                ajustement -= incertitude * 0.5
            
            # Effets jour de semaine
            effets_jours = {
                'Monday': 0.01, 'Tuesday': 0.00, 'Wednesday': -0.01,
                'Thursday': 0.00, 'Friday': 0.02, 'Saturday': -0.01, 'Sunday': -0.01
            }
            ajustement += effets_jours.get(ligne['Jour_Semaine'], 0)
            
            ajustements.append(ajustement)
        
        # Rendements finaux
        rendements_finaux = rendements_lisses + np.array(ajustements)
        rendements_finaux = np.clip(rendements_finaux, 0.1, 8.0)
        
        # DataFrame de résultats
        scenario_df_copie = scenario_df.copy()
        scenario_df_copie['Rendement_Predit'] = rendements_finaux
        scenario_df_copie['Scenario'] = nom_scenario
        
        # Intervalles de confiance
        for i, ligne in scenario_df_copie.iterrows():
            jours_ahead = ligne['Jours_Ahead']
            intervalle_base = mae_historique
            facteur_temps = 1 + (jours_ahead / 365) * 0.2
            intervalle_ajuste = intervalle_base * facteur_temps
            
            if ligne['Est_Weekend']:
                intervalle_ajuste *= 1.1
            
            ic_95 = intervalle_ajuste * 2
            
            scenario_df_copie.loc[i, 'Borne_Inf_95'] = max(0.1, ligne['Rendement_Predit'] - ic_95)
            scenario_df_copie.loc[i, 'Borne_Sup_95'] = min(8.0, ligne['Rendement_Predit'] + ic_95)
        
        predictions[nom_scenario] = scenario_df_copie
    
    return predictions

def generate_recommendations(predictions):
    """Générer les recommandations métier"""
    
    rendement_actuel = 2.54
    recommandations = {}
    
    for nom_scenario, pred_df in predictions.items():
        rendement_futur_moyen = pred_df['Rendement_Predit'].mean()
        changement_rendement = rendement_futur_moyen - rendement_actuel
        volatilite = pred_df['Rendement_Predit'].std()
        
        # Logique emprunteur corrigée
        if changement_rendement > 0.3:
            recommandation = "TAUX FIXE"
            raison = f"Rendements attendus en hausse de {changement_rendement:.2f}% en moyenne. Bloquer les taux actuels avant que les coûts d'emprunt n'augmentent."
        elif changement_rendement < -0.3:
            recommandation = "TAUX VARIABLE"
            raison = f"Rendements attendus en baisse de {abs(changement_rendement):.2f}% en moyenne. Utiliser des taux variables pour profiter de la diminution des coûts d'emprunt."
        else:
            recommandation = "STRATÉGIE FLEXIBLE"
            raison = f"Rendements relativement stables (±{abs(changement_rendement):.2f}%). Approche mixte selon les besoins."
        
        # Niveau de risque
        if volatilite < 0.2:
            niveau_risque = "FAIBLE"
        elif volatilite < 0.4:
            niveau_risque = "MOYEN"
        else:
            niveau_risque = "ÉLEVÉ"
        
        recommandations[nom_scenario] = {
            'recommandation': recommandation,
            'raison': raison,
            'niveau_risque': niveau_risque,
            'rendement_futur_moyen': rendement_futur_moyen,
            'changement_rendement': changement_rendement,
            'volatilite': volatilite
        }
    
    return recommandations

def main():
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🇲🇦 SOFAC - Modèle de Prédiction des Rendements 52-Semaines</h1>
        <p>Système d'aide à la décision pour optimiser les coûts de financement</p>
        <p><strong>Horizon:</strong> Juillet 2025 - Décembre 2026 | <strong>Perspective:</strong> Emprunteur</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("📊 Informations du Modèle")
        
        # Initialiser les données dans session_state si pas déjà fait
        if 'data_loaded' not in st.session_state:
            with st.spinner("Chargement des données..."):
                # Données historiques
                donnees_hist_dict = {
                    '2020-03': {'taux_directeur': 2.00, 'inflation': 0.8, 'pib': -0.3, 'rendement_52s': 2.35},
                    '2020-06': {'taux_directeur': 1.50, 'inflation': 0.7, 'pib': -15.8, 'rendement_52s': 2.00},
                    '2020-09': {'taux_directeur': 1.50, 'inflation': 0.3, 'pib': -7.2, 'rendement_52s': 1.68},
                    '2020-12': {'taux_directeur': 1.50, 'inflation': 0.3, 'pib': -4.8, 'rendement_52s': 1.93},
                    '2021-03': {'taux_directeur': 1.50, 'inflation': 0.6, 'pib': 0.3, 'rendement_52s': 1.53},
                    '2021-06': {'taux_directeur': 1.50, 'inflation': 1.1, 'pib': 13.9, 'rendement_52s': 1.52},
                    '2021-12': {'taux_directeur': 1.50, 'inflation': 3.6, 'pib': 7.8, 'rendement_52s': 1.56},
                    '2022-03': {'taux_directeur': 1.50, 'inflation': 4.8, 'pib': 2.1, 'rendement_52s': 1.61},
                    '2022-06': {'taux_directeur': 1.50, 'inflation': 7.5, 'pib': 4.3, 'rendement_52s': 1.79},
                    '2022-09': {'taux_directeur': 2.00, 'inflation': 7.4, 'pib': 3.7, 'rendement_52s': 2.18},
                    '2023-03': {'taux_directeur': 3.00, 'inflation': 7.9, 'pib': 4.1, 'rendement_52s': 3.41},
                    '2023-06': {'taux_directeur': 3.00, 'inflation': 5.3, 'pib': 2.6, 'rendement_52s': 3.34},
                    '2023-09': {'taux_directeur': 3.00, 'inflation': 4.4, 'pib': 3.2, 'rendement_52s': 3.24},
                    '2024-03': {'taux_directeur': 3.00, 'inflation': 2.1, 'pib': 3.5, 'rendement_52s': 2.94},
                    '2024-09': {'taux_directeur': 2.75, 'inflation': 2.2, 'pib': 5.4, 'rendement_52s': 2.69},
                    '2024-12': {'taux_directeur': 2.50, 'inflation': 2.3, 'pib': 4.6, 'rendement_52s': 2.53},
                    '2025-03': {'taux_directeur': 2.25, 'inflation': 1.4, 'pib': 3.8, 'rendement_52s': 2.54},
                    '2025-06': {'taux_directeur': 2.25, 'inflation': 1.3, 'pib': 3.7, 'rendement_52s': 1.75}
                }
                
                st.session_state.df_mensuel = create_monthly_dataset(donnees_hist_dict)
                st.session_state.modele, st.session_state.r2, st.session_state.mae, st.session_state.rmse, st.session_state.mae_vc = train_prediction_model(st.session_state.df_mensuel)
                st.session_state.scenarios = create_economic_scenarios()
                st.session_state.predictions = generate_predictions(st.session_state.scenarios, st.session_state.modele, st.session_state.mae)
                st.session_state.recommandations = generate_recommendations(st.session_state.predictions)
                st.session_state.data_loaded = True
        
        st.success("✅ Données chargées!")
        
        # Métriques du modèle
        st.subheader("🎯 Performance du Modèle")
        st.metric("R² Score", f"{st.session_state.r2:.1%}", help="Pourcentage de variance expliquée")
        st.metric("Précision", f"±{st.session_state.mae:.2f}%", help="Erreur absolue moyenne")
        st.metric("Validation Croisée", f"±{st.session_state.mae_vc:.2f}%", help="Erreur en validation croisée")
        
        # Situation actuelle
        st.subheader("📊 Situation Actuelle")
        st.metric("Rendement Actuel", "2.54%", help="Mars 2025 - Bank Al-Maghrib")
        st.metric("Taux Directeur", "2.25%", help="Juin 2025 - Bank Al-Maghrib")
        
    # Contenu principal
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue d'Ensemble", "🔮 Prédictions Détaillées", "💼 Recommandations", "📊 Analyses"])
    
    with tab1:
        st.header("📈 Vue d'Ensemble des Prédictions")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        cas_de_base = st.session_state.predictions['Cas_de_Base']
        rendement_moyen = cas_de_base['Rendement_Predit'].mean()
        changement = rendement_moyen - 2.54
        volatilite = cas_de_base['Rendement_Predit'].std()
        
        with col1:
            st.metric(
                "Rendement Moyen Prédit", 
                f"{rendement_moyen:.2f}%",
                delta=f"{changement:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Changement vs Actuel", 
                f"{changement:+.2f}%",
                delta="Baisse" if changement < 0 else "Hausse"
            )
        
        with col3:
            st.metric(
                "Volatilité", 
                f"{volatilite:.2f}%",
                help="Écart-type des prédictions"
            )
        
        with col4:
            jours_totaux = len(cas_de_base)
            st.metric(
                "Horizon de Prédiction", 
                f"{jours_totaux} jours",
                delta="18 mois"
            )
        
        # Graphique principal - Vue d'ensemble
        st.subheader("📊 Évolution des Rendements par Scénario")
        
        fig_overview = go.Figure()
        
        # Données historiques récentes
        df_recent = st.session_state.df_mensuel.tail(6)
        fig_overview.add_trace(
            go.Scatter(
                x=df_recent['Date'],
                y=df_recent['Rendement_52s'],
                mode='lines+markers',
                name='Historique 2025',
                line=dict(color='#60A5FA', width=4),
                marker=dict(size=8)
            )
        )
        
        # Prédictions par scénario (échantillon hebdomadaire)
        couleurs = {'Conservateur': '#FF6B6B', 'Cas_de_Base': '#4ECDC4', 'Optimiste': '#45B7D1'}
        
        for nom_scenario, pred_df in st.session_state.predictions.items():
            # Échantillon hebdomadaire pour clarté
            donnees_hebdo = pred_df[::7]
            
            fig_overview.add_trace(
                go.Scatter(
                    x=donnees_hebdo['Date'],
                    y=donnees_hebdo['Rendement_Predit'],
                    mode='lines+markers',
                    name=f'{nom_scenario}',
                    line=dict(color=couleurs[nom_scenario], width=3),
                    marker=dict(size=6)
                )
            )
        
        fig_overview.update_layout(
            title="Évolution des Rendements 52-Semaines: Historique et Prédictions",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # Résumé des recommandations
        st.subheader("🎯 Recommandations Rapides")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (scenario, rec) in enumerate(st.session_state.recommandations.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{scenario}</h4>
                    <p><strong>{rec['recommandation']}</strong></p>
                    <p>Changement: {rec['changement_rendement']:+.2f}%</p>
                    <p>Risque: {rec['niveau_risque']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("🔮 Prédictions Détaillées")
        
        # Sélection du scénario
        scenario_selectionne = st.selectbox(
            "Choisissez un scénario:",
            options=['Cas_de_Base', 'Conservateur', 'Optimiste'],
            index=0,
            help="Sélectionnez le scénario économique à analyser"
        )
        
        pred_scenario = st.session_state.predictions[scenario_selectionne]
        
        # Statistiques du scénario
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rendement Moyen", f"{pred_scenario['Rendement_Predit'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{pred_scenario['Rendement_Predit'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{pred_scenario['Rendement_Predit'].max():.2f}%")
        with col4:
            st.metric("Écart-type", f"{pred_scenario['Rendement_Predit'].std():.2f}%")
        
        # Graphique détaillé avec intervalles de confiance
        st.subheader(f"📊 Prédictions Quotidiennes - Scénario {scenario_selectionne}")
        
        # Échantillon pour affichage (tous les 3 jours pour lisibilité)
        donnees_affichage = pred_scenario[::3]
        
        fig_detail = go.Figure()
        
        couleurs = {'Conservateur': '#FF6B6B', 'Cas_de_Base': '#4ECDC4', 'Optimiste': '#45B7D1'}
        
        # Bandes de confiance
        fig_detail.add_trace(
            go.Scatter(
                x=list(donnees_affichage['Date']) + list(donnees_affichage['Date'][::-1]),
                y=list(donnees_affichage['Borne_Sup_95']) + list(donnees_affichage['Borne_Inf_95'][::-1]),
                fill='toself',
                fillcolor='rgba(74, 179, 209, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalle de Confiance 95%',
                showlegend=True
            )
        )
        
        # Prédictions principales
        fig_detail.add_trace(
            go.Scatter(
                x=donnees_affichage['Date'],
                y=donnees_affichage['Rendement_Predit'],
                mode='lines+markers',
                name='Prédiction',
                line=dict(color=couleurs[scenario_selectionne], width=3),
                marker=dict(size=4)
            )
        )
        
        fig_detail.update_layout(
            title=f"Prédictions Détaillées - {scenario_selectionne}",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Analyse mensuelle
        st.subheader("📅 Analyse Mensuelle")
        
        # Groupement par mois
        pred_scenario['Annee_Mois'] = pd.to_datetime(pred_scenario['Date']).dt.to_period('M')
        analyse_mensuelle = pred_scenario.groupby('Annee_Mois').agg({
            'Rendement_Predit': ['mean', 'min', 'max', 'std'],
            'Taux_Directeur': 'mean',
            'Inflation': 'mean',
            'Croissance_PIB': 'mean'
        }).round(3)
        
        analyse_mensuelle.columns = ['Rend_Moy', 'Rend_Min', 'Rend_Max', 'Volatilité', 
                                   'Taux_Dir', 'Inflation', 'PIB']
        analyse_mensuelle_display = analyse_mensuelle.reset_index()
        analyse_mensuelle_display['Annee_Mois'] = analyse_mensuelle_display['Annee_Mois'].astype(str)
        
        st.dataframe(
            analyse_mensuelle_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Export des données
        if st.button("📥 Télécharger les Prédictions"):
            csv = pred_scenario.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"sofac_predictions_{scenario_selectionne.lower()}.csv",
                mime="text/csv"
            ),
                    marker=dict(size=6)
                )
            )
        
        fig_overview.update_layout(
            title="Évolution des Rendements 52-Semaines: Historique et Prédictions",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # Résumé des recommandations
        st.subheader("🎯 Recommandations Rapides")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (scenario, rec) in enumerate(recommandations.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{scenario}</h4>
                    <p><strong>{rec['recommandation']}</strong></p>
                    <p>Changement: {rec['changement_rendement']:+.2f}%</p>
                    <p>Risque: {rec['niveau_risque']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("🔮 Prédictions Détaillées")
        
        # Sélection du scénario
        scenario_selectionne = st.selectbox(
            "Choisissez un scénario:",
            options=['Cas_de_Base', 'Conservateur', 'Optimiste'],
            index=0,
            help="Sélectionnez le scénario économique à analyser"
        )
        
        pred_scenario = st.session_state.predictions[scenario_selectionne]
        
        # Statistiques du scénario
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rendement Moyen", f"{pred_scenario['Rendement_Predit'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{pred_scenario['Rendement_Predit'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{pred_scenario['Rendement_Predit'].max():.2f}%")
        with col4:
            st.metric("Écart-type", f"{pred_scenario['Rendement_Predit'].std():.2f}%")
        
        # Graphique détaillé avec intervalles de confiance
        st.subheader(f"📊 Prédictions Quotidiennes - Scénario {scenario_selectionne}")
        
        # Échantillon pour affichage (tous les 3 jours pour lisibilité)
        donnees_affichage = pred_scenario[::3]
        
        fig_detail = go.Figure()
        
        couleurs = {'Conservateur': '#FF6B6B', 'Cas_de_Base': '#4ECDC4', 'Optimiste': '#45B7D1'}
        
        # Bandes de confiance
        fig_detail.add_trace(
            go.Scatter(
                x=list(donnees_affichage['Date']) + list(donnees_affichage['Date'][::-1]),
                y=list(donnees_affichage['Borne_Sup_95']) + list(donnees_affichage['Borne_Inf_95'][::-1]),
                fill='toself',
                fillcolor='rgba(74, 179, 209, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalle de Confiance 95%',
                showlegend=True
            )
        )
        
        # Prédictions principales
        fig_detail.add_trace(
            go.Scatter(
                x=donnees_affichage['Date'],
                y=donnees_affichage['Rendement_Predit'],
                mode='lines+markers',
                name='Prédiction',
                line=dict(color=couleurs[scenario_selectionne], width=3),
                marker=dict(size=4)
            )
        )
        
        fig_detail.update_layout(
            title=f"Prédictions Détaillées - {scenario_selectionne}",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Analyse mensuelle
        st.subheader("📅 Analyse Mensuelle")
        
        # Groupement par mois
        pred_scenario['Annee_Mois'] = pd.to_datetime(pred_scenario['Date']).dt.to_period('M')
        analyse_mensuelle = pred_scenario.groupby('Annee_Mois').agg({
            'Rendement_Predit': ['mean', 'min', 'max', 'std'],
            'Taux_Directeur': 'mean',
            'Inflation': 'mean',
            'Croissance_PIB': 'mean'
        }).round(3)
        
        analyse_mensuelle.columns = ['Rend_Moy', 'Rend_Min', 'Rend_Max', 'Volatilité', 
                                   'Taux_Dir', 'Inflation', 'PIB']
        analyse_mensuelle_display = analyse_mensuelle.reset_index()
        analyse_mensuelle_display['Annee_Mois'] = analyse_mensuelle_display['Annee_Mois'].astype(str)
        
        st.dataframe(
            analyse_mensuelle_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Export des données
        if st.button("📥 Télécharger les Prédictions"):
            csv = pred_scenario.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"sofac_predictions_{scenario_selectionne.lower()}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("💼 Recommandations Stratégiques")
        
        # Recommandation globale
        liste_recommandations = [rec['recommandation'] for rec in st.session_state.recommandations.values()]
        
        if liste_recommandations.count('TAUX VARIABLE') >= 2:
            strategie_globale = "TAUX VARIABLE"
            raison_globale = "Majorité des scénarios montrent des taux en baisse - profiter de la diminution des coûts"
            couleur_globale = "#28a745"
        elif liste_recommandations.count('TAUX FIXE') >= 2:
            strategie_globale = "TAUX FIXE" 
            raison_globale = "Majorité des scénarios montrent des taux en hausse - se protéger contre l'augmentation des coûts"
            couleur_globale = "#dc3545"
        else:
            strategie_globale = "STRATÉGIE FLEXIBLE"
            raison_globale = "Signaux mixtes suggèrent une approche diversifiée"
            couleur_globale = "#ffc107"
        
        st.markdown(f"""
        <div class="recommendation-box" style="background: linear-gradient(135deg, {couleur_globale} 0%, {couleur_globale}AA 100%);">
            <h2>🏆 RECOMMANDATION GLOBALE SOFAC</h2>
            <h3>{strategie_globale}</h3>
            <p>{raison_globale}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Détail par scénario
        st.subheader("📊 Analyse Détaillée par Scénario")
        
        couleurs = {'Conservateur': '#FF6B6B', 'Cas_de_Base': '#4ECDC4', 'Optimiste': '#45B7D1'}
        
        for nom_scenario, rec in st.session_state.recommandations.items():
            pred_df = st.session_state.predictions[nom_scenario]
            
            with st.expander(f"📈 Scénario {nom_scenario}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Recommandation:** {rec['recommandation']}
                    
                    **Justification:** {rec['raison']}
                    
                    **Métriques:**
                    - Rendement moyen prédit: {rec['rendement_futur_moyen']:.2f}%
                    - Changement vs actuel: {rec['changement_rendement']:+.2f}%
                    - Volatilité: {rec['volatilite']:.2f}%
                    - Niveau de risque: {rec['niveau_risque']}
                    """)
                
                with col2:
                    # Mini graphique
                    fig_mini = go.Figure()
                    echantillon_mini = pred_df[::30]  # Un point par mois
                    
                    fig_mini.add_trace(
                        go.Scatter(
                            x=echantillon_mini['Date'],
                            y=echantillon_mini['Rendement_Predit'],
                            mode='lines+markers',
                            name=nom_scenario,
                            line=dict(color=couleurs[nom_scenario], width=2)
                        )
                    )
                    
                    fig_mini.update_layout(
                        height=200,
                        showlegend=False,
                        template="plotly_white",
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    
                    st.plotly_chart(fig_mini, use_container_width=True)
        
        # Impact financier
        st.subheader("💰 Impact Financier Estimé")
        
        changement_cas_base = st.session_state.recommandations['Cas_de_Base']['changement_rendement']
        
        montant_emprunt = st.slider(
            "Montant d'emprunt (millions MAD):",
            min_value=1,
            max_value=100,
            value=10,
            step=1
        )
        
        if abs(changement_cas_base) > 0.3:
            if changement_cas_base < 0:  # Baisse
                economies = abs(changement_cas_base) * montant_emprunt * 1_000_000 / 100
                st.success(f"""
                💰 **Économies Potentielles avec TAUX VARIABLE:**
                
                - {economies:,.0f} MAD/an
                - Basé sur une baisse attendue de {abs(changement_cas_base):.2f}%
                - Pour un emprunt de {montant_emprunt}M MAD
                """)
            else:  # Hausse
                cout_evite = changement_cas_base * montant_emprunt * 1_000_000 / 100
                st.warning(f"""
                💰 **Coûts Évités avec TAUX FIXE:**
                
                - {cout_evite:,.0f} MAD/an
                - Basé sur une hausse attendue de {changement_cas_base:.2f}%
                - Pour un emprunt de {montant_emprunt}M MAD
                """)
        else:
            st.info(f"""
            💰 **Impact Financier Limité:**
            
            - Taux relativement stables (±{abs(changement_cas_base):.2f}%)
            - Approche flexible recommandée
            """)
    
    with tab4:
        st.header("📊 Analyses Approfondies")
        
        # Comparaison des scénarios
        st.subheader("⚖️ Comparaison des Scénarios")
        
        # Tableau comparatif
        donnees_comparaison = []
        for nom_scenario, pred_df in st.session_state.predictions.items():
            donnees_comparaison.append({
                'Scénario': nom_scenario,
                'Rendement_Moyen': f"{pred_df['Rendement_Predit'].mean():.2f}%",
                'Rendement_Min': f"{pred_df['Rendement_Predit'].min():.2f}%",
                'Rendement_Max': f"{pred_df['Rendement_Predit'].max():.2f}%",
                'Volatilité': f"{pred_df['Rendement_Predit'].std():.2f}%",
                'Recommandation': st.session_state.recommandations[nom_scenario]['recommandation'],
                'Niveau_Risque': st.session_state.recommandations[nom_scenario]['niveau_risque']
            })
        
        df_comparaison = pd.DataFrame(donnees_comparaison)
        st.dataframe(df_comparaison, use_container_width=True, hide_index=True)
        
        # Analyse de sensibilité
        st.subheader("🎯 Analyse de Sensibilité")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Impact des Variables Économiques:**")
            
            # Coefficients du modèle
            coefficients = {
                'Taux Directeur': st.session_state.modele.coef_[0],
                'Inflation': st.session_state.modele.coef_[1], 
                'Croissance PIB': st.session_state.modele.coef_[2]
            }
            
            fig_coef = go.Figure(data=[
                go.Bar(
                    x=list(coefficients.keys()),
                    y=list(coefficients.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )
            ])
            
            fig_coef.update_layout(
                title="Sensibilité du Modèle",
                yaxis_title="Impact par Point (%)",
                height=300,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_coef, use_container_width=True)
        
        with col2:
            st.markdown("**Distribution des Prédictions:**")
            
            cas_base_data = st.session_state.predictions['Cas_de_Base']['Rendement_Predit']
            
            fig_hist = go.Figure(data=[
                go.Histogram(
                    x=cas_base_data,
                    nbinsx=30,
                    marker_color='#4ECDC4',
                    opacity=0.7
                )
            ])
            
            fig_hist.update_layout(
                title="Distribution des Rendements",
                xaxis_title="Rendement (%)",
                yaxis_title="Fréquence",
                height=300,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Évolution temporelle
        st.subheader("📈 Évolution Temporelle Détaillée")
        
        # Graphique avec sous-graphiques
        fig_evolution = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Rendements Prédits', 'Taux Directeur', 'Inflation', 'Croissance PIB'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        cas_base_sample = st.session_state.predictions['Cas_de_Base'][::15]  # Échantillon bi-hebdomadaire
        
        # Rendements
        fig_evolution.add_trace(
            go.Scatter(x=cas_base_sample['Date'], y=cas_base_sample['Rendement_Predit'],
                      mode='lines', name='Rendement', line=dict(color='#4ECDC4')),
            row=1, col=1
        )
        
        # Taux directeur
        fig_evolution.add_trace(
            go.Scatter(x=cas_base_sample['Date'], y=cas_base_sample['Taux_Directeur'],
                      mode='lines+markers', name='Taux Directeur', line=dict(color='#FF6B6B')),
            row=1, col=2
        )
        
        # Inflation
        fig_evolution.add_trace(
            go.Scatter(x=cas_base_sample['Date'], y=cas_base_sample['Inflation'],
                      mode='lines', name='Inflation', line=dict(color='#45B7D1')),
            row=2, col=1
        )
        
        # PIB
        fig_evolution.add_trace(
            go.Scatter(x=cas_base_sample['Date'], y=cas_base_sample['Croissance_PIB'],
                      mode='lines', name='PIB', line=dict(color='#FFA500')),
            row=2, col=2
        )
        
        fig_evolution.update_layout(
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Métriques de performance du modèle
        st.subheader("🎯 Performance du Modèle")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Qualité de l'Ajustement</h4>
                <p><strong>R² Score:</strong> {st.session_state.r2:.1%}</p>
                <p><strong>Interprétation:</strong> {st.session_state.r2*100:.1f}% de la variance expliquée</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Précision</h4>
                <p><strong>Erreur Moyenne:</strong> ±{st.session_state.mae:.2f}%</p>
                <p><strong>Validation Croisée:</strong> ±{st.session_state.mae_vc:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Robustesse</h4>
                <p><strong>Échantillon:</strong> {len(st.session_state.df_mensuel)} observations</p>
                <p><strong>Horizon:</strong> 18 mois</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🇲🇦 <strong>SOFAC - Modèle de Prédiction des Rendements 52-Semaines</strong></p>
        <p>Données: Bank Al-Maghrib, HCP | Modèle: Régression Linéaire Multiple | Horizon: Juillet 2025 - Décembre 2026</p>
        <p><em>Les prédictions sont fournies à titre informatif et ne constituent pas des conseils financiers.</em></p>
    </div>
    """, unsafe_allow_html=True)>Qualité de l'Ajustement</h4>
                <p><strong>R² Score:</strong> {r2:.1%}</p>
                <p><strong>Interprétation:</strong> {r2*100:.1f}% de la variance expliquée</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Précision</h4>
                <p><strong>Erreur Moyenne:</strong> ±{mae:.2f}%</p>
                <p><strong>Validation Croisée:</strong> ±{mae_vc:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Robustesse</h4>
                <p><strong>Échantillon:</strong> {len(df_mensuel)} observations</p>
                <p><strong>Horizon:</strong> 18 mois</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🇲🇦 <strong>SOFAC - Modèle de Prédiction des Rendements 52-Semaines</strong></p>
        <p>Données: Bank Al-Maghrib, HCP | Modèle: Régression Linéaire Multiple | Horizon: Juillet 2025 - Décembre 2026</p>
        <p><em>Les prédictions sont fournies à titre informatif et ne constituent pas des conseils financiers.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
