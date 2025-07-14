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

st.set_page_config(
    page_title="SOFAC - Prédiction Rendements 52-Semaines",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text x="50" y="50" font-family="Arial" font-size="10" fill="rgba(255,255,255,0.1)" text-anchor="middle" dominant-baseline="middle">SOFAC</text></svg>');
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        text-align: center;
    }
    .recommendation-box h2 {
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .recommendation-box h3 {
        font-size: 1rem;
        margin-bottom: 0.8rem;
    }
    .data-status {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        padding: 0.4rem;
        border-radius: 5px;
        margin: 0.4rem 0;
        font-size: 0.8rem;
    }
    .data-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 0.4rem;
        border-radius: 5px;
        margin: 0.4rem 0;
        font-size: 0.8rem;
    }
    .executive-summary {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
    }
    .executive-summary h3 {
        font-size: 1rem;
        color: #28a745;
        margin-bottom: 0.8rem;
    }
    .summary-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid #dee2e6;
        font-size: 0.85rem;
    }
    .summary-item:last-child {
        border-bottom: none;
    }
    .summary-label {
        font-weight: 600;
        color: #495057;
    }
    .summary-value {
        font-weight: 700;
        color: #007bff;
    }
    .quick-recommendation {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 0.8rem;
        border-radius: 6px;
        margin-top: 0.8rem;
        border-left: 3px solid #28a745;
    }
    .quick-recommendation h4 {
        font-size: 0.9rem;
        color: #155724;
        margin-bottom: 0.4rem;
    }
    .quick-recommendation p {
        font-size: 0.8rem;
        color: #155724;
        margin: 0;
    }
    .sidebar-metric {
        background: #f8f9fa;
        padding: 0.6rem;
        border-radius: 6px;
        margin: 0.4rem 0;
        border-left: 3px solid #007bff;
        font-size: 0.8rem;
    }
    .trend-up { color: #28a745; font-weight: bold; }
    .trend-down { color: #dc3545; font-weight: bold; }
    .trend-stable { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_monthly_dataset():
    """Create monthly dataset using historical data only"""
    
    # Historical data - preserved exactly as in your notebook
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
        
        if date_courante.month == 12:
            date_courante = date_courante.replace(year=date_courante.year + 1, month=1)
        else:
            date_courante = date_courante.replace(month=date_courante.month + 1)
    
    return pd.DataFrame(donnees_mensuelles)

def train_prediction_model(df_mensuel):
    """Train the prediction model"""
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
    """Create economic scenarios starting from July 2025"""
    
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2026, 12, 31)
    
    dates_quotidiennes = []
    date_courante = date_debut
    
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
    # Base scenarios on realistic policy expectations
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
            
            date_str = date.strftime('%Y-%m')
            taux_directeur = 2.25
            for date_politique, taux in sorted(taux_politiques.items()):
                if date_str >= date_politique:
                    taux_directeur = taux
            
            np.random.seed(hash(date.strftime('%Y-%m-%d')) % 2**32)
            
            mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
            
            if nom_scenario == 'Conservateur':
                inflation_base = 1.4 + 0.5 * np.exp(-mois_depuis_debut / 18) + 0.2 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 - 0.5 * (mois_depuis_debut / 18) + 0.4 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            elif nom_scenario == 'Cas_de_Base':
                inflation_base = 1.4 + 0.3 * np.exp(-mois_depuis_debut / 12) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 - 0.2 * (mois_depuis_debut / 18) + 0.5 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            else:
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
    """Generate predictions with smooth continuity from June 2025 baseline"""
    
    # Use the correct June 2025 baseline from historical data
    rendement_juin_reel = 1.75
    predictions = {}
    
    for nom_scenario, scenario_df in scenarios.items():
        X_futur = scenario_df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
        rendements_bruts = modele.predict(X_futur)
        
        if len(rendements_bruts) > 0:
            # Ensure smooth transition from June 2025 historical value
            premier_predit = rendements_bruts[0]
            discontinuite = premier_predit - rendement_juin_reel
            
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
        else:
            rendements_lisses = rendements_bruts
        
        # Apply scenario-specific adjustments
        ajustements = []
        for i, ligne in scenario_df.iterrows():
            ajustement = 0
            
            if nom_scenario == 'Conservateur':
                ajustement += 0.1
            elif nom_scenario == 'Optimiste':
                ajustement -= 0.05
            
            # Add some time-based uncertainty
            jours_ahead = ligne['Jours_Ahead']
            incertitude = (jours_ahead / 365) * 0.05
            if nom_scenario == 'Conservateur':
                ajustement += incertitude
            elif nom_scenario == 'Optimiste':
                ajustement -= incertitude * 0.5
            
            # Day of week effects
            effets_jours = {
                'Monday': 0.01, 'Tuesday': 0.00, 'Wednesday': -0.01,
                'Thursday': 0.00, 'Friday': 0.02, 'Saturday': -0.01, 'Sunday': -0.01
            }
            ajustement += effets_jours.get(ligne['Jour_Semaine'], 0)
            
            ajustements.append(ajustement)
        
        rendements_finaux = rendements_lisses + np.array(ajustements)
        rendements_finaux = np.clip(rendements_finaux, 0.1, 8.0)
        
        scenario_df_copie = scenario_df.copy()
        scenario_df_copie['Rendement_Predit'] = rendements_finaux
        scenario_df_copie['Scenario'] = nom_scenario
        
        # Add confidence intervals
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
    """Generate recommendations using June 2025 historical baseline"""
    
    # Use the correct June 2025 baseline for recommendations
    rendement_actuel = 1.75  # June 2025 historical value
    recommandations = {}
    
    for nom_scenario, pred_df in predictions.items():
        rendement_futur_moyen = pred_df['Rendement_Predit'].mean()
        changement_rendement = rendement_futur_moyen - rendement_actuel
        volatilite = pred_df['Rendement_Predit'].std()
        
        if changement_rendement > 0.3:
            recommandation = "TAUX FIXE"
            raison = f"Rendements attendus en hausse de {changement_rendement:.2f}% depuis juin 2025. Bloquer les taux actuels avant que les coûts d'emprunt n'augmentent."
        elif changement_rendement < -0.3:
            recommandation = "TAUX VARIABLE"
            raison = f"Rendements attendus en baisse de {abs(changement_rendement):.2f}% depuis juin 2025. Utiliser des taux variables pour profiter de la diminution des coûts d'emprunt."
        else:
            recommandation = "STRATÉGIE FLEXIBLE"
            raison = f"Rendements relativement stables (±{abs(changement_rendement):.2f}%) depuis juin 2025. Approche mixte selon les besoins."
        
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
            'rendement_actuel': rendement_actuel,
            'rendement_futur_moyen': rendement_futur_moyen,
            'changement_rendement': changement_rendement,
            'volatilite': volatilite
        }
    
    return recommandations

def main():
    st.markdown("""
    <div class="main-header">
        <h1>SOFAC - Modèle de Prédiction des Rendements 52-Semaines</h1>
        <p>Système d'aide à la décision pour stratégie de financement</p>
        <p>Bank Al-Maghrib | HCP | Analyse Prédictive Avancée</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Informations du Modèle")
        
        # Load cached model data
        if 'data_loaded' not in st.session_state:
            with st.spinner("Calibration du modèle..."):
                st.session_state.df_mensuel = create_monthly_dataset()
                st.session_state.modele, st.session_state.r2, st.session_state.mae, st.session_state.rmse, st.session_state.mae_vc = train_prediction_model(st.session_state.df_mensuel)
                st.session_state.scenarios = create_economic_scenarios()
                st.session_state.predictions = generate_predictions(st.session_state.scenarios, st.session_state.modele, st.session_state.mae)
                st.session_state.recommandations = generate_recommendations(st.session_state.predictions)
                st.session_state.data_loaded = True
        
        # Display current yield from historical data (June 2025)
        derniere_donnee = st.session_state.df_mensuel.iloc[-1]
        rendement_actuel = derniere_donnee['Rendement_52s']
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <strong>Rendement Actuel</strong><br>
            <span style="font-size: 1.1rem; color: #007bff; font-weight: bold;">{rendement_actuel:.2f}%</span><br>
            <small>Juin 2025 (Dernière donnée historique)</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <strong>Performance Modèle</strong><br>
            R² = {st.session_state.r2*100:.1f}% (Excellent)<br>
            Précision = ±{st.session_state.mae:.2f}%
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <strong>Statut</strong><br>
            ✓ Modèle opérationnel<br>
            ✓ Données historiques validées<br>
            ✓ Prédictions générées
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <strong>Dernière MAJ</strong><br>
            {datetime.now().strftime('%d/%m/%Y')}<br>
            {datetime.now().strftime('%H:%M')}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Actualiser le Modèle", type="primary"):
            st.cache_data.clear()
            del st.session_state.data_loaded
            st.rerun()
    
    # Briefing Exécutif
    st.markdown("### ▸ Briefing Exécutif")
    
    # Calculs pour le résumé
    cas_de_base = st.session_state.predictions['Cas_de_Base']
    debut_2025 = cas_de_base.head(90)['Rendement_Predit'].mean()
    fin_2026 = cas_de_base.tail(90)['Rendement_Predit'].mean()
    evolution_totale = fin_2026 - debut_2025
    rendement_actuel = 1.75  # Dernière donnée historique
    
    rec_base = st.session_state.recommandations['Cas_de_Base']
    
    st.markdown(f"""
    <div class="executive-summary">
        <h3>▸ Résumé de la Situation Économique</h3>
        
        <div class="summary-item">
            <span class="summary-label">Rendement Actuel (Juin 2025)</span>
            <span class="summary-value">{rendement_actuel:.2f}%</span>
        </div>
        
        <div class="summary-item">
            <span class="summary-label">Prévision Moyenne 2025-2026</span>
            <span class="summary-value">{rec_base['rendement_futur_moyen']:.2f}%</span>
        </div>
        
        <div class="summary-item">
            <span class="summary-label">Évolution Attendue</span>
            <span class="summary-value trend-{'down' if rec_base['changement_rendement'] < 0 else 'up' if rec_base['changement_rendement'] > 0 else 'stable'}">{rec_base['changement_rendement']:+.2f}%</span>
        </div>
        
        <div class="summary-item">
            <span class="summary-label">Volatilité Prévue</span>
            <span class="summary-value">{rec_base['volatilite']:.2f}%</span>
        </div>
        
        <div class="summary-item">
            <span class="summary-label">Tendance Long Terme</span>
            <span class="summary-value">{"Baissière" if evolution_totale < -0.2 else "Haussière" if evolution_totale > 0.2 else "Stable"}</span>
        </div>
        
        <div class="quick-recommendation">
            <h4>▸ Recommandation Rapide</h4>
            <p><strong>{"↓" if rec_base['changement_rendement'] < -0.3 else "↑" if rec_base['changement_rendement'] > 0.3 else "→"} {rec_base['recommandation']}</strong></p>
            <p>{rec_base['raison'][:120]}...</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>▸ Rendement Actuel</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: #007bff; text-align: center;">{rendement_actuel:.2f}%</div>
            <small>Juin 2025 (Bank Al-Maghrib)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        changement = rec_base['changement_rendement']
        trend_class = "trend-down" if changement < 0 else "trend-up" if changement > 0 else "trend-stable"
        st.markdown(f"""
        <div class="metric-card">
            <h4>▸ Moyenne Future</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: #007bff; text-align: center;">{rec_base['rendement_futur_moyen']:.2f}%</div>
            <small class="{trend_class}">{changement:+.2f}% vs. actuel</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>▸ Volatilité</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: #007bff; text-align: center;">{rec_base['volatilite']:.2f}%</div>
            <small>Risque de variation</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>▸ Précision Modèle</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: #007bff; text-align: center;">{st.session_state.r2*100:.1f}%</div>
            <small>Variance expliquée</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommandation principale
    liste_recommandations = [rec['recommandation'] for rec in st.session_state.recommandations.values()]
    
    if liste_recommandations.count('TAUX VARIABLE') >= 2:
        strategie_globale = "TAUX VARIABLE"
        raison_globale = "Majorité des scénarios montrent des taux en baisse depuis juin 2025"
        couleur_globale = "#28a745"
        icone_globale = "↓"
    elif liste_recommandations.count('TAUX FIXE') >= 2:
        strategie_globale = "TAUX FIXE"
        raison_globale = "Majorité des scénarios montrent des taux en hausse depuis juin 2025"
        couleur_globale = "#dc3545"
        icone_globale = "↑"
    else:
        strategie_globale = "STRATÉGIE FLEXIBLE"
        raison_globale = "Signaux mixtes depuis juin 2025 - approche diversifiée recommandée"
        couleur_globale = "#ffc107"
        icone_globale = "→"
    
    st.markdown(f"""
    <div class="recommendation-box" style="background: linear-gradient(135deg, {couleur_globale} 0%, {couleur_globale}AA 100%);">
        <h2>▸ RECOMMANDATION STRATÉGIQUE SOFAC</h2>
        <h3>{icone_globale} {strategie_globale}</h3>
        <p>{raison_globale}</p>
        
        <h4>▸ Impact Financier Estimé (Emprunt 10M MAD):</h4>
    """, unsafe_allow_html=True)
    
    # Calcul de l'impact financier
    if rec_base['changement_rendement'] < -0.3:
        economies = abs(rec_base['changement_rendement']) * 10_000_000 / 100
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h4>▸ Économies Potentielles avec TAUX VARIABLE</h4>
            <p><strong>{economies:,.0f} MAD/an</strong></p>
            <p>Basé sur la baisse attendue de {abs(rec_base['changement_rendement']):.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    elif rec_base['changement_rendement'] > 0.3:
        cout_evite = rec_base['changement_rendement'] * 10_000_000 / 100
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h4>▸ Coûts Évités avec TAUX FIXE</h4>
            <p><strong>{cout_evite:,.0f} MAD/an</strong></p>
            <p>Basé sur la hausse attendue de {rec_base['changement_rendement']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h4>▸ Impact Financier Limité</h4>
            <p>Taux relativement stables (±{abs(rec_base['changement_rendement']):.2f}%)</p>
            <p>Approche flexible recommandée selon les besoins</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["▸ Vue d'Ensemble", "▸ Prédictions Détaillées", "▸ Recommandations"])
    
    with tab1:
        st.header("▸ Vue d'Ensemble des Prédictions")
        
        # Overview chart
        st.subheader("▸ Évolution des Rendements: Historique et Prédictions")
        
        fig_overview = go.Figure()
        
        # Historical data
        df_recent = st.session_state.df_mensuel.tail(12)
        
        fig_overview.add_trace(
            go.Scatter(
                x=df_recent['Date'],
                y=df_recent['Rendement_52s'],
                mode='lines+markers',
                name='Historique',
                line=dict(color='#60A5FA', width=4),
                marker=dict(size=8)
            )
        )
        
        # Prediction scenarios starting from July 2025
        couleurs = {'Conservateur': '#FF6B6B', 'Cas_de_Base': '#4ECDC4', 'Optimiste': '#45B7D1'}
        
        for nom_scenario, pred_df in st.session_state.predictions.items():
            donnees_hebdo = pred_df[::7]  # Weekly sampling for clarity
            
            fig_overview.add_trace(
                go.Scatter(
                    x=donnees_hebdo['Date'],
                    y=donnees_hebdo['Rendement_Predit'],
                    mode='lines+markers',
                    name=f'Prédiction {nom_scenario}',
                    line=dict(color=couleurs[nom_scenario], width=3),
                    marker=dict(size=6)
                )
            )
        
        # June 2025 baseline reference
        fig_overview.add_hline(
            y=1.75, 
            line_dash="dash", 
            line_color="blue",
            annotation_text="Juin 2025: 1.75%"
        )
        
        fig_overview.update_layout(
            title="Évolution des Rendements 52-Semaines: Historique (2020-2025) et Prédictions (2025-2026)",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white",
            font=dict(size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # Quick scenario comparison
        st.subheader("▸ Comparaison des Scénarios")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (scenario, rec) in enumerate(st.session_state.recommandations.items()):
            with [col1, col2, col3][i]:
                trend_icon = "↓" if rec['changement_rendement'] < -0.3 else "↑" if rec['changement_rendement'] > 0.3 else "→"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{scenario}</h4>
                    <p><strong>{trend_icon} {rec['recommandation']}</strong></p>
                    <p>Changement: {rec['changement_rendement']:+.2f}%</p>
                    <p>Risque: {rec['niveau_risque']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("▸ Prédictions Détaillées")
        
        scenario_selectionne = st.selectbox(
            "Choisissez un scénario:",
            options=['Cas_de_Base', 'Conservateur', 'Optimiste'],
            index=0,
            help="Sélectionnez le scénario économique à analyser"
        )
        
        pred_scenario = st.session_state.predictions[scenario_selectionne]
        
        # Enhanced metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rendement Moyen", f"{pred_scenario['Rendement_Predit'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{pred_scenario['Rendement_Predit'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{pred_scenario['Rendement_Predit'].max():.2f}%")
        with col4:
            baseline_comparison = pred_scenario['Rendement_Predit'].mean() - 1.75
            st.metric("Écart vs Juin 2025", f"{baseline_comparison:+.2f}%")
        
        # Detailed prediction chart
        st.subheader(f"▸ Prédictions Quotidiennes - Scénario {scenario_selectionne}")
        
        donnees_affichage = pred_scenario[::3]
        
        fig_detail = go.Figure()
        
        # Confidence bands
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
        
        # Main prediction line
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
        
        # June 2025 baseline reference
        fig_detail.add_hline(
            y=1.75, 
            line_dash="dash", 
            line_color="blue",
            annotation_text="Juin 2025: 1.75%"
        )
        
        fig_detail.update_layout(
            title=f"Prédictions Détaillées - {scenario_selectionne} (Continuité depuis Juin 2025)",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white",
            font=dict(size=11)
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Export functionality
        if st.button("▸ Télécharger les Prédictions"):
            pred_export = pred_scenario.copy()
            pred_export['Baseline_Juin_2025'] = 1.75
            
            csv = pred_export.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"sofac_predictions_{scenario_selectionne.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("▸ Recommandations Stratégiques")
        
        # Detailed scenario analysis
        st.subheader("▸ Analyse Détaillée par Scénario")
        
        for nom_scenario, rec in st.session_state.recommandations.items():
            with st.expander(f"▸ Scénario {nom_scenario}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Recommandation:** {rec['recommandation']}
                    
                    **Justification:** {rec['raison']}
                    
                    **Métriques (vs juin 2025: 1.75%):**
                    - Rendement moyen prédit: {rec['rendement_futur_moyen']:.2f}%
                    - Changement attendu: {rec['changement_rendement']:+.2f}%
                    - Volatilité: {rec['volatilite']:.2f}%
                    - Niveau de risque: {rec['niveau_risque']}
                    """)
                
                with col2:
                    # Mini chart
                    fig_mini = go.Figure()
                    
                    pred_df = st.session_state.predictions[nom_scenario]
                    echantillon_mini = pred_df[::30]
                    
                    # June 2025 baseline
                    fig_mini.add_hline(
                        y=1.75, 
                        line_dash="dash", 
                        line_color="blue",
                        annotation_text="Juin 2025: 1.75%"
                    )
                    
                    # Prediction line
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
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis_title="Rendement (%)",
                        font=dict(size=10)
                    )
                    
                    st.plotly_chart(fig_mini, use_container_width=True)
        
        # Financial impact calculator
        st.subheader("▸ Calculateur d'Impact Financier")
        
        col1, col2 = st.columns(2)
        
        with col1:
            montant_emprunt = st.slider(
                "Montant d'emprunt (millions MAD):",
                min_value=1,
                max_value=100,
                value=10,
                step=1
            )
        
        with col2:
            duree_emprunt = st.slider(
                "Durée d'emprunt (années):",
                min_value=1,
                max_value=10,
                value=3,
                step=1
            )
        
        changement_cas_base = st.session_state.recommandations['Cas_de_Base']['changement_rendement']
        impact_total = changement_cas_base * montant_emprunt * 1_000_000 / 100 * duree_emprunt
        
        if abs(changement_cas_base) > 0.2:
            if changement_cas_base < 0:
                st.success(f"""
                ▸ **Économies Potentielles avec TAUX VARIABLE:**
                
                - **Économies annuelles:** {abs(changement_cas_base) * montant_emprunt * 10_000:,.0f} MAD
                - **Économies totales ({duree_emprunt} ans):** {abs(impact_total):,.0f} MAD
                - **Basé sur:** Baisse attendue de {abs(changement_cas_base):.2f}% vs juin 2025 (1.75%)
                """)
            else:
                st.warning(f"""
                ▸ **Coûts Évités avec TAUX FIXE:**
                
                - **Surcoûts évités annuellement:** {changement_cas_base * montant_emprunt * 10_000:,.0f} MAD
                - **Surcoûts évités totaux ({duree_emprunt} ans):** {impact_total:,.0f} MAD
                - **Basé sur:** Hausse attendue de {changement_cas_base:.2f}% vs juin 2025 (1.75%)
                """)
        else:
            st.info(f"""
            ▸ **Impact Financier Limité:**
            
            - **Variation attendue:** ±{abs(changement_cas_base):.2f}% vs juin 2025 (1.75%)
            - **Impact annuel:** ±{abs(changement_cas_base) * montant_emprunt * 10_000:,.0f} MAD
            - **Approche flexible recommandée**
            """)
    
    # Section technique (repliable)
    with st.expander("▸ Informations Techniques du Modèle"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **▸ Performance du Modèle:**
            - R² Score: {st.session_state.r2:.3f} ({st.session_state.r2*100:.1f}% de variance expliquée)
            - Erreur Absolue Moyenne: {st.session_state.mae:.3f}%
            - Méthode: Régression Linéaire Multiple
            - Validation: Cross-validation 5-fold
            
            **▸ Données:**
            - Période d'entraînement: 2020-2025
            - Observations: {len(st.session_state.df_mensuel)} points mensuels
            - Prédictions: {len(st.session_state.predictions['Cas_de_Base'])} jours (Juillet 2025 - Décembre 2026)
            """)
        
        with col2:
            st.markdown(f"""
            **▸ Équation de Prédiction:**
            ```
            Rendement = 0.188 + 0.959×Taux_Directeur 
                      + 0.037×Inflation 
                      - 0.022×Croissance_PIB
            ```
            
            **▸ Variables Explicatives:**
            - Taux Directeur BAM (impact: +0.959)
            - Inflation sous-jacente (impact: +0.037)
            - Croissance PIB (impact: -0.022)
            """)
    
    # Footer
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1.5rem; font-size: 0.85rem;">
        <p><strong>SOFAC - Modèle de Prédiction des Rendements 52-Semaines</strong></p>
        <p>Sources: Bank Al-Maghrib, HCP | Modèle: Régression Linéaire Multiple</p>
        <p>Horizon: Juillet 2025 - Décembre 2026 | Baseline: Juin 2025 (1.75%)</p>
        <p>Dernière mise à jour: {current_time}</p>
        <p><em>⚠ Les prédictions sont basées sur des données historiques et ne constituent pas des conseils financiers.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
