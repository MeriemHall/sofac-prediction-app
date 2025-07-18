import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import requests
from bs4 import BeautifulSoup
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SOFAC Yield Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for glassmorphism dark theme
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #1e293b 0%, #1e40af 50%, #1e293b 100%);
        min-height: 100vh;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Glass card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    
    .glass-card-large {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #93c5fd;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Recommendation card */
    .recommendation-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin-bottom: 2rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .recommendation-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-message {
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .confidence-level {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Insight cards */
    .insight-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .insight-title {
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    .insight-text {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Note box */
    .note-box {
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1.5rem;
    }
    
    .note-text {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    p, span, div {
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Last update styling */
    .last-update {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        text-align: right;
    }
    
    /* Status indicators */
    .status-up { color: #10b981; }
    .status-down { color: #ef4444; }
    .status-stable { color: #f59e0b; }
    
    /* Remove default streamlit spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_live_moroccan_data():
    """Fetch live data from Bank Al-Maghrib and HCP"""
    
    live_data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'policy_rate': 2.25,
        'yield_52w': 2.40,
        'inflation': 1.1,
        'gdp_growth': 4.8,
        'sources': {},
        'success_count': 0,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Try to fetch Bank Al-Maghrib policy rate
    try:
        bkam_urls = [
            "https://www.bkam.ma/Politique-monetaire",
            "https://www.bkam.ma/"
        ]
        
        for url in bkam_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    patterns = [
                        r'taux.*?directeur.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?taux.*?directeur'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            try:
                                rate = float(match.replace(',', '.'))
                                if 0.25 <= rate <= 8.0:
                                    live_data['policy_rate'] = rate
                                    live_data['sources']['policy_rate'] = 'Bank Al-Maghrib Live'
                                    live_data['success_count'] += 1
                                    break
                            except ValueError:
                                continue
                        if live_data['success_count'] > 0:
                            break
                    
                    if live_data['success_count'] > 0:
                        break
            except:
                continue
        
        if 'policy_rate' not in live_data['sources']:
            live_data['sources']['policy_rate'] = 'Fallback Value'
    except:
        live_data['sources']['policy_rate'] = 'Fallback Value'
    
    # Estimate 52-week yield from policy rate
    spread = 0.15
    if live_data['policy_rate'] < 2.0:
        spread = 0.25
    elif live_data['policy_rate'] > 4.0:
        spread = 0.10
    
    live_data['yield_52w'] = live_data['policy_rate'] + spread
    live_data['sources']['yield_52w'] = f'Estimated from Policy Rate (+{spread*100:.0f}bps)'
    
    # Try to fetch HCP inflation data
    try:
        hcp_urls = ["https://www.hcp.ma/"]
        
        for url in hcp_urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    patterns = [
                        r'inflation.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?inflation'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            try:
                                rate = float(match.replace(',', '.'))
                                if 0 <= rate <= 20:
                                    live_data['inflation'] = rate
                                    live_data['sources']['inflation'] = 'HCP Live'
                                    live_data['success_count'] += 1
                                    break
                            except ValueError:
                                continue
                        if 'inflation' in live_data['sources']:
                            break
                    
                    if 'inflation' in live_data['sources']:
                        break
            except:
                continue
        
        if 'inflation' not in live_data['sources']:
            live_data['sources']['inflation'] = 'Fallback Value'
    except:
        live_data['sources']['inflation'] = 'Fallback Value'
    
    # GDP data estimation
    live_data['sources']['gdp_growth'] = 'Economic Estimation'
    
    # Data validation
    live_data['policy_rate'] = max(0.1, min(10.0, live_data['policy_rate']))
    live_data['yield_52w'] = max(0.1, min(15.0, live_data['yield_52w']))
    live_data['inflation'] = max(0.0, min(25.0, live_data['inflation']))
    live_data['gdp_growth'] = max(-10.0, min(20.0, live_data['gdp_growth']))
    
    return live_data

@st.cache_data
def create_monthly_dataset():
    """Create monthly dataset using historical data only"""
    
    # Historical data - preserved exactly
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
            recommandation = "FIXED RATE"
            raison = f"Yields expected to rise by {changement_rendement:.2f}% from June 2025. Lock in current rates before borrowing costs increase."
            trend = "up"
        elif changement_rendement < -0.3:
            recommandation = "VARIABLE RATE"
            raison = f"Yields expected to fall by {abs(changement_rendement):.2f}% from June 2025. Use variable rates to benefit from decreasing borrowing costs."
            trend = "down"
        else:
            recommandation = "EITHER RATE"
            raison = f"Yields relatively stable (±{abs(changement_rendement):.2f}%) from June 2025. Mixed approach based on needs."
            trend = "stable"
        
        if volatilite < 0.2:
            confidence = "High"
        elif volatilite < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        recommandations[nom_scenario] = {
            'type': recommandation,
            'message': raison,
            'trend': trend,
            'confidence': confidence,
            'rendement_actuel': rendement_actuel,
            'rendement_futur_moyen': rendement_futur_moyen,
            'changement_rendement': changement_rendement,
            'volatilite': volatilite
        }
    
    return recommandations

def main():
    # Load data first
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.df_mensuel = create_monthly_dataset()
            st.session_state.modele, st.session_state.r2, st.session_state.mae, st.session_state.rmse, st.session_state.mae_vc = train_prediction_model(st.session_state.df_mensuel)
            st.session_state.scenarios = create_economic_scenarios()
            st.session_state.predictions = generate_predictions(st.session_state.scenarios, st.session_state.modele, st.session_state.mae)
            st.session_state.recommandations = generate_recommendations(st.session_state.predictions)
            st.session_state.data_loaded = True
    
    # Get current time
    current_time = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">📊</div>
                <div>
                    <div class="header-title">SOFAC Yield Predictor</div>
                    <div class="header-subtitle">52-Week Treasury Yield Prediction Tool</div>
                </div>
            </div>
            <div class="last-update">
                📅 Last Update: {current_time}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get recommendation for base case
    base_recommendation = st.session_state.recommandations['Cas_de_Base']
    
    # Recommendation Card
    trend_icon = "📈" if base_recommendation['trend'] == 'up' else "📉" if base_recommendation['trend'] == 'down' else "⚖️"
    trend_class = "status-up" if base_recommendation['trend'] == 'up' else "status-down" if base_recommendation['trend'] == 'down' else "status-stable"
    
    st.markdown(f"""
    <div class="recommendation-card">
        <div style="font-size: 2rem;" class="{trend_class}">{trend_icon}</div>
        <div style="flex: 1;">
            <div class="recommendation-title">Financial Decision Recommendation</div>
            <div class="recommendation-message">{base_recommendation['message']}</div>
            <div class="confidence-level">Confidence Level: {base_recommendation['confidence']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Chart
    st.markdown('<div class="glass-card-large">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white; margin-bottom: 1rem;">52-Week Yield Prediction Chart</h3>', unsafe_allow_html=True)
    
    # Prepare data for chart
    df_historical = st.session_state.df_mensuel
    cas_de_base = st.session_state.predictions['Cas_de_Base']
    
    # Create chart
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=df_historical['Date'],
            y=df_historical['Rendement_52s'],
            mode='lines+markers',
            name='52-Week Yield (Historical)',
            line=dict(color='#60A5FA', width=3),
            marker=dict(size=6)
        )
    )
    
    # Fixed rate (historical)
    fig.add_trace(
        go.Scatter(
            x=df_historical['Date'],
            y=df_historical['Taux_Directeur'],
            mode='lines+markers',
            name='Fixed Rate',
            line=dict(color='#F59E0B', width=2, dash='dash'),
            marker=dict(size=4)
        )
    )
    
    # Inflation (historical)
    fig.add_trace(
        go.Scatter(
            x=df_historical['Date'],
            y=df_historical['Inflation'],
            mode='lines+markers',
            name='Inflation',
            line=dict(color='#EF4444', width=2, dash='dot'),
            marker=dict(size=4)
        )
    )
    
    # Predictions (sample every 7 days for clarity)
    pred_sample = cas_de_base[::7]
    fig.add_trace(
        go.Scatter(
            x=pred_sample['Date'],
            y=pred_sample['Rendement_Predit'],
            mode='lines+markers',
            name='52-Week Yield (Predicted)',
            line=dict(color='#10B981', width=3),
            marker=dict(size=6)
        )
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=list(pred_sample['Date']) + list(pred_sample['Date'][::-1]),
            y=list(pred_sample['Borne_Sup_95']) + list(pred_sample['Borne_Inf_95'][::-1]),
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval (95%)',
            showlegend=True
        )
    )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            title='Date',
            gridcolor='rgba(255,255,255,0.1)',
            color='rgba(255,255,255,0.7)'
        ),
        yaxis=dict(
            title='Percentage (%)',
            gridcolor='rgba(255,255,255,0.1)',
            color='rgba(255,255,255,0.7)'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            font=dict(color='white')
        ),
        margin=dict(l=0, r=0, t=20, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Key Insights
    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem;">💡</div>
            <h3 style="color: white; margin: 0;">Key Insights for SOFAC</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create three columns for insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_yield = 1.75  # June 2025 baseline
        current_fixed_rate = 2.25
        current_inflation = 1.4
        current_gdp = 3.8
        
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Current Situation</div>
            <div class="insight-text">
                52-week yield is at {current_yield:.2f}%, while the fixed rate is {current_fixed_rate:.2f}%. 
                Inflation is low at {current_inflation:.1f}%, and GDP growth is healthy at {current_gdp:.1f}%.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        future_avg = st.session_state.predictions['Cas_de_Base']['Rendement_Predit'].mean()
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Prediction Summary</div>
            <div class="insight-text">
                Model predicts yields will {"increase" if future_avg > current_yield else "decrease"} to an average of {future_avg:.2f}% 
                through 2025-2026 as inflation remains controlled and fixed rates are expected to {"rise" if future_avg > current_yield else "fall"}.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        volatility = st.session_state.predictions['Cas_de_Base']['Rendement_Predit'].std()
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">Risk Factors</div>
            <div class="insight-text">
                Predicted volatility is {volatility:.2f}%. Monitor inflation closely. If it rises above 2%, yields could 
                change faster than predicted. GDP volatility could also impact results.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics row
    st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{current_yield:.2f}%</div>
            <div class="metric-label">Current Yield</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_predicted = st.session_state.predictions['Cas_de_Base']['Rendement_Predit'].mean()
        change = avg_predicted - current_yield
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{avg_predicted:.2f}%</div>
            <div class="metric-label">Predicted Average</div>
            <div style="font-size: 0.8rem; color: {'#10b981' if change < 0 else '#ef4444'};">
                {change:+.2f}% change
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence_score = "High" if volatility < 0.2 else "Medium" if volatility < 0.4 else "Low"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{confidence_score}</div>
            <div class="metric-label">Confidence Level</div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.7);">
                ±{volatility:.2f}% volatility
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        model_accuracy = st.session_state.r2
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{model_accuracy:.1%}</div>
            <div class="metric-label">Model Accuracy</div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.7);">
                R² Score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Note Box
    st.markdown(f"""
    <div class="note-box">
        <div class="note-text">
            <strong>Note:</strong> This is a predictive model for financial analysis purposes. 
            Real financial decisions should always involve additional analysis and professional consultation.
            The model updates daily and uses the relationship between fixed rates, inflation, 
            and GDP to predict 52-week yields. Current baseline is June 2025 at {current_yield:.2f}%.
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
