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
    page_title="SOFAC - Prédiction Rendements 52-Semaines",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .data-status {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
    .data-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
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

def display_live_data_panel(live_data):
    """Display live data panel in sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Données en Temps Réel")
    
    live_sources = sum(1 for source in live_data['sources'].values() if 'Live' in source)
    total_sources = 4
    success_rate = (live_sources / total_sources) * 100
    
    if success_rate >= 50:
        st.sidebar.markdown('<div class="data-status">🟢 Données partiellement en direct</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="data-warning">🟡 Données principalement estimées</div>', unsafe_allow_html=True)
    
    st.sidebar.write(f"**Sources en direct:** {live_sources}/{total_sources} ({success_rate:.0f}%)")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        indicator = "🟢" if 'Live' in live_data['sources']['policy_rate'] else "🔴"
        st.metric(
            f"{indicator} Taux Directeur", 
            f"{live_data['policy_rate']:.2f}%",
            help=f"Source: {live_data['sources']['policy_rate']}"
        )
        
        indicator = "🟢" if 'Live' in live_data['sources']['inflation'] else "🔴"
        st.metric(
            f"{indicator} Inflation", 
            f"{live_data['inflation']:.2f}%",
            help=f"Source: {live_data['sources']['inflation']}"
        )
    
    with col2:
        indicator = "🟡"
        st.metric(
            f"{indicator} Rendement 52s", 
            f"{live_data['yield_52w']:.2f}%",
            delta=f"+{(live_data['yield_52w'] - live_data['policy_rate']):.2f}%",
            help=f"Source: {live_data['sources']['yield_52w']}"
        )
        
        st.metric(
            "🔴 Croissance PIB", 
            f"{live_data['gdp_growth']:.2f}%",
            help=f"Source: {live_data['sources']['gdp_growth']}"
        )
    
    st.sidebar.info(f"🕐 Dernière mise à jour: {live_data['last_updated']}")
    
    if st.sidebar.button("🔄 Actualiser"):
        st.cache_data.clear()
        st.rerun()

@st.cache_data
def create_monthly_dataset_with_live_data(live_data):
    """Create monthly dataset incorporating live data as future point only"""
    
    # Base historical data - DO NOT MODIFY
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
        '2025-06': {'taux_directeur': 2.25, 'inflation': 1.3, 'pib': 3.7, 'rendement_52s': 1.75}  # Keep original June value
    }
    
    # Only add live data for future months (after June 2025)
    current_month = datetime.now().strftime('%Y-%m')
    current_date = datetime.now()
    
    # Only add live data if we're past June 2025 and it's a new month
    if current_date > datetime(2025, 6, 30) and current_month not in donnees_historiques:
        donnees_historiques[current_month] = {
            'taux_directeur': live_data['policy_rate'],
            'inflation': live_data['inflation'],
            'pib': live_data['gdp_growth'],
            'rendement_52s': live_data['yield_52w']
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
    # Extend to current month only if we're past June 2025
    if current_date > datetime(2025, 6, 30):
        date_fin = datetime.now().replace(day=1) + timedelta(days=32)
        date_fin = date_fin.replace(day=1) - timedelta(days=1)
    else:
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
        
        # Mark if this is live data (only for months after June 2025)
        is_live_data = (current_date > datetime(2025, 6, 30) and date_str == current_month and date_str not in ['2025-06'])
        
        donnees_mensuelles.append({
            'Date': date_str,
            'Taux_Directeur': point_donnees['taux_directeur'],
            'Inflation': point_donnees['inflation'],
            'Croissance_PIB': point_donnees['pib'],
            'Rendement_52s': point_donnees['rendement_52s'],
            'Est_Point_Ancrage': est_ancrage,
            'Est_Live_Data': is_live_data
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
def create_economic_scenarios_with_live_base(live_data):
    """Create economic scenarios starting from live data"""
    
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2026, 12, 31)
    
    if datetime.now() > date_debut:
        date_debut = datetime.now().replace(day=1) + timedelta(days=32)
        date_debut = date_debut.replace(day=1)
    
    dates_quotidiennes = []
    date_courante = date_debut
    
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
    base_rate = live_data['policy_rate']
    
    decisions_politiques = {
        'Conservateur': {
            '2025-06': base_rate, 
            '2025-09': max(base_rate - 0.25, 1.5), 
            '2025-12': max(base_rate - 0.5, 1.25),
            '2026-03': max(base_rate - 0.75, 1.0), 
            '2026-06': max(base_rate - 0.75, 1.0), 
            '2026-09': max(base_rate - 1.0, 1.0), 
            '2026-12': max(base_rate - 1.0, 1.0)
        },
        'Cas_de_Base': {
            '2025-06': base_rate, 
            '2025-09': max(base_rate - 0.50, 1.0), 
            '2025-12': max(base_rate - 0.75, 0.75),
            '2026-03': max(base_rate - 1.0, 0.75), 
            '2026-06': max(base_rate - 1.0, 0.75), 
            '2026-09': max(base_rate - 1.25, 0.5), 
            '2026-12': max(base_rate - 1.25, 0.5)
        },
        'Optimiste': {
            '2025-06': base_rate, 
            '2025-09': max(base_rate - 0.75, 0.75), 
            '2025-12': max(base_rate - 1.0, 0.5),
            '2026-03': max(base_rate - 1.25, 0.5), 
            '2026-06': max(base_rate - 1.5, 0.25), 
            '2026-09': max(base_rate - 1.5, 0.25), 
            '2026-12': max(base_rate - 1.5, 0.25)
        }
    }
    
    scenarios = {}
    
    for nom_scenario in ['Conservateur', 'Cas_de_Base', 'Optimiste']:
        donnees_scenario = []
        taux_politiques = decisions_politiques[nom_scenario]
        
        base_inflation = live_data['inflation']
        base_gdp = live_data['gdp_growth']
        
        for i, date in enumerate(dates_quotidiennes):
            jours_ahead = i + 1
            
            date_str = date.strftime('%Y-%m')
            taux_directeur = base_rate
            for date_politique, taux in sorted(taux_politiques.items()):
                if date_str >= date_politique:
                    taux_directeur = taux
            
            np.random.seed(hash(date.strftime('%Y-%m-%d')) % 2**32)
            
            mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
            
            if nom_scenario == 'Conservateur':
                inflation_base = base_inflation + 0.3 * np.exp(-mois_depuis_debut / 18)
                pib_base = base_gdp - 0.3 * (mois_depuis_debut / 18)
            elif nom_scenario == 'Cas_de_Base':
                inflation_base = base_inflation + 0.1 * np.exp(-mois_depuis_debut / 12)
                pib_base = base_gdp - 0.1 * (mois_depuis_debut / 18)
            else:
                inflation_base = base_inflation - 0.2 * (mois_depuis_debut / 18)
                pib_base = base_gdp + 0.2 * (mois_depuis_debut / 18)
            
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

def generate_predictions_with_live_continuity(scenarios, modele, mae_historique, live_data):
    """Generate predictions with smooth continuity from June 2025 baseline"""
    
    # Use the correct June 2025 baseline from historical data
    rendement_juin_reel = 1.75  # This is the historical June 2025 value
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
            
            ic_95 = intervalle_ajuste * 2
            
            scenario_df_copie.loc[i, 'Borne_Inf_95'] = max(0.1, ligne['Rendement_Predit'] - ic_95)
            scenario_df_copie.loc[i, 'Borne_Sup_95'] = min(8.0, ligne['Rendement_Predit'] + ic_95)
        
        predictions[nom_scenario] = scenario_df_copie
    
    return predictions

def generate_recommendations_with_live_context(predictions, live_data):
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
            raison = f"Rendements attendus en hausse de {changement_rendement:.2f}% depuis juin 2025."
        elif changement_rendement < -0.3:
            recommandation = "TAUX VARIABLE"
            raison = f"Rendements attendus en baisse de {abs(changement_rendement):.2f}% depuis juin 2025."
        else:
            recommandation = "STRATÉGIE FLEXIBLE"
            raison = f"Rendements relativement stables (±{abs(changement_rendement):.2f}%) depuis juin 2025."
        
        if volatilite < 0.2:
            niveau_risque = "FAIBLE"
        elif volatilite < 0.4:
            niveau_risque = "MOYEN"
        else:
            niveau_risque = "ÉLEVÉ"
        
        live_sources_count = sum(1 for source in live_data['sources'].values() if 'Live' in source)
        if live_sources_count >= 2:
            confiance = "ÉLEVÉE"
        elif live_sources_count >= 1:
            confiance = "MOYENNE"
        else:
            confiance = "LIMITÉE"
        
        recommandations[nom_scenario] = {
            'recommandation': recommandation,
            'raison': raison,
            'niveau_risque': niveau_risque,
            'confiance': confiance,
            'rendement_actuel': rendement_actuel,
            'rendement_futur_moyen': rendement_futur_moyen,
            'changement_rendement': changement_rendement,
            'volatilite': volatilite,
            'live_data_quality': f"{live_sources_count}/4 sources directes"
        }
    
    return recommandations

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🇲🇦 SOFAC - Modèle de Prédiction des Rendements 52-Semaines</h1>
        <p>Système d'aide à la décision avec données automatiques Bank Al-Maghrib & HCP</p>
        <p><strong>Mise à jour:</strong> Horaire | <strong>Prochaine mise à jour:</strong> {}</p>
    </div>
    """.format((datetime.now() + timedelta(hours=1)).strftime('%H:%M')), unsafe_allow_html=True)
    
    # Fetch live data
    with st.spinner("🔄 Récupération des données en temps réel..."):
        live_data = fetch_live_moroccan_data()
    
    with st.sidebar:
        st.header("📊 Informations du Modèle")
        
        # Display live data panel
        display_live_data_panel(live_data)
        
        # Load cached model data
        if 'data_loaded' not in st.session_state or st.session_state.get('last_update') != live_data['date']:
            with st.spinner("🤖 Recalibration du modèle..."):
                st.session_state.df_mensuel = create_monthly_dataset_with_live_data(live_data)
                st.session_state.modele, st.session_state.r2, st.session_state.mae, st.session_state.rmse, st.session_state.mae_vc = train_prediction_model(st.session_state.df_mensuel)
                st.session_state.scenarios = create_economic_scenarios_with_live_base(live_data)
                st.session_state.predictions = generate_predictions_with_live_continuity(st.session_state.scenarios, st.session_state.modele, st.session_state.mae, live_data)
                st.session_state.recommandations = generate_recommendations_with_live_context(st.session_state.predictions, live_data)
                st.session_state.data_loaded = True
                st.session_state.last_update = live_data['date']
        
        st.success("✅ Modèle calibré avec données actuelles!")
        
        # Model performance metrics
        st.subheader("🎯 Performance du Modèle")
        st.metric("R² Score", f"{st.session_state.r2:.1%}", help="Pourcentage de variance expliquée")
        st.metric("Précision", f"±{st.session_state.mae:.2f}%", help="Erreur absolue moyenne")
        st.metric("Validation Croisée", f"±{st.session_state.mae_vc:.2f}%", help="Erreur en validation croisée")
        
        st.info("🔄 Le modèle est automatiquement recalibré avec les dernières données disponibles.")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Vue d'Ensemble", "🔮 Prédictions Détaillées", "💼 Recommandations"])
    
    with tab1:
        st.header("📈 Vue d'Ensemble des Prédictions")
        
        # Key metrics with live data context
        col1, col2, col3, col4 = st.columns(4)
        
        cas_de_base = st.session_state.predictions['Cas_de_Base']
        rendement_moyen = cas_de_base['Rendement_Predit'].mean()
        
        # Use the correct June 2025 baseline for comparison
        rendement_baseline = 1.75  # June 2025 historical value
        changement = rendement_moyen - rendement_baseline
        volatilite = cas_de_base['Rendement_Predit'].std()_Predit'].mean()
        
        # Use the correct June 2025 baseline for comparison
        rendement_baseline = 1.75  # June 2025 historical value
        changement = rendement_moyen - rendement_baseline
        volatilite = cas_de_base['Rendement_Predit'].std()
        
        with col1:
            st.metric(
                "Rendement Actuel (Juin 2025)", 
                f"{rendement_baseline:.2f}%",
                help="Dernière valeur historique - Juin 2025"
            )
        
        with col2:
            st.metric(
                "Rendement Moyen Prédit", 
                f"{rendement_moyen:.2f}%",
                delta=f"{changement:+.2f}%"
            )
        
        with col3:
            st.metric(
                "Volatilité Attendue", 
                f"{volatilite:.2f}%",
                help="Écart-type des prédictions"
            )
        
        with col4:
            quality_score = sum(1 for source in live_data['sources'].values() if 'Live' in source)
            st.metric(
                "Qualité des Données", 
                f"{quality_score}/4",
                delta="Direct" if quality_score >= 2 else "Mixte"
            )_Predit'].mean()
        changement = rendement_moyen - live_data['yield_52w']
        volatilite = cas_de_base['Rendement_Predit'].std()
        
        with col1:
            st.metric(
                "Rendement Actuel (Live)", 
                f"{live_data['yield_52w']:.2f}%",
                help=f"Source: {live_data['sources']['yield_52w']}"
            )
        
        with col2:
            st.metric(
                "Rendement Moyen Prédit", 
                f"{rendement_moyen:.2f}%",
                delta=f"{changement:+.2f}%"
            )
        
        with col3:
            st.metric(
                "Volatilité Attendue", 
                f"{volatilite:.2f}%",
                help="Écart-type des prédictions"
            )
        
        with col4:
            quality_score = sum(1 for source in live_data['sources'].values() if 'Live' in source)
            st.metric(
                "Qualité des Données", 
                f"{quality_score}/4",
                delta="Direct" if quality_score >= 2 else "Mixte"
            )
        
        # Overview chart
        st.subheader("📊 Évolution des Rendements: Historique et Prédictions")
        
        fig_overview = go.Figure()
        
        # Historical data
        df_recent = st.session_state.df_mensuel.tail(8)
        df_historical = df_recent[~df_recent.get('Est_Live_Data', False)]
        df_live = df_recent[df_recent.get('Est_Live_Data', False)]
        
        # Historical points
        if not df_historical.empty:
            fig_overview.add_trace(
                go.Scatter(
                    x=df_historical['Date'],
                    y=df_historical['Rendement_52s'],
                    mode='lines+markers',
                    name='Historique',
                    line=dict(color='#60A5FA', width=4),
                    marker=dict(size=8)
                )
            )
        
        # Live data point
        if not df_live.empty:
            fig_overview.add_trace(
                go.Scatter(
                    x=df_live['Date'],
                    y=df_live['Rendement_52s'],
                    mode='markers',
                    name='Données Live',
                    marker=dict(color='#22C55E', size=12, symbol='star'),
                    text=['Point de données en direct'],
                    textposition='top center'
                )
            )
        
        # Prediction scenarios
        couleurs = {'Conservateur': '#FF6B6B', 'Cas_de_Base': '#4ECDC4', 'Optimiste': '#45B7D1'}
        
        for nom_scenario, pred_df in st.session_state.predictions.items():
            donnees_hebdo = pred_df[::7]
            
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
        
        fig_overview.update_layout(
            title="Évolution des Rendements 52-Semaines avec Données Live",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # Quick recommendations
        st.subheader("🎯 Recommandations Rapides (Basées sur Données Live)")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (scenario, rec) in enumerate(st.session_state.recommandations.items()):
            with [col1, col2, col3][i]:
                confidence_color = "#4CAF50" if rec['confiance'] == "ÉLEVÉE" else "#FF9800" if rec['confiance'] == "MOYENNE" else "#F44336"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{scenario}</h4>
                    <p><strong>{rec['recommandation']}</strong></p>
                    <p>Changement: {rec['changement_rendement']:+.2f}%</p>
                    <p>Risque: {rec['niveau_risque']}</p>
                    <p style="color: {confidence_color};">Confiance: {rec['confiance']}</p>
                    <small>{rec['live_data_quality']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("🔮 Prédictions Détaillées")
        
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
            # Compare to June 2025 baseline instead of live data
            baseline_comparison = pred_scenario['Rendement_Predit'].mean() - 1.75
            st.metric("Écart vs Juin 2025", f"{baseline_comparison:+.2f}%")
        
        # Detailed prediction chart
        st.subheader(f"📊 Prédictions Quotidiennes - Scénario {scenario_selectionne}")
        
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
            template="plotly_white"
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Export functionality
        if st.button("📥 Télécharger les Prédictions"):
            pred_export = pred_scenario.copy()
            pred_export['Live_Baseline'] = live_data['yield_52w']
            pred_export['Live_Data_Quality'] = f"{sum(1 for s in live_data['sources'].values() if 'Live' in s)}/4"
            
            csv = pred_export.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV avec contexte live",
                data=csv,
                file_name=f"sofac_predictions_{scenario_selectionne.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("💼 Recommandations Stratégiques (Données Live)")
        
        # Global recommendation
        liste_recommandations = [rec['recommandation'] for rec in st.session_state.recommandations.values()]
        
        if liste_recommandations.count('TAUX VARIABLE') >= 2:
            strategie_globale = "TAUX VARIABLE"
            raison_globale = f"Majorité des scénarios montrent des taux en baisse depuis juin 2025 (1.75%)"
            couleur_globale = "#28a745"
        elif liste_recommandations.count('TAUX FIXE') >= 2:
            strategie_globale = "TAUX FIXE"
            raison_globale = f"Majorité des scénarios montrent des taux en hausse depuis juin 2025 (1.75%)"
            couleur_globale = "#dc3545"
        else:
            strategie_globale = "STRATÉGIE FLEXIBLE"
            raison_globale = f"Signaux mixtes depuis juin 2025 (1.75%) - approche diversifiée recommandée"
            couleur_globale = "#ffc107"
        
        # Data quality indicator
        quality_score = sum(1 for source in live_data['sources'].values() if 'Live' in source)
        quality_text = "Recommandation basée sur données majoritairement directes" if quality_score >= 2 else "Recommandation basée sur données mixtes" if quality_score >= 1 else "Recommandation basée sur estimations - à valider"
        
        st.markdown(f"""
        <div class="recommendation-box" style="background: linear-gradient(135deg, {couleur_globale} 0%, {couleur_globale}AA 100%);">
            <h2>🏆 RECOMMANDATION GLOBALE SOFAC</h2>
            <h3>{strategie_globale}</h3>
            <p>{raison_globale}</p>
            <small>{quality_text} ({quality_score}/4 sources directes)</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed scenario analysis
        st.subheader("📊 Analyse Détaillée par Scénario")
        
        for nom_scenario, rec in st.session_state.recommandations.items():
            with st.expander(f"📈 Scénario {nom_scenario}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Recommandation:** {rec['recommandation']}
                    
                    **Justification:** {rec['raison']}
                    
                    **Métriques (vs niveau live {live_data['yield_52w']:.2f}%):**
                    - Rendement moyen prédit: {rec['rendement_futur_moyen']:.2f}%
                    - Changement attendu: {rec['changement_rendement']:+.2f}%
                    - Volatilité: {rec['volatilite']:.2f}%
                    - Niveau de risque: {rec['niveau_risque']}
                    - Confiance: {rec['confiance']} ({rec['live_data_quality']})
                    """)
                
                with col2:
                    # Mini chart
                    fig_mini = go.Figure()
                    
                    pred_df = st.session_state.predictions[nom_scenario]
                    echantillon_mini = pred_df[::30]
                    
                    # Live baseline
                    fig_mini.add_hline(
                        y=live_data['yield_52w'], 
                        line_dash="dash", 
                        line_color="green",
                        annotation_text=f"Live: {live_data['yield_52w']:.2f}%"
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
                        yaxis_title="Rendement (%)"
                    )
                    
                    st.plotly_chart(fig_mini, use_container_width=True)
        
        # Financial impact calculator
        st.subheader("💰 Calculateur d'Impact Financier")
        
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
                💰 **Économies Potentielles avec TAUX VARIABLE:**
                
                - **Économies annuelles:** {abs(changement_cas_base) * montant_emprunt * 10_000:,.0f} MAD
                - **Économies totales ({duree_emprunt} ans):** {abs(impact_total):,.0f} MAD
                - **Basé sur:** Baisse attendue de {abs(changement_cas_base):.2f}% vs niveau live {live_data['yield_52w']:.2f}%
                - **Qualité prédiction:** {quality_score}/4 sources directes
                """)
            else:
                st.warning(f"""
                💰 **Coûts Évités avec TAUX FIXE:**
                
                - **Surcoûts évités annuellement:** {changement_cas_base * montant_emprunt * 10_000:,.0f} MAD
                - **Surcoûts évités totaux ({duree_emprunt} ans):** {impact_total:,.0f} MAD
                - **Basé sur:** Hausse attendue de {changement_cas_base:.2f}% vs niveau live {live_data['yield_52w']:.2f}%
                - **Qualité prédiction:** {quality_score}/4 sources directes
                """)
        else:
            st.info(f"""
            💰 **Impact Financier Limité:**
            
            - **Variation attendue:** ±{abs(changement_cas_base):.2f}% vs niveau live {live_data['yield_52w']:.2f}%
            - **Impact annuel:** ±{abs(changement_cas_base) * montant_emprunt * 10_000:,.0f} MAD
            - **Approche flexible recommandée**
            - **Qualité prédiction:** {quality_score}/4 sources directes
            """)
    
    # Footer
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    live_sources_count = sum(1 for source in live_data['sources'].values() if 'Live' in source)
    
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🇲🇦 <strong>SOFAC - Modèle de Prédiction des Rendements 52-Semaines avec Données Live</strong></p>
        <p>Sources: Bank Al-Maghrib ({live_sources_count > 0 and '🟢' or '🔴'}) | HCP ({live_sources_count > 1 and '🟢' or '🔴'}) | 
        Dernière mise à jour: {current_time}</p>
        <p>Qualité des données: {live_sources_count}/4 sources directes | 
        Prochaine actualisation automatique: {(datetime.now() + timedelta(hours=1)).strftime('%H:%M')}</p>
        <p><em>Les prédictions sont basées sur les dernières données disponibles et ne constituent pas des conseils financiers.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
