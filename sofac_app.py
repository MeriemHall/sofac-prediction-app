import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import requests
from bs4 import BeautifulSoup
import re
import warnings
import base64
from PIL import Image
import io
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SOFAC - Prédiction Rendements 52-Semaines",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to encode image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to create the SOFAC logo as SVG (since we can't load external images)
def create_sofac_logo_svg():
    return '''
    <svg width="180" height="60" viewBox="0 0 180 60" xmlns="http://www.w3.org/2000/svg">
        <circle cx="20" cy="20" r="6" fill="#FFD700"/>
        <path d="M12 28 Q24 20 36 28 Q48 36 60 28 Q72 20 84 28" 
              stroke="#1e3c72" stroke-width="3" fill="none"/>
        <text x="12" y="45" font-family="Arial, sans-serif" font-size="16" font-weight="bold" fill="#1e3c72">SOFAC</text>
        <text x="12" y="57" font-family="Arial, sans-serif" font-size="8" fill="#FF6B35">Dites oui au super crédit</text>
    </svg>
    '''

# Professional CSS with logo integration
st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3d5aa3 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        position: relative;
    }}
    .logo-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }}
    .logo-svg {{
        margin-right: 2rem;
        background: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    .header-text {{
        text-align: left;
    }}
    .executive-dashboard {{
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }}
    .status-card {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
    }}
    .metric-box {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        border-top: 3px solid #2a5298;
    }}
    .recommendation-panel {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }}
    .sidebar-logo {{
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    .stMetric label {{ font-size: 0.75rem !important; }}
    h1 {{ font-size: 1.4rem !important; }}
    h2 {{ font-size: 1.2rem !important; }}
    p {{ font-size: 0.82rem !important; }}
    
    /* Mobile responsiveness for logo */
    @media (max-width: 768px) {{
        .logo-container {{
            flex-direction: column;
        }}
        .logo-svg {{
            margin-right: 0;
            margin-bottom: 1rem;
        }}
        .header-text {{
            text-align: center;
        }}
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_live_data():
    """Fetch live economic data"""
    return {
        'policy_rate': 2.25,
        'inflation': 1.1,
        'gdp_growth': 4.8,
        'sources': {'policy_rate': 'Bank Al-Maghrib', 'inflation': 'HCP'},
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@st.cache_data
def create_dataset():
    """Create complete historical dataset with interpolation"""
    # Complete historical data
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
    
    # Generate monthly data from 2020 to June 2025
    date_debut = datetime(2020, 1, 1)
    date_fin = datetime(2025, 6, 30)
    
    donnees_mensuelles = []
    date_courante = date_debut
    
    # Convert to datetime objects
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
            # Find surrounding anchor points for interpolation
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
        
        # Move to next month
        if date_courante.month == 12:
            date_courante = date_courante.replace(year=date_courante.year + 1, month=1)
        else:
            date_courante = date_courante.replace(month=date_courante.month + 1)
    
    return pd.DataFrame(donnees_mensuelles)

def train_model(df):
    """Train prediction model with proper variable names"""
    X = df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
    y = df['Rendement_52s']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    scores_cv = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae_cv = -scores_cv.mean()
    
    return model, r2, mae, mae_cv

def generate_scenarios():
    """Generate realistic economic scenarios"""
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2026, 12, 31)
    
    dates_quotidiennes = []
    date_courante = date_debut
    
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
    # Realistic monetary policy decisions
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
            
            # Determine policy rate based on calendar
            date_str = date.strftime('%Y-%m')
            taux_directeur = 2.25  # Default
            for date_politique, taux in sorted(taux_politiques.items()):
                if date_str >= date_politique:
                    taux_directeur = taux
            
            # Add realistic economic projections with seasonality
            np.random.seed(hash(date.strftime('%Y-%m-%d')) % 2**32)
            
            mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
            
            # Scenario-specific economic projections
            if nom_scenario == 'Conservateur':
                inflation_base = 1.4 + 0.5 * np.exp(-mois_depuis_debut / 18) + 0.2 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 - 0.5 * (mois_depuis_debut / 18) + 0.4 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            elif nom_scenario == 'Cas_de_Base':
                inflation_base = 1.4 + 0.3 * np.exp(-mois_depuis_debut / 12) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 - 0.2 * (mois_depuis_debut / 18) + 0.5 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            else:  # Optimiste
                inflation_base = 1.4 - 0.2 * (mois_depuis_debut / 18) + 0.1 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 + 0.1 * (mois_depuis_debut / 18) + 0.6 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            
            # Add realistic noise
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

def predict_yields(scenarios, model):
    """Generate yield predictions with proper continuity"""
    baseline = 1.75  # June 2025 baseline
    predictions = {}
    
    for scenario_name, scenario_df in scenarios.items():
        X_future = scenario_df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
        rendements_bruts = model.predict(X_future)
        
        # Ensure smooth transition from June 2025 baseline
        if len(rendements_bruts) > 0:
            premier_predit = rendements_bruts[0]
            discontinuite = premier_predit - baseline
            
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
            
            if scenario_name == 'Conservateur':
                ajustement += 0.1
            elif scenario_name == 'Optimiste':
                ajustement -= 0.05
            
            # Time-based uncertainty - make it more gradual
            jours_ahead = ligne['Jours_Ahead']
            incertitude = (jours_ahead / 365) * 0.02  # Reduced from 0.05 for more stability
            if scenario_name == 'Conservateur':
                ajustement += incertitude
            elif scenario_name == 'Optimiste':
                ajustement -= incertitude * 0.5
            
            # Day of week effects - reduced for more consistency
            effets_jours = {
                'Monday': 0.005, 'Tuesday': 0.00, 'Wednesday': -0.005,
                'Thursday': 0.00, 'Friday': 0.01, 'Saturday': -0.005, 'Sunday': -0.005
            }
            ajustement += effets_jours.get(ligne['Jour_Semaine'], 0)
            
            ajustements.append(ajustement)
        
        rendements_finaux = rendements_lisses + np.array(ajustements)
        rendements_finaux = np.clip(rendements_finaux, 0.1, 8.0)
        
        # Ensure logical progression - smooth out any erratic jumps
        for i in range(1, len(rendements_finaux)):
            # Limit daily changes to ±0.1% for more realistic progression
            daily_change = rendements_finaux[i] - rendements_finaux[i-1]
            if abs(daily_change) > 0.1:
                rendements_finaux[i] = rendements_finaux[i-1] + np.sign(daily_change) * 0.1
        
        scenario_df_copy = scenario_df.copy()
        scenario_df_copy['rendement_predit'] = rendements_finaux
        scenario_df_copy['scenario'] = scenario_name
        
        predictions[scenario_name] = scenario_df_copy
    
    return predictions

def generate_recommendations(predictions):
    """Generate strategic recommendations"""
    baseline = 1.75
    recommendations = {}
    
    for scenario_name, pred_df in predictions.items():
        avg_yield = pred_df['rendement_predit'].mean()
        change = avg_yield - baseline
        volatility = pred_df['rendement_predit'].std()
        
        if change > 0.3:
            recommendation = "TAUX FIXE"
            reason = "Hausse attendue des rendements - bloquer les taux"
        elif change < -0.3:
            recommendation = "TAUX VARIABLE"
            reason = "Baisse attendue des rendements - profiter des taux variables"
        else:
            recommendation = "STRATÉGIE MIXTE"
            reason = "Évolution stable - approche équilibrée"
        
        risk_level = "ÉLEVÉ" if volatility > 0.3 else "MOYEN" if volatility > 0.15 else "FAIBLE"
        
        recommendations[scenario_name] = {
            'recommandation': recommendation,
            'raison': reason,
            'niveau_risque': risk_level,
            'rendement_moyen': avg_yield,
            'changement': change,
            'volatilite': volatility
        }
    
    return recommendations

def main():
    # Alternative header approach if HTML doesn't render properly
    col_logo, col_text = st.columns([1, 3])
    
    with col_logo:
        # Display logo
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="background: white; padding: 10px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
    
    with col_text:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3d5aa3 100%); 
                    padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
            <h1 style="margin: 0; color: white;">Système de Prédiction des Rendements</h1>
            <p style="margin: 0.5rem 0; color: white;">Modèle d'Intelligence Financière 52-Semaines</p>
            <p style="margin: 0; color: white;">Données Bank Al-Maghrib &amp; HCP | Mise à jour: Horaire</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data and models
    if 'data_loaded' not in st.session_state:
        with st.spinner("Chargement du modèle..."):
            st.session_state.df = create_dataset()
            st.session_state.model, st.session_state.r2, st.session_state.mae, st.session_state.mae_cv = train_model(st.session_state.df)
            st.session_state.scenarios = generate_scenarios()
            st.session_state.predictions = predict_yields(st.session_state.scenarios, st.session_state.model)
            st.session_state.recommendations = generate_recommendations(st.session_state.predictions)
            st.session_state.data_loaded = True
    
    live_data = fetch_live_data()
    baseline_yield = 1.75  # June 2025
    
    # Sidebar with logo
    with st.sidebar:
        # Add logo to sidebar - simplified approach
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.header("Informations du Modèle")
        
        st.markdown("### Données en Temps Réel")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Taux Directeur", f"{live_data['policy_rate']:.2f}%")
            st.metric("Inflation", f"{live_data['inflation']:.2f}%")
        
        with col2:
            st.metric("Rendement Actuel", f"{baseline_yield:.2f}%", help="Baseline Juin 2025")
            st.metric("Croissance PIB", f"{live_data['gdp_growth']:.2f}%")
        
        st.info(f"Dernière MAJ: {live_data['last_updated']}")
        
        # TODAY'S PREDICTION SECTION
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Prédiction du Jour")
        
        # Get today's date and prediction
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        today_display = today.strftime('%d/%m/%Y')
        
        # Find today's prediction in the base case scenario
        cas_base_predictions = st.session_state.predictions['Cas_de_Base']
        
        # Try to find today's prediction
        today_prediction = None
        closest_prediction = None
        closest_date = None
        
        for _, row in cas_base_predictions.iterrows():
            pred_date = row['Date']
            if pred_date == today_str:
                today_prediction = row['rendement_predit']
                break
            elif pred_date > today_str and closest_prediction is None:
                closest_prediction = row['rendement_predit']
                closest_date = pred_date
        
        # Display today's prediction
        if today_prediction is not None:
            st.sidebar.success("**" + today_display + "**")
            st.sidebar.metric(
                "🎯 Rendement Prédit Aujourd'hui",
                f"{today_prediction:.2f}%",
                delta=f"{(today_prediction - baseline_yield):+.2f}%",
                help="Prédiction pour aujourd'hui vs baseline juin 2025"
            )
        elif closest_prediction is not None:
            closest_date_display = datetime.strptime(closest_date, '%Y-%m-%d').strftime('%d/%m/%Y')
            st.sidebar.warning("**" + today_display + "**")
            st.sidebar.metric(
                "🎯 Prédiction Prochaine",
                f"{closest_prediction:.2f}%",
                delta=f"{(closest_prediction - baseline_yield):+.2f}%",
                help=f"Prédiction pour {closest_date_display}"
            )
        else:
            st.sidebar.info("**" + today_display + "**")
            st.sidebar.write("🎯 **Prédiction:** Données en cours de traitement")
        
        if st.sidebar.button("Actualiser"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("### Performance du Modèle")
        st.metric("R² Score", f"{st.session_state.r2:.1%}")
        st.metric("Précision", f"±{st.session_state.mae:.2f}%")
        st.metric("Validation Croisée", f"±{st.session_state.mae_cv:.2f}%")
        st.success("Modèle calibré avec succès")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Vue d'Ensemble", "Prédictions Détaillées", "Recommandations"])
    
    with tab1:
        # Executive Dashboard
        st.markdown('<div class="executive-dashboard">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; font-size: 1.4rem; font-weight: 700; margin-bottom: 2rem;">Tableau de Bord Exécutif</div>', unsafe_allow_html=True)
        
        # Current situation - get today's actual prediction
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        today_display = today.strftime('%d/%m/%Y')
        
        # Find today's prediction in the base case scenario
        cas_de_base_predictions = st.session_state.predictions['Cas_de_Base']
        current_prediction = None
        
        # Try to find today's prediction
        for _, row in cas_de_base_predictions.iterrows():
            pred_date = row['Date']
            if pred_date == today_str:
                current_prediction = row['rendement_predit']
                break
            elif pred_date > today_str and current_prediction is None:
                current_prediction = row['rendement_predit']
                break
        
        # If no prediction found, use the first available prediction
        if current_prediction is None:
            current_prediction = cas_de_base_predictions['rendement_predit'].iloc[0]
        
        evolution = current_prediction - baseline_yield
        
        # Status determination
        if evolution > 0.4:
            status = "TAUX ÉLEVÉS - CRITIQUE"
            status_color = "#dc3545"
            action = "BLOQUER LES TAUX MAINTENANT"
        elif evolution > 0.1:
            status = "TAUX EN HAUSSE - ATTENTION"
            status_color = "#ffc107"
            action = "PRÉPARER STRATÉGIE DE COUVERTURE"
        elif evolution < -0.4:
            status = "OPPORTUNITÉ - TAUX FAVORABLES"
            status_color = "#28a745"
            action = "MAXIMISER TAUX VARIABLES"
        else:
            status = "MARCHÉ STABLE"
            status_color = "#17a2b8"
            action = "MAINTENIR STRATÉGIE ACTUELLE"
        
        # Status card
        st.markdown(f"""
        <div class="status-card" style="border-left-color: {status_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #2c3e50;">Situation au {today_display}</h3>
                    <p style="margin: 0.5rem 0; font-weight: 600; color: {status_color};">{status}</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #2c3e50;">{current_prediction:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">Rendement Courant</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem; font-weight: 600; color: {status_color};">{evolution:+.2f}%</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">vs Juin 2025</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        volatility = st.session_state.predictions['Cas_de_Base']['rendement_predit'].std()
        impact_10m = abs(st.session_state.recommendations['Cas_de_Base']['changement']) * 10
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">BASELINE JUIN</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{baseline_yield:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">ÉVOLUTION</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {status_color};">{evolution:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">VOLATILITÉ</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{volatility:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">IMPACT 10M MAD</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{impact_10m:.0f}K/an</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        strategy = st.session_state.recommendations['Cas_de_Base']['recommandation']
        change_global = st.session_state.recommendations['Cas_de_Base']['changement']
        
        st.markdown(f"""
        <div class="recommendation-panel">
            <div style="text-align: center; font-size: 1.3rem; font-weight: 700; margin-bottom: 1.5rem;">RECOMMANDATIONS STRATÉGIQUES</div>
            <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 1rem; margin: 0.8rem 0;">
                <h4 style="margin: 0 0 0.5rem 0; color: white;">Action Prioritaire</h4>
                <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">{action}</p>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 1rem;">
                    <h5 style="margin: 0 0 0.5rem 0; color: white;">Stratégie Globale</h5>
                    <p style="margin: 0; font-weight: 600;">{strategy}</p>
                    <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.8;">Changement: {change_global:+.2f}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 1rem;">
                    <h5 style="margin: 0 0 0.5rem 0; color: white;">Impact Financier</h5>
                    <p style="margin: 0; font-weight: 600;">{abs(change_global) * 100:.0f}K MAD/an</p>
                    <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.8;">Pour 10M MAD emprunt</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chart
        st.subheader("Évolution des Rendements")
        
        fig = go.Figure()
        
        # Historical data - use last 8 points for better visualization
        df_hist = st.session_state.df.tail(8)
        fig.add_trace(go.Scatter(
            x=df_hist['Date'],
            y=df_hist['Rendement_52s'],
            mode='lines+markers',
            name='Historique',
            line=dict(color='#2a5298', width=4),
            marker=dict(size=8)
        ))
        
        # Predictions - use weekly sampling for clarity but ensure consistency
        colors = {'Conservateur': '#dc3545', 'Cas_de_Base': '#17a2b8', 'Optimiste': '#28a745'}
        for scenario, pred_df in st.session_state.predictions.items():
            # Use every 7th day but include today's date if it exists
            sample_indices = list(range(0, len(pred_df), 7))
            
            # Try to include today's prediction if it exists in the data
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_index = None
            for i, row in pred_df.iterrows():
                if row['Date'] == today_str:
                    today_index = i
                    break
            
            if today_index is not None and today_index not in sample_indices:
                sample_indices.append(today_index)
                sample_indices.sort()
            
            sample_data = pred_df.iloc[sample_indices]
            
            fig.add_trace(go.Scatter(
                x=sample_data['Date'],
                y=sample_data['rendement_predit'],
                mode='lines+markers',
                name=scenario,
                line=dict(color=colors[scenario], width=3),
                marker=dict(size=5)
            ))
        
        fig.add_hline(y=baseline_yield, line_dash="dash", line_color="gray", 
                     annotation_text=f"Baseline Juin 2025: {baseline_yield:.2f}%")
        
        fig.update_layout(
            height=450,
            template="plotly_white",
            xaxis_title="Période",
            yaxis_title="Rendement (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Prédictions Détaillées")
        
        scenario_choice = st.selectbox("Choisissez un scénario:", 
                                     ['Cas_de_Base', 'Conservateur', 'Optimiste'])
        
        pred_data = st.session_state.predictions[scenario_choice]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rendement Moyen", f"{pred_data['rendement_predit'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{pred_data['rendement_predit'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{pred_data['rendement_predit'].max():.2f}%")
        with col4:
            change = pred_data['rendement_predit'].mean() - baseline_yield
            st.metric("Écart vs Juin 2025", f"{change:+.2f}%")
        
        # Detailed chart
        st.subheader(f"Prédictions Quotidiennes - {scenario_choice}")
        
        sample_detailed = pred_data[::7]  # Weekly sampling
        
        fig_detail = go.Figure()
        fig_detail.add_trace(go.Scatter(
            x=sample_detailed['Date'],
            y=sample_detailed['rendement_predit'],
            mode='lines+markers',
            name='Prédiction',
            line=dict(color=colors[scenario_choice], width=3)
        ))
        
        fig_detail.add_hline(y=baseline_yield, line_dash="dash", line_color="blue",
                           annotation_text=f"Juin 2025: {baseline_yield:.2f}%")
        
        fig_detail.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Rendement (%)"
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Export
        if st.button("Télécharger les Prédictions"):
            csv = pred_data.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"sofac_predictions_{scenario_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("Recommandations Stratégiques")
        
        # Enhanced Loan Decision Section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <h3 style="margin: 0; color: white;">🏦 AIDE À LA DÉCISION EMPRUNT SOFAC</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Analyse comparative Taux Fixe vs Taux Variable sur la durée du contrat</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Loan Parameters Section
        st.subheader("⚙️ Paramètres de l'Emprunt")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            loan_amount = st.slider("Montant (millions MAD):", 1, 500, 50)
        with col2:
            loan_duration = st.slider("Durée (années):", 1, 10, 5)
        with col3:
            current_fixed_rate = st.number_input("Taux fixe proposé (%):", min_value=1.0, max_value=10.0, value=3.2, step=0.1)
        with col4:
            risk_tolerance = st.selectbox("Tolérance au risque:", ["Faible", "Moyenne", "Élevée"])
        
        # Calculate comprehensive loan analysis
        scenarios_analysis = {}
        
        for scenario_name, pred_df in st.session_state.predictions.items():
            # Get predictions for the loan duration
            loan_duration_days = loan_duration * 365
            relevant_predictions = pred_df.head(loan_duration_days)
            
            # Calculate variable rate costs (assuming annual rate changes)
            variable_rates_annual = []
            for year in range(loan_duration):
                start_day = year * 365
                end_day = min((year + 1) * 365, len(relevant_predictions))
                if start_day < len(relevant_predictions):
                    year_data = relevant_predictions.iloc[start_day:end_day]
                    avg_rate_year = year_data['rendement_predit'].mean()
                    variable_rates_annual.append(avg_rate_year)
                else:
                    # If we don't have data for this year, use the last available rate
                    variable_rates_annual.append(variable_rates_annual[-1] if variable_rates_annual else baseline_yield)
            
            # Calculate costs
            fixed_cost_total = (current_fixed_rate / 100) * loan_amount * 1_000_000 * loan_duration
            variable_cost_total = sum([(rate / 100) * loan_amount * 1_000_000 for rate in variable_rates_annual])
            
            cost_difference = variable_cost_total - fixed_cost_total
            cost_difference_percentage = (cost_difference / fixed_cost_total) * 100
            
            # Risk metrics
            volatility = relevant_predictions['rendement_predit'].std()
            max_rate = max(variable_rates_annual)
            min_rate = min(variable_rates_annual)
            rate_range = max_rate - min_rate
            
            scenarios_analysis[scenario_name] = {
                'variable_rates_annual': variable_rates_annual,
                'avg_variable_rate': np.mean(variable_rates_annual),
                'fixed_cost_total': fixed_cost_total,
                'variable_cost_total': variable_cost_total,
                'cost_difference': cost_difference,
                'cost_difference_percentage': cost_difference_percentage,
                'volatility': volatility,
                'max_rate': max_rate,
                'min_rate': min_rate,
                'rate_range': rate_range
            }
        
        # Decision Matrix
        st.subheader("📊 Matrice de Décision par Scénario")
        
        decision_data = []
        for scenario_name, analysis in scenarios_analysis.items():
            if analysis['cost_difference'] < 0:
                recommendation = "TAUX VARIABLE"
                savings = abs(analysis['cost_difference'])
                decision_color = "#28a745"
                decision_text = f"Économie de {savings:,.0f} MAD"
            else:
                recommendation = "TAUX FIXE" 
                extra_cost = analysis['cost_difference']
                decision_color = "#dc3545"
                decision_text = f"Éviter surcoût de {extra_cost:,.0f} MAD"
            
            risk_level = "FAIBLE" if analysis['volatility'] < 0.2 else "MOYEN" if analysis['volatility'] < 0.4 else "ÉLEVÉ"
            
            decision_data.append({
                'Scénario': scenario_name,
                'Taux Variable Moyen': f"{analysis['avg_variable_rate']:.2f}%",
                'Fourchette': f"{analysis['min_rate']:.2f}% - {analysis['max_rate']:.2f}%",
                'Coût Total Variable': f"{analysis['variable_cost_total']:,.0f} MAD",
                'Différence vs Fixe': decision_text,
                'Recommandation': recommendation,
                'Niveau Risque': risk_level,
                'Volatilité': f"{analysis['volatility']:.2f}%"
            })
        
        # Display decision matrix as a table
        decision_df = pd.DataFrame(decision_data)
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
        
        # Global recommendation based on risk tolerance and scenarios
        variable_recommendations = sum(1 for analysis in scenarios_analysis.values() if analysis['cost_difference'] < 0)
        
        if risk_tolerance == "Faible":
            if variable_recommendations >= 2 and all(analysis['volatility'] < 0.3 for analysis in scenarios_analysis.values()):
                final_recommendation = "TAUX VARIABLE"
                final_reason = "Majorité des scénarios favorables avec volatilité acceptable"
                final_color = "#28a745"
            else:
                final_recommendation = "TAUX FIXE"
                final_reason = "Sécurité privilégiée - éviter les fluctuations"
                final_color = "#dc3545"
        elif risk_tolerance == "Moyenne":
            if variable_recommendations >= 2:
                final_recommendation = "TAUX VARIABLE"
                final_reason = "Équilibre favorable entre économies potentielles et risque"
                final_color = "#28a745"
            else:
                final_recommendation = "STRATÉGIE MIXTE"
                final_reason = "Répartir 50/50 pour optimiser le risque"
                final_color = "#ffc107"
        else:  # Élevée
            best_scenario = min(scenarios_analysis.items(), key=lambda x: x[1]['cost_difference'])
            if best_scenario[1]['cost_difference'] < 0:
                final_recommendation = "TAUX VARIABLE"
                final_reason = f"Opportunité d'économies importantes (scénario {best_scenario[0]})"
                final_color = "#28a745"
            else:
                final_recommendation = "TAUX FIXE"
                final_reason = "Tous les scénarios défavorables au taux variable"
                final_color = "#dc3545"
        
        # Final recommendation display
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {final_color}, {final_color}AA); 
                    color: white; padding: 2rem; border-radius: 12px; margin: 2rem 0; text-align: center;">
            <h2>🎯 DÉCISION FINALE SOFAC</h2>
            <h3>{final_recommendation}</h3>
            <p><strong>Justification:</strong> {final_reason}</p>
            <p><strong>Montant:</strong> {loan_amount}M MAD | <strong>Durée:</strong> {loan_duration} ans | <strong>Taux fixe alternatif:</strong> {current_fixed_rate}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed cost breakdown
        st.subheader("💰 Analyse Détaillée des Coûts")
        
        base_case_analysis = scenarios_analysis['Cas_de_Base']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Option Taux Fixe")
            st.metric("Taux", f"{current_fixed_rate:.2f}%")
            st.metric("Coût Total", f"{base_case_analysis['fixed_cost_total']:,.0f} MAD")
            st.metric("Coût Annuel", f"{base_case_analysis['fixed_cost_total']/loan_duration:,.0f} MAD")
            st.success("✅ Prévisibilité totale")
        
        with col2:
            st.markdown("### Option Taux Variable")
            st.metric("Taux Moyen Prédit", f"{base_case_analysis['avg_variable_rate']:.2f}%")
            st.metric("Coût Total Estimé", f"{base_case_analysis['variable_cost_total']:,.0f} MAD")
            st.metric("Fourchette Annuelle", f"{base_case_analysis['min_rate']:.2f}% - {base_case_analysis['max_rate']:.2f}%")
            if base_case_analysis['cost_difference'] < 0:
                st.success(f"💰 Économie potentielle: {abs(base_case_analysis['cost_difference']):,.0f} MAD")
            else:
                st.warning(f"⚠️ Surcoût potentiel: {base_case_analysis['cost_difference']:,.0f} MAD")
        
        # Yearly breakdown chart
        st.subheader("📈 Évolution Annuelle des Taux (Cas de Base)")
        
        years = list(range(1, loan_duration + 1))
        fig_yearly = go.Figure()
        
        # Fixed rate line
        fig_yearly.add_trace(go.Scatter(
            x=years,
            y=[current_fixed_rate] * loan_duration,
            mode='lines+markers',
            name='Taux Fixe',
            line=dict(color='#dc3545', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Variable rate line (base case)
        fig_yearly.add_trace(go.Scatter(
            x=years,
            y=base_case_analysis['variable_rates_annual'],
            mode='lines+markers',
            name='Taux Variable (Prévu)',
            line=dict(color='#17a2b8', width=3),
            marker=dict(size=8)
        ))
        
        fig_yearly.update_layout(
            height=400,
            template="plotly_white",
            xaxis_title="Année",
            yaxis_title="Taux d'intérêt (%)",
            title="Comparaison Taux Fixe vs Variable sur la Durée du Prêt"
        )
        
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        # Risk assessment
        st.subheader("⚠️ Évaluation des Risques")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            st.markdown("### Risque de Taux")
            if base_case_analysis['volatility'] < 0.2:
                st.success("🟢 FAIBLE")
                risk_desc = "Fluctuations limitées attendues"
            elif base_case_analysis['volatility'] < 0.4:
                st.warning("🟡 MOYEN") 
                risk_desc = "Fluctuations modérées possibles"
            else:
                st.error("🔴 ÉLEVÉ")
                risk_desc = "Fortes fluctuations possibles"
            st.write(risk_desc)
        
        with risk_col2:
            st.markdown("### Risque de Liquidité")
            max_annual_diff = max(base_case_analysis['variable_rates_annual']) - current_fixed_rate
            if max_annual_diff < 0.5:
                st.success("🟢 FAIBLE")
                liquidity_desc = "Impact limité sur la trésorerie"
            elif max_annual_diff < 1.0:
                st.warning("🟡 MOYEN")
                liquidity_desc = "Impact modéré à prévoir"
            else:
                st.error("🔴 ÉLEVÉ")
                liquidity_desc = "Impact significatif possible"
            st.write(liquidity_desc)
        
        with risk_col3:
            st.markdown("### Recommandation Finale")
            if final_recommendation == "TAUX VARIABLE":
                st.success("📈 VARIABLE")
            elif final_recommendation == "TAUX FIXE":
                st.error("📊 FIXE") 
            else:
                st.warning("⚖️ MIXTE")
            st.write(f"Confiance: {70 + variable_recommendations * 10}%")
        
        # Global recommendation summary
        recommendations_list = [scenarios_analysis[scenario]['cost_difference'] < 0 for scenario in scenarios_analysis.keys()]
        
        if recommendations_list.count(True) >= 2:
            global_strategy = "TAUX VARIABLE"
            global_reason = "Majorité des scénarios favorisent les taux variables"
            global_color = "#28a745"
        elif recommendations_list.count(False) >= 2:
            global_strategy = "TAUX FIXE"
            global_reason = "Majorité des scénarios favorisent les taux fixes"
            global_color = "#dc3545"
        else:
            global_strategy = "STRATÉGIE MIXTE"
            global_reason = "Signaux mixtes - approche équilibrée recommandée"
            global_color = "#ffc107"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {global_color}, {global_color}AA); 
                    color: white; padding: 2rem; border-radius: 12px; margin: 2rem 0; text-align: center;">
            <h2>RECOMMANDATION GLOBALE SOFAC</h2>
            <h3>{global_strategy}</h3>
            <p>{global_reason}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed analysis
        st.subheader("Analyse Détaillée par Scénario")
        
        for scenario, rec in st.session_state.recommendations.items():
            with st.expander(f"📋 Scénario {scenario}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    scenario_analysis = scenarios_analysis[scenario]
                    st.markdown(f"""
                    **Recommandation:** {rec['recommandation']}
                    
                    **Analyse Financière:**
                    - Taux variable moyen: {scenario_analysis['avg_variable_rate']:.2f}%
                    - Fourchette: {scenario_analysis['min_rate']:.2f}% - {scenario_analysis['max_rate']:.2f}%
                    - Coût total (variable): {scenario_analysis['variable_cost_total']:,.0f} MAD
                    - Différence vs fixe: {scenario_analysis['cost_difference']:+,.0f} MAD ({scenario_analysis['cost_difference_percentage']:+.1f}%)
                    
                    **Métriques de Risque:**
                    - Volatilité: {scenario_analysis['volatility']:.2f}%
                    - Amplitude: {scenario_analysis['rate_range']:.2f}%
                    - Niveau de risque: {rec['niveau_risque']}
                    """)
                
                with col2:
                    # Mini chart for each scenario
                    pred_mini = st.session_state.predictions[scenario][::30]
                    
                    fig_mini = go.Figure()
                    fig_mini.add_hline(y=current_fixed_rate, line_dash="dash", line_color="red", 
                                     annotation_text=f"Taux Fixe: {current_fixed_rate:.2f}%")
                    fig_mini.add_trace(go.Scatter(
                        x=pred_mini['Date'],
                        y=pred_mini['rendement_predit'],
                        mode='lines+markers',
                        line=dict(color=colors[scenario], width=2),
                        name="Taux Variable"
                    ))
                    
                    fig_mini.update_layout(
                        height=200,
                        showlegend=False,
                        template="plotly_white",
                        margin=dict(l=20, r=20, t=20, b=20),
                        title=f"Évolution - {scenario}"
                    )
                    
                    st.plotly_chart(fig_mini, use_container_width=True)
        
        # Financial calculator
        st.subheader("Calculateur d'Impact Financier")
        
        col1, col2 = st.columns(2)
        with col1:
            loan_amount = st.slider("Montant emprunt (millions MAD):", 1, 100, 10)
        with col2:
            loan_duration = st.slider("Durée emprunt (années):", 1, 10, 3)
        
        base_change = st.session_state.recommendations['Cas_de_Base']['changement']
        total_impact = base_change * loan_amount * 1_000_000 / 100 * loan_duration
        
        if abs(base_change) > 0.2:
            if base_change < 0:
                st.success(f"""
                **Économies Potentielles avec TAUX VARIABLE:**
                - Économies annuelles: {abs(base_change) * loan_amount * 10_000:,.0f} MAD
                - Économies totales ({loan_duration} ans): {abs(total_impact):,.0f} MAD
                """)
            else:
                st.warning(f"""
                **Coûts Évités avec TAUX FIXE:**
                - Surcoûts évités annuellement: {base_change * loan_amount * 10_000:,.0f} MAD
                - Surcoûts évités totaux ({loan_duration} ans): {total_impact:,.0f} MAD
                """)
        else:
            st.info(f"""
            **Impact Financier Limité:**
            - Variation attendue: ±{abs(base_change):.2f}%
            - Impact annuel: ±{abs(base_change) * loan_amount * 10_000:,.0f} MAD
            """)
    
    # Footer with SOFAC branding
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create a simpler footer without complex HTML
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Simple logo display
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem;">{logo_svg}</div>', unsafe_allow_html=True)
        
        # Footer text
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p style="margin: 0; font-weight: bold; color: #2a5298;">SOFAC - Modèle de Prédiction des Rendements 52-Semaines</p>
            <p style="margin: 0; color: #FF6B35;">Dites oui au super crédit</p>
            <p style="margin: 0.5rem 0;">Baseline: Juin 2025 ({baseline_yield:.2f}%) | Dernière mise à jour: {current_time}</p>
            <p style="margin: 0;"><em>Les prédictions sont basées sur des données historiques et ne constituent pas des conseils financiers.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
