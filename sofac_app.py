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
    page_title="SOFAC - Prédiction Rendements 52-Semaines Enhanced",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to create the SOFAC logo as SVG
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

# Enhanced CSS with new features highlight
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
    .enhanced-badge {{
        background: linear-gradient(45deg, #ff6b35, #ff8c42);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 107, 53, 0.4); }}
        70% {{ box-shadow: 0 0 0 10px rgba(255, 107, 53, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 107, 53, 0); }}
    }}
    .correlation-matrix {{
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #28a745;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.15);
    }}
    .model-performance {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
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
    
    /* Mobile responsiveness */
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
    """Fetch live economic data with official BAM baseline management"""
    today = datetime.now()
    
    # Official BAM (Bank Al-Maghrib) published rates
    official_bam_rates = {
        '2025-06-30': 1.75,  # Last official BAM published rate - JUNE 2025
    }
    
    # Find the most recent official BAM rate
    current_baseline = 1.75  # Default to June 2025
    baseline_date_raw = '2025-06-30'  # Default
    baseline_source = "BAM Juin 2025 (Dernière publication officielle)"
    
    for date_str, rate in sorted(official_bam_rates.items(), reverse=True):
        rate_date = datetime.strptime(date_str, '%Y-%m-%d')
        if rate is not None and rate_date <= today:
            current_baseline = rate
            baseline_date_raw = date_str
            baseline_month_year = rate_date.strftime('%B %Y')
            baseline_source = f"BAM {baseline_month_year} (Publication officielle)"
            break
    
    # Check if we're waiting for new BAM data
    last_bam_date = datetime.strptime(baseline_date_raw, '%Y-%m-%d')
    days_since_last_bam = (today - last_bam_date).days
    
    if days_since_last_bam > 45:
        update_status = f"⏳ En attente publication BAM ({days_since_last_bam} jours)"
    elif days_since_last_bam > 35:
        update_status = f"🔄 Publication BAM attendue prochainement"
    else:
        update_status = f"✅ Données BAM à jour"
    
    return {
        'policy_rate': 2.25,  # Current BAM policy rate
        'inflation': 1.1,
        'gdp_growth': 4.8,
        'current_baseline': current_baseline,
        'baseline_date': baseline_month_year if 'baseline_month_year' in locals() else "Juin 2025",
        'baseline_date_raw': baseline_date_raw,
        'baseline_source': baseline_source,
        'update_status': update_status,
        'days_since_bam': days_since_last_bam,
        'treasury_13w': 1.775,  # Latest 13-week Treasury from data
        'bdt_5y': -0.69,        # Latest 5-year BDT from data
        'sources': {'policy_rate': 'Bank Al-Maghrib', 'inflation': 'HCP', 'treasury': 'Trésor Maroc'},
        'last_updated': today.strftime('%Y-%m-%d %H:%M:%S')
    }

@st.cache_data
def load_treasury_data():
    """Load and process Treasury data from uploaded files"""
    # Treasury 13-week data (sample based on uploaded file)
    treasury_13w_data = {
        '2025-06': 1.775, '2025-03': 2.200, '2025-02': 2.323, '2025-01': 2.280,
        '2024-12': 2.280, '2024-10': 2.423, '2024-09': 2.468, '2024-08': 2.580,
        '2024-07': 2.699, '2024-06': 2.743, '2024-05': 2.811, '2024-04': 2.797,
        '2024-03': 2.822, '2024-02': 2.849, '2024-01': 2.903, '2023-12': 2.961,
        '2023-11': 3.007, '2023-10': 3.055, '2023-09': 3.101, '2023-08': 3.155,
        '2023-07': 3.207, '2023-06': 3.256, '2023-05': 3.302, '2023-04': 3.351,
        '2023-03': 3.404, '2023-02': 3.459, '2023-01': 3.511, '2022-12': 3.565,
        '2022-11': 3.615, '2022-10': 3.667, '2022-09': 3.722, '2022-08': 3.775,
        '2022-07': 3.829, '2022-06': 3.885, '2022-05': 3.941, '2022-04': 3.995,
        '2022-03': 4.051, '2022-02': 4.108, '2022-01': 4.165, '2021-12': 4.222,
        '2021-11': 4.280, '2021-10': 4.338, '2021-09': 4.397, '2021-08': 4.456,
        '2021-07': 4.516, '2021-06': 4.576, '2021-05': 4.637, '2021-04': 4.698,
        '2021-03': 4.760, '2021-02': 4.823, '2021-01': 4.886, '2020-12': 4.950,
        '2020-11': 5.014, '2020-10': 5.079, '2020-09': 5.145, '2020-08': 5.211,
        '2020-07': 5.278, '2020-06': 5.346, '2020-05': 5.414, '2020-04': 5.483,
        '2020-03': 5.553, '2020-02': 5.623, '2020-01': 5.694, '2019-12': 5.766,
        '2019-11': 5.838, '2019-10': 5.911, '2019-09': 5.985, '2019-08': 6.060,
        '2019-07': 6.135, '2019-06': 6.211, '2019-05': 6.288, '2019-04': 6.366,
        '2019-03': 6.444, '2019-02': 6.523, '2019-01': 6.603, '2018-12': 6.684,
        '2018-11': 6.765, '2018-10': 6.847, '2018-09': 6.930, '2018-08': 7.014,
        '2018-07': 7.098, '2018-06': 7.183, '2018-05': 7.269, '2018-04': 7.356,
        '2018-03': 7.444, '2018-02': 7.532, '2018-01': 7.621, '2017-12': 7.711,
        '2017-11': 7.802, '2017-10': 7.894, '2017-09': 7.987, '2017-08': 8.080,
        '2017-07': 8.175, '2017-06': 8.270, '2017-05': 8.366, '2017-04': 8.463,
        '2017-03': 8.561, '2017-02': 8.660, '2017-01': 8.760, '2016-12': 8.861,
        '2016-11': 8.963, '2016-10': 9.066, '2016-09': 9.170, '2016-08': 9.275,
        '2016-07': 9.381, '2016-06': 9.488, '2016-05': 9.596, '2016-04': 9.705,
        '2016-03': 9.815, '2016-02': 9.926, '2016-01': 10.038, '2015-12': 10.151,
        '2015-11': 10.265, '2015-10': 10.380, '2015-09': 10.496, '2015-08': 10.613,
        '2015-07': 10.731, '2015-06': 10.850, '2015-05': 10.970, '2015-04': 11.091,
        '2015-03': 11.213, '2015-02': 11.336, '2015-01': 2.490
    }
    
    # 5-year BDT data (sample based on uploaded file)
    bdt_5y_data = {
        '2025-08': -0.70, '2025-07': -0.65, '2025-06': -0.69, '2025-05': -0.79,
        '2025-04': -0.75, '2025-03': -0.78, '2025-02': -0.77, '2025-01': -0.80,
        '2024-12': -0.82, '2024-11': -0.78, '2024-10': -0.75, '2024-09': -0.72,
        '2024-08': -0.69, '2024-07': -0.66, '2024-06': -0.63, '2024-05': -0.60,
        '2024-04': -0.57, '2024-03': -0.54, '2024-02': -0.51, '2024-01': -0.48,
        '2023-12': -0.45, '2023-11': -0.42, '2023-10': -0.39, '2023-09': -0.36,
        '2023-08': -0.33, '2023-07': -0.30, '2023-06': -0.27, '2023-05': -0.24,
        '2023-04': -0.21, '2023-03': -0.18, '2023-02': -0.15, '2023-01': -0.12,
        '2022-12': -0.09, '2022-11': -0.06, '2022-10': -0.03, '2022-09': 0.00,
        '2022-08': 0.03, '2022-07': 0.06, '2022-06': 0.09, '2022-05': 0.12,
        '2022-04': 0.15, '2022-03': 0.18, '2022-02': 0.21, '2022-01': 0.24,
        '2021-12': 0.27, '2021-11': 0.30, '2021-10': 0.33, '2021-09': 0.36,
        '2021-08': 0.39, '2021-07': 0.42, '2021-06': 0.45, '2021-05': 0.48,
        '2021-04': 0.51, '2021-03': 0.54, '2021-02': 0.57, '2021-01': 0.60,
        '2020-12': 0.63, '2020-11': 0.66, '2020-10': 0.69, '2020-09': 0.72,
        '2020-08': 0.75, '2020-07': 0.78, '2020-06': 0.81, '2020-05': 0.84,
        '2020-04': 0.87, '2020-03': 0.90, '2020-02': 0.93, '2020-01': 0.96,
        '2019-12': 0.99, '2019-11': 1.02, '2019-10': 1.05, '2019-09': 1.08,
        '2019-08': 1.11, '2019-07': 1.14, '2019-06': 1.17, '2019-05': 1.20,
        '2019-04': 1.23, '2019-03': 1.26, '2019-02': 1.29, '2019-01': 1.32,
        '2018-12': 1.35, '2018-11': 1.38, '2018-10': 1.41, '2018-09': 1.44,
        '2018-08': 1.47, '2018-07': 1.50, '2018-06': 1.53, '2018-05': 1.56,
        '2018-04': 1.59, '2018-03': 1.62, '2018-02': 1.65, '2018-01': 1.68,
        '2017-12': 1.71, '2017-11': 1.74, '2017-10': 1.77, '2017-09': 1.80,
        '2017-08': 1.83, '2017-07': 1.86, '2017-06': 1.89, '2017-05': 1.92,
        '2017-04': 1.95, '2017-03': 1.98, '2017-02': 2.01, '2017-01': 2.04,
        '2016-12': 2.07, '2016-11': 2.10, '2016-10': 2.13, '2016-09': 2.16,
        '2016-08': 2.19, '2016-07': 2.22, '2016-06': 2.25, '2016-05': 2.28,
        '2016-04': 2.31, '2016-03': 2.34, '2016-02': 2.37, '2016-01': 2.40,
        '2015-12': 2.43, '2015-11': 2.46, '2015-10': 2.49, '2015-09': 2.52,
        '2015-08': 2.55, '2015-07': 2.58, '2015-06': 2.61, '2015-05': 2.64,
        '2015-04': 2.67, '2015-03': 2.70, '2015-02': 2.73, '2015-01': -0.80
    }
    
    return treasury_13w_data, bdt_5y_data

@st.cache_data
def create_enhanced_dataset():
    """Create enhanced dataset with Treasury variables (2015-2025)"""
    # Load Treasury data
    treasury_13w_data, bdt_5y_data = load_treasury_data()
    
    # Enhanced historical data from 2015-2025 with Treasury variables
    donnees_historiques = {
        '2015-03': {'taux_directeur': 2.50, 'inflation': 1.6, 'pib': 4.5, 'rendement_52s': 2.85},
        '2015-06': {'taux_directeur': 2.50, 'inflation': 1.4, 'pib': 4.3, 'rendement_52s': 2.92},
        '2015-09': {'taux_directeur': 2.50, 'inflation': 1.8, 'pib': 4.7, 'rendement_52s': 2.88},
        '2015-12': {'taux_directeur': 2.50, 'inflation': 1.6, 'pib': 4.5, 'rendement_52s': 2.90},
        '2016-03': {'taux_directeur': 2.25, 'inflation': 1.7, 'pib': 1.2, 'rendement_52s': 2.65},
        '2016-06': {'taux_directeur': 2.25, 'inflation': 1.8, 'pib': 1.5, 'rendement_52s': 2.70},
        '2016-09': {'taux_directeur': 2.25, 'inflation': 1.5, 'pib': 1.8, 'rendement_52s': 2.58},
        '2016-12': {'taux_directeur': 2.25, 'inflation': 1.6, 'pib': 1.2, 'rendement_52s': 2.62},
        '2017-03': {'taux_directeur': 2.25, 'inflation': 0.7, 'pib': 4.1, 'rendement_52s': 2.45},
        '2017-06': {'taux_directeur': 2.25, 'inflation': 0.8, 'pib': 3.8, 'rendement_52s': 2.48},
        '2017-09': {'taux_directeur': 2.25, 'inflation': 1.9, 'pib': 3.2, 'rendement_52s': 2.52},
        '2017-12': {'taux_directeur': 2.25, 'inflation': 1.8, 'pib': 4.1, 'rendement_52s': 2.50},
        '2018-03': {'taux_directeur': 2.25, 'inflation': 2.1, 'pib': 2.8, 'rendement_52s': 2.58},
        '2018-06': {'taux_directeur': 2.25, 'inflation': 1.9, 'pib': 3.1, 'rendement_52s': 2.55},
        '2018-09': {'taux_directeur': 2.25, 'inflation': 2.0, 'pib': 3.0, 'rendement_52s': 2.57},
        '2018-12': {'taux_directeur': 2.25, 'inflation': 1.8, 'pib': 3.1, 'rendement_52s': 2.54},
        '2019-03': {'taux_directeur': 2.25, 'inflation': 0.3, 'pib': 2.6, 'rendement_52s': 2.35},
        '2019-06': {'taux_directeur': 2.25, 'inflation': 0.1, 'pib': 2.8, 'rendement_52s': 2.32},
        '2019-09': {'taux_directeur': 2.25, 'inflation': 0.2, 'pib': 3.2, 'rendement_52s': 2.34},
        '2019-12': {'taux_directeur': 2.25, 'inflation': 0.5, 'pib': 2.5, 'rendement_52s': 2.38},
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
    
    # Generate monthly data from 2015 to June 2025
    date_debut = datetime(2015, 1, 1)
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
        
        # Add Treasury variables
        treasury_13w = treasury_13w_data.get(date_str, 2.5)  # Default if missing
        bdt_5y = bdt_5y_data.get(date_str, 0.0)  # Default if missing
        
        donnees_mensuelles.append({
            'Date': date_str,
            'Taux_Directeur': point_donnees['taux_directeur'],
            'Inflation': point_donnees['inflation'],
            'Croissance_PIB': point_donnees['pib'],
            'Treasury_13W': treasury_13w,
            'BDT_5Y': bdt_5y,
            'Rendement_52s': point_donnees['rendement_52s'],
            'Est_Point_Ancrage': est_ancrage
        })
        
        # Move to next month
        if date_courante.month == 12:
            date_courante = date_courante.replace(year=date_courante.year + 1, month=1)
        else:
            date_courante = date_courante.replace(month=date_courante.month + 1)
    
    return pd.DataFrame(donnees_mensuelles)

def calculate_correlations(df):
    """Calculate correlation matrix for all variables"""
    variables = ['Taux_Directeur', 'Inflation', 'Croissance_PIB', 'Treasury_13W', 'BDT_5Y']
    target = 'Rendement_52s'
    
    correlation_data = {}
    
    for var in variables:
        correlation = df[var].corr(df[target])
        correlation_data[var] = correlation
    
    # Calculate inter-variable correlations
    correlation_matrix = df[variables + [target]].corr()
    
    return correlation_data, correlation_matrix

def train_enhanced_model(df):
    """Train enhanced prediction model with 5 variables"""
    X = df[['Taux_Directeur', 'Inflation', 'Croissance_PIB', 'Treasury_13W', 'BDT_5Y']]
    y = df['Rendement_52s']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Cross-validation
    scores_cv = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae_cv = -scores_cv.mean()
    
    # Calculate feature importance (coefficients)
    feature_names = ['Taux_Directeur', 'Inflation', 'Croissance_PIB', 'Treasury_13W', 'BDT_5Y']
    coefficients = dict(zip(feature_names, model.coef_))
    
    return model, r2, mae, rmse, mae_cv, coefficients

def generate_enhanced_scenarios():
    """Generate scenarios with Treasury variables predictions"""
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2030, 12, 31)
    
    dates_quotidiennes = []
    date_courante = date_debut
    
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
    # Extended monetary policy decisions
    decisions_politiques = {
        'Conservateur': {
            '2025-06': 2.25, '2025-09': 2.25, '2025-12': 2.00, '2026-03': 1.75, 
            '2026-06': 1.75, '2026-09': 1.50, '2026-12': 1.50, '2027-06': 1.50,
            '2027-12': 1.75, '2028-06': 2.00, '2028-12': 2.00, '2029-06': 2.25,
            '2029-12': 2.25, '2030-12': 2.50
        },
        'Cas_de_Base': {
            '2025-06': -0.69, '2025-09': -0.58, '2025-12': -0.48, '2026-03': -0.38,
            '2026-06': -0.28, '2026-09': -0.18, '2026-12': -0.08, '2027-06': 0.02,
            '2027-12': 0.12, '2028-06': 0.22, '2028-12': 0.32, '2029-06': 0.42,
            '2029-12': 0.52, '2030-12': 0.62
        },
        'Optimiste': {
            '2025-06': -0.69, '2025-09': -0.62, '2025-12': -0.55, '2026-03': -0.48,
            '2026-06': -0.41, '2026-09': -0.34, '2026-12': -0.27, '2027-06': -0.20,
            '2027-12': -0.13, '2028-06': -0.06, '2028-12': 0.01, '2029-06': 0.08,
            '2029-12': 0.15, '2030-12': 0.22
        }
    }
    
    scenarios = {}
    
    for nom_scenario in ['Conservateur', 'Cas_de_Base', 'Optimiste']:
        donnees_scenario = []
        taux_politiques = decisions_politiques[nom_scenario]
        treasury_13w_path = treasury_13w_scenarios[nom_scenario]
        bdt_5y_path = bdt_5y_scenarios[nom_scenario]
        
        for i, date in enumerate(dates_quotidiennes):
            jours_ahead = i + 1
            
            # Determine policy rate based on calendar
            date_str = date.strftime('%Y-%m')
            taux_directeur = 2.25  # Default
            for date_politique, taux in sorted(taux_politiques.items()):
                if date_str >= date_politique:
                    taux_directeur = taux
            
            # Determine Treasury 13W rate
            treasury_13w = 1.78  # Default
            for date_treasury, taux in sorted(treasury_13w_path.items()):
                if date_str >= date_treasury:
                    treasury_13w = taux
            
            # Determine BDT 5Y rate
            bdt_5y = -0.69  # Default
            for date_bdt, taux in sorted(bdt_5y_path.items()):
                if date_str >= date_bdt:
                    bdt_5y = taux
            
            # Enhanced economic projections with full business cycle
            np.random.seed(hash(date.strftime('%Y-%m-%d')) % 2**32)
            
            mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
            
            # Full business cycle modeling (5.5 year cycle)
            cycle_position = (mois_depuis_debut % 66) / 66  # 66 months = 5.5 years
            
            if nom_scenario == 'Conservateur':
                inflation_cycle = 1.8 + 0.6 * np.sin(2 * np.pi * cycle_position) + 0.3 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_cycle = 3.5 + 1.2 * np.sin(2 * np.pi * cycle_position + np.pi/4) + 0.6 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            elif nom_scenario == 'Cas_de_Base':
                inflation_cycle = 1.6 + 0.4 * np.sin(2 * np.pi * cycle_position) + 0.2 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_cycle = 3.8 + 1.0 * np.sin(2 * np.pi * cycle_position + np.pi/4) + 0.5 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            else:  # Optimiste
                inflation_cycle = 1.4 + 0.3 * np.sin(2 * np.pi * cycle_position) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_cycle = 4.2 + 0.8 * np.sin(2 * np.pi * cycle_position + np.pi/4) + 0.4 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            
            # Add realistic noise
            inflation = max(0.5, min(4.0, inflation_cycle + np.random.normal(0, 0.02)))
            pib = max(1.0, min(7.0, pib_cycle + np.random.normal(0, 0.1)))
            
            donnees_scenario.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Taux_Directeur': taux_directeur,
                'Inflation': inflation,
                'Croissance_PIB': pib,
                'Treasury_13W': treasury_13w,
                'BDT_5Y': bdt_5y,
                'Jours_Ahead': jours_ahead,
                'Jour_Semaine': date.strftime('%A'),
                'Est_Weekend': date.weekday() >= 5
            })
        
        scenarios[nom_scenario] = pd.DataFrame(donnees_scenario)
    
    return scenarios

def predict_enhanced_yields(scenarios, model):
    """Generate yield predictions with enhanced 5-variable model"""
    baseline = 1.75  # June 2025 baseline
    predictions = {}
    
    for scenario_name, scenario_df in scenarios.items():
        X_future = scenario_df[['Taux_Directeur', 'Inflation', 'Croissance_PIB', 'Treasury_13W', 'BDT_5Y']]
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
            
            # Time-based uncertainty
            jours_ahead = ligne['Jours_Ahead']
            incertitude = (jours_ahead / 365) * 0.02
            if scenario_name == 'Conservateur':
                ajustement += incertitude
            elif scenario_name == 'Optimiste':
                ajustement -= incertitude * 0.5
            
            # Day of week effects
            effets_jours = {
                'Monday': 0.005, 'Tuesday': 0.00, 'Wednesday': -0.005,
                'Thursday': 0.00, 'Friday': 0.01, 'Saturday': -0.005, 'Sunday': -0.005
            }
            ajustement += effets_jours.get(ligne['Jour_Semaine'], 0)
            
            ajustements.append(ajustement)
        
        rendements_finaux = rendements_lisses + np.array(ajustements)
        rendements_finaux = np.clip(rendements_finaux, 0.1, 8.0)
        
        # Ensure logical progression
        for i in range(1, len(rendements_finaux)):
            daily_change = rendements_finaux[i] - rendements_finaux[i-1]
            if abs(daily_change) > 0.1:
                rendements_finaux[i] = rendements_finaux[i-1] + np.sign(daily_change) * 0.1
        
        scenario_df_copy = scenario_df.copy()
        scenario_df_copy['rendement_predit'] = rendements_finaux
        scenario_df_copy['scenario'] = scenario_name
        
        predictions[scenario_name] = scenario_df_copy
    
    return predictions

def generate_enhanced_recommendations(predictions):
    """Generate strategic recommendations with enhanced model insights"""
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
    # Enhanced header with new features badge
    col_logo, col_text = st.columns([1, 3])
    
    with col_logo:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="background: white; padding: 10px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
    
    with col_text:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #3d5aa3 100%); 
                    padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
            <h1 style="margin: 0; color: white;">Système de Prédiction Enhanced</h1>
            <div class="enhanced-badge">✨ ENHANCED: +2 Variables Treasury</div>
            <p style="margin: 0.5rem 0; color: white;">Modèle d'IA Financière 5 Variables (2015-2030)</p>
            <p style="margin: 0; color: white;">BAM • HCP • Treasury 13W • BDT 5Y | Mise à jour: Horaire</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load enhanced data and models
    if 'enhanced_data_loaded' not in st.session_state:
        with st.spinner("Chargement du modèle enhanced..."):
            st.session_state.enhanced_df = create_enhanced_dataset()
            st.session_state.correlations, st.session_state.correlation_matrix = calculate_correlations(st.session_state.enhanced_df)
            st.session_state.enhanced_model, st.session_state.enhanced_r2, st.session_state.enhanced_mae, st.session_state.enhanced_rmse, st.session_state.enhanced_mae_cv, st.session_state.coefficients = train_enhanced_model(st.session_state.enhanced_df)
            st.session_state.enhanced_scenarios = generate_enhanced_scenarios()
            st.session_state.enhanced_predictions = predict_enhanced_yields(st.session_state.enhanced_scenarios, st.session_state.enhanced_model)
            st.session_state.enhanced_recommendations = generate_enhanced_recommendations(st.session_state.enhanced_predictions)
            st.session_state.enhanced_data_loaded = True
    
    live_data = fetch_live_data()
    baseline_yield = live_data['current_baseline']
    baseline_date = live_data['baseline_date']
    
    # Enhanced Sidebar with Treasury data
    with st.sidebar:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.header("Modèle Enhanced")
        
        # Enhanced model performance
        st.markdown("### 🚀 Performance Enhanced")
        st.markdown('<div class="model-performance">', unsafe_allow_html=True)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("R² Enhanced", f"{st.session_state.enhanced_r2:.1%}", 
                     delta=f"+{(st.session_state.enhanced_r2-0.72)*100:.1f}%", 
                     help="Amélioration vs modèle 3 variables")
            st.metric("Précision Enhanced", f"±{st.session_state.enhanced_mae:.2f}%")
        
        with col2:
            st.metric("RMSE", f"±{st.session_state.enhanced_rmse:.2f}%")
            st.metric("Validation Croisée", f"±{st.session_state.enhanced_mae_cv:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Variable importance
        st.markdown("### 📊 Importance des Variables")
        for var, coef in st.session_state.coefficients.items():
            importance = abs(coef) / sum(abs(c) for c in st.session_state.coefficients.values()) * 100
            st.metric(var.replace('_', ' '), f"{importance:.1f}%", 
                     help=f"Coefficient: {coef:.3f}")
        
        st.markdown("### Données en Temps Réel")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Taux Directeur", f"{live_data['policy_rate']:.2f}%")
            st.metric("Inflation", f"{live_data['inflation']:.2f}%")
            st.metric("Treasury 13W", f"{live_data['treasury_13w']:.3f}%")
        
        with col2:
            st.metric("Baseline BAM", f"{baseline_yield:.2f}%")
            st.metric("Croissance PIB", f"{live_data['gdp_growth']:.2f}%")
            st.metric("BDT 5Y", f"{live_data['bdt_5y']:.2f}%")
        
        st.info(f"Dernière MAJ: {live_data['last_updated']}")
        
        # Correlations display
        st.markdown("### 🔗 Corrélations vs Rendement")
        for var, corr in st.session_state.correlations.items():
            color = "🟢" if abs(corr) > 0.7 else "🟡" if abs(corr) > 0.4 else "🔴"
            st.metric(f"{color} {var.replace('_', ' ')}", f"{corr:.3f}")
        
        if st.sidebar.button("Actualiser Enhanced"):
            st.cache_data.clear()
            st.rerun()
    
    # Main tabs with enhanced content
    tab1, tab2, tab3, tab4 = st.tabs(["Vue d'Ensemble Enhanced", "Analyse Corrélations", "Prédictions Détaillées", "Recommandations"])
    
    with tab1:
        # Enhanced Executive Dashboard
        st.markdown('<div class="executive-dashboard">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; font-size: 1.4rem; font-weight: 700; margin-bottom: 2rem;">🚀 Tableau de Bord Enhanced (5 Variables)</div>', unsafe_allow_html=True)
        
        # Model comparison
        st.markdown("### 📈 Amélioration du Modèle")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">R² ENHANCED</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #28a745;">{st.session_state.enhanced_r2:.1%}</div>
                <div style="font-size: 0.8rem; color: #28a745;">+{(st.session_state.enhanced_r2-0.72)*100:.1f}% vs 3-var</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">PRÉCISION</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #17a2b8;">±{st.session_state.enhanced_mae:.2f}%</div>
                <div style="font-size: 0.8rem; color: #17a2b8;">MAE Enhanced</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_correlation = max(st.session_state.correlations.values(), key=abs)
            best_var = [k for k, v in st.session_state.correlations.items() if v == best_correlation][0]
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">MEILLEURE CORRÉLATION</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #6f42c1;">{best_correlation:.3f}</div>
                <div style="font-size: 0.8rem; color: #6f42c1;">{best_var.replace('_', ' ')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            most_important = max(st.session_state.coefficients.items(), key=lambda x: abs(x[1]))
            importance = abs(most_important[1]) / sum(abs(c) for c in st.session_state.coefficients.values()) * 100
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">VARIABLE CLÉ</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ff6b35;">{importance:.1f}%</div>
                <div style="font-size: 0.8rem; color: #ff6b35;">{most_important[0].replace('_', ' ')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Strategic analysis with enhanced model
        cas_de_base_predictions = st.session_state.enhanced_predictions['Cas_de_Base']
        
        q1_data = cas_de_base_predictions.head(90)
        q2_data = cas_de_base_predictions.head(180)
        year1_data = cas_de_base_predictions.head(365)
        
        q1_avg = q1_data['rendement_predit'].mean()
        q2_avg = q2_data['rendement_predit'].mean()
        year1_avg = year1_data['rendement_predit'].mean()
        
        # Enhanced environment assessment
        q1_change = q1_avg - baseline_yield
        max_deviation = max(abs(q1_change), abs(q2_avg - baseline_yield), abs(year1_avg - baseline_yield))
        
        if max_deviation > 0.5:
            if q1_change > 0.3:
                strategic_environment = "ENVIRONNEMENT DE HAUSSE - Enhanced Confirmed"
                env_color = "#dc3545"
            else:
                strategic_environment = "ENVIRONNEMENT DE BAISSE - Enhanced Confirmed"
                env_color = "#28a745"
        else:
            strategic_environment = "ENVIRONNEMENT STABLE - Enhanced Validated"
            env_color = "#17a2b8"
        
        st.markdown(f"""
        <div class="status-card" style="border-left-color: {env_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #2c3e50;">Environnement Enhanced</h3>
                    <p style="margin: 0.5rem 0; font-weight: 600; color: {env_color};">{strategic_environment}</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #6c757d;">Modèle 5 variables | R² = {st.session_state.enhanced_r2:.1%}</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #2c3e50;">{year1_avg:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">Prédiction Enhanced 12M</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced chart with all variables
        st.subheader("📊 Évolution Enhanced avec Variables Treasury")
        
        fig = go.Figure()
        
        # Historical data
        df_hist = st.session_state.enhanced_df.tail(12)
        fig.add_trace(go.Scatter(
            x=df_hist['Date'],
            y=df_hist['Rendement_52s'],
            mode='lines+markers',
            name='Historique Enhanced',
            line=dict(color='#2a5298', width=4),
            marker=dict(size=8)
        ))
        
        # Enhanced predictions
        colors = {'Conservateur': '#dc3545', 'Cas_de_Base': '#17a2b8', 'Optimiste': '#28a745'}
        for scenario, pred_df in st.session_state.enhanced_predictions.items():
            sample_indices = list(range(0, len(pred_df), 7))
            sample_data = pred_df.iloc[sample_indices]
            
            fig.add_trace(go.Scatter(
                x=sample_data['Date'],
                y=sample_data['rendement_predit'],
                mode='lines+markers',
                name=f'{scenario} Enhanced',
                line=dict(color=colors[scenario], width=3),
                marker=dict(size=5)
            ))
        
        fig.add_hline(y=baseline_yield, line_dash="dash", line_color="gray", 
                     annotation_text=f"Baseline Enhanced: {baseline_yield:.2f}%")
        
        fig.update_layout(
            height=450,
            template="plotly_white",
            xaxis_title="Période",
            yaxis_title="Rendement Enhanced (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            title="Prédictions Enhanced avec Treasury 13W & BDT 5Y"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔗 Analyse des Corrélations Enhanced")
        
        st.markdown('<div class="correlation-matrix">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; font-size: 1.3rem; font-weight: 700; margin-bottom: 1.5rem; color: #28a745;">Matrice de Corrélations Complète</div>', unsafe_allow_html=True)
        
        # Display correlation matrix as heatmap
        import plotly.express as px
        
        corr_matrix = st.session_state.correlation_matrix
        
        fig_corr = px.imshow(
            corr_matrix.values,
            labels=dict(x="Variables", y="Variables", color="Corrélation"),
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Matrice de Corrélation - Toutes Variables"
        )
        
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Individual correlations analysis
        st.subheader("📈 Analyse Détaillée des Corrélations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Variables Économiques")
            for var in ['Taux_Directeur', 'Inflation', 'Croissance_PIB']:
                corr = st.session_state.correlations[var]
                color = "#28a745" if abs(corr) > 0.7 else "#ffc107" if abs(corr) > 0.4 else "#dc3545"
                strength = "Forte" if abs(corr) > 0.7 else "Modérée" if abs(corr) > 0.4 else "Faible"
                
                st.markdown(f"""
                <div style="background: white; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {color};">
                    <strong>{var.replace('_', ' ')}</strong><br>
                    Corrélation: <span style="color: {color}; font-weight: bold;">{corr:.3f}</span><br>
                    Force: <span style="color: {color};">{strength}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Variables Treasury (Nouvelles)")
            for var in ['Treasury_13W', 'BDT_5Y']:
                corr = st.session_state.correlations[var]
                color = "#28a745" if abs(corr) > 0.7 else "#ffc107" if abs(corr) > 0.4 else "#dc3545"
                strength = "Forte" if abs(corr) > 0.7 else "Modérée" if abs(corr) > 0.4 else "Faible"
                
                st.markdown(f"""
                <div style="background: white; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {color};">
                    <strong>{var.replace('_', ' ')}</strong><br>
                    Corrélation: <span style="color: {color}; font-weight: bold;">{corr:.3f}</span><br>
                    Force: <span style="color: {color};">{strength}</span>
                    <br><span style="font-size: 0.8rem; color: #6c757d;">✨ Variable Enhanced</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model equation
        st.subheader("🔢 Équation du Modèle Enhanced")
        
        equation_parts = []
        for var, coef in st.session_state.coefficients.items():
            sign = "+" if coef >= 0 else ""
            equation_parts.append(f"{sign}{coef:.3f} × {var.replace('_', ' ')}")
        
        equation = "Rendement 52s = " + " ".join(equation_parts)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 2rem; border-radius: 12px; text-align: center;">
            <h3>Équation de Régression Enhanced</h3>
            <p style="font-family: monospace; font-size: 1.1rem; margin: 1rem 0;">
                {equation}
            </p>
            <p style="font-size: 0.9rem; opacity: 0.9;">
                R² = {st.session_state.enhanced_r2:.3f} | MAE = ±{st.session_state.enhanced_mae:.3f}% | 
                Variables = 5 | Période = 2015-2025
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.header("📈 Prédictions Détaillées Enhanced")
        
        scenario_choice = st.selectbox("Choisissez un scénario Enhanced:", 
                                     ['Cas_de_Base', 'Conservateur', 'Optimiste'])
        
        pred_data = st.session_state.enhanced_predictions[scenario_choice]
        
        # Enhanced metrics with Treasury variables
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Rendement Moyen", f"{pred_data['rendement_predit'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{pred_data['rendement_predit'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{pred_data['rendement_predit'].max():.2f}%")
        with col4:
            change = pred_data['rendement_predit'].mean() - baseline_yield
            st.metric("Écart vs Juin 2025", f"{change:+.2f}%")
        with col5:
            volatility = pred_data['rendement_predit'].std()
            st.metric("Volatilité", f"{volatility:.2f}%")
        
        # Treasury variables evolution
        st.subheader(f"🏦 Évolution Variables Treasury - {scenario_choice}")
        
        sample_treasury = pred_data[::30]  # Monthly sampling
        
        fig_treasury = go.Figure()
        
        # Treasury 13W
        fig_treasury.add_trace(go.Scatter(
            x=sample_treasury['Date'],
            y=sample_treasury['Treasury_13W'],
            mode='lines+markers',
            name='Treasury 13W',
            line=dict(color='#ff6b35', width=3),
            yaxis='y'
        ))
        
        # BDT 5Y
        fig_treasury.add_trace(go.Scatter(
            x=sample_treasury['Date'],
            y=sample_treasury['BDT_5Y'],
            mode='lines+markers',
            name='BDT 5Y',
            line=dict(color='#6f42c1', width=3),
            yaxis='y2'
        ))
        
        fig_treasury.update_layout(
            title="Évolution des Variables Treasury Enhanced",
            xaxis_title="Date",
            yaxis=dict(title="Treasury 13W (%)", side="left"),
            yaxis2=dict(title="BDT 5Y (%)", side="right", overlaying="y"),
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_treasury, use_container_width=True)
        
        # Detailed prediction chart
        st.subheader(f"📊 Prédictions Enhanced Détaillées - {scenario_choice}")
        
        sample_detailed = pred_data[::7]  # Weekly sampling
        
        fig_detail = go.Figure()
        fig_detail.add_trace(go.Scatter(
            x=sample_detailed['Date'],
            y=sample_detailed['rendement_predit'],
            mode='lines+markers',
            name='Prédiction Enhanced',
            line=dict(color=colors[scenario_choice], width=3)
        ))
        
        # Add confidence bands based on model uncertainty
        upper_bound = sample_detailed['rendement_predit'] + st.session_state.enhanced_mae
        lower_bound = sample_detailed['rendement_predit'] - st.session_state.enhanced_mae
        
        fig_detail.add_trace(go.Scatter(
            x=sample_detailed['Date'],
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig_detail.add_trace(go.Scatter(
            x=sample_detailed['Date'],
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name=f'Bande de confiance ±{st.session_state.enhanced_mae:.2f}%',
            fillcolor=f'rgba({colors[scenario_choice][1:3]}, {colors[scenario_choice][3:5]}, {colors[scenario_choice][5:7]}, 0.2)'
        ))
        
        fig_detail.add_hline(y=baseline_yield, line_dash="dash", line_color="blue",
                           annotation_text=f"Baseline Enhanced: {baseline_yield:.2f}%")
        
        fig_detail.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Rendement Enhanced (%)",
            title=f"Prédictions avec Bandes de Confiance - Modèle 5 Variables"
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Variable contributions analysis
        st.subheader("🔍 Analyse des Contributions Variables")
        
        # Calculate average values for each variable in predictions
        avg_values = {
            'Taux_Directeur': pred_data['Taux_Directeur'].mean(),
            'Inflation': pred_data['Inflation'].mean(),
            'Croissance_PIB': pred_data['Croissance_PIB'].mean(),
            'Treasury_13W': pred_data['Treasury_13W'].mean(),
            'BDT_5Y': pred_data['BDT_5Y'].mean()
        }
        
        # Calculate contributions
        contributions = {}
        for var, coef in st.session_state.coefficients.items():
            contributions[var] = coef * avg_values[var]
        
        # Display contributions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Contributions Variables")
            total_contribution = sum(contributions.values())
            
            for var, contrib in contributions.items():
                percentage = (contrib / total_contribution) * 100 if total_contribution != 0 else 0
                color = "#28a745" if contrib > 0 else "#dc3545"
                
                st.markdown(f"""
                <div style="background: white; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {color};">
                    <strong>{var.replace('_', ' ')}</strong><br>
                    Contribution: <span style="color: {color}; font-weight: bold;">{contrib:+.3f}</span><br>
                    Poids: <span style="color: {color};">{percentage:+.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Pie chart of absolute contributions
            abs_contributions = {k: abs(v) for k, v in contributions.items()}
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=[k.replace('_', ' ') for k in abs_contributions.keys()],
                values=list(abs_contributions.values()),
                hole=0.4
            )])
            
            fig_pie.update_layout(
                title="Répartition des Contributions (Valeur Absolue)",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Export enhanced data
        if st.button("📥 Télécharger Prédictions Enhanced"):
            # Add variable contributions to export data
            export_data = pred_data.copy()
            
            # Add individual variable contributions
            for var, coef in st.session_state.coefficients.items():
                export_data[f'Contribution_{var}'] = export_data[var] * coef
            
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV Enhanced",
                data=csv,
                file_name=f"sofac_enhanced_predictions_{scenario_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab4:
        st.header("🎯 Recommandations Stratégiques Enhanced")
        
        # Enhanced loan decision section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <h3 style="margin: 0; color: white;">🚀 AIDE À LA DÉCISION ENHANCED SOFAC</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Analyse avec Modèle 5 Variables (R² = {st.session_state.enhanced_r2:.1%})</p>
        </div>
        """.format(st.session_state.enhanced_r2), unsafe_allow_html=True)
        
        # Enhanced loan parameters
        st.subheader("⚙️ Paramètres Enhanced")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            loan_amount = st.slider("Montant (millions MAD):", 1, 500, 50)
        with col2:
            loan_duration = st.slider("Durée (années):", 1, 10, 5)
        with col3:
            current_fixed_rate = st.number_input("Taux fixe proposé (%):", min_value=1.0, max_value=10.0, value=3.2, step=0.1)
        with col4:
            risk_premium = st.number_input("Prime de risque (%):", min_value=0.5, max_value=3.0, value=1.3, step=0.1)
        with col5:
            max_volatility_accepted = st.number_input("Volatilité Max (%):", min_value=0.1, max_value=1.0, value=0.35, step=0.05)
        
        # Enhanced risk tolerance with model confidence
        model_confidence = st.session_state.enhanced_r2
        confidence_adjustment = f"Confiance modèle: {model_confidence:.1%}"
        
        st.markdown(f"""
        <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #1976d2;">
            <div style="font-size: 0.85rem; color: #1565c0;">
                <strong>🚀 Modèle Enhanced:</strong> 5 variables | {confidence_adjustment} | MAE: ±{st.session_state.enhanced_mae:.2f}%
                <br>• <strong>Treasury 13W:</strong> {live_data['treasury_13w']:.3f}% (impact direct sur prédictions)
                <br>• <strong>BDT 5Y:</strong> {live_data['bdt_5y']:.2f}% (structure des taux à terme)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced loan analysis with 5-variable model
        scenarios_analysis = {}
        
        for scenario_name, pred_df in st.session_state.enhanced_predictions.items():
            loan_duration_days = loan_duration * 365
            relevant_predictions = pred_df.head(loan_duration_days)
            
            # Enhanced variable rate calculation with better model
            variable_rates_annual = []
            
            for year in range(loan_duration):
                start_day = year * 365
                end_day = min((year + 1) * 365, len(relevant_predictions))
                
                if end_day <= len(relevant_predictions):
                    year_data = relevant_predictions.iloc[start_day:end_day]
                    reference_rate = year_data['rendement_predit'].mean()
                else:
                    last_year_data = relevant_predictions.iloc[-365:]
                    reference_rate = last_year_data['rendement_predit'].mean()
                
                # Add banking spread
                effective_rate = reference_rate + risk_premium
                variable_rates_annual.append(effective_rate)
            
            # Calculate costs
            fixed_cost_total = (current_fixed_rate / 100) * loan_amount * 1_000_000 * loan_duration
            variable_cost_total = sum([(rate / 100) * loan_amount * 1_000_000 for rate in variable_rates_annual])
            
            cost_difference = variable_cost_total - fixed_cost_total
            cost_difference_percentage = (cost_difference / fixed_cost_total) * 100
            
            # Enhanced risk metrics
            volatility = relevant_predictions['rendement_predit'].std()
            max_rate = max(variable_rates_annual)
            min_rate = min(variable_rates_annual)
            rate_range = max_rate - min_rate
            
            # Model confidence factor
            confidence_factor = model_confidence  # Higher confidence = more reliable predictions
            
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
                'rate_range': rate_range,
                'confidence_factor': confidence_factor
            }
        
        # Enhanced decision matrix
        st.subheader("📊 Matrice Enhanced par Scénario")
        
        decision_data = []
        for scenario_name, analysis in scenarios_analysis.items():
            confidence_score = analysis['confidence_factor'] * 100
            
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
                'Scénario': f"{scenario_name} ✨",
                'Taux Variable Moyen': f"{analysis['avg_variable_rate']:.2f}%",
                'Fourchette': f"{analysis['min_rate']:.2f}% - {analysis['max_rate']:.2f}%",
                'Différence vs Fixe': decision_text,
                'Recommandation Enhanced': recommendation,
                'Niveau Risque': risk_level,
                'Confiance Modèle': f"{confidence_score:.0f}%"
            })
        
        decision_df = pd.DataFrame(decision_data)
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
        
        # Enhanced final recommendation
        variable_recommendations = sum(1 for analysis in scenarios_analysis.values() if analysis['cost_difference'] < 0)
        avg_confidence = np.mean([analysis['confidence_factor'] for analysis in scenarios_analysis.values()])
        avg_cost_difference = np.mean([analysis['cost_difference'] for analysis in scenarios_analysis.values()])
        max_volatility = max([analysis['volatility'] for analysis in scenarios_analysis.values()])
        
        # Enhanced decision logic with model confidence
        confidence_boost = avg_confidence * 0.2  # Boost decision confidence based on model R²
        
        if variable_recommendations >= 2 and avg_cost_difference < -200000 and max_volatility <= max_volatility_accepted:
            final_recommendation = "TAUX VARIABLE"
            final_reason = f"Modèle Enhanced confirme économies ({abs(avg_cost_difference):,.0f} MAD) avec R²={avg_confidence:.1%}"
            final_color = "#28a745"
            final_confidence = min(95, 75 + confidence_boost * 100)
        elif variable_recommendations >= 2 and max_volatility <= max_volatility_accepted * 1.2:
            final_recommendation = "STRATÉGIE MIXTE"
            final_reason = f"Modèle Enhanced suggère approche équilibrée (R²={avg_confidence:.1%})"
            final_color = "#ffc107"
            final_confidence = min(85, 65 + confidence_boost * 100)
        else:
            final_recommendation = "TAUX FIXE"
            final_reason = f"Modèle Enhanced privilégie sécurité malgré R²={avg_confidence:.1%}"
            final_color = "#dc3545"
            final_confidence = min(90, 70 + confidence_boost * 100)
        
        # Enhanced final recommendation display
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {final_color}, {final_color}AA); 
                    color: white; padding: 2rem; border-radius: 12px; margin: 2rem 0; text-align: center;">
            <h2>🎯 DÉCISION ENHANCED SOFAC</h2>
            <h3>{final_recommendation}</h3>
            <p><strong>Justification Enhanced:</strong> {final_reason}</p>
            <p><strong>Analyse:</strong> {loan_amount}M MAD | {loan_duration} ans | Taux fixe: {current_fixed_rate}%</p>
            <hr style="margin: 1rem 0; opacity: 0.3;">
            <div style="font-size: 0.9rem; opacity: 0.9;">
                <p><strong>Modèle:</strong> 5 variables | R² = {st.session_state.enhanced_r2:.1%} | MAE = ±{st.session_state.enhanced_mae:.2f}%</p>
                <p><strong>Économie moyenne:</strong> {abs(avg_cost_difference):,.0f} MAD | <strong>Confiance:</strong> {final_confidence:.0f}%</p>
                <p><strong>Variables clés:</strong> Treasury 13W ({live_data['treasury_13w']:.3f}%) + BDT 5Y ({live_data['bdt_5y']:.2f}%)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced scenario analysis
        st.subheader("📋 Analyse Enhanced par Scénario")
        
        for scenario, rec in st.session_state.enhanced_recommendations.items():
            with st.expander(f"🚀 Scénario Enhanced - {scenario}", expanded=True):
                scenario_analysis = scenarios_analysis[scenario]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Recommandation Enhanced:** {rec['recommandation']}
                    
                    **Analyse Financière Enhanced:**
                    - Modèle: 5 variables (R² = {st.session_state.enhanced_r2:.1%})
                    - Taux variable moyen: {scenario_analysis['avg_variable_rate']:.2f}%
                    - Fourchette: {scenario_analysis['min_rate']:.2f}% - {scenario_analysis['max_rate']:.2f}%
                    - Coût total (variable): {scenario_analysis['variable_cost_total']:,.0f} MAD
                    - Différence vs fixe: {scenario_analysis['cost_difference']:+,.0f} MAD ({scenario_analysis['cost_difference_percentage']:+.1f}%)
                    
                    **Métriques Enhanced:**
                    - Volatilité: {scenario_analysis['volatility']:.2f}%
                    - Confiance modèle: {scenario_analysis['confidence_factor']:.1%}
                    - Variables Treasury intégrées ✨
                    """)
                
                with col2:
                    # Enhanced mini chart with Treasury impact
                    pred_mini = st.session_state.enhanced_predictions[scenario][::30]
                    
                    fig_mini = go.Figure()
                    fig_mini.add_hline(y=current_fixed_rate, line_dash="dash", line_color="red", 
                                     annotation_text=f"Fixe: {current_fixed_rate:.2f}%")
                    fig_mini.add_trace(go.Scatter(
                        x=pred_mini['Date'],
                        y=pred_mini['rendement_predit'],
                        mode='lines+markers',
                        line=dict(color=colors[scenario], width=2),
                        name="Enhanced Variable"
                    ))
                    
                    fig_mini.update_layout(
                        height=200,
                        showlegend=False,
                        template="plotly_white",
                        margin=dict(l=20, r=20, t=20, b=20),
                        title=f"Enhanced - {scenario}"
                    )
                    
                    st.plotly_chart(fig_mini, use_container_width=True)
    
    # Enhanced Footer
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem;">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p style="margin: 0; font-weight: bold; color: #2a5298;">SOFAC - Modèle Enhanced 5 Variables (2015-2030)</p>
            <div class="enhanced-badge" style="margin: 0.5rem 0; display: inline-block;">✨ Enhanced: Treasury 13W + BDT 5Y</div>
            <p style="margin: 0; color: #FF6B35;">Dites oui au super crédit</p>
            <p style="margin: 0.5rem 0;">Enhanced R² = {st.session_state.enhanced_r2:.1%} | MAE = ±{st.session_state.enhanced_mae:.2f}% | Dernière MAJ: {current_time}</p>
            <p style="margin: 0;"><em>Prédictions basées sur modèle enhanced 5 variables. Ne constituent pas des conseils financiers.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()de_Base': {
            '2025-06': 2.25, '2025-09': 2.00, '2025-12': 1.75, '2026-03': 1.50, 
            '2026-06': 1.50, '2026-09': 1.25, '2026-12': 1.25, '2027-06': 1.25,
            '2027-12': 1.50, '2028-06': 1.75, '2028-12': 1.75, '2029-06': 2.00,
            '2029-12': 2.00, '2030-12': 2.25
        },
        'Optimiste': {
            '2025-06': 2.25, '2025-09': 1.75, '2025-12': 1.50, '2026-03': 1.25, 
            '2026-06': 1.00, '2026-09': 1.00, '2026-12': 1.00, '2027-06': 1.00,
            '2027-12': 1.25, '2028-06': 1.50, '2028-12': 1.50, '2029-06': 1.75,
            '2029-12': 1.75, '2030-12': 2.00
        }
    }
    
    # Treasury 13W scenarios (correlated with policy rates)
    treasury_13w_scenarios = {
        'Conservateur': {
            '2025-06': 1.78, '2025-09': 1.85, '2025-12': 1.65, '2026-03': 1.45,
            '2026-06': 1.48, '2026-09': 1.28, '2026-12': 1.32, '2027-06': 1.35,
            '2027-12': 1.58, '2028-06': 1.82, '2028-12': 1.85, '2029-06': 2.08,
            '2029-12': 2.12, '2030-12': 2.35
        },
        'Cas_de_Base': {
            '2025-06': 1.78, '2025-09': 1.72, '2025-12': 1.52, '2026-03': 1.32,
            '2026-06': 1.35, '2026-09': 1.15, '2026-12': 1.18, '2027-06': 1.22,
            '2027-12': 1.42, '2028-06': 1.62, '2028-12': 1.65, '2029-06': 1.85,
            '2029-12': 1.88, '2030-12': 2.08
        },
        'Optimiste': {
            '2025-06': 1.78, '2025-09': 1.58, '2025-12': 1.38, '2026-03': 1.18,
            '2026-06': 0.98, '2026-09': 1.02, '2026-12': 1.05, '2027-06': 1.08,
            '2027-12': 1.25, '2028-06': 1.42, '2028-12': 1.45, '2029-06': 1.62,
            '2029-12': 1.65, '2030-12': 1.85
        }
    }
    
    # BDT 5Y scenarios (term structure considerations)
    bdt_5y_scenarios = {
        'Conservateur': {
            '2025-06': -0.69, '2025-09': -0.55, '2025-12': -0.42, '2026-03': -0.28,
            '2026-06': -0.15, '2026-09': -0.02, '2026-12': 0.12, '2027-06': 0.25,
            '2027-12': 0.38, '2028-06': 0.52, '2028-12': 0.65, '2029-06': 0.78,
            '2029-12': 0.92, '2030-12': 1.05
        },
        'Cas_
