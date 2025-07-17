# Quick recommendations with professional cards
        st.subheader("Recommandations Rapides")
        
        col1, col2, col3 = st.columns(3, gap="medium")
        
        for i, (scenario, rec) in enumerate(st.session_state.recommandations.items()):
            with [col1, col2, col3][i]:
                # Determine card color based on recommendation
                if rec['recommandation'] == 'TAUX VARIABLE':
                    card_color_rec = '#22c55e'
                elif rec['recommandation'] == 'TAUX FIXE':
                    card_color_rec = '#ef4444'
                else:
                    card_color_rec = '#f59e0b'
                
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {card_color_rec};">
                    <h4 style="color: #374151; font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem;">{scenario}</h4>
                    <div style="background: {card_color_rec}; color: white; padding: 0.5rem; border-radius: 6px; margin-bottom: 0.75rem; text-align: center;">
                        <p style="margin: 0; font-size: 0.9rem; font-weight: 600;">{rec['recommandation']}</p>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.8rem; color: #6b7280;">Changement:</span>
                        <span style="font-size: 0.8rem; font-weight: 500; color: #374151;">{rec['changement_rendement']:+.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-size: 0.8rem; color: #6b7280;">Risque:</span>
                        <span style="font-size: 0.8rem; font-weight: 500; color: #374151;">{rec['niveau_risque']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)import streamlit as st
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: #f8fafc;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 40"><rect x="80" y="10" width="30" height="20" rx="3" fill="white" fill-opacity="0.2"/><text x="85" y="25" font-family="Inter,sans-serif" font-size="12" font-weight="700" fill="white">SOFAC</text></svg>');
        background-repeat: no-repeat;
        background-position: right 24px center;
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 0;
        color: white;
        text-align: left;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .main-header h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
        color: white !important;
    }
    
    .main-header p {
        font-size: 0.95rem !important;
        margin: 0 !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 400 !important;
    }
    
    /* Card styles */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }
    
    .metric-card h4 {
        color: #374151 !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Professional recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .recommendation-box h2 {
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    .recommendation-box h3 {
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }
    
    /* Status indicators */
    .data-status {
        background: #f0fdf4;
        border: 1px solid #22c55e;
        color: #16a34a;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .data-warning {
        background: #fffbeb;
        border: 1px solid #f59e0b;
        color: #d97706;
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Briefing sections */
    .briefing-section {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    /* Executive summary cards */
    .executive-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    .executive-card h4 {
        color: #374151 !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #e5e7eb !important;
    }
    
    /* Improved metrics */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    .stMetric label {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #6b7280 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.025em !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }
    
    .css-1d391kg .stSubheader {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Tab improvements */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 0.25rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #6b7280;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e5e7eb;
        color: #374151;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1e40af !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    /* Button improvements */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.25rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    h4 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    p {
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
        color: #4b5563 !important;
    }
    
    /* Expander improvements */
    .streamlit-expanderHeader {
        background-color: #f8fafc !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background-color: white !important;
        border: 1px solid #e5e7eb !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Success/Warning/Info boxes */
    .stAlert {
        border-radius: 8px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
    }
    
    .stAlert > div {
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
    }
    
    /* Spinner improvements */
    .stSpinner {
        text-align: center;
        color: #6b7280;
    }
    
    /* Selectbox improvements */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    
    /* Slider improvements */
    .stSlider > div > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Make columns have consistent spacing */
    .stColumn {
        padding: 0 0.5rem;
    }
    
    /* Professional footer */
    .footer {
        background-color: #f8fafc;
        border-top: 1px solid #e5e7eb;
        padding: 2rem 0;
        margin-top: 3rem;
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        line-height: 1.6;
    }
    
    .footer strong {
        color: #374151;
        font-weight: 600;
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

def display_live_data_panel(live_data, last_historical_yield):
    """Display live data panel in sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Données en Temps Réel")
    
    live_sources = sum(1 for source in live_data['sources'].values() if 'Live' in source)
    total_sources = 4
    success_rate = (live_sources / total_sources) * 100
    
    if success_rate >= 50:
        st.sidebar.markdown('<div class="data-status">● Données partiellement en direct</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="data-warning">● Données principalement estimées</div>', unsafe_allow_html=True)
    
    st.sidebar.write(f"**Sources en direct:** {live_sources}/{total_sources} ({success_rate:.0f}%)")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        indicator = "●" if 'Live' in live_data['sources']['policy_rate'] else "○"
        st.metric(
            f"{indicator} Taux Directeur", 
            f"{live_data['policy_rate']:.2f}%",
            help=f"Source: {live_data['sources']['policy_rate']}"
        )
        
        indicator = "●" if 'Live' in live_data['sources']['inflation'] else "○"
        st.metric(
            f"{indicator} Inflation", 
            f"{live_data['inflation']:.2f}%",
            help=f"Source: {live_data['sources']['inflation']}"
        )
    
    with col2:
        # Show the actual historical yield instead of the estimated live yield
        st.metric(
            "▲ Rendement Actuel", 
            f"{last_historical_yield:.2f}%",
            help="Dernière valeur historique (Juin 2025)"
        )
        
        st.metric(
            "○ Croissance PIB", 
            f"{live_data['gdp_growth']:.2f}%",
            help=f"Source: {live_data['sources']['gdp_growth']}"
        )
    
    st.sidebar.info(f"⏰ Dernière mise à jour: {live_data['last_updated']}")
    
    if st.sidebar.button("↻ Actualiser"):
        st.cache_data.clear()
        st.rerun()

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
        <h1>SOFAC - Prédiction des Rendements 52-Semaines</h1>
        <p>Système d'aide à la décision • Données Bank Al-Maghrib & HCP • Mise à jour: {} • Prochaine: {}</p>
    </div>
    """.format(
        datetime.now().strftime('%H:%M'), 
        (datetime.now() + timedelta(hours=1)).strftime('%H:%M')
    ), unsafe_allow_html=True)
    
    # Fetch live data
    with st.spinner("↻ Récupération des données en temps réel..."):
        live_data = fetch_live_moroccan_data()
    
    # Load cached model data first to get the last historical yield
    if 'data_loaded' not in st.session_state:
        with st.spinner("⚙ Calibration du modèle..."):
            st.session_state.df_mensuel = create_monthly_dataset()
            st.session_state.modele, st.session_state.r2, st.session_state.mae, st.session_state.rmse, st.session_state.mae_vc = train_prediction_model(st.session_state.df_mensuel)
            st.session_state.scenarios = create_economic_scenarios()
            st.session_state.predictions = generate_predictions(st.session_state.scenarios, st.session_state.modele, st.session_state.mae)
            st.session_state.recommandations = generate_recommendations(st.session_state.predictions)
            st.session_state.data_loaded = True
    
    # Get the last historical yield value (June 2025)
    last_historical_yield = st.session_state.df_mensuel.iloc[-1]['Rendement_52s']  # 1.75%
    
    with st.sidebar:
        st.header("📊 Informations du Modèle")
        
        # Display live data panel with corrected historical yield
        display_live_data_panel(live_data, last_historical_yield)
        
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
        
        for _, row in cas_base_predictions.iterrows():
            pred_date = row['Date']
            if pred_date == today_str:
                today_prediction = row['Rendement_Predit']
                break
            elif pred_date > today_str and closest_prediction is None:
                closest_prediction = row['Rendement_Predit']
        
        # Display today's prediction
        if today_prediction is not None:
            st.sidebar.success(f"**{today_display}**")
            st.sidebar.metric(
                "🎯 Rendement Prédit Aujourd'hui",
                f"{today_prediction:.2f}%",
                delta=f"{(today_prediction - last_historical_yield):+.2f}%",
                help="Prédiction pour aujourd'hui vs baseline juin 2025"
            )
        elif closest_prediction is not None:
            st.sidebar.warning(f"**{today_display}**")
            st.sidebar.metric(
                "🎯 Prédiction Prochaine",
                f"{closest_prediction:.2f}%",
                delta=f"{(closest_prediction - last_historical_yield):+.2f}%",
                help="Prochaine prédiction disponible"
            )
        else:
            st.sidebar.info(f"**{today_display}**")
            st.sidebar.write("🎯 **Prédiction:** Données en cours de traitement")
        
        st.success("✓ Modèle calibré avec données historiques!")
        
        # Model performance metrics
        st.subheader("🎯 Performance du Modèle")
        st.metric("R² Score", f"{st.session_state.r2:.1%}", help="Pourcentage de variance expliquée")
        st.metric("Précision", f"±{st.session_state.mae:.2f}%", help="Erreur absolue moyenne")
        st.metric("Validation Croisée", f"±{st.session_state.mae_vc:.2f}%", help="Erreur en validation croisée")
        
        st.info("↻ Données live utilisées pour surveillance économique uniquement.")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Vue d'Ensemble", "🔮 Prédictions Détaillées", "💼 Recommandations"])
    
    with tab1:
        st.header("📈 Vue d'Ensemble des Prédictions")
        
        # ENHANCED EXECUTIVE BRIEFING
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        today_display = today.strftime('%d/%m/%Y')
        
        # Get current situation analysis
        cas_de_base = st.session_state.predictions['Cas_de_Base']
        current_prediction = None
        trend_direction = None
        trend_strength = "STABLE"
        
        # Find today's or closest prediction
        for i, row in cas_de_base.iterrows():
            pred_date = row['Date']
            if pred_date >= today_str:
                current_prediction = row['Rendement_Predit']
                # Analyze trend over next 30 days
                if i < len(cas_de_base) - 30:
                    future_avg = cas_de_base.iloc[i:i+30]['Rendement_Predit'].mean()
                    change = future_avg - current_prediction
                    if change > 0.1:
                        trend_direction = "HAUSSE"
                        trend_strength = "FORTE" if change > 0.3 else "MODÉRÉE"
                    elif change < -0.1:
                        trend_direction = "BAISSE"
                        trend_strength = "FORTE" if change < -0.3 else "MODÉRÉE"
                    else:
                        trend_direction = "STABLE"
                break
        
        if current_prediction is None:
            current_prediction = last_historical_yield
        
        # Get market sentiment and recommendations
        recommandation_base = st.session_state.recommandations['Cas_de_Base']
        evolution_vs_baseline = current_prediction - last_historical_yield
        changement_global = recommandation_base['changement_rendement']
        
        # Determine market condition and urgency
        if evolution_vs_baseline > 0.4 or changement_global > 0.4:
            market_status = "TAUX ÉLEVÉS - DANGER"
            status_emoji = "🔴"
            urgency = "IMMÉDIATE"
            action = "FIXER LES TAUX MAINTENANT"
            card_color = "#dc3545"
        elif evolution_vs_baseline > 0.1 or changement_global > 0.1:
            market_status = "TAUX EN HAUSSE - ATTENTION"
            status_emoji = "🟠"
            urgency = "ÉLEVÉE"
            action = "SURVEILLER ET PRÉPARER"
            card_color = "#fd7e14"
        elif evolution_vs_baseline < -0.4 or changement_global < -0.4:
            market_status = "TAUX FAVORABLES - OPPORTUNITÉ"
            status_emoji = "🟢"
            urgency = "MODÉRÉE"
            action = "UTILISER TAUX VARIABLES"
            card_color = "#28a745"
        elif evolution_vs_baseline < -0.1 or changement_global < -0.1:
            market_status = "TAUX EN BAISSE - FAVORABLE"
            status_emoji = "🟡"
            urgency = "NORMALE"
            action = "CONSIDÉRER TAUX VARIABLES"
            card_color = "#20c997"
        else:
            market_status = "TAUX STABLES - NEUTRE"
            status_emoji = "⚪"
            urgency = "FAIBLE"
            action = "STRATÉGIE ÉQUILIBRÉE"
            card_color = "#6c757d"
        
        # Calculate key metrics for decision makers
        volatilite_globale = cas_de_base['Rendement_Predit'].std()
        data_quality = sum(1 for source in live_data['sources'].values() if 'Live' in source)
        confidence_level = "ÉLEVÉE" if data_quality >= 3 else "MOYENNE" if data_quality >= 2 else "FAIBLE"
        
        # Impact financier estimation
        impact_10m = abs(changement_global) * 10  # Million MAD per year
        
        # EXECUTIVE BRIEFING SECTION - More professional design
        st.markdown(f"""
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 0; margin: 2rem 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
            <div style="background: {card_color}; color: white; padding: 1.5rem; border-radius: 12px 12px 0 0; text-align: center;">
                <h2 style="margin: 0; font-size: 1.3rem; font-weight: 600; color: white;">BRIEFING EXÉCUTIF</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: rgba(255,255,255,0.9);">{today_display}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for the main briefing content
        brief_col1, brief_col2, brief_col3 = st.columns([1, 1, 1], gap="medium")
        
        with brief_col1:
            st.markdown(f"""
            <div class="executive-card">
                <h4 style="color: #6b7280; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em; margin-bottom: 0.5rem;">SITUATION ACTUELLE</h4>
                <p style="font-size: 1.1rem; font-weight: 600; color: {card_color}; margin: 0;">{status_emoji} {market_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with brief_col2:
            st.markdown(f"""
            <div class="executive-card" style="text-align: center;">
                <h4 style="color: #6b7280; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em; margin-bottom: 0.5rem;">RENDEMENT ACTUEL</h4>
                <h1 style="font-size: 2.5rem; font-weight: 700; color: {card_color}; margin: 0; line-height: 1;">{current_prediction:.2f}%</h1>
                <p style="font-size: 0.8rem; color: #6b7280; margin: 0.5rem 0 0 0;">52 semaines</p>
            </div>
            """, unsafe_allow_html=True)
        
        with brief_col3:
            st.markdown(f"""
            <div class="executive-card">
                <h4 style="color: #6b7280; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.025em; margin-bottom: 0.5rem;">ÉVOLUTION</h4>
                <p style="font-size: 1.1rem; font-weight: 600; color: {card_color}; margin: 0;">{evolution_vs_baseline:+.2f}%</p>
                <p style="font-size: 0.8rem; color: #6b7280; margin: 0.25rem 0 0 0;">vs Juin 2025</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendation section
        st.markdown(f"""
        <div class="executive-card" style="background: {card_color}; color: white; text-align: center; border: none;">
            <h3 style="margin: 0 0 0.5rem 0; color: white; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.025em;">RECOMMANDATION IMMÉDIATE</h3>
            <p style="margin: 0; font-size: 1.25rem; font-weight: 700; color: white;">{action}</p>
            <p style="margin: 0.75rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.85rem;">
                <span style="font-weight: 500;">Urgence:</span> {urgency} • 
                <span style="font-weight: 500;">Tendance 30j:</span> {trend_direction} {trend_strength}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # KEY METRICS DASHBOARD
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "📊 Baseline Juin",
                f"{last_historical_yield:.2f}%",
                help="Dernière valeur historique"
            )
        
        with col2:
            st.metric(
                "📈 Évolution",
                f"{evolution_vs_baseline:+.2f}%",
                delta="vs Baseline"
            )
        
        with col3:
            st.metric(
                "⚡ Volatilité",
                f"{volatilite_globale:.2f}%",
                help="Risque de fluctuation"
            )
        
        with col4:
            st.metric(
                "💰 Impact 10M MAD",
                f"{impact_10m:.0f}K/an",
                help="Impact financier estimé"
            )
        
        with col5:
            st.metric(
                "🎯 Confiance",
                confidence_level,
                delta=f"{data_quality}/4 sources"
            )
        
        # QUICK ACTION SUMMARY
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 8px; border-left: 4px solid {card_color};">
                <h4 style="margin: 0 0 0.5rem 0; color: {card_color};">📋 RÉSUMÉ SITUATION</h4>
                <p style="margin: 0; font-size: 0.85rem;"><strong>Rendement actuel:</strong> {current_prediction:.2f}%</p>
                <p style="margin: 0; font-size: 0.85rem;"><strong>Changement prévu:</strong> {changement_global:+.2f}% (18 mois)</p>
                <p style="margin: 0; font-size: 0.85rem;"><strong>Tendance court terme:</strong> {trend_direction} {trend_strength}</p>
                <p style="margin: 0; font-size: 0.85rem;"><strong>Niveau de risque:</strong> {recommandation_base['niveau_risque']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            strategy = recommandation_base['recommandation']
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 8px; border-left: 4px solid {card_color};">
                <h4 style="margin: 0 0 0.5rem 0; color: {card_color};">⚡ ACTIONS IMMÉDIATES</h4>
                <p style="margin: 0; font-size: 0.85rem;"><strong>Stratégie:</strong> {strategy}</p>
                <p style="margin: 0; font-size: 0.85rem;"><strong>Horizon décision:</strong> {"Immédiat" if urgency in ["IMMÉDIATE", "ÉLEVÉE"] else "1-3 mois"}</p>
                <p style="margin: 0; font-size: 0.85rem;"><strong>Impact financier:</strong> {impact_10m:.0f}K MAD/an (10M MAD)</p>
                <p style="margin: 0; font-size: 0.85rem;"><strong>Recommandation:</strong> {action}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Separator
        st.markdown("---")
        
        # Overview chart
        st.subheader("📊 Évolution des Rendements: Historique et Prédictions")
        
        fig_overview = go.Figure()
        
        # Historical data
        df_recent = st.session_state.df_mensuel.tail(8)
        
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
        
        fig_overview.update_layout(
            title="Évolution des Rendements 52-Semaines: Historique (2020-2025) et Prédictions (2025-2026)",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
        
        # Quick recommendations
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
            baseline_comparison = pred_scenario['Rendement_Predit'].mean() - last_historical_yield
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
            y=last_historical_yield, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Juin 2025: {last_historical_yield:.2f}%"
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
        if st.button("⬇ Télécharger les Prédictions"):
            pred_export = pred_scenario.copy()
            pred_export['Baseline_Juin_2025'] = last_historical_yield
            
            csv = pred_export.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"sofac_predictions_{scenario_selectionne.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("💼 Recommandations Stratégiques")
        
        # Global recommendation
        liste_recommandations = [rec['recommandation'] for rec in st.session_state.recommandations.values()]
        
        if liste_recommandations.count('TAUX VARIABLE') >= 2:
            strategie_globale = "TAUX VARIABLE"
            raison_globale = f"Majorité des scénarios montrent des taux en baisse depuis juin 2025 ({last_historical_yield:.2f}%)"
            couleur_globale = "#28a745"
        elif liste_recommandations.count('TAUX FIXE') >= 2:
            strategie_globale = "TAUX FIXE"
            raison_globale = f"Majorité des scénarios montrent des taux en hausse depuis juin 2025 ({last_historical_yield:.2f}%)"
            couleur_globale = "#dc3545"
        else:
            strategie_globale = "STRATÉGIE FLEXIBLE"
            raison_globale = f"Signaux mixtes depuis juin 2025 ({last_historical_yield:.2f}%) - approche diversifiée recommandée"
            couleur_globale = "#ffc107"
        
        # Data quality indicator
        quality_score = sum(1 for source in live_data['sources'].values() if 'Live' in source)
        quality_text = "Surveillance économique en temps réel active" if quality_score >= 2 else "Surveillance économique limitée"
        
        st.markdown(f"""
        <div class="recommendation-box" style="background: linear-gradient(135deg, {couleur_globale} 0%, {couleur_globale}AA 100%);">
            <h2>▲ RECOMMANDATION GLOBALE SOFAC</h2>
            <h3>{strategie_globale}</h3>
            <p>{raison_globale}</p>
            <small>{quality_text}</small>
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
                    
                    **Métriques (vs juin 2025: {last_historical_yield:.2f}%):**
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
                        y=last_historical_yield, 
                        line_dash="dash", 
                        line_color="blue",
                        annotation_text=f"Juin 2025: {last_historical_yield:.2f}%"
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
                - **Basé sur:** Baisse attendue de {abs(changement_cas_base):.2f}% vs juin 2025 ({last_historical_yield:.2f}%)
                """)
            else:
                st.warning(f"""
                💰 **Coûts Évités avec TAUX FIXE:**
                
                - **Surcoûts évités annuellement:** {changement_cas_base * montant_emprunt * 10_000:,.0f} MAD
                - **Surcoûts évités totaux ({duree_emprunt} ans):** {impact_total:,.0f} MAD
                - **Basé sur:** Hausse attendue de {changement_cas_base:.2f}% vs juin 2025 ({last_historical_yield:.2f}%)
                """)
        else:
            st.info(f"""
            💰 **Impact Financier Limité:**
            
            - **Variation attendue:** ±{abs(changement_cas_base):.2f}% vs juin 2025 ({last_historical_yield:.2f}%)
            - **Impact annuel:** ±{abs(changement_cas_base) * montant_emprunt * 10_000:,.0f} MAD
            - **Approche flexible recommandée**
            """)
    
    # Professional footer
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    live_sources_count = sum(1 for source in live_data['sources'].values() if 'Live' in source)
    
    st.markdown(f"""
    <div class="footer">
        <p><strong>SOFAC</strong> - Modèle de Prédiction des Rendements 52-Semaines</p>
        <p>
            <strong>Sources:</strong> Bank Al-Maghrib, HCP • 
            <strong>Surveillance:</strong> {live_sources_count}/4 sources directes • 
            <strong>Modèle:</strong> Régression Linéaire Multiple
        </p>
        <p>
            <strong>Horizon:</strong> Juillet 2025 - Décembre 2026 • 
            <strong>Baseline:</strong> Juin 2025 ({last_historical_yield:.2f}%) • 
            <strong>Dernière mise à jour:</strong> {current_time}
        </p>
        <p><em>Les prédictions sont basées sur des données historiques et ne constituent pas des conseils financiers.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
