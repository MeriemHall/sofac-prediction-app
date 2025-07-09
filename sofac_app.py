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
import json
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
    .data-error {
        background: #f8d7da;
        border: 1px solid #dc3545;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour - updates every hour
def fetch_live_moroccan_data():
    """
    Fetch live data from Bank Al-Maghrib and HCP with enhanced error handling
    Updates automatically every hour
    """
    
    live_data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'policy_rate': 2.25,  # Current fallback
        'yield_52w': 2.40,    # Current fallback
        'inflation': 1.1,     # Current fallback
        'gdp_growth': 4.8,    # Current fallback
        'sources': {},
        'fetch_attempts': {},
        'success_count': 0,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8,ar;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    # =====================================================
    # 1. FETCH BANK AL-MAGHRIB POLICY RATE & 52-WEEK YIELD
    # =====================================================
    
    # Policy Rate from Bank Al-Maghrib
    try:
        bkam_urls = [
            "https://www.bkam.ma/Politique-monetaire/Cadre-strategique/Decision-de-la-politique-monetaire/Historique-des-decisions",
            "https://www.bkam.ma/Politique-monetaire",
            "https://www.bkam.ma/Marches/Principaux-indicateurs"
        ]
        
        policy_rate_found = False
        
        for url in bkam_urls:
            if policy_rate_found:
                break
                
            try:
                response = requests.get(url, headers=headers, timeout=15)
                live_data['fetch_attempts']['policy_rate'] = f"BAM Status: {response.status_code}"
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    # Enhanced patterns for policy rate detection
                    patterns = [
                        r'taux.*?directeur.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?taux.*?directeur',
                        r'politique.*?mon[eé]taire.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?politique.*?mon[eé]taire',
                        r'taux.*?r[eé]po.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?r[eé]po',
                        r'bank.*?al.*?maghrib.*?(\d+[,.]?\d*)%'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            try:
                                rate = float(match.replace(',', '.'))
                                if 0.25 <= rate <= 8.0:  # Reasonable policy rate range
                                    live_data['policy_rate'] = rate
                                    live_data['sources']['policy_rate'] = f'Bank Al-Maghrib Live'
                                    live_data['success_count'] += 1
                                    policy_rate_found = True
                                    break
                            except ValueError:
                                continue
                        if policy_rate_found:
                            break
                    
                    # Also look in tables
                    if not policy_rate_found:
                        tables = soup.find_all('table')
                        for table in tables:
                            rows = table.find_all('tr')
                            for row in rows[1:5]:  # Check first few data rows
                                cells = row.find_all(['td', 'th'])
                                if len(cells) >= 2:
                                    for cell in cells:
                                        cell_text = cell.get_text().strip()
                                        rate_match = re.search(r'(\d+[,.]?\d*)%?', cell_text)
                                        if rate_match:
                                            try:
                                                rate = float(rate_match.group(1).replace(',', '.'))
                                                if 0.25 <= rate <= 8.0:
                                                    live_data['policy_rate'] = rate
                                                    live_data['sources']['policy_rate'] = f'Bank Al-Maghrib Live (Table)'
                                                    live_data['success_count'] += 1
                                                    policy_rate_found = True
                                                    break
                                            except ValueError:
                                                continue
                                if policy_rate_found:
                                    break
                            if policy_rate_found:
                                break
                
                if policy_rate_found:
                    break
                    
            except requests.RequestException as e:
                live_data['fetch_attempts']['policy_rate'] = f"BAM Error: {str(e)[:50]}..."
                continue
        
        if not policy_rate_found:
            live_data['sources']['policy_rate'] = 'Fallback Value (Latest Known)'
    
    except Exception as e:
        live_data['fetch_attempts']['policy_rate'] = f"Exception: {str(e)[:50]}..."
        live_data['sources']['policy_rate'] = 'Fallback Value (Error)'
    
    # 52-Week Treasury Yield (Multiple approaches)
    try:
        treasury_yield_found = False
        
        # Method 1: Try Bank Al-Maghrib market data pages
        bkam_market_urls = [
            "https://www.bkam.ma/Marches/Principaux-indicateurs",
            "https://www.bkam.ma/Marches/Marche-monetaire",
            "https://www.bkam.ma/Statistiques"
        ]
        
        for url in bkam_market_urls:
            if treasury_yield_found:
                break
                
            try:
                response = requests.get(url, headers=headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    # Enhanced patterns for 52-week yield detection
                    patterns = [
                        r'52.*?semaines?.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?52.*?semaines?',
                        r'bons.*?tr[eé]sor.*?52.*?(\d+[,.]?\d*)',
                        r'treasury.*?52.*?week.*?(\d+[,.]?\d*)',
                        r'52w.*?(\d+[,.]?\d*)%?',
                        r'rendement.*?52.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?rendement.*?52'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            try:
                                rate = float(match.replace(',', '.'))
                                if 0.1 <= rate <= 12.0:  # Reasonable treasury yield range
                                    live_data['yield_52w'] = rate
                                    live_data['sources']['yield_52w'] = f'Bank Al-Maghrib Live (Markets)'
                                    live_data['success_count'] += 1
                                    treasury_yield_found = True
                                    break
                            except ValueError:
                                continue
                        if treasury_yield_found:
                            break
            except:
                continue
        
        # Method 2: Estimate from policy rate if not found directly
        if not treasury_yield_found:
            # Treasury yields typically trade at a spread to policy rate
            # Historical analysis shows 52-week yields are usually 10-50bp above policy rate
            current_spread = 0.15  # 15 basis points typical spread
            
            # Adjust spread based on economic conditions
            # If policy rate is low, spread tends to be higher
            if live_data['policy_rate'] < 2.0:
                current_spread = 0.25
            elif live_data['policy_rate'] > 4.0:
                current_spread = 0.10
            
            live_data['yield_52w'] = live_data['policy_rate'] + current_spread
            live_data['sources']['yield_52w'] = f'Estimated from Policy Rate (+{current_spread*100:.0f}bps)'
        
        live_data['fetch_attempts']['yield_52w'] = "Multiple BAM sources attempted"
        
    except Exception as e:
        live_data['fetch_attempts']['yield_52w'] = f"Error: {str(e)[:50]}..."
        live_data['sources']['yield_52w'] = 'Fallback Estimation'
    
    # =====================================================
    # 2. FETCH HCP INFLATION DATA
    # =====================================================
    try:
        hcp_inflation_urls = [
            "https://www.hcp.ma/Actualites_a5.html",
            "https://www.hcp.ma/Prix_r370.html",
            "https://www.hcp.ma/Indices-des-prix-a-la-consommation_r349.html",
            "https://www.hcp.ma/"
        ]
        
        inflation_found = False
        
        for url in hcp_inflation_urls:
            if inflation_found:
                break
                
            try:
                response = requests.get(url, headers=headers, timeout=15)
                live_data['fetch_attempts']['inflation'] = f"HCP Status: {response.status_code}"
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    # Enhanced patterns for inflation detection
                    patterns = [
                        r'inflation.*?(\d+[,.]?\d*)%.*?ann[ée]e?',
                        r'(\d+[,.]?\d*)%.*?inflation.*?ann[ée]e?',
                        r'hausse.*?prix.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?hausse.*?prix',
                        r'ipc.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?ipc',
                        r'indice.*?prix.*?consommation.*?(\d+[,.]?\d*)%',
                        r'variation.*?ann[ée]e?.*?(\d+[,.]?\d*)%',
                        r'glissement.*?annuel.*?(\d+[,.]?\d*)%'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            try:
                                rate = float(match.replace(',', '.'))
                                if 0 <= rate <= 20:  # Reasonable inflation range
                                    live_data['inflation'] = rate
                                    live_data['sources']['inflation'] = 'HCP Live'
                                    live_data['success_count'] += 1
                                    inflation_found = True
                                    break
                            except ValueError:
                                continue
                        if inflation_found:
                            break
            except requests.RequestException as e:
                live_data['fetch_attempts']['inflation'] = f"HCP Error: {str(e)[:50]}..."
                continue
        
        if not inflation_found:
            live_data['sources']['inflation'] = 'Fallback Value (Latest Known)'
    
    except Exception as e:
        live_data['fetch_attempts']['inflation'] = f"Exception: {str(e)[:50]}..."
        live_data['sources']['inflation'] = 'Fallback Value (Error)'
    
    # =====================================================
    # 3. FETCH HCP GDP GROWTH DATA
    # =====================================================
    try:
        hcp_gdp_urls = [
            "https://www.hcp.ma/Conjoncture-et-prevision-economique_r328.html",
            "https://www.hcp.ma/Comptes-nationaux_r302.html",
            "https://www.hcp.ma/Economie_r327.html",
            "https://www.hcp.ma/"
        ]
        
        gdp_found = False
        
        for url in hcp_gdp_urls:
            if gdp_found:
                break
                
            try:
                response = requests.get(url, headers=headers, timeout=15)
                live_data['fetch_attempts']['gdp_growth'] = f"HCP Status: {response.status_code}"
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    # Enhanced patterns for GDP growth detection
                    patterns = [
                        r'croissance.*?[ée]conomique.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?croissance.*?[ée]conomique',
                        r'pib.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?pib',
                        r'progression.*?(\d+[,.]?\d*)%.*?trimestre',
                        r'trimestre.*?(\d+[,.]?\d*)%.*?progression',
                        r'([ée]volution.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?ann[ée]e.*?2025',
                        r'2025.*?(\d+[,.]?\d*)%'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            try:
                                # Handle tuple results from some regex patterns
                                if isinstance(match, tuple):
                                    match = match[1] if match[1] else match[0]
                                
                                rate = float(match.replace(',', '.'))
                                if -5 <= rate <= 15:  # Reasonable GDP growth range
                                    live_data['gdp_growth'] = rate
                                    live_data['sources']['gdp_growth'] = 'HCP Live'
                                    live_data['success_count'] += 1
                                    gdp_found = True
                                    break
                            except (ValueError, IndexError):
                                continue
                        if gdp_found:
                            break
            except requests.RequestException as e:
                live_data['fetch_attempts']['gdp_growth'] = f"HCP Error: {str(e)[:50]}..."
                continue
        
        if not gdp_found:
            live_data['sources']['gdp_growth'] = 'Fallback Value (Latest Known)'
    
    except Exception as e:
        live_data['fetch_attempts']['gdp_growth'] = f"Exception: {str(e)[:50]}..."
        live_data['sources']['gdp_growth'] = 'Fallback Value (Error)'
    
    # =====================================================
    # 4. FINAL DATA VALIDATION AND PROCESSING
    # =====================================================
    
    # Data validation with reasonable bounds
    live_data['policy_rate'] = max(0.1, min(10.0, live_data['policy_rate']))
    live_data['yield_52w'] = max(0.1, min(15.0, live_data['yield_52w']))
    live_data['inflation'] = max(0.0, min(25.0, live_data['inflation']))
    live_data['gdp_growth'] = max(-10.0, min(20.0, live_data['gdp_growth']))
    
    # Ensure 52-week yield is not dramatically different from policy rate
    if abs(live_data['yield_52w'] - live_data['policy_rate']) > 3.0:
        live_data['yield_52w'] = live_data['policy_rate'] + 0.15
        live_data['sources']['yield_52w'] = 'Adjusted (Spread Correction)'
    
    # Set default sources if not already set
    for indicator in ['policy_rate', 'yield_52w', 'inflation', 'gdp_growth']:
        if indicator not in live_data['sources']:
            live_data['sources'][indicator] = 'Manual Fallback'
    
    return live_data

def display_live_data_panel(live_data):
    """Enhanced display panel with live data status"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Données en Temps Réel")
    
    # Calculate success rate
    live_sources = sum(1 for source in live_data['sources'].values() if 'Live' in source)
    total_sources = 4  # policy_rate, yield_52w, inflation, gdp_growth
    success_rate = (live_sources / total_sources) * 100
    
    # Status indicator
    if success_rate >= 75:
        st.sidebar.markdown('<div class="data-status">🟢 Excellent: Données majoritairement en direct</div>', unsafe_allow_html=True)
    elif success_rate >= 50:
        st.sidebar.markdown('<div class="data-warning">🟡 Bon: Données partiellement en direct</div>', unsafe_allow_html=True)
    elif success_rate >= 25:
        st.sidebar.markdown('<div class="data-warning">🟠 Moyen: Quelques données en direct</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="data-error">🔴 Limité: Données principalement estimées</div>', unsafe_allow_html=True)
    
    st.sidebar.write(f"**Sources en direct:** {live_sources}/{total_sources} ({success_rate:.0f}%)")
    
    # Current values with enhanced indicators
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # Policy rate
        if 'Live' in live_data['sources']['policy_rate']:
            indicator = "🟢"
            delta_color = "normal"
        else:
            indicator = "🔴"
            delta_color = "off"
        
        st.metric(
            f"{indicator} Taux Directeur", 
            f"{live_data['policy_rate']:.2f}%",
            help=f"Source: {live_data['sources']['policy_rate']}"
        )
        
        # Inflation
        if 'Live' in live_data['sources']['inflation']:
            indicator = "🟢"
        else:
            indicator = "🔴"
        
        st.metric(
            f"{indicator} Inflation", 
            f"{live_data['inflation']:.2f}%",
            help=f"Source: {live_data['sources']['inflation']}"
        )
    
    with col2:
        # 52-week yield
        if 'Live' in live_data['sources']['yield_52w']:
            indicator = "🟢"
        elif 'Estimated' in live_data['sources']['yield_52w']:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        st.metric(
            f"{indicator} Rendement 52s", 
            f"{live_data['yield_52w']:.2f}%",
            delta=f"+{(live_data['yield_52w'] - live_data['policy_rate']):.2f}%",
            help=f"Source: {live_data['sources']['yield_52w']}"
        )
        
        # GDP growth
        if 'Live' in live_data['sources']['gdp_growth']:
            indicator = "🟢"
        else:
            indicator = "🔴"
        
        st.metric(
            f"{indicator} Croissance PIB", 
            f"{live_data['gdp_growth']:.2f}%",
            help=f"Source: {live_data['sources']['gdp_growth']}"
        )
    
    # Last update and refresh controls
    st.sidebar.info(f"🕐 Dernière mise à jour: {live_data['last_updated']}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Actualiser", help="Force la mise à jour des données"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Détails", help="Afficher les détails des sources"):
            st.session_state.show_fetch_details = not st.session_state.get('show_fetch_details', False)
    
    # Detailed fetch information
    if st.session_state.get('show_fetch_details', False):
        with st.sidebar.expander("🔍 Détails des Sources", expanded=True):
            st.markdown("**Sources de Données:**")
            
            source_names = {
                'policy_rate': 'Taux Directeur BAM',
                'yield_52w': 'Rendement 52s BAM',
                'inflation': 'Inflation HCP',
                'gdp_growth': 'Croissance PIB HCP'
            }
            
            for key, source in live_data['sources'].items():
                if key in source_names:
                    if 'Live' in source:
                        status = "✅ Direct"
                    elif 'Estimated' in source or 'Adjusted' in source:
                        status = "⚠️ Estimé"
                    else:
                        status = "❌ Fallback"
                    
                    st.write(f"{status} **{source_names[key]}**")
                    st.write(f"   {source}")
            
            if 'fetch_attempts' in live_data and live_data['fetch_attempts']:
                st.markdown("**Tentatives de Récupération:**")
                for key, attempt in live_data['fetch_attempts'].items():
                    st.write(f"• {key}: {attempt}")

@st.cache_data
def create_monthly_dataset_with_live_data(live_data):
    """Create monthly dataset incorporating live data as the most recent point"""
    
    # Base historical data
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
    
    # Add current live data as the most recent point
    current_month = datetime.now().strftime('%Y-%m')
    if current_month not in donnees_historiques:
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
    date_fin = datetime.now().replace(day=1) + timedelta(days=32)
    date_fin = date_fin.replace(day=1) - timedelta(days=1)  # End of current month
    
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
            'Est_Point_Ancrage': est_ancrage,
            'Est_Live_Data': (date_str == current_month)
        })
        
        if date_courante.month == 12:
            date_courante = date_courante.replace(year=date_courante.year + 1, month=1)
        else:
            date_courante = date_courante.replace(month=date_courante.month + 1)
    
    return pd.DataFrame(donnees_mensuelles)

def train_prediction_model(df_mensuel):
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
    """Create economic scenarios starting from current live data"""
    
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2026, 12, 31)
    
    # If we're past July 2025, start from current date
    if datetime.now() > date_debut:
        date_debut = datetime.now().replace(day=1) + timedelta(days=32)
        date_debut = date_debut.replace(day=1)  # First day of next month
    
    dates_quotidiennes = []
    date_courante = date_debut
    
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
    # Policy decisions adapted to current situation
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
        
        # Base scenario parameters on current live data
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
            variation_marche = np.random.normal(0, 0.02)
            
            mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
            
            if nom_scenario == 'Conservateur':
                inflation_base = base_inflation + 0.3 * np.exp(-mois_depuis_debut / 18) + 0.2 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = base_gdp - 0.3 * (mois_depuis_debut / 18) + 0.4 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            elif nom_scenario == 'Cas_de_Base':
                inflation_base = base_inflation + 0.1 * np.exp(-mois_depuis_debut / 12) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = base_gdp - 0.1 * (mois_depuis_debut / 18) + 0.5 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            else:  # Optimiste
                inflation_base = base_inflation - 0.2 * (mois_depuis_debut / 18) + 0.1 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = base_gdp + 0.2 * (mois_depuis_debut / 18) + 0.6 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            
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
    """Generate predictions with smooth continuity from live data"""
    
    rendement_actuel = live_data['yield_52w']  # Use live 52-week yield as baseline
    predictions = {}
    
    for nom_scenario, scenario_df in scenarios.items():
        X_futur = scenario_df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
        rendements_bruts = modele.predict(X_futur)
        
        # Smooth continuity from current live data
        if len(rendements_bruts) > 0:
            premier_predit = rendements_bruts[0]
            discontinuite = premier_predit - rendement_actuel
            
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
        
        # Scenario-specific adjustments
        ajustements = []
        for i, ligne in scenario_df.iterrows():
            ajustement = 0
            
            if nom_scenario == 'Conservateur':
                ajustement += 0.1
            elif nom_scenario == 'Optimiste':
                ajustement -= 0.05
            
            jours_ahead = ligne['Jours_Ahead']
            incertitude = (jours_ahead / 365) * 0.1
            
            if nom_scenario == 'Conservateur':
                ajustement += incertitude
            elif nom_scenario == 'Optimiste':
                ajustement -= incertitude * 0.5
            
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
        
        # Enhanced confidence intervals
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

def generate_recommendations_with_live_context(predictions, live_data):
    """Generate recommendations using live data as baseline"""
    
    rendement_actuel = live_data['yield_52w']
    recommandations = {}
    
    for nom_scenario, pred_df in predictions.items():
        rendement_futur_moyen = pred_df['Rendement_Predit'].mean()
        changement_rendement = rendement_futur_moyen - rendement_actuel
        volatilite = pred_df['Rendement_Predit'].std()
        
        # Enhanced recommendation logic with live data context
        if changement_rendement > 0.3:
            recommandation = "TAUX FIXE"
            raison = f"Rendements attendus en hausse de {changement_rendement:.2f}% depuis le niveau actuel de {rendement_actuel:.2f}%. Sécuriser les taux maintenant."
        elif changement_rendement < -0.3:
            recommandation = "TAUX VARIABLE"
            raison = f"Rendements attendus en baisse de {abs(changement_rendement):.2f}% depuis le niveau actuel de {rendement_actuel:.2f}%. Profiter de la diminution des coûts."
        else:
            recommandation = "STRATÉGIE FLEXIBLE"
            raison = f"Rendements relativement stables autour du niveau actuel de {rendement_actuel:.2f}% (±{abs(changement_rendement):.2f}%)."
        
        # Risk assessment with volatility consideration
        if volatilite < 0.2:
            niveau_risque = "FAIBLE"
        elif volatilite < 0.4:
            niveau_risque = "MOYEN"
        else:
            niveau_risque = "ÉLEVÉ"
        
        # Confidence level based on data quality
        live_sources_count = sum(1 for source in live_data['sources'].values() if 'Live' in source)
        if live_sources_count >= 3:
            confiance = "ÉLEVÉE"
        elif live_sources_count >= 2:
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
    
    # Fetch live data first
    with st.spinner("🔄 Récupération des données en temps réel..."):
        live_data = fetch_live_moroccan_data()
    
    with st.sidebar:
        st.header("📊 Informations du Modèle")
        
        # Display live data panel
        display_live_data_panel(live_data)
        
        # Initialize or load cached model data
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
        
        # Live data incorporation notice
        st.info("🔄 Le modèle est automatiquement recalibré avec les dernières données disponibles.")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue d'Ensemble", "🔮 Prédictions Détaillées", "💼 Recommandations", "📊 Données Live"])
    
    with tab1:
        st.header("📈 Vue d'Ensemble des Prédictions")
        
        # Key metrics with live data context
        col1, col2, col3, col4 = st.columns(4)
        
        cas_de_base = st.session_state.predictions['Cas_de_Base']
        rendement_moyen = cas_de_base['Rendement_Predit'].mean()
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
                delta="Direct" if quality_score >= 3 else "Mixte"
            )
        
        # Enhanced overview chart with live data integration
        st.subheader("📊 Évolution des Rendements: Historique et Prédictions")
        
        fig_overview = go.Figure()
        
        # Historical data
        df_recent = st.session_state.df_mensuel.tail(8)
        
        # Separate live data points
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
            donnees_hebdo = pred_df[::7]  # Weekly sampling
            
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
        
        # Quick recommendations with live context
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
        
        # Enhanced metrics with live data context
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rendement Moyen", f"{pred_scenario['Rendement_Predit'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{pred_scenario['Rendement_Predit'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{pred_scenario['Rendement_Predit'].max():.2f}%")
        with col4:
            st.metric("Écart vs Live", f"{(pred_scenario['Rendement_Predit'].mean() - live_data['yield_52w']):+.2f}%")
        
        # Detailed prediction chart
        st.subheader(f"📊 Prédictions Quotidiennes - Scénario {scenario_selectionne}")
        
        donnees_affichage = pred_scenario[::3]  # Every 3 days for clarity
        
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
        
        # Add current live data point for reference
        fig_detail.add_trace(
            go.Scatter(
                x=[datetime.now().strftime('%Y-%m-%d')],
                y=[live_data['yield_52w']],
                mode='markers',
                name='Point Live Actuel',
                marker=dict(color='#22C55E', size=15, symbol='star'),
                text=[f"Live: {live_data['yield_52w']:.2f}%"],
                textposition='top center'
            )
        )
        
        fig_detail.update_layout(
            title=f"Prédictions Détaillées - {scenario_selectionne} (Continuité depuis données live)",
            xaxis_title="Date",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        # Export functionality
        if st.button("📥 Télécharger les Prédictions"):
            # Add live data context to export
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
        
        # Global recommendation with live data integration
        liste_recommandations = [rec['recommandation'] for rec in st.session_state.recommandations.values()]
        
        if liste_recommandations.count('TAUX VARIABLE') >= 2:
            strategie_globale = "TAUX VARIABLE"
            raison_globale = f"Majorité des scénarios montrent des taux en baisse depuis le niveau live actuel de {live_data['yield_52w']:.2f}%"
            couleur_globale = "#dc3545"
        else:
            strategie_globale = "STRATÉGIE FLEXIBLE"
            raison_globale = f"Signaux mixtes depuis le niveau live actuel de {live_data['yield_52w']:.2f}% - approche diversifiée recommandée"
            couleur_globale = "#ffc107"
        
        # Data quality indicator for recommendations
        quality_score = sum(1 for source in live_data['sources'].values() if 'Live' in source)
        quality_text = "Recommandation basée sur données majoritairement directes" if quality_score >= 3 else \
                      "Recommandation basée sur données mixtes" if quality_score >= 2 else \
                      "Recommandation basée sur estimations - à valider"
        
        st.markdown(f"""
        <div class="recommendation-box" style="background: linear-gradient(135deg, {couleur_globale} 0%, {couleur_globale}AA 100%);">
            <h2>🏆 RECOMMANDATION GLOBALE SOFAC</h2>
            <h3>{strategie_globale}</h3>
            <p>{raison_globale}</p>
            <small>{quality_text} ({quality_score}/4 sources directes)</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed scenario analysis with live context
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
                    # Mini chart with live baseline
                    fig_mini = go.Figure()
                    
                    pred_df = st.session_state.predictions[nom_scenario]
                    echantillon_mini = pred_df[::30]  # Monthly sampling
                    
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
        
        # Enhanced financial impact calculator
        st.subheader("💰 Calculateur d'Impact Financier (Basé sur Données Live)")
        
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
            if changement_cas_base < 0:  # Baisse attendue
                st.success(f"""
                💰 **Économies Potentielles avec TAUX VARIABLE:**
                
                - **Économies annuelles:** {abs(changement_cas_base) * montant_emprunt * 10_000:,.0f} MAD
                - **Économies totales ({duree_emprunt} ans):** {abs(impact_total):,.0f} MAD
                - **Basé sur:** Baisse attendue de {abs(changement_cas_base):.2f}% vs niveau live {live_data['yield_52w']:.2f}%
                - **Qualité prédiction:** {quality_score}/4 sources directes
                """)
            else:  # Hausse attendue
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
    
    with tab4:
        st.header("📊 Tableau de Bord des Données Live")
        
        # Live data dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Sources de Données en Temps Réel")
            
            source_status = []
            indicators = {
                'policy_rate': ('Bank Al-Maghrib', 'Taux Directeur', '🏦'),
                'yield_52w': ('Bank Al-Maghrib', 'Rendement 52s', '📈'),
                'inflation': ('HCP', 'Inflation', '📊'),
                'gdp_growth': ('HCP', 'Croissance PIB', '💹')
            }
            
            for key, (institution, indicator, icon) in indicators.items():
                source = live_data['sources'][key]
                if 'Live' in source:
                    status = "🟢 Direct"
                    status_color = "#4CAF50"
                elif 'Estimated' in source or 'Adjusted' in source:
                    status = "🟡 Estimé"
                    status_color = "#FF9800"
                else:
                    status = "🔴 Fallback"
                    status_color = "#F44336"
                
                st.markdown(f"""
                <div style="border: 1px solid {status_color}; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                    <h4>{icon} {indicator}</h4>
                    <p><strong>Institution:</strong> {institution}</p>
                    <p><strong>Statut:</strong> <span style="color: {status_color};">{status}</span></p>
                    <p><strong>Valeur:</strong> {live_data[key]:.2f}%</p>
                    <small>{source}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔄 Historique des Mises à Jour")
            
            # Fetch attempts information
            if live_data.get('fetch_attempts'):
                st.markdown("**Dernières Tentatives de Récupération:**")
                for source, attempt in live_data['fetch_attempts'].items():
                    source_name = {
                        'policy_rate': 'Taux Directeur',
                        'yield_52w': 'Rendement 52s',
                        'inflation': 'Inflation',
                        'gdp_growth': 'Croissance PIB'
                    }.get(source, source)
                    
                    st.write(f"**{source_name}:** {attempt}")
            
            # Update schedule
            st.markdown("**Calendrier de Mise à Jour:**")
            st.write("• **Automatique:** Toutes les heures")
            st.write("• **Manuel:** Bouton 'Actualiser'")
            st.write("• **Recalibration:** Lors de nouvelles données")
            
            next_update = (datetime.now() + timedelta(hours=1)).strftime('%H:%M')
            st.write(f"• **Prochaine mise à jour:** {next_update}")
            
            # Data quality trend (placeholder for future enhancement)
            st.markdown("**Qualité des Données (24h):**")
            current_quality = sum(1 for source in live_data['sources'].values() if 'Live' in source)
            st.progress(current_quality / 4)
            st.write(f"Score actuel: {current_quality}/4 sources directes")
        
        # Live data visualization
        st.subheader("📈 Évolution des Indicateurs Clés")
        
        # Create a comparison chart showing current vs recent historical values
        fig_indicators = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Taux Directeur (%)', 'Rendement 52s (%)', 'Inflation (%)', 'Croissance PIB (%)'],
            vertical_spacing=0.12
        )
        
        # Historical trend (last 6 months) vs current live value
        recent_data = st.session_state.df_mensuel.tail(6)
        
        indicators_data = [
            ('Taux_Directeur', live_data['policy_rate'], 1, 1),
            ('Rendement_52s', live_data['yield_52w'], 1, 2),
            ('Inflation', live_data['inflation'], 2, 1),
            ('Croissance_PIB', live_data['gdp_growth'], 2, 2)
        ]
        
        for col, live_val, row, col_pos in indicators_data:
            # Historical trend
            fig_indicators.add_trace(
                go.Scatter(
                    x=recent_data['Date'],
                    y=recent_data[col],
                    mode='lines+markers',
                    name=f'Historique {col}',
                    line=dict(color='#60A5FA', width=2),
                    showlegend=False
                ),
                row=row, col=col_pos
            )
            
            # Current live value
            fig_indicators.add_trace(
                go.Scatter(
                    x=[datetime.now().strftime('%Y-%m')],
                    y=[live_val],
                    mode='markers',
                    name=f'Live {col}',
                    marker=dict(color='#22C55E', size=12, symbol='star'),
                    showlegend=False
                ),
                row=row, col=col_pos
            )
        
        fig_indicators.update_layout(
            height=500,
            template="plotly_white",
            title_text="Indicateurs Économiques: Évolution Récente et Valeurs Live"
        )
        
        st.plotly_chart(fig_indicators, use_container_width=True)
        
        # Export live data report
        if st.button("📥 Exporter Rapport de Données Live"):
            report_data = {
                'timestamp': live_data['last_updated'],
                'data_quality_score': f"{sum(1 for s in live_data['sources'].values() if 'Live' in s)}/4",
                'indicators': {
                    'policy_rate': {'value': live_data['policy_rate'], 'source': live_data['sources']['policy_rate']},
                    'yield_52w': {'value': live_data['yield_52w'], 'source': live_data['sources']['yield_52w']},
                    'inflation': {'value': live_data['inflation'], 'source': live_data['sources']['inflation']},
                    'gdp_growth': {'value': live_data['gdp_growth'], 'source': live_data['sources']['gdp_growth']}
                },
                'fetch_attempts': live_data.get('fetch_attempts', {}),
                'recommendations_summary': {
                    scenario: rec['recommandation'] 
                    for scenario, rec in st.session_state.recommandations.items()
                }
            }
            
            report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="Télécharger Rapport JSON",
                data=report_json,
                file_name=f"sofac_live_data_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    # Enhanced footer with live data attribution
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
        elif liste_recommandations.count('TAUX FIXE') >= 2:
            strategie_globale = "TAUX FIXE"
            raison_globale = f"Majorité des scénarios montrent des taux en hausse depuis le niveau live actuel de {live_data['yield_52w']:.2f}%"
            couleur_globale =
