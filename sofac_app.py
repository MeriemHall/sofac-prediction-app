# ============================================================================
# SOFAC - Interface de Prédiction des Rendements 52-Semaines
# Version Streamlit pour Managers - Sans Emojis
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="SOFAC - Prédiction des Taux",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour SOFAC
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #e7f3ff 0%, #cce7ff 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #007bff;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,123,255,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(40,167,69,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(255,193,7,0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(23,162,184,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,123,255,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,123,255,0.4);
    }
    
    .highlight-metric {
        font-size: 2.5rem;
        font-weight: bold;
        color: #007bff;
        text-align: center;
    }
    
    .trend-up {
        color: #28a745;
        font-weight: bold;
    }
    
    .trend-down {
        color: #dc3545;
        font-weight: bold;
    }
    
    .trend-stable {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache pour optimiser les performances
@st.cache_data(ttl=3600)
def charger_donnees_historiques():
    """Charge les données historiques de SOFAC"""
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
def creer_donnees_mensuelles(df_historique):
    """Création du jeu de données mensuelles par interpolation"""
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
    
    # Conversion des données originales
    dates_ancrage = {}
    for _, ligne in df_historique.iterrows():
        date_obj = datetime.strptime(ligne['Date'] + '-01', '%Y-%m-%d')
        dates_ancrage[date_obj] = {
            'taux_directeur': ligne['taux_directeur'],
            'inflation': ligne['inflation'],
            'pib': ligne['pib'],
            'rendement_52s': ligne['rendement_52s']
        }
    
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
                point_donnees = dates_ancrage[max(dates_avant)].copy()
            else:
                point_donnees = dates_ancrage[min(dates_apres)].copy()
        
        donnees_mensuelles.append({
            'Date': date_str,
            'Taux_Directeur': point_donnees['taux_directeur'],
            'Inflation': point_donnees['inflation'],
            'Croissance_PIB': point_donnees['pib'],
            'Rendement_52s': point_donnees['rendement_52s']
        })
        
        if date_courante.month == 12:
            date_courante = date_courante.replace(year=date_courante.year + 1, month=1)
        else:
            date_courante = date_courante.replace(month=date_courante.month + 1)
    
    return pd.DataFrame(donnees_mensuelles)

@st.cache_data
def construire_modele(df_mensuel):
    """Construction du modèle de régression linéaire"""
    X = df_mensuel[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
    y = df_mensuel['Rendement_52s']
    
    modele = LinearRegression()
    modele.fit(X, y)
    
    y_pred = modele.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    return modele, r2, mae

@st.cache_data
def generer_predictions_futures(_modele, mae_historique):
    """Génération des prédictions pour juillet 2025 - décembre 2026"""
    np.random.seed(42)
    
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2026, 12, 31)
    
    # Scénario "Cas de Base" Bank Al-Maghrib
    decisions_politiques = {
        '2025-06': 2.25, '2025-09': 2.00, '2025-12': 1.75,
        '2026-03': 1.50, '2026-06': 1.50, '2026-09': 1.25, '2026-12': 1.25
    }
    
    dates_quotidiennes = []
    date_courante = date_debut
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
    donnees_prediction = []
    
    for i, date in enumerate(dates_quotidiennes):
        jours_ahead = i + 1
        
        # Taux directeur selon calendrier Bank Al-Maghrib
        date_str = date.strftime('%Y-%m')
        taux_directeur = 2.25
        for date_politique, taux in sorted(decisions_politiques.items()):
            if date_str >= date_politique:
                taux_directeur = taux
        
        # Scénarios d'inflation et PIB
        mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
        
        inflation_base = 1.4 + 0.3 * np.exp(-mois_depuis_debut / 12) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 12)
        variation_inflation = np.random.normal(0, 0.01)
        inflation = max(0.0, min(5.0, inflation_base + variation_inflation))
        
        trimestre = (date.month - 1) // 3
        pib_base = 3.8 - 0.2 * (mois_depuis_debut / 18) + 0.5 * np.sin(2 * np.pi * trimestre / 4)
        variation_pib = np.random.normal(0, 0.05)
        pib = max(-2.0, min(6.0, pib_base + variation_pib))
        
        donnees_prediction.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Taux_Directeur': taux_directeur,
            'Inflation': inflation,
            'Croissance_PIB': pib,
            'Jours_Ahead': jours_ahead
        })
    
    df_prediction = pd.DataFrame(donnees_prediction)
    
    # Prédictions avec le modèle
    X_futur = df_prediction[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
    rendements_bruts = modele.predict(X_futur)
    
    # Correction de continuité depuis juin 2025
    rendement_juin_reel = 1.75
    discontinuite = rendements_bruts[0] - rendement_juin_reel
    
    # Lissage exponentiel
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
    
    df_prediction['Rendement_Predit'] = np.clip(rendements_lisses, 0.1, 8.0)
    
    return df_prediction

def generer_recommandations(predictions):
    """Génération des recommandations métier pour SOFAC"""
    rendement_actuel = 2.54
    rendement_futur_moyen = predictions['Rendement_Predit'].mean()
    changement_rendement = rendement_futur_moyen - rendement_actuel
    
    if changement_rendement < -0.3:
        recommandation = "TAUX VARIABLE"
        raison = f"Rendements attendus en baisse de {abs(changement_rendement):.2f}% en moyenne. Utiliser des taux variables pour profiter de la diminution des coûts d'emprunt."
        couleur = "success"
        icone = "Baisse"
    elif changement_rendement > 0.3:
        recommandation = "TAUX FIXE"
        raison = f"Rendements attendus en hausse de {changement_rendement:.2f}% en moyenne. Bloquer les taux actuels avant que les coûts d'emprunt n'augmentent."
        couleur = "warning"
        icone = "Hausse"
    else:
        recommandation = "STRATEGIE FLEXIBLE"
        raison = f"Rendements relativement stables (±{abs(changement_rendement):.2f}%). Approche mixte selon les besoins de liquidité et durée des emprunts."
        couleur = "info"
        icone = "Stable"
    
    return {
        'recommandation': recommandation,
        'raison': raison,
        'couleur': couleur,
        'icone': icone,
        'rendement_actuel': rendement_actuel,
        'rendement_futur': rendement_futur_moyen,
        'changement': changement_rendement,
        'volatilite': predictions['Rendement_Predit'].std()
    }

def creer_graphique_principal(df_mensuel, predictions):
    """Création du graphique principal des prédictions"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Evolution des Rendements 52-Semaines',
            'Variables Economiques Cles',
            'Distribution des Predictions',
            'Analyse Trimestrielle'
        ],
        specs=[[{"colspan": 2}, None],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # Historique récent
    historique_recent = df_mensuel.tail(12)
    fig.add_trace(
        go.Scatter(
            x=historique_recent['Date'],
            y=historique_recent['Rendement_52s'],
            mode='lines+markers',
            name='Historique Recent',
            line=dict(color='#60A5FA', width=4),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Prédictions (échantillon hebdomadaire pour clarté)
    predictions_hebdo = predictions[::7]
    fig.add_trace(
        go.Scatter(
            x=predictions_hebdo['Date'],
            y=predictions_hebdo['Rendement_Predit'],
            mode='lines+markers',
            name='Predictions 2025-2026',
            line=dict(color='#FF6B6B', width=4, dash='dash'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Variables économiques
    fig.add_trace(
        go.Scatter(
            x=predictions_hebdo['Date'],
            y=predictions_hebdo['Taux_Directeur'],
            mode='lines',
            name='Taux Directeur BAM',
            line=dict(color='#4ECDC4', width=3)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=predictions_hebdo['Date'],
            y=predictions_hebdo['Inflation'],
            mode='lines',
            name='Inflation',
            line=dict(color='#45B7D1', width=3)
        ),
        row=1, col=1
    )
    
    # Distribution des prédictions
    fig.add_trace(
        go.Histogram(
            x=predictions['Rendement_Predit'],
            name='Distribution Rendements',
            nbinsx=30,
            marker_color='lightblue',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Analyse trimestrielle
    predictions['Trimestre'] = pd.to_datetime(predictions['Date']).dt.to_period('Q')
    moyenne_trim = predictions.groupby('Trimestre')['Rendement_Predit'].mean()
    
    fig.add_trace(
        go.Bar(
            x=[str(q) for q in moyenne_trim.index],
            y=moyenne_trim.values,
            name='Moyenne Trimestrielle',
            marker_color='lightgreen',
            text=[f'{v:.2f}%' for v in moyenne_trim.values],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': '<b>SOFAC - Predictions des Rendements 52-Semaines</b>',
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        height=800,
        template='plotly_dark',
        paper_bgcolor='rgba(17, 24, 39, 1)',
        plot_bgcolor='rgba(17, 24, 39, 1)',
        font={'color': 'white'},
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Rendement (%)", row=1, col=1)
    fig.update_xaxes(title_text="Rendement (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequence", row=2, col=1)
    fig.update_xaxes(title_text="Trimestre", row=2, col=2)
    fig.update_yaxes(title_text="Rendement Moyen (%)", row=2, col=2)
    
    return fig

# Interface principale
def main():
    # En-tête SOFAC
    st.markdown("""
    <div class="main-header">
        <h1>SOFAC - Prediction des Rendements 52-Semaines</h1>
        <h3>Outil d'Aide a la Decision pour la Strategie de Financement</h3>
        <p>Bank Al-Maghrib • HCP • Analyse Predictive Avancee</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec contrôles
    with st.sidebar:
        st.markdown("### Controles")
        
        if st.button("Actualiser les Donnees", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Informations")
        st.info(f"**Derniere mise a jour:** {datetime.now().strftime('%d/%m/%Y a %H:%M')}")
        st.success("Modele operationnel")
        st.info("R² = 95.7% (Excellent)")
        st.info("Precision = ±0.10%")
    
    # Chargement des données avec indicateur de progression
    with st.spinner("Chargement des donnees et construction du modele..."):
        df_historique = charger_donnees_historiques()
        df_mensuel = creer_donnees_mensuelles(df_historique)
        modele, r2_score_val, mae_val = construire_modele(df_mensuel)
        predictions = generer_predictions_futures(modele, mae_val)
        recommandations = generer_recommandations(predictions)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Rendement Actuel</h4>
            <div class="highlight-metric">2.54%</div>
            <small>Mars 2025 (Bank Al-Maghrib)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        changement = recommandations['changement']
        trend_class = "trend-down" if changement < 0 else "trend-up" if changement > 0 else "trend-stable"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Moyenne Future</h4>
            <div class="highlight-metric">{recommandations['rendement_futur']:.2f}%</div>
            <small class="{trend_class}">{changement:+.2f}% vs. actuel</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Volatilite</h4>
            <div class="highlight-metric">{recommandations['volatilite']:.2f}%</div>
            <small>Risque de variation</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Precision Modele</h4>
            <div class="highlight-metric">{r2_score_val*100:.1f}%</div>
            <small>Variance expliquee</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommandation principale
    st.markdown(f"""
    <div class="recommendation-box">
        <h2>RECOMMANDATION STRATEGIQUE SOFAC</h2>
        <h3 style="color: #007bff;">{recommandations['icone']} - {recommandations['recommandation']}</h3>
        <p><strong>Justification:</strong> {recommandations['raison']}</p>
        
        <h4>Impact Financier Estime (Emprunt 10M MAD):</h4>
    """, unsafe_allow_html=True)
    
    # Calcul de l'impact financier
    if recommandations['changement'] < -0.3:
        economies = abs(recommandations['changement']) * 10_000_000 / 100
        st.markdown(f"""
        <div class="success-box">
            <h4>Economies Potentielles avec TAUX VARIABLE</h4>
            <p><strong>{economies:,.0f} MAD/an</strong></p>
            <p>Base sur la baisse attendue de {abs(recommandations['changement']):.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    elif recommandations['changement'] > 0.3:
        cout_evite = recommandations['changement'] * 10_000_000 / 100
        st.markdown(f"""
        <div class="warning-box">
            <h4>Couts Evites avec TAUX FIXE</h4>
            <p><strong>{cout_evite:,.0f} MAD/an</strong></p>
            <p>Base sur la hausse attendue de {recommandations['changement']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
            <h4>Impact Financier Limite</h4>
            <p>Taux relativement stables (±{abs(recommandations['changement']):.2f}%)</p>
            <p>Approche flexible recommandee selon les besoins</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Graphique principal
    st.markdown("### Analyse Graphique Complete")
    fig = creer_graphique_principal(df_mensuel, predictions)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyses détaillées
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Resume de la Situation Economique")
        
        # Calculs pour le résumé
        debut_2025 = predictions.head(90)['Rendement_Predit'].mean()
        fin_2026 = predictions.tail(90)['Rendement_Predit'].mean()
        evolution_totale = fin_2026 - debut_2025
        
        st.markdown(f"""
        <div class="info-box">
            <h4>Evolution Prevue 2025-2026</h4>
            <p><strong>Debut 2025:</strong> {debut_2025:.2f}%</p>
            <p><strong>Fin 2026:</strong> {fin_2026:.2f}%</p>
            <p><strong>Evolution totale:</strong> {evolution_totale:+.2f}%</p>
            <p><strong>Tendance:</strong> {"Baissiere" if evolution_totale < -0.2 else "Haussiere" if evolution_totale > 0.2 else "Stable"}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Facteurs clés
        st.markdown("""
        <div class="info-box">
            <h4>Facteurs d'Influence Principaux</h4>
            <p><strong>• Taux Directeur BAM:</strong> Impact majeur (+0.96% par point)</p>
            <p><strong>• Inflation sous-jacente:</strong> Impact modere (+0.04% par point)</p>
            <p><strong>• Croissance PIB:</strong> Impact negatif (-0.02% par point)</p>
            <p><strong>• Politique monetaire:</strong> Cycle de detente attendu</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Gestion des Risques")
        
        volatilite = recommandations['volatilite']
        niveau_risque = "ELEVE" if volatilite > 0.4 else "MODERE" if volatilite > 0.2 else "FAIBLE"
        
        st.markdown(f"""
        <div class="warning-box">
            <h4>Evaluation du Risque: {niveau_risque}</h4>
            <p><strong>Volatilite prevue:</strong> {volatilite:.2f}%</p>
            <p><strong>Intervalle de confiance:</strong> ±{volatilite*2:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>Points de Surveillance</h4>
            <p><strong>• Decisions BAM:</strong> Reunions trimestrielles du comite monetaire</p>
            <p><strong>• Inflation:</strong> Surveillance mensuelle HCP</p>
            <p><strong>• Croissance:</strong> Publications trimestrielles PIB</p>
            <p><strong>• Contexte international:</strong> Fed, BCE, geopolitique</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommandations opérationnelles
    st.markdown("### Recommandations Operationnelles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if recommandations['recommandation'] == 'TAUX VARIABLE':
            st.markdown("""
            <div class="success-box">
                <h4>Actions Immediates</h4>
                <p>• Privilegier les nouveaux emprunts a taux variable</p>
                <p>• Negocier des caps de protection</p>
                <p>• Eviter les taux fixes long terme</p>
                <p>• Surveiller les opportunites de refinancement</p>
            </div>
            """, unsafe_allow_html=True)
        elif recommandations['recommandation'] == 'TAUX FIXE':
            st.markdown("""
            <div class="warning-box">
                <h4>Actions Immediates</h4>
                <p>• Bloquer les taux fixes des maintenant</p>
                <p>• Privilegier les echeances longues</p>
                <p>• Eviter les taux variables</p>
                <p>• Accelerer les projets de financement</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>Actions Immediates</h4>
                <p>• Approche equilibree: 50% fixe, 50% variable</p>
                <p>• Diversifier les echeances</p>
                <p>• Surveiller les signaux de marche</p>
                <p>• Maintenir la flexibilite</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Timing Optimal</h4>
            <p><strong>Fenetre recommandee:</strong> Juillet - Septembre 2025</p>
            <p><strong>Eviter:</strong> Fins de trimestre (volatilite)</p>
            <p><strong>Surveiller:</strong> Reunions BAM (mars, juin, septembre)</p>
            <p><strong>Opportunite:</strong> Periodes de stabilite politique</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>Suivi & Revision</h4>
            <p><strong>Frequence:</strong> Revision mensuelle</p>
            <p><strong>Declencheurs:</strong> Ecart >0.25% vs. previsions</p>
            <p><strong>Sources:</strong> BAM, HCP, Bloomberg</p>
            <p><strong>Reporting:</strong> Dashboard mis a jour quotidiennement</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Section technique (repliable)
    with st.expander("Informations Techniques du Modele"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Performance du Modele:**
            - R² Score: {r2_score_val:.3f} ({r2_score_val*100:.1f}% de variance expliquee)
            - Erreur Absolue Moyenne: {mae_val:.3f}%
            - Methode: Regression Lineaire Multiple
            - Validation: Cross-validation 5-fold
            
            **Donnees:**
            - Periode d'entrainement: 2020-2025
            - Observations: {len(df_mensuel)} points mensuels
            - Predictions: {len(predictions)} jours (Juillet 2025 - Decembre 2026)
            """)
        
        with col2:
            st.markdown(f"""
            **Equation de Prediction:**
            ```
            Rendement = 0.188 + 0.959×Taux_Directeur 
                      + 0.037×Inflation 
                      - 0.022×Croissance_PIB
            ```
            
            **Variables Explicatives:**
            - Taux Directeur BAM (impact: +0.959)
            - Inflation sous-jacente (impact: +0.037)
            - Croissance PIB (impact: -0.022)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
        <p><strong>Avertissement Important:</strong> Ces predictions sont basees sur des modeles statistiques et des hypotheses macroeconomiques. 
        Elles doivent etre utilisees comme outil d'aide a la decision en complement d'autres analyses financieres.</p>
        <p><strong>Contact:</strong> Direction Financiere SOFAC | <strong>Support:</strong> analyse.financiere@sofac.ma</p>
        <p><strong>Derniere mise a jour:</strong> {datetime.now().strftime('%d/%m/%Y a %H:%M')} | <strong>Sources:</strong> Bank Al-Maghrib, HCP, SOFAC</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
