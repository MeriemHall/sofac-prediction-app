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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_monthly_dataset():
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
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2026, 12, 31)
    
    dates_quotidiennes = []
    date_courante = date_debut
    
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
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
            variation_marche = np.random.normal(0, 0.02)
            
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
    rendement_juin_reel = 1.75
    predictions = {}
    
    for nom_scenario, scenario_df in scenarios.items():
        X_futur = scenario_df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
        rendements_bruts = modele.predict(X_futur)
        
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
    rendement_actuel = 2.54
    recommandations = {}
    
    for nom_scenario, pred_df in predictions.items():
        rendement_futur_moyen = pred_df['Rendement_Predit'].mean()
        changement_rendement = rendement_futur_moyen - rendement_actuel
        volatilite = pred_df['Rendement_Predit'].std()
        
        if changement_rendement > 0.3:
            recommandation = "TAUX FIXE"
            raison = f"Rendements attendus en hausse de {changement_rendement:.2f}% en moyenne."
        elif changement_rendement < -0.3:
            recommandation = "TAUX VARIABLE"
            raison = f"Rendements attendus en baisse de {abs(changement_rendement):.2f}% en moyenne."
        else:
            recommandation = "STRATÉGIE FLEXIBLE"
            raison = f"Rendements relativement stables (±{abs(changement_rendement):.2f}%)."
        
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
    st.markdown("""
    <div class="main-header">
        <h1>🇲🇦 SOFAC - Modèle de Prédiction des Rendements 52-Semaines</h1>
        <p>Système d'aide à la décision pour optimiser les coûts de financement</p>
        <p><strong>Horizon:</strong> Juillet 2025 - Décembre 2026 | <strong>Perspective:</strong> Emprunteur</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📊 Informations du Modèle")
        
        if 'data_loaded' not in st.session_state:
            with st.spinner("Chargement des données..."):
                st.session_state.df_mensuel = create_monthly_dataset()
                st.session_state.modele, st.session_state.r2, st.session_state.mae, st.session_state.rmse, st.session_state.mae_vc = train_prediction_model(st.session_state.df_mensuel)
                st.session_state.scenarios = create_economic_scenarios()
                st.session_state.predictions = generate_predictions(st.session_state.scenarios, st.session_state.modele, st.session_state.mae)
                st.session_state.recommandations = generate_recommendations(st.session_state.predictions)
                st.session_state.data_loaded = True
        
        st.success("✅ Données chargées!")
        
        st.subheader("🎯 Performance du Modèle")
        st.metric("R² Score", f"{st.session_state.r2:.1%}")
        st.metric("Précision", f"±{st.session_state.mae:.2f}%")
        st.metric("Validation Croisée", f"±{st.session_state.mae_vc:.2f}%")
        
        st.subheader("📊 Situation Actuelle")
        st.metric("Rendement Actuel", "2.54%")
        st.metric("Taux Directeur", "2.25%")
    
    tab1, tab2, tab3 = st.tabs(["📈 Vue d'Ensemble", "🔮 Prédictions Détaillées", "💼 Recommandations"])
    
    with tab1:
        st.header("📈 Vue d'Ensemble des Prédictions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        cas_de_base = st.session_state.predictions['Cas_de_Base']
        rendement_moyen = cas_de_base['Rendement_Predit'].mean()
        changement = rendement_moyen - 2.54
        volatilite = cas_de_base['Rendement_Predit'].std()
        
        with col1:
            st.metric("Rendement Moyen Prédit", f"{rendement_moyen:.2f}%", delta=f"{changement:+.2f}%")
        with col2:
            st.metric("Changement vs Actuel", f"{changement:+.2f}%")
        with col3:
            st.metric("Volatilité", f"{volatilite:.2f}%")
        with col4:
            st.metric("Horizon de Prédiction", f"{len(cas_de_base)} jours")
        
        st.subheader("📊 Évolution des Rendements par Scénario")
        
        fig_overview = go.Figure()
        
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
        
        couleurs = {'Conservateur': '#FF6B6B', 'Cas_de_Base': '#4ECDC4', 'Optimiste': '#45B7D1'}
        
        for nom_scenario, pred_df in st.session_state.predictions.items():
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
            template="plotly_white"
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
        
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
            index=0
        )
        
        pred_scenario = st.session_state.predictions[scenario_selectionne]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rendement Moyen", f"{pred_scenario['Rendement_Predit'].mean():.2f}%")
        with col2:
            st.metric("Rendement Min", f"{pred_scenario['Rendement_Predit'].min():.2f}%")
        with col3:
            st.metric("Rendement Max", f"{pred_scenario['Rendement_Predit'].max():.2f}%")
        with col4:
            st.metric("Écart-type", f"{pred_scenario['Rendement_Predit'].std():.2f}%")
        
        st.subheader(f"📊 Prédictions Quotidiennes - Scénario {scenario_selectionne}")
        
        donnees_affichage = pred_scenario[::3]
        
        fig_detail = go.Figure()
        
        couleurs = {'Conservateur': '#FF6B6B', 'Cas_de_Base': '#4ECDC4', 'Optimiste': '#45B7D1'}
        
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
    
    with tab3:
        st.header("💼 Recommandations Stratégiques")
        
        liste_recommandations = [rec['recommandation'] for rec in st.session_state.recommandations.values()]
        
        if liste_recommandations.count('TAUX VARIABLE') >= 2:
            strategie_globale = "TAUX VARIABLE"
            raison_globale = "Majorité des scénarios montrent des taux en baisse"
            couleur_globale = "#28a745"
        elif liste_recommandations.count('TAUX FIXE') >= 2:
            strategie_globale = "TAUX FIXE"
            raison_globale = "Majorité des scénarios montrent des taux en hausse"
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
        
        st.subheader("📊 Analyse Détaillée par Scénario")
        
        for nom_scenario, rec in st.session_state.recommandations.items():
            with st.expander(f"📈 Scénario {nom_scenario}", expanded=True):
                st.markdown(f"""
                **Recommandation:** {rec['recommandation']}
                
                **Justification:** {rec['raison']}
                
                **Métriques:**
                - Rendement moyen prédit: {rec['rendement_futur_moyen']:.2f}%
                - Changement vs actuel: {rec['changement_rendement']:+.2f}%
                - Volatilité: {rec['volatilite']:.2f}%
                - Niveau de risque: {rec['niveau_risque']}
                """)
        
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
            if changement_cas_base < 0:
                economies = abs(changement_cas_base) * montant_emprunt * 1_000_000 / 100
                st.success(f"""
                💰 **Économies Potentielles avec TAUX VARIABLE:**
                
                - {economies:,.0f} MAD/an
                - Basé sur une baisse attendue de {abs(changement_cas_base):.2f}%
                - Pour un emprunt de {montant_emprunt}M MAD
                """)
            else:
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
# Enhanced Data Fetcher with 52-Week Treasury Yield
# Add this to your existing sofac_app.py

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_enhanced_moroccan_data():
    """Enhanced fetcher including 52-week treasury yields"""
    
    live_data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'policy_rate': 2.25,  # Fallback
        'yield_52w': 2.40,    # Fallback
        'inflation': 1.1,     # Fallback
        'gdp_growth': 4.8,    # Fallback
        'sources': {},
        'fetch_attempts': {},
        'fetch_success': False
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.8,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    }
    
    # =====================================================
    # 1. FETCH BANK AL-MAGHRIB POLICY RATE
    # =====================================================
    try:
        bkam_policy_url = "https://www.bkam.ma/Politique-monetaire/Cadre-strategique/Decision-de-la-politique-monetaire/Historique-des-decisions"
        
        response = requests.get(bkam_policy_url, headers=headers, timeout=15)
        live_data['fetch_attempts']['policy_rate'] = f"Status: {response.status_code}"
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for the most recent policy rate in the table
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:3]:  # Check first few data rows
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        date_cell = cells[0].get_text().strip()
                        rate_cell = cells[1].get_text().strip()
                        
                        # Extract rate percentage
                        rate_match = re.search(r'(\d+[,.]?\d*)', rate_cell)
                        if rate_match:
                            rate = float(rate_match.group(1).replace(',', '.'))
                            if 0.5 <= rate <= 10:  # Reasonable range
                                live_data['policy_rate'] = rate
                                live_data['sources']['policy_rate'] = f'Bank Al-Maghrib Live ({date_cell})'
                                live_data['fetch_success'] = True
                                break
                
                if live_data['fetch_success']:
                    break
        
    except Exception as e:
        live_data['fetch_attempts']['policy_rate'] = f"Error: {str(e)[:50]}..."
        live_data['sources']['policy_rate'] = 'Fallback (Fetch failed)'
    
    # =====================================================
    # 2. FETCH 52-WEEK TREASURY YIELD (Multiple Approaches)
    # =====================================================
    
    # Approach A: Try to access treasury data through alternative routes
    try:
        # Method 1: Try main Bank Al-Maghrib markets page
        bkam_markets_urls = [
            "https://www.bkam.ma/Marches/Principaux-indicateurs",
            "https://www.bkam.ma/Marches",
            "https://www.bkam.ma/Statistiques/Indicateurs-monétaires-et-financiers"
        ]
        
        treasury_yield_found = False
        
        for url in bkam_markets_urls:
            if treasury_yield_found:
                break
                
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    # Search for 52-week patterns
                    patterns = [
                        r'52.*?semaines?.*?(\d+[,.]?\d*)%',
                        r'(\d+[,.]?\d*)%.*?52.*?semaines?',
                        r'bons.*?tr[eé]sor.*?52.*?(\d+[,.]?\d*)',
                        r'treasury.*?52.*?week.*?(\d+[,.]?\d*)',
                        r'52w.*?(\d+[,.]?\d*)%'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            rate = float(match.replace(',', '.'))
                            if 0.1 <= rate <= 15:  # Reasonable treasury yield range
                                live_data['yield_52w'] = rate
                                live_data['sources']['yield_52w'] = f'Bank Al-Maghrib Live (Markets Page)'
                                treasury_yield_found = True
                                break
                        if treasury_yield_found:
                            break
            except:
                continue
        
        # Method 2: Estimate from policy rate if not found
        if not treasury_yield_found:
            # Treasury yields typically trade at a small spread to policy rate
            policy_spread = 0.10  # 10 basis points typical spread
            live_data['yield_52w'] = live_data['policy_rate'] + policy_spread
            live_data['sources']['yield_52w'] = 'Estimated from Policy Rate (+10bps)'
        
        live_data['fetch_attempts']['yield_52w'] = "Multiple approaches attempted"
        
    except Exception as e:
        live_data['fetch_attempts']['yield_52w'] = f"Error: {str(e)[:50]}..."
        live_data['sources']['yield_52w'] = 'Fallback (Estimation)'
    
    # =====================================================
    # 3. FETCH HCP INFLATION DATA
    # =====================================================
    try:
        hcp_inflation_urls = [
            "https://www.hcp.ma/Actualite-Indices-des-prix-a-la-consommation-IPC_r349.html",
            "https://www.hcp.ma/Economie_r327.html"
        ]
        
        inflation_found = False
        
        for url in hcp_inflation_urls:
            if inflation_found:
                break
                
            try:
                response = requests.get(url, headers=headers, timeout=15)
                live_data['fetch_attempts']['inflation'] = f"Status: {response.status_code}"
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    # Look for recent inflation patterns
                    patterns = [
                        r'inflation.*?sous[- ]?jacente.*?(\d+[,.]?\d*)%.*?ann[eé]e',
                        r'indicateur.*?inflation.*?(\d+[,.]?\d*)%.*?ann[eé]e',
                        r'hausse.*?(\d+[,.]?\d*)%.*?ann[eé]e',
                        r'(\d+[,.]?\d*)%.*?ann[eé]e.*?inflation',
                        r'inflation.*?(\d+[,.]?\d*)%'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            rate = float(match.replace(',', '.'))
                            if 0 <= rate <= 15:  # Reasonable inflation range
                                live_data['inflation'] = rate
                                live_data['sources']['inflation'] = 'HCP Live'
                                inflation_found = True
                                break
                        if inflation_found:
                            break
            except:
                continue
        
    except Exception as e:
        live_data['fetch_attempts']['inflation'] = f"Error: {str(e)[:50]}..."
        live_data['sources']['inflation'] = 'Fallback (Fetch failed)'
    
    # =====================================================
    # 4. FETCH HCP GDP GROWTH DATA
    # =====================================================
    try:
        hcp_gdp_urls = [
            "https://www.hcp.ma/Conjoncture-et-prevision-economique_r328.html",
            "https://www.hcp.ma/Economie_r327.html"
        ]
        
        gdp_found = False
        
        for url in hcp_gdp_urls:
            if gdp_found:
                break
                
            try:
                response = requests.get(url, headers=headers, timeout=15)
                live_data['fetch_attempts']['gdp_growth'] = f"Status: {response.status_code}"
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text().lower()
                    
                    # Look for GDP growth patterns
                    patterns = [
                        r'croissance.*?[eé]conomique.*?(\d+[,.]?\d*)%',
                        r'pib.*?(\d+[,.]?\d*)%',
                        r'progression.*?(\d+[,.]?\d*)%.*?trimestre',
                        r'(\d+[,.]?\d*)%.*?premier.*?trimestre.*?2025',
                        r'am[eé]lioration.*?(\d+[,.]?\d*)%'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            rate = float(match.replace(',', '.'))
                            if 0 <= rate <= 15:  # Reasonable GDP growth range
                                live_data['gdp_growth'] = rate
                                live_data['sources']['gdp_growth'] = 'HCP Live'
                                gdp_found = True
                                break
                        if gdp_found:
                            break
            except:
                continue
        
    except Exception as e:
        live_data['fetch_attempts']['gdp_growth'] = f"Error: {str(e)[:50]}..."
        live_data['sources']['gdp_growth'] = 'Fallback (Fetch failed)'
    
    # =====================================================
    # 5. SET DEFAULT SOURCES AND FINAL PROCESSING
    # =====================================================
    
    # Set default sources if not already set
    if 'policy_rate' not in live_data['sources']:
        live_data['sources']['policy_rate'] = 'Manual Fallback'
    if 'yield_52w' not in live_data['sources']:
        live_data['sources']['yield_52w'] = 'Manual Fallback'
    if 'inflation' not in live_data['sources']:
        live_data['sources']['inflation'] = 'Manual Fallback'
    if 'gdp_growth' not in live_data['sources']:
        live_data['sources']['gdp_growth'] = 'Manual Fallback'
    
    # Final data validation
    live_data['policy_rate'] = max(0.1, min(10.0, live_data['policy_rate']))
    live_data['yield_52w'] = max(0.1, min(15.0, live_data['yield_52w']))
    live_data['inflation'] = max(0.0, min(20.0, live_data['inflation']))
    live_data['gdp_growth'] = max(-10.0, min(20.0, live_data['gdp_growth']))
    
    live_data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return live_data

def display_enhanced_data_panel(live_data):
    """Enhanced display panel with detailed fetch information"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Données en Temps Réel")
    
    # Success rate indicator
    live_sources = sum(1 for source in live_data['sources'].values() if 'Live' in source)
    total_sources = len([k for k in live_data['sources'].keys() if k != 'last_updated'])
    success_rate = (live_sources / total_sources) * 100
    
    if success_rate >= 75:
        st.sidebar.success(f"🟢 {live_sources}/{total_sources} sources en direct ({success_rate:.0f}%)")
    elif success_rate >= 50:
        st.sidebar.warning(f"🟡 {live_sources}/{total_sources} sources en direct ({success_rate:.0f}%)")
    elif success_rate >= 25:
        st.sidebar.warning(f"🟠 {live_sources}/{total_sources} sources en direct ({success_rate:.0f}%)")
    else:
        st.sidebar.error(f"🔴 {live_sources}/{total_sources} sources en direct ({success_rate:.0f}%)")
    
    # Current values with source indicators
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # Policy rate with indicator
        source_indicator = "🟢" if 'Live' in live_data['sources']['policy_rate'] else "🔴"
        st.metric(
            f"{source_indicator} Taux Directeur", 
            f"{live_data['policy_rate']:.2f}%",
            help=f"Source: {live_data['sources']['policy_rate']}"
        )
        
        # 52-week yield with indicator
        source_indicator = "🟢" if 'Live' in live_data['sources']['yield_52w'] else "🟡" if 'Estimated' in live_data['sources']['yield_52w'] else "🔴"
        st.metric(
            f"{source_indicator} Rendement 52s", 
            f"{live_data['yield_52w']:.2f}%",
            help=f"Source: {live_data['sources']['yield_52w']}"
        )
    
    with col2:
        # Inflation with indicator
        source_indicator = "🟢" if 'Live' in live_data['sources']['inflation'] else "🔴"
        st.metric(
            f"{source_indicator} Inflation", 
            f"{live_data['inflation']:.2f}%",
            help=f"Source: {live_data['sources']['inflation']}"
        )
        
        # GDP with indicator
        source_indicator = "🟢" if 'Live' in live_data['sources']['gdp_growth'] else "🔴"
        st.metric(
            f"{source_indicator} Croissance PIB", 
            f"{live_data['gdp_growth']:.2f}%",
            help=f"Source: {live_data['sources']['gdp_growth']}"
        )
    
    # Last update and manual refresh
    st.sidebar.info(f"🕐 Mis à jour: {live_data['last_updated']}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Actualiser"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Détails"):
            st.session_state.show_fetch_details = not st.session_state.get('show_fetch_details', False)
    
    # Detailed fetch information
    if st.session_state.get('show_fetch_details', False):
        with st.sidebar.expander("🔍 Détails des Tentatives", expanded=True):
            st.markdown("**Sources de Données:**")
            for key, source in live_data['sources'].items():
                indicator_name = {
                    'policy_rate': 'Taux Directeur',
                    'yield_52w': 'Rendement 52s',
                    'inflation': 'Inflation', 
                    'gdp_growth': 'Croissance PIB'
                }
                
                if key in indicator_name:
                    status = "✅" if 'Live' in source else "⚠️" if 'Estimated' in source else "❌"
                    st.write(f"{status} **{indicator_name[key]}:** {source}")
            
            if 'fetch_attempts' in live_data:
                st.markdown("**Tentatives de Récupération:**")
                for key, attempt in live_data['fetch_attempts'].items():
                    st.write(f"• {key}: {attempt}")

# Usage in your main function:
def main_with_enhanced_data():
    st.markdown("""
    <div class="main-header">
        <h1>🇲🇦 SOFAC - Modèle de Prédiction des Rendements 52-Semaines</h1>
        <p>Système d'aide à la décision avec données automatiques Bank Al-Maghrib & HCP</p>
        <p><strong>Rendement 52s:</strong> Automatique | <strong>Mise à jour:</strong> Horaire</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch enhanced live data (including 52-week yields)
    live_data = fetch_enhanced_moroccan_data()
    
    with st.sidebar:
        st.header("📊 Informations du Modèle")
        
        # Display enhanced data panel
        display_enhanced_data_panel(live_data)
        
        # Rest of your existing sidebar code...
    
    # Your existing tabs and content...

# Add this notice for the 52-week yield
def display_yield_notice():
    if st.sidebar.button("ℹ️ À propos du Rendement 52s"):
        st.sidebar.info("""
        **Rendement 52-semaines:**
        
        Les données sont extraites de:
        - Bank Al-Maghrib (pages marchés)
        - Estimation basée sur le taux directeur
        - Spread typique: +10 à +50 points de base
        
        **Note:** Certaines pages BAM limitent l'accès automatique. En cas d'échec, nous utilisons une estimation fiable basée sur la relation historique avec le taux directeur.
        """)
