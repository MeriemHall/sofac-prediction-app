fig_detail.add_trace(
            go.Histogram(
                x=pred_scenario['Rendement_Predit'],
                nbinsx=30,
                name='Distribution',
                marker_color=colors[scenario_selectionne],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        pred_scenario['Mois'] = pd.to_datetime(pred_scenario['Date']).dt.to_period('M')
        monthly_avg = pred_scenario.groupby('Mois')['Rendement_Predit'].mean()
        
        fig_detail.add_trace(
            go.Scatter(
                x=[str(m) for m in monthly_avg.index],
                y=monthly_avg.values,
                mode='lines+markers',
                name='Moyenne Mensuelle',
                line=dict(color=colors[scenario_selectionne], width=3),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        fig_detail.update_layout(
            height=800,
            title_text=f"Analyse Détaillée - Scénario {scenario_selectionne}",
            template="plotly_white",
            showlegend=True
        )
        
        fig_detail.update_xaxes(title_text="Date", row=1, col=1)
        fig_detail.update_yaxes(title_text="Rendement (%)", row=1, col=1)
        fig_detail.update_xaxes(title_text="Rendement (%)", row=2, col=1)
        fig_detail.update_yaxes(title_text="Fréquence", row=2, col=1)
        fig_detail.update_xaxes(title_text="Mois", row=2, col=2)
        fig_detail.update_yaxes(title_text="Rendement Moyen (%)", row=2, col=2)
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        st.subheader("Analyse Statistique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Statistiques Descriptives")
            
            stats_df = pd.DataFrame({
                'Métrique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum', 'Q1', 'Q3'],
                'Valeur (%)': [
                    f"{pred_scenario['Rendement_Predit'].mean():.3f}",
                    f"{pred_scenario['Rendement_Predit'].median():.3f}",
                    f"{pred_scenario['Rendement_Predit'].std():.3f}",
                    f"{pred_scenario['Rendement_Predit'].min():.3f}",
                    f"{pred_scenario['Rendement_Predit'].max():.3f}",
                    f"{pred_scenario['Rendement_Predit'].quantile(0.25):.3f}",
                    f"{pred_scenario['Rendement_Predit'].quantile(0.75):.3f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Analyse de Risque")
            
            var_95 = np.percentile(pred_scenario['Rendement_Predit'], 5)
            var_99 = np.percentile(pred_scenario['Rendement_Predit'], 1)
            prob_above_baseline = (pred_scenario['Rendement_Predit'] > last_historical_yield).mean() * 100
            
            risk_df = pd.DataFrame({
                'Métrique de Risque': [
                    'VaR 95% (pire cas 5%)',
                    'VaR 99% (pire cas 1%)',
                    'Prob. > Référence',
                    'Volatilité Annualisée',
                    'Ratio Sharpe (estimé)'
                ],
                'Valeur': [
                    f"{var_95:.3f}%",
                    f"{var_99:.3f}%",
                    f"{prob_above_baseline:.1f}%",
                    f"{pred_scenario['Rendement_Predit'].std() * np.sqrt(252):.3f}%",
                    f"{(pred_scenario['Rendement_Predit'].mean() - last_historical_yield) / pred_scenario['Rendement_Predit'].std():.2f}"
                ]
            })
            st.dataframe(risk_df, use_container_width=True)
        
        st.subheader("Export des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Télécharger Prédictions Détaillées", use_container_width=True):
                pred_export = pred_scenario.copy()
                pred_export['Scenario'] = scenario_selectionne
                pred_export['Baseline_Reference'] = last_historical_yield
                pred_export['Ecart_vs_Baseline'] = pred_export['Rendement_Predit'] - last_historical_yield
                
                csv = pred_export.to_csv(index=False)
                st.download_button(
                    label="💾 Télécharger CSV",
                    data=csv,
                    file_name=f"sofac_predictions_{scenario_selectionne.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Télécharger Analyse Statistique", use_container_width=True):
                analysis_data = {
                    'Scenario': [scenario_selectionne],
                    'Periode_Analyse': [f"{pred_scenario['Date'].iloc[0]} à {pred_scenario['Date'].iloc[-1]}"],
                    'Rendement_Moyen_Pct': [pred_scenario['Rendement_Predit'].mean()],
                    'Ecart_vs_Baseline_Pct': [pred_scenario['Rendement_Predit'].mean() - last_historical_yield],
                    'Volatilite_Pct': [pred_scenario['Rendement_Predit'].std()],
                    'VaR_95_Pct': [var_95],
                    'Probabilite_Hausse_Pct': [prob_above_baseline],
                    'Recommandation': [st.session_state.recommandations[scenario_selectionne]['recommandation']],
                    'Niveau_Risque': [st.session_state.recommandations[scenario_selectionne]['niveau_risque']]
                }
                
                analysis_df = pd.DataFrame(analysis_data)
                csv_analysis = analysis_df.to_csv(index=False)
                st.download_button(
                    label="💾 Télécharger Analyse",
                    data=csv_analysis,
                    file_name=f"sofac_analysis_{scenario_selectionne.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    with tab3:
        st.header("Recommandations Stratégiques Complètes")
        
        liste_recommandations = [rec['recommandation'] for rec in st.session_state.recommandations.values()]
        
        if liste_recommandations.count('TAUX VARIABLE') >= 2:
            strategie_globale = "TAUX VARIABLE"
            raison_globale = f"Majorité des scénarios anticipent une baisse des taux depuis la référence juin 2025 ({last_historical_yield:.2f}%)"
            couleur_globale = "#16a34a"
            icone_globale = "📉"
        elif liste_recommandations.count('TAUX FIXE') >= 2:
            strategie_globale = "TAUX FIXE"
            raison_globale = f"Majorité des scénarios anticipent une hausse des taux depuis la référence juin 2025 ({last_historical_yield:.2f}%)"
            couleur_globale = "#dc2626"
            icone_globale = "📈"
        else:
            strategie_globale = "STRATÉGIE MIXTE"
            raison_globale = f"Signaux contrastés depuis la référence juin 2025 ({last_historical_yield:.2f}%) - diversification recommandée"
            couleur_globale = "#f59e0b"
            icone_globale = "⚖️"
        
        quality_score = sum(1 for source in live_data['sources'].values() if 'Live' in source)
        quality_text = "Surveillance économique en temps réel active" if quality_score >= 2 else "Surveillance économique limitée"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {couleur_globale} 0%, {couleur_globale}CC 100%); color: white; padding: 2rem; border-radius: 16px; margin: 2rem 0; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
            <h2 style="margin: 0 0 1rem 0; font-size: 2rem;">{icone_globale} STRATÉGIE GLOBALE SOFAC</h2>
            <h3 style="margin: 0 0 1rem 0; font-size: 2.5rem; font-weight: 700;">{strategie_globale}</h3>
            <p style="margin: 0 0 1rem 0; font-size: 1.2rem; opacity: 0.9;">{raison_globale}</p>
            <p style="margin: 0; font-size: 1rem; opacity: 0.8;">{quality_text} • Confiance du modèle: {st.session_state.r2:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📋 Analyse Comparative des Scénarios")
        
        for nom_scenario, rec in st.session_state.recommandations.items():
            with st.expander(f"📊 Analyse Détaillée - Scénario {nom_scenario}", expanded=True):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if rec['recommandation'] == 'TAUX VARIABLE':
                        rec_color = "#16a34a"
                        rec_icon = "📉"
                    elif rec['recommandation'] == 'TAUX FIXE':
                        rec_color = "#dc2626"
                        rec_icon = "📈"
                    else:
                        rec_color = "#f59e0b"
                        rec_icon = "⚖️"
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {rec_color}; padding-left: 1rem; margin: 1rem 0;">
                        <h4 style="color: {rec_color}; margin-bottom: 1rem;">{rec_icon} {rec['recommandation']}</h4>
                        <p style="margin-bottom: 1rem; font-size: 1.1rem;"><strong>Justification:</strong> {rec['raison']}</p>
                        
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0;">
                            <div>
                                <p><strong>📊 Métriques Financières:</strong></p>
                                <ul>
                                    <li>Rendement de référence: {last_historical_yield:.2f}%</li>
                                    <li>Rendement moyen prédit: {rec['rendement_futur_moyen']:.2f}%</li>
                                    <li>Changement attendu: {rec['changement_rendement']:+.2f}%</li>
                                    <li>Volatilité: {rec['volatilite']:.2f}%</li>
                                </ul>
                            </div>
                            <div>
                                <p><strong>⚠️ Évaluation des Risques:</strong></p>
                                <ul>
                                    <li>Niveau de risque: {rec['niveau_risque']}</li>
                                    <li>Horizon de confiance: 18 mois</li>
                                    <li>Facteurs d'incertitude: Politique monétaire</li>
                                    <li>Révision recommandée: Trimestrielle</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    pred_df = st.session_state.predictions[nom_scenario]
                    sample_mini = pred_df[::20]
                    
                    fig_mini = go.Figure()
                    
                    fig_mini.add_hline(
                        y=last_historical_yield, 
                        line_dash="dash", 
                        line_color="#6b7280",
                        annotation_text=f"Référence: {last_historical_yield:.2f}%"
                    )
                    
                    colors = {'Conservateur': '#ef4444', 'Cas_de_Base': '#3b82f6', 'Optimiste': '#10b981'}
                    fig_mini.add_trace(
                        go.Scatter(
                            x=sample_mini['Date'],
                            y=sample_mini['Rendement_Predit'],
                            mode='lines+markers',
                            name=nom_scenario,
                            line=dict(color=colors[nom_scenario], width=3),
                            marker=dict(size=6)
                        )
                    )
                    
                    fig_mini.update_layout(
                        height=250,
                        showlegend=False,
                        template="plotly_white",
                        margin=dict(l=20, r=20, t=20, b=20),
                        title=f"Évolution Prévue - {nom_scenario}",
                        xaxis_title="Date",
                        yaxis_title="Rendement (%)"
                    )
                    
                    st.plotly_chart(fig_mini, use_container_width=True)
        
        st.subheader("💰 Simulateur d'Impact Financier")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            montant_emprunt = st.slider(
                "Montant d'emprunt (millions MAD):",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                help="Montant total de l'emprunt à analyser"
            )
        
        with col2:
            duree_emprunt = st.slider(
                "Durée d'emprunt (années):",
                min_value=1,
                max_value=15,
                value=5,
                step=1,
                help="Durée de l'emprunt en années"
            )
        
        with col3:
            scenario_calcul = st.selectbox(
                "Scénario pour le calcul:",
                options=['Cas_de_Base', 'Conservateur', 'Optimiste'],
                help="Scénario économique pour l'estimation"
            )
        
        changement_scenario = st.session_state.recommandations[scenario_calcul]['changement_rendement']
        impact_annuel = changement_scenario * montant_emprunt * 1_000_000 / 100
        impact_total = impact_annuel * duree_emprunt
        
        col1, col2 = st.columns(2)
        
        with col1:
            if abs(changement_scenario) > 0.2:
                if changement_scenario < 0:
                    st.success(f"""
                    ### 💰 Économies Potentielles avec TAUX VARIABLE
                    
                    **📊 Détails du Calcul:**
                    - **Montant:** {montant_emprunt:,} millions MAD
                    - **Durée:** {duree_emprunt} années
                    - **Scénario:** {scenario_calcul}
                    - **Variation attendue:** {changement_scenario:.2f}%
                    
                    **💵 Impact Financier:**
                    - **Économies annuelles:** {abs(impact_annuel):,.0f} MAD
                    - **Économies totales:** {abs(impact_total):,.0f} MAD
                    
                    **📈 Basé sur:** Baisse attendue de {abs(changement_scenario):.2f}% vs référence juin 2025 ({last_historical_yield:.2f}%)
                    """)
                else:
                    st.warning(f"""
                    ### 💰 Coûts Évités avec TAUX FIXE
                    
                    **📊 Détails du Calcul:**
                    - **Montant:** {montant_emprunt:,} millions MAD
                    - **Durée:** {duree_emprunt} années
                    - **Scénario:** {scenario_calcul}
                    - **Variation attendue:** +{changement_scenario:.2f}%
                    
                    **💵 Impact Financier:**
                    - **Surcoûts évités/an:** {impact_annuel:,.0f} MAD
                    - **Surcoûts évités totaux:** {impact_total:,.0f} MAD
                    
                    **📈 Basé sur:** Hausse attendue de {changement_scenario:.2f}% vs référence juin 2025 ({last_historical_yield:.2f}%)
                    """)
            else:
                st.info(f"""
                ### 💰 Impact Financier Limité
                
                **📊 Détails du Calcul:**
                - **Montant:** {montant_emprunt:,} millions MAD
                - **Durée:** {duree_emprunt} années
                - **Scénario:** {scenario_calcul}
                - **Variation attendue:** {changement_scenario:+.2f}%
                
                **💵 Impact Financier:**
                - **Variation annuelle:** ±{abs(impact_annuel):,.0f} MAD
                - **Impact total:** ±{abs(impact_total):,.0f} MAD
                
                **⚖️ Recommandation:** Stratégie flexible selon les besoins de liquidité
                """)
        
        with col2:
            scenarios_impact = []
            for scenario in ['Conservateur', 'Cas_de_Base', 'Optimiste']:
                change = st.session_state.recommandations[scenario]['changement_rendement']
                annual_impact = change * montant_emprunt * 1_000_000 / 100
                scenarios_impact.append({
                    'Scénario': scenario,
                    'Impact Annuel (MAD)': annual_impact,
                    'Impact Total (MAD)': annual_impact * duree_emprunt
                })
            
            impact_df = pd.DataFrame(scenarios_impact)
            
            fig_impact = go.Figure()
            
            colors = ['#ef4444', '#3b82f6', '#10b981']
            fig_impact.add_trace(
                go.Bar(
                    x=impact_df['Scénario'],
                    y=impact_df['Impact Annuel (MAD)'],
                    marker_color=colors,
                    text=[f"{val:,.0f}" for val in impact_df['Impact Annuel (MAD)']],
                    textposition='auto',
                    name='Impact Annuel'
                )
            )
            
            fig_impact.update_layout(
                title=f"Impact Financier par Scénario<br>({montant_emprunt}M MAD sur {duree_emprunt} ans)",
                xaxis_title="Scénario",
                yaxis_title="Impact Annuel (MAD)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig_impact, use_container_width=True)
    
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    live_sources_count = sum(1 for source in live_data['sources'].values() if 'Live' in source)
    
    st.markdown(f"""
    <div class="footer">
        <p><strong>SOFAC</strong> - Système de Prédiction des Rendements 52-Semaines</p>
        <p>
            <strong>Sources de données:</strong> Bank Al-Maghrib, HCP • 
            <strong>Surveillance temps réel:</strong> {live_sources_count}/4 sources directes • 
            <strong>Modèle:</strong> Régression Linéaire Multiple (R² = {st.session_state.r2:.1%})
        </p>
        <p>
            <strong>Horizon de prédiction:</strong> Juillet 2025 - Décembre 2026 • 
            <strong>Référence historique:</strong> Juin 2025 ({last_historical_yield:.2f}%) • 
            <strong>Dernière mise à jour:</strong> {current_time}
        </p>
        <p><em>Les prédictions sont basées sur des modèles statistiques et ne constituent pas des conseils financiers. 
        Consultez vos conseillers financiers pour toute décision d'investissement.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() mois_depuis_debut / 12)
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
        
        if len(rendements_bruts) > 0:
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
        
        ajustements = []
        for i, ligne in scenario_df.iterrows():
            ajustement = 0
            
            if nom_scenario == 'Conservateur':
                ajustement += 0.1
            elif nom_scenario == 'Optimiste':
                ajustement -= 0.05
            
            jours_ahead = ligne['Jours_Ahead']
            incertitude = (jours_ahead / 365) * 0.05
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
    rendement_actuel = 1.75
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

def create_gauge_chart(value, min_val, max_val, title, color_ranges):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': color_ranges,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_trend_chart(data, title):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6, color='#1e40af'),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Période",
        yaxis_title="Valeur (%)",
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    st.markdown("""
    <div class="main-header">
        <h1>SOFAC</h1>
        <div class="subtitle">Système de Prédiction des Rendements 52-Semaines</div>
        <div class="update-info">Données Bank Al-Maghrib & HCP • Mise à jour: {} • Prochaine: {}</div>
    </div>
    """.format(
        datetime.now().strftime('%H:%M'), 
        (datetime.now() + timedelta(hours=1)).strftime('%H:%M')
    ), unsafe_allow_html=True)
    
    with st.spinner("↻ Récupération des données en temps réel..."):
        live_data = fetch_live_moroccan_data()
    
    if 'data_loaded' not in st.session_state:
        with st.spinner("⚙ Calibration du modèle..."):
            st.session_state.df_mensuel = create_monthly_dataset()
            st.session_state.modele, st.session_state.r2, st.session_state.mae, st.session_state.rmse, st.session_state.mae_vc = train_prediction_model(st.session_state.df_mensuel)
            st.session_state.scenarios = create_economic_scenarios()
            st.session_state.predictions = generate_predictions(st.session_state.scenarios, st.session_state.modele, st.session_state.mae)
            st.session_state.recommandations = generate_recommendations(st.session_state.predictions)
            st.session_state.data_loaded = True
    
    last_historical_yield = st.session_state.df_mensuel.iloc[-1]['Rendement_52s']
    
    with st.sidebar:
        st.header("Informations du Modèle")
        display_live_data_panel(live_data, last_historical_yield)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Prédiction du Jour")
        
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        today_display = today.strftime('%d/%m/%Y')
        
        cas_base_predictions = st.session_state.predictions['Cas_de_Base']
        
        today_prediction = None
        closest_prediction = None
        
        for _, row in cas_base_predictions.iterrows():
            pred_date = row['Date']
            if pred_date == today_str:
                today_prediction = row['Rendement_Predit']
                break
            elif pred_date > today_str and closest_prediction is None:
                closest_prediction = row['Rendement_Predit']
        
        if today_prediction is not None:
            st.sidebar.success(f"**{today_display}**")
            st.sidebar.metric(
                "Rendement Prédit",
                f"{today_prediction:.2f}%",
                delta=f"{(today_prediction - last_historical_yield):+.2f}%"
            )
        elif closest_prediction is not None:
            st.sidebar.warning(f"**{today_display}**")
            st.sidebar.metric(
                "Prédiction Prochaine",
                f"{closest_prediction:.2f}%",
                delta=f"{(closest_prediction - last_historical_yield):+.2f}%"
            )
        else:
            st.sidebar.info(f"**{today_display}**")
            st.sidebar.write("**Prédiction:** En cours...")
        
        st.success("✓ Modèle calibré")
        
        st.subheader("Performance Modèle")
        st.metric("R² Score", f"{st.session_state.r2:.1%}")
        st.metric("Précision", f"±{st.session_state.mae:.2f}%")
        st.metric("Validation", f"±{st.session_state.mae_vc:.2f}%")
        
        st.info("↻ Surveillance économique active")
    
    tab1, tab2, tab3 = st.tabs(["📊 Briefing Exécutif", "📈 Analyse Détaillée", "💼 Recommandations"])
    
    with tab1:
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')
        today_display = today.strftime('%d/%m/%Y')
        
        cas_de_base = st.session_state.predictions['Cas_de_Base']
        current_prediction = None
        trend_direction = None
        trend_strength = "STABLE"
        
        for i, row in cas_de_base.iterrows():
            pred_date = row['Date']
            if pred_date >= today_str:
                current_prediction = row['Rendement_Predit']
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
        
        recommandation_base = st.session_state.recommandations['Cas_de_Base']
        evolution_vs_baseline = current_prediction - last_historical_yield
        changement_global = recommandation_base['changement_rendement']
        
        if evolution_vs_baseline > 0.4 or changement_global > 0.4:
            market_status = "ALERTE ROUGE"
            status_color = "#dc2626"
            urgency = "IMMÉDIATE"
            action = "BLOQUER TOUS LES TAUX MAINTENANT"
        elif evolution_vs_baseline > 0.1 or changement_global > 0.1:
            market_status = "ATTENTION REQUISE"
            status_color = "#f59e0b"
            urgency = "ÉLEVÉE"
            action = "PRÉPARER STRATÉGIE DÉFENSIVE"
        elif evolution_vs_baseline < -0.4 or changement_global < -0.4:
            market_status = "OPPORTUNITÉ MAJEURE"
            status_color = "#16a34a"
            urgency = "MODÉRÉE"
            action = "MAXIMISER TAUX VARIABLES"
        elif evolution_vs_baseline < -0.1 or changement_global < -0.1:
            market_status = "CONTEXTE FAVORABLE"
            status_color = "#059669"
            urgency = "NORMALE"
            action = "CONSIDÉRER TAUX VARIABLES"
        else:
            market_status = "SITUATION STABLE"
            status_color = "#6b7280"
            urgency = "FAIBLE"
            action = "MAINTENIR STRATÉGIE ACTUELLE"
        
        volatilite_globale = cas_de_base['Rendement_Predit'].std()
        data_quality = sum(1 for source in live_data['sources'].values() if 'Live' in source)
        impact_10m = abs(changement_global) * 10
        
        st.markdown(f"""
        <div class="briefing-container">
            <div class="briefing-header">
                <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">BRIEFING EXÉCUTIF SOFAC</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">{today_display} • Mise à jour en temps réel</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="status-card" style="border-color: {status_color};">
                <div class="metric-label">RENDEMENT ACTUEL</div>
                <div class="metric-value" style="color: {status_color};">{current_prediction:.2f}%</div>
                <div class="metric-change" style="color: {status_color};">{evolution_vs_baseline:+.2f}% vs Juin</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            trend_color = "#dc2626" if trend_direction == "HAUSSE" else "#16a34a" if trend_direction == "BAISSE" else "#6b7280"
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-label">TENDANCE 30 JOURS</div>
                <div class="metric-value" style="color: {trend_color}; font-size: 1.8rem;">{trend_direction}</div>
                <div class="metric-change" style="color: {trend_color};">{trend_strength}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            urgency_color = "#dc2626" if urgency == "IMMÉDIATE" else "#f59e0b" if urgency == "ÉLEVÉE" else "#16a34a"
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-label">NIVEAU D'URGENCE</div>
                <div class="metric-value" style="color: {urgency_color}; font-size: 1.8rem;">{urgency}</div>
                <div class="metric-change" style="color: {urgency_color};">Action requise</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            impact_color = "#dc2626" if impact_10m > 50 else "#f59e0b" if impact_10m > 20 else "#16a34a"
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-label">IMPACT FINANCIER</div>
                <div class="metric-value" style="color: {impact_color};">{impact_10m:.0f}K</div>
                <div class="metric-change" style="color: {impact_color};">MAD/an (10M)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h2 style="margin: 0 0 1rem 0; font-size: 1.5rem;">🎯 RECOMMANDATION IMMÉDIATE</h2>
            <h3 style="margin: 0 0 1rem 0; font-size: 2rem; font-weight: 700;">{action}</h3>
            <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                Statut: {market_status} • Stratégie: {recommandation_base['recommandation']} • Horizon: {"Immédiat" if urgency in ["IMMÉDIATE", "ÉLEVÉE"] else "1-3 mois"}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Analyse Détaillée")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="action-item">
                <h4 style="color: #1e3a8a; margin-bottom: 1rem;">💹 SITUATION FINANCIÈRE</h4>
                <p><strong>Rendement de référence (Juin 2025):</strong> {last_historical_yield:.2f}%</p>
                <p><strong>Évolution prévue (18 mois):</strong> {changement_global:+.2f}%</p>
                <p><strong>Volatilité attendue:</strong> {volatilite_globale:.2f}%</p>
                <p><strong>Niveau de risque:</strong> {recommandation_base['niveau_risque']}</p>
                <p><strong>Confiance du modèle:</strong> {st.session_state.r2:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            gauge_colors = [
                {'range': [0, 1], 'color': "lightgreen"},
                {'range': [1, 2], 'color': "yellow"},
                {'range': [2, 3], 'color': "orange"},
                {'range': [3, 5], 'color': "red"}
            ]
            
            fig_gauge = create_gauge_chart(
                current_prediction, 0, 5, 
                "Rendement Actuel (%)", 
                gauge_colors
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="action-item">
                <h4 style="color: #1e3a8a; margin-bottom: 1rem;">🎯 ACTIONS RECOMMANDÉES</h4>
                <p><strong>Stratégie principale:</strong> {recommandation_base['recommandation']}</p>
                <p><strong>Justification:</strong> {recommandation_base['raison'][:100]}...</p>
                <p><strong>Impact estimé:</strong> {impact_10m:.0f}K MAD/an sur 10M MAD</p>
                <p><strong>Qualité des données:</strong> {data_quality}/4 sources directes</p>
                <p><strong>Prochaine révision:</strong> {(datetime.now() + timedelta(hours=1)).strftime('%H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            recent_predictions = cas_de_base.head(30)['Rendement_Predit']
            fig_trend = create_trend_chart(recent_predictions, "Tendance 30 Prochains Jours")
            st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("### 📊 Comparaison des Scénarios")
        
        scenario_cols = st.columns(3)
        scenarios_data = [
            ('Conservateur', '#ef4444', st.session_state.recommandations['Conservateur']),
            ('Cas de Base', '#3b82f6', st.session_state.recommandations['Cas_de_Base']),
            ('Optimiste', '#10b981', st.session_state.recommandations['Optimiste'])
        ]
        
        for i, (scenario, color, rec) in enumerate(scenarios_data):
            with scenario_cols[i]:
                trend_class = "trend-up" if rec['changement_rendement'] > 0.1 else "trend-down" if rec['changement_rendement'] < -0.1 else "trend-stable"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: {color}; text-align: center; margin-bottom: 1rem;">{scenario}</h4>
                    <div style="text-align: center;">
                        <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin: 0.5rem 0;">
                            {rec['rendement_futur_moyen']:.2f}%
                        </div>
                        <div class="trend-indicator {trend_class}">
                            {rec['changement_rendement']:+.2f}%
                        </div>
                        <p style="margin: 1rem 0 0.5rem 0; font-weight: 600;">{rec['recommandation']}</p>
                        <p style="margin: 0; font-size: 0.9rem; color: #6b7280;">Risque: {rec['niveau_risque']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Évolution Historique et Projections")
        
        fig_overview = go.Figure()
        
        df_recent = st.session_state.df_mensuel.tail(12)
        fig_overview.add_trace(
            go.Scatter(
                x=df_recent['Date'],
                y=df_recent['Rendement_52s'],
                mode='lines+markers',
                name='Historique',
                line=dict(color='#1e3a8a', width=4),
                marker=dict(size=8, color='#1e3a8a')
            )
        )
        
        fig_overview.add_hline(
            y=last_historical_yield, 
            line_dash="dash", 
            line_color="#6b7280",
            annotation_text=f"Référence Juin 2025: {last_historical_yield:.2f}%"
        )
        
        colors = {'Conservateur': '#ef4444', 'Cas_de_Base': '#3b82f6', 'Optimiste': '#10b981'}
        for nom_scenario, pred_df in st.session_state.predictions.items():
            sample_data = pred_df[::15]
            fig_overview.add_trace(
                go.Scatter(
                    x=sample_data['Date'],
                    y=sample_data['Rendement_Predit'],
                    mode='lines',
                    name=f'Projection {nom_scenario}',
                    line=dict(color=colors[nom_scenario], width=3),
                    opacity=0.8
                )
            )
        
        fig_overview.update_layout(
            title="Évolution des Rendements: Historique et Projections SOFAC",
            xaxis_title="Période",
            yaxis_title="Rendement (%)",
            height=500,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
    
    with tab2:
        st.header("Analyse Détaillée des Prédictions")
        
        scenario_selectionne = st.selectbox(
            "Sélectionnez un scénario d'analyse:",
            options=['Cas_de_Base', 'Conservateur', 'Optimiste'],
            index=0,
            help="Choisissez le scénario économique pour l'analyse détaillée"
        )
        
        pred_scenario = st.session_state.predictions[scenario_selectionne]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_val = pred_scenario['Rendement_Predit'].mean()
            st.metric("Rendement Moyen", f"{avg_val:.2f}%")
            
        with col2:
            min_val = pred_scenario['Rendement_Predit'].min()
            st.metric("Minimum", f"{min_val:.2f}%")
            
        with col3:
            max_val = pred_scenario['Rendement_Predit'].max()
            st.metric("Maximum", f"{max_val:.2f}%")
            
        with col4:
            baseline_comparison = avg_val - last_historical_yield
            st.metric("Écart vs Référence", f"{baseline_comparison:+.2f}%")
        
        st.subheader(f"Analyse Complète - Scénario {scenario_selectionne}")
        
        fig_detail = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Prédictions Quotidiennes avec Intervalles de Confiance',
                'Distribution des Rendements',
                'Évolution Mensuelle',
                'Volatilité dans le Temps'
            ],
            specs=[[{"colspan": 2}, None], 
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        sample_data = pred_scenario[::3]
        
        fig_detail.add_trace(
            go.Scatter(
                x=list(sample_data['Date']) + list(sample_data['Date'][::-1]),
                y=list(sample_data['Borne_Sup_95']) + list(sample_data['Borne_Inf_95'][::-1]),
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalle 95%',
                showlegend=True
            ),
            row=1, col=1
        )
        
        colors = {'Conservateur': '#ef4444', 'Cas_de_Base': '#3b82f6', 'Optimiste': '#10b981'}
        fig_detail.add_trace(
            go.Scatter(
                x=sample_data['Date'],
                y=sample_data['Rendement_Predit'],
                mode='lines+markers',
                name='Prédiction',
                line=dict(color=colors[scenario_selectionne], width=3),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig_detail.add_hline(
            y=last_historical_yield, 
            line_dash="dash", 
            line_color="#6b7280",
            annotation_text=f"Référence: {last_historical_yield:.2f}%",
            row=1, col=1
        )
        
        fig_detail.add_trace(
            go.Histogram(import streamlit as st
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: #f8fafc;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 2rem;
        border-radius: 0;
        color: white;
        text-align: center;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header .subtitle {
        font-size: 1.2rem !important;
        margin: 0 0 0.5rem 0 !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500 !important;
    }
    
    .main-header .update-info {
        font-size: 0.9rem !important;
        margin: 0 !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 400 !important;
    }
    
    .briefing-container {
        background: white;
        border-radius: 16px;
        padding: 0;
        margin: 2rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    .briefing-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px 16px 0 0;
        text-align: center;
    }
    
    .status-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-change {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.3);
    }
    
    .action-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .action-item {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #3b82f6;
    }
    
    .css-1d391kg {
        background-color: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }
    
    .css-1d391kg h2 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
    }
    
    .css-1d391kg h3 {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .css-1d391kg p {
        font-size: 0.75rem !important;
        line-height: 1.4 !important;
        color: #6b7280 !important;
        margin: 0.25rem 0 !important;
    }
    
    .css-1d391kg .stMetric label {
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        color: #6b7280 !important;
    }
    
    .css-1d391kg .stMetric [data-testid="metric-value"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #111827 !important;
    }
    
    .css-1d391kg .stAlert {
        font-size: 0.7rem !important;
        padding: 0.5rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .data-status {
        background: #f0fdf4;
        border: 1px solid #22c55e;
        color: #16a34a;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    .data-warning {
        background: #fffbeb;
        border: 1px solid #f59e0b;
        color: #d97706;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.7rem;
        font-weight: 500;
    }
    
    .trend-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem;
    }
    
    .trend-up {
        background: #fee2e2;
        color: #dc2626;
    }
    
    .trend-down {
        background: #dcfce7;
        color: #16a34a;
    }
    
    .trend-stable {
        background: #f3f4f6;
        color: #6b7280;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 1rem 2rem;
        font-weight: 500;
        color: #6b7280;
        border: none;
        font-size: 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e5e7eb;
        color: #374151;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1e3a8a !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.4);
    }
    
    h1, h2, h3, h4 {
        color: #111827 !important;
        font-weight: 600 !important;
    }
    
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.75rem !important; }
    h3 { font-size: 1.5rem !important; }
    h4 { font-size: 1.25rem !important; }
    
    p {
        font-size: 1rem !important;
        line-height: 1.6 !important;
        color: #4b5563 !important;
    }
    
    .footer {
        background-color: #f8fafc;
        border-top: 2px solid #e5e7eb;
        padding: 2rem 0;
        margin-top: 3rem;
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
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
    
    spread = 0.15
    if live_data['policy_rate'] < 2.0:
        spread = 0.25
    elif live_data['policy_rate'] > 4.0:
        spread = 0.10
    
    live_data['yield_52w'] = live_data['policy_rate'] + spread
    live_data['sources']['yield_52w'] = f'Estimated from Policy Rate (+{spread*100:.0f}bps)'
    
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
    
    live_data['sources']['gdp_growth'] = 'Economic Estimation'
    
    live_data['policy_rate'] = max(0.1, min(10.0, live_data['policy_rate']))
    live_data['yield_52w'] = max(0.1, min(15.0, live_data['yield_52w']))
    live_data['inflation'] = max(0.0, min(25.0, live_data['inflation']))
    live_data['gdp_growth'] = max(-10.0, min(20.0, live_data['gdp_growth']))
    
    return live_data

def display_live_data_panel(live_data, last_historical_yield):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Données Temps Réel")
    
    live_sources = sum(1 for source in live_data['sources'].values() if 'Live' in source)
    total_sources = 4
    success_rate = (live_sources / total_sources) * 100
    
    if success_rate >= 50:
        st.sidebar.markdown('<div class="data-status">● Données partiellement directes</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="data-warning">○ Données principalement estimées</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown(f"**Sources directes:** {live_sources}/{total_sources} ({success_rate:.0f}%)")
    
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
    
    st.sidebar.info(f"⏰ Mise à jour: {live_data['last_updated']}")
    
    if st.sidebar.button("↻ Actualiser"):
        st.cache_data.clear()
        st.rerun()

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
            
            mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
            
            if nom_scenario == 'Conservateur':
                inflation_base = 1.4 + 0.5 * np.exp(-mois_depuis_debut / 18) + 0.2 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 - 0.5 * (mois_depuis_debut / 18) + 0.4 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            elif nom_scenario == 'Cas_de_Base':
                inflation_base = 1.4 + 0.3 * np.exp(-mois_depuis_debut / 12) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 12)
                pib_base = 3.8 - 0.2 * (mois_depuis_debut / 18) + 0.5 * np.sin(2 * np.pi * ((date.month - 1) // 3) / 4)
            else:
                inflation_base = 1.4 - 0.2 * (mois_depuis_debut / 18) + 0.1 * np.sin(2 * np.pi *
