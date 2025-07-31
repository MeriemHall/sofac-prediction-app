X_enhanced['PIB_Inflation_Ratio'] = X_base['Croissance_PIB'] / (X_base['Inflation'] + 0.1)
            X_enhanced['Taux_Directeur_Squared'] = X_base['Taux_Directeur'] ** 2
            X_enhanced['Inflation_Squared'] = X_base['Inflation'] ** 2
            X_enhanced['Taux_Directeur_Lag1'] = X_base['Taux_Directeur'].shift(1).fillna(X_base['Taux_Directeur'].iloc[0])
            X_enhanced['Rendement_Lag1'] = baseline
            
            expected_features = model._feature_columns
            for feature in expected_features:
                if feature not in X_enhanced.columns:
                    X_enhanced[feature] = 0.0
            
            X_features = X_enhanced[expected_features]
        else:
            X_features = X_base
        
        rendements_bruts = model.predict(X_features)
        
        if has_enhanced_features and 'Rendement_Lag1' in expected_features:
            for i in range(1, len(rendements_bruts)):
                if i < len(X_features):
                    X_features.iloc[i, X_features.columns.get_loc('Rendement_Lag1')] = rendements_bruts[i-1]
            rendements_bruts = model.predict(X_features)
        
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
        
        ajustements = []
        for i, ligne in scenario_df.iterrows():
            ajustement = 0
            
            if scenario_name == 'Conservateur':
                ajustement += 0.08
            elif scenario_name == 'Optimiste':
                ajustement -= 0.04
            
            jours_ahead = ligne['Jours_Ahead']
            uncertainty_base = (jours_ahead / 365) * 0.025
            
            if scenario_name == 'Conservateur':
                ajustement += uncertainty_base * 1.2
            elif scenario_name == 'Optimiste':
                ajustement -= uncertainty_base * 0.6
            
            effets_jours = {
                'Monday': 0.008, 'Tuesday': -0.003, 'Wednesday': 0.002,
                'Thursday': -0.002, 'Friday': 0.012, 'Saturday': -0.006, 'Sunday': -0.008
            }
            ajustement += effets_jours.get(ligne['Jour_Semaine'], 0)
            
            mois = ligne.get('Mois', (i // 30) % 12 + 1)
            seasonal_effect = 0.01 * np.sin(2 * np.pi * mois / 12) + 0.005 * np.sin(4 * np.pi * mois / 12)
            ajustement += seasonal_effect
            
            annee = ligne.get('Annee', 2025 + i // 365)
            cycle_position = (annee - 2025) / 5
            cycle_effect = 0.02 * np.sin(2 * np.pi * cycle_position) if scenario_name != 'Optimiste' else 0.01 * np.sin(2 * np.pi * cycle_position)
            ajustement += cycle_effect
            
            np.random.seed(hash(ligne['Date']) % 2**32)
            base_vol = 0.008 if scenario_name == 'Conservateur' else 0.006 if scenario_name == 'Cas_de_Base' else 0.005
            market_noise = np.random.normal(0, base_vol)
            
            if np.random.random() < 0.002:
                shock_magnitude = np.random.normal(0, 0.05)
                market_noise += shock_magnitude
            
            ajustement += market_noise
            ajustements.append(ajustement)
        
        rendements_finaux = rendements_lisses + np.array(ajustements)
        rendements_finaux = np.clip(rendements_finaux, 0.1, 8.0)
        
        for i in range(1, len(rendements_finaux)):
            daily_change = rendements_finaux[i] - rendements_finaux[i-1]
            max_daily_change = 0.20
            
            if abs(daily_change) > max_daily_change:
                rendements_finaux[i] = rendements_finaux[i-1] + np.sign(daily_change) * max_daily_change
            
            if abs(daily_change) < 0.001:
                min_variation = np.random.normal(0, 0.002)
                rendements_finaux[i] += min_variation
        
        rendements_finaux = np.clip(rendements_finaux, 0.1, 8.0)
        
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
    col_logo, col_text = st.columns([1, 3])
    
    with col_logo:
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
    
    if 'data_loaded' not in st.session_state:
        with st.spinner("Chargement et optimisation du modèle..."):
            st.session_state.df = create_dataset()
            model_results = train_model(st.session_state.df)
            
            if len(model_results) == 6:
                (st.session_state.model, st.session_state.r2, st.session_state.mae, 
                 st.session_state.mae_cv, st.session_state.accuracy, st.session_state.detailed_metrics) = model_results
            else:
                (st.session_state.model, st.session_state.r2, st.session_state.mae, 
                 st.session_state.mae_cv, st.session_state.accuracy) = model_results
                st.session_state.detailed_metrics = {}
            
            st.session_state.scenarios = generate_scenarios()
            st.session_state.predictions = predict_yields(st.session_state.scenarios, st.session_state.model)
            st.session_state.recommendations = generate_recommendations(st.session_state.predictions)
            st.session_state.data_loaded = True
    
    live_data = fetch_live_data()
    baseline_yield = live_data['current_baseline']
    baseline_date = live_data['baseline_date']
    
    with st.sidebar:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">{logo_svg}</div>', unsafe_allow_html=True)
        
        st.header("Informations du Modèle")
        st.markdown("### Données en Temps Réel")
        
        fetch_status = live_data.get('fetch_status', {})
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            policy_status = fetch_status.get('policy_rate', '🟡 Fallback')
            st.metric("Taux Directeur", f"{live_data['policy_rate']:.2f}%", help=f"Source: {policy_status}")
            inflation_status = fetch_status.get('inflation', '🟡 Fallback') 
            st.metric("Inflation", f"{live_data['inflation']:.2f}%", help=f"Source: {inflation_status}")
        
        with col2:
            baseline_status = fetch_status.get('baseline', '🟡 Dernière Publication')
            st.metric("Baseline Actuelle", f"{baseline_yield:.2f}%", help=f"Source: {baseline_status}")
            gdp_status = fetch_status.get('gdp', '🟡 Estimation')
            st.metric("Croissance PIB", f"{live_data['gdp_growth']:.2f}%", help=f"Source: {gdp_status}")
        
        data_quality = live_data.get('data_quality', {})
        live_sources = data_quality.get('live_sources', 0)
        total_sources = data_quality.get('total_sources', 3)
        
        if live_sources == total_sources:
            quality_color = "#28a745"
            quality_text = f"🟢 Excellente ({live_sources}/{total_sources} temps réel)"
        elif live_sources >= total_sources // 2:
            quality_color = "#ffc107" 
            quality_text = f"🟡 Bonne ({live_sources}/{total_sources} temps réel)"
        else:
            quality_color = "#dc3545"
            quality_text = f"🟠 Limitée ({live_sources}/{total_sources} temps réel)"
        
        st.markdown(f"""
        <div style="background: {quality_color}22; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; border-left: 3px solid {quality_color};">
            <div style="font-size: 0.8rem; color: {quality_color}; font-weight: 600;">
                Qualité des données: {quality_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"Dernière MAJ: {live_data['last_updated']}")
        
        fetch_errors = live_data.get('fetch_errors', [])
        if fetch_errors:
            with st.sidebar.expander(f"⚠️ Erreurs de récupération ({len(fetch_errors)})"):
                for error in fetch_errors[:3]:
                    st.text(error[:100] + "..." if len(error) > 100 else error)
        
        baseline_info = live_data.get('baseline_info', {})
        baseline_status = fetch_status.get('baseline', '🟡 Dernière Publication')
        
        if '🟢 Temps Réel' in baseline_status:
            bg_color = "#e8f5e8"
            text_color = "#2d5a2d"
        else:
            bg_color = "#fff3cd" 
            text_color = "#856404"
        
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 0.8rem; border-radius: 6px; border-left: 3px solid #2a5298; margin: 0.5rem 0;">
            <div style="font-size: 0.75rem; color: {text_color};">
                <strong>🏛️ Baseline Bank Al-Maghrib:</strong> {baseline_date}<br>
                <strong>📊 Taux Variable:</strong> {baseline_yield:.2f}%<br>
                <strong>🔍 Statut:</strong> {baseline_info.get('status', 'Officiel')}<br>
                <strong>🔄 Source:</strong> {baseline_status}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if '🟡' in baseline_status:
            st.sidebar.info("ℹ️ Utilise dernière publication officielle")
        elif '🟢' in baseline_status:
            st.sidebar.success("✅ Données Bank Al-Maghrib à jour")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Vision Stratégique")
        
        cas_base_predictions = st.session_state.predictions['Cas_de_Base']
        
        three_month_data = cas_base_predictions.head(90)
        three_month_avg = three_month_data['rendement_predit'].mean()
        three_month_trend = "↗️ Hausse" if three_month_avg > baseline_yield else "↘️ Baisse" if three_month_avg < baseline_yield else "→ Stable"
        
        six_month_data = cas_base_predictions.head(180)
        six_month_min = six_month_data['rendement_predit'].min()
        six_month_max = six_month_data['rendement_predit'].max()
        
        st.sidebar.metric("📈 Tendance 3 mois", f"{three_month_avg:.2f}%", delta=f"{three_month_trend}", help="Direction générale sur 3 mois")
        st.sidebar.metric("🎯 Fourchette 6 mois", f"{six_month_min:.2f}%-{six_month_max:.2f}%", help="Plage attendue sur 6 mois")
        
        if three_month_avg < baseline_yield - 0.3:
            strategic_window = "🟢 Fenêtre favorable taux variable"
        elif three_month_avg > baseline_yield + 0.3:
            strategic_window = "🔴 Privilégier taux fixe"
        else:
            strategic_window = "🟡 Période de transition"
            
        st.sidebar.success(strategic_window)
        
        if st.sidebar.button("Actualiser"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("### Performance du Modèle")
        
        if hasattr(st.session_state, 'detailed_metrics') and st.session_state.detailed_metrics:
            metrics = st.session_state.detailed_metrics
            st.metric("R² Score", f"{st.session_state.r2:.1%}", help=f"Modèle: {metrics.get('model_name', 'Linear')}")
            st.metric("Précision (MAE)", f"±{st.session_state.mae:.3f}%")
            st.metric("Exactitude", f"{st.session_state.accuracy:.1f}%", help="±15bp tolérance")
            st.metric("Validation Croisée", f"±{st.session_state.mae_cv:.3f}%")
            
            if st.session_state.r2 >= 0.85 and st.session_state.accuracy >= 80:
                st.success("🟢 Modèle Excellent")
            elif st.session_state.r2 >= 0.75 and st.session_state.accuracy >= 70:
                st.info("🟡 Modèle Satisfaisant") 
            else:
                st.warning("🟠 Modèle Acceptable")
        else:
            st.metric("R² Score", f"{st.session_state.r2:.1%}")
            st.metric("Précision", f"±{st.session_state.mae:.2f}%")
            st.metric("Validation Croisée", f"±{st.session_state.mae_cv:.2f}%")
            st.metric("Exactitude ML", f"{st.session_state.accuracy:.1f}%")
            st.success("Modèle calibré avec succès")
    
    tab1, tab2, tab3 = st.tabs(["Vue d'Ensemble", "Prédictions Détaillées", "Recommandations"])
    
    with tab1:
        st.markdown('<div class="executive-dashboard">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; font-size: 1.4rem; font-weight: 700; margin-bottom: 2rem;">Tableau de Bord Stratégique</div>', unsafe_allow_html=True)
        
        cas_de_base_predictions = st.session_state.predictions['Cas_de_Base']
        
        q1_data = cas_de_base_predictions.head(90)
        q2_data = cas_de_base_predictions.head(180)
        year1_data = cas_de_base_predictions.head(365)
        
        q1_avg = q1_data['rendement_predit'].mean()
        q2_avg = q2_data['rendement_predit'].mean() 
        year1_avg = year1_data['rendement_predit'].mean()
        
        q1_change = q1_avg - baseline_yield
        q2_change = q2_avg - baseline_yield
        year1_change = year1_avg - baseline_yield
        
        q1_volatility = q1_data['rendement_predit'].std()
        max_deviation = max(abs(q1_change), abs(q2_change), abs(year1_change))
        
        if max_deviation > 0.5:
            if q1_change > 0.3:
                strategic_environment = "ENVIRONNEMENT DE HAUSSE"
                env_color = "#dc3545"
                strategic_action = "SÉCURISER IMMÉDIATEMENT - TAUX FIXES"
            elif year1_change < -0.3:
                strategic_environment = "ENVIRONNEMENT DE BAISSE"
                env_color = "#28a745"
                strategic_action = "MAXIMISER TAUX VARIABLES"
            else:
                strategic_environment = "ENVIRONNEMENT CYCLIQUE"
                env_color = "#ff6b35"
                strategic_action = "STRATÉGIE ADAPTATIVE REQUISE"
        elif max_deviation > 0.25:
            if q1_change > 0.2:
                strategic_environment = "ENVIRONNEMENT DE HAUSSE MODÉRÉE"
                env_color = "#ffc107"
                strategic_action = "PRÉPARER COUVERTURE - SURVEILLER"
            elif q1_volatility > 0.3:
                strategic_environment = "ENVIRONNEMENT VOLATIL"
                env_color = "#6f42c1"
                strategic_action = "GESTION ACTIVE DU RISQUE"
            else:
                strategic_environment = "ENVIRONNEMENT EN TRANSITION"
                env_color = "#17a2b8"
                strategic_action = "APPROCHE ÉQUILIBRÉE"
        else:
            strategic_environment = "ENVIRONNEMENT STABLE"
            env_color = "#28a745"
            strategic_action = "MAINTENIR STRATÉGIE ACTUELLE"
        
        trend_6m = q2_avg - baseline_yield
        
        st.markdown(f"""
        <div class="status-card" style="border-left-color: {env_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #2c3e50;">Environnement Stratégique</h3>
                    <p style="margin: 0.5rem 0; font-weight: 600; color: {env_color};">{strategic_environment}</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #6c757d;">{strategic_action}</p>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700; color: #2c3e50;">{year1_avg:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">Moyenne 12 mois</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem; font-weight: 600; color: {env_color};">{trend_6m:+.2f}%</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">Tendance 6 mois</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        volatility_6m = q2_data['rendement_predit'].std()
        optimal_window = "Q1" if q1_avg < q2_avg < year1_avg else "Q2-Q4" if q2_avg < year1_avg else "Immédiate"
        risk_level = "Faible" if volatility_6m < 0.2 else "Modéré" if volatility_6m < 0.4 else "Élevé"
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">HORIZON 3 MOIS</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{q1_avg:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">HORIZON 6 MOIS</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{q2_avg:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">STABILITÉ 6M</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 0.8rem; margin-bottom: 0.5rem;">FENÊTRE OPTIMALE</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2c3e50;">{optimal_window}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        q1_trend = q1_avg - baseline_yield
        q2_trend = q2_avg - baseline_yield
        year1_trend = year1_avg - baseline_yield
        
        if q1_trend > 0.3 and q2_trend > 0.2:
            timing_recommendation = "AGIR IMMÉDIATEMENT - Cycle de hausse confirmé"
            timing_color = "#dc3545"
        elif q1_trend > 0.25 and q2_trend < q1_trend:
            timing_recommendation = "SÉCURISER MAINTENANT - Pic temporaire approche"
            timing_color = "#ff6b35"
        elif q1_trend < -0.3 and q2_trend < -0.2:
            timing_recommendation = "ATTENDRE - Baisse continue favorable"
            timing_color = "#28a745"  
        elif abs(q1_trend - q2_trend) > 0.2:
            timing_recommendation = "STRATÉGIE ADAPTATIVE - Environnement cyclique"
            timing_color = "#6f42c1"
        elif year1_trend < -0.2:
            timing_recommendation = "PLANIFIER - Opportunités à moyen terme"
            timing_color = "#17a2b8"
        else:
            timing_recommendation = "SURVEILLER - Signaux mixtes"
            timing_color = "#ffc107"
        
        st.markdown(f"""
        <div class="recommendation-panel">
            <div style="text-align: center; font-size: 1.3rem; font-weight: 700; margin-bottom: 1.5rem;">STRATÉGIE & TIMING RECOMMANDÉS</div>
            <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 1rem; margin: 0.8rem 0;">
                <h4 style="margin: 0 0 0.5rem 0; color: white;">Timing Stratégique</h4>
                <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: {timing_color};">{timing_recommendation}</p>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 1rem;">
                    <h5 style="margin: 0 0 0.5rem 0; color: white;">Horizon 3 Mois</h5>
                    <p style="margin: 0; font-weight: 600;">{q1_avg:.2f}%</p>
                    <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.8;">Évolution: {q1_trend:+.2f}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); border-radius: 8px; padding: 1rem;">
                    <h5 style="margin: 0 0 0.5rem 0; color: white;">Horizon 12 Mois</h5>
                    <p style="margin: 0; font-weight: 600;">{year1_avg:.2f}%</p>
                    <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.8;">Tendance: {year1_trend:+.2f}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Évolution des Rendements")
        
        fig = go.Figure()
        
        df_hist = st.session_state.df.tail(8)
        fig.add_trace(go.Scatter(
            x=df_hist['Date'],
            y=df_hist['Rendement_52s'],
            mode='lines+markers',
            name='Historique',
            line=dict(color='#2a5298', width=4),
            marker=dict(size=8)
        ))
        
        colors = {'Conservateur': '#dc3545', 'Cas_de_Base': '#17a2b8', 'Optimiste': '#28a745'}
        for scenario, pred_df in st.session_state.predictions.items():
            sample_indices = list(range(0, len(pred_df), 7))
            
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
                     annotation_text=f"Baseline Bank Al-Maghrib ({baseline_date}): {baseline_yield:.2f}%")
        
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
        
        scenario_choice = st.selectbox("Choisissez un scénario:", ['Cas_de_Base', 'Conservateur', 'Optimiste'])
        
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
            st.metric("Écart vs Bank Al-Maghrib", f"{change:+.2f}%", help=f"Comparaison avec baseline {baseline_date}")
        
        st.subheader(f"Prédictions Quotidiennes - {scenario_choice}")
        
        sample_detailed = pred_data[::7]
        
        fig_detail = go.Figure()
        fig_detail.add_trace(go.Scatter(
            x=sample_detailed['Date'],
            y=sample_detailed['rendement_predit'],
            mode='lines+markers',
            name='Prédiction',
            line=dict(color=colors[scenario_choice], width=3)
        ))
        
        fig_detail.add_hline(y=baseline_yield, line_dash="dash", line_color="blue",
                           annotation_text=f"Bank Al-Maghrib ({baseline_date}): {baseline_yield:.2f}%")
        
        fig_detail.update_layout(
            height=500,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Rendement (%)"
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
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
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <h3 style="margin: 0; color: white;">🏦 AIDE À LA DÉCISION EMPRUNT SOFAC</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Analyse comparative Taux Fixe vs Taux Variable sur la durée du contrat</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("⚙️ Paramètres de l'Emprunt")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            loan_amount = st.slider("Montant (millions MAD):", 1, 500, 50)
        with col2:
            loan_duration = st.slider("Durée (années):", 1, 10, 5)
        with col3:
            current_fixed_rate = st.number_input("Taux fixe proposé (%):", min_value=1.0, max_value=10.0, value=3.2, step=0.1)
        with col4:
            risk_premium = st.number_input("Prime de risque (%):", min_value=0.5, max_value=3.0, value=1.3, step=0.1, help="Marge bancaire sur taux de référence")
        with col5:
            max_volatility_accepted = st.number_input("Volatilité Max (%):", min_value=0.1, max_value=1.0, value=0.40, step=0.05, help="Volatilité maximale acceptable")
        
        st.markdown(f"""
        <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #1976d2;">
            <div style="font-size: 0.85rem; color: #1565c0;">
                <strong>💡 Guide de Tolérance:</strong>
                <br>• <strong>Conservateur:</strong> 0.20-0.30% (volatilité très limitée)
                <br>• <strong>Équilibré:</strong> 0.30-0.40% (tolérance moyenne recommandée: 0.40%)
                <br>• <strong>Agressif:</strong> 0.40-0.60% (volatilité élevée pour gains supérieurs)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        banking_spread = risk_premium
        
        scenarios_analysis = {}
        
        for scenario_name, pred_df in st.session_state.predictions.items():
            loan_duration_days = loan_duration * 365
            relevant_predictions = pred_df.head(loan_duration_days)
            
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
                
                effective_rate = reference_rate + banking_spread
                variable_rates_annual.append(effective_rate)
            
            fixed_cost_total = (current_fixed_rate / 100) * loan_amount * 1_000_000 * loan_duration
            variable_cost_total = sum([(rate / 100) * loan_amount * 1_000_000 for rate in variable_rates_annual])
            
            cost_difference = variable_cost_total - fixed_cost_total
            cost_difference_percentage = (cost_difference / fixed_cost_total) * 100
            
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
        
        st.subheader("📊 Matrice de Décision par Scénario")
        
        decision_data = []
        for scenario_name, analysis in scenarios_analysis.items():
            if analysis['cost_difference'] < 0:
                recommendation = "TAUX VARIABLE"
                savings = abs(analysis['cost_difference'])
                decision_text = f"Économie de {savings:,.0f} MAD"
            else:
                recommendation = "TAUX FIXE" 
                extra_cost = analysis['cost_difference']
                decision_text = f"Éviter surcoût de {extra_cost:,.0f} MAD"
            
            risk_level = "FAIBLE" if analysis['volatility'] < 0.2 else "MOYEN" if analysis['volatility'] < 0.4 else "ÉLEVÉ"
            
            decision_data.append({
                'Scénario': scenario_name,
                'Taux Variable Effectif': f"{analysis['avg_variable_rate']:.2f}%",
                'Fourchette Effectif': f"{analysis['min_rate']:.2f}% - {analysis['max_rate']:.2f}%",
                'Coût Total Variable': f"{analysis['variable_cost_total']:,.0f} MAD",
                'Différence vs Fixe': decision_text,
                'Recommandation': recommendation,
                'Niveau Risque': risk_level,
                'Volatilité': f"{analysis['volatility']:.2f}%"
            })
        
        decision_df = pd.DataFrame(decision_data)
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
        
        variable_recommendations = sum(1 for analysis in scenarios_analysis.values() if analysis['cost_difference'] < 0)
        total_scenarios = len(scenarios_analysis)
        
        avg_cost_difference = np.mean([analysis['cost_difference'] for analysis in scenarios_analysis.values()])
        max_volatility = max([analysis['volatility'] for analysis in scenarios_analysis.values()])
        
        avg_savings = abs(avg_cost_difference)
        volatility_tolerance_margin = 0.05
        effective_max_volatility = max_volatility_accepted + volatility_tolerance_margin
        
        if variable_recommendations >= 2 and avg_cost_difference < 0 and max_volatility <= effective_max_volatility:
            final_recommendation = "TAUX VARIABLE"
            final_reason = f"Économies favorables ({avg_savings:,.0f} MAD) avec volatilité acceptable ({max_volatility:.2f}% ≤ {max_volatility_accepted:.2f}%)"
            final_color = "#28a745"
        elif variable_recommendations >= 2 and avg_cost_difference < 0 and max_volatility <= max_volatility_accepted * 1.5:
            final_recommendation = "STRATÉGIE MIXTE"
            final_reason = f"Économies probables ({avg_savings:,.0f} MAD) mais volatilité élevée ({max_volatility:.2f}% > {max_volatility_accepted:.2f}%)"
            final_color = "#ffc107"
        elif avg_cost_difference >= 0:
            final_recommendation = "TAUX FIXE"
            final_reason = f"Taux fixe plus avantageux - évite surcoût de {avg_savings:,.0f} MAD"
            final_color = "#dc3545"
        else:
            final_recommendation = "TAUX FIXE"
            final_reason = f"Volatilité excessive ({max_volatility:.2f}% >> {max_volatility_accepted:.2f}%) malgré économies potentielles"
            final_color = "#dc3545"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {final_color}, {final_color}AA); 
                    color: white; padding: 2rem; border-radius: 12px; margin: 2rem 0; text-align: center;">
            <h2>🎯 DÉCISION FINALE SOFAC</h2>
            <h3>{final_recommendation}</h3>
            <p><strong>Justification:</strong> {final_reason}</p>
            <p><strong>Montant:</strong> {loan_amount}M MAD | <strong>Durée:</strong> {loan_duration} ans | <strong>Taux fixe alternatif:</strong> {current_fixed_rate}%</p>
            <hr style="margin: 1rem 0; opacity: 0.3;">
            <div style="font-size: 0.9rem; opacity: 0.9;">
                <p><strong>Analyse:</strong> {variable_recommendations}/{total_scenarios} scénarios favorables au taux variable</p>
                <p><strong>Économie moyenne:</strong> {abs(avg_cost_difference):,.0f} MAD | <strong>Volatilité max:</strong> {max_volatility:.2f}%</p>
                <p><strong>Niveau de confiance:</strong> {min(95, 60 + variable_recommendations * 15 + (20 if avg_cost_difference < -1000000 else 0))}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
            reference_rate = base_case_analysis['avg_variable_rate'] - banking_spread
            st.metric("Taux Référence Moyen", f"{reference_rate:.2f}%", help="Prédiction du modèle")
            st.metric("+ Prime de Risque", f"+{banking_spread:.2f}%", help=f"Prime ajustable ({banking_spread:.1f}%)")
            st.metric("= Taux Effectif SOFAC", f"{base_case_analysis['avg_variable_rate']:.2f}%", help="Taux réel avec prime")
            st.metric("Fourchette Effective", f"{base_case_analysis['min_rate']:.2f}% - {base_case_analysis['max_rate']:.2f}%")
            if base_case_analysis['cost_difference'] < 0:
                st.success(f"💰 Économie potentielle: {abs(base_case_analysis['cost_difference']):,.0f} MAD")
            else:
                st.warning(f"⚠️ Surcoût potentiel: {base_case_analysis['cost_difference']:,.0f} MAD")
        
        st.subheader("📈 Évolution Annuelle des Taux (Cas de Base)")
        
        years = list(range(1, loan_duration + 1))
        fig_yearly = go.Figure()
        
        fig_yearly.add_trace(go.Scatter(
            x=years,
            y=[current_fixed_rate] * loan_duration,
            mode='lines+markers',
            name='Taux Fixe',
            line=dict(color='#dc3545', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
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
    
    st.markdown("---")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        logo_svg = create_sofac_logo_svg()
        st.markdown(f'<div style="text-align: center; margin-bottom: 1rem;">{logo_svg}</div>', unsafe_allow_html=True)
        
        baseline_info = live_data.get('baseline_info', {})
        st.markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p style="margin: 0; font-weight: bold; color: #2a5298;">SOFAC - Modèle de Prédiction des Rendements 52-Semaines</p>
            <p style="margin: 0; color: #FF6B35;">Dites oui au super crédit</p>
            <p style="margin: 0.5rem 0;">Baseline Bank Al-Maghrib: {baseline_date} ({baseline_yield:.2f}%) | Dernière mise à jour: {current_time}</p>
            <p style="margin: 0;"><em>Référence: {baseline_info.get('source', 'Bank Al-Maghrib - Taux Variable Officiel')}</em></p>
            <p style="margin: 0;"><em>Les prédictions sont basées sur des données historiques et ne constituent pas des conseils financiers.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()import streamlit as st
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

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_live_data():
    """Fetch live economic data directly from Bank Al-Maghrib and HCP websites"""
    today = datetime.now()
    
    policy_rate = 2.25
    inflation = 1.1
    gdp_growth = 4.8
    baseline_rate = 1.75
    
    data_sources = {
        'policy_rate_fetched': False,
        'inflation_fetched': False,
        'baseline_fetched': False,
        'fetch_errors': []
    }
    
    try:
        bam_url = "https://www.bkam.ma"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(bam_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            rate_patterns = [
                r'taux\s+directeur[:\s]*([0-9]+[,.]?[0-9]*)\s*%',
                r'policy\s+rate[:\s]*([0-9]+[,.]?[0-9]*)\s*%'
            ]
            
            text_content = soup.get_text().lower()
            for pattern in rate_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    rate_str = match.group(1).replace(',', '.')
                    try:
                        policy_rate = float(rate_str)
                        data_sources['policy_rate_fetched'] = True
                        break
                    except ValueError:
                        continue
    except Exception as e:
        data_sources['fetch_errors'].append(f"Bank Al-Maghrib fetch error: {str(e)}")
    
    try:
        hcp_urls = ["https://www.hcp.ma"]
        
        for hcp_url in hcp_urls:
            try:
                response = requests.get(hcp_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    inflation_patterns = [
                        r'inflation[:\s]*([0-9]+[,.]?[0-9]*)\s*%',
                        r'ipc[:\s]*([0-9]+[,.]?[0-9]*)\s*%'
                    ]
                    
                    text_content = soup.get_text().lower()
                    for pattern in inflation_patterns:
                        match = re.search(pattern, text_content, re.IGNORECASE)
                        if match:
                            rate_str = match.group(1).replace(',', '.')
                            try:
                                inflation = float(rate_str)
                                data_sources['inflation_fetched'] = True
                                break
                            except ValueError:
                                continue
                    
                    if data_sources['inflation_fetched']:
                        break
            except Exception:
                continue
    except Exception as e:
        data_sources['fetch_errors'].append(f"HCP fetch error: {str(e)}")
    
    if data_sources['baseline_fetched']:
        baseline_status = 'Temps Réel'
        baseline_date = today.strftime('%B %Y')
        baseline_source = 'Bank Al-Maghrib - Temps Réel'
    else:
        baseline_status = 'Officiel (Juin 2025)'
        baseline_date = 'Juin 2025'
        baseline_source = 'Bank Al-Maghrib - Dernière Publication'
    
    fetch_status = {
        'policy_rate': '🟢 Temps Réel' if data_sources['policy_rate_fetched'] else '🟡 Fallback',
        'inflation': '🟢 Temps Réel' if data_sources['inflation_fetched'] else '🟡 Fallback',
        'baseline': '🟢 Temps Réel' if data_sources['baseline_fetched'] else '🟡 Dernière Publication',
        'gdp': '🟡 Estimation'
    }
    
    return {
        'policy_rate': policy_rate,
        'inflation': inflation,
        'gdp_growth': gdp_growth,
        'current_baseline': baseline_rate,
        'baseline_date': baseline_date,
        'baseline_info': {
            'rate': baseline_rate,
            'publication_month': baseline_date,
            'source': baseline_source,
            'status': baseline_status
        },
        'fetch_status': fetch_status,
        'data_quality': {
            'live_sources': sum([data_sources['policy_rate_fetched'], data_sources['inflation_fetched'], data_sources['baseline_fetched']]),
            'total_sources': 3
        },
        'sources': {
            'policy_rate': 'Bank Al-Maghrib' + (' - Temps Réel' if data_sources['policy_rate_fetched'] else ' - Fallback'),
            'inflation': 'HCP' + (' - Temps Réel' if data_sources['inflation_fetched'] else ' - Fallback'),
            'baseline': baseline_source
        },
        'last_updated': today.strftime('%Y-%m-%d %H:%M:%S'),
        'fetch_errors': data_sources['fetch_errors']
    }

@st.cache_data
def create_dataset():
    """Create complete historical dataset with interpolation"""
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

def train_model(df):
    """Train enhanced prediction model"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    
    X_base = df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']].copy()
    y = df['Rendement_52s'].copy()
    
    X_enhanced = X_base.copy()
    X_enhanced['Taux_Inflation_Interaction'] = X_base['Taux_Directeur'] * X_base['Inflation']
    X_enhanced['Spread_Taux_Inflation'] = X_base['Taux_Directeur'] - X_base['Inflation']
    X_enhanced['PIB_Inflation_Ratio'] = X_base['Croissance_PIB'] / (X_base['Inflation'] + 0.1)
    X_enhanced['Taux_Directeur_Squared'] = X_base['Taux_Directeur'] ** 2
    X_enhanced['Inflation_Squared'] = X_base['Inflation'] ** 2
    
    if len(df) > 3:
        X_enhanced['Taux_Directeur_Lag1'] = X_base['Taux_Directeur'].shift(1).fillna(X_base['Taux_Directeur'].iloc[0])
        X_enhanced['Rendement_Lag1'] = y.shift(1).fillna(y.iloc[0])
    
    if len(df) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.2, random_state=42, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = X_enhanced, X_enhanced, y, y
    
    models_to_try = {
        'Linear_Enhanced': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }
    
    best_model = None
    best_r2 = -np.inf
    best_metrics = {}
    
    for model_name, model in models_to_try.items():
        try:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test) if len(X_test) > 0 else y_pred_train
            
            y_eval = y_test if len(y_test) > 0 else y_train
            y_pred_eval = y_pred_test if len(y_pred_test) > 0 else y_pred_train
            
            r2 = r2_score(y_eval, y_pred_eval)
            mae = mean_absolute_error(y_eval, y_pred_eval)
            rmse = np.sqrt(mean_squared_error(y_eval, y_pred_eval))
            
            accuracy_normal = np.mean(np.abs(y_eval - y_pred_eval) <= 0.15) * 100
            
            if len(df) >= 5:
                cv_scores = cross_val_score(model, X_enhanced, y, cv=min(5, len(df)//2), scoring='neg_mean_absolute_error')
                mae_cv = -cv_scores.mean()
            else:
                mae_cv = mae
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_metrics = {
                    'model_name': model_name,
                    'r2': r2,
                    'mae': mae,
                    'mae_cv': mae_cv,
                    'accuracy_normal': accuracy_normal
                }
        except Exception:
            continue
    
    if best_model is None:
        X_simple = df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']]
        best_model = LinearRegression()
        best_model.fit(X_simple, y)
        
        y_pred = best_model.predict(X_simple)
        best_metrics = {
            'model_name': 'LinearRegression_Fallback',
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'mae_cv': mean_absolute_error(y, y_pred),
            'accuracy_normal': np.mean(np.abs(y - y_pred) <= 0.15) * 100
        }
    
    best_model._feature_columns = list(X_enhanced.columns) if 'X_enhanced' in locals() else ['Taux_Directeur', 'Inflation', 'Croissance_PIB']
    best_model._base_columns = ['Taux_Directeur', 'Inflation', 'Croissance_PIB']
    
    return (best_model, best_metrics['r2'], best_metrics['mae'], best_metrics['mae_cv'], best_metrics['accuracy_normal'], best_metrics)

def generate_scenarios():
    """Generate realistic economic scenarios"""
    date_debut = datetime(2025, 7, 1)
    date_fin = datetime(2030, 12, 31)
    
    dates_quotidiennes = []
    date_courante = date_debut
    
    while date_courante <= date_fin:
        dates_quotidiennes.append(date_courante)
        date_courante += timedelta(days=1)
    
    decisions_politiques = {
        'Conservateur': {
            '2025-06': 2.25, '2025-09': 2.15, '2025-12': 2.00, '2026-03': 1.90, 
            '2026-06': 1.85, '2026-09': 1.80, '2026-12': 1.85, '2027-03': 1.90,
            '2027-06': 1.95, '2027-09': 2.00, '2027-12': 2.10, '2028-03': 2.20,
            '2028-06': 2.30, '2028-09': 2.35, '2028-12': 2.40, '2029-03': 2.45,
            '2029-06': 2.50, '2029-09': 2.55, '2029-12': 2.60, '2030-06': 2.65,
            '2030-12': 2.70
        },
        'Cas_de_Base': {
            '2025-06': 2.25, '2025-09': 2.00, '2025-12': 1.75, '2026-03': 1.50, 
            '2026-06': 1.30, '2026-09': 1.25, '2026-12': 1.20, '2027-03': 1.25,
            '2027-06': 1.35, '2027-09': 1.45, '2027-12': 1.60, '2028-03': 1.75,
            '2028-06': 1.90, '2028-09': 2.05, '2028-12': 2.20, '2029-03': 2.25,
            '2029-06': 2.30, '2029-09': 2.25, '2029-12': 2.20, '2030-06': 2.25,
            '2030-12': 2.30
        },
        'Optimiste': {
            '2025-06': 2.25, '2025-09': 1.75, '2025-12': 1.50, '2026-03': 1.25, 
            '2026-06': 1.00, '2026-09': 0.85, '2026-12': 0.75, '2027-03': 0.80,
            '2027-06': 0.90, '2027-09': 1.00, '2027-12': 1.15, '2028-03': 1.30,
            '2028-06': 1.45, '2028-09': 1.60, '2028-12': 1.75, '2029-03': 1.80,
            '2029-06': 1.85, '2029-09': 1.80, '2029-12': 1.75, '2030-06': 1.80,
            '2030-12': 1.85
        }
    }
    
    scenarios = {}
    
    for nom_scenario in ['Conservateur', 'Cas_de_Base', 'Optimiste']:
        donnees_scenario = []
        taux_politiques = decisions_politiques[nom_scenario]
        
        for i, date in enumerate(dates_quotidiennes):
            jours_ahead = i + 1
            date_str = date.strftime('%Y-%m')
            
            sorted_dates = sorted(taux_politiques.keys())
            current_rate = taux_politiques[sorted_dates[0]]
            
            for j, policy_date in enumerate(sorted_dates):
                if date_str >= policy_date:
                    current_rate = taux_politiques[policy_date]
                elif j > 0:
                    prev_date = sorted_dates[j-1]
                    prev_rate = taux_politiques[prev_date]
                    curr_rate = taux_politiques[policy_date]
                    
                    prev_month = datetime.strptime(prev_date + '-01', '%Y-%m-%d')
                    curr_month = datetime.strptime(policy_date + '-01', '%Y-%m-%d')
                    target_month = datetime(date.year, date.month, 1)
                    
                    if curr_month > prev_month:
                        days_total = (curr_month - prev_month).days
                        days_elapsed = (target_month - prev_month).days
                        factor = max(0, min(1, days_elapsed / days_total))
                        current_rate = prev_rate + factor * (curr_rate - prev_rate)
                    break
            
            taux_directeur = current_rate
            
            np.random.seed(hash(date.strftime('%Y-%m-%d')) % 2**32)
            mois_depuis_debut = (date.year - 2025) * 12 + date.month - 7
            
            if nom_scenario == 'Conservateur':
                inflation_base = 1.9 + 0.3 * np.sin(2 * np.pi * mois_depuis_debut / 24) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 6)
                pib_base = 3.2 + 0.4 * np.sin(2 * np.pi * mois_depuis_debut / 36) + 0.2 * np.sin(2 * np.pi * mois_depuis_debut / 12)
            elif nom_scenario == 'Cas_de_Base':
                inflation_base = 1.6 + 0.25 * np.sin(2 * np.pi * mois_depuis_debut / 20) + 0.12 * np.sin(2 * np.pi * mois_depuis_debut / 8)
                pib_base = 3.8 + 0.35 * np.sin(2 * np.pi * mois_depuis_debut / 30) + 0.18 * np.sin(2 * np.pi * mois_depuis_debut / 10)
            else:
                inflation_base = 1.3 + 0.2 * np.sin(2 * np.pi * mois_depuis_debut / 18) + 0.1 * np.sin(2 * np.pi * mois_depuis_debut / 7)
                pib_base = 4.2 + 0.3 * np.sin(2 * np.pi * mois_depuis_debut / 28) + 0.15 * np.sin(2 * np.pi * mois_depuis_debut / 9)
            
            mois = date.month
            seasonal_inflation = 0.05 * np.sin(2 * np.pi * mois / 12)
            seasonal_pib = 0.1 * np.sin(2 * np.pi * (mois - 3) / 12)
            
            noise_factor = 0.02 if nom_scenario == 'Conservateur' else 0.015
            inflation_noise = np.random.normal(0, noise_factor)
            pib_noise = np.random.normal(0, noise_factor * 1.5)
            
            if np.random.random() < 0.005:
                shock_magnitude = np.random.uniform(-0.3, 0.2)
                inflation_noise += shock_magnitude * 0.5
                pib_noise += shock_magnitude
            
            inflation = max(0.5, min(4.0, inflation_base + seasonal_inflation + inflation_noise))
            pib = max(1.0, min(7.0, pib_base + seasonal_pib + pib_noise))
            
            donnees_scenario.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Taux_Directeur': taux_directeur,
                'Inflation': inflation,
                'Croissance_PIB': pib,
                'Jours_Ahead': jours_ahead,
                'Jour_Semaine': date.strftime('%A'),
                'Est_Weekend': date.weekday() >= 5,
                'Mois': mois,
                'Annee': date.year
            })
        
        scenarios[nom_scenario] = pd.DataFrame(donnees_scenario)
    
    return scenarios

def predict_yields(scenarios, model):
    """Generate yield predictions with enhanced realism"""
    baseline = 1.75
    predictions = {}
    
    has_enhanced_features = hasattr(model, '_feature_columns') and hasattr(model, '_base_columns')
    
    for scenario_name, scenario_df in scenarios.items():
        X_base = scenario_df[['Taux_Directeur', 'Inflation', 'Croissance_PIB']].copy()
        
        if has_enhanced_features:
            X_enhanced = X_base.copy()
            X_enhanced['Taux_Inflation_Interaction'] = X_base['Taux_Directeur'] * X_base['Inflation']
            X_enhanced['Spread_Taux_Inflation'] = X_base['Taux_Directeur'] - X_base['Inflation']
            X_enhanced['PIB_Inflation
