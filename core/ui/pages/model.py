"""Step 2 -- Model page."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import (
    RESULTS_PATH, FIXTURES_PATH, TEAM_STATS_PATH, MODEL_PATH,
)
from core.data.strength import TeamStrengthCalculator
from core.models.poisson_model import PoissonModel
from core.utils.helpers import normalize_team_names
from core.ui.helpers import (
    stepper, blocked, next_button, load_results, load_model,
)


def render():
    stepper()
    st.title("Step 2 \u2014 Model")

    with st.expander("\U0001f4d0 Dixon-Coles model \u2014 how it works", expanded=True):
        st.markdown(r"""
**Core idea:** Dixon & Coles (1997) bivariate-Poisson model fitted by maximum pseudo-likelihood with exponential time-weighting.

$$\lambda = \exp(\alpha_{\text{home}} + \beta_{\text{away}} + \gamma) \qquad \mu = \exp(\alpha_{\text{away}} + \beta_{\text{home}})$$

| Symbol | Meaning | Typical value |
|--------|---------|--------------|
| **\u03b1** | Attack strength (log-scale). Higher = scores more goals | top teams \u2248 +0.4 \u2013 +0.6 |
| **\u03b2** | Defence weakness (log-scale). **Lower (more negative) = concedes less** | top teams \u2248 \u22120.4 \u2013 \u22120.6 |
| **\u03b3** | Home advantage (log-scale) | 0.16 \u2013 0.35 |
| **\u03c1** | Low-score dependence \u2014 corrects 0-0 / 1-1 joint probabilities | \u22120.15 \u2013 0.05 |

Probabilities are computed from a full score-line grid (P(h,a) for h,a up to 10) with the DC \u03c4 correction applied to the four low-score cells. Results sum to exactly 1 after normalisation.

Time-weighting uses **\u03c6(t) = exp(\u2212\u03be\u00b7t)** where t is days before training and \u03be \u2248 0.0018 day\u207b\u00b9 (a match 1 year ago retains ~52% weight; 2 years ago ~27%).
        """)

    if not st.session_state.data_loaded:
        blocked("Data", "Download data first before training the model.")

    col_train, col_result = st.columns([1, 2])

    with col_train:
        st.subheader("Train")

        if st.button("Train Model", type="primary"):
            with st.spinner("Calculating team strengths\u2026"):
                try:
                    results = load_results()
                    strength_calc = TeamStrengthCalculator(use_odds_integration=False)
                    team_stats = strength_calc.calculate_strengths(results)
                    TEAM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    team_stats.to_csv(TEAM_STATS_PATH)
                except Exception as e:
                    st.error(f"Strength calculation failed: {e}")
                    st.stop()

            with st.spinner("Fitting Dixon-Coles model\u2026"):
                try:
                    team_stats = pd.read_csv(TEAM_STATS_PATH, index_col=0)
                    model = PoissonModel()
                    model.fit(results, team_stats)
                    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    model.save(str(MODEL_PATH))
                    st.session_state.model_trained = True
                    st.success("Model trained and saved.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    with col_result:
        if st.session_state.model_trained:
            try:
                model = load_model()
                st.subheader("Parameters")
                trained_label = model.last_trained.strftime("%Y-%m-%d %H:%M") if getattr(model, "last_trained", None) else "Unknown"
                st.caption(f"Dixon-Coles \u00b7 Trained: **{trained_label}**")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Home advantage \u03b3",  f"{model.home_advantage:.3f}",
                           help="Log-scale home advantage parameter. Typical range 0.16\u20130.35.")
                mc2.metric("Low-score \u03c1",       f"{model.rho:.4f}",
                           help="Dixon-Coles low-score dependence. Negative = 0-0/1-1 more frequent than independent Poisson predicts.")
                mc3.metric("Held-out RPS",      f"{model.validation_score:.4f}" if model.validation_score else "N/A",
                           help="Rank Probability Score on last 10% of training data. Lower is better.")

                if model.attack_rates:
                    st.subheader("Team Strengths (DC parameters)")
                    st.caption(
                        "\u03b1 (attack): log-scale, fitted by Dixon-Coles \u2014 higher = more goals scored. "
                        "\u03b2 (defence): log-scale \u2014 lower (more negative) = fewer goals conceded."
                    )

                    _current_teams: set = set()
                    try:
                        if RESULTS_PATH.exists():
                            _res = pd.read_csv(RESULTS_PATH)
                            _res = normalize_team_names(_res)
                            if "SeasonStart" in _res.columns:
                                _latest = int(_res["SeasonStart"].dropna().max())
                                _res = _res[_res["SeasonStart"] == _latest]
                            _current_teams = set(_res["HomeTeam"].tolist() + _res["AwayTeam"].tolist())
                        if FIXTURES_PATH.exists():
                            _fix = pd.read_csv(FIXTURES_PATH)
                            _fix = normalize_team_names(_fix)
                            _current_teams |= set(_fix["HomeTeam"].tolist() + _fix["AwayTeam"].tolist())
                    except Exception:
                        pass

                    team_filter = st.radio(
                        "Show teams", ["Current season", "All teams"],
                        horizontal=True, key="model_team_filter"
                    )

                    dc_df = pd.DataFrame({
                        "\u03b1 (Attack)":  pd.Series(model.attack_rates),
                        "\u03b2 (Defence)": pd.Series(model.defense_rates),
                    }).sort_values("\u03b1 (Attack)", ascending=False)
                    dc_df.index.name = "Team"

                    if team_filter == "Current season" and _current_teams:
                        dc_df = dc_df[dc_df.index.isin(_current_teams)]

                    fig = go.Figure()
                    fig.add_bar(x=dc_df.index, y=dc_df["\u03b1 (Attack)"],  name="Attack (\u03b1)",              marker_color="#2196F3")
                    fig.add_bar(x=dc_df.index, y=dc_df["\u03b2 (Defence)"], name="Defence (\u03b2, lower=better)", marker_color="#F44336")
                    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="\u03b1=0 (league avg)")
                    fig.update_layout(
                        barmode="group", xaxis_tickangle=-45,
                        height=320, margin=dict(t=20, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(dc_df.round(3), use_container_width=True)

                if model.attack_rates:
                    st.subheader("Try a prediction")
                    teams_sorted = sorted(model.attack_rates.keys())
                    _pc1, _pc2 = st.columns(2)
                    ex_home = _pc1.selectbox("Home team", teams_sorted, key="ex_home")
                    ex_away_opts = [t for t in teams_sorted if t != ex_home]
                    ex_away = _pc2.selectbox("Away team", ex_away_opts, key="ex_away")

                    ah  = model.attack_rates.get(ex_home, 0.0)
                    dh  = model.defense_rates.get(ex_home, 0.0)
                    aa  = model.attack_rates.get(ex_away, 0.0)
                    da  = model.defense_rates.get(ex_away, 0.0)
                    gam = model.home_advantage
                    mu_h, mu_a = model.predict_match(ex_home, ex_away)

                    st.markdown(
                        f"**\u03bb_home** = exp(\u03b1_{ex_home}({ah:+.3f}) + \u03b2_{ex_away}({da:+.3f}) + \u03b3({gam:+.3f})) "
                        f"= **{mu_h:.2f} xG**  \n"
                        f"**\u03bb_away** = exp(\u03b1_{ex_away}({aa:+.3f}) + \u03b2_{ex_home}({dh:+.3f})) "
                        f"= **{mu_a:.2f} xG**"
                    )
                    try:
                        probs = model.predict_outcome_probabilities(ex_home, ex_away)
                        _m1, _m2, _m3 = st.columns(3)
                        _m1.metric(f"{ex_home} win", f"{probs['home_win']:.0%}")
                        _m2.metric("Draw",           f"{probs['draw']:.0%}")
                        _m3.metric(f"{ex_away} win", f"{probs['away_win']:.0%}")
                        st.caption(f"Dixon-Coles \u03c1 = {model.rho:.4f}")
                    except Exception:
                        pass

            except Exception as e:
                st.warning(f"Could not display model info: {e}")
        else:
            st.info("No model trained yet. Choose settings and click Train Model.")

    if st.session_state.model_trained:
        next_button("Simulate")
