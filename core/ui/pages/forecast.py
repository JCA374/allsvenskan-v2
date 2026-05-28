"""Step 4 -- Forecast page."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import (
    RESULTS_PATH, FIXTURES_PATH, SIM_PATH, FORECAST_CACHE_PATH,
    GAMES_PER_TEAM, RELEGATION_SPOTS, EUROPEAN_SPOTS,
)
from core.utils.helpers import normalize_team_names
from core.ui.helpers import (
    stepper, blocked, next_button,
    load_sim, run_forecast_computation, save_forecast_cache, load_forecast_cache,
)


def render():
    stepper()
    st.title("Step 4 \u2014 Forecast")

    if not st.session_state.sim_complete:
        blocked("Simulate", "Run a simulation first to see the season forecast.")

    try:
        if FORECAST_CACHE_PATH.exists():
            table, champ, releg, europe, pos_probs, summary, updated_at = load_forecast_cache()
        else:
            sim = load_sim(SIM_PATH.stat().st_mtime if SIM_PATH.exists() else 0)
            table, champ, releg, europe, pos_probs, summary = run_forecast_computation(sim)
            try:
                result = save_forecast_cache()
                updated_at = result[-1]
            except Exception:
                updated_at = None
    except Exception as e:
        st.error(f"Could not load forecast: {e}")
        st.stop()

    if updated_at:
        st.caption(f"Updated at: {updated_at.strftime('%Y-%m-%d %H:%M')}")

    n_teams = len(table)

    # ── Expected Final Table ─────────────────────────────────────────────────
    st.subheader("Expected Final Standings")

    try:
        all_results = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
        all_results = normalize_team_names(all_results)
        if "SeasonStart" in all_results.columns and not all_results.empty:
            latest_season = int(all_results["SeasonStart"].dropna().max())
            season_results = all_results[all_results["SeasonStart"] == latest_season].copy()
        else:
            season_results = all_results.copy()
    except Exception:
        season_results = pd.DataFrame()

    try:
        season_fixtures = pd.read_csv(FIXTURES_PATH, parse_dates=["Date"])
        season_fixtures = normalize_team_names(season_fixtures)
    except Exception:
        season_fixtures = pd.DataFrame()

    def _games_for_team(team, results_df, fixtures_df):
        played   = int((results_df["HomeTeam"] == team).sum() + (results_df["AwayTeam"] == team).sum()) if not results_df.empty else 0
        upcoming = int((fixtures_df["HomeTeam"] == team).sum() + (fixtures_df["AwayTeam"] == team).sum()) if not fixtures_df.empty else 0
        total = played + upcoming
        return total if total >= GAMES_PER_TEAM else GAMES_PER_TEAM

    tbl = table.copy()
    tbl["GP"]           = tbl["Team"].map(lambda t: _games_for_team(t, season_results, season_fixtures))
    tbl["Title %"]      = tbl["Team"].map(lambda t: champ.get(t, 0) * 100)
    tbl["Europe %"]     = tbl["Team"].map(lambda t: europe.get(t, 0) * 100)
    tbl["Relegation %"] = tbl["Team"].map(lambda t: releg.get(t, 0) * 100)
    if not summary.empty:
        std_map = dict(zip(summary["Team"], summary["Std_Points"]))
        tbl["Pts \u00b1"] = tbl["Team"].map(lambda t: std_map.get(t, 0))

    # ── Mobile-first primary table ───────────────────────────────────────────
    tbl_primary = tbl[["Position", "Team", "GP", "Expected_Points", "Title %", "Relegation %"]].copy()
    tbl_primary = tbl_primary.rename(columns={
        "Position": "#", "Expected_Points": "Pts", "Title %": "Title", "Relegation %": "Rel"
    }).reset_index(drop=True)

    def _row_color_primary(row):
        pos = int(row["#"])
        if pos == 1:                          return ["background-color: #fffde7"] * len(row)
        if pos <= EUROPEAN_SPOTS:             return ["background-color: #e3f2fd"] * len(row)
        if pos > n_teams - RELEGATION_SPOTS:  return ["background-color: #fce4ec"] * len(row)
        return [""] * len(row)

    fmt_primary = {"GP": "{:.0f}", "Pts": "{:.0f}", "Title": "{:.1f}%", "Rel": "{:.1f}%"}
    _tbl_height = n_teams * 35 + 38

    st.dataframe(
        tbl_primary.style.apply(_row_color_primary, axis=1).format(fmt_primary),
        use_container_width=True,
        hide_index=True,
        height=_tbl_height,
    )

    st.caption("\U0001f7e1 1st \u00b7 \U0001f535 2nd\u20133rd \u2014 Europe \u00b7 \U0001f534 Relegation   |   **Pts** = expected final points \u00b7 **Title/Rel** = % of simulations")

    with st.expander("Full table \u2014 GP, spread, European spots"):
        tbl_full = tbl[["Position", "Team", "GP", "Expected_Points"] +
                       (["Pts \u00b1"] if "Pts \u00b1" in tbl.columns else []) +
                       ["Title %", "Europe %", "Relegation %"]].copy()
        tbl_full = tbl_full.rename(columns={"Expected_Points": "Exp Pts"}).reset_index(drop=True)

        def _row_color_full(row):
            pos = int(row["Position"])
            if pos == 1:                          return ["background-color: #fffde7"] * len(row)
            if pos <= EUROPEAN_SPOTS:             return ["background-color: #e3f2fd"] * len(row)
            if pos > n_teams - RELEGATION_SPOTS:  return ["background-color: #fce4ec"] * len(row)
            return [""] * len(row)

        fmt_full = {"Exp Pts": "{:.1f}", "Title %": "{:.1f}%", "Europe %": "{:.1f}%", "Relegation %": "{:.1f}%"}
        if "Pts \u00b1" in tbl_full.columns:
            fmt_full["Pts \u00b1"] = "\u00b1{:.1f}"

        st.dataframe(
            tbl_full.style.apply(_row_color_full, axis=1).format(fmt_full),
            use_container_width=True,
            hide_index=True,
            height=_tbl_height,
        )

    st.divider()

    # ── Charts ───────────────────────────────────────────────────────────────
    tab_title, tab_europe, tab_releg, tab_heat = st.tabs(
        ["Title Race", "European Spots", "Relegation", "Heatmap"]
    )

    def _sorted_bar(probs: dict, color: str, threshold: float = 0.005):
        data = {k: v * 100 for k, v in probs.items() if v > threshold}
        df   = pd.DataFrame({"Team": list(data.keys()), "Probability": list(data.values())})
        df   = df.sort_values("Probability", ascending=True)
        fig  = px.bar(
            df, x="Probability", y="Team", orientation="h",
            text=df["Probability"].map(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=[color],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="Probability (%)", yaxis_title="",
            margin=dict(l=10, r=50, t=10, b=30),
            height=min(max(220, len(df) * 30), 480),
            showlegend=False,
        )
        return fig

    with tab_title:
        st.plotly_chart(_sorted_bar(champ, "#FFC107"), use_container_width=True)
    with tab_europe:
        st.plotly_chart(_sorted_bar(europe, "#2196F3"), use_container_width=True)
    with tab_releg:
        st.plotly_chart(_sorted_bar(releg, "#F44336"), use_container_width=True)
    with tab_heat:
        if pos_probs:
            teams_ordered = tbl["Team"].tolist()
            n_pos  = len(teams_ordered)
            matrix = np.array([
                [pos_probs.get(team, [0] * n_pos)[p] for p in range(n_pos)]
                for team in teams_ordered
            ])
            fig = go.Figure(go.Heatmap(
                z=matrix * 100,
                x=[f"#{p+1}" for p in range(n_pos)],
                y=teams_ordered,
                colorscale="Blues",
                showscale=False,
                hovertemplate="%{y} \u2192 position %{x}: %{z:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                xaxis_title="", yaxis_title="",
                height=max(380, n_pos * 28),
                margin=dict(l=10, r=10, t=10, b=30),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(side="bottom", tickfont=dict(size=11)),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"scrollZoom": False, "displayModeBar": False},
            )
        else:
            st.info("Position probabilities not available.")

    next_button("Predictions")
