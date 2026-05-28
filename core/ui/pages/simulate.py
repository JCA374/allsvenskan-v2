"""Step 3 -- Simulate page."""
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import (
    RESULTS_PATH, FIXTURES_PATH, MODEL_PATH, SIM_PATH,
    FORECAST_CACHE_PATH, GAMES_PER_TEAM,
)
from core.simulation.simulator import MonteCarloSimulator
from core.utils.helpers import normalize_team_names, build_standings
from core.ui.helpers import (
    stepper, blocked, next_button,
    load_fixtures, load_model, load_results, load_sim,
    run_forecast_computation, save_forecast_cache, load_forecast_cache,
)


def render():
    stepper()
    st.title("Step 3 \u2014 Simulate")

    with st.expander("\U0001f4d0 How the Monte Carlo simulation works"):
        st.markdown("""
**What it does:** plays out every remaining fixture *N* times, drawing random Poisson-distributed scorelines each time.

For each remaining fixture:
1. Compute \u03bc_home and \u03bc_away from the fitted model
2. Draw `home_goals ~ Poisson(\u03bc_home)`, `away_goals ~ Poisson(\u03bc_away)`
3. Award 3 pts (win) / 1 pt each (draw) / 0 pts (loss)
4. Accumulate points on top of actual current standings

After *N* simulations each team has a distribution of final points.
From that distribution we read championship, European, and relegation probabilities directly.

**Current standings seed:** the simulation starts from each team's *real* points already earned this season.
This is critical \u2014 without it, early-season leaders would get no advantage.

**Accuracy vs speed:** each doubling of *N* halves the standard error on probabilities.
- 1 000 sims \u2192 \u00b13% uncertainty on a 50% probability
- 10 000 sims \u2192 \u00b11%
- 50 000 sims \u2192 \u00b10.4%
        """)

    if not st.session_state.model_trained:
        blocked("Model", "Train the model before running a simulation.")

    col_cfg, col_info = st.columns([1, 2])

    with col_cfg:
        st.subheader("Settings")

        n_sims = st.slider(
            "Simulations",
            min_value=500, max_value=50_000, value=10_000, step=500,
            help="More = more accurate probabilities, but slower.",
        )

        upcoming_path = Path("data/clean/upcoming_fixtures.csv")
        uf = pd.DataFrame()
        if upcoming_path.exists():
            try:
                uf = pd.read_csv(upcoming_path, parse_dates=["Date"])
            except Exception:
                pass
        n_fixtures = len(uf) if not uf.empty else len(load_fixtures())

        st.metric("Upcoming fixtures", n_fixtures)
        if not uf.empty and "Date" in uf.columns:
            d_min = uf["Date"].min()
            d_max = uf["Date"].max()
            st.caption(f"Fixtures: {d_min.strftime('%d %b %Y')} \u2013 {d_max.strftime('%d %b %Y')}")

        if MODEL_PATH.exists():
            try:
                _m = load_model()
                _trained  = getattr(_m, "last_trained", None)
                _win_lbl  = f"{_m.training_window} season(s)" if getattr(_m, "training_window", None) else "all seasons"
                _date_str = _trained.strftime("%d %b %Y %H:%M") if _trained else "unknown"
                st.caption(f"Model: **Dixon-Coles**, window **{_win_lbl}**, trained {_date_str}")
            except Exception:
                pass

        if n_fixtures == 0:
            st.warning("No upcoming fixtures. Re-download data on the Data page.")

        if st.button("Run Simulation", type="primary", disabled=n_fixtures == 0):
            progress = st.progress(0, text="Starting\u2026")

            def _cb(pct):
                progress.progress(int(pct), text=f"{pct:.0f}%")

            try:
                model = load_model()
                simulator = MonteCarloSimulator.from_upcoming_fixtures(model)

                try:
                    _all_res = normalize_team_names(pd.read_csv(RESULTS_PATH, parse_dates=["Date"]))
                    if "SeasonStart" in _all_res.columns and not _all_res.empty:
                        _latest  = int(_all_res["SeasonStart"].dropna().max())
                        _cur     = _all_res[_all_res["SeasonStart"] == _latest]
                    else:
                        _cur = pd.DataFrame()
                    _stnd = build_standings(_cur) if not _cur.empty else pd.DataFrame()
                    current_pts = dict(zip(_stnd["Team"], _stnd["Pts"])) if not _stnd.empty else {}
                except Exception:
                    current_pts = {}

                if current_pts:
                    sim_results = simulator.run_monte_carlo_with_standings(
                        n_simulations=n_sims,
                        current_standings=current_pts,
                        progress_callback=_cb,
                    )
                else:
                    sim_results = simulator.run(n_simulations=n_sims, progress_callback=_cb)

                SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
                sim_results.to_csv(SIM_PATH, index=False)
                progress.progress(100, text="Computing forecast\u2026")
                try:
                    save_forecast_cache()
                except Exception as _fe:
                    st.warning(f"Forecast cache failed: {_fe}")
                st.session_state.sim_complete = True
                progress.progress(100, text="Done!")
                st.success(f"Completed {n_sims:,} simulations over {n_fixtures} fixtures.")
                st.rerun()
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    with col_info:
        with st.expander("\U0001f50d Data sanity check", expanded=True):
            issues = []
            try:
                _san_model    = load_model() if MODEL_PATH.exists() else None
                _san_fix_path = Path("data/clean/upcoming_fixtures.csv")
                _san_fix      = pd.read_csv(_san_fix_path) if _san_fix_path.exists() else pd.DataFrame()
                _san_res      = load_results() if RESULTS_PATH.exists() else pd.DataFrame()

                model_teams   = set(_san_model.attack_rates.keys()) if _san_model else set()
                fixture_teams = set(_san_fix["HomeTeam"].tolist() + _san_fix["AwayTeam"].tolist()) if not _san_fix.empty else set()
                result_teams  = set(pd.unique(_san_res[["HomeTeam", "AwayTeam"]].values.ravel())) if not _san_res.empty else set()

                in_fixtures_not_model = fixture_teams - model_teams

                st.markdown(f"**Model teams:** {len(model_teams)}  \n"
                            f"**Fixture teams:** {len(fixture_teams)}  \n"
                            f"**Overlap:** {len(model_teams & fixture_teams)}")

                if in_fixtures_not_model:
                    st.warning(f"In fixtures but NOT in model (will use default \u03b1=\u03b2=1.0): {sorted(in_fixtures_not_model)}")
                    issues.append("team mismatch")
                else:
                    st.success("All fixture teams are in the model \u2705")

                if not _san_res.empty and "SeasonStart" in _san_res.columns:
                    latest = int(_san_res["SeasonStart"].dropna().max())
                    cur    = _san_res[_san_res["SeasonStart"] == latest]
                    cur_teams = set(pd.unique(cur[["HomeTeam", "AwayTeam"]].values.ravel()))
                    seed_mismatch = cur_teams - fixture_teams
                    if seed_mismatch:
                        st.warning(f"Teams in current standings but NOT in upcoming fixtures (standings seed may be incomplete): {sorted(seed_mismatch)}")
                        issues.append("standings seed mismatch")
                    else:
                        st.success(f"Current season standings seed ({len(cur_teams)} teams) consistent with fixtures \u2705")

                    _all_fix = normalize_team_names(pd.read_csv(FIXTURES_PATH)) if FIXTURES_PATH.exists() else _san_fix
                    _all_fix_teams = set(_all_fix["HomeTeam"].tolist() + _all_fix["AwayTeam"].tolist()) if not _all_fix.empty else set()
                    _gp_gaps = []
                    for _t in sorted(cur_teams & _all_fix_teams):
                        _played = int((cur["HomeTeam"] == _t).sum() + (cur["AwayTeam"] == _t).sum())
                        _upcoming = int((_all_fix["HomeTeam"] == _t).sum() + (_all_fix["AwayTeam"] == _t).sum())
                        if _played + _upcoming != GAMES_PER_TEAM:
                            _gp_gaps.append((_t, _played, _upcoming))
                    if _gp_gaps:
                        _max_gap = max(GAMES_PER_TEAM - (p + u) for _, p, u in _gp_gaps)
                        if _max_gap > 2:
                            st.warning(f"{len(_gp_gaps)} teams missing >{_max_gap} games \u2014 data may be incomplete.")
                            issues.append("GP mismatch")
                        else:
                            _played_total = _gp_gaps[0][1]
                            st.success(f"All teams: {GAMES_PER_TEAM} GP \u00b7 {_played_total} played \u00b7 {GAMES_PER_TEAM - _played_total} simulated \u2705")
                    else:
                        _played_sample = int((cur["HomeTeam"] == list(cur_teams)[0]).sum() + (cur["AwayTeam"] == list(cur_teams)[0]).sum())
                        st.success(f"All teams: {GAMES_PER_TEAM} GP \u00b7 {_played_sample} played \u00b7 {GAMES_PER_TEAM - _played_sample} simulated \u2705")
            except Exception as e:
                st.caption(f"Sanity check skipped: {e}")

        if st.session_state.sim_complete:
            try:
                if FORECAST_CACHE_PATH.exists():
                    _, _, _, _, _, summary, _ = load_forecast_cache()
                else:
                    sim = load_sim(SIM_PATH.stat().st_mtime if SIM_PATH.exists() else 0)
                    _, _, _, _, _, summary = run_forecast_computation(sim)
                if not summary.empty:
                    st.subheader("Last Simulation \u2014 Top 3")
                    for _, row in summary.head(3).iterrows():
                        st.metric(row["Team"], f"{row['Mean_Points']:.1f} pts avg")
            except Exception as e:
                st.warning(f"Could not load previous results: {e}")
        else:
            st.info("Configure settings and click Run Simulation.")

    if st.session_state.sim_complete:
        next_button("Forecast")
