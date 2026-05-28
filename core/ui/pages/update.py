"""Update Everything page -- state-machine pipeline."""
import pandas as pd
import streamlit as st

from core.config import (
    RESULTS_PATH, FIXTURES_PATH, HISTORICAL_PATH,
    TEAM_STATS_PATH, MODEL_PATH, SIM_PATH,
)
from core.data.scraper import AllsvenskanScraper
from core.data.cleaner import DataCleaner
from core.data.strength import TeamStrengthCalculator
from core.models.poisson_model import PoissonModel
from core.simulation.simulator import MonteCarloSimulator
from core.utils.helpers import normalize_team_names, build_standings
from core.ui.helpers import nav, load_results, load_model, save_forecast_cache


def render():
    # ── Session-state defaults for this page ─────────────────────────────────
    for _k, _v in {
        "upd_step":  None,
        "upd_stop":  False,
        "upd_log":   [],
    }.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    upd_step = st.session_state.upd_step
    upd_stop = st.session_state.upd_stop

    st.title("\U0001f504 Update Everything")

    # ── Completed-step log ───────────────────────────────────────────────────
    for entry in st.session_state.upd_log:
        if entry["state"] == "complete":
            st.success(f"\u2705 {entry['label']} \u2014 {entry['msg']}")
        elif entry["state"] == "skipped":
            st.info(f"\u23ed {entry['label']} \u2014 {entry['msg']}")
        elif entry["state"] == "error":
            st.error(f"\u274c {entry['label']} \u2014 {entry['msg']}")
        elif entry["state"] == "stopped":
            st.warning(f"\u23f9 {entry['label']} \u2014 {entry['msg']}")

    # ── Idle ─────────────────────────────────────────────────────────────────
    if upd_step is None:
        st.caption("Fetch fresh data \u2192 train model \u2192 10 000 simulations \u2192 Predictions")

        current_year   = pd.Timestamp.now().year
        data_age_hours = None
        if RESULTS_PATH.exists():
            try:
                mtime          = RESULTS_PATH.stat().st_mtime
                data_age_hours = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(mtime)).total_seconds() / 3600
            except Exception:
                pass

        history_ok = False
        if HISTORICAL_PATH.exists():
            try:
                history_ok = len(pd.read_csv(HISTORICAL_PATH)) > 0
            except Exception:
                pass
        if not history_ok and RESULTS_PATH.exists():
            try:
                _r = pd.read_csv(RESULTS_PATH)
                if "SeasonStart" in _r.columns:
                    history_ok = _r["SeasonStart"].nunique() > 1
                else:
                    history_ok = len(_r) > 100
            except Exception:
                pass

        if data_age_hours is not None:
            if data_age_hours < 1:
                st.success(f"Data is fresh ({data_age_hours * 60:.0f} min ago) \u2014 fetch will be skipped.")
            else:
                st.info(f"Data last fetched {data_age_hours:.1f}h ago.")
        else:
            st.warning("No data on disk yet.")

        if not history_ok:
            st.warning(
                "Historical data not downloaded. Go to **Data** and click "
                "**Download All History** first."
            )

        if st.button("Update Everything", type="primary", disabled=not history_ok):
            st.session_state.upd_step = "data"
            st.session_state.upd_stop = False
            st.session_state.upd_log  = []
            st.rerun()

    # ── Done or stopped ──────────────────────────────────────────────────────
    elif upd_step == "done":
        if upd_stop:
            st.warning("Stopped \u2014 completed steps have been saved.")
        else:
            st.success("All steps complete!")

        col_a, col_b = st.columns(2)
        if col_a.button("Go to Predictions", type="primary"):
            st.session_state.upd_step = None
            nav("Predictions")
        if col_b.button("Run again"):
            st.session_state.upd_step = None
            st.session_state.upd_log  = []
            st.rerun()

    # ── Running a step ───────────────────────────────────────────────────────
    else:
        if st.button("\u23f9 Stop after this step", type="secondary"):
            st.session_state.upd_stop = True
            st.rerun()

        current_year = pd.Timestamp.now().year

        def _log(label, state, msg):
            st.session_state.upd_log.append({"label": label, "state": state, "msg": msg})

        def _advance(next_step):
            st.session_state.upd_step = "done" if st.session_state.upd_stop else next_step
            st.rerun()

        # ── Step 1: Data ─────────────────────────────────────────────────────
        if upd_step == "data":
            data_age_hours = None
            if RESULTS_PATH.exists():
                try:
                    mtime          = RESULTS_PATH.stat().st_mtime
                    data_age_hours = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(mtime)).total_seconds() / 3600
                except Exception:
                    pass

            skip_fetch = data_age_hours is not None and data_age_hours < 1

            st.subheader("Step 1/3 \u2014 Match data")
            bar  = st.progress(0, text="Starting\u2026")

            if skip_fetch:
                bar.progress(100, text="Skipped (data < 1h old)")
                _log("Step 1/3 \u2014 Data", "skipped",
                     f"data is {data_age_hours * 60:.0f} min old (< 1 hour)")
                _advance("model")

            try:
                bar.progress(10, text="Connecting to football-data.co.uk\u2026")
                scraper = AllsvenskanScraper()
                raw = scraper.scrape_matches(seasons=[current_year])
                if raw.empty:
                    raise RuntimeError("No data returned from source")

                bar.progress(40, text="Cleaning and normalising team names\u2026")
                cleaner      = DataCleaner()
                cur_results, cur_fixtures = cleaner.clean_data(raw)
                cur_results  = normalize_team_names(cur_results)
                cur_fixtures = normalize_team_names(cur_fixtures)

                all_fix_teams = pd.concat([cur_fixtures["HomeTeam"], cur_fixtures["AwayTeam"]]).unique()
                bad_djurg = [t for t in all_fix_teams
                             if ("jurg" in str(t).lower() or "\u00e5rd" in str(t).lower())
                             and t != "Djurgarden"]
                if bad_djurg:
                    st.warning(f"Unexpected Djurg\u00e5rden variant after normalisation: {bad_djurg}")

                bar.progress(70, text="Merging with historical data\u2026")
                hist_df  = normalize_team_names(pd.read_csv(HISTORICAL_PATH, parse_dates=["Date"]))
                combined = pd.concat([hist_df, cur_results], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["Date", "HomeTeam", "AwayTeam"], keep="last"
                )

                bar.progress(90, text="Saving\u2026")
                RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(RESULTS_PATH, index=False)
                cur_fixtures.to_csv(FIXTURES_PATH, index=False)
                _today = pd.Timestamp.now().normalize()
                _fix_col = "Date" if "Date" in cur_fixtures.columns else cur_fixtures.columns[0]
                _upcoming = cur_fixtures[pd.to_datetime(cur_fixtures[_fix_col]) >= _today].copy()
                _upcoming.to_csv("data/clean/upcoming_fixtures.csv", index=False)
                st.session_state.data_loaded = True

                bar.progress(100, text="Done")
                _log("Step 1/3 \u2014 Data", "complete",
                     f"{len(cur_results)} matches fetched \u00b7 {len(cur_fixtures)} upcoming fixtures")
                _advance("model")

            except Exception as e:
                bar.progress(100, text="Failed")
                _log("Step 1/3 \u2014 Data", "error", str(e))
                st.session_state.upd_step = "done"
                st.rerun()

        # ── Step 2: Model ────────────────────────────────────────────────────
        elif upd_step == "model":
            st.subheader("Step 2/3 \u2014 Training model")
            bar = st.progress(0, text="Loading data\u2026")

            try:
                results = load_results()

                bar.progress(20, text="Calculating team strengths\u2026")
                strength_calc = TeamStrengthCalculator(use_odds_integration=False)
                team_stats    = strength_calc.calculate_strengths(results)
                TEAM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
                team_stats.to_csv(TEAM_STATS_PATH)

                bar.progress(50, text="Fitting Dixon-Coles model\u2026")
                model = PoissonModel()
                model.fit(results, team_stats)

                bar.progress(90, text="Saving model\u2026")
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(MODEL_PATH))
                st.session_state.model_trained = True

                n_teams = len(model.attack_rates)
                bar.progress(100, text="Done")
                _log("Step 2/3 \u2014 Model", "complete",
                     f"{n_teams} teams \u00b7 {len(results)} matches")
                _advance("sim")

            except Exception as e:
                bar.progress(100, text="Failed")
                _log("Step 2/3 \u2014 Model", "error", str(e))
                st.session_state.upd_step = "done"
                st.rerun()

        # ── Step 3: Simulate ─────────────────────────────────────────────────
        elif upd_step == "sim":
            st.subheader("Step 3/3 \u2014 Simulating 10 000 seasons")
            bar = st.progress(0, text="Loading model and fixtures\u2026")

            try:
                model     = load_model()
                simulator = MonteCarloSimulator.from_upcoming_fixtures(model)

                try:
                    _all_res = normalize_team_names(pd.read_csv(RESULTS_PATH, parse_dates=["Date"]))
                    if "SeasonStart" in _all_res.columns and not _all_res.empty:
                        _latest  = int(_all_res["SeasonStart"].dropna().max())
                        _cur_res = _all_res[_all_res["SeasonStart"] == _latest]
                    else:
                        _cur_res = pd.DataFrame()
                    _stnd       = build_standings(_cur_res) if not _cur_res.empty else pd.DataFrame()
                    current_pts = dict(zip(_stnd["Team"], _stnd["Pts"])) if not _stnd.empty else {}
                except Exception:
                    current_pts = {}

                def _sim_cb(pct):
                    bar.progress(int(pct), text=f"Simulating\u2026 {pct:.0f}%")

                bar.progress(5, text="Starting simulations\u2026")
                if current_pts:
                    sim_results = simulator.run_monte_carlo_with_standings(
                        n_simulations=10_000,
                        current_standings=current_pts,
                        progress_callback=_sim_cb,
                    )
                else:
                    sim_results = simulator.run(
                        n_simulations=10_000,
                        progress_callback=_sim_cb,
                    )

                bar.progress(98, text="Saving results\u2026")
                SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
                sim_results.to_csv(SIM_PATH, index=False)
                bar.progress(99, text="Computing forecast cache\u2026")
                try:
                    save_forecast_cache()
                except Exception as _fe:
                    st.warning(f"Forecast cache failed: {_fe}")
                st.session_state.sim_complete = True

                bar.progress(100, text="Done")
                _log("Step 3/3 \u2014 Simulate", "complete", "10 000 simulations complete")
                st.session_state.upd_step = "done"
                st.rerun()

            except Exception as e:
                bar.progress(100, text="Failed")
                _log("Step 3/3 \u2014 Simulate", "error", str(e))
                st.session_state.upd_step = "done"
                st.rerun()
