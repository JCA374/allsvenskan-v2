"""Step 1 -- Data page."""
import pandas as pd
import streamlit as st

from core.config import (
    RESULTS_PATH, FIXTURES_PATH, HISTORICAL_PATH,
)
from core.data.scraper import AllsvenskanScraper
from core.data.cleaner import DataCleaner
from core.utils.helpers import normalize_team_names, build_standings
from core.ui.helpers import (
    stepper, next_button, load_results, load_fixtures, color_table_row,
)


def render():
    stepper()
    st.title("Step 1 \u2014 Data")

    with st.expander("\U0001f4d0 How the pipeline works"):
        st.markdown("""
**Data \u2192 Model \u2192 Simulate \u2192 Forecast**

| Step | What happens |
|------|-------------|
| **Data** | Download completed match results + upcoming fixtures from football-data.co.uk |
| **Model** | Fits a Dixon-Coles model to estimate each team's attack & defence strength |
| **Simulate** | Plays out the remaining fixtures 10 000\u00d7 using random Poisson draws |
| **Forecast** | Aggregates the 10 000 simulations into finish probabilities |

**What the raw data contains:** `FTHG` = Full-Time Home Goals, `FTAG` = Full-Time Away Goals,
`SeasonStart` = calendar year the season started (e.g. 2025 for the 2025 season).
Rows without goal values are upcoming fixtures.
        """)

    col_load, col_status = st.columns([1, 2])

    with col_load:
        current_year = pd.Timestamp.now().year
        history_exists = False
        if HISTORICAL_PATH.exists():
            try:
                history_exists = len(pd.read_csv(HISTORICAL_PATH)) > 0
            except Exception:
                pass

        st.subheader("Historical data (one-time)")
        if history_exists:
            hist_df      = pd.read_csv(HISTORICAL_PATH)
            hist_seasons = sorted(hist_df["SeasonStart"].dropna().astype(int).unique())
            st.success(f"Cached: seasons {hist_seasons[0]}\u2013{hist_seasons[-1]} ({len(hist_df)} matches)")
        else:
            st.info("No historical data cached yet.")

        if st.button("Download All History", disabled=history_exists):
            past_years = list(range(2012, current_year))
            with st.spinner(f"Downloading seasons {past_years[0]}\u2013{past_years[-1]}\u2026"):
                try:
                    scraper = AllsvenskanScraper()
                    raw = scraper.scrape_matches(seasons=past_years)
                    if raw.empty:
                        st.error("No historical data returned.")
                    else:
                        cleaner = DataCleaner()
                        hist_results, _ = cleaner.clean_data(raw)
                        hist_results = normalize_team_names(hist_results)
                        HISTORICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                        hist_results.to_csv(HISTORICAL_PATH, index=False)
                        st.success(f"Saved {len(hist_results)} historical matches.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Download failed: {e}")

        st.divider()

        st.subheader(f"Current season ({current_year})")
        if st.button("Refresh Current Season", type="primary"):
            with st.spinner(f"Fetching {current_year} data\u2026"):
                try:
                    scraper = AllsvenskanScraper()
                    raw = scraper.scrape_matches(seasons=[current_year])
                    if raw.empty:
                        st.error(f"No data returned for {current_year}.")
                    else:
                        cleaner = DataCleaner()
                        cur_results, cur_fixtures = cleaner.clean_data(raw)

                        cur_results  = normalize_team_names(cur_results)
                        cur_fixtures = normalize_team_names(cur_fixtures)

                        if history_exists:
                            hist_df  = normalize_team_names(pd.read_csv(HISTORICAL_PATH, parse_dates=["Date"]))
                            combined = pd.concat([hist_df, cur_results], ignore_index=True)
                            combined = combined.drop_duplicates(
                                subset=["Date", "HomeTeam", "AwayTeam"], keep="last"
                            )
                        else:
                            combined = cur_results
                        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                        FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
                        combined.to_csv(RESULTS_PATH, index=False)
                        cur_fixtures.to_csv(FIXTURES_PATH, index=False)
                        _today = pd.Timestamp.now().normalize()
                        _fix_col = "Date" if "Date" in cur_fixtures.columns else cur_fixtures.columns[0]
                        _upcoming = cur_fixtures[pd.to_datetime(cur_fixtures[_fix_col]) >= _today].copy()
                        _upcoming.to_csv("data/clean/upcoming_fixtures.csv", index=False)

                        st.session_state.data_loaded = True
                        st.success(
                            f"Updated: {len(cur_results)} matches this season, "
                            f"{len(cur_fixtures)} upcoming fixtures. "
                            f"{len(combined)} total results saved."
                        )
                        st.rerun()
                except Exception as e:
                    st.error(f"Refresh failed: {e}")

    with col_status:
        if st.session_state.data_loaded:
            try:
                results  = load_results()
                fixtures = load_fixtures()
                c1, c2, c3 = st.columns(3)
                c1.metric("Results", len(results))
                c2.metric("Upcoming fixtures", len(fixtures))
                seasons = sorted(results["SeasonStart"].dropna().astype(int).unique()) if "SeasonStart" in results.columns else []
                c3.metric("Seasons loaded", len(seasons) if seasons else "\u2014")
            except Exception as e:
                st.warning(f"Could not read data files: {e}")
        else:
            st.info("Download historical data then refresh the current season to get started.")

    if st.session_state.data_loaded:
        st.divider()
        st.subheader("Current Season Standings")
        try:
            results = load_results()
            if "SeasonStart" in results.columns:
                results_current = results[results["SeasonStart"] == results["SeasonStart"].max()]
            else:
                results_current = results
            standings = build_standings(results_current)
            n = len(standings)
            styled = (
                standings.style
                .apply(color_table_row, n_teams=n, axis=1)
                .format({"GD": "{:+d}"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption("\U0001f7e2 Title contender  |  \U0001f535 European spots  |  \U0001f534 Relegation zone")
        except Exception as e:
            st.error(f"Could not build standings: {e}")

        st.divider()
        st.subheader("Browse historical matches")
        try:
            all_results = load_results()
            teams = sorted(pd.unique(all_results[["HomeTeam", "AwayTeam"]].values.ravel()))
            fc1, fc2, fc3 = st.columns(3)
            sel_home   = fc1.selectbox("Home team", ["Any"] + teams, key="hist_home")
            sel_away   = fc2.selectbox("Away team", ["Any"] + teams, key="hist_away")
            seasons_avail = sorted(all_results["SeasonStart"].dropna().astype(int).unique(), reverse=True) if "SeasonStart" in all_results.columns else []
            sel_season = fc3.selectbox("Season", ["All"] + [str(s) for s in seasons_avail], key="hist_season")

            view = all_results.copy()
            if sel_home   != "Any":  view = view[view["HomeTeam"] == sel_home]
            if sel_away   != "Any":  view = view[view["AwayTeam"] == sel_away]
            if sel_season != "All":  view = view[view["SeasonStart"] == int(sel_season)]
            view = view.sort_values("Date", ascending=False)
            st.caption(f"{len(view)} matches")
            st.dataframe(
                view[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam", "SeasonStart"]]
                .rename(columns={"FTHG": "HG", "FTAG": "AG", "SeasonStart": "Season"})
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
        except Exception as e:
            st.error(f"Could not load match history: {e}")

    if st.session_state.data_loaded:
        next_button("Model")
