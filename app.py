import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data.scraper import AllsvenskanScraper
from core.data.cleaner import DataCleaner
from core.data.strength import TeamStrengthCalculator
from core.models.poisson_model import PoissonModel
from core.simulation.simulator import MonteCarloSimulator
from core.analysis.aggregator import ResultsAggregator
from core.utils.helpers import TEAM_NAME_MAP

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Allsvenskan Forecast",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="auto",
)

# Responsive CSS: stack Streamlit columns on narrow screens
st.markdown("""
<style>
@media (max-width: 640px) {
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* Prevent wide tables / charts from overflowing */
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
        max-width: 100vw;
        overflow-x: auto;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_PATH   = Path("data/clean/results.csv")
FIXTURES_PATH  = Path("data/clean/fixtures.csv")
HISTORICAL_PATH = Path("data/clean/historical_results.csv")
TEAM_STATS_PATH = Path("data/processed/team_stats.csv")
MODEL_PATH     = Path("models/poisson_params.pkl")
SIM_PATH            = Path("reports/simulations/sim_results_latest.csv")
FORECAST_CACHE_PATH = Path("reports/simulations/forecast_cache.pkl")
RELEGATION_SPOTS = 3
EUROPEAN_SPOTS   = 3   # 1st: CL qualifying · 2nd–3rd: ECL qualifying (cup winner gets EL separately)

# ── Pipeline steps definition ──────────────────────────────────────────────────
STEPS = [
    ("Data",        "🗄️",  "data_loaded"),
    ("Model",       "🧠",  "model_trained"),
    ("Simulate",    "🎲",  "sim_complete"),
    ("Forecast",    "📊",  "sim_complete"),
    ("Predictions", "📅",  "model_trained"),
]

def _nav(page: str):
    st.session_state.active_page = page
    # No st.rerun() here — button on_click callbacks trigger a rerun automatically.

def _step_done(key: str) -> bool:
    return bool(st.session_state.get(key, False))

# ── Session-state defaults ─────────────────────────────────────────────────────
for key, default in {
    "data_loaded":    False,
    "model_trained":  False,
    "sim_complete":   False,
    "active_page":    "Forecast",
    "admin_unlocked": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Detect on-disk state so page refreshes don't lose context
if not st.session_state.data_loaded and RESULTS_PATH.exists():
    try:
        if len(pd.read_csv(RESULTS_PATH)) > 0:
            st.session_state.data_loaded = True
    except Exception:
        pass

if not st.session_state.model_trained and MODEL_PATH.exists():
    st.session_state.model_trained = True

if not st.session_state.sim_complete and SIM_PATH.exists():
    try:
        if len(pd.read_csv(SIM_PATH)) > 0:
            st.session_state.sim_complete = True
    except Exception:
        pass

# ── Sidebar ────────────────────────────────────────────────────────────────────
PUBLIC_PAGES  = {"Forecast", "Predictions"}
ADMIN_PAGES   = {"Data", "Model", "Simulate", "Update"}

with st.sidebar:
    st.title("⚽ Allsvenskan")
    st.caption("Monte Carlo Forecast")
    st.divider()

    admin = st.session_state.admin_unlocked
    visible_steps = STEPS if admin else [s for s in STEPS if s[0] in PUBLIC_PAGES]

    for i, (name, icon, gate) in enumerate(STEPS, 1):
        if name not in PUBLIC_PAGES and not admin:
            continue
        done   = _step_done(gate)
        active = st.session_state.active_page == name
        if active:
            label = f"**{icon} {name}**"
        elif done:
            label = f"{icon} {name} ✅"
        else:
            label = f"{icon} {name}"
        st.button(
            label,
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if active else "secondary",
            on_click=_nav,
            args=(name,),
        )

    if admin:
        st.divider()
        _upd_active = st.session_state.active_page == "Update"
        st.button(
            "**🔄 Update Everything**" if _upd_active else "🔄 Update Everything",
            key="nav_Update",
            use_container_width=True,
            type="primary" if _upd_active else "secondary",
            on_click=_nav,
            args=("Update",),
        )

    # ── Admin login / logout ───────────────────────────────────────────────
    st.divider()
    if not admin:
        with st.expander("🔒 Admin login"):
            pwd = st.text_input("Password", type="password", key="admin_pwd_input", label_visibility="collapsed")
            if st.button("Unlock", key="admin_login_btn", use_container_width=True):
                expected = st.secrets.get("ADMIN_PASSWORD", "")
                if expected and pwd == expected:
                    st.session_state.admin_unlocked = True
                    st.rerun()
                else:
                    st.error("Wrong password")
    else:
        st.caption("🔓 Admin mode")
        if st.button("Lock", key="admin_lock_btn", use_container_width=True):
            st.session_state.admin_unlocked = False
            if st.session_state.active_page in ADMIN_PAGES:
                st.session_state.active_page = "Forecast"
            st.rerun()

page = st.session_state.active_page

# Redirect non-admin users away from admin pages
if page in ADMIN_PAGES and not st.session_state.admin_unlocked:
    st.session_state.active_page = "Forecast"
    page = "Forecast"
    st.rerun()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TEAM_NAME_MAP to HomeTeam/AwayTeam — ensures e.g. 'Djurgården' → 'Djurgarden'."""
    df = df.copy()
    for col in ("HomeTeam", "AwayTeam"):
        if col in df.columns:
            df[col] = df[col].map(lambda t: TEAM_NAME_MAP.get(str(t).strip(), str(t).strip()))
    return df

def _load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
    df = _normalize_teams(df)
    return df[df["FTHG"].notna() & df["FTAG"].notna()].copy()

def _load_fixtures() -> pd.DataFrame:
    if FIXTURES_PATH.exists():
        df = pd.read_csv(FIXTURES_PATH, parse_dates=["Date"])
        if "FTHG" in df.columns:
            df = df[df["FTHG"].isna()]
        df = _normalize_teams(df[["Date", "HomeTeam", "AwayTeam"]].copy())
        return df
    return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])

def _load_model() -> PoissonModel:
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, PoissonModel):
        return data
    m = PoissonModel()
    m.load(str(MODEL_PATH))
    return m

@st.cache_data
def _load_sim(_mtime: float = 0) -> pd.DataFrame:
    df = pd.read_csv(SIM_PATH)
    # Normalise column names (team names) and merge duplicate variants.
    # e.g. both "Djurgården" and "Djurgarden" should collapse into "Djurgarden".
    rename_map = {col: TEAM_NAME_MAP.get(col, col) for col in df.columns}
    df = df.rename(columns=rename_map)
    # If renaming created duplicate columns, sum them (they represent the same team).
    df = df.T.groupby(level=0).sum().T
    return df

def _run_forecast_computation(sim: pd.DataFrame):
    """Pure computation — no caching. Call this once after simulation then save to disk."""
    rename_map = {col: TEAM_NAME_MAP.get(col, col) for col in sim.columns}
    sim = sim.rename(columns=rename_map).T.groupby(level=0).sum().T
    agg       = ResultsAggregator()
    table     = agg.generate_final_table_prediction(sim)
    champ     = agg.calculate_championship_odds(sim)
    releg     = agg.calculate_relegation_odds(sim, relegation_spots=RELEGATION_SPOTS)
    europe    = agg.calculate_european_qualification_odds(sim, european_spots=EUROPEAN_SPOTS)
    pos_probs = agg.calculate_position_probabilities(sim)
    summary   = agg.analyze_results(sim)
    return table, champ, releg, europe, pos_probs, summary

def _save_forecast_cache():
    """Compute forecast from current sim file and persist to disk."""
    sim = _load_sim(SIM_PATH.stat().st_mtime if SIM_PATH.exists() else 0)
    result = _run_forecast_computation(sim)
    FORECAST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FORECAST_CACHE_PATH, "wb") as f:
        pickle.dump(result, f)
    return result

def _load_forecast_cache():
    """Load pre-computed forecast from disk. Raises if not available."""
    with open(FORECAST_CACHE_PATH, "rb") as f:
        return pickle.load(f)

def _standings_from_results(results: pd.DataFrame) -> pd.DataFrame:
    teams = pd.unique(results[["HomeTeam", "AwayTeam"]].values.ravel())
    cols  = ["GP", "W", "D", "L", "GF", "GA", "GD", "Pts"]
    tbl   = pd.DataFrame(0, index=teams, columns=cols)
    for _, r in results.iterrows():
        h, a   = r["HomeTeam"], r["AwayTeam"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        tbl.at[h, "GP"] += 1; tbl.at[a, "GP"] += 1
        tbl.at[h, "GF"] += hg; tbl.at[h, "GA"] += ag
        tbl.at[a, "GF"] += ag; tbl.at[a, "GA"] += hg
        if hg > ag:
            tbl.at[h, "W"] += 1; tbl.at[a, "L"] += 1; tbl.at[h, "Pts"] += 3
        elif ag > hg:
            tbl.at[a, "W"] += 1; tbl.at[h, "L"] += 1; tbl.at[a, "Pts"] += 3
        else:
            tbl.at[h, "D"] += 1; tbl.at[a, "D"] += 1
            tbl.at[h, "Pts"] += 1; tbl.at[a, "Pts"] += 1
    tbl["GD"] = tbl["GF"] - tbl["GA"]
    return (
        tbl.sort_values(["Pts", "GD", "GF"], ascending=False)
           .reset_index().rename(columns={"index": "Team"})
    )

def _color_table_row(row, n_teams):
    pos = row.name + 1
    if pos == 1:                          return ["background-color: #d4edda"] * len(row)  # CL
    if pos <= EUROPEAN_SPOTS:             return ["background-color: #e3f2fd"] * len(row)  # ECL (2nd–3rd)
    if pos > n_teams - RELEGATION_SPOTS:  return ["background-color: #fce4ec"] * len(row)  # relegation
    return [""] * len(row)

def _stepper():
    """Render a compact progress strip at the top of each page."""
    parts = []
    for i, (name, icon, gate) in enumerate(STEPS, 1):
        done   = _step_done(gate)
        active = page == name
        if active:
            parts.append(f"**▶ {icon} {name}**")
        elif done:
            parts.append(f"✅ {name}")
        else:
            parts.append(f"<span style='color:#aaa'>{i}. {name}</span>")
    st.markdown("&nbsp; · &nbsp;".join(parts), unsafe_allow_html=True)
    st.divider()

def _next_button(next_page: str):
    """Render a 'Next step' button at the bottom of a page."""
    st.divider()
    if st.button(f"Next → {next_page}", type="primary"):
        _nav(next_page)

def _blocked(required_page: str, message: str):
    """Show a friendly blocker and stop rendering."""
    st.info(message)
    if st.button(f"Go to {required_page}"):
        _nav(required_page)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════
if page == "Data":
    _stepper()
    st.title("Step 1 — Data")

    with st.expander("📐 How the pipeline works"):
        st.markdown("""
**Data → Model → Simulate → Forecast**

| Step | What happens |
|------|-------------|
| **Data** | Download completed match results + upcoming fixtures from football-data.co.uk |
| **Model** | Fits a Dixon-Coles model to estimate each team's attack & defence strength |
| **Simulate** | Plays out the remaining fixtures 10 000× using random Poisson draws |
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
            st.success(f"Cached: seasons {hist_seasons[0]}–{hist_seasons[-1]} ({len(hist_df)} matches)")
        else:
            st.info("No historical data cached yet.")

        if st.button("Download All History", disabled=history_exists):
            past_years = list(range(2012, current_year))
            with st.spinner(f"Downloading seasons {past_years[0]}–{past_years[-1]}…"):
                try:
                    scraper = AllsvenskanScraper()
                    raw = scraper.scrape_matches(seasons=past_years)
                    if raw.empty:
                        st.error("No historical data returned.")
                    else:
                        cleaner = DataCleaner()
                        hist_results, _ = cleaner.clean_data(raw)
                        hist_results = _normalize_teams(hist_results)
                        HISTORICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
                        hist_results.to_csv(HISTORICAL_PATH, index=False)
                        st.success(f"Saved {len(hist_results)} historical matches.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Download failed: {e}")

        st.divider()

        st.subheader(f"Current season ({current_year})")
        if st.button("Refresh Current Season", type="primary"):
            with st.spinner(f"Fetching {current_year} data…"):
                try:
                    scraper = AllsvenskanScraper()
                    raw = scraper.scrape_matches(seasons=[current_year])
                    if raw.empty:
                        st.error(f"No data returned for {current_year}.")
                    else:
                        cleaner = DataCleaner()
                        cur_results, cur_fixtures = cleaner.clean_data(raw)

                        cur_results  = _normalize_teams(cur_results)
                        cur_fixtures = _normalize_teams(cur_fixtures)

                        if history_exists:
                            hist_df  = _normalize_teams(pd.read_csv(HISTORICAL_PATH, parse_dates=["Date"]))
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
                        cur_fixtures.to_csv("data/clean/upcoming_fixtures.csv", index=False)

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
                results  = _load_results()
                fixtures = _load_fixtures()
                c1, c2, c3 = st.columns(3)
                c1.metric("Results", len(results))
                c2.metric("Upcoming fixtures", len(fixtures))
                seasons = sorted(results["SeasonStart"].dropna().astype(int).unique()) if "SeasonStart" in results.columns else []
                c3.metric("Seasons loaded", len(seasons) if seasons else "—")
            except Exception as e:
                st.warning(f"Could not read data files: {e}")
        else:
            st.info("Download historical data then refresh the current season to get started.")

    if st.session_state.data_loaded:
        st.divider()
        st.subheader("Current Season Standings")
        try:
            results = _load_results()
            if "SeasonStart" in results.columns:
                results_current = results[results["SeasonStart"] == results["SeasonStart"].max()]
            else:
                results_current = results
            standings = _standings_from_results(results_current)
            n = len(standings)
            styled = (
                standings.style
                .apply(_color_table_row, n_teams=n, axis=1)
                .format({"GD": "{:+d}"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption("🟢 Title contender  |  🔵 European spots  |  🔴 Relegation zone")
        except Exception as e:
            st.error(f"Could not build standings: {e}")

        st.divider()
        st.subheader("Browse historical matches")
        try:
            all_results = _load_results()
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
        _next_button("Model")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model":
    _stepper()
    st.title("Step 2 — Model")

    with st.expander("📐 Dixon-Coles model — how it works"):
        st.markdown(r"""
**Core idea:** Dixon & Coles (1997) bivariate-Poisson model fitted by maximum pseudo-likelihood with exponential time-weighting.

$$\lambda = \exp(\alpha_{\text{home}} + \beta_{\text{away}} + \gamma) \qquad \mu = \exp(\alpha_{\text{away}} + \beta_{\text{home}})$$

| Symbol | Meaning | Typical value |
|--------|---------|--------------|
| **α** | Attack strength (log-scale). Higher = scores more goals | top teams ≈ +0.4 – +0.6 |
| **β** | Defence weakness (log-scale). **Lower (more negative) = concedes less** | top teams ≈ −0.4 – −0.6 |
| **γ** | Home advantage (log-scale) | 0.16 – 0.35 |
| **ρ** | Low-score dependence — corrects 0-0 / 1-1 joint probabilities | −0.15 – 0.05 |

Probabilities are computed from a full score-line grid (P(h,a) for h,a up to 10) with the DC τ correction applied to the four low-score cells. Results sum to exactly 1 after normalisation.

Time-weighting uses **φ(t) = exp(−ξ·t)** where t is days before training and ξ ≈ 0.0018 day⁻¹ (a match 1 year ago retains ~52% weight; 2 years ago ~27%).
        """)

    if not st.session_state.data_loaded:
        _blocked("Data", "Download data first before training the model.")

    col_train, col_result = st.columns([1, 2])

    with col_train:
        st.subheader("Train")

        if st.button("Train Model", type="primary"):
            with st.spinner("Calculating team strengths…"):
                try:
                    results = _load_results()
                    strength_calc = TeamStrengthCalculator(use_odds_integration=False)
                    team_stats = strength_calc.calculate_strengths(results)
                    TEAM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    team_stats.to_csv(TEAM_STATS_PATH)
                except Exception as e:
                    st.error(f"Strength calculation failed: {e}")
                    st.stop()

            with st.spinner("Fitting Dixon-Coles model…"):
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
                model = _load_model()
                st.subheader("Parameters")
                trained_label = model.last_trained.strftime("%Y-%m-%d %H:%M") if getattr(model, "last_trained", None) else "Unknown"
                st.caption(f"Dixon-Coles · Trained: **{trained_label}**")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Home advantage γ",  f"{model.home_advantage:.3f}",
                           help="Log-scale home advantage parameter. Typical range 0.16–0.35.")
                mc2.metric("Low-score ρ",       f"{model.rho:.4f}",
                           help="Dixon-Coles low-score dependence. Negative = 0-0/1-1 more frequent than independent Poisson predicts.")
                mc3.metric("Held-out RPS",      f"{model.validation_score:.4f}" if model.validation_score else "N/A",
                           help="Rank Probability Score on last 10% of training data. Lower is better.")

                if model.attack_rates:
                    st.subheader("Team Strengths (DC parameters)")
                    st.caption(
                        "α (attack): log-scale, fitted by Dixon-Coles — higher = more goals scored. "
                        "β (defence): log-scale — lower (more negative) = fewer goals conceded."
                    )
                    dc_df = pd.DataFrame({
                        "α (Attack)":  pd.Series(model.attack_rates),
                        "β (Defence)": pd.Series(model.defense_rates),
                    }).sort_values("α (Attack)", ascending=False)
                    dc_df.index.name = "Team"

                    fig = go.Figure()
                    fig.add_bar(x=dc_df.index, y=dc_df["α (Attack)"],  name="Attack (α)",              marker_color="#2196F3")
                    fig.add_bar(x=dc_df.index, y=dc_df["β (Defence)"], name="Defence (β, lower=better)", marker_color="#F44336")
                    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="α=0 (league avg)")
                    fig.update_layout(
                        barmode="group", xaxis_tickangle=-45,
                        height=320, margin=dict(t=20, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(dc_df.round(3), use_container_width=True)

                # ── Interactive prediction breakdown ──────────────────────────
                if model.attack_rates:
                    st.subheader("Try a prediction")
                    teams_sorted = sorted(model.attack_rates.keys())
                    _pc1, _pc2 = st.columns(2)
                    ex_home = _pc1.selectbox("Home team", teams_sorted, key="ex_home")
                    ex_away_opts = [t for t in teams_sorted if t != ex_home]
                    ex_away = _pc2.selectbox("Away team", ex_away_opts, key="ex_away")

                    # Dixon-Coles log-additive: λ = exp(α_home + β_away + γ)
                    ah  = model.attack_rates.get(ex_home, 0.0)
                    dh  = model.defense_rates.get(ex_home, 0.0)
                    aa  = model.attack_rates.get(ex_away, 0.0)
                    da  = model.defense_rates.get(ex_away, 0.0)
                    gam = model.home_advantage
                    mu_h, mu_a = model.predict_match(ex_home, ex_away)

                    st.markdown(
                        f"**λ_home** = exp(α_{ex_home}({ah:+.3f}) + β_{ex_away}({da:+.3f}) + γ({gam:+.3f})) "
                        f"= **{mu_h:.2f} xG**  \n"
                        f"**λ_away** = exp(α_{ex_away}({aa:+.3f}) + β_{ex_home}({dh:+.3f})) "
                        f"= **{mu_a:.2f} xG**"
                    )
                    try:
                        probs = model.predict_outcome_probabilities(ex_home, ex_away)
                        _m1, _m2, _m3 = st.columns(3)
                        _m1.metric(f"{ex_home} win", f"{probs['home_win']:.0%}")
                        _m2.metric("Draw",           f"{probs['draw']:.0%}")
                        _m3.metric(f"{ex_away} win", f"{probs['away_win']:.0%}")
                        st.caption(f"Dixon-Coles ρ = {model.rho:.4f}")
                    except Exception:
                        pass

            except Exception as e:
                st.warning(f"Could not display model info: {e}")
        else:
            st.info("No model trained yet. Choose settings and click Train Model.")

    if st.session_state.model_trained:
        _next_button("Simulate")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SIMULATE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Simulate":
    _stepper()
    st.title("Step 3 — Simulate")

    with st.expander("📐 How the Monte Carlo simulation works"):
        st.markdown("""
**What it does:** plays out every remaining fixture *N* times, drawing random Poisson-distributed scorelines each time.

For each remaining fixture:
1. Compute μ_home and μ_away from the fitted model
2. Draw `home_goals ~ Poisson(μ_home)`, `away_goals ~ Poisson(μ_away)`
3. Award 3 pts (win) / 1 pt each (draw) / 0 pts (loss)
4. Accumulate points on top of actual current standings

After *N* simulations each team has a distribution of final points.
From that distribution we read championship, European, and relegation probabilities directly.

**Current standings seed:** the simulation starts from each team's *real* points already earned this season.
This is critical — without it, early-season leaders would get no advantage.

**Accuracy vs speed:** each doubling of *N* halves the standard error on probabilities.
- 1 000 sims → ±3% uncertainty on a 50% probability
- 10 000 sims → ±1%
- 50 000 sims → ±0.4%
        """)

    if not st.session_state.model_trained:
        _blocked("Model", "Train the model before running a simulation.")

    col_cfg, col_info = st.columns([1, 2])

    with col_cfg:
        st.subheader("Settings")

        n_sims = st.slider(
            "Simulations",
            min_value=500, max_value=50_000, value=10_000, step=500,
            help="More = more accurate probabilities, but slower.",
        )

        # Fixture info
        upcoming_path = Path("data/clean/upcoming_fixtures.csv")
        uf = pd.DataFrame()
        if upcoming_path.exists():
            try:
                uf = pd.read_csv(upcoming_path, parse_dates=["Date"])
            except Exception:
                pass
        n_fixtures = len(uf) if not uf.empty else len(_load_fixtures())

        st.metric("Upcoming fixtures", n_fixtures)
        if not uf.empty and "Date" in uf.columns:
            d_min = uf["Date"].min()
            d_max = uf["Date"].max()
            st.caption(f"Fixtures: {d_min.strftime('%d %b %Y')} – {d_max.strftime('%d %b %Y')}")

        # Show what model will be used — no retrain here
        if MODEL_PATH.exists():
            try:
                _m = _load_model()
                _trained  = getattr(_m, "last_trained", None)
                _win_lbl  = f"{_m.training_window} season(s)" if getattr(_m, "training_window", None) else "all seasons"
                _date_str = _trained.strftime("%d %b %Y %H:%M") if _trained else "unknown"
                st.caption(f"Model: **Dixon-Coles**, window **{_win_lbl}**, trained {_date_str}")
            except Exception:
                pass

        if n_fixtures == 0:
            st.warning("No upcoming fixtures. Re-download data on the Data page.")

        if st.button("Run Simulation", type="primary", disabled=n_fixtures == 0):
            progress = st.progress(0, text="Starting…")

            def _cb(pct):
                progress.progress(int(pct), text=f"{pct:.0f}%")

            try:
                # Always use the model exactly as trained — settings are
                # fixed on the Model page; no silent retrain here.
                model = _load_model()
                simulator = MonteCarloSimulator.from_upcoming_fixtures(model)

                try:
                    _all_res = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
                    if "SeasonStart" in _all_res.columns and not _all_res.empty:
                        _latest  = int(_all_res["SeasonStart"].dropna().max())
                        _cur     = _all_res[_all_res["SeasonStart"] == _latest]
                    else:
                        _cur = pd.DataFrame()
                    _stnd = _standings_from_results(_cur) if not _cur.empty else pd.DataFrame()
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
                progress.progress(100, text="Computing forecast…")
                try:
                    _save_forecast_cache()
                except Exception as _fe:
                    st.warning(f"Forecast cache failed: {_fe}")
                st.session_state.sim_complete = True
                progress.progress(100, text="Done!")
                st.success(f"Completed {n_sims:,} simulations over {n_fixtures} fixtures.")
                st.rerun()
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    with col_info:
        # ── Data sanity check ─────────────────────────────────────────────
        with st.expander("🔍 Data sanity check", expanded=True):
            issues = []
            try:
                _san_model    = _load_model() if MODEL_PATH.exists() else None
                _san_fix_path = Path("data/clean/upcoming_fixtures.csv")
                _san_fix      = pd.read_csv(_san_fix_path) if _san_fix_path.exists() else pd.DataFrame()
                _san_res      = _load_results() if RESULTS_PATH.exists() else pd.DataFrame()

                model_teams   = set(_san_model.attack_rates.keys()) if _san_model else set()
                fixture_teams = set(_san_fix["HomeTeam"].tolist() + _san_fix["AwayTeam"].tolist()) if not _san_fix.empty else set()
                result_teams  = set(pd.unique(_san_res[["HomeTeam", "AwayTeam"]].values.ravel())) if not _san_res.empty else set()

                in_fixtures_not_model = fixture_teams - model_teams
                in_model_not_fixtures = model_teams - fixture_teams

                st.markdown(f"**Model teams:** {len(model_teams)}  \n"
                            f"**Fixture teams:** {len(fixture_teams)}  \n"
                            f"**Overlap:** {len(model_teams & fixture_teams)}")

                if in_fixtures_not_model:
                    st.warning(f"In fixtures but NOT in model (will use default α=β=1.0): {sorted(in_fixtures_not_model)}")
                    issues.append("team mismatch")
                else:
                    st.success("All fixture teams are in the model ✅")

                if not _san_res.empty and "SeasonStart" in _san_res.columns:
                    latest = int(_san_res["SeasonStart"].dropna().max())
                    cur    = _san_res[_san_res["SeasonStart"] == latest]
                    cur_teams = set(pd.unique(cur[["HomeTeam", "AwayTeam"]].values.ravel()))
                    seed_mismatch = cur_teams - fixture_teams
                    if seed_mismatch:
                        st.warning(f"Teams in current standings but NOT in upcoming fixtures (standings seed may be incomplete): {sorted(seed_mismatch)}")
                        issues.append("standings seed mismatch")
                    else:
                        st.success(f"Current season standings seed ({len(cur_teams)} teams) consistent with fixtures ✅")
            except Exception as e:
                st.caption(f"Sanity check skipped: {e}")

        if st.session_state.sim_complete:
            try:
                if FORECAST_CACHE_PATH.exists():
                    _, _, _, _, _, summary = _load_forecast_cache()
                else:
                    sim = _load_sim(SIM_PATH.stat().st_mtime if SIM_PATH.exists() else 0)
                    _, _, _, _, _, summary = _run_forecast_computation(sim)
                if not summary.empty:
                    st.subheader("Last Simulation — Top 3")
                    for _, row in summary.head(3).iterrows():
                        st.metric(row["Team"], f"{row['Mean_Points']:.1f} pts avg")
            except Exception as e:
                st.warning(f"Could not load previous results: {e}")
        else:
            st.info("Configure settings and click Run Simulation.")

    if st.session_state.sim_complete:
        _next_button("Forecast")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":
    _stepper()
    st.title("Step 4 — Forecast")

    if not st.session_state.sim_complete:
        _blocked("Simulate", "Run a simulation first to see the season forecast.")

    try:
        if FORECAST_CACHE_PATH.exists():
            table, champ, releg, europe, pos_probs, summary = _load_forecast_cache()
        else:
            # First time fallback: compute and save
            sim = _load_sim(SIM_PATH.stat().st_mtime if SIM_PATH.exists() else 0)
            table, champ, releg, europe, pos_probs, summary = _run_forecast_computation(sim)
            try:
                _save_forecast_cache()
            except Exception:
                pass
    except Exception as e:
        st.error(f"Could not load forecast: {e}")
        st.stop()

    n_teams = len(table)

    # ── Expected Final Table ──────────────────────────────────────────────────
    st.subheader("Expected Final Standings")

    try:
        all_results = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
        # Use the most recent season in the data, not a hardcoded calendar year,
        # so the filter works regardless of when the app is run.
        if "SeasonStart" in all_results.columns and not all_results.empty:
            latest_season = int(all_results["SeasonStart"].dropna().max())
            season_results = all_results[all_results["SeasonStart"] == latest_season].copy()
        else:
            season_results = all_results.copy()
    except Exception:
        season_results = pd.DataFrame()

    # Use upcoming_fixtures.csv — same file the simulator reads — so the counts
    # are consistent with what was actually simulated.
    _upcoming_path = Path("data/clean/upcoming_fixtures.csv")
    try:
        season_fixtures = pd.read_csv(
            _upcoming_path if _upcoming_path.exists() else FIXTURES_PATH,
            parse_dates=["Date"],
        )
        season_fixtures = _normalize_teams(season_fixtures)
    except Exception:
        season_fixtures = pd.DataFrame()

    def _games_for_team(team, results_df, fixtures_df):
        played   = int((results_df["HomeTeam"] == team).sum() + (results_df["AwayTeam"] == team).sum()) if not results_df.empty else 0
        upcoming = int((fixtures_df["HomeTeam"] == team).sum() + (fixtures_df["AwayTeam"] == team).sum()) if not fixtures_df.empty else 0
        return played + upcoming

    tbl = table.copy()
    tbl["GP"]           = tbl["Team"].map(lambda t: _games_for_team(t, season_results, season_fixtures))
    tbl["Title %"]      = tbl["Team"].map(lambda t: champ.get(t, 0) * 100)
    tbl["Europe %"]     = tbl["Team"].map(lambda t: europe.get(t, 0) * 100)
    tbl["Relegation %"] = tbl["Team"].map(lambda t: releg.get(t, 0) * 100)
    if not summary.empty:
        std_map = dict(zip(summary["Team"], summary["Std_Points"]))
        tbl["Pts ±"] = tbl["Team"].map(lambda t: std_map.get(t, 0))

    cols_order = ["Position", "Team", "GP", "Expected_Points"]
    if "Pts ±" in tbl.columns:
        cols_order.append("Pts ±")
    cols_order += ["Title %", "Europe %", "Relegation %"]
    tbl = tbl[cols_order].rename(columns={"Expected_Points": "Exp Pts"}).reset_index(drop=True)

    def _row_color(row):
        pos = int(row["Position"])
        if pos == 1:                          return ["background-color: #fffde7"] * len(row)  # CL
        if pos <= EUROPEAN_SPOTS:             return ["background-color: #e3f2fd"] * len(row)  # ECL
        if pos > n_teams - RELEGATION_SPOTS:  return ["background-color: #fce4ec"] * len(row)  # relegation
        return [""] * len(row)

    fmt = {"Exp Pts": "{:.1f}", "Title %": "{:.1f}%", "Europe %": "{:.1f}%", "Relegation %": "{:.1f}%"}
    if "Pts ±" in tbl.columns:
        fmt["Pts ±"] = "±{:.1f}"

    st.dataframe(tbl.style.apply(_row_color, axis=1).format(fmt), use_container_width=True, hide_index=True, height=(len(tbl) + 1) * 35 + 3)

    legend_cols = st.columns(3)
    legend_cols[0].caption("🟡 1st — Champions League")
    legend_cols[1].caption("🔵 2nd–3rd — Conference League")
    legend_cols[2].caption("🔴 Relegation zone")

    st.caption(
        "**GP** = games played this season + remaining fixtures.  "
        "**Exp Pts** = mean final points across all simulations (includes actual points already earned).  "
        "**± Pts** = standard deviation — how spread-out the point outcomes are.  "
        "**Title/Europe/Relegation %** = fraction of simulations where the team finished in that zone."
    )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    tab_title, tab_europe, tab_releg, tab_heat = st.tabs(
        ["Title Race", "European Qualification", "Relegation Battle", "Position Heatmap"]
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
            margin=dict(l=10, r=20, t=20, b=10),
            height=max(300, len(df) * 32), showlegend=False,
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
                showscale=True,
                colorbar=dict(title="Probability (%)"),
                hovertemplate="%{y} → position %{x}: %{z:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                xaxis_title="Final Position", yaxis_title="",
                height=max(400, n_pos * 36),
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Position probabilities not available.")

    _next_button("Predictions")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictions":
    _stepper()
    st.title("Step 5 — Predictions")

    if not st.session_state.model_trained:
        _blocked("Model", "Train the model first to generate fixture predictions.")

    fixtures = _load_fixtures()
    if fixtures.empty:
        st.info("No upcoming fixtures found. Download data for the current season on the Data page.")
        st.stop()

    try:
        model = _load_model()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    all_teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        team1 = fc1.selectbox("Team 1 (highlight)", ["All teams"] + all_teams, key="pred_team1")
    with fc2:
        team2_opts = ["All teams"] + [t for t in all_teams if t != team1] if team1 != "All teams" else ["All teams"] + all_teams
        team2 = fc2.selectbox("Team 2", team2_opts, key="pred_team2")
    with fc3:
        if "Date" in fixtures.columns and fixtures["Date"].notna().any():
            dates       = sorted(fixtures["Date"].dt.date.dropna().unique())
            date_filter = fc3.selectbox("Date", ["All dates"] + [str(d) for d in dates], key="pred_date")
        else:
            date_filter = "All dates"

    # ── Build predictions for ALL fixtures (cache stays valid across filter changes) ──
    @st.cache_data(show_spinner=False)
    def _predict_fixtures(fixture_records: tuple, model_mtime: float):
        rows = []
        for home, away, date in fixture_records:
            try:
                pred = model.predict_outcome_probabilities(home, away)
                rows.append({
                    "Date":     date,
                    "Home":     home,
                    "Away":     away,
                    "Home Win": pred["home_win"],
                    "Draw":     pred["draw"],
                    "Away Win": pred["away_win"],
                    "xG Home":  pred["mu_home"],
                    "xG Away":  pred["mu_away"],
                })
            except Exception:
                pass
        return rows

    _model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
    _all_records = tuple(
        (r["HomeTeam"], r["AwayTeam"], r["Date"].date() if pd.notna(r.get("Date")) else "—")
        for _, r in fixtures.iterrows()
    )
    all_rows = _predict_fixtures(_all_records, _model_mtime)

    if not all_rows:
        st.info("No predictions could be generated.")
        st.stop()

    pred_df = pd.DataFrame(all_rows)

    # Apply filters
    disp_df = pred_df.copy()
    if team1 != "All teams" and team2 != "All teams":
        disp_df = disp_df[
            ((disp_df["Home"] == team1) & (disp_df["Away"] == team2)) |
            ((disp_df["Home"] == team2) & (disp_df["Away"] == team1))
        ]
    elif team1 != "All teams":
        disp_df = disp_df[(disp_df["Home"] == team1) | (disp_df["Away"] == team1)]
    elif team2 != "All teams":
        disp_df = disp_df[(disp_df["Home"] == team2) | (disp_df["Away"] == team2)]
    if date_filter != "All dates":
        disp_df = disp_df[disp_df["Date"].astype(str) == date_filter]

    st.caption(f"Showing {len(disp_df)} fixture{'s' if len(disp_df) != 1 else ''}")

    if disp_df.empty:
        st.info("No fixtures match the selected filters.")
        st.stop()

    # ── Match cards (above the table) ─────────────────────────────────────────
    st.subheader("Match Details")
    cards_html = ["""<style>
.match-card {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #e0e0e0;
}
.match-team { flex: 1 1 30%; min-width: 80px; }
.match-team-away { text-align: right; }
.match-center { flex: 1 1 30%; min-width: 120px; text-align: center; }
@media (max-width: 480px) {
    .match-card { flex-direction: column; align-items: stretch; }
    .match-team, .match-team-away, .match-center { text-align: left; min-width: unset; }
}
</style>"""]
    for r in disp_df.to_dict("records"):
        hw   = float(r["Home Win"]) * 100
        dw   = float(r["Draw"])     * 100
        aw   = float(r["Away Win"]) * 100
        xgh  = float(r["xG Home"])
        xga  = float(r["xG Away"])
        t1_home = team1 != "All teams" and r["Home"] == team1
        t1_away = team1 != "All teams" and r["Away"] == team1
        hw_str = f"<strong>{hw:.0f}%</strong>" if t1_home else f"{hw:.0f}%"
        aw_str = f"<strong>{aw:.0f}%</strong>" if t1_away else f"{aw:.0f}%"
        dw_str = f"{dw:.0f}%"
        cards_html.append(f"""
<div class="match-card">
  <div class="match-team">
    <strong>{r['Home']}</strong><br>
    <span style="color:#888;font-size:0.8em">xG {xgh:.2f}</span>
  </div>
  <div class="match-center">
    <div style="color:#888;font-size:0.75em;margin-bottom:4px">{r['Date']}</div>
    <div style="font-size:1em;margin-bottom:6px">{hw_str} · {dw_str} · {aw_str}</div>
    <div style="display:flex;height:10px;border-radius:4px;overflow:hidden">
      <div style="width:{hw:.1f}%;background:#2196F3"></div>
      <div style="width:{dw:.1f}%;background:#9E9E9E"></div>
      <div style="width:{aw:.1f}%;background:#F44336"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.7em;color:#888;margin-top:2px">
      <span>Home</span><span>Draw</span><span>Away</span>
    </div>
  </div>
  <div class="match-team match-team-away">
    <strong>{r['Away']}</strong><br>
    <span style="color:#888;font-size:0.8em">xG {xga:.2f}</span>
  </div>
</div>""")
    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

    st.divider()

    # ── Table view ────────────────────────────────────────────────────────────
    st.subheader("All Fixtures")

    def _bold_team1_odds(row):
        styles = [""] * len(row)
        cols   = list(row.index)
        if team1 != "All teams":
            if row.get("Home") == team1 and "Home Win" in cols:
                styles[cols.index("Home Win")] = "font-weight: bold"
            elif row.get("Away") == team1 and "Away Win" in cols:
                styles[cols.index("Away Win")] = "font-weight: bold"
        return styles

    table_df = disp_df.copy()
    table_df["Home Win"] = table_df["Home Win"].map(lambda x: f"{x:.0%}")
    table_df["Draw"]     = table_df["Draw"].map(lambda x: f"{x:.0%}")
    table_df["Away Win"] = table_df["Away Win"].map(lambda x: f"{x:.0%}")
    table_df["xG Home"]  = table_df["xG Home"].map(lambda x: f"{x:.2f}")
    table_df["xG Away"]  = table_df["xG Away"].map(lambda x: f"{x:.2f}")
    styled = table_df.style.apply(_bold_team1_odds, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPDATE EVERYTHING  (state-machine: each step is a separate rerun so that
# the Stop button — which queues a rerun — takes effect between steps)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Update":
    # ── Session-state defaults for this page ─────────────────────────────────
    for _k, _v in {
        "upd_step":  None,   # None | "data" | "model" | "sim" | "done"
        "upd_stop":  False,
        "upd_log":   [],     # list of dicts {label, state, msg}
    }.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    upd_step = st.session_state.upd_step
    upd_stop = st.session_state.upd_stop

    st.title("🔄 Update Everything")

    # ── Completed-step log (persists across reruns) ───────────────────────────
    for entry in st.session_state.upd_log:
        if entry["state"] == "complete":
            st.success(f"✅ {entry['label']} — {entry['msg']}")
        elif entry["state"] == "skipped":
            st.info(f"⏭ {entry['label']} — {entry['msg']}")
        elif entry["state"] == "error":
            st.error(f"❌ {entry['label']} — {entry['msg']}")
        elif entry["state"] == "stopped":
            st.warning(f"⏹ {entry['label']} — {entry['msg']}")

    # ── Idle — show start UI ──────────────────────────────────────────────────
    if upd_step is None:
        st.caption("Fetch fresh data → train model → 10 000 simulations → Predictions")

        current_year   = pd.Timestamp.now().year
        data_age_hours = None
        if RESULTS_PATH.exists():
            try:
                mtime          = RESULTS_PATH.stat().st_mtime
                data_age_hours = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(mtime)).total_seconds() / 3600
            except Exception:
                pass

        # history_ok: accept either a dedicated historical file OR results.csv
        # that already contains multiple seasons (i.e. historical was merged in previously).
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
                    history_ok = len(_r) > 100  # enough rows to imply historical data
            except Exception:
                pass

        if data_age_hours is not None:
            if data_age_hours < 1:
                st.success(f"Data is fresh ({data_age_hours * 60:.0f} min ago) — fetch will be skipped.")
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

    # ── Done or stopped ───────────────────────────────────────────────────────
    elif upd_step == "done":
        if upd_stop:
            st.warning("Stopped — completed steps have been saved.")
        else:
            st.success("All steps complete!")

        col_a, col_b = st.columns(2)
        if col_a.button("Go to Predictions", type="primary"):
            st.session_state.upd_step = None
            _nav("Predictions")
        if col_b.button("Run again"):
            st.session_state.upd_step = None
            st.session_state.upd_log  = []
            st.rerun()

    # ── Running a step ────────────────────────────────────────────────────────
    else:
        # Stop button — sets flag; takes effect after the current step finishes
        if st.button("⏹ Stop after this step", type="secondary"):
            st.session_state.upd_stop = True
            st.rerun()

        current_year = pd.Timestamp.now().year

        def _log(label, state, msg):
            st.session_state.upd_log.append({"label": label, "state": state, "msg": msg})

        def _advance(next_step):
            """Move to next step, or to done if stop was requested."""
            st.session_state.upd_step = "done" if st.session_state.upd_stop else next_step
            st.rerun()

        # ── Step 1: Data ──────────────────────────────────────────────────────
        if upd_step == "data":
            data_age_hours = None
            if RESULTS_PATH.exists():
                try:
                    mtime          = RESULTS_PATH.stat().st_mtime
                    data_age_hours = (pd.Timestamp.now() - pd.Timestamp.fromtimestamp(mtime)).total_seconds() / 3600
                except Exception:
                    pass

            skip_fetch = data_age_hours is not None and data_age_hours < 1

            st.subheader("Step 1/3 — Match data")
            bar  = st.progress(0, text="Starting…")

            if skip_fetch:
                bar.progress(100, text="Skipped (data < 1h old)")
                _log("Step 1/3 — Data", "skipped",
                     f"data is {data_age_hours * 60:.0f} min old (< 1 hour)")
                _advance("model")

            try:
                bar.progress(10, text="Connecting to football-data.co.uk…")
                scraper = AllsvenskanScraper()
                raw = scraper.scrape_matches(seasons=[current_year])
                if raw.empty:
                    raise RuntimeError("No data returned from source")

                bar.progress(40, text="Cleaning and normalising team names…")
                cleaner      = DataCleaner()
                cur_results, cur_fixtures = cleaner.clean_data(raw)
                cur_results  = _normalize_teams(cur_results)
                cur_fixtures = _normalize_teams(cur_fixtures)

                # Warn on any unexpected Djurgården variant
                all_fix_teams = pd.concat([cur_fixtures["HomeTeam"], cur_fixtures["AwayTeam"]]).unique()
                bad_djurg = [t for t in all_fix_teams
                             if ("jurg" in str(t).lower() or "ård" in str(t).lower())
                             and t != "Djurgarden"]
                if bad_djurg:
                    st.warning(f"Unexpected Djurgården variant after normalisation: {bad_djurg}")

                bar.progress(70, text="Merging with historical data…")
                hist_df  = _normalize_teams(pd.read_csv(HISTORICAL_PATH, parse_dates=["Date"]))
                combined = pd.concat([hist_df, cur_results], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["Date", "HomeTeam", "AwayTeam"], keep="last"
                )

                bar.progress(90, text="Saving…")
                RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                FIXTURES_PATH.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(RESULTS_PATH, index=False)
                cur_fixtures.to_csv(FIXTURES_PATH, index=False)
                cur_fixtures.to_csv("data/clean/upcoming_fixtures.csv", index=False)
                st.session_state.data_loaded = True

                bar.progress(100, text="Done")
                _log("Step 1/3 — Data", "complete",
                     f"{len(cur_results)} matches fetched · {len(cur_fixtures)} upcoming fixtures")
                _advance("model")

            except Exception as e:
                bar.progress(100, text="Failed")
                _log("Step 1/3 — Data", "error", str(e))
                st.session_state.upd_step = "done"
                st.rerun()

        # ── Step 2: Model ─────────────────────────────────────────────────────
        elif upd_step == "model":
            st.subheader("Step 2/3 — Training model")
            bar = st.progress(0, text="Loading data…")

            try:
                results = _load_results()

                bar.progress(20, text="Calculating team strengths…")
                strength_calc = TeamStrengthCalculator(use_odds_integration=False)
                team_stats    = strength_calc.calculate_strengths(results)
                TEAM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
                team_stats.to_csv(TEAM_STATS_PATH)

                bar.progress(50, text="Fitting Dixon-Coles model…")
                model = PoissonModel()
                model.fit(results, team_stats)

                bar.progress(90, text="Saving model…")
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(MODEL_PATH))
                st.session_state.model_trained = True

                n_teams = len(model.attack_rates)
                bar.progress(100, text="Done")
                _log("Step 2/3 — Model", "complete",
                     f"{n_teams} teams · {len(results)} matches")
                _advance("sim")

            except Exception as e:
                bar.progress(100, text="Failed")
                _log("Step 2/3 — Model", "error", str(e))
                st.session_state.upd_step = "done"
                st.rerun()

        # ── Step 3: Simulate ──────────────────────────────────────────────────
        elif upd_step == "sim":
            st.subheader("Step 3/3 — Simulating 10 000 seasons")
            bar = st.progress(0, text="Loading model and fixtures…")

            try:
                model     = _load_model()
                simulator = MonteCarloSimulator.from_upcoming_fixtures(model)

                try:
                    _all_res = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
                    if "SeasonStart" in _all_res.columns and not _all_res.empty:
                        _latest  = int(_all_res["SeasonStart"].dropna().max())
                        _cur_res = _all_res[_all_res["SeasonStart"] == _latest]
                    else:
                        _cur_res = pd.DataFrame()
                    _stnd       = _standings_from_results(_cur_res) if not _cur_res.empty else pd.DataFrame()
                    current_pts = dict(zip(_stnd["Team"], _stnd["Pts"])) if not _stnd.empty else {}
                except Exception:
                    current_pts = {}

                def _sim_cb(pct):
                    bar.progress(int(pct), text=f"Simulating… {pct:.0f}%")

                bar.progress(5, text="Starting simulations…")
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

                bar.progress(98, text="Saving results…")
                SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
                sim_results.to_csv(SIM_PATH, index=False)
                bar.progress(99, text="Computing forecast cache…")
                try:
                    _save_forecast_cache()
                except Exception as _fe:
                    st.warning(f"Forecast cache failed: {_fe}")
                st.session_state.sim_complete = True

                bar.progress(100, text="Done")
                _log("Step 3/3 — Simulate", "complete", "10 000 simulations complete")
                st.session_state.upd_step = "done"
                st.rerun()

            except Exception as e:
                bar.progress(100, text="Failed")
                _log("Step 3/3 — Simulate", "error", str(e))
                st.session_state.upd_step = "done"
                st.rerun()
