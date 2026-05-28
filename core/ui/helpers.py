"""Shared helpers for the Streamlit UI pages."""
import pickle

import pandas as pd
import streamlit as st

from core.config import (
    RESULTS_PATH, FIXTURES_PATH, HISTORICAL_PATH,
    MODEL_PATH, SIM_PATH, FORECAST_CACHE_PATH,
    GAMES_PER_TEAM, RELEGATION_SPOTS, EUROPEAN_SPOTS,
)
from core.models.poisson_model import PoissonModel
from core.analysis.aggregator import ResultsAggregator
from core.utils.helpers import TEAM_NAME_MAP, normalize_team_names, build_standings

# ── Pipeline steps definition ────────────────────────────────────────────────
STEPS = [
    ("Data",        "\U0001f5c4\ufe0f",  "data_loaded"),
    ("Model",       "\U0001f9e0",  "model_trained"),
    ("Simulate",    "\U0001f3b2",  "sim_complete"),
    ("Forecast",    "\U0001f4ca",  "sim_complete"),
    ("Predictions", "\U0001f4c5",  "model_trained"),
]


def nav(page: str):
    st.session_state.active_page = page


def step_done(key: str) -> bool:
    return bool(st.session_state.get(key, False))


def stepper():
    """Render a compact progress strip at the top of each page."""
    parts = []
    page = st.session_state.active_page
    for i, (name, icon, gate) in enumerate(STEPS, 1):
        done = step_done(gate)
        active = page == name
        if active:
            parts.append(f"**\u25b6 {icon} {name}**")
        elif done:
            parts.append(f"\u2705 {name}")
        else:
            parts.append(f"<span style='color:#aaa'>{i}. {name}</span>")
    st.markdown("&nbsp; \u00b7 &nbsp;".join(parts), unsafe_allow_html=True)
    st.divider()


def next_button(next_page: str):
    """Render a 'Next step' button at the bottom of a page."""
    st.divider()
    if st.button(f"Next \u2192 {next_page}", type="primary"):
        nav(next_page)


def blocked(required_page: str, message: str):
    """Show a friendly blocker and stop rendering."""
    st.info(message)
    if st.button(f"Go to {required_page}"):
        nav(required_page)
    st.stop()


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH, parse_dates=["Date"])
    df = normalize_team_names(df)
    return df[df["FTHG"].notna() & df["FTAG"].notna()].copy()


def load_fixtures() -> pd.DataFrame:
    if FIXTURES_PATH.exists():
        df = pd.read_csv(FIXTURES_PATH, parse_dates=["Date"])
        if "FTHG" in df.columns:
            df = df[df["FTHG"].isna()]
        df = normalize_team_names(df[["Date", "HomeTeam", "AwayTeam"]].copy())
        return df
    return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])


def load_model() -> PoissonModel:
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, PoissonModel):
        return data
    m = PoissonModel()
    m.load(str(MODEL_PATH))
    return m


@st.cache_data
def load_sim(_mtime: float = 0) -> pd.DataFrame:
    df = pd.read_csv(SIM_PATH)
    rename_map = {col: TEAM_NAME_MAP.get(col, col) for col in df.columns}
    df = df.rename(columns=rename_map)
    df = df.T.groupby(level=0).sum().T
    return df


def run_forecast_computation(sim: pd.DataFrame):
    """Pure computation -- no caching."""
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


def save_forecast_cache():
    """Compute forecast from current sim file and persist to disk."""
    sim = pd.read_csv(SIM_PATH)
    rename_map = {col: TEAM_NAME_MAP.get(col, col) for col in sim.columns}
    sim = sim.rename(columns=rename_map).T.groupby(level=0).sum().T
    result = run_forecast_computation(sim)
    from datetime import datetime
    updated_at = datetime.now()
    FORECAST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FORECAST_CACHE_PATH, "wb") as f:
        pickle.dump((*result, updated_at), f)
    return (*result, updated_at)


def load_forecast_cache():
    """Load pre-computed forecast from disk."""
    with open(FORECAST_CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    if len(data) == 6:
        return (*data, None)
    return data


def color_table_row(row, n_teams):
    pos = row.name + 1
    if pos == 1:                          return ["background-color: #d4edda"] * len(row)
    if pos <= EUROPEAN_SPOTS:             return ["background-color: #e3f2fd"] * len(row)
    if pos > n_teams - RELEGATION_SPOTS:  return ["background-color: #fce4ec"] * len(row)
    return [""] * len(row)
