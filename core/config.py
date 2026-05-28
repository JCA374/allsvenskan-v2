"""Centralized configuration: file paths and league constants.

Every entry point (app.py, cli.py, scripts/daily_update.py) should import
paths and constants from here so they stay in sync.
"""
from pathlib import Path

# Project root — works whether imported from project root or a subdirectory.
ROOT = Path(__file__).resolve().parent.parent

# ── File paths ────────────────────────────────────────────────────────────────
RESULTS_PATH        = ROOT / "data/clean/results.csv"
FIXTURES_PATH       = ROOT / "data/clean/fixtures.csv"
UPCOMING_PATH       = ROOT / "data/clean/upcoming_fixtures.csv"
HISTORICAL_PATH     = ROOT / "data/clean/historical_results.csv"
TEAM_STATS_PATH     = ROOT / "data/processed/team_stats.csv"
MODEL_PATH          = ROOT / "models/poisson_params.pkl"
SIM_PATH            = ROOT / "reports/simulations/sim_results_latest.csv"
FORECAST_CACHE_PATH = ROOT / "reports/simulations/forecast_cache.pkl"

# ── League constants ──────────────────────────────────────────────────────────
GAMES_PER_TEAM   = 30   # 16 teams × 2 meetings = (16-1)*2 = 30
RELEGATION_SPOTS = 3
EUROPEAN_SPOTS   = 3    # 1st: CL qualifying · 2nd–3rd: ECL qualifying
