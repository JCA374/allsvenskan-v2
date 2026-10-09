"""Centralized configuration: file paths and league constants."""
from pathlib import Path

# Project root — works whether imported from project root or a subdirectory.
ROOT = Path(__file__).resolve().parent.parent

# ── File paths ────────────────────────────────────────────────────────────────
RESULTS_PATH    = ROOT / "data/clean/results.csv"
UPCOMING_PATH   = ROOT / "data/clean/upcoming_fixtures.csv"
HISTORICAL_PATH = ROOT / "data/clean/historical_results.csv"
SITE_DATA_PATH  = ROOT / "site/data.json"

# ── League constants ──────────────────────────────────────────────────────────
GAMES_PER_TEAM = 30   # 16 teams × 2 meetings = (16-1)*2 = 30
