#!/usr/bin/env python3
"""
Daily update script — mirrors the app's "Update Everything" pipeline.
Saves to the exact file paths the Streamlit app expects, including the
pre-computed forecast cache so the Forecast page loads instantly.

Run: python scripts/daily_update.py
"""
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import (
    HISTORICAL_PATH, RESULTS_PATH, FIXTURES_PATH, UPCOMING_PATH,
    TEAM_STATS_PATH, MODEL_PATH, SIM_PATH, FORECAST_CACHE_PATH,
    GAMES_PER_TEAM, RELEGATION_SPOTS, EUROPEAN_SPOTS,
)
from core.data.scraper import AllsvenskanScraper
from core.data.cleaner import DataCleaner
from core.data.strength import TeamStrengthCalculator
from core.models.poisson_model import PoissonModel
from core.simulation.simulator import MonteCarloSimulator
from core.analysis.aggregator import ResultsAggregator
from core.utils.helpers import TEAM_NAME_MAP, validate_games_per_team, normalize_team_names, build_standings


def step_fetch_data():
    print("── Step 1: Fetch current season ──────────────────────────────────")
    current_year = pd.Timestamp.now().year
    scraper = AllsvenskanScraper()
    raw = scraper.scrape_matches(seasons=[current_year])
    if raw.empty:
        raise RuntimeError(f"No data returned for {current_year}")

    cleaner = DataCleaner()
    cur_results, cur_fixtures = cleaner.clean_data(raw)
    cur_results  = normalize_team_names(cur_results)
    cur_fixtures = normalize_team_names(cur_fixtures)

    # Build combined results: merge all available historical sources with fresh data.
    # Priority: historical_results.csv > existing results.csv > current season only.
    # Safety: never let results.csv shrink (protects against source outages).
    base_df = pd.DataFrame()
    if HISTORICAL_PATH.exists():
        base_df = normalize_team_names(pd.read_csv(HISTORICAL_PATH, parse_dates=["Date"]))
        print(f"  Historical base: {len(base_df)} rows from historical_results.csv")
    if RESULTS_PATH.exists():
        existing = normalize_team_names(pd.read_csv(RESULTS_PATH, parse_dates=["Date"]))
        if len(existing) > len(base_df):
            base_df = existing
            print(f"  Historical base: {len(base_df)} rows from results.csv (larger)")
        elif base_df.empty:
            base_df = existing
            print(f"  Historical base: {len(base_df)} rows from results.csv")

    if not base_df.empty:
        combined = pd.concat([base_df, cur_results], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
    else:
        combined = cur_results
        print("  WARNING: no historical data found — using current season only")

    # Safety check: never shrink results.csv
    if RESULTS_PATH.exists():
        existing_len = len(pd.read_csv(RESULTS_PATH))
        if len(combined) < existing_len * 0.9:
            print(f"  ERROR: combined ({len(combined)}) is much smaller than existing ({existing_len}) — aborting save")
            raise RuntimeError("Refusing to overwrite results.csv with fewer rows (data loss protection)")

    for p in (RESULTS_PATH, FIXTURES_PATH, UPCOMING_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Sanity check: every team must have exactly 30 GP (results + fixtures)
    bad = validate_games_per_team(cur_results, cur_fixtures, GAMES_PER_TEAM)
    if bad:
        for team, played, upcoming, total in bad:
            print(f"  WARNING: {team}: {played} played + {upcoming} upcoming = {total} (expected {GAMES_PER_TEAM})")
        raise RuntimeError(
            f"GP sanity check failed — {len(bad)} team(s) do not have {GAMES_PER_TEAM} games"
        )

    combined.to_csv(RESULTS_PATH, index=False)
    cur_fixtures.to_csv(FIXTURES_PATH, index=False)
    cur_fixtures.to_csv(UPCOMING_PATH, index=False)
    print(f"  {len(cur_results)} matches this season · {len(cur_fixtures)} upcoming · {len(combined)} total saved")
    return combined, cur_fixtures


def step_train_model(results: pd.DataFrame):
    print("── Step 2: Train model ────────────────────────────────────────────")
    strength_calc = TeamStrengthCalculator(use_odds_integration=False)
    team_stats    = strength_calc.calculate_strengths(results)
    TEAM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_stats.to_csv(TEAM_STATS_PATH)

    model = PoissonModel()
    model.fit(results, team_stats)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"  Dixon-Coles trained on {len(results)} matches · {len(model.attack_rates)} teams")
    return model


def step_simulate(model: PoissonModel, results: pd.DataFrame):
    print("── Step 3: Simulate ───────────────────────────────────────────────")
    simulator = MonteCarloSimulator.from_upcoming_fixtures(model)

    # Seed with actual current standings
    current_pts: dict = {}
    try:
        if "SeasonStart" in results.columns and not results.empty:
            latest   = int(results["SeasonStart"].dropna().max())
            cur      = results[results["SeasonStart"] == latest]
            standings = build_standings(cur)
            current_pts = dict(zip(standings["Team"], standings["Pts"]))
    except Exception as e:
        print(f"  WARNING: could not build standings seed: {e}")

    def _cb(pct):
        print(f"  {pct:.0f}%", end="\r", flush=True)

    if current_pts:
        sim_results = simulator.run_monte_carlo_with_standings(
            n_simulations=10_000,
            current_standings=current_pts,
            progress_callback=_cb,
        )
    else:
        sim_results = simulator.run(n_simulations=10_000, progress_callback=_cb)

    print()
    SIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    sim_results.to_csv(SIM_PATH, index=False)
    print(f"  10 000 simulations saved → {SIM_PATH}")
    return sim_results


def step_compute_forecast(sim_results: pd.DataFrame):
    print("── Step 4: Compute forecast cache ────────────────────────────────")
    rename_map = {col: TEAM_NAME_MAP.get(col, col) for col in sim_results.columns}
    sim = sim_results.rename(columns=rename_map).T.groupby(level=0).sum().T

    agg       = ResultsAggregator()
    table     = agg.generate_final_table_prediction(sim)
    champ     = agg.calculate_championship_odds(sim)
    releg     = agg.calculate_relegation_odds(sim, relegation_spots=RELEGATION_SPOTS)
    europe    = agg.calculate_european_qualification_odds(sim, european_spots=EUROPEAN_SPOTS)
    pos_probs = agg.calculate_position_probabilities(sim)
    summary   = agg.analyze_results(sim)

    updated_at = datetime.now()
    cache = (table, champ, releg, europe, pos_probs, summary, updated_at)
    FORECAST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FORECAST_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Forecast cache saved → {FORECAST_CACHE_PATH}")
    return cache


def main():
    print("🚀 Allsvenskan daily update\n")
    try:
        results, _ = step_fetch_data()
        model       = step_train_model(results)
        sim_results = step_simulate(model, results)
        step_compute_forecast(sim_results)
        print("\n✅ Done!")
    except Exception as e:
        import traceback
        print(f"\n❌ Update failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
