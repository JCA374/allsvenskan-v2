"""Fetch fresh match data and save it to data/clean/."""
import pandas as pd

from core.config import GAMES_PER_TEAM, HISTORICAL_PATH, RESULTS_PATH, UPCOMING_PATH
from core.data.cleaner import DataCleaner
from core.data.scraper import AllsvenskanScraper
from core.utils.helpers import normalize_team_names, validate_games_per_team


def fetch_and_save():
    """Fetch the current season, merge it into results.csv and save the remaining fixtures.

    Raises RuntimeError if the source returns nothing, the merged results would
    shrink, or any team does not add up to GAMES_PER_TEAM games.
    """
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

    for p in (RESULTS_PATH, UPCOMING_PATH):
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
    cur_fixtures.to_csv(UPCOMING_PATH, index=False)
    print(f"  {len(cur_results)} matches this season · {len(cur_fixtures)} upcoming · {len(combined)} total saved")
    return combined, cur_fixtures
