"""Export results + remaining fixtures as one compact JSON file for the static site.

The browser fits the model and runs the simulation itself (site/app.js); this
script only ships the raw data it needs.

Usage:
    python scripts/export_site_data.py           # export from data/clean/*.csv
    python scripts/export_site_data.py --fetch   # fetch fresh data first
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ROOT, RESULTS_PATH, UPCOMING_PATH, GAMES_PER_TEAM
from core.utils.helpers import normalize_team_names

OUT_PATH = ROOT / "site/data.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="fetch fresh data before exporting")
    args = parser.parse_args()

    if args.fetch:
        from scripts.daily_update import step_fetch_data
        step_fetch_data()

    results = normalize_team_names(pd.read_csv(RESULTS_PATH, parse_dates=["Date"]))
    results = results.dropna(subset=["FTHG", "FTAG"]).sort_values("Date")
    fixtures = normalize_team_names(pd.read_csv(UPCOMING_PATH, parse_dates=["Date"]))
    fixtures = fixtures.sort_values("Date")

    season = int(results["Date"].dt.year.max())
    teams = sorted(set(results["HomeTeam"]) | set(results["AwayTeam"])
                   | set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}

    data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "season": season,
        "gamesPerTeam": GAMES_PER_TEAM,
        "teams": teams,
        # [date, home, away, home goals, away goals]
        "results": [
            [r.Date.strftime("%Y-%m-%d"), idx[r.HomeTeam], idx[r.AwayTeam], int(r.FTHG), int(r.FTAG)]
            for r in results.itertuples()
        ],
        # [date, home, away]
        "fixtures": [
            [r.Date.strftime("%Y-%m-%d"), idx[r.HomeTeam], idx[r.AwayTeam]]
            for r in fixtures.itertuples()
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}: {len(data['results'])} results, "
          f"{len(data['fixtures'])} fixtures, {OUT_PATH.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
