"""
Sanity-check that results + fixtures always total 30 GP per team.

This test runs against the live data source (football-data.co.uk / ESPN)
so it doubles as a pre-deployment gate: if the upstream data is broken
or the cleaner introduces a mismatch, this test will fail.
"""

import pandas as pd
import pytest

from core.data.scraper import AllsvenskanScraper
from core.data.cleaner import DataCleaner
from core.utils.helpers import (
    GAMES_PER_TEAM,
    normalize_team_names,
    validate_games_per_team,
)


# ── Unit tests (fast, no network) ────────────────────────────────────────────

def _make_round_robin(n_teams=16):
    """Generate a perfect round-robin fixture list (home+away) for n teams."""
    teams = [f"Team{i}" for i in range(1, n_teams + 1)]
    rows = []
    for h in teams:
        for a in teams:
            if h != a:
                rows.append({"HomeTeam": h, "AwayTeam": a, "Date": "2026-05-01"})
    return pd.DataFrame(rows)


def test_validate_all_correct():
    """When results + fixtures = 30 GP for all teams, no errors."""
    full = _make_round_robin()
    # Split: first 74 are results, rest are fixtures
    results = full.iloc[:74].copy()
    fixtures = full.iloc[74:].copy()
    bad = validate_games_per_team(results, fixtures, GAMES_PER_TEAM)
    assert bad == [], f"Expected no issues, got: {bad}"


def test_validate_detects_missing_games():
    """Dropping a fixture should be caught."""
    full = _make_round_robin()
    results = full.iloc[:74].copy()
    fixtures = full.iloc[74:-1].copy()  # drop last fixture
    bad = validate_games_per_team(results, fixtures, GAMES_PER_TEAM)
    assert len(bad) > 0, "Should detect teams with != 30 games"
    for team, played, upcoming, total in bad:
        assert total == GAMES_PER_TEAM - 1


def test_validate_detects_extra_games():
    """Duplicating a fixture should be caught."""
    full = _make_round_robin()
    results = full.iloc[:74].copy()
    fixtures = full.iloc[74:].copy()
    # Duplicate one fixture
    extra = fixtures.iloc[[0]].copy()
    fixtures = pd.concat([fixtures, extra], ignore_index=True)
    bad = validate_games_per_team(results, fixtures, GAMES_PER_TEAM)
    assert len(bad) > 0, "Should detect teams with > 30 games"


def test_validate_empty_fixtures():
    """All 30 games played, zero fixtures — should pass."""
    full = _make_round_robin()
    empty = pd.DataFrame(columns=["HomeTeam", "AwayTeam", "Date"])
    bad = validate_games_per_team(full, empty, GAMES_PER_TEAM)
    assert bad == []


# ── Integration test (hits network) ──────────────────────────────────────────

@pytest.mark.network
def test_live_data_games_per_team():
    """Fetch current season from upstream and verify 30 GP per team."""
    current_year = pd.Timestamp.now().year
    scraper = AllsvenskanScraper()
    raw = scraper.scrape_matches(seasons=[current_year])
    assert not raw.empty, f"No data returned for {current_year}"

    cleaner = DataCleaner()
    results, fixtures = cleaner.clean_data(raw)
    results = normalize_team_names(results)
    fixtures = normalize_team_names(fixtures)

    bad = validate_games_per_team(results, fixtures, GAMES_PER_TEAM)
    assert bad == [], (
        f"GP mismatch for {len(bad)} team(s): "
        + ", ".join(f"{t}={tot}" for t, _, _, tot in bad)
    )
