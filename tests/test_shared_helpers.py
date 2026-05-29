"""Tests for the shared helper functions introduced during refactoring:
normalize_team_names(), build_standings(), and core.config constants.
"""
import pandas as pd
import pytest

from core.config import (
    GAMES_PER_TEAM, RELEGATION_SPOTS, EUROPEAN_SPOTS,
    RESULTS_PATH, FIXTURES_PATH, MODEL_PATH, ROOT,
)
from core.utils.helpers import (
    GAMES_PER_TEAM as HELPERS_GAMES_PER_TEAM,
    TEAM_NAME_MAP,
    normalize_team_name,
    normalize_team_names,
    build_standings,
)


# ── core.config ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_constants_have_expected_values(self):
        assert GAMES_PER_TEAM == 30
        assert RELEGATION_SPOTS == 3
        assert EUROPEAN_SPOTS == 3

    def test_paths_are_under_root(self):
        for path in (RESULTS_PATH, FIXTURES_PATH, MODEL_PATH):
            assert str(path).startswith(str(ROOT)), f"{path} not under ROOT {ROOT}"

    def test_helpers_reexport_matches_config(self):
        """GAMES_PER_TEAM from helpers should be the same object as from config."""
        assert HELPERS_GAMES_PER_TEAM == GAMES_PER_TEAM


# ── normalize_team_names ─────────────────────────────────────────────────────

class TestNormalizeTeamNames:
    def test_normalizes_swedish_chars(self):
        df = pd.DataFrame({
            "HomeTeam": ["Djurgårdens IF", "Malmö FF"],
            "AwayTeam": ["IFK Göteborg", "BK Häcken"],
        })
        result = normalize_team_names(df)
        assert result["HomeTeam"].tolist() == ["Djurgarden", "Malmo FF"]
        assert result["AwayTeam"].tolist() == ["Goteborg", "Hacken"]

    def test_leaves_canonical_names_unchanged(self):
        df = pd.DataFrame({
            "HomeTeam": ["AIK", "Elfsborg"],
            "AwayTeam": ["Hammarby", "Kalmar"],
        })
        result = normalize_team_names(df)
        assert result["HomeTeam"].tolist() == ["AIK", "Elfsborg"]
        assert result["AwayTeam"].tolist() == ["Hammarby", "Kalmar"]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"HomeTeam": ["Malmö FF"], "AwayTeam": ["AIK"]})
        original_value = df["HomeTeam"].iloc[0]
        normalize_team_names(df)
        assert df["HomeTeam"].iloc[0] == original_value

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"SomeOther": ["value"]})
        result = normalize_team_names(df)
        assert "SomeOther" in result.columns

    def test_strips_whitespace(self):
        df = pd.DataFrame({
            "HomeTeam": ["  AIK  "],
            "AwayTeam": [" Malmö FF "],
        })
        result = normalize_team_names(df)
        assert result["HomeTeam"].iloc[0] == "AIK"
        assert result["AwayTeam"].iloc[0] == "Malmo FF"

    def test_consistent_with_single_normalize(self):
        """normalize_team_names on a DF should match normalize_team_name per cell."""
        names = ["Djurgårdens IF", "IFK Göteborg", "AIK", "Malmö FF"]
        df = pd.DataFrame({"HomeTeam": names, "AwayTeam": names})
        result = normalize_team_names(df)
        for i, name in enumerate(names):
            assert result["HomeTeam"].iloc[i] == normalize_team_name(name)


# ── build_standings ──────────────────────────────────────────────────────────

def _make_results():
    """3 matches between 3 teams: A beats B, B beats C, A draws C."""
    return pd.DataFrame({
        "HomeTeam": ["A", "B", "A"],
        "AwayTeam": ["B", "C", "C"],
        "FTHG":     [2,   3,   1],
        "FTAG":     [1,   0,   1],
    })


class TestBuildStandings:
    def test_returns_correct_columns(self):
        standings = build_standings(_make_results())
        expected_cols = {"Team", "GP", "W", "D", "L", "GF", "GA", "GD", "Pts"}
        assert set(standings.columns) == expected_cols

    def test_points_calculation(self):
        standings = build_standings(_make_results())
        pts = dict(zip(standings["Team"], standings["Pts"]))
        # A: win + draw = 3+1 = 4
        assert pts["A"] == 4
        # B: loss + win = 0+3 = 3
        assert pts["B"] == 3
        # C: loss + draw = 0+1 = 1
        assert pts["C"] == 1

    def test_games_played(self):
        standings = build_standings(_make_results())
        gp = dict(zip(standings["Team"], standings["GP"]))
        assert gp["A"] == 2
        assert gp["B"] == 2
        assert gp["C"] == 2

    def test_goal_difference(self):
        standings = build_standings(_make_results())
        gd = dict(zip(standings["Team"], standings["GD"]))
        # A: scored 3 (2+1), conceded 2 (1+1) → GD = +1
        assert gd["A"] == 1
        # B: scored 4 (1+3), conceded 2 (2+0) → GD = +2
        assert gd["B"] == 2
        # C: scored 1 (0+1), conceded 4 (3+1) → GD = -3
        assert gd["C"] == -3

    def test_sorted_by_points_then_gd(self):
        standings = build_standings(_make_results())
        teams = standings["Team"].tolist()
        # A has 4 pts (1st), B has 3 pts (2nd), C has 1 pt (3rd)
        assert teams == ["A", "B", "C"]

    def test_win_draw_loss_counts(self):
        standings = build_standings(_make_results())
        row_a = standings[standings["Team"] == "A"].iloc[0]
        assert row_a["W"] == 1
        assert row_a["D"] == 1
        assert row_a["L"] == 0

    def test_full_round_robin(self):
        """A complete 4-team round-robin: each team plays 6 games."""
        teams = ["W", "X", "Y", "Z"]
        rows = []
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append({"HomeTeam": h, "AwayTeam": a, "FTHG": 1, "FTAG": 0})
        results = pd.DataFrame(rows)
        standings = build_standings(results)
        # Every team plays 6 (3 home wins, 3 away losses)
        for _, row in standings.iterrows():
            assert row["GP"] == 6
            assert row["W"] == 3   # all home wins
            assert row["L"] == 3   # all away losses
            assert row["Pts"] == 9


# ── Import smoke tests ─────────────────────────────────────────────────────

class TestEntryPointImports:
    """Verify that all entry points can import their dependencies.

    These catch missing packages in requirements.txt or broken import
    chains before they reach CI or Streamlit Cloud.
    """

    def test_daily_update_imports(self):
        """scripts/daily_update.py imports should resolve."""
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
        from core.utils.helpers import (
            TEAM_NAME_MAP, validate_games_per_team,
            normalize_team_names, build_standings,
        )

    def test_app_imports(self):
        """app.py imports should resolve."""
        pytest.importorskip("streamlit")
        from core.config import RESULTS_PATH, MODEL_PATH, SIM_PATH
        from core.ui.helpers import STEPS, nav, step_done

    def test_cli_imports(self):
        """cli.py imports should resolve."""
        from core.config import (
            RESULTS_PATH, FIXTURES_PATH, TEAM_STATS_PATH, MODEL_PATH,
        )
        from core.data.scraper import AllsvenskanScraper
        from core.data.cleaner import DataCleaner
