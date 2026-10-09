"""Tests for core.utils.helpers and core.config."""
import pandas as pd

from core.config import GAMES_PER_TEAM, RESULTS_PATH, UPCOMING_PATH, SITE_DATA_PATH, ROOT
from core.utils.helpers import (
    normalize_team_name,
    normalize_team_names,
    drop_playoff_matches,
)


# ── core.config ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_games_per_team(self):
        assert GAMES_PER_TEAM == 30

    def test_paths_are_under_root(self):
        for path in (RESULTS_PATH, UPCOMING_PATH, SITE_DATA_PATH):
            assert str(path).startswith(str(ROOT)), f"{path} not under ROOT {ROOT}"


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


class TestOsterAlias:
    def test_osters_maps_to_oster(self):
        assert normalize_team_name("Osters") == "Oster"


# ── drop_playoff_matches ─────────────────────────────────────────────────────

class TestDropPlayoffMatches:
    def test_drops_both_playoff_legs(self):
        # A and B play a full 12-match "season"; C only appears in a two-leg playoff vs B.
        dates = pd.date_range("2024-04-01", periods=14, freq="7D")
        df = pd.DataFrame({
            "Date": dates,
            "HomeTeam": ["A", "B"] * 6 + ["C", "B"],
            "AwayTeam": ["B", "A"] * 6 + ["B", "C"],
        })
        out = drop_playoff_matches(df)
        assert len(out) == 12
        assert "C" not in set(out["HomeTeam"]) | set(out["AwayTeam"])

    def test_counts_per_season(self):
        # The same team pair across two seasons keeps both seasons.
        df = pd.DataFrame({
            "Date": list(pd.date_range("2023-04-01", periods=10, freq="7D"))
                    + list(pd.date_range("2024-04-01", periods=10, freq="7D")),
            "HomeTeam": ["A"] * 20,
            "AwayTeam": ["B"] * 20,
        })
        assert len(drop_playoff_matches(df)) == 20
