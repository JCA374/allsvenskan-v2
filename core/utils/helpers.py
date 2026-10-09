"""Team-name normalisation and match-data sanity helpers."""
import pandas as pd

from core.config import GAMES_PER_TEAM

# Single source of truth: maps any known alias → canonical football-data.co.uk name.
# Canonical names are the ASCII short names used throughout results/fixtures CSVs.
TEAM_NAME_MAP: dict = {
    # IFK Göteborg
    'IFK Göteborg': 'Goteborg',
    'IFK Goteborg': 'Goteborg',
    # Malmö FF
    'Malmö FF': 'Malmo FF',
    'Malmö': 'Malmo FF',
    'Malmo': 'Malmo FF',
    # Djurgården
    'Djurgårdens IF': 'Djurgarden',
    'Djurgardens IF': 'Djurgarden',
    'Djurgarden IF': 'Djurgarden',
    'Djurgården': 'Djurgarden',
    # BK Häcken
    'BK Häcken': 'Hacken',
    'BK Hacken': 'Hacken',
    # Hammarby
    'Hammarby IF': 'Hammarby',
    # Halmstad
    'Halmstads BK': 'Halmstad',
    # Kalmar
    'Kalmar FF': 'Kalmar',
    # Mjällby
    'Mjällby AIF': 'Mjallby',
    'Mjallby AIF': 'Mjallby',
    # Norrköping
    'IFK Norrköping': 'Norrkoping',
    'IFK Norrkoping': 'Norrkoping',
    'Norrkopings IFK': 'Norrkoping',
    # Örgryte
    'Örgryte IS': 'Orgryte',
    'Orgryte IS': 'Orgryte',
    # Sirius
    'IK Sirius': 'Sirius',
    # Västerås SK
    'Västerås SK': 'Vasteras SK',
    # Värnamo
    'IFK Värnamo': 'Varnamo',
    'IFK Varnamo': 'Varnamo',
    # Elfsborg
    'IF Elfsborg': 'Elfsborg',
    # Brommapojkarna
    'IF Brommapojkarna': 'Brommapojkarna',
    # Degerfors
    'Degerfors IF': 'Degerfors',
    # Östersund
    'Östersunds FK': 'Ostersunds',
    'Ostersunds FK': 'Ostersunds',
    # Sundsvall
    'GIF Sundsvall': 'Sundsvall',
    # Östers
    'Östers IF': 'Oster',
    'Osters IF': 'Oster',
    'Osters': 'Oster',
    # Örebro
    'Örebro SK': 'Orebro',
    'Orebro SK': 'Orebro',
    # Landskrona
    'Landskrona BoIS': 'Landskrona',
    # Varberg
    'Varberg BoIS': 'Varberg',
    # Brage
    'IK Brage': 'Brage',
    # Gefle
    'Gefle IF': 'Gefle',
    # Helsingborg
    'Helsingborgs IF': 'Helsingborg',
    # Ljungskile
    'Ljungskile SK': 'Ljungskile',
    # GAIS (already canonical)
    # AIK (already canonical)
}


def normalize_team_name(team_name):
    """Normalize team name to canonical football-data.co.uk form."""
    if pd.isna(team_name):
        return team_name
    return TEAM_NAME_MAP.get(str(team_name).strip(), str(team_name).strip())


def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TEAM_NAME_MAP to HomeTeam/AwayTeam columns of a DataFrame."""
    df = df.copy()
    for col in ("HomeTeam", "AwayTeam"):
        if col in df.columns:
            df[col] = df[col].map(lambda t: TEAM_NAME_MAP.get(str(t).strip(), str(t).strip()))
    return df


def validate_games_per_team(results_df, fixtures_df, expected=GAMES_PER_TEAM):
    """Check that every team has exactly `expected` total games (results + fixtures).

    Returns a list of (team, played, upcoming, total) tuples for teams that
    don't match.  An empty list means everything is correct.
    """
    teams = set()
    for df in (results_df, fixtures_df):
        if df is not None and not df.empty:
            teams |= set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

    bad = []
    for team in sorted(teams):
        played = 0
        upcoming = 0
        if results_df is not None and not results_df.empty:
            played = int((results_df["HomeTeam"] == team).sum()
                         + (results_df["AwayTeam"] == team).sum())
        if fixtures_df is not None and not fixtures_df.empty:
            upcoming = int((fixtures_df["HomeTeam"] == team).sum()
                           + (fixtures_df["AwayTeam"] == team).sum())
        total = played + upcoming
        if total != expected:
            bad.append((team, played, upcoming, total))
    return bad


def drop_playoff_matches(df: pd.DataFrame, min_matches: int = 10) -> pd.DataFrame:
    """Drop relegation-playoff matches that the source files list as league games.

    A playoff is two legs against a Superettan side, which therefore appears with
    only 2 matches in that season. Any match involving a team with fewer than
    `min_matches` games in a season is dropped.
    """
    season = df["Date"].dt.year
    counts = pd.concat([
        pd.DataFrame({"season": season, "team": df["HomeTeam"]}),
        pd.DataFrame({"season": season, "team": df["AwayTeam"]}),
    ]).value_counts()
    home_n = [counts[(s, t)] for s, t in zip(season, df["HomeTeam"])]
    away_n = [counts[(s, t)] for s, t in zip(season, df["AwayTeam"])]
    keep = (pd.Series(home_n, index=df.index) >= min_matches) & (pd.Series(away_n, index=df.index) >= min_matches)
    return df[keep]
