import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Single source of truth: maps any known alias → canonical football-data.co.uk name.
# Canonical names are the ASCII short names used throughout results/fixtures CSVs
# and the Poisson model's attack/defense rate dictionaries.
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

from core.config import GAMES_PER_TEAM  # re-export for backward compat


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


def ensure_directory_exists(directory_path):
    """Ensure a directory exists, create if it doesn't"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False

def validate_dataframe(df, required_columns, df_name="DataFrame"):
    """Validate DataFrame structure and content"""
    validation_errors = []
    
    try:
        # Check if DataFrame is empty
        if df.empty:
            validation_errors.append(f"{df_name} is empty")
            return validation_errors
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_errors.append(f"{df_name} missing columns: {missing_columns}")
        
        # Check for null values in required columns
        for col in required_columns:
            if col in df.columns and df[col].isna().any():
                null_count = df[col].isna().sum()
                validation_errors.append(f"{df_name}[{col}] has {null_count} null values")
        
        return validation_errors
        
    except Exception as e:
        return [f"Error validating {df_name}: {str(e)}"]

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers, returning default if division by zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

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


def build_standings(results: pd.DataFrame) -> pd.DataFrame:
    """Build a league table from completed match results.

    Returns DataFrame with columns: Team, GP, W, D, L, GF, GA, GD, Pts
    sorted by Pts > GD > GF descending.
    """
    teams = pd.unique(results[["HomeTeam", "AwayTeam"]].values.ravel())
    cols = ["GP", "W", "D", "L", "GF", "GA", "GD", "Pts"]
    tbl = pd.DataFrame(0, index=teams, columns=cols)
    for _, r in results.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        tbl.at[h, "GP"] += 1; tbl.at[a, "GP"] += 1
        tbl.at[h, "GF"] += hg; tbl.at[h, "GA"] += ag
        tbl.at[a, "GF"] += ag; tbl.at[a, "GA"] += hg
        if hg > ag:
            tbl.at[h, "W"] += 1; tbl.at[a, "L"] += 1; tbl.at[h, "Pts"] += 3
        elif ag > hg:
            tbl.at[a, "W"] += 1; tbl.at[h, "L"] += 1; tbl.at[a, "Pts"] += 3
        else:
            tbl.at[h, "D"] += 1; tbl.at[a, "D"] += 1
            tbl.at[h, "Pts"] += 1; tbl.at[a, "Pts"] += 1
    tbl["GD"] = tbl["GF"] - tbl["GA"]
    return (
        tbl.sort_values(["Pts", "GD", "GF"], ascending=False)
           .reset_index()
           .rename(columns={"index": "Team"})
    )


def calculate_points_from_result(home_goals, away_goals):
    """Calculate points awarded for a match result"""
    try:
        home_goals = int(home_goals)
        away_goals = int(away_goals)
        
        if home_goals > away_goals:
            return 3, 0  # Home win
        elif home_goals == away_goals:
            return 1, 1  # Draw
        else:
            return 0, 3  # Away win
            
    except (ValueError, TypeError):
        return 0, 0  # Invalid input

def format_percentage(value, decimal_places=1):
    """Format a decimal as a percentage string"""
    try:
        return f"{value * 100:.{decimal_places}f}%"
    except (TypeError, ValueError):
        return "0.0%"

def get_season_progress(completed_matches, total_matches):
    """Calculate season progress as a percentage"""
    try:
        if total_matches <= 0:
            return 0
        return min(100, (completed_matches / total_matches) * 100)
    except (TypeError, ZeroDivisionError):
        return 0

def parse_match_score(score_string):
    """Parse a score string like '2-1' into home and away goals"""
    try:
        if pd.isna(score_string) or not score_string:
            return None, None
        
        parts = str(score_string).split('-')
        if len(parts) == 2:
            home_goals = int(parts[0].strip())
            away_goals = int(parts[1].strip())
            return home_goals, away_goals
        
        return None, None
        
    except (ValueError, AttributeError):
        return None, None

def create_league_table(results_df):
    """Create current league table from results"""
    try:
        if results_df.empty:
            return pd.DataFrame()
        
        # Get all teams
        home_teams = set(results_df['HomeTeam'].unique())
        away_teams = set(results_df['AwayTeam'].unique())
        all_teams = list(home_teams | away_teams)
        
        # Initialize table
        table_data = []
        
        for team in all_teams:
            # Home matches
            home_matches = results_df[results_df['HomeTeam'] == team]
            away_matches = results_df[results_df['AwayTeam'] == team]
            
            # Calculate stats
            games_played = len(home_matches) + len(away_matches)
            goals_for = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
            goals_against = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
            goal_difference = goals_for - goals_against
            
            # Calculate points
            points = 0
            wins = 0
            draws = 0
            losses = 0
            
            # Home points
            for _, match in home_matches.iterrows():
                home_pts, away_pts = calculate_points_from_result(match['FTHG'], match['FTAG'])
                points += home_pts
                if home_pts == 3:
                    wins += 1
                elif home_pts == 1:
                    draws += 1
                else:
                    losses += 1
            
            # Away points
            for _, match in away_matches.iterrows():
                home_pts, away_pts = calculate_points_from_result(match['FTHG'], match['FTAG'])
                points += away_pts
                if away_pts == 3:
                    wins += 1
                elif away_pts == 1:
                    draws += 1
                else:
                    losses += 1
            
            table_data.append({
                'Team': team,
                'Games': games_played,
                'Wins': wins,
                'Draws': draws,
                'Losses': losses,
                'Goals_For': goals_for,
                'Goals_Against': goals_against,
                'Goal_Difference': goal_difference,
                'Points': points
            })
        
        # Create DataFrame and sort by points, then goal difference
        table_df = pd.DataFrame(table_data)
        table_df = table_df.sort_values(['Points', 'Goal_Difference', 'Goals_For'], 
                                       ascending=[False, False, False])
        table_df['Position'] = range(1, len(table_df) + 1)
        
        return table_df
        
    except Exception as e:
        print(f"Error creating league table: {e}")
        return pd.DataFrame()

def save_data_safely(data, filepath, format='csv'):
    """Safely save data to file with error handling"""
    try:
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:
            ensure_directory_exists(directory)
        
        if format.lower() == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                pd.DataFrame(data).to_csv(filepath, index=False)
        elif format.lower() == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        return False

def load_data_safely(filepath, format='csv'):
    """Safely load data from file with error handling"""
    try:
        if not os.path.exists(filepath):
            return None
        
        if format.lower() == 'csv':
            return pd.read_csv(filepath)
        elif format.lower() == 'json':
            import json
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return None
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None

def calculate_current_standings_from_url():
    """Calculate current Allsvenskan standings using latest scraped results."""
    try:
        from datetime import datetime
        from core.data.scraper import AllsvenskanScraper

        scraper = AllsvenskanScraper()
        current_year = datetime.now().year
        season_data = scraper.scrape_matches([current_year])

        if season_data.empty:
            return pd.DataFrame()

        results = season_data.dropna(subset=['FTHG', 'FTAG']).copy()
        if results.empty:
            return pd.DataFrame()

        teams = pd.unique(results[['HomeTeam', 'AwayTeam']].values.ravel())
        columns = ['GP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
        table = pd.DataFrame(0, index=teams, columns=columns)

        for _, row in results.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            home_goals = int(row['FTHG'])
            away_goals = int(row['FTAG'])

            table.at[home, 'GP'] += 1
            table.at[away, 'GP'] += 1
            table.at[home, 'GF'] += home_goals
            table.at[home, 'GA'] += away_goals
            table.at[away, 'GF'] += away_goals
            table.at[away, 'GA'] += home_goals

            if home_goals > away_goals:
                table.at[home, 'W'] += 1
                table.at[away, 'L'] += 1
                table.at[home, 'Pts'] += 3
            elif away_goals > home_goals:
                table.at[away, 'W'] += 1
                table.at[home, 'L'] += 1
                table.at[away, 'Pts'] += 3
            else:
                table.at[home, 'D'] += 1
                table.at[away, 'D'] += 1
                table.at[home, 'Pts'] += 1
                table.at[away, 'Pts'] += 1

        table['GD'] = table['GF'] - table['GA']
        standings = (
            table
            .sort_values(['Pts', 'GD', 'GF'], ascending=False)
            .reset_index()
            .rename(columns={'index': 'Team'})
        )
        return standings

    except Exception as e:
        print(f"Error calculating Allsvenskan standings: {e}")
        return pd.DataFrame()

def calculate_current_standings(results_df):
    """Calculate current league standings from completed matches (fallback)"""
    try:
        if results_df.empty:
            return {}
        
        # Get all teams
        home_teams = set(results_df['HomeTeam'].unique())
        away_teams = set(results_df['AwayTeam'].unique())
        all_teams = list(home_teams | away_teams)
        
        # Initialize standings
        standings = {}
        for team in all_teams:
            standings[team] = {
                'played': 0,
                'won': 0,
                'drawn': 0,
                'lost': 0,
                'goals_for': 0,
                'goals_against': 0,
                'goal_diff': 0,
                'points': 0
            }
        
        def _get_match_points(match):
            """Standard football points: win=3, draw=1, loss=0."""
            try:
                home_goals = int(match['FTHG'])
                away_goals = int(match['FTAG'])
            except (ValueError, TypeError):
                return 0, 0, 0, 0

            if home_goals > away_goals:
                return 1, 0, (3, 0)
            elif away_goals > home_goals:
                return 0, 1, (0, 3)
            else:
                return 0, 0, (1, 1)

        # Process each match
        for _, match in results_df.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            home_goals = int(match['FTHG']) if pd.notna(match['FTHG']) else 0
            away_goals = int(match['FTAG']) if pd.notna(match['FTAG']) else 0
            
            # Update games played
            standings[home_team]['played'] += 1
            standings[away_team]['played'] += 1
            
            # Update goals
            standings[home_team]['goals_for'] += home_goals
            standings[home_team]['goals_against'] += away_goals
            standings[away_team]['goals_for'] += away_goals
            standings[away_team]['goals_against'] += home_goals
            
            # Update results and points using standard football rules (3/1/0)
            home_win, away_win, (home_points, away_points) = _get_match_points(match)

            if home_win:
                standings[home_team]['won'] += 1
                standings[away_team]['lost'] += 1
            elif away_win:
                standings[away_team]['won'] += 1
                standings[home_team]['lost'] += 1
            else:
                standings[home_team]['drawn'] += 1
                standings[away_team]['drawn'] += 1

            standings[home_team]['points'] += home_points
            standings[away_team]['points'] += away_points
        
        # Calculate goal difference
        for team in standings:
            standings[team]['goal_diff'] = standings[team]['goals_for'] - standings[team]['goals_against']
        
        return standings
        
    except Exception as e:
        print(f"Error calculating standings: {e}")
        return {}

def get_current_points_table(standings):
    """Extract just team:points from full standings"""
    return {team: data['points'] for team, data in standings.items()}
