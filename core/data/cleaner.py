import pandas as pd
from datetime import datetime

class DataCleaner:
    def __init__(self):
        # API data is already clean and normalized
        pass
    
    def clean_data(self, df):
        """Clean API data and split into results and fixtures"""
        try:
            df_clean = df.copy()
            
            # API data is already well-formatted, just ensure proper data types
            # Split into results (completed matches) and fixtures (upcoming matches)
            results = df_clean.dropna(subset=['FTHG', 'FTAG']).copy()
            fixtures = df_clean[df_clean['FTHG'].isna() | df_clean['FTAG'].isna()].copy()
            
            # Clean up results dataframe - keep authentic team names from API
            if not results.empty:
                base_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                extra_cols = [col for col in ['ScoreDetail', 'Season', 'SeasonStart'] if col in results.columns]
                results = results[base_cols + extra_cols].copy()
                results['FTHG'] = results['FTHG'].astype(int)
                results['FTAG'] = results['FTAG'].astype(int)
            
            # Clean up fixtures dataframe - keep authentic team names from API
            if not fixtures.empty:
                base_cols = ['Date', 'HomeTeam', 'AwayTeam']
                extra_cols = [col for col in ['ScoreDetail', 'Season', 'SeasonStart'] if col in fixtures.columns]
                fixtures = fixtures[base_cols + extra_cols].copy()
            else:
                # If no fixtures found, get upcoming fixtures using enhanced scraper
                print("No fixtures in data - getting upcoming fixtures from enhanced scraper")
                try:
                    from .scraper import AllsvenskanScraper
                    scraper = AllsvenskanScraper()
                    upcoming_fixtures = scraper.get_upcoming_fixtures()

                    if not upcoming_fixtures.empty:
                        fixtures = upcoming_fixtures[['Date', 'HomeTeam', 'AwayTeam']].copy()
                        # Remove any fixtures that already appear in results
                        # (ESPN may include games that football-data.co.uk
                        #  already has as completed results)
                        if not results.empty:
                            played = set(
                                zip(results["HomeTeam"], results["AwayTeam"])
                            )
                            before = len(fixtures)
                            fixtures = fixtures[
                                ~fixtures.apply(
                                    lambda r: (r["HomeTeam"], r["AwayTeam"]) in played,
                                    axis=1,
                                )
                            ].copy()
                            dropped = before - len(fixtures)
                            if dropped:
                                print(f"Dropped {dropped} fixtures already in results")
                        print(f"Added {len(fixtures)} upcoming fixtures from enhanced scraper")
                    else:
                        print("No upcoming fixtures available from any source")

                except Exception as e:
                    print(f"Could not get upcoming fixtures: {e}")
            
            return results, fixtures
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def validate_data(self, results, fixtures):
        """Validate cleaned data"""
        try:
            # Check results
            if not results.empty:
                required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
                if not all(col in results.columns for col in required_columns):
                    print("Missing required columns in results")
                    return False
                
                # Check for valid scores
                if (results['FTHG'] < 0).any() or (results['FTAG'] < 0).any():
                    print("Invalid negative scores found")
                    return False
            
            # Check fixtures
            if not fixtures.empty:
                required_columns = ['Date', 'HomeTeam', 'AwayTeam']
                if not all(col in fixtures.columns for col in required_columns):
                    print("Missing required columns in fixtures")
                    return False
            
            print(f"Data validation passed: {len(results)} results, {len(fixtures)} fixtures")
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
