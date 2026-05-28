import io
import logging
import re
from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd
import requests

from core.utils.helpers import TEAM_NAME_MAP

logger = logging.getLogger(__name__)


class AllsvenskanScraper:
    """
    Scraper for Allsvenskan (Swedish top-flight football) match data.

    Primary source: football-data.co.uk free CSV download (no API key required).
    The "new" format file (SWE.csv) contains all historical seasons and the current
    partial season. Upcoming fixtures appear as rows where HG/AG are empty.

    Returns a combined DataFrame with completed results (FTHG/FTAG filled) and
    upcoming fixtures (FTHG/FTAG set to NaN).
    """

    # football-data.co.uk "new" format — all Allsvenskan seasons in one file
    FDUK_NEW_URL = "https://www.football-data.co.uk/new/SWE.csv"

    # Historical standard-format files.
    # football-data.co.uk encodes Swedish seasons (single calendar year) with
    # consecutive-year codes matching their other European leagues.
    HISTORICAL_SEASONS: dict = {
        2024: "2425",
        2023: "2324",
        2022: "2223",
        2021: "2122",
        2020: "2021",
        2019: "1920",
        2018: "1819",
        2017: "1718",
        2016: "1617",
        2015: "1516",
    }
    HISTORICAL_BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/S1.csv"

    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 30):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
            }
        )
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def scrape_matches(self, seasons: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """
        Fetch Allsvenskan results and upcoming fixtures.

        Args:
            seasons: Iterable of season years (e.g. [2023, 2024]). When None, fetches
                     all available data from the aggregated football-data.co.uk file.

        Returns:
            DataFrame with columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, Season, SeasonStart.
            Completed matches have integer FTHG/FTAG; upcoming fixtures have NaN.
        """
        df = self._fetch_new_format()

        if df.empty:
            self.logger.warning("New-format file unavailable; trying historical season files")
            df = self._fetch_historical_seasons(seasons)

        if df.empty:
            self.logger.error("No Allsvenskan data fetched from any source")
            return pd.DataFrame(
                columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season", "SeasonStart"]
            )

        if seasons is not None:
            season_list = list(seasons) if not isinstance(seasons, list) else seasons
            df = df[df["SeasonStart"].isin(season_list)].copy()
            if df.empty:
                self.logger.warning("No data found for requested seasons: %s", season_list)
                return df

        df["Match"] = df["HomeTeam"] + " - " + df["AwayTeam"]
        self.logger.info("Fetched %d rows (results + fixtures)", len(df))
        return df.reset_index(drop=True)

    # ESPN name → canonical name: use the shared TEAM_NAME_MAP from helpers
    ESPN_TEAM_NAME_MAP = TEAM_NAME_MAP

    ESPN_ALLSVENSKAN_URL = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer/swe.1/scoreboard"
    )

    def get_upcoming_fixtures(self, seasons: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """Return only upcoming fixtures (rows without scores) from today onwards.

        Primary: football-data.co.uk (includes upcoming rows when available).
        Fallback: ESPN public API, which always carries the full season schedule.
        """
        if seasons is None:
            current_year = datetime.now().year
            seasons = [current_year]

        df = self.scrape_matches(seasons)
        fixtures = df[df["FTHG"].isna()].copy()

        today = pd.Timestamp.now().normalize()
        if "Date" in fixtures.columns:
            fixtures = fixtures[fixtures["Date"] >= today]

        if fixtures.empty:
            self.logger.info(
                "No upcoming fixtures in primary source; fetching from ESPN API"
            )
            fixtures = self._fetch_espn_fixtures()

        cols = [
            c for c in ["Date", "HomeTeam", "AwayTeam", "Season", "SeasonStart", "Match"]
            if c in fixtures.columns
        ]
        return fixtures[cols].reset_index(drop=True)

    def _fetch_espn_fixtures(self) -> pd.DataFrame:
        """Fetch upcoming Allsvenskan fixtures from the ESPN public scoreboard API."""
        today = datetime.now()
        season_end = today.replace(month=11, day=30)
        date_from = today.strftime("%Y%m%d")
        date_to = season_end.strftime("%Y%m%d")
        current_year = today.year

        url = (
            f"{self.ESPN_ALLSVENSKAN_URL}"
            f"?dates={date_from}-{date_to}&limit=300"
        )
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self.logger.warning("ESPN API request failed: %s", exc)
            return pd.DataFrame()

        rows = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {})
            if status.get("completed", False):
                continue  # skip already-played matches

            date_str = event.get("date", "")
            try:
                date = pd.Timestamp(date_str).tz_localize(None).normalize()
            except Exception:
                try:
                    date = pd.Timestamp(date_str).tz_convert(None).normalize()
                except Exception:
                    continue

            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            home_name = self.ESPN_TEAM_NAME_MAP.get(
                home.get("team", {}).get("displayName", ""),
                home.get("team", {}).get("displayName", ""),
            )
            away_name = self.ESPN_TEAM_NAME_MAP.get(
                away.get("team", {}).get("displayName", ""),
                away.get("team", {}).get("displayName", ""),
            )
            rows.append(
                {
                    "Date": date,
                    "HomeTeam": home_name,
                    "AwayTeam": away_name,
                    "Season": str(current_year),
                    "SeasonStart": current_year,
                    "Match": f"{home_name} - {away_name}",
                }
            )

        if not rows:
            self.logger.warning("ESPN API returned no upcoming fixtures")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        before = len(df)
        df = df.drop_duplicates(subset=["HomeTeam", "AwayTeam"], keep="first")
        if len(df) < before:
            self.logger.info("Dropped %d duplicate ESPN fixtures", before - len(df))
        self.logger.info("Fetched %d upcoming fixtures from ESPN API", len(df))
        return df

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _fetch_new_format(self) -> pd.DataFrame:
        """Download and parse the football-data.co.uk new-format aggregated CSV for Sweden."""
        raw = self._get_csv(self.FDUK_NEW_URL)
        if raw is None:
            return pd.DataFrame()

        try:
            df = pd.read_csv(raw, encoding="utf-8", on_bad_lines="skip")
        except Exception as exc:
            self.logger.error("Failed to parse new-format CSV: %s", exc)
            return pd.DataFrame()

        return self._normalise_new_format(df)

    def _fetch_historical_seasons(self, seasons: Optional[Iterable[int]]) -> pd.DataFrame:
        """Fetch individual season CSV files (old/standard format) as fallback."""
        if seasons is None:
            seasons = list(self.HISTORICAL_SEASONS.keys())

        frames: List[pd.DataFrame] = []
        for year in seasons:
            code = self.HISTORICAL_SEASONS.get(year)
            if not code:
                self.logger.warning("No season code configured for year %s", year)
                continue

            url = self.HISTORICAL_BASE_URL.format(season=code)
            raw = self._get_csv(url)
            if raw is None:
                continue

            try:
                df = pd.read_csv(raw, encoding="utf-8", on_bad_lines="skip")
                df = self._normalise_standard_format(df, year)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                self.logger.error("Failed to parse season %s (%s): %s", year, url, exc)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _get_csv(self, url: str):
        """Fetch a URL and return a StringIO, or None on failure."""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return io.StringIO(resp.text)
        except requests.RequestException as exc:
            self.logger.warning("HTTP error fetching %s: %s", url, exc)
            return None

    def _normalise_new_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map football-data.co.uk new-format columns to our standard schema.

        New format columns (subset): Country, League, Season, Date, Time, Home, Away, HG, AG, Res
        HG = Home Goals, AG = Away Goals (equivalent to FTHG/FTAG).
        Note: the file may have a BOM, producing a column named 'ï»¿Country'.
        """
        # Strip whitespace and remove BOM from column names
        df.columns = [c.strip().lstrip("\ufeff").lstrip("ï»¿") for c in df.columns]

        # Keep only Allsvenskan rows.
        # New format uses Country/League; old format used Div = "S1".
        if "League" in df.columns:
            df = df[df["League"].str.strip().str.lower() == "allsvenskan"].copy()
        elif "Div" in df.columns:
            df = df[df["Div"].str.strip().str.upper() == "S1"].copy()

        if df.empty:
            self.logger.warning("No Allsvenskan rows found in SWE.csv")
            return df

        # Rename Home/Away → HomeTeam/AwayTeam (new format uses shorter names)
        rename = {}
        if "Home" in df.columns and "HomeTeam" not in df.columns:
            rename["Home"] = "HomeTeam"
        if "Away" in df.columns and "AwayTeam" not in df.columns:
            rename["Away"] = "AwayTeam"
        # Rename goal columns to our standard names
        if "HG" in df.columns:
            rename["HG"] = "FTHG"
        if "AG" in df.columns:
            rename["AG"] = "FTAG"
        if rename:
            df = df.rename(columns=rename)

        # Parse season start year
        if "Season" in df.columns:
            df["SeasonStart"] = df["Season"].apply(self._parse_season_year)
        else:
            df["SeasonStart"] = None

        # Parse dates — football-data.co.uk uses dd/mm/yy or dd/mm/yyyy
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date"])

        # Goals: numeric; upcoming fixtures will become NaN
        for col in ["FTHG", "FTAG"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        required = {"Date", "HomeTeam", "AwayTeam"}
        if not required.issubset(df.columns):
            self.logger.error(
                "New-format CSV missing required columns: %s",
                required - set(df.columns),
            )
            return pd.DataFrame()

        keep = [
            c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season", "SeasonStart"]
            if c in df.columns
        ]
        return df[keep].copy()

    def _normalise_standard_format(self, df: pd.DataFrame, season_year: int) -> pd.DataFrame:
        """
        Map football-data.co.uk standard/old-format columns to our schema.

        Standard format already uses FTHG/FTAG column names.
        """
        df.columns = [c.strip() for c in df.columns]

        required = {"Date", "HomeTeam", "AwayTeam"}
        if not required.issubset(df.columns):
            self.logger.warning("Standard-format file missing columns: %s", required - set(df.columns))
            return pd.DataFrame()

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date"])

        for col in ["FTHG", "FTAG"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["SeasonStart"] = season_year
        df["Season"] = str(season_year)

        keep = [
            c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season", "SeasonStart"]
            if c in df.columns
        ]
        return df[keep].copy()

    @staticmethod
    def _parse_season_year(season_str) -> Optional[int]:
        """Extract the start year from strings like '2024', '24/25', '2024-25'."""
        if pd.isna(season_str):
            return None
        s = str(season_str).strip()
        m = re.match(r"^(\d{4})", s)
        if m:
            return int(m.group(1))
        # "24/25" -> 2024
        m = re.match(r"^(\d{2})/\d{2}$", s)
        if m:
            yr = int(m.group(1))
            return 2000 + yr if yr < 50 else 1900 + yr
        return None
