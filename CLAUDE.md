# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Allsvenskan (Swedish top-flight football) season forecast. A static site on GitHub Pages
(https://jca374.github.io/allsvenskan-v2/) where **the visitor's browser** fits a Dixon-Coles model and
runs the Monte Carlo season simulation. Python only fetches and exports the match data.

## Layout

| Path | Purpose |
|------|---------|
| `site/index.html`, `site/style.css` | The page (plain HTML/CSS, no build step) |
| `site/app.js` | UI: table, tiles, heatmap, what-if picks, score grids; talks to the workers |
| `site/model.js` | Dixon-Coles fit, score grids, season simulation, round replay, backtest (JS port of `core/models/poisson_model.py`) |
| `site/worker.js` | Web Worker wrapper around `model.js`; the page runs two (interactive forecast + background replay/backtest) |
| `site/charts.js` | Hand-written SVG charts (line + crosshair, scatter, calibration) and the shared tooltip |
| `site/data.json` | Results + remaining fixtures, written by the export script |
| `scripts/export_site_data.py` | `--fetch` refreshes `data/clean/*.csv` via `core/data/pipeline.py`, then writes `site/data.json` |
| `core/data/` | `scraper.py` (football-data.co.uk + ESPN fixtures), `cleaner.py`, `pipeline.py` (`fetch_and_save`) |
| `core/utils/helpers.py` | `TEAM_NAME_MAP`, `normalize_team_names`, `validate_games_per_team`, `drop_playoff_matches` |
| `core/models/poisson_model.py` | Python reference model; the JS port is tested against it |
| `.github/workflows/pages.yml` | Daily 05:00 UTC + on push: tests → fetch data → deploy `site/` to Pages |

## Commands

```bash
python scripts/export_site_data.py --fetch      # refresh data and site/data.json
python -m http.server -d site 8000               # preview at http://localhost:8000
python -m pytest tests/ -m "not network"         # Python tests (drop -m for live-data tests)
node --test "tests/js/*.test.mjs"                # JS model tests
python tests/js/make_parity_fixture.py           # regenerate the Python-vs-JS parity fixture
```

## Data

- Source: `https://www.football-data.co.uk/new/SWE.csv` (results; no CORS, so the browser cannot fetch it directly). Upcoming fixtures come from the ESPN scoreboard API, queried **one day at a time** (date-range queries return nothing).
- Canonical team names are football-data's ASCII names (`Malmo FF`, `Goteborg`); `TEAM_NAME_MAP` maps aliases. Display names with å/ä/ö live in `DISPLAY_NAMES` in `site/app.js`.
- The source lists relegation-playoff legs as league games; `drop_playoff_matches` removes them at export.
- `fetch_and_save` refuses to write if results would shrink or any team does not total 30 games (16 teams, double round-robin).

## Model notes

- λ = exp(α_home + β_away + γ), μ = exp(α_away + β_home), time decay ξ = 0.0018/day, low-score correction ρ. ξ was checked by backtest (0.001 / 0.0018 / 0.003; 0.0018 best).
- Teams with little recent history (weighted matches < 5, e.g. newly promoted) are blended toward the average of the 3 weakest teams of the latest season. The Python model does not do this, so parity is only tested on teams with full history.
- If you change the model maths, change both `site/model.js` and `core/models/poisson_model.py`, then regenerate the parity fixture.
