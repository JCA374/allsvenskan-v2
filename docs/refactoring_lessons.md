# Codebase Refactoring: What, Why, and How

## The Core Principle

**Every piece of knowledge should live in exactly one place.**

When the same logic, constant, or function exists in multiple files, you create a maintenance trap: fix a bug in one copy and forget the other, and your app silently breaks. This principle is called **DRY** (Don't Repeat Yourself), and most of what was done here was applying it.

---

## Lesson 1: Centralize Configuration

### Problem
The same constants and file paths were defined in 3 separate files:

```python
# app.py
RESULTS_PATH = Path("data/clean/results.csv")
GAMES_PER_TEAM = 30
RELEGATION_SPOTS = 3

# scripts/daily_update.py  (same values, defined again)
RESULTS_PATH = ROOT / "data/clean/results.csv"
GAMES_PER_TEAM = 30
RELEGATION_SPOTS = 3

# core/utils/helpers.py  (yet again)
GAMES_PER_TEAM = 30
```

If you change the number of relegation spots from 3 to 2, you need to remember to update it in all three files. You *will* forget one eventually.

### Solution
Create one file that owns these values:

```python
# core/config.py  (the SINGLE source of truth)
RESULTS_PATH     = ROOT / "data/clean/results.csv"
GAMES_PER_TEAM   = 30
RELEGATION_SPOTS = 3
```

Everyone else imports from it:

```python
from core.config import RESULTS_PATH, GAMES_PER_TEAM, RELEGATION_SPOTS
```

### Rule of thumb
> If you see the same literal value in more than one file, extract it to a config module.

---

## Lesson 2: Extract Duplicated Functions

### Problem
Two functions were copy-pasted across `app.py` and `daily_update.py`:

```python
# Identical in both files:
def _normalize_teams(df):
    df = df.copy()
    for col in ("HomeTeam", "AwayTeam"):
        if col in df.columns:
            df[col] = df[col].map(lambda t: TEAM_NAME_MAP.get(str(t).strip(), str(t).strip()))
    return df

def _standings_from_results(results):
    # ... 20 lines of standings logic ...
```

This is worse than duplicated constants because the logic is complex enough that a bugfix in one copy might not match the other.

### Solution
Move the function to the shared module (`core/utils/helpers.py`) under a clear public name:

```python
# core/utils/helpers.py
def normalize_team_names(df): ...
def build_standings(results): ...
```

Both entry points now import and use the same function.

### Rule of thumb
> If you're about to copy-paste a function, stop. Put it somewhere both callers can import it.

---

## Lesson 3: Break Up Monolithic Files

### Problem
`app.py` was **1,494 lines** containing:
- Streamlit page config and CSS
- Sidebar navigation
- Session state management
- Data loading helpers
- 6 different page sections (Data, Model, Simulate, Forecast, Predictions, Update)
- Each page was 170-270 lines of mixed UI + business logic

This creates two problems:
1. **Cognitive load**: To edit the Forecast page, you need to scroll past 800 lines of unrelated code.
2. **Blast radius**: A typo on line 1300 (Update page) can break line 200 (Data page) because they share the same file scope.

### Solution: Separation by responsibility

Split into a hierarchy where **each file has one job**:

```
app.py                          (137 lines -- config, sidebar, routing)
core/ui/helpers.py              (shared UI helpers -- loaders, nav, stepper)
core/ui/pages/
    data.py                     (Data page only)
    model.py                    (Model page only)
    simulate.py                 (Simulate page only)
    forecast.py                 (Forecast page only)
    predictions.py              (Predictions page only)
    update.py                   (Update page only)
```

`app.py` becomes a thin router:

```python
if page == "Data":
    from core.ui.pages.data import render
    render()
elif page == "Model":
    from core.ui.pages.model import render
    render()
# ...
```

### Rule of thumb
> If a file has multiple "sections" separated by big comment banners, each section probably wants to be its own file. A good file is **one you can read top-to-bottom without losing track** -- typically under 300 lines.

---

## Lesson 4: Use Proper Python Packages

### Problem
The subdirectories under `core/` had no `__init__.py` files:

```
core/analysis/aggregator.py     # no __init__.py
core/data/scraper.py            # no __init__.py
core/models/poisson_model.py    # no __init__.py
```

Python 3.3+ treats these as "namespace packages" so imports still work, but it's unconventional. IDEs, type checkers, and other developers expect `__init__.py` to mark a directory as a package.

### Solution
Add empty `__init__.py` files to every package directory. It costs nothing and makes the intent explicit.

### Rule of thumb
> Every directory that contains Python modules should have an `__init__.py`.

---

## Lesson 5: Remove Dead Weight

### Problem
- `core/config/` and `core/database/` were empty directories (only `__pycache__`)
- `CLAUDE.md` referenced a `models/hybrid_model.py` that doesn't exist
- Session state key docs listed keys that aren't used (`db_manager`, `odds_fetched`, etc.)

Stale documentation is *worse* than no documentation because it actively misleads.

### Solution
Delete the empty directories, remove false references, update docs to match reality.

### Rule of thumb
> If something doesn't exist in the code, it shouldn't exist in the docs.

---

## Summary: The Structure Checklist

When evaluating (or building) a codebase, ask:

| Question | If "no" then... |
|----------|-----------------|
| Does every constant live in exactly one place? | Extract to a config module |
| Does every function live in exactly one place? | Extract to a shared module |
| Can you read each file without scrolling past unrelated code? | Split by responsibility |
| Does every directory have `__init__.py`? | Add them |
| Does the documentation match the code? | Update or remove stale docs |

The goal is that when you need to change *one thing*, you only need to touch *one file*, and you can understand that file without reading 1,400 lines of context.

---

## What Changed (reference)

| File | Before | After |
|------|--------|-------|
| `app.py` | 1,494 lines (monolith) | 137 lines (router) |
| `core/config.py` | did not exist | centralized paths + constants |
| `core/ui/pages/*.py` | did not exist | 6 page modules, ~170-240 lines each |
| `core/ui/helpers.py` | did not exist | shared UI helpers |
| `scripts/daily_update.py` | had duplicated functions + constants | imports from shared modules |
| `cli.py` | hardcoded path strings | imports from `core.config` |
| `core/*/` | missing `__init__.py` | proper packages |
