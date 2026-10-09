# Repository Guidelines

See `CLAUDE.md` for the project layout, commands and model notes.

- The product is the static site in `site/` (plain ES modules, no build step, no dependencies). Python in `core/` only fetches data.
- Run `python -m pytest tests/ -m "not network"` and `node --test "tests/js/*.test.mjs"` before committing; CI runs both before deploying.
- Python: PEP 8, four-space indentation. JS: two-space indentation, ES modules, no frameworks.
- Commit messages are short and imperative; keep clean-ups, fixes and features in separate commits.
- Insert team names into the DOM with `textContent` or the `esc()` helper in `site/app.js`, never raw `innerHTML`.
