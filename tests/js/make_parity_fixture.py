"""Regenerate tests/js/parity_fixture.json: a slice of site/data.json plus the
Python Dixon-Coles fit on it. model.test.mjs checks the JS port against this.

    python tests/js/make_parity_fixture.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.models.poisson_model import PoissonModel  # noqa: E402

data = json.loads((ROOT / "site/data.json").read_text())
results = [r for r in data["results"] if "2022-01-01" <= r[0] <= "2025-12-31"]
used = sorted({r[1] for r in results} | {r[2] for r in results})
remap = {old: new for new, old in enumerate(used)}
teams = [data["teams"][i] for i in used]
results = [[d, remap[h], remap[a], hg, ag] for d, h, a, hg, ag in results]

df = pd.DataFrame(results, columns=["Date", "h", "a", "FTHG", "FTAG"])
df["HomeTeam"] = [teams[i] for i in df.h]
df["AwayTeam"] = [teams[i] for i in df.a]
model = PoissonModel()
model.fit(df)

out = {
    "data": {"season": 2025, "gamesPerTeam": 30, "teams": teams, "results": results, "fixtures": []},
    "python": {
        "gamma": model.home_advantage,
        "rho": model.rho,
        "attack": [model.attack_rates[t] for t in teams],
        "defence": [model.defense_rates[t] for t in teams],
    },
}
(Path(__file__).parent / "parity_fixture.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"{len(results)} matches, {len(teams)} teams")
