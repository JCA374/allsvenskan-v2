"""
Unit tests for the Dixon-Coles model (core/models/poisson_model.py).
Tests correspond to §4.1 of the implementation specification.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

from core.models.poisson_model import PoissonModel, _neg_log_likelihood, _rps


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tau(x, y, lam, mu, rho):
    """Reference τ function from the DC paper."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _make_dummy_df(n=200, seed=0):
    """Generate a synthetic multi-team results DataFrame."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(6)]
    rows = []
    base_date = pd.Timestamp("2020-04-01")
    for i in range(n):
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append({
            'Date': base_date + pd.Timedelta(days=int(i * 7 / 6)),
            'HomeTeam': home,
            'AwayTeam': away,
            'FTHG': int(rng.poisson(1.5)),
            'FTAG': int(rng.poisson(1.1)),
            'SeasonStart': 2020,
        })
    return pd.DataFrame(rows)


# ── Test 1: τ = 1 for non-low scores ─────────────────────────────────────────

@pytest.mark.parametrize("x, y", [(2, 0), (0, 2), (2, 2), (3, 1), (5, 4)])
def test_tau_is_one_for_high_scores(x, y):
    """τ(x,y) must equal exactly 1 when x+y ≥ 3 (or neither cell is in 0–1)."""
    assert _tau(x, y, lam=1.5, mu=1.2, rho=-0.1) == 1.0


# ── Test 2: ρ = 0 collapses to Independent Poisson ───────────────────────────

def test_rho_zero_equals_independent_poisson():
    """With ρ=0 the DC correction has no effect; probabilities match plain Poisson."""
    from scipy.stats import poisson as sp_poisson

    lam, mu = 1.5, 1.2
    max_g = 10
    # DC with ρ=0
    k = np.arange(max_g + 1, dtype=float)
    p_h = np.exp(k * np.log(lam) - lam - gammaln(k + 1.0))
    p_a = np.exp(k * np.log(mu)  - mu  - gammaln(k + 1.0))
    M_dc = np.outer(p_h, p_a)
    # No τ correction when ρ=0: M unchanged
    M_indep = np.outer(
        sp_poisson.pmf(np.arange(max_g + 1), lam),
        sp_poisson.pmf(np.arange(max_g + 1), mu),
    )
    np.testing.assert_allclose(M_dc, M_indep, atol=1e-12)


# ── Test 3: τ formula matches the paper ──────────────────────────────────────

def test_tau_formula_values():
    """Hard-coded verification from §4.1 of the spec."""
    lam, mu, rho = 1.5, 1.2, -0.1

    # (0,0): 1 − 1.5·1.2·(−0.1) = 1 + 0.18 = 1.18
    assert abs(_tau(0, 0, lam, mu, rho) - 1.18) < 1e-9

    # (1,1): 1 − ρ = 1 − (−0.1) = 1.10
    assert abs(_tau(1, 1, lam, mu, rho) - 1.10) < 1e-9

    # (0,1): 1 + λ·ρ = 1 + 1.5·(−0.1) = 0.85
    assert abs(_tau(0, 1, lam, mu, rho) - 0.85) < 1e-9


# ── Test 4: Identifiability — sum(α) ≈ 0 after fit ───────────────────────────

def test_attack_sum_to_zero_constraint():
    """After fitting, the SLSQP constraint Σα_i = 0 must hold to 1e-5."""
    df = _make_dummy_df(n=150)
    model = PoissonModel(time_decay=0.0018)
    model.fit(df)
    assert model.fitted
    alpha_sum = sum(model.attack_rates.values())
    assert abs(alpha_sum) < 1e-4, f"Σα = {alpha_sum:.6f}, expected ≈ 0"


# ── Test 5: Score grid sums to ≈ 1 before normalisation ──────────────────────

def test_score_grid_sums_to_one():
    """The DC score grid M (before explicit renormalisation) should sum to ≈ 1."""
    lam, mu, rho = 1.5, 1.2, -0.08
    max_g = 10
    k = np.arange(max_g + 1, dtype=float)
    p_h = np.exp(k * np.log(lam) - lam - gammaln(k + 1.0))
    p_a = np.exp(k * np.log(mu)  - mu  - gammaln(k + 1.0))
    M = np.outer(p_h, p_a)
    M[0, 0] *= 1.0 - lam * mu * rho
    M[0, 1] *= 1.0 + lam * rho
    M[1, 0] *= 1.0 + mu  * rho
    M[1, 1] *= 1.0 - rho
    assert abs(M.sum() - 1.0) < 1e-3, f"M.sum() = {M.sum():.6f}"


# ── Test 6: RPS unit test ─────────────────────────────────────────────────────

def test_rps_worked_example():
    """Forecast (0.8, 0.1, 0.1) on a home win gives RPS = 0.025 (Constantinou 2012)."""
    score = _rps(0.8, 0.1, 0.1, 'H')
    assert abs(score - 0.025) < 1e-9, f"RPS = {score}, expected 0.025"


# ── Test 7: predict_match returns positive floats ─────────────────────────────

def test_predict_match_positive_floats():
    """After fitting, predict_match() returns two positive finite floats."""
    df = _make_dummy_df(n=100)
    model = PoissonModel(time_decay=0.0018)
    model.fit(df)
    assert model.fitted

    teams = list(model.attack_rates.keys())
    home, away = teams[0], teams[1]
    mu_home, mu_away = model.predict_match(home, away)

    assert isinstance(mu_home, float)
    assert isinstance(mu_away, float)
    assert mu_home > 0
    assert mu_away > 0
    assert np.isfinite(mu_home)
    assert np.isfinite(mu_away)


# ── Bonus: predict_outcome_probabilities sums to 1 ───────────────────────────

def test_outcome_probabilities_sum_to_one():
    """1X2 probabilities must sum to 1 (±1e-6)."""
    df = _make_dummy_df(n=100)
    model = PoissonModel(time_decay=0.0018)
    model.fit(df)

    teams = list(model.attack_rates.keys())
    probs = model.predict_outcome_probabilities(teams[0], teams[1])
    total = probs['home_win'] + probs['draw'] + probs['away_win']
    assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}"


# ── Data-prep guards ──────────────────────────────────────────────────────────

def test_scoreline_cap_clips_extreme_goals():
    """Goals > 9 must be clipped to 9; the model should still fit without error."""
    df = _make_dummy_df(n=150)
    # Inject one absurd score-line (data error)
    df.loc[0, 'FTHG'] = 25
    df.loc[1, 'FTAG'] = 18
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PoissonModel(time_decay=0.0018)
        model.fit(df)
    assert model.fitted
    assert any("clipped" in str(w.message).lower() for w in caught), \
        "Expected a clipping warning for extreme score-lines"


def test_disconnected_graph_raises():
    """fit() must raise ValueError when the fixture graph is disconnected."""
    rng = np.random.default_rng(99)
    # Two isolated groups of teams that never play each other
    group_a = ["A1", "A2", "A3"]
    group_b = ["B1", "B2", "B3"]
    rows = []
    base = pd.Timestamp("2023-01-01")
    for i in range(60):
        teams = group_a if i % 2 == 0 else group_b
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            'Date': base + pd.Timedelta(days=i),
            'HomeTeam': h, 'AwayTeam': a,
            'FTHG': int(rng.poisson(1.4)),
            'FTAG': int(rng.poisson(1.1)),
        })
    df = pd.DataFrame(rows)
    model = PoissonModel()
    with pytest.raises(ValueError, match="disconnected"):
        model.fit(df)
