# CLAUDE.md — Dixon-Coles football prediction model implementation

## Overview and goals

This document is the build specification for a **classic Dixon-Coles (1997) bivariate-Poisson football prediction model** with **RPS-optimised exponential time-decay weighting** (Pearson, Livingston & King, 2019). It is written so an LLM coding agent can implement, test, and integrate the model into the user's existing Python football prediction system without any further design input.

The user already has 10 years of historical match data (date, home team, away team, home goals, away goals). Your job is to add a fully working Dixon-Coles fit/predict module that:

1. Estimates team attack/defence ratings, a global home-advantage parameter γ, and the Dixon-Coles low-score dependence parameter ρ, using a time-weighted pseudo-likelihood with weight φ(t) = exp(−ξ·t), t in **days**.
2. Selects ξ by walk-forward cross-validation that **minimises mean Rank Probability Score (RPS)** on a held-out window — not by maximum likelihood (ξ would degenerate; see §3).
3. Outputs calibrated `P(home win)`, `P(draw)`, `P(away win)` (and a full score-line grid for over/under, BTTS, Asian handicap markets).
4. Beats an Independent Poisson baseline on RPS — the empirical ranking confirmed across five European leagues by Pearson, Livingston & King (2019).

Pearson et al. (2019) found Dixon-Coles to be the best overall model on aggregate across the EPL, Bundesliga, Serie A, La Liga, and Ligue 1, edging out the Bivariate Weibull Count model and beating Independent Poisson outright. We therefore implement the classic DC formulation as the core, exactly as the source paper recommends.

**Language conventions.** British English throughout (`colour`, `optimise`, `analyse`, `defence`, `parameterise`).

---

## 1. Mathematical foundation

### 1.1 Independent Poisson base model (Maher 1982)

For home team *i* playing away team *j*, let `X_ij` and `Y_ij` be home and away goals. The base model assumes independence:

```
X_ij ~ Poisson(λ),   λ = exp(α_i + β_j + γ)
Y_ij ~ Poisson(μ),   μ = exp(α_j + β_i)
```

- **α_i** = team *i*'s attack strength (higher = scores more goals).
- **β_i** = team *i*'s defence strength. **In our log-link parameterisation, higher β means the team concedes more (worse defence).** Watch the sign convention: Dixon & Coles' original multiplicative paper used β as a *defensive weakness* multiplier; in log-space it remains a "concedes-more" coefficient. Top teams should therefore have **high α and low (very negative) β**.
- **γ** = global home-advantage (log-scale, expected ≈ 0.25–0.35 for European leagues, i.e. ~28–42% goal uplift).

We deliberately use the **additive log-space form** `λ = exp(α + β + γ)` rather than the multiplicative form `λ = α·β·γ` from the original paper. This guarantees λ > 0 for any real parameter values, lets the optimiser explore ℝ unconstrained, and is what every modern Python implementation does (penaltyblog, dashee87, opisthokonta).

### 1.2 Dixon-Coles low-score correction

Independent Poisson under-predicts the joint frequencies of 0–0 and 1–1 (and slightly over-predicts 1–0 / 0–1) versus what is observed in real football data. Dixon & Coles introduced a multiplicative correction τ on only the four low-score cells:

```
τ(x, y; λ, μ, ρ) =
   1 − λ·μ·ρ      if (x, y) = (0, 0)
   1 + λ·ρ        if (x, y) = (0, 1)
   1 + μ·ρ        if (x, y) = (1, 0)
   1 − ρ          if (x, y) = (1, 1)
   1              otherwise
```

The corrected joint pmf is:

```
P(X = x, Y = y) = τ(x, y; λ, μ, ρ) · Pois(x; λ) · Pois(y; μ)
```

ρ is a single scalar that captures the dependence between low scores. Its admissible range is `max(−1/λ, −1/μ) ≤ ρ ≤ min(1/(λμ), 1)`; in fitted European leagues you should expect **ρ ≈ −0.20 to +0.10** (typically slightly negative, ≈ −0.05 to −0.13). Setting ρ = 0 reduces the model exactly to Independent Poisson — use this as a unit test (§5).

### 1.3 Exponential time-weighting

Recent matches are more informative than old ones. Dixon & Coles introduced

```
φ(t) = exp(−ξ · t)
```

where t is the time elapsed (in our implementation: **days**) between match k and the prediction reference date. Dixon & Coles' original paper used half-weeks and ξ = 0.0065 per half-week; converting, that is **0.0065 / 3.5 ≈ 0.00186 per day**. Pearson, Livingston & King (2019, §3.2) explicitly use **days** as the time unit. We must do the same — never mix half-weeks and days.

### 1.4 Time-weighted pseudo-likelihood

The objective for fitting team strengths at reference time T is:

```
L_T(θ) = Σ_{k : t_k < T}  φ(T − t_k) · ℓ_k(θ)

where ℓ_k(θ) =  log τ(x_k, y_k; λ_k, μ_k, ρ)
              + x_k · log(λ_k) − λ_k − log(x_k!)
              + y_k · log(μ_k) − μ_k − log(y_k!)
```

This is a **pseudo-likelihood**: it is no longer a true product of densities (the weights destroy that), but the maximiser is still consistent and asymptotically normal under standard conditions. We minimise the **negative** of L_T with `scipy.optimize.minimize`.

### 1.5 Identifiability constraint

α and β are jointly identifiable only up to an additive shift (you can add c to all α and subtract c from γ without changing any λ). To pin down a unique solution, impose **one** equality constraint. Two equivalent options exist; use whichever is cleaner for your optimiser:

| Constraint | Form | Optimiser implication |
|---|---|---|
| **Mean attack = 1** (Dixon-Coles original) | `(1/n) Σ_i α_i = 1`, i.e. `Σ α_i = n` | Equality constraint → must use **SLSQP** or `trust-constr` |
| **Sum-to-zero** (opisthokonta speed-up) | Estimate n−1 attacks, set α_n = −Σ_{i<n} α_i, mean attack = 0 | No constraint → use **L-BFGS-B** (~10× faster) |

The modern Python convention (penaltyblog) is the first; the opisthokonta R package uses the second. Either is correct provided you stay consistent. **Recommend: implement sum-to-zero reparameterisation by default for speed; expose the SLSQP constraint version as an option for direct comparability with literature numbers.**

### 1.6 Total parameter count

For a league of n teams: **2n + 2 free parameters** (n attacks, n defences, γ home advantage, ρ low-score dependence). For a 20-team league: 42. Under sum-to-zero reparameterisation you store 2n + 2 but optimise only 2n + 1 numerically (one attack is derived) — or 2n + 2 if you also reparameterise defences.

### 1.7 Rank Probability Score (RPS)

For an ordinal r-outcome forecast (here r = 3 with order **H, D, A**) and indicator vector e ∈ {0,1}³ of the actual result:

```
RPS = (1 / (r − 1)) · Σ_{i=1}^{r−1} ( Σ_{j=1}^{i} (p_j − e_j) )²
    = (1/2) · [ (p_H − e_H)²  +  (p_H + p_D − e_H − e_D)² ]   for football
```

RPS ∈ [0, 1], **lower is better**, and unlike Brier it is **sensitive to the ordinal distance** between forecast mass and the realised outcome (Constantinou & Fenton 2012). Worked check: forecast (0.8, 0.1, 0.1) on a home win gives RPS = ½·(0.04 + 0.01) = 0.025 ✓.

We use **mean RPS** over the validation window as the objective for ξ-tuning.

---

## 2. Implementation steps (in order)

### Step A — Data preparation

Input: a `pandas.DataFrame` with at minimum these columns:

| column | dtype | notes |
|---|---|---|
| `date` | datetime64[ns] | match kick-off date |
| `home_team` | str | |
| `away_team` | str | |
| `home_goals` | int | |
| `away_goals` | int | |

Recommended preprocessing:

1. Sort by `date` ascending; drop duplicates.
2. Build a **stable team index map** `{team_name: i}`. Persist it; if you re-fit later with a new team appearing (promotion), append rather than reshuffle.
3. **Connectivity check**. The team×team played-graph must be connected, otherwise the model is unidentifiable. Verify with `scipy.sparse.csgraph.connected_components` on an adjacency matrix where `A[i,j] = 1` if i and j have ever played (either direction). If fragmented, refuse to fit and raise a clear error.
4. Compute the **time gap in days** for each match relative to the reference date `T` (typically the first un-played fixture date or "now"): `t_k = (T − match_date).days`.
5. Compute weights `w_k = exp(−ξ · t_k)` once per fit (re-computed inside the ξ search).
6. **Sanity-cap extreme score-lines.** Real top-flight matches almost never exceed 9 goals per side; if you have data errors (e.g. cricket scores), clip or flag.

### Step B — Negative log-likelihood (vectorised)

This is the performance-critical function. Implement it in pure numpy/scipy with `gammaln` for log-factorials.

```python
import numpy as np
from scipy.special import gammaln

def neg_log_likelihood(params, home_idx, away_idx, gh, ga, weights, n_teams):
    """
    Vectorised weighted negative log-likelihood for the Dixon-Coles model.

    params layout: [α_1..α_n,  β_1..β_n,  γ,  ρ]   (length 2n+2)
    home_idx, away_idx : int arrays of team indices, shape (m,)
    gh, ga             : home/away goals, shape (m,)
    weights            : exp(-ξ * t_k), shape (m,)
    """
    attack  = params[0:n_teams]
    defence = params[n_teams:2*n_teams]
    gamma   = params[-2]
    rho     = params[-1]

    # Per-match Poisson means
    log_lam = attack[home_idx] + defence[away_idx] + gamma   # home expected log-goals
    log_mu  = attack[away_idx] + defence[home_idx]           # away expected log-goals
    lam     = np.exp(log_lam)
    mu      = np.exp(log_mu)

    # Log-Poisson (use gammaln for stability)
    log_p_h = gh * log_lam - lam - gammaln(gh + 1.0)
    log_p_a = ga * log_mu  - mu  - gammaln(ga + 1.0)

    # Vectorised tau correction (Dixon-Coles low-score adjustment)
    tau = np.ones_like(lam)
    m00 = (gh == 0) & (ga == 0); tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    m01 = (gh == 0) & (ga == 1); tau[m01] = 1.0 + lam[m01] * rho
    m10 = (gh == 1) & (ga == 0); tau[m10] = 1.0 + mu[m10]  * rho
    m11 = (gh == 1) & (ga == 1); tau[m11] = 1.0 - rho

    # Guard: tau must be positive — penalise the optimiser if not
    if np.any(tau <= 0.0):
        return 1e10

    log_lik = weights * (np.log(tau) + log_p_h + log_p_a)
    return -np.sum(log_lik)
```

**Why this is the right shape**:

- Vectorised over all matches at once — no Python loop.
- Uses `gammaln(k+1)` for log(k!) — stable and fast versus `np.log(scipy.stats.poisson.pmf(...))` which underflows for high goal counts.
- `log_lam` / `log_mu` computed first, exponentiated once — avoids redundant `exp`/`log`.
- Returns a large finite penalty (`1e10`) when τ ≤ 0 instead of `np.nan`/`np.inf`, so the optimiser smoothly retreats.

### Step C — Parameter optimisation

```python
from scipy.optimize import minimize

def fit_dixon_coles(df, n_teams, weights, init=None,
                    use_constraint=True, max_iter=200):
    if init is None:
        # Reasonable defaults: small positive attacks, small negative defences
        init = np.concatenate([
            np.full(n_teams,  0.10),    # α (attack)  — start near 0 in log-space
            np.full(n_teams, -0.10),    # β (defence)
            [0.25],                     # γ home advantage (typical EPL value)
            [-0.05],                    # ρ low-score dependence
        ])

    args = (df['home_idx'].values, df['away_idx'].values,
            df['home_goals'].values, df['away_goals'].values,
            weights, n_teams)

    if use_constraint:
        # Mean-attack-equals-zero (cleanest log-space analogue of DC's mean=1)
        constraint = {'type': 'eq',
                      'fun': lambda p: np.sum(p[:n_teams])}
        bounds = [(None, None)] * (2 * n_teams) + [(None, None), (-0.2, 0.2)]
        result = minimize(neg_log_likelihood, init, args=args,
                          method='SLSQP', constraints=[constraint],
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-9})
    else:
        # Reparameterised: use L-BFGS-B (faster); sum-to-zero baked into NLL
        bounds = [(None, None)] * (2 * n_teams) + [(None, None), (-0.2, 0.2)]
        result = minimize(neg_log_likelihood, init, args=args,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-9})

    if not result.success:
        raise RuntimeError(f"DC optimiser did not converge: {result.message}")
    return result
```

**Notes**:

- **Method choice.** SLSQP when you carry the mean-attack equality constraint; L-BFGS-B when you have reparameterised. `trust-constr` works too but is slower. SciPy ≥ 1.10 offers improved convergence diagnostics — use them.
- **Bounds on ρ.** Keep ρ in (−0.2, 0.2) — values outside are pathological and almost guarantee τ ≤ 0 for plausible λ, μ.
- **Initialisation matters.** Bad inits can produce a non-PSD numerical Hessian. If convergence fails, try a multi-start: 5 random initialisations from `np.random.normal(0, 0.1, ...)`, keep the lowest NLL.
- **Analytic gradients (optional, big win).** Penaltyblog v1.5+ ships analytic Jacobians (Cython) and reports ~5–10× speed-up. For a from-scratch numpy implementation, autodiff via JAX is the easiest route to the same gain — `jax.grad(nll)` and pass `jac=` to `minimize`. This is **strongly recommended** for production because the ξ-grid search refits the model dozens of times.

### Step D — ξ-tuning by walk-forward RPS

```python
def rps(p, outcome_idx):
    """Three-outcome (H,D,A) Rank Probability Score. Lower is better."""
    e = np.zeros(3); e[outcome_idx] = 1.0
    cum_p = np.cumsum(p)
    cum_e = np.cumsum(e)
    return np.sum((cum_p - cum_e) ** 2) / 2.0   # divide by r-1 = 2
```

**Walk-forward backtest schema**:

1. Choose the validation window: typically **the most recent 1–2 full seasons**, **excluding the first ~10 match-days of each season** (newly promoted teams have unstable estimates early on).
2. For each candidate ξ on a grid `np.arange(0.0, 0.0050, 0.0001)`:
   1. Iterate over each match-day d in the validation window.
   2. Refit the DC model on **all** matches with date < d, with weights `exp(−ξ · (d − date_k).days)`.
   3. For each match in match-day d, compute (P_H, P_D, P_A) and its RPS against the actual result.
   4. Accumulate.
   5. Store mean RPS for this ξ.
3. The **optimal ξ** minimises mean RPS.

**Why grid search not joint MLE.** As ξ → ∞, every weight → 0 and the weighted NLL → 0 (its supremum). So MLE on ξ is a degenerate maximum: the model trivially "fits" by ignoring all data. You must hold ξ out as a hyperparameter and select by predictive performance. `scipy.optimize.minimize_scalar(method='bounded', bounds=(0, 0.01))` works because the RPS-vs-ξ curve is empirically smooth and unimodal, but **grid search is preferred** in practice — it gives you a diagnostic plot (you should see a clean U-shape), parallelises trivially across cores, and tolerates flat regions.

**Expected optima** (for context, in days⁻¹):
- Dixon & Coles original (translated): **0.00186**.
- Opisthokonta (multi-season top divisions): EPL 0.0018, Bundesliga 0.0023, Eredivisie 0.0019, Ligue 1 0.0019.
- Pearson, Livingston & King (2019, EPL multi-season): ~0.0033.
- **Sensible default if untuned: 0.0018.**

**Performance tip.** Refits are independent across ξ — parallelise with `joblib.Parallel` over the ξ grid. Within a fixed ξ, warm-start each match-day's fit from the previous match-day's solution.

### Step E — Prediction (match outcome probabilities)

Given fitted parameters and a future fixture (home_team, away_team):

```python
def predict_match(params, home_idx, away_idx, n_teams, max_goals=10):
    attack  = params[0:n_teams]
    defence = params[n_teams:2*n_teams]
    gamma, rho = params[-2], params[-1]

    lam = np.exp(attack[home_idx] + defence[away_idx] + gamma)
    mu  = np.exp(attack[away_idx] + defence[home_idx])

    k = np.arange(max_goals + 1)
    # Log-Poisson then exp — stable for moderate λ, μ
    log_h = k * np.log(lam) - lam - gammaln(k + 1.0)
    log_a = k * np.log(mu)  - mu  - gammaln(k + 1.0)
    p_h = np.exp(log_h)
    p_a = np.exp(log_a)

    M = np.outer(p_h, p_a)                 # (G+1, G+1) joint pmf

    # Apply DC tau correction to the 2x2 low-score block
    M[0, 0] *= 1.0 - lam * mu * rho
    M[0, 1] *= 1.0 + lam * rho
    M[1, 0] *= 1.0 + mu  * rho
    M[1, 1] *= 1.0 - rho

    # (Optional) renormalise so M sums to exactly 1 — recommended for downstream markets
    M /= M.sum()

    p_home_win = np.tril(M, -1).sum()      # i > j (home goals > away goals)
    p_draw     = np.trace(M)
    p_away_win = np.triu(M,  1).sum()
    return np.array([p_home_win, p_draw, p_away_win]), M
```

**Score-grid size.** Use `max_goals = 10` for 1×2 markets (residual mass < 1e-6 for typical λ ≈ 1.5). Use `max_goals = 15` if you also serve over/under or correct-score markets at high totals. Penaltyblog defaults to 15.

The full M matrix gives you everything else: over/under N goals = sum of cells with i+j > N; BTTS = sum of cells with i ≥ 1 and j ≥ 1; Asian handicap = appropriate triangle sums. Build these as helper methods on a `FootballProbabilityGrid` class.

---

## 3. Suggested code structure

```
dixon_coles/
├── __init__.py
├── data.py            # load, validate, build team index, connectivity check
├── model.py           # DixonColesModel class: fit, predict, score
├── likelihood.py      # neg_log_likelihood (numpy) + optional JAX version
├── weights.py         # dixon_coles_weights(dates, xi, ref_date) -> np.array
├── tuning.py          # walk_forward_rps, optimise_xi
├── metrics.py         # rps, brier, log_loss, calibration_curve
├── prediction.py      # FootballProbabilityGrid: 1X2, OU, BTTS, AH helpers
└── tests/
    ├── test_likelihood.py
    ├── test_tau.py
    ├── test_constraint.py
    ├── test_rps.py
    └── test_against_penaltyblog.py
```

**Public API (mirror penaltyblog's ergonomics)**:

```python
class DixonColesModel:
    def __init__(self, df: pd.DataFrame, *, xi: float = 0.0018,
                 reference_date: pd.Timestamp | None = None,
                 reparameterise: bool = True):
        ...

    def fit(self, *, max_iter: int = 200, multi_start: int = 1,
            jac: Callable | None = None) -> "DixonColesModel":
        ...

    def predict(self, home_team: str, away_team: str,
                max_goals: int = 10, normalise: bool = True
                ) -> FootballProbabilityGrid:
        ...

    def get_params(self) -> dict[str, float]: ...
    def team_ratings(self) -> pd.DataFrame: ...    # team, attack, defence
    def score(self, df: pd.DataFrame, metric: str = "rps") -> float: ...

def optimise_xi(df: pd.DataFrame, *,
                grid: np.ndarray = np.arange(0.0, 0.0050, 0.0001),
                validation_seasons: int = 1,
                skip_matchdays_per_season: int = 10,
                n_jobs: int = -1) -> tuple[float, pd.DataFrame]:
    """Returns (best_xi, full_curve_df)."""
```

---

## 4. Verification and validation

### 4.1 Unit tests on the model itself

1. **τ reduces to 1 for non-low scores.** `tau(2, 3, ...)` must be exactly 1 regardless of ρ.
2. **ρ = 0 collapses to Independent Poisson.** Fit DC with ρ fixed at 0 and a vanilla independent Poisson model on the same data; the team strengths must match to within optimiser tolerance.
3. **τ formula matches the paper.** Hard-code `(0,0,1.5,1.2,-0.1)` → `1 − 1.5·1.2·(−0.1) = 1.18`; `(1,1,*,*,−0.1)` → `1.10`; `(0,1,1.5,*,−0.1)` → `1 − 0.15 = 0.85`. (Note signs.)
4. **Identifiability constraint holds post-fit.** If you used the SLSQP `Σα = n` (or sum-to-zero) constraint, assert `abs(np.sum(α) - target) < 1e-6` after fit.
5. **Scaling invariance.** Add a constant c to all α and subtract c from γ — λ and μ for every match must be unchanged (sanity check that the model is genuinely identifiable up to that one shift).
6. **RPS unit test.** Forecast (0.8, 0.1, 0.1) on a home result must give 0.025 (Constantinou & Fenton 2012, worked example).
7. **Score grid sums to ≈ 1.** Before any normalisation, `M.sum()` should be within ~1e-4 of 1 for plausible λ, μ at `max_goals=10`. Larger discrepancies indicate over-/under-flow or a bug in τ application.

### 4.2 Sanity checks on fitted parameters (real data)

After fitting on a full 10-year dataset for, say, the EPL:

| Quantity | Expected range | Diagnostic if outside |
|---|---|---|
| **γ (home advantage)** | 0.20 – 0.40 (log-scale) | Below 0: severe bug. Above 0.5: data error (wrong home/away column). |
| **ρ (low-score dependence)** | −0.20 to +0.10, usually slightly negative | Outside (−0.3, 0.3): bound is binding; revisit. |
| **Top-club α (attack)** | Highest in the league | If a relegation team tops attack ranking: index-mapping bug. |
| **Top-club β (defence)** | Lowest (most negative) in the league | Same. |
| **|α_max − α_min|** | ~0.6–1.2 in log-space | A factor of 2× in goal-scoring rate between best and worst is typical. |

### 4.3 Reference benchmarks to compare against

- **Penaltyblog** (`pip install penaltyblog`): fit `pb.models.DixonColesGoalModel` on the same dataset; your `home_advantage` and `rho` should match within ±0.02 and team ratings (after de-meaning) within ±0.05.
- **opisthokonta blog Part 1–4** (R): if you can call R via `rpy2`, the fitted ρ on EPL 2011/12 should be ≈ −0.13 — dashee87's Python port reproduces this.
- Dashee87's notebook (https://github.com/dashee87/blogScripts) — direct Python apples-to-apples.

### 4.4 Predictive validation

1. **Walk-forward out-of-sample RPS** on a held-out final season. Compare:
   - Independent Poisson baseline.
   - Dixon-Coles with ξ = 0 (no time decay).
   - Dixon-Coles with tuned ξ.
   - (Optional) Bookmaker consensus (de-vigged closing odds).
   Target: tuned DC < untuned DC < Independent Poisson on mean RPS, with bookmakers as an upper-performance ceiling.
2. **Brier score** as secondary metric: `BS = (1/N) Σ Σ_{j∈{H,D,A}} (p_j − e_j)²`. Report alongside RPS but do not optimise on it (insensitive to ordinal structure).
3. **Log-loss** as a third metric: `−(1/N) Σ log(p_observed)`. Strictly local proper score, sensitive to overconfident wrong predictions.
4. **Calibration / reliability diagrams.** Bin predicted P(home win) into deciles; for each bin plot empirical home-win frequency against mean predicted probability with Wilson 95% bands. Repeat for draw and away. A well-calibrated DC tends to over-forecast home wins below ~0.6 and under-forecast draws — this is a known quirk of Poisson-family models (see Wheatcroft 2021, arXiv:2106.14345).
5. **Goodness-of-fit χ² on score-line frequencies.** Bin observed (home goals, away goals) frequencies on the test set; compute expected counts as the sum over matches of model-implied joint pmfs; compute Σ (O−E)²/E over bins with E ≥ 5. Degrees of freedom = #bins − #parameters − 1. The DC adjustment should specifically reduce residuals at (0,0), (1,0), (0,1), (1,1) versus Independent Poisson. Show this side-by-side as the central empirical justification for the τ correction.

---

## 5. Common pitfalls and edge cases

1. **Time units.** Mixing days and half-weeks invalidates ξ entirely. Always use **days** internally and document this loudly. ξ ≈ 0.0018/day is the right order of magnitude; ξ = 0.0065 in days would correspond to a 107-day half-life — far too aggressive.
2. **τ negativity.** Large |ρ| combined with large λ·μ drives `1 − λ·μ·ρ` ≤ 0 → log domain error. Bound ρ in (−0.2, 0.2) and add a finite penalty (`return 1e10`) inside the NLL when any τ ≤ 0. Never let `np.nan` propagate.
3. **Newly promoted teams.** They have no historical data. Three workable heuristics:
   - Initialise their α, β at the **league mean** (zero in our log-space) for the first season. Expect 5–10 noisy match-days of poor predictions.
   - Use a **Bayesian hierarchical prior** (penaltyblog has `BayesianHierarchicalGoalModel`); the prior shrinks them toward the league mean automatically.
   - Bridge from second-tier data with a strength offset (Constantinou's *Dolores* approach).
4. **Disconnected fixture graph** (especially early-season or cup data): the model becomes unidentifiable. Always run the connectivity check (Step A.3) before fitting.
5. **Season boundaries.** Do not reset team strengths between seasons — exponential decay handles staleness gracefully. Modern Bayesian state-space variants (e.g. Owen et al. 2025, *JRSSC* 74:717) treat each season as a discrete shock with a random-walk prior; if you need the absolute best, layer that on top.
6. **Large goal counts.** `np.log(scipy.stats.poisson.pmf(k, λ))` underflows for k ≥ ~20. Always use `gammaln(k+1)` and the closed-form log-Poisson `k·log(λ) − λ − gammaln(k+1)`.
7. **Overflow in `np.exp(log_lam)`.** Add a clip `np.exp(np.clip(log_lam, -10, 5))` if a degenerate optimisation step pushes log_lam to extreme values; otherwise λ overflows to inf and ruins the gradient.
8. **Optimiser non-convergence.** SLSQP can stall on stiff likelihoods. Fall back to (i) supplying analytic Jacobians, (ii) multi-start with 5 random inits, (iii) the L-BFGS-B + reparameterisation route.
9. **Refit cost during ξ search.** A 10-year dataset × 38-match validation window × 50 ξ values × 200 LBFGS iterations is ~380,000 NLL calls. Vectorise the NLL (already done above), parallelise the ξ axis (`joblib`), warm-start sequential refits, and consider JAX-jitted gradients. Without these, the ξ search can take hours.
10. **Retraining frequency in production.** Refit weekly (or per match-day) once tuned. Re-tune ξ once per season — its optimum is stable across short windows.
11. **Sign conventions.** Many papers and blog posts flip α/β (attack vs defence). Within your codebase pick one convention (we use: high α = good attack; high β = bad defence in the log-additive form) and unit-test that the highest-α team is one of the league's top scorers.
12. **Probability normalisation.** After applying the τ correction the score grid sums to slightly more or less than 1 (the correction is multiplicative and not designed to preserve mass exactly). Renormalising `M /= M.sum()` is recommended for downstream betting markets.

---

## 6. When to reach beyond classic Dixon-Coles

The user asked for the "best option available". The classic Dixon-Coles with RPS-tuned ξ is the right **core** because it is precisely what Pearson, Livingston & King (2019) found to be the strongest model on aggregate. However, if you want incremental gains:

- **Bayesian hierarchical Dixon-Coles** (Baio & Blangiardo 2010 framework; available out-of-the-box in penaltyblog ≥ 1.5, Stan-backed, with PyMC and a new native Cython MCMC engine in v1.9). Gives uncertainty intervals, automatic shrinkage for promoted teams, and slightly improved RPS in low-data regimes.
- **State-space Dixon-Coles** (Owen et al. 2025, *JRSSC* 74:717). Replaces the fixed-ξ exponential weighting with a learned random-walk evolution variance σ²_η — strictly more flexible, recently published. Worth implementing as a v2 in numpyro/JAX.
- **Hybrid: DC ratings as features in a gradient booster.** Several recent papers (Razali et al. 2022 with pi-ratings; Mendes-Neves et al. 2025) show that pure XGBoost/NN do *not* beat well-implemented DC, but a hybrid using DC's λ, μ, plus pi-ratings, plus rest days, plus market odds, fed to CatBoost can shave another 1–2% off RPS.
- **Bivariate Weibull Count** (Boshnakov, Kharrat & McHale 2017) is the third model in the source paper. It beats DC on the EPL specifically but not on aggregate.

For now, **classic DC with RPS-tuned ξ is the recommendation** — it is the empirical winner across 5 leagues in the source paper, has a 30-year track record, and compounds cleanly with anything you build on top.

---

## 7. References (with URLs)

**Foundational papers**

- Maher, M. J. (1982). "Modelling Association Football Scores." *Statistica Neerlandica* 36(3): 109–118. DOI: https://doi.org/10.1111/j.1467-9574.1982.tb00782.x — open mirror PDF: http://www.90minut.pl/misc/maher.pdf
- Dixon, M. J. & Coles, S. G. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market." *Journal of the Royal Statistical Society Series C* 46(2): 265–280. DOI: https://doi.org/10.1111/1467-9876.00065 — Lancaster repo metadata: https://research.lancaster-university.uk/en/publications/modelling-association-football-scores-and-inefficiencies-in-the-f/
- Constantinou, A. C. & Fenton, N. E. (2012). "Solving the Problem of Inadequate Scoring Rules for Assessing Probabilistic Football Forecast Models." *Journal of Quantitative Analysis in Sports* 8(1), Art. 1. Open author PDF: http://constantinou.info/downloads/papers/solvingTheProblem.pdf
- Boshnakov, G., Kharrat, T. & McHale, I. G. (2017). "A Bivariate Weibull Count Model for Forecasting Association Football Scores." *International Journal of Forecasting* 33(2): 458–466. DOI: https://doi.org/10.1016/j.ijforecast.2016.11.006
- **Pearson, M., Livingston Jr, G. & King, R. (2020).** "An exploration of predictive football modelling." *Journal of Quantitative Analysis in Sports* 16(1): 27–39 (online Dec 2019). DOI: https://doi.org/10.1515/jqas-2019-0075 — full text via ResearchGate: https://www.researchgate.net/publication/338028907_An_exploration_of_predictive_football_modelling
- Baio, G. & Blangiardo, M. (2010). "Bayesian hierarchical model for the prediction of football results." *Journal of Applied Statistics* 37(2): 253–264.

**Modern/Bayesian/state-space**

- Owen, A. et al. (2025). "Bayesian state-space models for the modelling and prediction of EPL football." *JRSSC* 74(3): 717. https://academic.oup.com/jrsssc/article/74/3/717/7929974
- Michels et al. (2023/2025). "Extending the Dixon and Coles model: an application to women's football data." arXiv:2307.02139.
- Fischer & Heuer (2024). arXiv:2408.08331 — empirical comparison of NN/RF/Poisson on five top European leagues.
- Wheatcroft, E. (2021). "More on verification of probability forecasts for football outcomes." arXiv:2106.14345 — calibration plots and Brier decomposition.
- Wheatcroft, E. (2019). arXiv:1908.08980 — dissenting view on RPS for football.

**Reference Python implementations**

- **penaltyblog** (Martin Eastwood) — production-grade, Cython-accelerated, actively maintained (v1.9 Feb 2026): https://github.com/martineastwood/penaltyblog · PyPI: https://pypi.org/project/penaltyblog/ · Docs: https://docs.pena.lt/y/models/dixon_coles.html · Tutorial blog: https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- **dashee87/blogScripts** — pedagogical numpy/scipy port of the opisthokonta series: https://github.com/dashee87/blogScripts/blob/master/Jupyter/2018-09-13-predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting.ipynb · Companion blog: https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/
- **Torvaney/soccerstan** — Stan/PyStan Dixon-Coles + Karlis-Ntzoufras: https://github.com/Torvaney/soccerstan
- **opisthokonta/goalmodel** (R, callable from Python via rpy2) — most thorough academic implementation: https://github.com/opisthokonta/goalmodel
- GitHub topic page (browse community variants including PyMC Bayesian forks): https://github.com/topics/dixon-coles

**Reference R / blog series** (the canonical exposition; ports it to your taste)

- Opisthokonta (Jonas Christoffer Lindstrøm) Part 1: https://opisthokonta.net/?p=890
- Part 2: https://opisthokonta.net/?p=913 · Part 3: https://opisthokonta.net/?p=927 · Part 4 (sum-to-zero speed-up): https://opisthokonta.net/?p=939
- Time-weighted Poisson regression: https://opisthokonta.net/?p=1013
- Simple re-implementation: https://opisthokonta.net/?p=1685

**Tooling docs**

- SciPy `optimize.minimize`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- SciPy `special.gammaln`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaln.html
- JAX (for autodiff acceleration): https://jax.readthedocs.io/

---

## 8. Acceptance criteria

The implementation is considered complete when **all** of the following hold:

1. All seven unit tests in §4.1 pass.
2. On the user's 10-year dataset, fit time for a single ξ is < 30 seconds (vectorised numpy) or < 5 seconds (with JAX/analytic gradient).
3. Fitted γ ∈ [0.20, 0.40], ρ ∈ [−0.20, 0.10], top-3 attack teams match domain expectations.
4. Tuned-ξ DC strictly beats untuned DC, which strictly beats Independent Poisson, on mean RPS over a held-out final season.
5. Penaltyblog cross-check on the same data agrees on `home_advantage` and `rho` to within ±0.02.
6. Calibration plots, χ² goodness-of-fit, and walk-forward RPS curve are produced as standard diagnostic outputs.
7. The module exposes the public API of §3 and is importable as a single package from the existing prediction system.