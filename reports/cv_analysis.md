# Cross-Validation Analysis: Correctness & Overfitting Assessment

**Date:** 2026-04-26  
**Scope:** `core/models/poisson_model.py` — `_cross_validate`, `walk_forward_cv`, `_fit_mle`

---

## TL;DR

The model does **not** use cross-validation in production — it is commented out. The
`walk_forward_cv` static method is architecturally sound but has a subtle implementation
discrepancy that makes its scores optimistic. The MLE path contains a real bug: it ignores
time-decay weights, contradicting the model's core design intent. True overfitting in the
parameter sense is unlikely given model complexity vs. data size, but there are structural
reasons the out-of-sample metrics can be misleading.

---

## 1. Is CV actually running?

**No.** The call to `_cross_validate` is commented out in `fit()` (lines 70–73):

```python
# Skip validation by default for faster training
# Uncomment below for validation if needed
# if len(results_df) > 100:
#     self.validation_score = self._cross_validate(results_df)
```

`validation_score` is always `None`. Any UI display showing "N/A" is accurate.

---

## 2. `_cross_validate` — structural problems if re-enabled

The `_cross_validate` method (lines 283–360) uses `TimeSeriesSplit`, which is the **correct
choice** for temporal data (train always precedes test). However, if re-enabled it would
produce misleading scores for two reasons:

### 2a. Different training path than production

In production `fit()`, the model is initialized with proper team statistics from
`TeamStrengthCalculator.calculate_strengths()` before `_refine_parameters` is called.
Inside each fold of `_cross_validate`, `temp_model` skips this — it initializes all
attack/defense rates to 1.0 and only runs `_refine_parameters`. The CV model is
systematically weaker than the production model, so CV scores would not be representative
of actual model quality.

### 2b. No team strength calculation inside the fold

`TeamStrengthCalculator.calculate_strengths(train_data)` is never called per fold. This
means the fold's `team_stats_df` — which initializes the Poisson parameters — is derived
from the full dataset (passed at the outer `fit()` level), not from the fold's training
partition. This is **data leakage**: team strength statistics include future (validation)
match data when initializing model parameters, making CV scores overly optimistic.

---

## 3. `walk_forward_cv` — mostly correct, one important gap

The static method `walk_forward_cv` (lines 611–750) is the better-designed evaluation path:

- Trains on k prior completed seasons, validates on the last completed season (historical).
- Trains on prior seasons + early current season, validates on the held-out tail (current-season split).
- Each fold calls `TeamStrengthCalculator.calculate_strengths(train_df)` on training data only → **no leakage**.

### What works correctly
- Temporal ordering is respected at the season and match level.
- Team strength stats are computed from training data only.
- Separate historical and in-progress season evaluations address different forecasting horizons.

### The gap: train model ≠ production model

`_train_and_eval` creates models with `use_mle=False, use_dixon_coles=False` (lines 678–679),
regardless of the production model's configuration. If the deployed model uses MLE and
Dixon-Coles, the walk-forward CV scores describe a simpler model and are not a valid estimate
of production model accuracy.

### Structural CV challenge: team identity across seasons

Allsvenskan is a promotion/relegation league. Teams promoted/relegated between seasons
may appear in the validation season but not in the training seasons. For those matches,
the model falls back to default attack/defense of 1.0. This is handled gracefully but it
means CV results are evaluated partly on "known teams" (where the model has real
parameters) and partly on defaults. The log-loss on the known-team subset will look better
than the true expected performance on a season with roster changes.

---

## 4. MLE ignores time-decay weights — a real bug

In `_fit_mle` (lines 129–210), the negative log-likelihood sums over all matches equally:

```python
log_probs = (poisson.logpmf(home_goals_arr, mu_home) +
             poisson.logpmf(away_goals_arr, mu_away))
...
return -np.sum(np.clip(log_probs, -10, None)) + reg
```

No time-decay weights are applied. A match from 2020 contributes identically to a match
from last week in the MLE objective. This directly contradicts the model's stated purpose:
*"More recent matches get higher weights"*.

The L2 regularization (`lam = 0.01 * len(valid_df)`) penalizes deviation from 1.0 but
does not compensate for ignoring recency. The Bayesian shrinkage afterward (lines 196–201)
is applied to the MLE result but cannot retroactively make the MLE time-aware.

**Fix:** Apply `exp(-time_decay * days_from_recent)` weights inside the MLE objective,
replacing `np.sum(log_probs)` with `np.dot(weights, log_probs)`.

---

## 5. Is the model overfitting?

### Parameter count vs. data size

For a 16-team Allsvenskan season (~240 matches), the model has:
- 16 attack rates + 16 defense rates + 1 home_advantage + 1 league_avg = **34 parameters**
- Ratio: ~7 matches per parameter — reasonable, not obviously overparameterized.

With Bayesian shrinkage and L2 regularization, the model is actively guarded against
extreme estimates for data-sparse teams.

### `_refine_parameters` is very simple

The refinement step (lines 386–441) computes weighted averages of observed goals and
blends with initialized values. This cannot meaningfully overfit — it's essentially a
smoothed empirical average.

### Conclusion on overfitting

**The model is unlikely to overfit in the classical parameter-overfitting sense.** The
concern is rather the opposite:

- The model may be **underfit** for teams with very few games (only `_refine_parameters`
  with few data points, falling back toward 1.0).
- The **MLE path may mildly overfit** relative to time-weighted fit: old matches drag
  attack/defense estimates away from recent form.
- The **validation metrics are optimistic** due to the issues above (leakage in
  `_cross_validate`, simplified model in `walk_forward_cv`).

---

## 6. Can cross-validation be validly used here?

**Yes, with temporal splits — which is what `walk_forward_cv` does.** Standard k-fold CV
assumes i.i.d. data and is **not** valid for football data. `TimeSeriesSplit` and
walk-forward evaluation are the right tools.

However, three domain-specific caveats apply:

1. **Small N:** 240 matches/season means each fold evaluates on ~50–80 matches. Log-loss
   estimates will have high variance across folds; treat them as directional, not precise.

2. **Non-stationarity:** The model trained on 2021–2023 is evaluated on 2024. But the 2024
   Allsvenskan may include 2–4 teams that were not in the 2023 season (promotion/relegation),
   plus squad turnover within surviving teams. Walk-forward CV will systematically
   underestimate difficulty for newly promoted teams.

3. **The "best lookback window" finding is expected to be small:** As the docstring notes
   (lines 629–633), the time-decay of 0.01/day means 2-year-old matches have weight
   `e^(-730*0.01) ≈ 0.0007` — essentially zero. Adding more historical seasons doesn't
   change the fit much. Walk-forward CV will confirm this but should not be interpreted
   as "history doesn't matter" — it means the exponential decay already discounts old data
   aggressively enough that extra seasons add little signal.

---

## 7. Recommendations (priority order)

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | MLE ignores time-decay weights | High | Apply `weights` vector inside `_fit_mle` log-likelihood sum |
| 2 | `_cross_validate` uses simplified model, potential leakage | Medium | If re-enabling, call `TeamStrengthCalculator` per fold and match production config |
| 3 | `walk_forward_cv` uses `use_mle=False` regardless of production config | Medium | Pass production config flags into `_train_and_eval` |
| 4 | CV is disabled by default | Low | Consider enabling `walk_forward_cv` as a model-quality check after training, logged but not blocking |
| 5 | Promotion/relegation teams not flagged in CV output | Low | Add a metadata field to CV results indicating "unknown teams" hit defaults |

---

## Appendix: Data sizes

- Allsvenskan: 16 teams, 30 rounds = 240 matches/season (full season)
- With 3-fold `TimeSeriesSplit` on one season: train ≈ 160, test ≈ 80 matches
- With `walk_forward_cv` on 3 seasons: train ≈ 480 matches, val ≈ 240 matches (historical)
- Current-season split (40% held out): train ≈ 144 matches, val ≈ 96 matches (if mid-season)
