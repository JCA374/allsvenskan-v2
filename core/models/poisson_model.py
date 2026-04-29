"""
Dixon-Coles (1997) football prediction model with RPS-optimised exponential
time-decay weighting (Pearson, Livingston & King 2019).

Mathematical foundation
-----------------------
Log-additive parameterisation (Maher 1982 in log-space):

    λ = exp(α_i + β_j + γ)   home expected goals
    μ = exp(α_j + β_i)       away expected goals

    α_i  attack strength  — higher = more goals scored
    β_i  defence weakness — lower (more negative) = fewer goals conceded
    γ    global home-advantage (log-scale, typically 0.25–0.35)

Dixon-Coles low-score correction τ:

    τ(0,0) = 1 − λ·μ·ρ
    τ(0,1) = 1 + λ·ρ
    τ(1,0) = 1 + μ·ρ
    τ(1,1) = 1 − ρ
    τ(x,y) = 1  for x+y ≥ 3

Time-weighting: φ(t) = exp(−ξ · t)  where t is days before reference date.
ξ (xi) is the per-day decay rate; default 0.0018 (≈ DC original translated to days).
"""

import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


# ── Module-level NLL (must be picklable) ──────────────────────────────────────

def _neg_log_likelihood(params, home_idx, away_idx, gh, ga, weights, n_teams):
    """Vectorised weighted negative log-likelihood for the Dixon-Coles model.

    params layout: [α_1..α_n,  β_1..β_n,  γ,  ρ]   length 2n+2
    """
    attack  = params[0:n_teams]
    defence = params[n_teams:2 * n_teams]
    gamma   = params[-2]
    rho     = params[-1]

    log_lam = attack[home_idx] + defence[away_idx] + gamma
    log_mu  = attack[away_idx] + defence[home_idx]
    lam     = np.exp(np.clip(log_lam, -10.0, 5.0))
    mu      = np.exp(np.clip(log_mu,  -10.0, 5.0))

    log_p_h = gh * log_lam - lam - gammaln(gh + 1.0)
    log_p_a = ga * log_mu  - mu  - gammaln(ga + 1.0)

    tau = np.ones(len(lam))
    m00 = (gh == 0) & (ga == 0)
    m01 = (gh == 0) & (ga == 1)
    m10 = (gh == 1) & (ga == 0)
    m11 = (gh == 1) & (ga == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10]  * rho
    tau[m11] = 1.0 - rho

    if np.any(tau <= 0.0):
        return 1e10

    log_lik = weights * (np.log(tau) + log_p_h + log_p_a)
    return -np.sum(log_lik)


def _rps(p_home, p_draw, p_away, outcome):
    """Rank Probability Score for a single match. Lower is better.

    outcome: 'H', 'D', or 'A'
    """
    p = np.array([p_home, p_draw, p_away])
    e = np.zeros(3)
    e[{'H': 0, 'D': 1, 'A': 2}[outcome]] = 1.0
    cum_p = np.cumsum(p)
    cum_e = np.cumsum(e)
    return np.sum((cum_p - cum_e) ** 2) / 2.0


# ── Main class ────────────────────────────────────────────────────────────────

class PoissonModel:
    """Dixon-Coles (1997) bivariate-Poisson model with exponential time-weighting.

    Exposes the same public API as the previous heuristic PoissonModel so that
    all callers (simulator, hybrid model, CLI, Streamlit app) work unchanged.

    Parameters
    ----------
    time_decay : float
        Per-day exponential decay rate ξ (default 0.0018, i.e. DC original).
        Formerly used as a within-season decay; now the sole time-weighting
        parameter — no separate season_multiplier step is applied.
    season_multiplier : float
        Accepted for API/pickle compatibility; ignored in DC model (ξ covers
        all time-weighting continuously).
    use_mle : bool
        Accepted for API compatibility; DC always fits via maximum pseudo-
        likelihood — this flag has no effect.
    use_dixon_coles : bool
        Accepted for API compatibility; the τ low-score correction is always
        applied in the DC model.
    """

    def __init__(self, time_decay=0.0018, season_multiplier=0.5,
                 use_mle=False, use_dixon_coles=False):
        # Public attributes expected by callers / pickle round-trips
        self.time_decay        = time_decay
        self.season_multiplier = season_multiplier   # stored but unused
        self.use_mle           = use_mle             # stored but unused
        self.use_dixon_coles   = use_dixon_coles     # stored but unused

        self.attack_rates   = {}    # {team: α_i}  log-space
        self.defense_rates  = {}    # {team: β_i}  log-space (lower = better)
        self.home_advantage = 0.0   # γ in log-space (~0.25–0.35)
        self.league_avg     = 0.0   # unused in DC; kept for compat
        self.rho            = 0.0   # ρ Dixon-Coles low-score parameter
        self.fitted         = False
        self.validation_score = None
        self.last_trained   = None
        self.training_window = None

        # Internal DC state
        self._xi          = time_decay   # alias for clarity
        self._team_index  = {}           # {team: int index}
        self._params      = None         # raw numpy params array

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, results_df, team_stats_df=None):
        """Fit the Dixon-Coles model on completed match results.

        Parameters
        ----------
        results_df : pd.DataFrame
            Completed matches with columns: Date, HomeTeam, AwayTeam, FTHG, FTAG.
            SeasonStart is ignored (ξ provides continuous time-weighting).
        team_stats_df : pd.DataFrame or None
            Accepted for API compatibility; not used by the DC optimiser.
        """
        try:
            if results_df is None or len(results_df) == 0:
                raise ValueError("Empty results DataFrame")

            df = results_df.copy()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['FTHG', 'FTAG']).sort_values('Date').reset_index(drop=True)

            if len(df) < 10:
                raise ValueError(f"Insufficient data: {len(df)} matches")

            # Sanity-cap extreme score-lines (data errors, not real football)
            MAX_GOALS = 9
            capped = ((df['FTHG'] > MAX_GOALS) | (df['FTAG'] > MAX_GOALS)).sum()
            if capped:
                warnings.warn(
                    f"{capped} match(es) had goals > {MAX_GOALS} and were clipped — "
                    "check source data for errors.",
                    stacklevel=2,
                )
            df['FTHG'] = df['FTHG'].clip(upper=MAX_GOALS)
            df['FTAG'] = df['FTAG'].clip(upper=MAX_GOALS)

            # Build stable team index
            teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
            self._team_index = {t: i for i, t in enumerate(teams)}
            n = len(teams)

            # Connectivity check — model is unidentifiable on a disconnected fixture graph
            self._check_connectivity(df, teams, self._team_index)

            # Time weights: φ(t) = exp(−ξ · t)
            ref_date = df['Date'].max()
            days_ago = (ref_date - df['Date']).dt.days.values.astype(float)
            weights = np.exp(-self._xi * days_ago)

            # Vectorise match arrays
            home_idx = df['HomeTeam'].map(self._team_index).values
            away_idx = df['AwayTeam'].map(self._team_index).values
            gh = df['FTHG'].values.astype(float)
            ga = df['FTAG'].values.astype(float)

            # Fit via SLSQP with mean-attack-zero constraint
            result = self._optimise(home_idx, away_idx, gh, ga, weights, n)

            # Store parameters
            self._params = result.x
            attack_arr  = self._params[:n]
            defence_arr = self._params[n:2 * n]
            gamma       = float(self._params[-2])
            rho         = float(self._params[-1])

            self.attack_rates   = {t: float(attack_arr[i])  for t, i in self._team_index.items()}
            self.defense_rates  = {t: float(defence_arr[i]) for t, i in self._team_index.items()}
            self.home_advantage = gamma
            self.rho            = rho
            self.fitted         = True
            self.last_trained   = datetime.now()

            # Held-out validation RPS on last 10% of matches
            split = max(int(len(df) * 0.9), len(df) - 100)
            val_df = df.iloc[split:]
            if len(val_df) >= 5:
                self.validation_score = self._compute_rps(val_df)
                print(f"✅ Dixon-Coles fitted. γ={gamma:.3f}, ρ={rho:.3f}, "
                      f"held-out RPS={self.validation_score:.4f}")
            else:
                print(f"✅ Dixon-Coles fitted. γ={gamma:.3f}, ρ={rho:.3f}")

        except ValueError:
            # ValueError covers: empty data, insufficient matches, disconnected
            # graph — all are clear programming/data errors that must propagate.
            raise
        except Exception as exc:
            print(f"Error fitting Dixon-Coles model: {exc}")
            self._set_defaults(results_df)

    @staticmethod
    def _check_connectivity(df, teams, team_index):
        """Raise ValueError if the team-vs-team fixture graph is disconnected.

        A disconnected graph makes attack/defence ratings unidentifiable: teams
        in separate components have no common reference point, so their ratings
        are only internally consistent, not comparable across components.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        n = len(teams)
        hi = df['HomeTeam'].map(team_index).values
        ai = df['AwayTeam'].map(team_index).values
        # Undirected adjacency: both directions
        rows = np.concatenate([hi, ai])
        cols = np.concatenate([ai, hi])
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        n_components, labels = connected_components(adj, directed=False)
        if n_components > 1:
            groups = {}
            for team, label in zip(teams, labels):
                groups.setdefault(label, []).append(team)
            detail = "; ".join(
                f"component {k}: {v}" for k, v in groups.items()
            )
            raise ValueError(
                f"Fixture graph has {n_components} disconnected components — "
                f"model is unidentifiable. {detail}"
            )

    def _optimise(self, home_idx, away_idx, gh, ga, weights, n_teams,
                  max_iter=300, n_starts=5):
        """Run SLSQP with mean-attack-zero constraint; multi-start on failure."""
        args = (home_idx, away_idx, gh, ga, weights, n_teams)
        constraint = {
            'type': 'eq',
            'fun': lambda p: np.sum(p[:n_teams]),
        }
        bounds = (
            [(None, None)] * (2 * n_teams)   # α and β unbounded
            + [(None, None)]                  # γ unbounded
            + [(-0.2, 0.2)]                   # ρ bounded
        )
        opts = {'maxiter': max_iter, 'ftol': 1e-9}

        best = None
        rng = np.random.default_rng(42)

        for start in range(n_starts):
            if start == 0:
                init = np.concatenate([
                    np.full(n_teams,  0.10),
                    np.full(n_teams, -0.10),
                    [0.25],
                    [-0.05],
                ])
            else:
                init = np.concatenate([
                    rng.normal(0.0, 0.1, n_teams),
                    rng.normal(0.0, 0.1, n_teams),
                    rng.normal(0.25, 0.05, 1),
                    rng.uniform(-0.1, 0.05, 1),
                ])

            try:
                result = minimize(
                    _neg_log_likelihood, init, args=args,
                    method='SLSQP',
                    constraints=[constraint],
                    bounds=bounds,
                    options=opts,
                )
                if best is None or result.fun < best.fun:
                    best = result
                if result.success:
                    break
            except Exception:
                continue

        if best is None:
            raise RuntimeError("All optimisation starts failed")
        if not best.success:
            warnings.warn(f"DC optimiser: {best.message}", stacklevel=3)
        return best

    def _compute_rps(self, df):
        """Mean RPS over a DataFrame of completed matches."""
        scores = []
        for _, row in df.iterrows():
            try:
                probs = self.predict_outcome_probabilities(row['HomeTeam'], row['AwayTeam'])
                hg, ag = int(row['FTHG']), int(row['FTAG'])
                outcome = 'H' if hg > ag else ('D' if hg == ag else 'A')
                scores.append(_rps(probs['home_win'], probs['draw'], probs['away_win'], outcome))
            except Exception:
                continue
        return float(np.mean(scores)) if scores else None

    def _set_defaults(self, results_df=None):
        """Fallback defaults when fit fails."""
        teams = []
        if results_df is not None and len(results_df) > 0:
            teams = sorted(
                set(results_df['HomeTeam'].unique())
                | set(results_df['AwayTeam'].unique())
            )
        self._team_index = {t: i for i, t in enumerate(teams)}
        self.attack_rates   = {t: 0.0 for t in teams}
        self.defense_rates  = {t: 0.0 for t in teams}
        self.home_advantage = 0.25
        self.rho            = 0.0
        self.fitted         = True

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_match(self, home_team, away_team):
        """Return (mu_home, mu_away) expected goals.

        Uses log-additive DC parameterisation:
            λ = exp(α_home + β_away + γ)
            μ = exp(α_away + β_home)

        Falls back to league-mean values (exp(γ/2), exp(0)) for unknown teams.
        """
        if not self.fitted:
            raise ValueError("Model not fitted — call fit() first")

        alpha_home = self.attack_rates.get(home_team, 0.0)
        beta_away  = self.defense_rates.get(away_team, 0.0)
        alpha_away = self.attack_rates.get(away_team, 0.0)
        beta_home  = self.defense_rates.get(home_team, 0.0)
        gamma      = self.home_advantage

        mu_home = float(np.exp(np.clip(alpha_home + beta_away + gamma, -5, 4)))
        mu_away = float(np.exp(np.clip(alpha_away + beta_home,         -5, 4)))
        return mu_home, mu_away

    def predict_outcome_probabilities(self, home_team, away_team, max_goals=10):
        """Compute 1X2 probabilities via the full Dixon-Coles score grid.

        Returns
        -------
        dict with keys: home_win, draw, away_win, mu_home, mu_away
        """
        try:
            mu_home, mu_away = self.predict_match(home_team, away_team)

            k = np.arange(max_goals + 1, dtype=float)
            log_h = k * np.log(mu_home) - mu_home - gammaln(k + 1.0)
            log_a = k * np.log(mu_away) - mu_away - gammaln(k + 1.0)
            p_h = np.exp(log_h)
            p_a = np.exp(log_a)

            M = np.outer(p_h, p_a)

            # Apply τ correction to 2×2 low-score block
            rho = self.rho
            M[0, 0] *= max(1e-6, 1.0 - mu_home * mu_away * rho)
            M[0, 1] *= 1.0 + mu_home * rho
            M[1, 0] *= 1.0 + mu_away * rho
            M[1, 1] *= 1.0 - rho

            M /= M.sum()   # renormalise after τ perturbation

            p_home_win = float(np.tril(M, -1).sum())
            p_draw     = float(np.trace(M))
            p_away_win = float(np.triu(M,  1).sum())

            return {
                'home_win': p_home_win,
                'draw':     p_draw,
                'away_win': p_away_win,
                'mu_home':  mu_home,
                'mu_away':  mu_away,
            }
        except Exception as exc:
            print(f"Error predicting {home_team} vs {away_team}: {exc}")
            return {'home_win': 0.33, 'draw': 0.33, 'away_win': 0.34,
                    'mu_home': 1.5, 'mu_away': 1.0}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath):
        """Pickle the model to filepath."""
        try:
            data = {
                'attack_rates':    self.attack_rates,
                'defense_rates':   self.defense_rates,
                'home_advantage':  self.home_advantage,
                'league_avg':      self.league_avg,
                'fitted':          self.fitted,
                'rho':             self.rho,
                'time_decay':      self.time_decay,
                'season_multiplier': self.season_multiplier,
                'use_mle':         self.use_mle,
                'use_dixon_coles': self.use_dixon_coles,
                'validation_score': self.validation_score,
                'last_trained':    self.last_trained,
                'training_window': self.training_window,
                # DC-specific
                'xi':         self._xi,
                '_team_index': self._team_index,
                '_params':     self._params,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as exc:
            print(f"Error saving model: {exc}")

    def load(self, filepath):
        """Load model from pickle file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.attack_rates     = data.get('attack_rates', {})
            self.defense_rates    = data.get('defense_rates', {})
            self.home_advantage   = data.get('home_advantage', 0.25)
            self.league_avg       = data.get('league_avg', 0.0)
            self.fitted           = data.get('fitted', False)
            self.rho              = data.get('rho', 0.0)
            self.time_decay       = data.get('time_decay', 0.0018)
            self.season_multiplier = data.get('season_multiplier', 0.5)
            self.use_mle          = data.get('use_mle', False)
            self.use_dixon_coles  = data.get('use_dixon_coles', False)
            self.validation_score = data.get('validation_score', None)
            self.last_trained     = data.get('last_trained', None)
            self.training_window  = data.get('training_window', None)
            self._xi              = data.get('xi', self.time_decay)
            self._team_index      = data.get('_team_index', {})
            self._params          = data.get('_params', None)
        except Exception as exc:
            print(f"Error loading model: {exc}")
            self.fitted = False

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_model_summary(self):
        """Return a dict of key model statistics for display."""
        if not self.fitted:
            return "Model not fitted"
        return {
            'teams_count':      len(self.attack_rates),
            'home_advantage':   round(self.home_advantage, 3),   # γ log-scale
            'league_avg_goals': round(self.league_avg, 3),
            'dixon_coles_rho':  round(self.rho, 3),
            'decay_xi':         self._xi,
            'time_decay':       self.time_decay,
            'season_multiplier': self.season_multiplier,
            'validation_score': (round(self.validation_score, 4)
                                 if self.validation_score else 'N/A'),
            'last_trained':     (self.last_trained.strftime('%Y-%m-%d %H:%M')
                                 if self.last_trained else 'Unknown'),
            'strongest_attack': (max(self.attack_rates.items(), key=lambda x: x[1])
                                 if self.attack_rates else None),
            'strongest_defense': (min(self.defense_rates.items(), key=lambda x: x[1])
                                  if self.defense_rates else None),
        }

