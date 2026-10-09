// Dixon-Coles model + Monte Carlo season simulation, run in the visitor's browser.
// Mirrors core/models/poisson_model.py:
//   λ = exp(α_home + β_away + γ)   μ = exp(α_away + β_home)
//   time weights φ(t) = exp(−ξ·days_ago), low-score correction τ with parameter ρ.
// Fitting uses closed-form coordinate updates for the Poisson part, then a 1-D
// search for ρ (the Python code fits everything jointly with SLSQP; results agree
// to within rounding).

const MAX_GOALS = 10;
const DAY_MS = 86400000;

export function fitModel(data, { xi = 0.0018, iterations = 200 } = {}) {
  const n = data.teams.length;
  const rows = data.results;
  const ref = Date.parse(rows[rows.length - 1][0]);

  const m = rows.length;
  const H = new Int32Array(m), A = new Int32Array(m);
  const GH = new Float64Array(m), GA = new Float64Array(m), W = new Float64Array(m);
  rows.forEach(([date, h, a, gh, ga], k) => {
    H[k] = h; A[k] = a;
    GH[k] = Math.min(gh, 9); GA[k] = Math.min(ga, 9);   // same cap as Python
    W[k] = Math.exp(-xi * (ref - Date.parse(date)) / DAY_MS);
  });

  // Goals scored / conceded per team (weighted) — the numerators never change.
  const scored = new Float64Array(n), conceded = new Float64Array(n);
  let homeGoals = 0;
  for (let k = 0; k < m; k++) {
    scored[H[k]] += W[k] * GH[k]; scored[A[k]] += W[k] * GA[k];
    conceded[A[k]] += W[k] * GH[k]; conceded[H[k]] += W[k] * GA[k];
    homeGoals += W[k] * GH[k];
  }

  const att = new Float64Array(n), def = new Float64Array(n);
  let gamma = 0.25;
  const den = new Float64Array(n);

  for (let it = 0; it < iterations; it++) {
    // α update
    den.fill(0);
    const eg = Math.exp(gamma);
    for (let k = 0; k < m; k++) {
      den[H[k]] += W[k] * Math.exp(def[A[k]]) * eg;
      den[A[k]] += W[k] * Math.exp(def[H[k]]);
    }
    for (let i = 0; i < n; i++) att[i] = Math.log(Math.max(scored[i], 1e-9) / den[i]);

    // β update
    den.fill(0);
    for (let k = 0; k < m; k++) {
      den[A[k]] += W[k] * Math.exp(att[H[k]]) * eg;
      den[H[k]] += W[k] * Math.exp(att[A[k]]);
    }
    for (let i = 0; i < n; i++) def[i] = Math.log(Math.max(conceded[i], 1e-9) / den[i]);

    // γ update
    let gd = 0;
    for (let k = 0; k < m; k++) gd += W[k] * Math.exp(att[H[k]] + def[A[k]]);
    gamma = Math.log(homeGoals / gd);

    // identifiability: mean attack = 0 (shift is absorbed by defence)
    const shift = att.reduce((s, v) => s + v, 0) / n;
    for (let i = 0; i < n; i++) { att[i] -= shift; def[i] += shift; }
  }

  // ρ: maximise Σ w·log τ over [−0.2, 0.2] by golden-section search
  const lowScore = [];
  for (let k = 0; k < m; k++) {
    if (GH[k] <= 1 && GA[k] <= 1) {
      const lam = Math.exp(att[H[k]] + def[A[k]] + gamma), mu = Math.exp(att[A[k]] + def[H[k]]);
      lowScore.push([GH[k], GA[k], lam, mu, W[k]]);
    }
  }
  const ll = (rho) => {
    let s = 0;
    for (const [x, y, lam, mu, w] of lowScore) {
      const t = tau(x, y, lam, mu, rho);
      if (t <= 0) return -Infinity;
      s += w * Math.log(t);
    }
    return s;
  };
  let lo = -0.2, hi = 0.2;
  const g = (Math.sqrt(5) - 1) / 2;
  for (let it = 0; it < 60; it++) {
    const c = hi - g * (hi - lo), d = lo + g * (hi - lo);
    if (ll(c) > ll(d)) hi = d; else lo = c;
  }
  const rho = (lo + hi) / 2;

  return { attack: Array.from(att), defence: Array.from(def), gamma, rho };
}

function tau(x, y, lam, mu, rho) {
  if (x === 0 && y === 0) return 1 - lam * mu * rho;
  if (x === 0 && y === 1) return 1 + lam * rho;
  if (x === 1 && y === 0) return 1 + mu * rho;
  if (x === 1 && y === 1) return 1 - rho;
  return 1;
}

export function expectedGoals(model, home, away) {
  const clip = (v) => Math.min(4, Math.max(-5, v));
  return [
    Math.exp(clip(model.attack[home] + model.defence[away] + model.gamma)),
    Math.exp(clip(model.attack[away] + model.defence[home])),
  ];
}

function poissonPmf(mean) {
  const p = new Float64Array(MAX_GOALS + 1);
  p[0] = Math.exp(-mean);
  for (let k = 1; k <= MAX_GOALS; k++) p[k] = p[k - 1] * mean / k;
  return p;
}

// Full score grid, flattened: grid[h * (MAX_GOALS+1) + a] = P(home h, away a)
export function scoreGrid(model, home, away) {
  const [lam, mu] = expectedGoals(model, home, away);
  const ph = poissonPmf(lam), pa = poissonPmf(mu);
  const S = MAX_GOALS + 1;
  const grid = new Float64Array(S * S);
  let total = 0;
  for (let h = 0; h < S; h++) {
    for (let a = 0; a < S; a++) {
      let p = ph[h] * pa[a];
      if (h <= 1 && a <= 1) p *= Math.max(1e-6, tau(h, a, lam, mu, model.rho));
      grid[h * S + a] = p;
      total += p;
    }
  }
  for (let i = 0; i < grid.length; i++) grid[i] /= total;
  return { grid, lam, mu };
}

export function matchProbabilities(model, home, away) {
  const { grid, lam, mu } = scoreGrid(model, home, away);
  const S = MAX_GOALS + 1;
  let win = 0, draw = 0, loss = 0, best = 0, bestScore = [0, 0];
  for (let h = 0; h < S; h++) {
    for (let a = 0; a < S; a++) {
      const p = grid[h * S + a];
      if (h > a) win += p; else if (h === a) draw += p; else loss += p;
      if (p > best) { best = p; bestScore = [h, a]; }
    }
  }
  return { home: win, draw, away: loss, lam, mu, likelyScore: bestScore };
}

// Current table from this season's results.
export function currentTable(data) {
  const n = data.teams.length;
  const t = Array.from({ length: n }, () => ({ played: 0, pts: 0, gf: 0, ga: 0 }));
  const active = new Set();
  for (const [date, h, a, gh, ga] of data.results) {
    if (!date.startsWith(String(data.season))) continue;
    active.add(h); active.add(a);
    t[h].played++; t[a].played++;
    t[h].gf += gh; t[h].ga += ga; t[a].gf += ga; t[a].ga += gh;
    if (gh > ga) t[h].pts += 3; else if (gh < ga) t[a].pts += 3; else { t[h].pts++; t[a].pts++; }
  }
  for (const [, h, a] of data.fixtures) { active.add(h); active.add(a); }
  return { table: t, teams: [...active].sort((x, y) => x - y) };
}

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let r = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

// Simulate the rest of the season `nSims` times. Each match score is drawn from
// its Dixon-Coles score grid. Ranking: points, goal difference, goals scored, then random.
export function simulateSeason(data, model, { nSims = 10000, seed = 1 } = {}) {
  const rand = mulberry32(seed);
  const { table, teams } = currentTable(data);
  const T = teams.length;
  const slot = new Map(teams.map((t, i) => [t, i]));
  const S = MAX_GOALS + 1;

  const matches = data.fixtures.map(([, h, a]) => {
    const { grid } = scoreGrid(model, h, a);
    const cdf = new Float64Array(grid.length);
    let c = 0;
    for (let i = 0; i < grid.length; i++) { c += grid[i]; cdf[i] = c; }
    return { h: slot.get(h), a: slot.get(a), cdf };
  });

  const positions = Array.from({ length: T }, () => new Float64Array(T));
  const ptsSum = new Float64Array(T);
  const pts = new Int32Array(T), gd = new Int32Array(T), gf = new Int32Array(T);
  const tiebreak = new Float64Array(T);
  const order = teams.map((_, i) => i);

  for (let s = 0; s < nSims; s++) {
    for (let i = 0; i < T; i++) {
      const row = table[teams[i]];
      pts[i] = row.pts; gd[i] = row.gf - row.ga; gf[i] = row.gf; tiebreak[i] = rand();
    }
    for (const { h, a, cdf } of matches) {
      const u = rand() * cdf[cdf.length - 1];
      let lo = 0, hi = cdf.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
      const hg = Math.floor(lo / S), ag = lo % S;
      gd[h] += hg - ag; gd[a] += ag - hg; gf[h] += hg; gf[a] += ag;
      if (hg > ag) pts[h] += 3; else if (hg < ag) pts[a] += 3; else { pts[h]++; pts[a]++; }
    }
    order.sort((x, y) => pts[y] - pts[x] || gd[y] - gd[x] || gf[y] - gf[x] || tiebreak[y] - tiebreak[x]);
    for (let p = 0; p < T; p++) positions[order[p]][p]++;
    for (let i = 0; i < T; i++) ptsSum[i] += pts[i];
  }

  return teams.map((team, i) => ({
    team,
    ...table[team],
    expectedPts: ptsSum[i] / nSims,
    positions: Array.from(positions[i], (c) => c / nSims),
  }));
}
