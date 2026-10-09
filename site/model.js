// Dixon-Coles model + Monte Carlo season simulation, run in the visitor's browser.
// Mirrors core/models/poisson_model.py:
//   λ = exp(α_home + β_away + γ)   μ = exp(α_away + β_home)
//   time weights φ(t) = exp(−ξ·days_ago), low-score correction τ with parameter ρ.
// Fitting uses closed-form coordinate updates for the Poisson part, then a 1-D
// search for ρ (the Python code fits everything jointly with SLSQP; results agree
// to within rounding).

export const MAX_GOALS = 10;
const S = MAX_GOALS + 1;
const DAY_MS = 86400000;
const MIN_WEIGHT = 5;   // teams with less (time-weighted) match history lean on the promoted-team prior

export function fitModel(data, { xi = 0.0018, maxIter = 500, tol = 1e-7, init = null } = {}) {
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

  // Goals scored / conceded and total match weight per team — constant across iterations.
  const scored = new Float64Array(n), conceded = new Float64Array(n), weight = new Float64Array(n);
  let homeGoals = 0;
  for (let k = 0; k < m; k++) {
    scored[H[k]] += W[k] * GH[k]; scored[A[k]] += W[k] * GA[k];
    conceded[A[k]] += W[k] * GH[k]; conceded[H[k]] += W[k] * GA[k];
    weight[H[k]] += W[k]; weight[A[k]] += W[k];
    homeGoals += W[k] * GH[k];
  }
  const hasData = Array.from(weight, (w) => w > 0);
  const nData = hasData.filter(Boolean).length;

  const att = new Float64Array(n), def = new Float64Array(n);
  let gamma = 0.25;
  if (init) { att.set(init.attack); def.set(init.defence); gamma = init.gamma; }
  const den = new Float64Array(n);

  for (let it = 0; it < maxIter; it++) {
    let change = 0;
    const eg = Math.exp(gamma);

    // α update
    den.fill(0);
    for (let k = 0; k < m; k++) {
      den[H[k]] += W[k] * Math.exp(def[A[k]]) * eg;
      den[A[k]] += W[k] * Math.exp(def[H[k]]);
    }
    for (let i = 0; i < n; i++) {
      const v = hasData[i] ? Math.log(Math.max(scored[i], 1e-9) / den[i]) : 0;
      change = Math.max(change, Math.abs(v - att[i])); att[i] = v;
    }

    // β update
    den.fill(0);
    for (let k = 0; k < m; k++) {
      den[A[k]] += W[k] * Math.exp(att[H[k]]) * eg;
      den[H[k]] += W[k] * Math.exp(att[A[k]]);
    }
    for (let i = 0; i < n; i++) {
      const v = hasData[i] ? Math.log(Math.max(conceded[i], 1e-9) / den[i]) : 0;
      change = Math.max(change, Math.abs(v - def[i])); def[i] = v;
    }

    // γ update
    let gd = 0;
    for (let k = 0; k < m; k++) gd += W[k] * Math.exp(att[H[k]] + def[A[k]]);
    gamma = Math.log(homeGoals / gd);

    // identifiability: mean attack = 0 over teams with data (shift is absorbed by defence)
    let shift = 0;
    for (let i = 0; i < n; i++) if (hasData[i]) shift += att[i];
    shift /= nData;
    for (let i = 0; i < n; i++) if (hasData[i]) { att[i] -= shift; def[i] += shift; }

    if (change < tol) break;
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

  // Promoted-team prior: teams with little or no recent history (e.g. newly promoted)
  // are pulled toward the average of the 3 weakest teams of the latest season.
  const latestYear = rows[m - 1][0].slice(0, 4);
  const latestTeams = new Set();
  for (const [date, h, a] of rows) if (date.startsWith(latestYear)) { latestTeams.add(h); latestTeams.add(a); }
  const weakest = [...latestTeams].filter((i) => weight[i] >= MIN_WEIGHT)
    .sort((x, y) => (att[x] - def[x]) - (att[y] - def[y])).slice(0, 3);
  if (weakest.length) {
    const priorAtt = weakest.reduce((s, i) => s + att[i], 0) / weakest.length;
    const priorDef = weakest.reduce((s, i) => s + def[i], 0) / weakest.length;
    for (let i = 0; i < n; i++) {
      if (weight[i] >= MIN_WEIGHT) continue;
      const b = weight[i] / MIN_WEIGHT;
      att[i] = b * att[i] + (1 - b) * priorAtt;
      def[i] = b * def[i] + (1 - b) * priorDef;
    }
  }

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
  const p = new Float64Array(S);
  p[0] = Math.exp(-mean);
  for (let k = 1; k < S; k++) p[k] = p[k - 1] * mean / k;
  return p;
}

// Full score grid, flattened: grid[h * S + a] = P(home h, away a)
export function scoreGrid(model, home, away) {
  const [lam, mu] = expectedGoals(model, home, away);
  const ph = poissonPmf(lam), pa = poissonPmf(mu);
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
  let win = 0, draw = 0, loss = 0, best = 0, bestScore = [0, 0];
  for (let h = 0; h < S; h++) {
    for (let a = 0; a < S; a++) {
      const p = grid[h * S + a];
      if (h > a) win += p; else if (h === a) draw += p; else loss += p;
      if (p > best) { best = p; bestScore = [h, a]; }
    }
  }
  return { home: win, draw, away: loss, lam, mu, likelyScore: bestScore, grid };
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

export function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let r = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

const outcomeOf = (h, a) => (h > a ? 'H' : h === a ? 'D' : 'A');

// Simulate the rest of the season `nSims` times. Each match score is drawn from
// its Dixon-Coles score grid. `overrides` maps a fixture index to 'H' | 'D' | 'A';
// such a match is drawn from the grid restricted to that outcome, so goal
// difference stays realistic. Ranking: points, goal difference, goals scored, then random.
export function simulateSeason(data, model, { nSims = 10000, seed = 1, overrides = {} } = {}) {
  const rand = mulberry32(seed);
  const { table, teams } = currentTable(data);
  const T = teams.length;
  const slot = new Map(teams.map((t, i) => [t, i]));

  const matches = data.fixtures.map(([, h, a], idx) => {
    const { grid } = scoreGrid(model, h, a);
    const fixed = overrides[idx];
    const cdf = new Float64Array(grid.length);
    let c = 0;
    for (let i = 0; i < grid.length; i++) {
      if (!fixed || outcomeOf(Math.floor(i / S), i % S) === fixed) c += grid[i];
      cdf[i] = c;
    }
    return { h: slot.get(h), a: slot.get(a), cdf };
  });

  const MAX_PTS = 3 * (data.gamesPerTeam ?? 30) + 1;
  const positions = Array.from({ length: T }, () => new Float64Array(T));
  const ptsHist = Array.from({ length: T }, () => new Float64Array(MAX_PTS));
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
      while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] <= u) lo = mid + 1; else hi = mid; }
      const hg = Math.floor(lo / S), ag = lo % S;
      gd[h] += hg - ag; gd[a] += ag - hg; gf[h] += hg; gf[a] += ag;
      if (hg > ag) pts[h] += 3; else if (hg < ag) pts[a] += 3; else { pts[h]++; pts[a]++; }
    }
    order.sort((x, y) => pts[y] - pts[x] || gd[y] - gd[x] || gf[y] - gf[x] || tiebreak[y] - tiebreak[x]);
    for (let p = 0; p < T; p++) positions[order[p]][p]++;
    for (let i = 0; i < T; i++) { ptsSum[i] += pts[i]; ptsHist[i][Math.min(pts[i], MAX_PTS - 1)]++; }
  }

  const quantile = (hist, q) => {
    let c = 0;
    for (let p = 0; p < hist.length; p++) { c += hist[p]; if (c >= q * nSims) return p; }
    return hist.length - 1;
  };
  return teams.map((team, i) => ({
    team,
    ...table[team],
    expectedPts: ptsSum[i] / nSims,
    ptsLow: quantile(ptsHist[i], 0.1),
    ptsHigh: quantile(ptsHist[i], 0.9),
    positions: Array.from(positions[i], (c) => c / nSims),
  }));
}

// Split this season into rounds: cutoff k is the date of the (k·T/2)-th match.
// Returns [{ round, date }] with round 0 = before the season started.
export function seasonRounds(data) {
  const season = data.results.filter((r) => r[0].startsWith(String(data.season)));
  const perRound = currentTable(data).teams.length / 2;
  const rounds = [{ round: 0, date: null }];
  for (let k = 1; k * perRound <= season.length; k++) rounds.push({ round: k, date: season[k * perRound - 1][0] });
  return rounds;
}

// The data as it looked after `date` (null = before the season): later results become fixtures.
export function dataAsOf(data, date) {
  const seasonStart = `${data.season}-01-01`;
  const cutoff = date ?? `${data.season - 1}-12-31`;
  const results = data.results.filter((r) => r[0] <= cutoff);
  const replayed = data.results.filter((r) => r[0] > cutoff && r[0] >= seasonStart).map(([d, h, a]) => [d, h, a]);
  return { ...data, results, fixtures: [...replayed, ...data.fixtures] };
}

// Rolling out-of-sample test: predict every match from `from` onward with a model
// fitted only on earlier matches (refitted every `refitDays`). Returns the mean
// ranked probability score (RPS) for the model and for a base-rate baseline,
// plus calibration bins.
export function backtest(data, { from, refitDays = 14, onProgress } = {}) {
  const test = data.results.filter((r) => r[0] >= from);
  const bins = Array.from({ length: 10 }, () => ({ predicted: 0, observed: 0, n: 0 }));
  const rpsOf = (P, o) => {
    const E = [0, 0, 0]; E[o] = 1;
    return ((P[0] - E[0]) ** 2 + (P[0] + P[1] - E[0] - E[1]) ** 2) / 2;
  };
  let model = null, refitAfter = '', base = null, rps = 0, rpsBase = 0;
  test.forEach((r, k) => {
    if (!model || r[0] > refitAfter) {
      const past = data.results.filter((x) => x[0] < r[0]);
      model = fitModel({ ...data, results: past }, { init: model });
      const counts = [0, 0, 0];
      for (const x of past) counts[x[3] > x[4] ? 0 : x[3] === x[4] ? 1 : 2]++;
      base = counts.map((c) => c / past.length);
      const next = new Date(r[0]); next.setDate(next.getDate() + refitDays);
      refitAfter = next.toISOString().slice(0, 10);
      onProgress?.(k / test.length);
    }
    const p = matchProbabilities(model, r[1], r[2]);
    const P = [p.home, p.draw, p.away];
    const o = r[3] > r[4] ? 0 : r[3] === r[4] ? 1 : 2;
    rps += rpsOf(P, o); rpsBase += rpsOf(base, o);
    for (let j = 0; j < 3; j++) {
      const b = bins[Math.min(9, Math.floor(P[j] * 10))];
      b.predicted += P[j]; b.observed += j === o ? 1 : 0; b.n++;
    }
  });
  return {
    n: test.length,
    rps: rps / test.length,
    rpsBase: rpsBase / test.length,
    bins: bins.filter((b) => b.n).map((b) => ({ predicted: b.predicted / b.n, observed: b.observed / b.n, n: b.n })),
  };
}
