// Run with: node --test tests/js/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  fitModel, matchProbabilities, simulateSeason, seasonRounds, dataAsOf, backtest,
} from '../../site/model.js';

const fixture = JSON.parse(fs.readFileSync(new URL('./parity_fixture.json', import.meta.url)));
const data = fixture.data;
const model = fitModel(data);

const latestTeams = [...new Set(data.results.filter((r) => r[0].startsWith('2025')).flatMap((r) => [r[1], r[2]]))];
// a two-round mini season among the 2025 teams, so simulations have something to play
const miniFixtures = latestTeams.flatMap((h, i) => latestTeams.filter((a) => a !== h && (i % 2)).slice(0, 1).map((a) => ['2026-04-01', h, a]));

test('matches the Python Dixon-Coles fit', () => {
  assert.ok(Math.abs(model.gamma - fixture.python.gamma) < 0.005, `gamma ${model.gamma} vs ${fixture.python.gamma}`);
  assert.ok(Math.abs(model.rho - fixture.python.rho) < 0.005, `rho ${model.rho} vs ${fixture.python.rho}`);
  let worst = 0;
  for (const h of latestTeams) for (const a of latestTeams) {
    if (h === a) continue;
    const js = matchProbabilities(model, h, a), py = matchProbabilities(fixture.python, h, a);
    worst = Math.max(worst, Math.abs(js.home - py.home), Math.abs(js.draw - py.draw), Math.abs(js.away - py.away));
  }
  assert.ok(worst < 0.005, `largest 1X2 difference ${worst}`);
});

test('a team with no history gets a finite, below-average rating', () => {
  const withNewcomer = { ...data, teams: [...data.teams, 'Newcomer'] };
  const m = fitModel(withNewcomer);
  const k = data.teams.length;
  assert.ok(Number.isFinite(m.attack[k]) && Number.isFinite(m.defence[k]));
  assert.ok(m.attack[k] - m.defence[k] < 0, 'newcomer should be rated below average');
  const p = matchProbabilities(m, k, latestTeams[0]);
  assert.ok(Number.isFinite(p.home + p.draw + p.away));
});

test('1X2 probabilities sum to 1', () => {
  const p = matchProbabilities(model, latestTeams[0], latestTeams[1]);
  assert.ok(Math.abs(p.home + p.draw + p.away - 1) < 1e-9);
});

test('position probabilities sum to 1 per team and per position', () => {
  const season = { ...data, season: 2026, fixtures: miniFixtures };
  const rows = simulateSeason(season, model, { nSims: 2000 });
  for (const r of rows) assert.ok(Math.abs(r.positions.reduce((s, v) => s + v, 0) - 1) < 1e-9);
  for (let p = 0; p < rows.length; p++) assert.ok(Math.abs(rows.reduce((s, r) => s + r.positions[p], 0) - 1) < 1e-9);
  for (const r of rows) assert.ok(r.ptsLow <= r.expectedPts && r.expectedPts <= r.ptsHigh);
});

test('a fixed home win always gives the home team 3 points', () => {
  const [, h, a] = miniFixtures[0];
  const season = { ...data, season: 2026, fixtures: [miniFixtures[0]] };
  const rows = simulateSeason(season, model, { nSims: 500, overrides: { 0: 'H' } });
  assert.equal(rows.find((r) => r.team === h).ptsLow, 3);
  assert.equal(rows.find((r) => r.team === h).ptsHigh, 3);
  assert.equal(rows.find((r) => r.team === a).ptsHigh, 0);
});

test('replaying a round turns later results back into fixtures', () => {
  const season = { ...data, season: 2025, fixtures: [] };
  const rounds = seasonRounds(season);
  assert.equal(rounds[0].round, 0);
  const asOf = dataAsOf(season, rounds[5].date);
  assert.equal(asOf.results.length + asOf.fixtures.length, data.results.length);
  assert.ok(asOf.results.every((r) => r[0] <= rounds[5].date));
});

test('backtest beats the base-rate baseline', () => {
  const bt = backtest(data, { from: '2025-06-01', refitDays: 60 });
  assert.ok(bt.n > 50);
  assert.ok(bt.rps < bt.rpsBase, `model ${bt.rps} vs baseline ${bt.rpsBase}`);
});
