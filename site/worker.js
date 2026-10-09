// Runs the model off the main thread. The page starts two of these: one answers
// forecast requests (what-if clicks), the other does the slower background jobs.
import {
  fitModel, matchProbabilities, simulateSeason, seasonRounds, dataAsOf, backtest,
} from './model.js';

let data, model;

self.onmessage = ({ data: msg }) => {
  const t0 = performance.now();

  if (msg.type === 'init') {
    data = msg.data;
    model = fitModel(data);
    const matches = data.fixtures.map(([, h, a]) => matchProbabilities(model, h, a));
    self.postMessage({ type: 'model', model, matches, ms: performance.now() - t0 });
  }

  if (msg.type === 'forecast') {
    const rows = simulateSeason(data, model, msg);
    self.postMessage({ type: 'forecast', id: msg.id, rows, ms: performance.now() - t0 });
  }

  if (msg.type === 'race') {
    const rounds = seasonRounds(data);
    const out = [];
    let m = null;
    rounds.forEach((r, k) => {
      const asOf = dataAsOf(data, r.date);
      m = fitModel(asOf, { init: m });
      const rows = simulateSeason(asOf, m, { nSims: msg.nSims ?? 2000, seed: k + 1 });
      out.push({ ...r, positions: Object.fromEntries(rows.map((x) => [x.team, x.positions])) });
      self.postMessage({ type: 'race-progress', done: k + 1, total: rounds.length });
    });
    self.postMessage({ type: 'race', rounds: out, ms: performance.now() - t0 });
  }

  if (msg.type === 'backtest') {
    const result = backtest(data, {
      from: `${data.season - 1}-01-01`,
      onProgress: (p) => self.postMessage({ type: 'backtest-progress', done: p }),
    });
    self.postMessage({ type: 'backtest', ...result, from: `${data.season - 1}`, ms: performance.now() - t0 });
  }
};
