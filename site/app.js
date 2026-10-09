import { lineChart, scatter, calibration, showTip, hideTip, SERIES } from './charts.js';

const DISPLAY_NAMES = {
  Djurgarden: 'Djurgården', Goteborg: 'IFK Göteborg', Hacken: 'BK Häcken', 'Malmo FF': 'Malmö FF',
  Mjallby: 'Mjällby', Orgryte: 'Örgryte', 'Vasteras SK': 'Västerås SK', Norrkoping: 'Norrköping',
  Varnamo: 'Värnamo', Orebro: 'Örebro', Oster: 'Öster', Ostersunds: 'Östersund',
  Jonkopings: 'Jönköpings Södra', Atvidabergs: 'Åtvidaberg',
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const pct = (p) => (p >= 0.995 ? '>99%' : p > 0 && p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`);
const fmtDate = (iso, opts) => new Date(iso + 'T12:00:00').toLocaleDateString('en-GB', opts);
const sum = (arr, from, to) => arr.slice(from, to).reduce((s, v) => s + v, 0);

const forecaster = new Worker('worker.js', { type: 'module' });
const background = new Worker('worker.js', { type: 'module' });

let data, model, matches, name;
let rows = [];                 // latest forecast
let overrides = {};            // fixture index → 'H' | 'D' | 'A'
let requestId = 0;
let raceRounds = null, raceMetric = 'title';

// ── start-up ────────────────────────────────────────────────────────────────

async function init() {
  data = await (await fetch('data.json', { cache: 'no-cache' })).json();
  name = (i) => DISPLAY_NAMES[data.teams[i]] ?? data.teams[i];
  const lastResult = data.results[data.results.length - 1][0];
  $('#season').textContent = data.season;
  $('#meta').textContent =
    `Results up to ${fmtDate(lastResult, { day: 'numeric', month: 'long', year: 'numeric' })} · ` +
    `${data.fixtures.length} matches left · data updated ${fmtDate(data.generated.slice(0, 10), { day: 'numeric', month: 'short' })}`;
  overrides = readHash();

  forecaster.onmessage = ({ data: msg }) => {
    if (msg.type === 'model') {
      ({ model, matches } = msg);
      renderFixtures();
      syncOverrideUi();
      renderStrength();
      run(1);
    }
    if (msg.type === 'forecast' && msg.id === requestId) showForecast(msg);
  };
  forecaster.postMessage({ type: 'init', data });

  background.onmessage = ({ data: msg }) => {
    if (msg.type === 'race-progress') $('#race-progress').value = msg.done / msg.total;
    if (msg.type === 'race') { raceRounds = msg.rounds; renderRace(); background.postMessage({ type: 'backtest' }); }
    if (msg.type === 'backtest-progress') $('#acc-progress').value = msg.done;
    if (msg.type === 'backtest') renderAccuracy(msg);
  };
  background.postMessage({ type: 'init', data });
  background.postMessage({ type: 'race', nSims: 2000 });

  $('#run').addEventListener('click', () => run(Math.floor(Math.random() * 2 ** 31)));
  $('#nsims').addEventListener('change', () => run(1));
  $('#reset').addEventListener('click', () => { overrides = {}; afterOverrideChange(); });
  $('#race-tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-metric]');
    if (!b) return;
    raceMetric = b.dataset.metric;
    for (const t of $('#race-tabs').children) t.setAttribute('aria-selected', t === b);
    renderRace();
  });
}

function run(seed) {
  const nSims = Number($('#nsims').value);
  $('#run').disabled = true;
  $('#status').textContent = `Simulating ${nSims.toLocaleString('en')} seasons…`;
  forecaster.postMessage({ type: 'forecast', id: ++requestId, nSims, seed, overrides });
}

function showForecast(msg) {
  rows = msg.rows.sort((a, b) => b.expectedPts - a.expectedPts);
  renderTiles();
  renderTable();
  renderHeatmap();
  const nSims = Number($('#nsims').value);
  $('#status').textContent = `Played out ${nSims.toLocaleString('en')} seasons in your browser in ${(msg.ms / 1000).toFixed(1)} s`;
  $('#run').disabled = false;
}

// ── what-if overrides, kept in the URL hash as Home-Away:H ──────────────────

const fixtureKey = (i) => `${data.teams[data.fixtures[i][1]]}-${data.teams[data.fixtures[i][2]]}`;

function readHash() {
  const m = location.hash.match(/fix=([^&]*)/);
  if (!m) return {};
  const wanted = new Map(decodeURIComponent(m[1]).split(',').map((s) => {
    const k = s.lastIndexOf(':');
    return [s.slice(0, k), s.slice(k + 1)];
  }));
  const out = {};
  data.fixtures.forEach((_, i) => {
    const o = wanted.get(fixtureKey(i));
    if (['H', 'D', 'A'].includes(o)) out[i] = o;
  });
  return out;
}

function afterOverrideChange() {
  const parts = Object.entries(overrides).map(([i, o]) => `${fixtureKey(i)}:${o}`);
  history.replaceState(null, '', parts.length ? `#fix=${encodeURIComponent(parts.join(','))}` : location.pathname);
  syncOverrideUi();
  run(1);
}

function syncOverrideUi() {
  const n = Object.keys(overrides).length;
  $('#fixed-chip').hidden = !n;
  $('#fixed-count').textContent = `${n} match${n === 1 ? '' : 'es'} fixed`;
  for (const btn of document.querySelectorAll('.pick button')) {
    btn.setAttribute('aria-pressed', overrides[btn.dataset.idx] === btn.dataset.o);
  }
  for (const m of document.querySelectorAll('.match')) m.classList.toggle('fixed', !!overrides[m.dataset.idx]);
}

// ── tiles ───────────────────────────────────────────────────────────────────

function renderTiles() {
  const T = rows.length;
  const best = (f) => rows.reduce((a, b) => (f(b) > f(a) ? b : a));
  const champ = best((r) => r.positions[0]);
  const playoff = best((r) => r.positions[T - 3]);
  const down = best((r) => sum(r.positions, T - 2, T));
  const tile = (label, team, p, note) =>
    `<div class="tile"><div class="tile-label">${label}</div><div class="tile-value">${pct(p)}</div><div class="tile-team">${esc(name(team))}</div><div class="tile-note">${note}</div></div>`;
  $('#tiles').innerHTML =
    tile('Title favourite', champ.team, champ.positions[0], 'chance of winning the league') +
    tile('Likeliest in the playoff', playoff.team, playoff.positions[T - 3], 'chance of finishing 14th') +
    tile('Likeliest relegated', down.team, sum(down.positions, T - 2, T), 'chance of finishing 15th or 16th');
}

// ── table with points range ─────────────────────────────────────────────────

function probCell(p) {
  return `<td class="num prob">${pct(p)}<span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span></td>`;
}

function renderTable() {
  const T = rows.length;
  const lo = Math.floor(Math.min(...rows.map((r) => r.pts)) / 10) * 10;
  const hi = Math.ceil(Math.max(...rows.map((r) => r.ptsHigh)) / 10) * 10;
  const at = (v) => `${(((v - lo) / (hi - lo)) * 100).toFixed(2)}%`;
  const scale = [];
  for (let v = lo; v <= hi; v += 10) scale.push(`<span style="left:${at(v)}">${v}</span>`);
  $('#range-scale').innerHTML = scale.join('');

  $('#table tbody').innerHTML = rows.map((r, i) => {
    const zone = i < 3 ? 'zone-eu' : i >= T - 3 ? 'zone-down' : '';
    const gd = r.gf - r.ga;
    return `<tr class="${zone}">
      <td class="num dim">${i + 1}</td>
      <td class="team">${esc(name(r.team))}</td>
      <td class="num dim opt">${r.played}</td>
      <td class="num">${r.pts}</td>
      <td class="num dim opt">${gd > 0 ? '+' : ''}${gd}</td>
      <td class="range" data-team="${r.team}"><div class="range-track">
        <span class="rbar" style="left:${at(r.ptsLow)};width:calc(${at(r.ptsHigh)} - ${at(r.ptsLow)})"></span>
        <span class="rdot" style="left:${at(r.pts)}"></span>
        <span class="rtick" style="left:${at(r.expectedPts)}"></span>
      </div></td>
      <td class="num"><b>${r.expectedPts.toFixed(1)}</b></td>
      ${probCell(r.positions[0])}
      ${probCell(sum(r.positions, 0, 3))}
      ${probCell(r.positions[T - 3])}
      ${probCell(sum(r.positions, T - 2, T))}
    </tr>`;
  }).join('');

  for (const cell of document.querySelectorAll('td.range')) {
    const r = rows.find((x) => x.team === Number(cell.dataset.team));
    cell.addEventListener('pointermove', (e) => showTip(e, name(r.team), [
      { value: r.pts, label: 'points now' },
      { value: r.expectedPts.toFixed(1), label: 'expected final points' },
      { value: `${r.ptsLow}–${r.ptsHigh}`, label: 'likely range (80%)' },
    ]));
    cell.addEventListener('pointerleave', hideTip);
  }
}

// ── position heatmap ────────────────────────────────────────────────────────

function ordinal(n) {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function renderHeatmap() {
  const head = `<tr><th></th>${rows.map((_, p) => `<th>${p + 1}</th>`).join('')}</tr>`;
  const body = rows.map((r) => `<tr><th class="rowhead">${esc(name(r.team))}</th>${r.positions.map((p, pos) => {
    const share = Math.round(Math.min(1, p / 0.6) * 100);   // 60%+ = full colour
    const label = p >= 0.005 ? Math.round(p * 100) : '';
    const tip = `${name(r.team)} finishes ${ordinal(pos + 1)}: ${(p * 100).toFixed(1)}%`;
    return `<td class="${share > 45 ? 'dark' : 'light'}" title="${esc(tip)}"
      style="background:color-mix(in oklab, var(--heat) ${share}%, var(--bg))">${label}</td>`;
  }).join('')}</tr>`).join('');
  $('#heatmap').innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
}

// ── race over the season ────────────────────────────────────────────────────

const METRICS = {
  title: { label: 'win the league', of: (pos) => pos[0] },
  top3: { label: 'finish top 3', of: (pos) => sum(pos, 0, 3) },
  down: { label: 'be relegated', of: (pos) => sum(pos, pos.length - 2, pos.length) },
};

function renderRace() {
  if (!raceRounds) return;
  const metric = METRICS[raceMetric];
  const teams = Object.keys(raceRounds[raceRounds.length - 1].positions).map(Number);
  const all = teams.map((t) => ({ team: t, values: raceRounds.map((r) => (r.positions[t] ? metric.of(r.positions[t]) : 0)) }));
  const ranked = all.filter((s) => Math.max(...s.values) >= 0.05)
    .sort((a, b) => b.values[b.values.length - 1] - a.values[a.values.length - 1] || Math.max(...b.values) - Math.max(...a.values));
  const shown = new Set(ranked.slice(0, SERIES.length).map((s) => s.team));
  const series = all.map((s) => {
    const k = ranked.findIndex((x) => x.team === s.team);
    return { name: name(s.team), values: s.values, color: SERIES[k], muted: !shown.has(s.team) };
  });
  const legend = $('#race-legend');
  legend.replaceChildren(...ranked.slice(0, SERIES.length).map((s, k) => {
    const item = document.createElement('span');
    item.className = 'key';
    item.innerHTML = `<i class="kline" style="background:${SERIES[k]}"></i>`;
    item.append(name(s.team));
    return item;
  }));
  lineChart($('#race'), {
    series,
    xLabel: 'Round',
    xTickLabel: (i) => (i === 0 ? 'Start' : i),
    tipTitle: (i) => (i === 0 ? 'Before the season' : `After round ${i} (${fmtDate(raceRounds[i].date, { day: 'numeric', month: 'short' })})`),
  });
  $('#race').setAttribute('aria-label', `Chance to ${metric.label} after each round`);
}

// ── team strength scatter ───────────────────────────────────────────────────

function renderStrength() {
  const teams = [...new Set(data.fixtures.flatMap(([, h, a]) => [h, a]))];
  const mean = (f) => teams.reduce((s, t) => s + f(t), 0) / teams.length;
  const avgAtt = mean((t) => model.attack[t]), avgDef = mean((t) => model.defence[t]);
  const half = model.gamma / 2;
  const points = teams.map((t) => {
    const scores = Math.exp(model.attack[t] + avgDef + half);
    const concedes = Math.exp(avgAtt + model.defence[t] + half);
    return {
      x: scores, y: concedes, label: name(t),
      tip: [{ value: scores.toFixed(2), label: 'goals scored per match' }, { value: concedes.toFixed(2), label: 'goals conceded per match' }],
    };
  });
  scatter($('#strength'), {
    points,
    xLabel: 'Goals scored per match →',
    yLabel: '← Goals conceded per match',
    xRef: Math.exp(avgAtt + avgDef + half),
    yRef: Math.exp(avgAtt + avgDef + half),
    yInvert: true,
    quadrants: ['Weak attack · tight defence', 'Strong attack · tight defence', 'Weak attack · leaky defence', 'Strong attack · leaky defence'],
  });
}

// ── fixtures with what-if picks and score grids ─────────────────────────────

function renderFixtures() {
  const byDate = new Map();
  data.fixtures.forEach((f, i) => {
    if (!byDate.has(f[0])) byDate.set(f[0], []);
    byDate.get(f[0]).push(i);
  });
  // show the next two rounds (one match per team each); the rest behind a button
  const nTeams = new Set(data.fixtures.flatMap(([, h, a]) => [h, a])).size;
  let shown = 0;
  const firstHidden = [...byDate].findIndex(([, idxs]) => (shown += idxs.length) > nTeams);
  $('#fixtures').innerHTML = [...byDate].map(([date, idxs], d) => `
    <div class="day"${firstHidden >= 0 && d >= firstHidden ? ' data-later hidden' : ''}>
      <h3>${fmtDate(date, { weekday: 'long', day: 'numeric', month: 'long' })}</h3>
      ${idxs.map((i) => {
        const [, h, a] = data.fixtures[i];
        const p = matches[i];
        const pick = (o, label, title) =>
          `<button type="button" data-idx="${i}" data-o="${o}" aria-pressed="${overrides[i] === o}" title="${esc(title)}">${label}</button>`;
        return `<div class="match${overrides[i] ? ' fixed' : ''}" data-idx="${i}">
          <button type="button" class="teams" aria-expanded="false">${esc(name(h))} – ${esc(name(a))}</button>
          <div>
            <div class="split" role="img" aria-label="Home ${pct(p.home)}, draw ${pct(p.draw)}, away ${pct(p.away)}">
              <i class="home" style="flex:${p.home}"></i><i class="draw" style="flex:${p.draw}"></i><i class="away" style="flex:${p.away}"></i>
            </div>
            <div class="split-nums"><span>${pct(p.home)}</span><span>${pct(p.draw)}</span><span>${pct(p.away)}</span></div>
          </div>
          <div class="pick" role="group" aria-label="Fix the result">
            ${pick('H', '1', `${name(h)} win`)}${pick('D', 'X', 'Draw')}${pick('A', '2', `${name(a)} win`)}
          </div>
          <div class="grid-slot" hidden></div>
        </div>`;
      }).join('')}
    </div>`).join('') + (firstHidden >= 0
    ? `<button type="button" id="show-all" class="show-all">Show all ${data.fixtures.length} remaining matches</button>` : '');

  $('#fixtures').addEventListener('click', (e) => {
    if (e.target.id === 'show-all') {
      for (const d of document.querySelectorAll('.day[data-later]')) d.hidden = false;
      e.target.remove();
      return;
    }
    const pickBtn = e.target.closest('.pick button');
    if (pickBtn) {
      const { idx, o } = pickBtn.dataset;
      if (overrides[idx] === o) delete overrides[idx]; else overrides[idx] = o;
      afterOverrideChange();
      return;
    }
    const teamsBtn = e.target.closest('button.teams');
    if (teamsBtn) {
      const match = teamsBtn.closest('.match');
      const slot = match.querySelector('.grid-slot');
      const open = slot.hidden;
      slot.hidden = !open;
      teamsBtn.setAttribute('aria-expanded', open);
      if (open && !slot.childElementCount) slot.innerHTML = scoreGridHtml(Number(match.dataset.idx));
    }
  });
}

function scoreGridHtml(i) {
  const [, h, a] = data.fixtures[i];
  const p = matches[i];
  const N = 6, S = 11;   // show 0–5 goals; the grid holds 0–10
  const max = Math.max(...p.grid);
  let rowsHtml = '';
  for (let x = 0; x < N; x++) {
    rowsHtml += `<tr><th>${x}</th>`;
    for (let y = 0; y < N; y++) {
      const v = p.grid[x * S + y];
      const share = Math.round((v / max) * 100);
      const best = x === p.likelyScore[0] && y === p.likelyScore[1];
      rowsHtml += `<td class="${share > 55 ? 'dark' : 'light'}${best ? ' best' : ''}" title="${x}–${y}: ${(v * 100).toFixed(1)}%"
        style="background:color-mix(in oklab, var(--heat) ${share}%, var(--bg))">${(v * 100).toFixed(v < 0.1 ? 1 : 0)}</td>`;
    }
    rowsHtml += '</tr>';
  }
  return `<div class="score-grid">
    <p class="note">Chance of each exact score, in %. Rows: ${esc(name(h))} goals · columns: ${esc(name(a))} goals.
      Expected goals ${p.lam.toFixed(2)}–${p.mu.toFixed(2)}; most likely ${p.likelyScore.join('–')}.</p>
    <table class="heatmap"><thead><tr><th></th>${[...Array(N).keys()].map((y) => `<th>${y}</th>`).join('')}</tr></thead><tbody>${rowsHtml}</tbody></table>
  </div>`;
}

// ── model accuracy ──────────────────────────────────────────────────────────

function renderAccuracy(bt) {
  calibration($('#calibration'), bt.bins);
  const better = (1 - bt.rps / bt.rpsBase) * 100;
  $('#acc-note').textContent =
    `Every match since January ${bt.from} (${bt.n.toLocaleString('en')} matches) was predicted using only earlier results. ` +
    'Dots on the diagonal mean the percentages can be trusted: things the model calls 40% happen about 40% of the time.';
  $('#acc-tiles').innerHTML = `
    <div class="tile"><div class="tile-label">Prediction error (RPS)</div><div class="tile-value">${bt.rps.toFixed(3)}</div><div class="tile-note">lower is better</div></div>
    <div class="tile"><div class="tile-label">Naive guess</div><div class="tile-value">${bt.rpsBase.toFixed(3)}</div><div class="tile-note">same home/draw/away odds for every match</div></div>
    <div class="tile"><div class="tile-label">Improvement</div><div class="tile-value">${better.toFixed(0)}%</div><div class="tile-note">smaller error than the naive guess</div></div>`;
}

init().catch((err) => {
  console.error(err);
  $('#meta').textContent = 'Could not load the data. Try reloading the page.';
});
