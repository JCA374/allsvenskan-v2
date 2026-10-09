import { fitModel, matchProbabilities, simulateSeason } from './model.js';

const DISPLAY_NAMES = {
  Djurgarden: 'Djurgården', Goteborg: 'IFK Göteborg', Hacken: 'BK Häcken', 'Malmo FF': 'Malmö FF',
  Mjallby: 'Mjällby', Orgryte: 'Örgryte', 'Vasteras SK': 'Västerås SK', Norrkoping: 'Norrköping',
  Varnamo: 'Värnamo', Orebro: 'Örebro', Osters: 'Öster', Brommapojkarna: 'Brommapojkarna',
};

const $ = (sel) => document.querySelector(sel);
const pct = (p) => (p >= 0.995 ? '>99%' : p > 0 && p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`);
const fmtDate = (iso, opts) => new Date(iso + 'T12:00:00').toLocaleDateString('en-GB', opts);

let data, model, name;

async function init() {
  data = await (await fetch('data.json', { cache: 'no-cache' })).json();
  name = (i) => DISPLAY_NAMES[data.teams[i]] ?? data.teams[i];
  const lastResult = data.results[data.results.length - 1][0];
  $('#season').textContent = data.season;
  $('#meta').textContent =
    `Results up to ${fmtDate(lastResult, { day: 'numeric', month: 'long', year: 'numeric' })} · ` +
    `${data.fixtures.length} matches left · data updated ${fmtDate(data.generated.slice(0, 10), { day: 'numeric', month: 'short' })}`;

  model = fitModel(data);
  renderFixtures();
  run(1);
  $('#run').addEventListener('click', () => run(Math.floor(Math.random() * 2 ** 31)));
}

function run(seed) {
  const nSims = Number($('#nsims').value);
  $('#run').disabled = true;
  $('#status').textContent = `Simulating ${nSims.toLocaleString('en')} seasons…`;
  // let the status paint before the main thread gets busy
  setTimeout(() => {
    const t0 = performance.now();
    const rows = simulateSeason(data, model, { nSims, seed });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    rows.sort((a, b) => b.expectedPts - a.expectedPts);
    renderTable(rows);
    renderHeatmap(rows);
    $('#status').textContent = `Played out ${nSims.toLocaleString('en')} seasons in your browser in ${secs} s`;
    $('#run').disabled = false;
  }, 30);
}

function probCell(p) {
  return `<td class="num prob">${pct(p)}<span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span></td>`;
}

function renderTable(rows) {
  const T = rows.length;
  const sum = (r, from, to) => r.positions.slice(from, to).reduce((s, v) => s + v, 0);
  $('#table tbody').innerHTML = rows.map((r, i) => {
    const zone = i < 3 ? 'zone-eu' : i >= T - 3 ? 'zone-down' : '';
    const gd = r.gf - r.ga;
    return `<tr class="${zone}">
      <td class="num dim">${i + 1}</td>
      <td class="team">${name(r.team)}</td>
      <td class="num dim">${r.played}</td>
      <td class="num">${r.pts}</td>
      <td class="num dim">${gd > 0 ? '+' : ''}${gd}</td>
      <td class="num"><b>${r.expectedPts.toFixed(1)}</b></td>
      ${probCell(r.positions[0])}
      ${probCell(sum(r, 0, 3))}
      ${probCell(r.positions[T - 3])}
      ${probCell(sum(r, T - 2, T))}
    </tr>`;
  }).join('');
}

function renderHeatmap(rows) {
  const T = rows.length;
  const head = `<tr><th></th>${rows.map((_, p) => `<th>${p + 1}</th>`).join('')}</tr>`;
  const body = rows.map((r) => `<tr><th class="rowhead">${name(r.team)}</th>${r.positions.map((p, pos) => {
    const share = Math.round(Math.min(1, p / 0.6) * 100);   // 60%+ = full colour
    const label = p >= 0.005 ? Math.round(p * 100) : '';
    const tip = `${name(r.team)} finishes ${ordinal(pos + 1)}: ${(p * 100).toFixed(1)}%`;
    return `<td class="${share > 45 ? 'dark' : 'light'}" title="${tip}"
      style="background:color-mix(in oklab, var(--heat) ${share}%, var(--bg))">${label}</td>`;
  }).join('')}</tr>`).join('');
  $('#heatmap').innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
  $('#heatmap').setAttribute('aria-label', `Final position probabilities for ${T} teams`);
}

function ordinal(n) {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function renderFixtures() {
  const byDate = new Map();
  for (const f of data.fixtures) {
    if (!byDate.has(f[0])) byDate.set(f[0], []);
    byDate.get(f[0]).push(f);
  }
  $('#fixtures').innerHTML = [...byDate].map(([date, fixtures]) => `
    <div class="day">
      <h3>${fmtDate(date, { weekday: 'long', day: 'numeric', month: 'long' })}</h3>
      ${fixtures.map(([, h, a]) => {
        const p = matchProbabilities(model, h, a);
        return `<div class="match">
          <div class="teams">${name(h)} – ${name(a)}</div>
          <div>
            <div class="split" role="img" aria-label="Home ${pct(p.home)}, draw ${pct(p.draw)}, away ${pct(p.away)}">
              <i class="home" style="flex:${p.home}"></i><i class="draw" style="flex:${p.draw}"></i><i class="away" style="flex:${p.away}"></i>
            </div>
            <div class="split-nums"><span>${pct(p.home)}</span><span>${pct(p.draw)}</span><span>${pct(p.away)}</span></div>
          </div>
          <div class="extra">xG ${p.lam.toFixed(1)}–${p.mu.toFixed(1)} · likely ${p.likelyScore.join('–')}</div>
        </div>`;
      }).join('')}
    </div>`).join('');
}

init().catch((err) => {
  console.error(err);
  $('#meta').textContent = 'Could not load the data. Try reloading the page.';
});
