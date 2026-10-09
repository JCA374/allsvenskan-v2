// Small hand-written SVG charts: line chart with crosshair, scatter, calibration plot.
// Charts render at their container's pixel width and redraw on resize.

const NS = 'http://www.w3.org/2000/svg';
export const SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'].map((v) => `var(${v})`);

export function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent?.appendChild(node);
  return node;
}

function text(parent, x, y, str, attrs = {}) {
  const t = el('text', { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}

// ── tooltip (one shared element) ────────────────────────────────────────────

const tip = document.createElement('div');
tip.className = 'tooltip';
tip.hidden = true;
document.body.appendChild(tip);

// rows: [{ value, label, color? }] — value is the strong element, label secondary
export function showTip(event, title, rows) {
  tip.replaceChildren();
  if (title) { const h = document.createElement('div'); h.className = 'tip-title'; h.textContent = title; tip.append(h); }
  for (const r of rows) {
    const row = document.createElement('div'); row.className = 'tip-row';
    if (r.color) { const k = document.createElement('i'); k.style.background = r.color; row.append(k); }
    const v = document.createElement('b'); v.textContent = r.value;
    const l = document.createElement('span'); l.textContent = r.label;
    row.append(v, l); tip.append(row);
  }
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + w > window.innerWidth - 8) x = event.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = event.clientY - h - pad;
  tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
}
export const hideTip = () => { tip.hidden = true; };

// Re-render `draw(width)` whenever the container changes width.
export function responsive(container, draw) {
  let last = 0;
  const run = () => {
    const w = Math.floor(container.clientWidth);
    if (w && w !== last) { last = w; draw(w); }
  };
  new ResizeObserver(run).observe(container);
  run();
  return () => { last = 0; run(); };   // force redraw
}

function ticks(min, max, count) {
  const step = niceStep((max - min) / count);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

// Spread label y-positions so they are at least `gap` apart, keeping order.
function spread(items, gap, top, bottom) {
  items.sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + gap);
  const over = items.length ? items[items.length - 1].y - bottom : 0;
  if (over > 0) for (const it of items) it.y -= over;
  for (let i = items.length - 2; i >= 0; i--) items[i].y = Math.min(items[i].y, items[i + 1].y - gap);
  for (const it of items) it.y = Math.max(it.y, top);
}

// ── line chart ──────────────────────────────────────────────────────────────
// series: [{ name, color, values: number[] (0..1, index = x) , muted? }]
export function lineChart(container, { series, xLabel, xTickLabel = (x) => x, tipTitle }) {
  return responsive(container, (W) => {
    container.replaceChildren();
    const labelW = W < 520 ? 0 : 120;
    const m = { top: 12, right: 12 + labelW, bottom: 34, left: 40 };
    const H = Math.round(Math.min(380, Math.max(260, W * 0.45)));
    const svg = el('svg', { width: W, height: H, class: 'chart', role: 'img' }, container);
    const n = Math.max(...series.map((s) => s.values.length));
    const x = (i) => m.left + (i / Math.max(1, n - 1)) * (W - m.left - m.right);
    const y = (v) => m.top + (1 - v) * (H - m.top - m.bottom);

    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      el('line', { x1: m.left, x2: W - m.right, y1: y(v), y2: y(v), class: v ? 'grid' : 'axis' }, svg);
      text(svg, m.left - 6, y(v) + 4, `${v * 100}%`, { class: 'tick', 'text-anchor': 'end' });
    }
    const step = Math.max(1, Math.ceil(n / (W / 60)));
    for (let i = 0; i < n; i += step) text(svg, x(i), H - m.bottom + 16, xTickLabel(i), { class: 'tick', 'text-anchor': 'middle' });
    text(svg, m.left + (W - m.left - m.right) / 2, H - 4, xLabel, { class: 'axis-label', 'text-anchor': 'middle' });

    const ordered = [...series].sort((a, b) => (a.muted === b.muted ? 0 : a.muted ? -1 : 1));
    for (const s of ordered) {
      const d = s.values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
      el('path', { d, class: s.muted ? 'line muted' : 'line', style: s.muted ? '' : `stroke:${s.color}` }, svg);
    }
    if (labelW) {
      // only label lines that end visibly above zero; the rest are in the legend and tooltip
      const labels = series.filter((s) => !s.muted && s.values[s.values.length - 1] >= 0.03).map((s) => ({ s, y: y(s.values[s.values.length - 1]) }));
      spread(labels, 14, m.top + 4, H - m.bottom);
      for (const { s, y: ly } of labels) {
        el('circle', { cx: x(s.values.length - 1), cy: y(s.values[s.values.length - 1]), r: 3, style: `fill:${s.color}` }, svg);
        text(svg, W - m.right + 8, ly + 4, `${s.name} ${Math.round(s.values[s.values.length - 1] * 100)}%`, { class: 'end-label' });
      }
    }

    // crosshair
    const hair = el('line', { y1: m.top, y2: H - m.bottom, class: 'crosshair', visibility: 'hidden' }, svg);
    const hit = el('rect', { x: m.left, y: m.top, width: W - m.left - m.right, height: H - m.top - m.bottom, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', (e) => {
      const r = svg.getBoundingClientRect();
      const i = Math.round(((e.clientX - r.left - m.left) / (W - m.left - m.right)) * (n - 1));
      if (i < 0 || i >= n) return;
      hair.setAttribute('x1', x(i)); hair.setAttribute('x2', x(i)); hair.setAttribute('visibility', 'visible');
      const rows = series.filter((s) => !s.muted && s.values[i] != null)
        .sort((a, b) => b.values[i] - a.values[i])
        .map((s) => ({ value: `${(s.values[i] * 100).toFixed(1)}%`, label: s.name, color: s.color }));
      showTip(e, tipTitle(i), rows);
    });
    hit.addEventListener('pointerleave', () => { hair.setAttribute('visibility', 'hidden'); hideTip(); });
  });
}

// ── scatter ─────────────────────────────────────────────────────────────────
// points: [{ x, y, label, tip: rows }]; yInvert puts low values at the top.
export function scatter(container, { points, xLabel, yLabel, xRef, yRef, yInvert = false, quadrants = [] }) {
  return responsive(container, (W) => {
    container.replaceChildren();
    const narrow = W < 520;
    const H = Math.round(narrow ? W * 1.15 : Math.min(460, Math.max(320, W * 0.6)));
    const m = { top: 16, right: 16, bottom: 40, left: 48 };
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const pad = (lo, hi) => [lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08];
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs));
    const [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
    const x = (v) => m.left + ((v - x0) / (x1 - x0)) * (W - m.left - m.right);
    const yy = (v) => { const t = (v - y0) / (y1 - y0); return m.top + (yInvert ? t : 1 - t) * (H - m.top - m.bottom); };
    const svg = el('svg', { width: W, height: H, class: 'chart', role: 'img' }, container);

    for (const v of ticks(x0, x1, 5)) {
      el('line', { x1: x(v), x2: x(v), y1: m.top, y2: H - m.bottom, class: 'grid' }, svg);
      text(svg, x(v), H - m.bottom + 16, v.toFixed(1), { class: 'tick', 'text-anchor': 'middle' });
    }
    for (const v of ticks(y0, y1, 5)) {
      el('line', { x1: m.left, x2: W - m.right, y1: yy(v), y2: yy(v), class: 'grid' }, svg);
      text(svg, m.left - 6, yy(v) + 4, v.toFixed(1), { class: 'tick', 'text-anchor': 'end' });
    }
    text(svg, m.left + (W - m.left - m.right) / 2, H - 6, xLabel, { class: 'axis-label', 'text-anchor': 'middle' });
    const yl = text(svg, 0, 0, yLabel, { class: 'axis-label', 'text-anchor': 'middle' });
    yl.setAttribute('transform', `translate(12 ${m.top + (H - m.top - m.bottom) / 2}) rotate(-90)`);

    el('line', { x1: x(xRef), x2: x(xRef), y1: m.top, y2: H - m.bottom, class: 'ref' }, svg);
    el('line', { x1: m.left, x2: W - m.right, y1: yy(yRef), y2: yy(yRef), class: 'ref' }, svg);
    const corners = [[m.left + 6, m.top + 14, 'start'], [W - m.right - 6, m.top + 14, 'end'],
      [m.left + 6, H - m.bottom - 8, 'start'], [W - m.right - 6, H - m.bottom - 8, 'end']];
    if (!narrow) quadrants.forEach((q, i) => text(svg, corners[i][0], corners[i][1], q, { class: 'quadrant', 'text-anchor': corners[i][2] }));

    const labels = points.map((p) => ({ p, y: yy(p.y) - 9 }));
    for (const p of points) el('circle', { cx: x(p.x), cy: yy(p.y), r: 5, class: 'dot' }, svg);
    // nudge labels apart within columns of nearby x
    labels.sort((a, b) => a.p.x - b.p.x);
    for (let i = 0; i < labels.length; i++) {
      for (let j = 0; j < i; j++) {
        const a = labels[j], b = labels[i];
        if (Math.abs(x(a.p.x) - x(b.p.x)) < 70 && Math.abs(a.y - b.y) < 12) b.y = a.y + (b.y >= a.y ? 12 : -12);
      }
    }
    for (const { p, y } of labels) {
      const px = x(p.x);
      const anchor = px > W - m.right - 40 ? 'end' : px < m.left + 40 ? 'start' : 'middle';
      text(svg, px + (anchor === 'end' ? 6 : anchor === 'start' ? -6 : 0), y, p.label, { class: 'dot-label', 'text-anchor': anchor });
    }
    for (const p of points) {
      const hitArea = el('circle', { cx: x(p.x), cy: yy(p.y), r: 14, fill: 'transparent' }, svg);
      hitArea.addEventListener('pointermove', (e) => showTip(e, p.label, p.tip));
      hitArea.addEventListener('pointerleave', hideTip);
    }
  });
}

// ── calibration ─────────────────────────────────────────────────────────────
// bins: [{ predicted, observed, n }]
export function calibration(container, bins) {
  return responsive(container, (W) => {
    container.replaceChildren();
    const size = Math.min(W, 420);
    const m = { top: 12, right: 12, bottom: 40, left: 48 };
    const svg = el('svg', { width: size, height: size, class: 'chart', role: 'img' }, container);
    const s = (v) => v * (size - m.left - m.right);
    const x = (v) => m.left + s(v), y = (v) => size - m.bottom - v * (size - m.top - m.bottom);
    for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      el('line', { x1: x(v), x2: x(v), y1: y(0), y2: y(1), class: 'grid' }, svg);
      el('line', { x1: x(0), x2: x(1), y1: y(v), y2: y(v), class: 'grid' }, svg);
      text(svg, x(v), size - m.bottom + 16, `${v * 100}%`, { class: 'tick', 'text-anchor': 'middle' });
      text(svg, m.left - 6, y(v) + 4, `${v * 100}%`, { class: 'tick', 'text-anchor': 'end' });
    }
    el('line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), class: 'ref' }, svg);
    text(svg, x(0.97), y(0.97) + 16, 'perfect', { class: 'quadrant', 'text-anchor': 'end' });
    text(svg, m.left + (size - m.left - m.right) / 2, size - 6, 'Model said', { class: 'axis-label', 'text-anchor': 'middle' });
    const yl = text(svg, 0, 0, 'Actually happened', { class: 'axis-label', 'text-anchor': 'middle' });
    yl.setAttribute('transform', `translate(12 ${m.top + (size - m.top - m.bottom) / 2}) rotate(-90)`);
    const maxN = Math.max(...bins.map((b) => b.n));
    for (const b of bins) {
      el('circle', { cx: x(b.predicted), cy: y(b.observed), r: 4 + 8 * Math.sqrt(b.n / maxN), class: 'dot' }, svg);
      const hitArea = el('circle', { cx: x(b.predicted), cy: y(b.observed), r: 16, fill: 'transparent' }, svg);
      hitArea.addEventListener('pointermove', (e) => showTip(e, `Predicted ≈ ${Math.round(b.predicted * 100)}%`, [
        { value: `${Math.round(b.observed * 100)}%`, label: 'actually happened' },
        { value: b.n.toLocaleString('en'), label: 'predictions' },
      ]));
      hitArea.addEventListener('pointerleave', hideTip);
    }
  });
}
