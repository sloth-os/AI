// Small 2D charts for the HUD. Thin marks, hairline grid, fixed categorical colors, a legend for
// two or more series, and a hover read-out so no value is hidden behind a guess.
import { h } from './ui.js';
import { SERIES, seqCss, divCss } from './colors.js';

const INK = '#e6ecf8';
const INK2 = '#b4bfd6';
const MUTED = '#7f8aa5';
const GRID = 'rgba(255,255,255,0.07)';
const AXIS = 'rgba(255,255,255,0.16)';
const SURFACE = '#0b101c';
const FONT = '11px "Inter Variable", system-ui, sans-serif';

function niceTicks(lo, hi, n = 4) {
  if (!(hi > lo)) hi = lo + 1;
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(+v.toPrecision(12));
  return ticks;
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return +(v / 1e9).toPrecision(3) + 'B';
  if (a >= 1e6) return +(v / 1e6).toPrecision(3) + 'M';
  if (a >= 1e4) return +(v / 1e3).toPrecision(3) + 'K';
  if (a !== 0 && a < 0.01) return v.toExponential(0);
  return String(+v.toPrecision(3));
}

function setupCanvas(canvas, w, hgt) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(hgt * dpr);
  canvas.style.width = w + 'px';
  canvas.style.height = hgt + 'px';
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}

/**
 * Line chart with a crosshair read-out.
 * @param {{
 *   height?: number, series: {name:string, color?:string, data:(number[]|[number,number][])}[],
 *   xLabel?: string, yLabel?: string, yLog?: boolean, xLog?: boolean, yDomain?: [number,number], xDomain?: [number,number],
 *   yFormat?: (v:number)=>string, xFormat?: (v:number)=>string, legend?: boolean, title?: string
 * }} o   data is y-values (x = index) or [x, y] pairs. Colors default to the fixed series order.
 * @returns {{el: HTMLElement, update: (series?: any[]) => void}}
 */
export function lineChart(o) {
  let series = o.series || [];
  const height = o.height ?? 150;
  const canvas = h('canvas');
  const tipEl = h('div.chart-tip', { hidden: true });
  const legendEl = h('div.legend');
  const plot = h('div.chart-plot', canvas, tipEl);
  const el = h('figure.chart', o.title && h('figcaption', o.title), legendEl, plot);
  const pad = { l: 40, r: 12, t: o.yLabel ? 20 : 8, b: o.xLabel ? 30 : 20 };
  let hoverX = null;
  let geom = null;
  let rafQueued = false;

  const pts = (s) => s.data.map((d, i) => (Array.isArray(d) ? d : [i, d]));
  const tx = (v) => (o.xLog ? Math.log10(Math.max(v, 1e-12)) : v);
  const ty = (v) => (o.yLog ? Math.log10(Math.max(v, 1e-12)) : v);

  function draw() {
    rafQueued = false;
    const w = plot.clientWidth || 280;
    const g = setupCanvas(canvas, w, height);
    g.clearRect(0, 0, w, height);
    const all = series.map(pts);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const arr of all)
      for (const [x, y] of arr) {
        if (!Number.isFinite(y)) continue;
        const X = tx(x), Y = ty(y);
        if (X < x0) x0 = X;
        if (X > x1) x1 = X;
        if (Y < y0) y0 = Y;
        if (Y > y1) y1 = Y;
      }
    if (!Number.isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
    if (o.xDomain) [x0, x1] = o.xDomain.map(tx);
    if (o.yDomain) [y0, y1] = o.yDomain.map(ty);
    else if (!o.yLog) {
      if (y0 > 0 && y0 < (y1 - y0) * 2) y0 = 0;
      const m = (y1 - y0) * 0.06 || 0.5;
      y1 += m;
      if (y0 !== 0) y0 -= m;
    }
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) y1 = y0 + 1;
    const iw = w - pad.l - pad.r;
    const ih = height - pad.t - pad.b;
    const sx = (X) => pad.l + ((X - x0) / (x1 - x0)) * iw;
    const sy = (Y) => pad.t + ih - ((Y - y0) / (y1 - y0)) * ih;
    geom = { x0, x1, y0, y1, iw, ih, sx, sy, all, w };

    g.font = FONT;
    g.lineWidth = 1;
    g.textBaseline = 'middle';
    g.textAlign = 'right';
    const yFmt = o.yFormat ?? fmtTick;
    const xFmt = o.xFormat ?? fmtTick;
    const yTicks = o.yLog ? logTicks(y0, y1) : niceTicks(y0, y1, 4);
    for (const t of yTicks) {
      const y = Math.round(sy(t)) + 0.5;
      g.strokeStyle = GRID;
      g.beginPath();
      g.moveTo(pad.l, y);
      g.lineTo(w - pad.r, y);
      g.stroke();
      g.fillStyle = MUTED;
      g.fillText(yFmt(o.yLog ? Math.pow(10, t) : t), pad.l - 6, y);
    }
    g.textAlign = 'center';
    g.textBaseline = 'top';
    const xTicks = o.xLog ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(2, Math.floor(iw / 64)));
    for (const t of xTicks) {
      const x = sx(t);
      if (x < pad.l - 1 || x > w - pad.r + 1) continue;
      g.fillStyle = MUTED;
      g.fillText(xFmt(o.xLog ? Math.pow(10, t) : t), x, pad.t + ih + 5);
    }
    g.strokeStyle = AXIS;
    g.beginPath();
    g.moveTo(pad.l, pad.t + ih + 0.5);
    g.lineTo(w - pad.r, pad.t + ih + 0.5);
    g.stroke();
    if (o.xLabel) {
      g.fillStyle = MUTED;
      g.textAlign = 'right';
      g.fillText(o.xLabel, w - pad.r, height - 12);
    }
    if (o.yLabel) {
      g.save();
      g.fillStyle = MUTED;
      g.textAlign = 'left';
      g.textBaseline = 'top';
      g.fillText(o.yLabel, 2, 0);
      g.restore();
    }

    g.save();
    g.beginPath();
    g.rect(pad.l, pad.t - 2, iw + 2, ih + 4);
    g.clip();
    series.forEach((s, si) => {
      const arr = all[si];
      if (!arr.length) return;
      const color = s.color || SERIES[si % SERIES.length];
      g.strokeStyle = color;
      g.lineWidth = s.width ?? 2;
      g.lineJoin = 'round';
      g.lineCap = 'round';
      g.globalAlpha = s.dim ? 0.35 : 1;
      g.beginPath();
      // decimate long series to roughly one point per pixel column
      const stride = Math.max(1, Math.floor(arr.length / (iw * 1.5)));
      let started = false;
      for (let i = 0; i < arr.length; i += stride) {
        const [x, y] = arr[i];
        if (!Number.isFinite(y)) continue;
        const X = sx(tx(x)), Y = sy(ty(y));
        if (started) g.lineTo(X, Y);
        else {
          g.moveTo(X, Y);
          started = true;
        }
      }
      const last = arr[arr.length - 1];
      if (stride > 1 && Number.isFinite(last[1])) g.lineTo(sx(tx(last[0])), sy(ty(last[1])));
      g.stroke();
      // end marker with a surface ring
      if (Number.isFinite(last[1])) {
        const X = sx(tx(last[0])), Y = sy(ty(last[1]));
        g.fillStyle = SURFACE;
        g.beginPath();
        g.arc(X, Y, 5.5, 0, Math.PI * 2);
        g.fill();
        g.fillStyle = color;
        g.beginPath();
        g.arc(X, Y, 3.5, 0, Math.PI * 2);
        g.fill();
      }
      g.globalAlpha = 1;
    });
    g.restore();

    if (hoverX != null) {
      const X = Math.max(pad.l, Math.min(w - pad.r, hoverX));
      g.strokeStyle = 'rgba(255,255,255,0.35)';
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(Math.round(X) + 0.5, pad.t);
      g.lineTo(Math.round(X) + 0.5, pad.t + ih);
      g.stroke();
    }
  }

  function logTicks(lo, hi) {
    // decades, plus 2x and 5x steps when the range is short
    const decades = [];
    for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) decades.push(e);
    const inRange = (v) => v >= lo - 1e-9 && v <= hi + 1e-9;
    let out = decades.filter(inRange);
    if (out.length < 3) {
      out = [];
      for (const e of decades) for (const m of [1, 2, 5]) if (inRange(e + Math.log10(m))) out.push(e + Math.log10(m));
    }
    if (out.length < 2) return niceTicks(lo, hi, 3);
    while (out.length > 6) out = out.filter((_, i) => i % 2 === 0);
    return out;
  }

  function showTip(clientX) {
    if (!geom) return;
    const rect = canvas.getBoundingClientRect();
    hoverX = clientX - rect.left;
    const X = geom.x0 + ((hoverX - pad.l) / geom.iw) * (geom.x1 - geom.x0);
    const rows = [];
    let xShown = null;
    series.forEach((s, si) => {
      const arr = geom.all[si];
      if (!arr.length) return;
      let best = 0, bd = Infinity;
      for (let i = 0; i < arr.length; i++) {
        const d = Math.abs(tx(arr[i][0]) - X);
        if (d < bd) { bd = d; best = i; }
      }
      xShown ??= arr[best][0];
      rows.push(h('div.r', h('i.line', { style: { '--c': s.color || SERIES[si % SERIES.length] } }), h('b', (o.yFormat ?? fmtTick)(arr[best][1])), h('span', s.name)));
    });
    tipEl.replaceChildren(h('div.x', `${o.xLabel || 'x'} ${(o.xFormat ?? fmtTick)(xShown ?? 0)}`), ...rows);
    tipEl.hidden = false;
    const left = hoverX + 14 + 150 > geom.w ? hoverX - 14 - tipEl.offsetWidth : hoverX + 14;
    tipEl.style.left = Math.max(0, left) + 'px';
    queue();
  }
  canvas.addEventListener('pointermove', (e) => showTip(e.clientX));
  canvas.addEventListener('pointerleave', () => {
    hoverX = null;
    tipEl.hidden = true;
    queue();
  });

  function renderLegend() {
    const show = o.legend ?? series.length >= 2;
    legendEl.hidden = !show;
    if (!show) return;
    legendEl.replaceChildren(...series.map((s, si) => h('span.li', h('i.line', { style: { '--c': s.color || SERIES[si % SERIES.length] } }), s.name)));
  }
  const queue = () => {
    if (rafQueued) return;
    rafQueued = true;
    requestAnimationFrame(draw);
  };
  renderLegend();
  new ResizeObserver(queue).observe(plot);
  queue();
  return {
    el,
    update(next) {
      if (next) {
        const names = series.map((s) => s.name).join('|');
        series = next;
        if (names !== series.map((s) => s.name).join('|')) renderLegend();
      }
      queue();
    },
  };
}

/**
 * Ranked horizontal bars, one row per item. rows: [{label, value, color?, note?, title?, active?}]
 * One color for all bars unless a row's color identifies an entity that is colored the same elsewhere.
 */
export function barChart(o) {
  const el = h('figure.chart.bars', o.title && h('figcaption', o.title));
  const body = h('div.bars-body');
  el.appendChild(body);
  const api = {
    el,
    update(rows) {
      const list = o.sort ? [...rows].sort((a, b) => b.value - a.value) : rows;
      const max = o.max ?? Math.max(1e-12, ...list.map((r) => Math.abs(r.value)));
      const f = o.format ?? fmtTick;
      body.replaceChildren(
        ...list.map((r) => {
          const pct = Math.max(0.5, (Math.abs(r.value) / max) * 100);
          const row = h(
            'div.bar-row' + (r.active ? '.on' : ''),
            { title: r.title ?? `${r.label}: ${f(r.value)}${o.unit ? ' ' + o.unit : ''}` },
            h('span.bl', r.label),
            h('span.bt', h('i', { style: { width: pct + '%', '--c': r.color || o.color || SERIES[0] } })),
            h('span.bv', r.note ?? f(r.value) + (o.unit ? ' ' + o.unit : '')),
          );
          if (r.onClick) {
            row.classList.add('click');
            row.addEventListener('click', r.onClick);
          }
          return row;
        }),
      );
    },
  };
  api.update(o.rows || []);
  return api;
}

/**
 * Heatmap of a matrix. mode 'seq' expects values in [0, max]; 'div' expects signed values in [-max, max].
 * @param {{rows: string[], cols: string[], values: number[][], mode?: 'seq'|'div', max?: number, cell?: number,
 *          format?: (v:number)=>string, title?: string, rowLabel?: string, colLabel?: string, onHover?: (r:number,c:number)=>void}} o
 */
export function heatmap(o) {
  const canvas = h('canvas');
  const tipEl = h('div.chart-tip', { hidden: true });
  const plot = h('div.chart-plot', canvas, tipEl);
  const el = h('figure.chart.heat', o.title && h('figcaption', o.title), plot);
  let data = o;
  let lay = null;
  function draw() {
    const { rows, cols, values } = data;
    const mode = data.mode ?? 'seq';
    const w = plot.clientWidth || 280;
    const g0 = canvas.getContext('2d');
    g0.font = FONT;
    const labelW = Math.min(110, Math.max(24, ...rows.map((r) => g0.measureText(r).width)) + 8);
    const top = data.colLabels === false ? 4 : Math.min(70, Math.max(16, ...cols.map((c) => g0.measureText(c).width)) + 6);
    const cell = Math.max(3, Math.min(data.cell ?? 22, (w - labelW - 2) / Math.max(1, cols.length)));
    const hgt = top + cell * rows.length + 2;
    const g = setupCanvas(canvas, w, hgt);
    g.clearRect(0, 0, w, hgt);
    g.font = FONT;
    let max = data.max;
    if (max == null) {
      max = 1e-12;
      for (const r of values) for (const v of r) max = Math.max(max, Math.abs(v));
    }
    for (let r = 0; r < rows.length; r++) {
      g.fillStyle = INK2;
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      if (cell >= 9) g.fillText(rows[r], labelW - 6, top + cell * (r + 0.5));
      for (let c = 0; c < cols.length; c++) {
        const v = values[r]?.[c];
        if (v == null || Number.isNaN(v)) continue;
        g.fillStyle = mode === 'div' ? divCss(Math.max(-1, Math.min(1, v / max))) : seqCss(Math.max(0, Math.min(1, v / max)));
        const gap = cell > 6 ? 1 : 0;
        g.fillRect(labelW + c * cell + gap, top + r * cell + gap, cell - gap * 2, cell - gap * 2);
      }
    }
    if (data.colLabels !== false && cell >= 9) {
      g.fillStyle = INK2;
      for (let c = 0; c < cols.length; c++) {
        g.save();
        g.translate(labelW + cell * (c + 0.5), top - 4);
        g.rotate(-Math.PI / 2);
        g.textAlign = 'left';
        g.textBaseline = 'middle';
        g.fillText(cols[c], 0, 0);
        g.restore();
      }
    }
    lay = { labelW, top, cell, w };
  }
  canvas.addEventListener('pointermove', (e) => {
    if (!lay) return;
    const rect = canvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left - lay.labelW) / lay.cell);
    const r = Math.floor((e.clientY - rect.top - lay.top) / lay.cell);
    const v = data.values[r]?.[c];
    if (v == null || c < 0) {
      tipEl.hidden = true;
      return;
    }
    const f = data.format ?? ((x) => x.toFixed(2));
    tipEl.replaceChildren(h('div.r', h('b', f(v))), h('div.x', `${data.rowLabel ? data.rowLabel + ' ' : ''}${data.rows[r]} → ${data.colLabel ? data.colLabel + ' ' : ''}${data.cols[c]}`));
    tipEl.hidden = false;
    const x = e.clientX - rect.left;
    tipEl.style.left = Math.max(0, x + 14 + tipEl.offsetWidth > lay.w ? x - 14 - tipEl.offsetWidth : x + 14) + 'px';
    tipEl.style.top = Math.max(0, e.clientY - rect.top - 10) + 'px';
    data.onHover?.(r, c);
  });
  canvas.addEventListener('pointerleave', () => {
    tipEl.hidden = true;
    data.onHover?.(-1, -1);
  });
  new ResizeObserver(() => requestAnimationFrame(draw)).observe(plot);
  requestAnimationFrame(draw);
  return {
    el,
    update(next) {
      data = { ...data, ...next };
      requestAnimationFrame(draw);
    },
  };
}

/**
 * Radar ("profile") chart for comparing a few options across named criteria. Pair it with a meter
 * list or table for exact values.
 * @param {{axes: string[], series: {name:string, color?:string, values:number[]}[], max?: number, size?: number}} o
 */
export function radar(o) {
  const size = o.size ?? 220;
  const el = h('figure.chart.radar');
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const api = {
    el,
    update(series) {
      const n = o.axes.length;
      const max = o.max ?? 5;
      const cx = size / 2;
      const cy = size / 2;
      const R = size / 2 - 34;
      const pt = (i, v) => {
        const a = -Math.PI / 2 + (i / n) * Math.PI * 2;
        return [cx + Math.cos(a) * R * (v / max), cy + Math.sin(a) * R * (v / max)];
      };
      const svg = svgEl('svg', { viewBox: `0 0 ${size} ${size}`, width: '100%', role: 'img', 'aria-label': 'Profile across ' + o.axes.join(', ') });
      for (let ring = 1; ring <= max; ring++) {
        const p = o.axes.map((_, i) => pt(i, ring).join(',')).join(' ');
        svg.appendChild(svgEl('polygon', { points: p, fill: 'none', stroke: ring === max ? AXIS : GRID, 'stroke-width': 1 }));
      }
      o.axes.forEach((name, i) => {
        const [x, y] = pt(i, max);
        svg.appendChild(svgEl('line', { x1: cx, y1: cy, x2: x, y2: y, stroke: GRID, 'stroke-width': 1 }));
        const [lx, ly] = pt(i, max * 1.2);
        const t = svgEl('text', { x: lx, y: ly, fill: INK2, 'font-size': 10.5, 'text-anchor': Math.abs(lx - cx) < 8 ? 'middle' : lx > cx ? 'start' : 'end', 'dominant-baseline': 'middle' });
        t.textContent = name;
        svg.appendChild(t);
      });
      series.forEach((s, si) => {
        const color = s.color || SERIES[si % SERIES.length];
        const p = s.values.map((v, i) => pt(i, v).join(',')).join(' ');
        svg.appendChild(svgEl('polygon', { points: p, fill: color, 'fill-opacity': s.dim ? 0.03 : 0.12, stroke: color, 'stroke-width': s.dim ? 1 : 2, 'stroke-linejoin': 'round', 'stroke-opacity': s.dim ? 0.45 : 1 }));
        if (!s.dim)
          s.values.forEach((v, i) => {
            const [x, y] = pt(i, v);
            svg.appendChild(svgEl('circle', { cx: x, cy: y, r: 3, fill: color, stroke: SURFACE, 'stroke-width': 1.5 }));
          });
      });
      const kids = [svg];
      if (series.length >= 2) kids.push(h('div.legend', series.map((s, si) => h('span.li', h('i.line', { style: { '--c': s.color || SERIES[si % SERIES.length] } }), s.name))));
      el.replaceChildren(...kids);
    },
  };
  api.update(o.series || []);
  return api;
}

/** Tiny inline trend line (no axes). */
export function sparkline(values, o = {}) {
  const w = o.width ?? 80;
  const hgt = o.height ?? 22;
  const canvas = h('canvas.spark');
  const draw = (vals) => {
    const g = setupCanvas(canvas, w, hgt);
    g.clearRect(0, 0, w, hgt);
    if (vals.length < 2) return;
    const lo = Math.min(...vals), hi = Math.max(...vals);
    g.strokeStyle = o.color || MUTED;
    g.lineWidth = 1.5;
    g.beginPath();
    vals.forEach((v, i) => {
      const x = 2 + (i / (vals.length - 1)) * (w - 4);
      const y = hgt - 3 - ((v - lo) / (hi - lo || 1)) * (hgt - 6);
      if (i) g.lineTo(x, y);
      else g.moveTo(x, y);
    });
    g.stroke();
  };
  draw(values);
  return { el: canvas, update: draw };
}

export { INK, INK2, MUTED, SURFACE };
