// Quantization, measured: round the real GPT's weights to fewer bits and see what it costs.
// A real weight matrix is drawn as a field of bars; lowering the bit width snaps the bars onto a
// coarser and coarser grid. Perplexity is measured live on held-out text.
import * as THREE from 'three';
import '../labs/gpt.css';
import { loadTinyGPT, quantLevels } from '../ml/tinygpt.js';
import { PAL, SERIES, div, glow } from '../core/colors.js';
import { instanced, gridFloor, flushInst } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, segmented, stat, statRow, table, tour, fmt } from '../core/ui.js';
import { lineChart } from '../core/charts.js';
import { TECHNIQUES_2026 } from '../data/landscape.js';

const BITS = [16, 8, 6, 5, 4, 3, 2];
const METHODS = [
  { id: 'mse', name: 'Clipped (least error)', color: SERIES[0] },
  { id: 'minmax', name: 'Min-max (naive)', color: SERIES[1] },
];
const G = 96; // side of the square of weights that is drawn
const BINS = 81;
const EVAL_WINDOWS = 10;
const PROMPT = 'Once upon a time, there was a little girl named';

export async function enter(ctx) {
  const { app, root } = ctx;
  ctx.ui.loading('Loading the model');
  const { model, tokenizer, evalSet } = await loadTinyGPT(ctx.asset);
  const d = model.config.d;
  app.setView({ pos: [4.5, 6.2, 15], target: [0, 1.0, -0.9], fit: 8.8, min: 4, max: 70, maxPolar: Math.PI * 0.55 });
  root.add(gridFloor({ size: 70, y: -3, color: '#5a3350', opacity: 0.3 }));

  // ------------------------------------------------------------------ state
  const S = { bits: 16, method: 'mse' };
  const models = new Map(); // "bits|method" -> TinyGPT
  const results = new Map(); // "bits|method" -> { loss, ppl }
  const samples = new Map();
  const key = (bits, method) => (bits >= 16 ? '16' : bits + '|' + method);
  const getModel = (bits, method) => {
    const k = key(bits, method);
    if (!models.has(k)) models.set(k, model.quantized(bits, { method }));
    return models.get(k);
  };

  // ------------------------------------------------------------------ 3D: a real weight matrix
  // back: the matrix as a low relief; front: ONE of its rows as tall bars against the grid of
  // allowed values (a row shares one scale, so its levels are straight lines); far back: histogram.
  const NAME = 'h0.proj.w';
  const baseW = model.t.get(NAME); // [d, d]
  let maxAbs = 0;
  for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) maxAbs = Math.max(maxAbs, Math.abs(baseW[r * d + c]));
  const FIELD = 10;
  const CELL = FIELD / G;
  const FIELD_Z = -1.4;
  const RELIEF = 0.75;
  const barGeo = new THREE.BoxGeometry(CELL * 0.9, 1, CELL * 0.9);
  barGeo.translate(0, 0.5, 0);
  const field = instanced(barGeo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), G * G);
  root.add(field);
  const cur = new Float32Array(G * G);
  const target = new Float32Array(G * G);
  let animating = 1;
  const m4 = new THREE.Matrix4();
  const col = new THREE.Color();

  // the row with the most pronounced outlier: it shows best why clipping helps
  let ROW = 0;
  let bestRatio = 0;
  for (let r = 0; r < G; r++) {
    let mx = 0;
    let ss = 0;
    for (let c = 0; c < d; c++) {
      mx = Math.max(mx, Math.abs(baseW[r * d + c]));
      ss += baseW[r * d + c] ** 2;
    }
    const ratio = mx / Math.sqrt(ss / d);
    if (ratio > bestRatio) {
      bestRatio = ratio;
      ROW = r;
    }
  }
  let rowMaxAbs = 0;
  for (let c = 0; c < d; c++) rowMaxAbs = Math.max(rowMaxAbs, Math.abs(baseW[ROW * d + c]));
  const ROW_Z = 5.3;
  const ROW_W = 12.4;
  const ROW_H = 2.3;
  const rowGeo = new THREE.BoxGeometry((ROW_W / d) * 0.74, 1, 0.16);
  rowGeo.translate(0, 0.5, 0);
  const rowBars = instanced(rowGeo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), d);
  rowBars.position.set(0, 0, ROW_Z);
  root.add(rowBars);
  const rowCur = new Float32Array(d);
  const rowTarget = new Float32Array(d);
  const MAX_LINES = 31;
  const lvlGeo = new THREE.BufferGeometry();
  lvlGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_LINES * 6), 3));
  const lvlLines = new THREE.LineSegments(lvlGeo, new THREE.LineBasicMaterial({ color: glow('#ffffff', 1), transparent: true, opacity: 0.3, depthWrite: false }));
  lvlLines.position.set(0, 0, ROW_Z - 0.12);
  lvlLines.frustumCulled = false;
  root.add(lvlLines);
  // marks the row inside the relief
  const rowMark = new THREE.Mesh(new THREE.BoxGeometry(FIELD + 0.3, 0.02, CELL * 1.6), new THREE.MeshBasicMaterial({ color: glow('#ffffff', 1.6), transparent: true, opacity: 0.55, depthWrite: false }));
  rowMark.position.set(0, RELIEF + 0.12, FIELD_Z + (ROW - (G - 1) / 2) * CELL);
  root.add(rowMark);

  function paintField(force) {
    let moving = false;
    for (let r = 0; r < G; r++)
      for (let c = 0; c < G; c++) {
        const i = r * G + c;
        const dlt = target[i] - cur[i];
        if (Math.abs(dlt) > 1e-5) {
          cur[i] += force ? dlt : dlt * 0.22;
          moving = true;
        } else cur[i] = target[i];
        const v = cur[i] / maxAbs;
        const hgt = v * RELIEF;
        m4.makeScale(1, Math.abs(hgt) < 0.004 ? 0.004 : hgt, 1);
        m4.setPosition((c - (G - 1) / 2) * CELL, 0, FIELD_Z + (r - (G - 1) / 2) * CELL);
        field.setMatrixAt(i, m4);
        div(Math.max(-1, Math.min(1, v * 1.7)), col).multiplyScalar(0.55 + Math.min(1, Math.abs(v) * 1.7) * 0.75);
        field.setColorAt(i, col);
      }
    flushInst(field);
    for (let c = 0; c < d; c++) {
      const dlt = rowTarget[c] - rowCur[c];
      if (Math.abs(dlt) > 1e-6) {
        rowCur[c] += force ? dlt : dlt * 0.2;
        moving = true;
      }
      const v = rowCur[c] / rowMaxAbs;
      const hgt = v * ROW_H;
      m4.makeScale(1, Math.abs(hgt) < 0.006 ? 0.006 : hgt, 1);
      m4.setPosition((c - (d - 1) / 2) * (ROW_W / d), 0, 0);
      rowBars.setMatrixAt(c, m4);
      div(Math.max(-1, Math.min(1, v * 1.5)), col).multiplyScalar(0.8 + Math.abs(v) * 0.9);
      rowBars.setColorAt(c, col);
    }
    flushInst(rowBars);
    return moving;
  }

  // histogram of the same weights, each expressed as a fraction of its row's largest weight
  const HIST_Z = -7.4;
  const histGeo = new THREE.BoxGeometry((12 / BINS) * 0.8, 1, 0.16);
  histGeo.translate(0, 0.5, 0);
  const hist = instanced(histGeo, new THREE.MeshBasicMaterial(), BINS);
  hist.position.set(0, 0, HIST_Z);
  root.add(hist);
  const histCur = new Float32Array(BINS);
  const histTarget = new Float32Array(BINS);
  const histLabel = textSprite('how often each value occurs in the matrix', { size: 0.3, color: '#93a1bf', weight: 500 });
  histLabel.position.set(0, -0.4, HIST_Z);
  const levelLabel = textSprite(' ', { size: 0.46, font: 'display', weight: 700, color: '#eef3ff' });
  levelLabel.position.set(0, 5.5, HIST_Z);
  const rowLabel = textSprite('one row of the matrix, and the values its weights may take', { size: 0.3, color: '#93a1bf', weight: 500 });
  rowLabel.position.set(0, -ROW_H - 0.5, ROW_Z);
  root.add(histLabel, levelLabel, rowLabel);

  function setTargets() {
    const W = getModel(S.bits, S.method).t.get(NAME);
    histTarget.fill(0);
    for (let r = 0; r < d; r++) {
      let rowMax = 0;
      for (let c = 0; c < d; c++) rowMax = Math.max(rowMax, Math.abs(baseW[r * d + c]));
      for (let c = 0; c < d; c++) {
        const b = Math.round(((W[r * d + c] / rowMax + 1) / 2) * (BINS - 1));
        histTarget[Math.max(0, Math.min(BINS - 1, b))]++;
      }
    }
    let hm = 0;
    for (let i = 0; i < BINS; i++) hm = Math.max(hm, histTarget[i]);
    for (let i = 0; i < BINS; i++) histTarget[i] = Math.pow(histTarget[i] / hm, 0.6) * 4.6;
    for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) target[r * G + c] = W[r * d + c];
    // the row view and its level lines: the allowed values are k * step for k = -qmax..qmax
    let step = 0;
    for (let c = 0; c < d; c++) {
      rowTarget[c] = W[ROW * d + c];
      const a = Math.abs(W[ROW * d + c]);
      if (a > 1e-12 && (step === 0 || a < step)) step = a;
    }
    const lp = lvlGeo.attributes.position.array;
    lp.fill(0);
    const qmax = Math.pow(2, S.bits - 1) - 1;
    let lines = 0;
    if (S.bits <= 5 && step > 0) {
      for (let k = -qmax; k <= qmax && lines < MAX_LINES; k++) {
        const y = ((k * step) / rowMaxAbs) * ROW_H;
        lp.set([-ROW_W / 2 - 0.25, y, 0, ROW_W / 2 + 0.25, y, 0], lines * 6);
        lines++;
      }
    }
    lvlGeo.setDrawRange(0, lines * 2);
    lvlGeo.attributes.position.needsUpdate = true;
    animating = 1;
    levelLabel.userData.setText(S.bits >= 16 ? '16-bit floats: 65,536 possible values per weight' : `${S.bits} bits: ${quantLevels(S.bits)} possible values per weight`);
  }
  function paintHist(force) {
    for (let i = 0; i < BINS; i++) {
      histCur[i] += (histTarget[i] - histCur[i]) * (force ? 1 : 0.2);
      m4.makeScale(1, Math.max(0.01, histCur[i]), 1);
      m4.setPosition((i - (BINS - 1) / 2) * (12 / BINS), 0, 0);
      hist.setMatrixAt(i, m4);
      hist.setColorAt(i, col.copy(glow(PAL.coral, 0.6 + Math.min(1.1, histCur[i] * 0.25))));
    }
    flushInst(hist);
  }

  // ------------------------------------------------------------------ measurement (time-sliced so the page stays smooth)
  let job = null;
  const queue = [];
  function startNext() {
    const next = queue.shift();
    if (!next) {
      job = null;
      progress.textContent = '';
      measureBtn.disabled = false;
      return;
    }
    job = { ...next, m: getModel(next.bits, next.method), w: 0, p: 0, cache: null, loss: 0, count: 0 };
  }
  function runJob(budgetMs) {
    if (!job) return;
    const V = model.config.vocab;
    const t0 = performance.now();
    while (performance.now() - t0 < budgetMs) {
      const win = evalSet.windows[job.w];
      if (job.p === 0) job.cache = job.m.newCache();
      const logits = job.m.step(win[job.p], job.cache);
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(logits[i] - max);
      job.loss += Math.log(sum) + max - logits[win[job.p + 1]];
      job.count++;
      if (++job.p >= win.length - 1) {
        job.p = 0;
        if (++job.w >= EVAL_WINDOWS) {
          const loss = job.loss / job.count;
          results.set(key(job.bits, job.method), { loss, ppl: Math.exp(loss), tokens: job.count });
          refresh();
          startNext();
          return;
        }
      }
    }
    const done = (job.w * 128 + job.p) / (EVAL_WINDOWS * 128);
    progress.textContent = `measuring ${job.bits >= 16 ? '16-bit baseline' : job.bits + '-bit, ' + METHODS.find((m) => m.id === job.method).name.toLowerCase()} … ${(done * 100).toFixed(0)}%${queue.length ? ` (${queue.length} more queued)` : ''}`;
  }
  function measure(bits, method) {
    const k = key(bits, method);
    if (results.has(k) || (job && key(job.bits, job.method) === k) || queue.some((q) => key(q.bits, q.method) === k)) return;
    queue.push({ bits, method });
    if (!job) startNext();
  }
  function measureAll() {
    measureBtn.disabled = true;
    for (const b of BITS) measure(b, S.method);
    for (const b of BITS) if (b < 16) measure(b, S.method === 'mse' ? 'minmax' : 'mse');
  }

  /** Greedy continuation of a fixed prompt: the qualitative side of the damage. */
  function sampleText(bits, method) {
    const k = key(bits, method);
    if (samples.has(k)) return samples.get(k);
    const m = getModel(bits, method);
    const ids = [tokenizer.eot, ...tokenizer.encode(PROMPT)];
    const cache = m.newCache();
    let logits;
    for (const id of ids) logits = m.step(id, cache);
    const out = [];
    for (let i = 0; i < 28; i++) {
      let best = 0;
      for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
      if (best === tokenizer.eot) break;
      out.push(best);
      logits = m.step(best, cache);
    }
    const s = tokenizer.decode(out);
    samples.set(k, s);
    return s;
  }

  // ------------------------------------------------------------------ HUD
  const bitsSeg = segmented(BITS.map((b) => ({ value: b, label: b === 16 ? '16 (base)' : String(b) })), S.bits, (b) => select(b, S.method), { wrap: true });
  const methodSeg = segmented(METHODS.map((m) => ({ value: m.id, label: m.name })), S.method, (m) => select(S.bits, m));
  const stSize = stat({ label: 'Model size', value: '–' });
  const stRatio = stat({ label: 'Smaller by', value: '–' });
  const stPpl = stat({ label: 'Perplexity', value: '–', sub: 'lower is better' });
  const stErr = stat({ label: 'Weight error', value: '–', sub: 'RMS, relative' });
  const sampleEl = h('div.gpt-out');
  const progress = h('div.hint');
  const measureBtn = button('Measure every bit width', { kind: 'primary', onClick: measureAll });
  const chart = lineChart({ height: 150, yLog: true, xLabel: 'bits per weight', yLabel: 'perplexity on held-out stories (log)', series: [], xDomain: [2, 16] });
  const tbl = h('div');
  let rms = 0;
  for (const n of model.matrixNames) for (const v of model.t.get(n)) rms += v * v;
  let nMat = 0;
  for (const n of model.matrixNames) nMat += model.t.get(n).length;
  rms = Math.sqrt(rms / nMat);

  function refresh() {
    const m = getModel(S.bits, S.method);
    const bytes = model.sizeBytes(S.bits);
    stSize.set(fmt.bytes(bytes));
    stRatio.set((model.sizeBytes(16) / bytes).toFixed(1) + '×', 'than 16-bit');
    const r = results.get(key(S.bits, S.method));
    const base = results.get('16');
    stPpl.set(r ? r.ppl.toFixed(2) : '…', r && base ? (S.bits >= 16 ? 'baseline' : `${r.ppl >= base.ppl ? '+' : ''}${(((r.ppl - base.ppl) / base.ppl) * 100).toFixed(1)}% vs 16-bit`) : 'measuring');
    stErr.set(S.bits >= 16 ? '0%' : ((m.meta.rmse / rms) * 100).toFixed(1) + '%');
    chart.update(
      METHODS.map((mt) => ({
        name: mt.name,
        color: mt.color,
        data: BITS.map((b) => [b, results.get(key(b, mt.id))?.ppl]).filter((p) => p[1] != null).sort((a, b) => a[0] - b[0]),
      })),
    );
    tbl.replaceChildren(
      table(
        ['Bits', 'Size', 'Clipped', 'Min-max'],
        BITS.map((b) => [b === 16 ? '16 (base)' : String(b), fmt.bytes(model.sizeBytes(b)), results.get(key(b, 'mse'))?.ppl.toFixed(2) ?? '–', results.get(key(b, 'minmax'))?.ppl.toFixed(2) ?? '–']),
        { compact: true },
      ),
    );
  }
  function select(bits, method) {
    S.bits = bits;
    S.method = method;
    bitsSeg.set(bits);
    methodSeg.set(method);
    setTargets();
    measure(16, 'mse');
    measure(bits, method);
    sampleEl.replaceChildren(document.createTextNode(PROMPT), h('b', sampleText(bits, method)));
    refresh();
  }

  const steps = [
    {
      title: 'Weights are just numbers',
      text: `A model is its weights: this one has **${fmt.int(model.paramCount)}** of them, stored as 16-bit floats (${fmt.bytes(model.sizeBytes(16))}).\n\nThe carpet in the middle is one real matrix from the first block (${G} × ${G} of its ${d} × ${d} weights): red is positive, blue negative, and most weights are close to zero. The tall bars in front are a single row of it, the one marked by the white line. The bars at the back count how often each value occurs.`,
      enter: () => select(16, S.method),
    },
    {
      title: 'Round them onto a grid',
      text: 'Quantization keeps the model and throws away precision: every weight is rounded to the nearest of a few allowed values, with one scale factor per row. At **4 bits** a weight can take only 15 values. Watch the bars snap into terraces and the histogram collapse into spikes.',
      body: () => callout('Clipping matters. The naive way stretches the grid to cover the largest weight in the row, wasting levels on rare outliers. Clipping the range to minimize the rounding error gives the many small weights a finer grid.', 'info'),
      enter: () => select(4, S.method),
    },
    {
      title: 'Measure the damage',
      text: `**Perplexity** is how surprised the model is by text it has never seen (${EVAL_WINDOWS * 128} tokens of held-out stories here): the lower, the better. Press the button and the model is re-evaluated at every bit width, on this device. Typically 8 bits costs nothing, 4 bits costs a little, and 2 bits breaks a model that was not trained for it.`,
      body: () => callout('The text sample on the right is the same prompt continued greedily by the quantized model: watch it degrade.', 'key'),
      enter: () => {
        select(S.bits === 16 ? 4 : S.bits, S.method);
        measureAll();
      },
    },
    {
      title: 'What production systems do',
      text: `Rounding after training, as done here, is the crude version. Real quantizers (GPTQ, AWQ) use sample data to choose the rounding that preserves each layer’s outputs, and the best results come from **training with quantization in the loop**.\n\n${TECHNIQUES_2026.filter((t) => /4-bit|Ternary/.test(t.name)).map((t) => `**${t.name}.** ${t.text}`).join('\n\n')}\n\nLarge models tolerate quantization better than this tiny one: with billions of weights there is more redundancy to absorb the rounding error.`,
      body: () =>
        section(
          'Why it matters: memory',
          table(
            ['Model', '16-bit', '8-bit', '4-bit'],
            [7e9, 70e9, 1e12].map((p) => [fmt.compact(p) + ' parameters', fmt.bytes((p * 16) / 8), fmt.bytes((p * 8) / 8), fmt.bytes((p * 4) / 8)]),
            { compact: true },
          ),
          text('Weights only (parameters × bits ÷ 8). What is measured on this page is **quality**. The speed gain in practice comes from moving fewer bytes through memory and from integer or low-precision kernels on the hardware.', 'small'),
        ),
      enter: () => select(S.bits, S.method),
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'KV cache →', onFinish: () => ctx.router.go('perf/kvcache') });
  ctx.ui.left(header({ kicker: 'Experiment · Quantization', title: 'Fewer bits, same model?', lead: 'The real GPT from the lab, squeezed from 16 bits per weight down to 2, with the damage measured live.' }), guide.body);
  ctx.ui.right(
    section('Bits per weight', bitsSeg),
    section('Rounding method', methodSeg),
    statRow(stSize, stRatio),
    statRow(stPpl, stErr),
    section('The same prompt, continued by this version', sampleEl),
    h('div.btnrow', measureBtn),
    progress,
    section('Quality against size', chart, legendNote()),
    tbl,
  );
  ctx.ui.dock(guide.dock);

  function legendNote() {
    return text('Measured on this page with the weights rounded in place. Each point is the same held-out text.', 'small');
  }

  setTargets();
  paintField(true);
  paintHist(true);
  return {
    update() {
      runJob(9);
      if (animating) {
        const moving = paintField(false);
        paintHist(false);
        if (!moving) animating = 0;
      }
    },
  };
}
