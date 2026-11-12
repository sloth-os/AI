// KV cache, measured: generate the same text twice with the real GPT, once recomputing every
// earlier token at every step and once reusing their cached keys and values. Both sides run for
// real on this device; the grids light up exactly the (token, block) cells being computed.
import * as THREE from 'three';
import '../labs/gpt.css';
import { loadTinyGPT } from '../ml/tinygpt.js';
import { SERIES, glow } from '../core/colors.js';
import { instanced, setInst, flushInst, gridFloor, edges } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, slider, stat, statRow, table, tour, legend, fmt } from '../core/ui.js';
import { lineChart } from '../core/charts.js';

const PROMPT = 'Once upon a time, there was a little girl named';
const COLOR = { none: SERIES[1], kv: SERIES[0] };

export async function enter(ctx) {
  const { app, root } = ctx;
  ctx.ui.loading('Loading the model');
  const { model, tokenizer } = await loadTinyGPT(ctx.asset);
  const { layers: L, d, ctx: CTX } = model.config;
  const promptIds = [tokenizer.eot, ...tokenizer.encode(PROMPT)];
  const P = promptIds.length;
  let NEW = 64; // tokens to generate

  app.setView({ pos: [1.5, 3, 15], target: [0, 0.3, 0], fit: 8.7, min: 4, max: 60 });
  root.add(gridFloor({ size: 70, y: -5.4, color: '#25477f', opacity: 0.3 }));

  // ------------------------------------------------------------------ 3D: two grids of (token position x block) cells
  const COLS = 96;
  const CW = 15 / COLS;
  const RH = 0.92; // row height
  const cellGeo = new THREE.BoxGeometry(CW * 0.78, RH * 0.82, 0.3);
  function makeGrid(y, title, sub, color) {
    const g = new THREE.Group();
    g.position.y = y;
    const mesh = instanced(cellGeo, new THREE.MeshBasicMaterial(), COLS * L);
    for (let c = 0; c < COLS; c++) for (let l = 0; l < L; l++) setInst(mesh, c * L + l, (c - (COLS - 1) / 2) * CW, (l - (L - 1) / 2) * RH, 0);
    flushInst(mesh);
    const top = (L * RH) / 2;
    const frame = edges(new THREE.BoxGeometry(15.4, L * RH + 0.24, 0.5), color, { k: 1.1, opacity: 0.55 });
    const t = textSprite(title, { size: 0.44, font: 'display', weight: 700, color: '#eef3ff', anchor: [0, 0.5] });
    t.position.set(-7.7, top + 0.5, 0);
    const s = textSprite(sub, { size: 0.25, color: '#93a1bf', weight: 500, anchor: [0, 0.5] });
    s.position.set(-7.7 + t.scale.x + 0.35, top + 0.47, 0);
    const readout = textSprite(' ', { size: 0.32, font: 'mono', weight: 650, color: '#eef3ff', anchor: [1, 0.5] });
    readout.position.set(7.7, top + 0.5, 0);
    const axis = textSprite('token position →', { size: 0.21, color: '#7f8aa5', weight: 500, anchor: [1, 0.5] });
    axis.position.set(7.7, -top - 0.36, 0);
    g.add(mesh, frame, t, s, readout, axis);
    root.add(g);
    return { g, mesh, readout, heat: new Float32Array(COLS * L), stored: new Uint8Array(COLS), color: glow(color, 1) };
  }
  const grids = {
    none: makeGrid(3.1, 'No cache', 'every step recomputes every earlier token', COLOR.none),
    kv: makeGrid(-2.7, 'KV cache', 'every step computes one new token and reuses the rest', COLOR.kv),
  };
  const blockLab = textSprite(`${L} blocks`, { size: 0.21, color: '#7f8aa5', weight: 500, anchor: [1, 0.5] });
  blockLab.position.set(-7.95, 3.1, 0);
  root.add(blockLab);

  const col = new THREE.Color();
  function paintGrid(G, dt) {
    for (let c = 0; c < COLS; c++)
      for (let l = 0; l < L; l++) {
        const i = c * L + l;
        G.heat[i] *= Math.exp(-dt * 5);
        const base = G.stored[c] ? 0.34 : 0.045;
        G.mesh.setColorAt(i, col.copy(G.color).multiplyScalar(base + G.heat[i] * 2.6));
      }
    G.mesh.instanceColor.needsUpdate = true;
  }
  const touch = (G, c) => {
    if (c >= COLS) return;
    for (let l = 0; l < L; l++) G.heat[c * L + l] = 1;
  };

  // ------------------------------------------------------------------ the race (real computation, time-sliced)
  // Both sides produce the same greedy continuation. "No cache" replays the whole sequence through
  // the network for every new token; "KV cache" feeds only the newest token.
  const R = { running: false, done: false };
  const argmax = (a) => {
    let b = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
    return b;
  };
  function resetRace() {
    R.running = false;
    R.done = false;
    R.ids = [...promptIds];
    R.kv = { cache: model.newCache(), fed: 0, ms: [], total: 0, ops: 0, tokens: 0, logits: null };
    R.none = { cache: null, pos: 0, len: 0, ms: [], total: 0, ops: 0, tokens: 0, stepStart: 0, acc: 0 };
    R.tokens = []; // the greedy continuation, decided by the cached side
    for (const G of Object.values(grids)) {
      G.heat.fill(0);
      G.stored.fill(0);
    }
    refresh();
  }

  /** Cached side: prefill the prompt once, then one step per token. */
  function runKv(budget) {
    const k = R.kv;
    const t0 = performance.now();
    while (performance.now() - t0 < budget && k.tokens < NEW) {
      const ops0 = model.ops;
      const s0 = performance.now();
      if (k.fed < P) {
        k.logits = model.step(R.ids[k.fed], k.cache);
        touch(grids.kv, k.fed);
        grids.kv.stored[Math.min(COLS - 1, k.fed)] = 1;
        k.fed++;
        k.prefill = (k.prefill || 0) + performance.now() - s0;
        k.ops += model.ops - ops0;
        continue;
      }
      const next = argmax(k.logits);
      R.tokens.push(next);
      R.ids.push(next);
      k.logits = model.step(next, k.cache);
      const ms = performance.now() - s0;
      touch(grids.kv, k.fed);
      grids.kv.stored[Math.min(COLS - 1, k.fed)] = 1;
      k.fed++;
      k.tokens++;
      k.ms.push([P + k.tokens, ms]);
      k.total += ms;
      k.ops += model.ops - ops0;
    }
  }

  /** Uncached side: for each new token, run the whole sequence so far from scratch. */
  function runNone(budget) {
    const n = R.none;
    const t0 = performance.now();
    while (performance.now() - t0 < budget && n.tokens < NEW && n.tokens < R.tokens.length) {
      if (n.pos === 0) {
        n.cache = model.newCache(); // a fresh, empty cache = nothing is remembered between steps
        n.len = P + n.tokens;
        n.acc = 0;
        n.ops0 = model.ops;
        grids.none.stored.fill(0);
      }
      const s0 = performance.now();
      model.step(R.ids[n.pos], n.cache);
      n.acc += performance.now() - s0;
      touch(grids.none, n.pos);
      if (++n.pos >= n.len) {
        n.pos = 0;
        n.tokens++;
        n.ms.push([P + n.tokens, n.acc]);
        n.total += n.acc;
        n.ops += model.ops - n.ops0;
      }
    }
  }

  // ------------------------------------------------------------------ HUD
  const runBtn = button('▶ Run the race', { kind: 'primary', onClick: () => (R.done ? (resetRace(), (R.running = true)) : (R.running = !R.running), syncBtn()) });
  const syncBtn = () => (runBtn.textContent = R.running ? '❚❚ Pause' : R.done ? '↻ Run again' : '▶ Run the race');
  const stKv = stat({ label: 'KV cache', value: '–', unit: 'ms', color: COLOR.kv });
  const stNone = stat({ label: 'No cache', value: '–', unit: 'ms', color: COLOR.none });
  const stRatio = stat({ label: 'Speed-up', value: '–' });
  const stMem = stat({ label: 'Cache memory', value: '–', sub: 'keys + values' });
  const outEl = h('div.gpt-out');
  const chart = lineChart({ height: 150, xLabel: 'tokens in context', yLabel: 'time for one more token (ms)', series: [] });
  const tbl = h('div');
  const lenSlider = slider({ label: 'Tokens to generate', min: 16, max: CTX - P - 2, step: 1, value: NEW, format: (v) => String(v), onChange: (v) => ((NEW = v), resetRace(), syncBtn()), onInput: () => {} });

  function refresh() {
    const k = R.kv;
    const n = R.none;
    stKv.set(k.total ? k.total.toFixed(0) : '–', `${k.tokens} tokens`);
    stNone.set(n.total ? n.total.toFixed(0) : '–', `${n.tokens} tokens`);
    // fair ratio: compare the time both sides needed for the tokens the slow side has finished
    const m = n.tokens;
    const kvSame = k.ms.slice(0, m).reduce((s, x) => s + x[1], 0);
    stRatio.set(m > 2 && kvSame > 0 ? (n.total / kvSame).toFixed(1) + '×' : '–', m > 2 ? `over the first ${m} tokens` : '');
    const kvBytes = 2 * L * d * 2 * (P + k.tokens);
    stMem.set(fmt.bytes(kvBytes), `${fmt.bytes(2 * L * d * 2)} per token`);
    grids.kv.readout.userData.setText(k.total ? `${k.total.toFixed(0)} ms · ${fmt.compact(k.ops)} multiply-adds` : ' ');
    grids.none.readout.userData.setText(n.total ? `${n.total.toFixed(0)} ms · ${fmt.compact(n.ops)} multiply-adds` : ' ');
    chart.update([
      { name: 'KV cache', color: COLOR.kv, data: k.ms },
      { name: 'No cache', color: COLOR.none, data: n.ms },
    ]);
    outEl.replaceChildren(document.createTextNode(PROMPT), h('b', tokenizer.decode(R.tokens.slice(0, Math.max(n.tokens, 0)))), h('span', { style: { opacity: 0.45 } }, tokenizer.decode(R.tokens.slice(n.tokens))));
    tbl.replaceChildren(
      table(
        ['', 'KV cache', 'No cache'],
        [
          ['Tokens generated', String(k.tokens), String(n.tokens)],
          ['Total time', k.total ? fmt.ms(k.total) : '–', n.total ? fmt.ms(n.total) : '–'],
          ['Last token took', k.ms.length ? fmt.ms(k.ms[k.ms.length - 1][1]) : '–', n.ms.length ? fmt.ms(n.ms[n.ms.length - 1][1]) : '–'],
          ['Multiply-adds', fmt.compact(k.ops), fmt.compact(n.ops)],
        ],
        { compact: true },
      ),
    );
  }

  const steps = [
    {
      title: 'The waste in generation',
      text: 'A language model generates one token at a time, and each new token attends to all earlier ones. The naive way is to run the **whole sequence** through the network again for every new token. But the earlier tokens have not changed, and with a causal mask their vectors cannot change either.\n\nPress **Run the race**. Both rows generate the same text with the real model; each lit cell is one token passing through one block.',
      enter: () => {},
    },
    {
      title: 'Keep the keys and values',
      text: 'Attention needs only two things from an earlier token: its **key** and its **value** in every block. Store them once (the dim cells), and each new token costs one pass through the network instead of a pass for every token so far.\n\nSo the cached line in the chart stays flat while the uncached one climbs with the length of the context: total work grows with n instead of n².',
      body: () => legend([{ color: COLOR.kv, label: 'KV cache', shape: 'line' }, { color: COLOR.none, label: 'No cache', shape: 'line' }]),
      enter: () => {},
    },
    {
      title: 'The price is memory',
      text: `The cache holds one key and one value vector per token, per block. For this toy that is ${fmt.bytes(2 * L * d * 2)} per token. For a 70-billion-parameter model of the Llama-3 class (80 blocks, 8 key-value heads of 128 dimensions, 16-bit) it is 2 × 80 × 1,024 × 2 bytes ≈ **0.33 MB per token**, so a 100,000-token conversation pins about 33 GB of GPU memory for one user.\n\nThat is why long contexts are expensive, why providers bill cached prompt tokens separately, and why serving engines page and share this memory between requests.`,
      body: () => h('div.btnrow', button('See batching →', { onClick: () => ctx.router.go('perf/batching') }), button('Back to the GPT lab', { onClick: () => ctx.router.go('labs/gpt') })),
      enter: () => {},
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'Batching →', onFinish: () => ctx.router.go('perf/batching') });
  ctx.ui.left(header({ kicker: 'Experiment · KV cache', title: 'Why generation is fast', lead: 'Same model, same text, generated twice: once the slow way, once with the cache every real system uses. Timed on this device.' }), guide.body);
  ctx.ui.right(
    h('div.btnrow', runBtn, button('Reset', { onClick: () => (resetRace(), syncBtn()) })),
    lenSlider,
    statRow(stKv, stNone),
    statRow(stRatio, stMem),
    section('Time per new token', chart),
    section('Generated text (bright = finished by both)', outEl),
    tbl,
    callout('Timings are real but small: this model has one million parameters. The shape of the two curves is what carries over to large models.', 'info'),
  );
  ctx.ui.dock(guide.dock);
  ctx.key(' ', () => runBtn.click());

  resetRace();
  let uiTimer = 0;
  return {
    update(dt) {
      if (R.running) {
        runKv(3);
        runNone(9);
        if (R.none.tokens >= NEW) {
          R.running = false;
          R.done = true;
          syncBtn();
          refresh();
        }
      }
      paintGrid(grids.none, dt);
      paintGrid(grids.kv, dt);
      uiTimer += dt;
      if (R.running && uiTimer > 0.15) {
        uiTimer = 0;
        refresh();
      }
    },
  };
}
