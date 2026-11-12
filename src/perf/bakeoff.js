// Model Bake-off: six classic models trained live on five datasets. Each pillar is one model on one
// dataset: its height is the test accuracy, and its top face shows the decision regions it learned.
// All models and the benchmark loop are in ../ml/classical.js.
import * as THREE from 'three';
import { MODELS, bakeoff, winnerOf } from '../ml/classical.js';
import { DATASETS, DOMAIN } from '../ml/mlp.js';
import { PAL, SERIES, glow, divCss } from '../core/colors.js';
import { gridFloor, edges } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, slider, table, stat, statRow, fmt } from '../core/ui.js';
import { heatmap, barChart } from '../core/charts.js';
import { damp } from '../core/tween.js';

const STEP = 2.7; // spacing of the pillars
const SIDE = 2.2;
const RES = 30; // resolution of the decision-surface picture
const WHY = {
  linear: 'A straight line separates the classes, so the simplest model is already right. Flexible models can only match it, and with noisy data they tend to fit the noise instead.',
  moons: 'Two interlocking crescents: no straight line works. Models that can bend their boundary (neighbors, trees, a network) follow the curve; the linear ones are stuck near the best straight cut.',
  circles: 'One class surrounds the other. A line through a ring is wrong half the time, while naive Bayes wins cheaply here: the classes differ in spread, which is exactly what its bell curves measure.',
  spiral: 'Two arms wound around each other: the hardest shape here. Only models that follow fine local structure cope. k-NN simply looks at the nearest points; a small network needs more training than it gets.',
  xor: 'Opposite corners share a class. Each input alone says nothing, which defeats models that treat inputs separately (logistic regression, naive Bayes). A tree solves it with two axis-aligned cuts.',
};

export async function enter(ctx) {
  const { app, root } = ctx;
  const M = MODELS.length;
  const D = DATASETS.length;
  app.setView({ pos: [6.5, 17, 16], target: [-1.3, 1.1, 0.6], fit: 12.6, min: 4, max: 70, maxPolar: Math.PI * 0.48 });
  root.add(gridFloor({ size: 80, y: -0.02, color: '#6a4a1c', opacity: 0.25 }));

  const S = { noise: 0.3, trainSize: 300, seeds: 3, run: null, cells: null, done: 0, total: 0, sel: { m: -1, d: 0 }, baseSeed: 1, running: false };
  const px = (d) => (d - (D - 1) / 2) * STEP;
  const pz = (m) => (m - (M - 1) / 2) * STEP;
  const heightOf = (acc) => 0.25 + 3.6 * Math.max(0, (acc - 0.4) / 0.6);

  // ------------------------------------------------------------------ 3D: 30 pillars
  const boxGeo = new THREE.BoxGeometry(SIDE, 1, SIDE);
  boxGeo.translate(0, 0.5, 0);
  const topGeo = new THREE.PlaneGeometry(SIDE, SIDE);
  topGeo.rotateX(-Math.PI / 2);
  const pillars = [];
  for (let m = 0; m < M; m++)
    for (let d = 0; d < D; d++) {
      const g = new THREE.Group();
      g.position.set(px(d), 0, pz(m));
      const body = new THREE.Mesh(boxGeo, new THREE.MeshStandardMaterial({ color: '#131d33', roughness: 0.5, metalness: 0.2, transparent: true, opacity: 0.92 }));
      const outline = edges(boxGeo, PAL.amber, { k: 0.8, opacity: 0.35 });
      body.add(outline);
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = RES * 4;
      const tex = new THREE.CanvasTexture(canvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      const top = new THREE.Mesh(topGeo, new THREE.MeshBasicMaterial({ map: tex }));
      const label = textSprite('–', { size: 0.34, font: 'mono', weight: 700, color: '#eef3ff', bg: 'rgba(5,8,16,0.65)' });
      g.add(body, top, label);
      root.add(g);
      const p = { m, d, g, body, outline, top, canvas, tex, label, hgt: 0.25, target: 0.25, painted: false };
      body.scale.y = 0.25;
      ctx.pick.add([body, top], {
        click: () => select(m, d),
        tip: () => {
          const c = S.cells?.[m][d];
          return `${MODELS[m].name} on “${DATASETS[d].name}”\n${c?.runs ? `test accuracy ${(c.accuracy * 100).toFixed(1)}% (${c.runs} run${c.runs > 1 ? 's' : ''})` : 'not measured yet'}`;
        },
      });
      pillars.push(p);
    }
  MODELS.forEach((mo, m) => {
    const s = textSprite(mo.short, { size: 0.5, font: 'display', weight: 700, color: '#eef3ff', anchor: [1, 0.5] });
    s.position.set(px(0) - SIDE / 2 - 0.5, 0.3, pz(m));
    root.add(s);
  });
  DATASETS.forEach((ds, d) => {
    const s = textSprite(ds.name, { size: 0.5, font: 'display', weight: 700, color: '#ffd9a0', anchor: [0.5, 0.5] });
    s.position.set(px(d), 0.3, pz(M - 1) + SIDE / 2 + 0.9);
    root.add(s);
  });

  /** Draw the decision regions of a trained model and its training points onto a pillar's top. */
  function paintTop(p, model, train) {
    const g = p.canvas.getContext('2d');
    const X = new Float64Array(RES * RES * 2);
    for (let j = 0; j < RES; j++)
      for (let i = 0; i < RES; i++) {
        X[(j * RES + i) * 2] = ((i + 0.5) / RES - 0.5) * 2 * DOMAIN;
        X[(j * RES + i) * 2 + 1] = (0.5 - (j + 0.5) / RES) * 2 * DOMAIN;
      }
    const prob = model.predictProba(X, RES * RES);
    for (let j = 0; j < RES; j++)
      for (let i = 0; i < RES; i++) {
        g.fillStyle = divCss((prob[j * RES + i] - 0.5) * 1.7);
        g.fillRect(i * 4, j * 4, 4, 4);
      }
    const n = Math.min(train.n, 160);
    for (let k = 0; k < n; k++) {
      const x = (train.X[k * 2] / (2 * DOMAIN) + 0.5) * RES * 4;
      const y = (0.5 - train.X[k * 2 + 1] / (2 * DOMAIN)) * RES * 4;
      g.fillStyle = train.y[k] ? '#ffd0d0' : '#cfe2ff';
      g.beginPath();
      g.arc(x, y, 1.6, 0, Math.PI * 2);
      g.fill();
    }
    p.tex.needsUpdate = true;
    p.painted = true;
  }

  // ------------------------------------------------------------------ HUD
  const progress = h('div.hint');
  const runBtn = button('↻ Run again with new data', { kind: 'primary', onClick: () => start(S.baseSeed + 1000) });
  const heat = heatmap({ rows: MODELS.map((m) => m.short), cols: DATASETS.map((d) => d.name), values: MODELS.map(() => DATASETS.map(() => NaN)), max: 1, cell: 30, format: (v) => (v * 100).toFixed(1) + '%', title: 'Test accuracy (brighter is better)' });
  const bars = barChart({ rows: [], max: 1, format: (v) => (v * 100).toFixed(1) + '%' });
  const dsTitle = h('h3');
  const whyEl = h('div');
  const timeEl = h('div');
  const stBest = stat({ label: 'Most wins', value: '–' });
  const stMean = stat({ label: 'Best average', value: '–' });

  function refresh() {
    if (!S.cells) return;
    heat.update({ values: S.cells.map((row) => row.map((c) => (c.runs ? Math.max(0, (c.accuracy - 0.4) / 0.6) : NaN))), format: (v) => (40 + v * 60).toFixed(1) + '%' });
    const d = S.sel.d;
    const ds = DATASETS[d];
    const win = winnerOf(S.cells, d);
    dsTitle.textContent = `Dataset: ${ds.name}`;
    bars.update(
      MODELS.map((mo, m) => ({ label: mo.short, value: S.cells[m][d].runs ? S.cells[m][d].accuracy : 0, color: m === win ? SERIES[1] : SERIES[0], active: m === S.sel.m, note: S.cells[m][d].runs ? (S.cells[m][d].accuracy * 100).toFixed(1) + '%' : '…', onClick: () => select(m, d) }))
        .sort((a, b) => b.value - a.value),
    );
    whyEl.replaceChildren(text(`${ds.blurb} ${WHY[ds.id] ?? ''}`), win >= 0 ? callout(`Winner so far: **${MODELS[win].name}** (${MODELS[win].settings}).`, 'key') : '');
    timeEl.replaceChildren(
      table(
        ['Model', 'Train', 'Predict 1,000 points'],
        MODELS.map((mo, m) => [mo.short, S.cells[m][d].runs ? fmt.ms(S.cells[m][d].train) : '…', S.cells[m][d].runs ? fmt.ms(S.cells[m][d].predict) : '…']),
        { compact: true },
      ),
    );
    // overall: wins per model and mean accuracy
    const wins = MODELS.map(() => 0);
    for (let k = 0; k < D; k++) {
      const w = winnerOf(S.cells, k);
      if (w >= 0) wins[w]++;
    }
    const means = MODELS.map((_, m) => S.cells[m].reduce((s, c) => s + (c.runs ? c.accuracy : 0), 0) / D);
    const bw = wins.indexOf(Math.max(...wins));
    const bm = means.indexOf(Math.max(...means));
    if (S.done >= M * D) {
      stBest.set(MODELS[bw].short, `${wins[bw]} of ${D} datasets`);
      stMean.set(MODELS[bm].short, (means[bm] * 100).toFixed(1) + '% on average');
    }
  }
  function select(m, d) {
    S.sel = { m, d };
    refresh();
  }
  function start(seed = 1) {
    S.baseSeed = seed;
    S.cells = null;
    S.done = 0;
    S.total = S.seeds * M * D;
    S.running = true;
    for (const p of pillars) {
      p.target = 0.25;
      p.label.userData.setText('–');
    }
    const cells = MODELS.map(() => DATASETS.map(() => null));
    S.run = bakeoff({ trainSize: S.trainSize, testSize: 1000, noise: S.noise, seeds: S.seeds, baseSeed: seed });
    S.pending = cells;
    stBest.set('–', '');
    stMean.set('–', '');
  }
  function pump(budgetMs) {
    if (!S.running) return;
    const t0 = performance.now();
    while (performance.now() - t0 < budgetMs) {
      const r = S.run.next();
      if (r.done) {
        S.running = false;
        S.cells = r.value;
        progress.textContent = `Finished: ${S.total} trainings on this device (${S.seeds} fresh datasets per cell).`;
        refresh();
        return;
      }
      const ev = r.value;
      if (!ev) continue;
      S.done = ev.done;
      S.pending[ev.m][ev.d] = ev.cell;
      const p = pillars[ev.m * D + ev.d];
      if (ev.seed === 0) paintTop(p, ev.model, ev.train);
      p.target = heightOf(ev.cell.accuracy);
      p.label.userData.setText((ev.cell.accuracy * 100).toFixed(0) + '%');
      // a results table that fills in as cells arrive
      S.cells = S.pending.map((row) => row.map((c) => c ?? { runs: 0, accuracy: NaN }));
      progress.textContent = `training ${MODELS[ev.m].short} on ${DATASETS[ev.d].name} … ${ev.done} / ${ev.total}`;
      refresh();
    }
  }

  ctx.ui.left(
    header({ kicker: 'Experiment · Model bake-off', title: 'No free lunch', lead: 'Six classic models, five datasets, trained right now in your browser. **No model wins everywhere**: the best one depends on the shape of the problem.' }),
    text('Every pillar is one model on one dataset. Its **height** is the accuracy on 1,000 test points it never saw; the picture on **top** is what the model believes: red where it predicts one class, blue the other, with the training points drawn in.\n\nClick a pillar to read its column.'),
    h('div', dsTitle),
    whyEl,
    callout('Measured on this device with fresh random data, so numbers shift a little from run to run. Model settings are fixed and listed below each winner: none of them was tuned per dataset.', 'info'),
    h('div.btnrow', button('Next: hall of records →', { kind: 'ghost', onClick: () => ctx.router.go('perf/records') })),
  );
  ctx.ui.right(
    h('div.btnrow', runBtn),
    progress,
    slider({ label: 'Noise in the data', min: 0, max: 0.7, step: 0.05, value: S.noise, format: (v) => v.toFixed(2), onInput: (v) => (S.noise = v), onChange: () => start(S.baseSeed) }),
    slider({ label: 'Training points', min: 40, max: 600, step: 20, value: S.trainSize, format: (v) => String(v), onInput: (v) => (S.trainSize = v), onChange: () => start(S.baseSeed) }),
    statRow(stBest, stMean),
    heat,
    section('Ranking on the selected dataset', bars),
    section('Speed on the selected dataset', timeEl),
  );
  ctx.ui.dock(h('div.dockrow', h('span.hint', 'Height = test accuracy (40% to 100%) · top = learned decision regions · click a pillar')));

  start(1);
  return {
    update(dt) {
      pump(9);
      for (const p of pillars) {
        p.hgt = damp(p.hgt, p.target, 0.18, dt);
        p.body.scale.y = p.hgt;
        p.top.position.y = p.hgt + 0.012;
        p.top.visible = p.painted;
        p.label.position.y = p.hgt + 0.55;
        const win = S.cells && winnerOf(S.cells, p.d) === p.m && S.cells[p.m][p.d].runs > 0;
        const sel = S.sel.d === p.d;
        p.outline.material.color.copy(glow(win ? '#ffd166' : PAL.amber, win ? 2.6 : sel ? 1.2 : 0.6));
        p.outline.material.opacity = win ? 1 : sel ? 0.7 : 0.3;
      }
    },
  };
}
