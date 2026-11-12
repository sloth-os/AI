// Diffusion Lab: noise becomes shape. Particles start as Gaussian noise and are denoised step by
// step into a 3D shape, using the exact denoiser in ../ml/diffusion.js (no trained network).
import * as THREE from 'three';
import { SHAPES, SAMPLERS, Sampler, schedule, quality, rng } from '../ml/diffusion.js';
import { PAL, SERIES, glow } from '../core/colors.js';
import { points, gridFloor } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, segmented, slider, tour, stat, statRow, code } from '../core/ui.js';
import { lineChart } from '../core/charts.js';

const N = 1600; // particles
const M = 420; // target points per shape
const SCALE = 6; // world units per model unit
const SCOLOR = Object.fromEntries(SAMPLERS.map((s, i) => [s.id, SERIES[i]]));
const COMPARE_STEPS = [4, 8, 12, 16, 24, 32, 48];

export async function enter(ctx) {
  const { app, root } = ctx;
  app.setView({ pos: [3, 3.5, 16], target: [0, 0.3, 0], fit: 12, min: 3, max: 90, autoRotate: 0.5 });
  root.add(gridFloor({ size: 110, y: -9, color: '#5a2a55', opacity: 0.25 }));

  const seedRnd = rng(7);
  const targets = Object.fromEntries(SHAPES.map((s) => [s.id, s.make(M, seedRnd)]));
  const pool = new Float32Array(SHAPES.length * 110 * 3);
  SHAPES.forEach((s, i) => pool.set(s.make(110, seedRnd), i * 330));

  const S = { shape: 'knot', sampler: 'ddim', steps: 30, guidance: 1, run: null, mode: 'idle', tFwd: 0, lerp: 1, wait: 0, results: {}, compare: null };

  // ------------------------------------------------------------------ 3D
  const pos = new Float32Array(N * 3);
  const from = new Float32Array(N * 3);
  const to = new Float32Array(N * 3);
  const colr = new Float32Array(N * 3);
  const size = new Float32Array(N).fill(0.17);
  const cloud = points(pos, colr, size);
  root.add(cloud);
  const GN = M;
  const gpos = new Float32Array(GN * 3);
  const gcol = new Float32Array(GN * 3).fill(0.16);
  const ghost = points(gpos, gcol, new Float32Array(GN).fill(0.09));
  root.add(ghost);
  const arrowGeo = new THREE.BufferGeometry();
  const ARROWS = 70;
  arrowGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ARROWS * 6), 3));
  const arrows = new THREE.LineSegments(arrowGeo, new THREE.LineBasicMaterial({ color: glow('#ffffff', 1.6), transparent: true, opacity: 0.55, depthWrite: false }));
  arrows.frustumCulled = false;
  arrows.visible = false;
  root.add(arrows);
  const caption = textSprite(' ', { size: 0.62, font: 'mono', weight: 600, color: '#e9d3f2' });
  caption.position.set(0, 9.4, 0);
  root.add(caption);

  const base = glow(PAL.magenta, 1);
  function paintColors(heat) {
    // heat 1 = pure noise (dim, bluish), heat 0 = finished (bright)
    for (let i = 0; i < N; i++) {
      const k = 0.55 + (1 - heat) * 1.9;
      colr[i * 3] = base.r * k * (1 - heat * 0.5);
      colr[i * 3 + 1] = base.g * k + heat * 0.25;
      colr[i * 3 + 2] = base.b * k + heat * 0.5;
    }
    cloud.geometry.attributes.color.needsUpdate = true;
  }
  function setGhost() {
    const T = targets[S.shape];
    for (let i = 0; i < GN * 3; i++) gpos[i] = T[i] * SCALE;
    ghost.geometry.attributes.position.needsUpdate = true;
  }
  const show = (src) => {
    for (let i = 0; i < N * 3; i++) pos[i] = src[i] * SCALE;
    cloud.geometry.attributes.position.needsUpdate = true;
  };

  // ------------------------------------------------------------------ actions
  function generate() {
    S.run = new Sampler({ sampler: S.sampler, steps: S.steps, n: N, targets: targets[S.shape], pool, guidance: S.guidance, seed: (Math.random() * 1e9) | 0 });
    S.mode = 'reverse';
    S.lerp = 1;
    S.wait = 0;
    S.t0 = performance.now();
    from.set(S.run.x);
    to.set(S.run.x);
    show(S.run.x);
    paintColors(1);
    genBtn.textContent = '❚❚ Stop';
    refreshStats();
    if (!app._userMoved) app.frame({ target: [0, 0.3, 0], dir: [0.2, 0.22, 1], fit: 12 }, 0.8);
  }
  function stop() {
    S.mode = 'idle';
    genBtn.textContent = '▶ Generate';
  }
  let fwdBase = null;
  let fwdNoise = null;
  function dissolve() {
    stop();
    const T = targets[S.shape];
    const r = rng((Math.random() * 1e9) | 0);
    fwdBase = new Float32Array(N * 3);
    fwdNoise = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const j = (r() * M) | 0;
      for (let c = 0; c < 3; c++) {
        fwdBase[i * 3 + c] = T[j * 3 + c];
        fwdNoise[i * 3 + c] = r.normal();
      }
    }
    S.tFwd = 0;
    S.mode = 'forward';
  }
  function startCompare() {
    stop();
    const jobs = [];
    for (const s of SAMPLERS) for (const st of COMPARE_STEPS) jobs.push({ sampler: s.id, steps: st });
    S.results = {};
    S.compare = { jobs, i: 0, run: null, total: jobs.length };
    cmpBtn.disabled = true;
  }
  function pumpCompare(budget) {
    const c = S.compare;
    if (!c) return;
    const t0 = performance.now();
    while (performance.now() - t0 < budget) {
      if (!c.run) {
        if (c.i >= c.jobs.length) {
          S.compare = null;
          cmpBtn.disabled = false;
          progress.textContent = 'Done. Lower is sharper: the curve that drops first needs the fewest steps.';
          return;
        }
        const j = c.jobs[c.i];
        c.run = new Sampler({ sampler: j.sampler, steps: j.steps, n: 500, targets: targets[S.shape], seed: 11 });
      }
      if (!c.run.step()) {
        const j = c.jobs[c.i];
        const q = quality(c.run.x, 500, targets[S.shape]);
        (S.results[j.sampler] ??= []).push([j.steps, q.sharpness]);
        c.run = null;
        c.i++;
        progress.textContent = `comparing … ${c.i} / ${c.total}`;
        chart.update(SAMPLERS.map((s) => ({ name: s.name, color: SCOLOR[s.id], data: S.results[s.id] ?? [] })));
      }
    }
  }

  // ------------------------------------------------------------------ HUD
  const genBtn = button('▶ Generate', { kind: 'primary', onClick: () => (S.mode === 'reverse' ? stop() : generate()) });
  const cmpBtn = button('Compare samplers', { onClick: startCompare });
  const progress = h('div.hint');
  const stStep = stat({ label: 'Model calls', value: '0' });
  const stSharp = stat({ label: 'Distance', value: '–', sub: 'to the shape; lower is sharper' });
  const stCover = stat({ label: 'Coverage', value: '–', sub: 'of the shape' });
  const chart = lineChart({ height: 150, yLog: true, xLabel: 'steps', yLabel: 'mean distance to the shape (log)', series: [] });
  const samplerInfo = h('div');
  const shapeSeg = segmented(SHAPES.map((s) => ({ value: s.id, label: s.name })), S.shape, (v) => ((S.shape = v), setGhost(), generate()), { wrap: true });
  const samplerSeg = segmented(SAMPLERS.map((s) => ({ value: s.id, label: s.name })), S.sampler, (v) => setSampler(v));
  const stepSlider = slider({ label: 'Steps', min: 2, max: 100, step: 1, value: S.steps, format: String, onInput: (v) => (S.steps = v), onChange: () => generate() });
  const guideSlider = slider({ label: 'Guidance strength', min: 0, max: 4, step: 0.1, value: S.guidance, format: (v) => v.toFixed(1), onInput: (v) => (S.guidance = v), onChange: () => generate() });
  function setSampler(v) {
    S.sampler = v;
    samplerSeg.set(v);
    samplerInfo.replaceChildren(text(SAMPLERS.find((s) => s.id === v).blurb, 'small'));
    generate();
  }
  function refreshStats() {
    const r = S.run;
    if (!r) return;
    stStep.set(String(r.evals), `step ${r.i} of ${r.steps}`);
    if (r.i >= r.steps || r.i % 4 === 0) {
      const q = quality(r.x, 400, targets[S.shape]);
      stSharp.set(q.sharpness.toFixed(3));
      stCover.set((q.coverage * 100).toFixed(0) + '%');
    }
  }

  const steps = [
    {
      title: 'Data dissolves into noise',
      text: 'Start from the end. Take real data (here, points that form a shape) and add a little random noise, again and again. After enough steps nothing of the shape is left: only a featureless Gaussian cloud. This **forward process** needs no intelligence at all.',
      enter: () => dissolve(),
    },
    {
      title: 'What the model learns',
      text: 'The model’s whole job is to look at a noisy point and answer: **where did this probably come from?** The white lines show that answer for a few particles. Early on, in heavy noise, every particle is pulled toward the middle of the shape; as the noise falls, the answers sharpen.',
      body: () => code('x̂₀ = E[ clean point | noisy point, noise level ]'),
      enter: () => ((arrows.visible = true), (S.steps = 40), stepSlider.set(40), setSampler('ddim')),
      leave: () => (arrows.visible = false),
    },
    {
      title: 'Sculpting from noise',
      text: 'Generation runs the film backward. Begin with pure noise, ask the model where each particle belongs, move a small step that way, and repeat at the next lower noise level. Sixty small steps of the original **DDPM** sampler turn static into structure.',
      enter: () => ((S.steps = 60), stepSlider.set(60), setSampler('ddpm')),
    },
    {
      title: 'Fewer steps',
      text: 'Every step is a full run of the network, so steps are what generation costs. **DDIM** drops the random kick and can stride further. **Flow matching** trains toward straight paths from noise to data, and straight paths tolerate big steps. Set the steps to 8, switch samplers, then press **Compare samplers** to measure all three.',
      enter: () => ((S.steps = 8), stepSlider.set(8), setSampler('flow')),
    },
    {
      title: 'Guidance',
      text: 'Image models are trained with and without the prompt. At sampling time the two predictions are mixed: **prediction = unconditional + w × (conditional − unconditional)**. Here “unconditional” knows all five shapes at once. At w = 0 you get a generic blur of everything; at w = 1 the plain conditional model; push w higher and the shape is over-sharpened and thins out.',
      enter: () => ((S.steps = 30), stepSlider.set(30), (S.guidance = 0), guideSlider.set(0), setSampler('ddim')),
      leave: () => ((S.guidance = 1), guideSlider.set(1)),
    },
    {
      title: 'From particles to pictures',
      text: 'An image model does the same thing with millions of numbers instead of three: a text encoder turns the prompt into vectors, a network denoises a small **latent** image (for example 4 × 64 × 64) over a few dozen steps, and a decoder turns the result into pixels.',
      body: () => [
        callout('This lab uses the mathematically exact denoiser, which can only reproduce the points it was given. A real network cannot fit that exact answer; it has to generalize, and that imperfection is exactly why it can produce images nobody has seen.', 'info', 'One honest difference'),
        h('div.btnrow', button('Follow a text-to-image pipeline →', { onClick: () => ctx.router.go('flow/diffusion') })),
      ],
      enter: () => ((S.steps = 30), stepSlider.set(30), setSampler('flow')),
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'Next lab →', onFinish: () => ctx.router.go('labs/world') });
  ctx.ui.left(header({ kicker: 'Lab · Diffusion', title: 'Noise becomes shape', lead: `${N.toLocaleString('en-US')} particles, three samplers, and the exact mathematics of denoising.` }), guide.body);
  ctx.ui.right(
    section('Shape', shapeSeg),
    section('Sampler', samplerSeg, samplerInfo),
    stepSlider,
    guideSlider,
    h('div.btnrow', genBtn, button('Dissolve', { title: 'Run the forward process', onClick: dissolve })),
    statRow(stStep, stSharp, stCover),
    h('div.btnrow', cmpBtn),
    progress,
    section('Quality against steps', chart),
  );
  ctx.ui.dock(guide.dock);
  ctx.key(' ', () => genBtn.click());

  setGhost();
  samplerInfo.replaceChildren(text(SAMPLERS.find((s) => s.id === S.sampler).blurb, 'small'));
  paintColors(1);

  return {
    update(dt) {
      pumpCompare(7);
      if (S.mode === 'forward') {
        S.tFwd = Math.min(1, S.tFwd + dt / 3.2);
        const { a, s } = schedule(S.tFwd);
        for (let i = 0; i < N * 3; i++) pos[i] = (a * fwdBase[i] + s * fwdNoise[i]) * SCALE;
        cloud.geometry.attributes.position.needsUpdate = true;
        paintColors(S.tFwd);
        caption.userData.setText(`forward · noise level ${(s * 100).toFixed(0)}%`);
        if (S.tFwd >= 1) S.mode = 'idle';
      } else if (S.mode === 'reverse' && S.run) {
        const r = S.run;
        // glide between sampler steps so the motion stays smooth while the math runs every few frames
        const gap = Math.max(0.03, Math.min(0.14, 2.6 / r.steps));
        S.lerp = Math.min(1, S.lerp + dt / gap);
        for (let i = 0; i < N * 3; i++) pos[i] = (from[i] + (to[i] - from[i]) * S.lerp) * SCALE;
        cloud.geometry.attributes.position.needsUpdate = true;
        if (S.lerp >= 1) {
          if (r.i < r.steps) {
            from.set(r.x);
            r.step();
            to.set(r.x);
            S.lerp = 0;
            paintColors(r.t);
            refreshStats();
            if (arrows.visible) {
              const ap = arrowGeo.attributes.position.array;
              for (let k = 0; k < ARROWS; k++) {
                const i = k * 17;
                for (let c = 0; c < 3; c++) {
                  ap[k * 6 + c] = from[i * 3 + c] * SCALE;
                  ap[k * 6 + 3 + c] = r.x0[i * 3 + c] * SCALE;
                }
              }
              arrowGeo.attributes.position.needsUpdate = true;
            }
          } else {
            stop();
            refreshStats();
            if (!app._userMoved) app.frame({ target: [0, 0, 0], dir: [0.25, 0.5, 1], fit: 7.4 }, 1.4);
          }
        }
        caption.userData.setText(`${SAMPLERS.find((s) => s.id === r.kind).name} · step ${r.i} / ${r.steps}`);
      }
    },
  };
}
