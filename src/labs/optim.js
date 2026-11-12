// Optimizer Arena: five gradient-descent optimizers race on the same loss landscape.
// The math lives in ../ml/optim.js; this file draws it and wires the controls.
import * as THREE from 'three';
import { OPTIMIZERS, SURFACES, Run } from '../ml/optim.js';
import { SERIES, glow } from '../core/colors.js';
import { glowMat, gridFloor, ring, line } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, segmented, slider, tour, code, table, fmt } from '../core/ui.js';
import { lineChart } from '../core/charts.js';
import { damp } from '../core/tween.js';

const S = 6; // half-size of the landscape in world units
const H = 4.4; // height of the tallest point
const N = 128; // grid resolution
const MAX_TRAIL = 4000;
const COLOR = Object.fromEntries(OPTIMIZERS.map((o, i) => [o.id, SERIES[i]]));

export async function enter(ctx) {
  const { app, root } = ctx;
  app.setView({ pos: [7, 8, 11], target: [0, 1.2, 0], fit: 8.4, min: 5, max: 60, maxPolar: Math.PI * 0.49 });
  root.add(gridFloor({ size: 60, y: -0.02, color: '#31508c', opacity: 0.35 }));

  // ---------------------------------------------------------------- landscape mesh
  let surface = SURFACES[0];
  let fmax = 1;
  const heightOf = (f) => {
    const u = Math.min(1.4, Math.max(0, (f - surface.fmin) / (fmax - surface.fmin)));
    return (H * Math.log1p(surface.curve * u)) / Math.log1p(surface.curve);
  };
  const toWorld = (x, y, f, out = new THREE.Vector3()) => {
    const [x0, x1, y0, y1] = surface.domain;
    return out.set(((x - x0) / (x1 - x0) - 0.5) * 2 * S, heightOf(f ?? surface.f([x, y])), ((y - y0) / (y1 - y0) - 0.5) * 2 * S);
  };
  const fromWorld = (p) => {
    const [x0, x1, y0, y1] = surface.domain;
    return [x0 + (p.x / (2 * S) + 0.5) * (x1 - x0), y0 + (p.z / (2 * S) + 0.5) * (y1 - y0)];
  };

  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array((N + 1) * (N + 1) * 3);
  const aH = new Float32Array((N + 1) * (N + 1));
  const idx = [];
  for (let j = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const a = j * (N + 1) + i;
      idx.push(a, a + N + 1, a + 1, a + 1, a + N + 1, a + N + 2);
    }
  geo.setIndex(idx);
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('aH', new THREE.BufferAttribute(aH, 1));
  const surfMat = new THREE.ShaderMaterial({
    uniforms: {
      uLow: { value: new THREE.Color('#081428') },
      uMid: { value: new THREE.Color('#1c5cab') },
      uHigh: { value: new THREE.Color('#9ec5f4') },
    },
    vertexShader: /* glsl */ `
      attribute float aH;
      varying float vH;
      varying vec3 vWorld;
      void main() {
        vH = aH;
        vec4 w = modelMatrix * vec4(position, 1.0);
        vWorld = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uLow; uniform vec3 uMid; uniform vec3 uHigh;
      varying float vH;
      varying vec3 vWorld;
      void main() {
        vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
        float diff = 0.5 + 0.5 * abs(dot(n, normalize(vec3(0.35, 1.0, 0.25))));
        vec3 col = mix(uLow, uMid, smoothstep(0.0, 0.45, vH));
        col = mix(col, uHigh, smoothstep(0.45, 1.0, vH));
        float v = vH * 22.0;
        float w = fwidth(v);
        float f = fract(v);
        float contour = 1.0 - smoothstep(0.0, w * 1.4, min(f, 1.0 - f));
        col = col * diff + contour * 0.22 * uHigh;
        gl_FragColor = vec4(col, 0.97);
      }`,
    side: THREE.DoubleSide,
    transparent: true,
  });
  const mesh = new THREE.Mesh(geo, surfMat);
  root.add(mesh);

  const markers = new THREE.Group();
  root.add(markers);
  const startRing = ring(0.28, '#ffffff', { k: 2.2, segments: 40 });
  root.add(startRing);

  function buildSurface() {
    const [x0, x1, y0, y1] = surface.domain;
    fmax = -Infinity;
    for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) fmax = Math.max(fmax, surface.f([x0 + (i / N) * (x1 - x0), y0 + (j / N) * (y1 - y0)]));
    const v = new THREE.Vector3();
    for (let j = 0; j <= N; j++)
      for (let i = 0; i <= N; i++) {
        const k = j * (N + 1) + i;
        toWorld(x0 + (i / N) * (x1 - x0), y0 + (j / N) * (y1 - y0), undefined, v);
        v.toArray(pos, k * 3);
        aH[k] = v.y / H;
      }
    geo.attributes.position.needsUpdate = true;
    geo.attributes.aH.needsUpdate = true;
    geo.computeBoundingSphere();
    geo.computeBoundingBox();
    // mark the minima
    markers.clear();
    for (const m of surface.minima) {
      const p = toWorld(m[0], m[1]);
      const r = ring(0.2, '#ffe9a8', { k: 2.4, segments: 32 });
      r.position.copy(p).y += 0.03;
      const beam = line([p, p.clone().setY(p.y + 1.1)], '#ffe9a8', { k: 1.6, opacity: 0.55 });
      const lab = textSprite(surface.minima.length > 1 ? 'a minimum' : 'minimum', { size: 0.24, color: '#ffe9a8', weight: 600 });
      lab.position.copy(p).y += 1.3;
      markers.add(r, beam, lab);
    }
  }

  const axisX = textSprite('weight 1 →', { size: 0.3, color: '#93a1bf' });
  axisX.position.set(0, 0.1, S + 0.7);
  const axisZ = textSprite('weight 2 →', { size: 0.3, color: '#93a1bf' });
  axisZ.position.set(S + 1.1, 0.1, 0);
  const axisY = textSprite('↑ loss', { size: 0.3, color: '#93a1bf' });
  axisY.position.set(-S - 0.6, H * 0.75, -S - 0.6);
  root.add(axisX, axisZ, axisY);

  // ---------------------------------------------------------------- racers
  const racers = OPTIMIZERS.map((opt) => {
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.17, 20, 14), glowMat(COLOR[opt.id], { k: 2.3 }));
    const tgeo = new THREE.BufferGeometry();
    const tpos = new Float32Array(MAX_TRAIL * 3);
    tgeo.setAttribute('position', new THREE.BufferAttribute(tpos, 3));
    tgeo.setDrawRange(0, 0);
    const trail = new THREE.Line(tgeo, new THREE.LineBasicMaterial({ color: glow(COLOR[opt.id], 1.7), transparent: true, opacity: 0.95, depthWrite: false }));
    trail.frustumCulled = false;
    const label = textSprite(opt.name, { size: 0.26, color: '#eef3ff', weight: 650, bg: 'rgba(5,8,16,0.7)', onTop: true });
    root.add(ball, trail, label);
    return { opt, ball, trail, tpos, label, count: 0, run: null, active: true, target: new THREE.Vector3() };
  });

  let start = [...surface.start];
  let lrScale = 1;
  let noise = 0;
  let speed = 40; // optimizer steps per second
  let playing = false;
  let acc = 0;
  let finishedNotified = false;

  function pushTrail(r, p) {
    if (r.count >= MAX_TRAIL) {
      // thin the trail: keep every other point
      for (let i = 0; i < MAX_TRAIL / 2; i++) r.tpos.copyWithin(i * 3, i * 6, i * 6 + 3);
      r.count = MAX_TRAIL / 2;
    }
    p.toArray(r.tpos, r.count * 3);
    r.count++;
    r.trail.geometry.setDrawRange(0, r.count);
    r.trail.geometry.attributes.position.needsUpdate = true;
  }

  function reset() {
    acc = 0;
    finishedNotified = false;
    const p0 = toWorld(start[0], start[1]);
    startRing.position.copy(p0).y += 0.04;
    for (const r of racers) {
      r.run = new Run(surface, r.opt, { start, lrScale, noise });
      r.count = 0;
      r.target.copy(p0).y += 0.17;
      r.ball.position.copy(r.target);
      r.ball.visible = r.trail.visible = r.label.visible = r.active;
      pushTrail(r, p0.clone().setY(p0.y + 0.05));
    }
    refresh(true);
  }

  function stepAll(n = 1) {
    const v = new THREE.Vector3();
    for (let s = 0; s < n; s++)
      for (const r of racers) {
        if (!r.active || r.run.diverged) continue;
        r.run.step();
        if (r.run.diverged) continue;
        toWorld(r.run.p[0], r.run.p[1], r.run.loss, v);
        r.target.copy(v).y += 0.17;
        v.y += 0.05;
        pushTrail(r, v);
      }
  }

  // ---------------------------------------------------------------- HUD
  const chart = lineChart({ height: 132, yLog: true, xLabel: 'step', yLabel: 'loss above the minimum (log)', series: [] });
  const board = h('div');
  const playBtn = button('▶ Run', { kind: 'primary', onClick: () => setPlaying(!playing) });
  const surfSeg = segmented(SURFACES.map((s) => ({ value: s.id, label: s.name })), surface.id, (id) => setSurface(id), { wrap: true });
  const blurb = h('div.txt.small');
  const optChips = h('div.chips');

  function renderChips() {
    optChips.replaceChildren(
      ...racers.map((r) => {
        const c = h('span.chip.click', { style: { '--c': COLOR[r.opt.id], opacity: r.active ? 1 : 0.4 }, title: 'Click to include or exclude' }, h('i', { style: { width: '14px', height: '3px', borderRadius: '2px', background: COLOR[r.opt.id], display: 'inline-block' } }), r.opt.name);
        c.addEventListener('click', () => {
          r.active = !r.active;
          renderChips();
          reset();
        });
        return c;
      }),
    );
  }

  function refresh(force) {
    const act = racers.filter((r) => r.active);
    chart.update(act.map((r) => ({ name: r.opt.name, color: COLOR[r.opt.id], data: r.run.history.map(([s, l]) => [s, Math.max(l - surface.fmin, 1e-7)]) })));
    const ranked = [...act].sort((a, b) => (a.run.solvedAt ?? 1e9) - (b.run.solvedAt ?? 1e9) || a.run.loss - b.run.loss);
    board.replaceChildren(
      table(
        ['Optimizer', 'Solved at step', 'Loss now'],
        ranked.map((r, i) => [
          h('span', h('i', { style: { display: 'inline-block', width: '14px', height: '3px', borderRadius: '2px', background: COLOR[r.opt.id], marginRight: '7px', verticalAlign: 'middle' } }), r.opt.name + (i === 0 && r.run.solvedAt != null ? '  ★' : '')),
          r.run.diverged ? 'diverged' : r.run.solvedAt != null ? fmt.int(r.run.solvedAt) : 'not yet',
          r.run.diverged ? '∞' : (r.run.loss - surface.fmin).toExponential(1),
        ]),
        { compact: true },
      ),
    );
    if (force) blurb.replaceChildren(text(surface.blurb).firstChild);
  }

  function setPlaying(v) {
    playing = v;
    playBtn.textContent = playing ? '❚❚ Pause' : '▶ Run';
  }

  function setSurface(id, opts = {}) {
    surface = SURFACES.find((s) => s.id === id);
    surfSeg.set(id);
    start = [...surface.start];
    speed = { ravine: 40, banana: 260, saddle: 25, four: 30, bumpy: 30 }[id];
    speedSlider.set(Math.log10(speed));
    buildSurface();
    if (opts.active) racers.forEach((r) => (r.active = opts.active.includes(r.opt.id)));
    renderChips();
    reset();
    setPlaying(opts.play ?? false);
  }

  const lrSlider = slider({ label: 'Step size (learning rate)', min: -1, max: 0.7, step: 0.01, value: 0, format: (v) => '×' + Math.pow(10, v).toFixed(2), onInput: (v) => { lrScale = Math.pow(10, v); reset(); } });
  const noiseSlider = slider({ label: 'Gradient noise (mini-batch)', min: 0, max: 1.5, step: 0.01, value: 0, format: (v) => v.toFixed(2), onInput: (v) => { noise = v; reset(); } });
  const speedSlider = slider({ label: 'Simulation speed', min: 0.3, max: 2.8, step: 0.01, value: Math.log10(speed), format: (v) => Math.round(Math.pow(10, v)) + ' steps/s', onInput: (v) => (speed = Math.pow(10, v)) });

  const steps = [
    {
      title: 'A loss landscape',
      text: 'Training a model means searching for weights that make its error (the **loss**) small. With only two weights we can draw every possibility: each point on the floor is a pair of weights, and the height is how wrong the model would be there.\n\nReal models have billions of weights, so the landscape has billions of dimensions, but the game is the same: **walk downhill without being able to see the map**.',
      body: () => callout('Click anywhere on the surface to drop the optimizers at that spot.', 'key'),
      enter: () => setSurface('four', { active: ['sgd'], play: false }),
    },
    {
      title: 'Follow the slope',
      text: 'The **gradient** points uphill, so step the other way. That is all of stochastic gradient descent (SGD). In this ravine the slope across is 100 times steeper than the slope along, so a step size small enough to be stable across the valley is painfully slow along it.',
      body: () => code(OPTIMIZERS[0].formula),
      enter: () => setSurface('ravine', { active: ['sgd'], play: true }),
    },
    {
      title: 'Add inertia',
      text: '**Momentum** keeps a running velocity: gradients that agree add up, gradients that flip sign cancel. The ball accelerates along the valley floor. **Nesterov** looks ahead to where the velocity is about to take it and corrects sooner.',
      body: () => code(OPTIMIZERS[1].formula + '\n\n' + OPTIMIZERS[2].formula),
      enter: () => setSurface('ravine', { active: ['sgd', 'momentum', 'nesterov'], play: true }),
    },
    {
      title: 'Adapt each direction',
      text: '**RMSProp** divides every weight’s step by the recent size of its own gradient, so steep directions get small steps and flat ones large steps. **Adam** adds momentum on top and is the default for training transformers. Notice RMSProp hovering near the bottom: its step never shrinks.',
      body: () => code(OPTIMIZERS[3].formula + '\n\n' + OPTIMIZERS[4].formula),
      enter: () => setSurface('ravine', { active: ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam'], play: true }),
    },
    {
      title: 'Traps',
      text: 'On the bumpy bowl plain descent stops in the first dip it finds, while momentum rolls through. Now raise **gradient noise**: the jitter of mini-batches shakes SGD out of shallow dips. Noise is not only a cost, it is part of why SGD generalizes.',
      enter: () => setSurface('bumpy', { active: ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam'], play: true }),
    },
    {
      title: 'Your arena',
      text: 'Pick any landscape, change the step size, switch optimizers on and off, and click the surface to choose a starting point. On **Four minima** the adaptive optimizers end in a different valley from the others: the optimizer shapes _which_ solution you get, not only how fast.',
      body: () => callout('Each optimizer starts at about 60% of its own best step size for the landscape (found by a sweep), the safety margin people use in practice. The step-size slider scales all of them together.', 'info', 'Is the race fair?'),
      enter: () => setSurface('four', { active: ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam'], play: true }),
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'Next lab →', onFinish: () => ctx.router.go('labs/cnn') });

  ctx.ui.left(
    header({ kicker: 'Lab · Optimizer Arena', title: 'How a model finds its way downhill', lead: 'Five optimizers, one landscape, the same starting point.' }),
    guide.body,
  );
  ctx.ui.right(
    section('Landscape', surfSeg, blurb),
    h('div.btnrow', playBtn, button('Step', { onClick: () => { setPlaying(false); stepAll(1); refresh(); } }), button('Reset', { onClick: () => { reset(); setPlaying(false); } })),
    section('Optimizers', optChips),
    lrSlider,
    noiseSlider,
    speedSlider,
    section('Loss', chart),
    section('Scoreboard', board),
  );
  ctx.ui.dock(guide.dock);

  ctx.pick.add(mesh, {
    cursor: 'crosshair',
    tip: (hit) => {
      const q = fromWorld(hit.point);
      return `loss ${surface.f(q).toPrecision(3)}   (click to start here)`;
    },
    click: (hit) => {
      start = fromWorld(hit.point);
      reset();
      setPlaying(true);
    },
  });
  ctx.key(' ', () => setPlaying(!playing));

  buildSurface();
  renderChips();
  reset();

  let uiTimer = 0;
  return {
    update(dt) {
      if (playing) {
        acc += dt * speed;
        const n = Math.min(80, Math.floor(acc));
        if (n > 0) {
          acc -= n;
          stepAll(n);
        }
        const act = racers.filter((r) => r.active);
        const done = act.length > 0 && act.every((r) => r.run.diverged || r.run.solvedAt != null || r.run.steps >= 6000);
        if (done && !finishedNotified) {
          finishedNotified = true;
          setPlaying(false);
          refresh();
        }
      }
      for (const r of racers) {
        if (!r.active) continue;
        r.ball.position.x = damp(r.ball.position.x, r.target.x, 0.05, dt);
        r.ball.position.y = damp(r.ball.position.y, r.target.y, 0.05, dt);
        r.ball.position.z = damp(r.ball.position.z, r.target.z, 0.05, dt);
        r.label.position.copy(r.ball.position).y += 0.45;
      }
      uiTimer += dt;
      if (playing && uiTimer > 0.2) {
        uiTimer = 0;
        refresh();
      }
    },
  };
}
