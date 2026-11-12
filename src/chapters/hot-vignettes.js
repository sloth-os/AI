// The ten vignettes of the Hot 10 chapter: one bespoke, procedural scene per issue, plus the small
// emblem that stands for it on its monolith in the overview.
//
//   VIGNETTES[viz].build(env)   -> { group, update(dt), panel }      env = { ctx, issue, low }
//   VIGNETTES[viz].emblem(color) -> Group about 1.6 units across, animated by userData.tick(t)
//   VIGNETTES[viz].view          -> camera framing in the vignette's own coordinates
//
// A vignette is built when its issue is opened and disposed of when the visitor moves on. Scenes
// stand on a floor at y = 0, are centred on the origin and face +z.
//
// Honesty: every figure a scene displays is read out of the fact text in src/data/hot.js (see
// `figure`), or is computed live. Where a scene runs a toy model, its panel says so.
import * as THREE from 'three';
import { HOT_AS_OF } from '../data/hot.js';
import { PAL, SERIES, glow, seq } from '../core/colors.js';
import { glowMat, stdMat, glassMat, line, segments, edges, tube, arc, rbox, ring, halo, points, instanced, setInst, flushInst, gridFloor, rng } from '../core/geom.js';
import { textSprite, FONTS } from '../core/labels.js';
import { h, section, slider, segmented, toggle, button, fmt } from '../core/ui.js';
import { barChart } from '../core/charts.js';
import { damp, clamp, lerp, smoothstep, ease } from '../core/tween.js';

const TAU = Math.PI * 2;
const INK = '#e6ecf8';
const DIM = '#a3b0cc';
const BAD = PAL.red; // an attack, an escape, a retraction
const WARN = PAL.amber;
const GOOD = PAL.mint;
const UP = new THREE.Vector3(0, 1, 0);
const _v = new THREE.Vector3();
const _w = new THREE.Vector3();
const _c = new THREE.Color();
const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();

// ---------------------------------------------------------------------------- shared helpers
/**
 * Read a figure out of the dated facts, so that a scene never shows a number that is not in the data
 * file. Returns the RegExp match (or null, and the scene leaves that label out).
 */
function figure(issue, re) {
  for (const f of issue.facts) {
    const m = f.text.match(re);
    if (m) return m;
  }
  if (import.meta.env?.DEV) console.error(`[hot] figure not found in src/data/hot.js for "${issue.id}": ${re}`);
  return null;
}
const num = (s) => Number(String(s).replace(/,/g, ''));

const sph = (r, mat, seg = 16) => new THREE.Mesh(new THREE.SphereGeometry(r, seg, Math.max(6, seg - 4)), mat);
const at = (obj, x, y, z) => {
  obj.position.set(x, y, z);
  return obj;
};
/** Static label in the scene. */
function lbl(str, x, y, z, o = {}) {
  return at(textSprite(str, { size: 0.3, color: DIM, weight: 600, ...o }), x, y, z);
}

/**
 * Text sprite for read-outs that change while the scene runs. `textSprite` keeps one texture per
 * distinct string until the chapter is left, which suits labels but not counters, so this one
 * redraws a single fixed-size canvas instead. sprite.userData.set(text) updates it.
 */
function liveLabel(o = {}) {
  const size = o.size ?? 0.4; // world height of one em
  const ems = o.width ?? 6; // canvas width in em
  const rows = o.rows ?? 1;
  const PX = 64;
  const canvas = document.createElement('canvas');
  canvas.width = Math.ceil(ems * PX);
  canvas.height = Math.ceil(rows * PX * 1.25);
  const g = canvas.getContext('2d');
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: !o.onTop, fog: false }));
  const align = o.align ?? 'center';
  sprite.center.set(align === 'left' ? 0 : align === 'right' ? 1 : 0.5, 0.5);
  sprite.scale.set(size * ems, size * rows * 1.25, 1);
  sprite.renderOrder = 12;
  let shown = null;
  sprite.userData.set = (str) => {
    if (str === shown) return;
    shown = str;
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.font = `${o.weight ?? 650} ${PX}px ${FONTS[o.font ?? 'display']}`;
    g.textBaseline = 'middle';
    g.textAlign = align;
    g.fillStyle = o.color ?? INK;
    const x = align === 'left' ? 6 : align === 'right' ? canvas.width - 6 : canvas.width / 2;
    String(str).split('\n').forEach((l, i) => g.fillText(l, x, PX * 1.25 * (i + 0.5) + 3));
    tex.needsUpdate = true;
  };
  sprite.userData.set(o.text ?? '');
  return sprite;
}

/** A pool of soft glowing dots drawn in one call. Park a dot with hide(i). */
function dots(n, color, size = 0.14, k = 2.4) {
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  const siz = new Float32Array(n).fill(size);
  const c = glow(color, k);
  for (let i = 0; i < n; i++) {
    col[i * 3] = c.r;
    col[i * 3 + 1] = c.g;
    col[i * 3 + 2] = c.b;
    pos[i * 3 + 1] = -1e4;
  }
  const obj = points(pos, col, siz);
  const geo = obj.geometry;
  return {
    obj,
    pos,
    col,
    siz,
    n,
    set(i, x, y, z) {
      pos[i * 3] = x;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = z;
    },
    hide(i) {
      pos[i * 3 + 1] = -1e4;
    },
    tint(i, color3, mul = 1) {
      col[i * 3] = color3.r * mul;
      col[i * 3 + 1] = color3.g * mul;
      col[i * 3 + 2] = color3.b * mul;
    },
    flush(all) {
      geo.attributes.position.needsUpdate = true;
      if (all) geo.attributes.color.needsUpdate = geo.attributes.size.needsUpdate = true;
    },
  };
}

/** Sample a curve once so that particles can follow it without allocating. */
function sample(curve, n = 48) {
  const a = new Float32Array((n + 1) * 3);
  for (let i = 0; i <= n; i++) curve.getPoint(i / n, _v).toArray(a, i * 3);
  return a;
}
/** Write the point at fraction f of a sampled path into out[o..o+2]. */
function along(a, f, out, o = 0) {
  const n = a.length / 3 - 1;
  const x = Math.min(0.99999, Math.max(0, f)) * n;
  const i = Math.floor(x);
  const k = x - i;
  const j = i * 3;
  out[o] = a[j] + (a[j + 3] - a[j]) * k;
  out[o + 1] = a[j + 1] + (a[j + 4] - a[j + 1]) * k;
  out[o + 2] = a[j + 2] + (a[j + 5] - a[j + 2]) * k;
}
const _p3 = [0, 0, 0];
/** Move an object to fraction f of a sampled path. */
function ride(obj, a, f) {
  along(a, f, _p3);
  obj.position.set(_p3[0], _p3[1], _p3[2]);
}

/** Holographic floor under a scene. */
function stage(group, color, o = {}) {
  group.add(gridFloor({ size: 54, y: 0, color, opacity: o.opacity ?? 0.36, fade: o.fade ?? 15, step: o.step ?? 1 }));
}

/** A line whose two end points are rewritten every frame. */
function beamLine(color, k = 2) {
  const l = line([[0, 0, 0], [0, 0, 0]], color, { k, opacity: 0.99 });
  l.frustumCulled = false;
  l.visible = false;
  const a = l.geometry.attributes.position;
  l.userData.put = (x0, y0, z0, x1, y1, z1) => {
    a.setXYZ(0, x0, y0, z0);
    a.setXYZ(1, x1, y1, z1);
    a.needsUpdate = true;
  };
  return l;
}

/** "What you are looking at" list for the right panel. items: [color, text, shape?] */
function key(items) {
  return h('div.hot-key', items.map(([c, label, shape]) => h('div.li', h('i' + (shape ? '.' + shape : ''), { style: { '--c': c } }), h('span', label))));
}
/** Small table of live counters. rows: [label, color?] -> returns { el, set(i, value) } */
function tally(rows) {
  const cells = rows.map(() => h('dd', '0'));
  const el = h('dl.hot-tally', rows.map(([label, c], i) => [h('dt', c && h('i', { style: { '--c': c } }), label), cells[i]]));
  return { el, set: (i, v) => (cells[i].textContent = String(v)) };
}
const fine = (...kids) => h('div.hot-fine', ...kids);
const css = (color) => '#' + new THREE.Color(color).getHexString();

const pct = (v) => {
  if (v >= 0.9995) return '100%';
  if (v >= 0.1) return (v * 100).toFixed(1) + '%';
  if (v >= 0.0001) return (v * 100).toFixed(2) + '%';
  return 'under 0.01%';
};

// ---------------------------------------------------------------------------- 1 · containment
// An agent in a sandbox: tool calls stay inside, stray probes hit the walls and are logged by the
// monitor, and now and then one finds a gap. On the floor, a chain of n steps shows how per-step
// reliability p compounds into p^n.
function containment({ low }) {
  const g = new THREE.Group();
  stage(g, '#2b5d92');
  const C = 2.15; // half-size of the sandbox
  const MID = new THREE.Vector3(0, 3.7, 0);
  const box = at(new THREE.Group(), MID.x, MID.y, MID.z);
  g.add(box);

  const cubeGeo = new THREE.BoxGeometry(C * 2, C * 2, C * 2);
  box.add(new THREE.Mesh(cubeGeo, glassMat('#11305c', 0.1, { emissive: PAL.cyan, ek: 0.1 })));
  box.add(edges(cubeGeo, PAL.cyan, { k: 1.9 }));
  g.add(lbl('sandbox', MID.x - C + 0.1, MID.y + C + 0.3, MID.z + C, { color: '#bfe9ff', anchor: [0, 0.5] }));

  // the gap: a small misconfigured opening in the right-hand wall
  const GAP = new THREE.Vector3(C, 0.5, 0.5);
  const gs = 0.3;
  box.add(line([[C, GAP.y - gs, GAP.z - gs], [C, GAP.y + gs, GAP.z - gs], [C, GAP.y + gs, GAP.z + gs], [C, GAP.y - gs, GAP.z + gs]], WARN, { k: 2.4, closed: true }));
  g.add(lbl('gap', MID.x + C + 0.15, MID.y + GAP.y + 0.62, MID.z + GAP.z, { color: '#ffd9a0', size: 0.26, anchor: [0, 0.5] }));

  // the agent and its tools
  const core = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 2), glowMat('#d8f1ff', { k: 1.35 }));
  const shell = edges(new THREE.IcosahedronGeometry(0.78, 1), PAL.cyan, { k: 1.5, opacity: 0.75 });
  box.add(core, shell, halo(PAL.cyan, 3.2, 0.3));
  g.add(lbl('agent', MID.x, MID.y + 1.15, MID.z, { color: INK, size: 0.27 }));
  const TOOLS = [[-1.4, 1.2, 1.15], [1.3, -1.25, 1.2], [-1.25, -1.2, -1.3], [1.2, 1.3, -1.15]];
  const toolGeo = rbox(0.4, 0.4, 0.4, 0.08);
  const toolEdge = new THREE.BoxGeometry(0.42, 0.42, 0.42);
  const spokes = [];
  for (const p of TOOLS) {
    const m = at(new THREE.Mesh(toolGeo, stdMat('#10261f', { emissive: GOOD, ek: 0.55 })), ...p);
    m.add(edges(toolEdge, GOOD, { k: 1.8 }));
    box.add(m);
    spokes.push([0, 0, 0], p);
  }
  box.add(segments(spokes, GOOD, { k: 1.1, opacity: 0.5 }));
  g.add(lbl('tools in scope', MID.x + TOOLS[0][0], MID.y + TOOLS[0][1] + 0.5, MID.z + TOOLS[0][2], { size: 0.24, color: '#b9f2dd' }));
  const calls = dots(TOOLS.length * 2, GOOD, 0.15, 2.6);
  box.add(calls.obj);

  // probes that wander off-task and hit a wall
  const NP = low ? 6 : 10;
  const probes = dots(NP, WARN, 0.17, 2.6);
  box.add(probes.obj);
  const pr = Array.from({ length: NP }, () => ({ on: false, x: 0, y: 0, z: 0, dx: 0, dy: 0, dz: 0, fade: 0 }));
  const ripples = Array.from({ length: 5 }, () => {
    const r = ring(1, WARN, { segments: 28, k: 2.2, opacity: 0.99 });
    r.visible = false;
    box.add(r);
    return { obj: r, t: 1 };
  });
  let nextRipple = 0;
  function ripple(x, y, z, nx, ny, nz) {
    const r = ripples[nextRipple++ % ripples.length];
    r.obj.position.set(x, y, z);
    r.obj.quaternion.setFromUnitVectors(UP, _v.set(nx, ny, nz));
    r.t = 0;
    r.obj.visible = true;
  }

  // the monitor: a ring that orbits the sandbox, reads every wall hit and writes it to a log
  const tilt = at(new THREE.Group(), MID.x, MID.y, MID.z);
  tilt.rotation.set(1.12, 0, 0.3);
  const spin = new THREE.Group();
  tilt.add(spin);
  g.add(tilt);
  const MR = 4.0;
  spin.add(new THREE.Mesh(new THREE.TorusGeometry(MR, 0.026, 8, 128), glowMat('#cfe0ff', { k: 1.25, opacity: 0.85 })));
  const sensor = at(sph(0.16, glowMat('#ffffff', { k: 2.4 })), MR, 0, 0);
  sensor.add(halo('#bcd6ff', 1.5, 0.5));
  spin.add(sensor, at(sph(0.07, glowMat('#cfe0ff', { k: 1.8 }), 10), -MR, 0, 0));
  const sensorAt = new THREE.Vector3();
  const Z = new THREE.Vector3(0, 0, 1);
  const beams = Array.from({ length: 3 }, () => {
    const l = beamLine('#ffffff', 2.2);
    g.add(l);
    return { obj: l, t: 1 };
  });
  let nextBeam = 0;
  function report(x, y, z, color) {
    const b = beams[nextBeam++ % beams.length];
    b.obj.userData.put(sensorAt.x, sensorAt.y, sensorAt.z, x, y, z);
    b.obj.material.color.copy(color);
    b.t = 0;
    b.obj.visible = true;
  }
  const LOGN = 9;
  const LOG = { x: -6.5, y: 6.3, z: 0 };
  const logMesh = instanced(new THREE.BoxGeometry(1, 0.085, 0.02), new THREE.MeshBasicMaterial({ color: 0xffffff }), LOGN);
  g.add(logMesh, lbl('monitor log', LOG.x, LOG.y + 0.42, LOG.z, { size: 0.25, anchor: [0, 0.5] }));
  const logs = [];
  const C_HIT = glow(WARN, 1.5);
  const C_ESC = glow(BAD, 2.2);
  const C_WHITE = glow('#ffffff', 2.2);
  function writeLog(color) {
    logs.unshift({ w: 0.5 + Math.random() * 0.8, color });
    if (logs.length > LOGN) logs.pop();
    for (let i = 0; i < LOGN; i++) {
      const e = logs[i];
      if (e) setInst(logMesh, i, LOG.x + e.w / 2, LOG.y - i * 0.23, LOG.z, e.w, 1, 1, _c.copy(e.color).multiplyScalar(1 - (i / LOGN) * 0.8));
      else setInst(logMesh, i, 0, 0, 0, 0);
    }
    flushInst(logMesh);
  }
  for (let i = 0; i < LOGN; i++) setInst(logMesh, i, 0, 0, 0, 0);

  // the world outside, and the probe that gets there
  const OUT = new THREE.Vector3(6.2, 4.9, -0.5);
  const world = at(new THREE.Group(), OUT.x + 0.5, OUT.y, OUT.z);
  for (const [x, y, z, s] of [[0, 0, 0, 0.44], [0.5, 0.42, -0.2, 0.3], [0.52, -0.4, 0.25, 0.26]]) {
    const geo = new THREE.BoxGeometry(s, s, s);
    const m = at(new THREE.Mesh(geo, stdMat('#1b2236', { emissive: PAL.steel, ek: 0.25 })), x, y, z);
    m.add(edges(geo, PAL.steel, { k: 1.2, opacity: 0.7 }));
    world.add(m);
  }
  g.add(world, lbl('outside systems', OUT.x + 0.75, OUT.y + 0.95, OUT.z, { size: 0.25 }));
  const escDot = sph(0.12, glowMat(WARN, { k: 2.6 }), 12);
  escDot.add(halo(BAD, 1.3, 0.55));
  const cage = new THREE.Group();
  const cageA = ring(0.5, BAD, { segments: 32, k: 2.6, opacity: 0.99 });
  const cageB = ring(0.5, BAD, { segments: 32, k: 2.6, opacity: 0.99 });
  cageB.rotation.x = Math.PI / 2;
  cage.add(cageA, cageB);
  const caughtLbl = lbl('caught', 0, 0, 0, { size: 0.27, color: '#ffd0d5', weight: 700 });
  g.add(escDot, cage, caughtLbl);
  escDot.visible = cage.visible = caughtLbl.visible = false;
  const gapAt = new THREE.Vector3().copy(MID).add(GAP);
  const stopAt = new THREE.Vector3().lerpVectors(gapAt, OUT, 0.7);
  const esc = { phase: 0, t: 0, next: 2.4 };
  const C_AMBER = glow(WARN, 2.6);
  const C_RED = glow(BAD, 2.8);

  // ---- the chain of steps on the floor
  const NB = 500;
  const beadXZ = new Float32Array(NB * 2);
  {
    let ang = 0.5;
    let rad = 3.7;
    for (let k = 0; k < NB; k++) {
      beadXZ[k * 2] = Math.sin(ang) * rad;
      beadXZ[k * 2 + 1] = Math.cos(ang) * rad;
      ang += 0.262 / rad;
      rad += (0.44 / TAU) * (0.262 / rad);
    }
  }
  const beads = instanced(new THREE.SphereGeometry(0.086, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }), NB);
  g.add(beads);
  const runner = halo('#ffffff', 0.75, 0.95, 2.4);
  const flash = halo(BAD, 1.5, 0, 2.4);
  const endRing = ring(0.2, '#ffffff', { segments: 20, k: 2 });
  const endLbl = liveLabel({ size: 0.25, width: 5, color: DIM, weight: 600, font: 'body' });
  g.add(runner, flash, endRing, endLbl);
  const big = liveLabel({ size: 1.05, width: 4.6, weight: 700 });
  const bigCap = liveLabel({ size: 0.27, width: 12, color: DIM, weight: 600, font: 'body' });
  const READ = { x: 5.4, y: 1.9, z: 4.2 };
  g.add(at(big, READ.x, READ.y, READ.z), at(bigCap, READ.x, READ.y - 0.82, READ.z));

  let p = 0.99;
  let n = 200;
  const run = { i: 0, acc: 0, wait: 0, runs: 0, clean: 0 };
  const counts = { hits: 0, caught: 0 };

  const bigEl = h('div.hot-big', h('div.l', 'Chance that every step goes right'), h('div.v', ''), h('div.f', ''));
  const runsEl = fine('');
  const guard = tally([['Probes stopped at a wall and logged', css(WARN)], ['Escapes through the gap, caught', css(BAD)]]);

  function refresh() {
    const pn = Math.pow(p, n);
    for (let k = 0; k < NB; k++) {
      if (k < n) {
        const alive = Math.pow(p, k + 1);
        seq(0.2 + 0.8 * alive, _c).multiplyScalar(0.75 + 1.5 * alive);
        setInst(beads, k, beadXZ[k * 2], 0.1, beadXZ[k * 2 + 1], 1, 1, 1, _c);
      } else setInst(beads, k, 0, 0, 0, 0);
    }
    flushInst(beads);
    endRing.position.set(beadXZ[(n - 1) * 2], 0.1, beadXZ[(n - 1) * 2 + 1]);
    endLbl.position.set(beadXZ[(n - 1) * 2], 0.62, beadXZ[(n - 1) * 2 + 1]);
    endLbl.userData.set(n === 1 ? 'step 1' : `step ${n}`);
    big.userData.set(pct(pn));
    bigCap.userData.set(n === 1 ? 'chance that the one step goes right' : `chance that all ${n} steps go right`);
    bigEl.querySelector('.v').textContent = pct(pn);
    bigEl.querySelector('.f').textContent = `p^n = ${p.toFixed(3)}^${n}`;
    run.i = run.acc = run.wait = run.runs = run.clean = 0;
    runsEl.textContent = 'The dot on the floor runs the chain over and over, failing each step with probability 1 − p.';
  }

  const pSlider = slider({ label: 'Reliability of each step (p)', min: 90, max: 99.9, step: 0.1, value: 99, format: (v) => v.toFixed(1) + '%', onInput: (v) => ((p = v / 100), refresh()) });
  const nSlider = slider({ label: 'Steps in the task (n)', min: 1, max: 500, step: 1, value: 200, format: (v) => String(Math.round(v)), onInput: (v) => ((n = Math.round(v)), refresh()) });
  refresh();

  let T = 0;
  let spawn = 0.3;
  let ui = 0;
  return {
    group: g,
    panel: [
      section('Try it · how reliability compounds', pSlider, nSlider, bigEl, runsEl),
      section(
        'In the scene',
        key([
          [css(PAL.cyan), 'The sandbox around the agent'],
          [css(GOOD), 'Tool calls that stay in scope'],
          [css(WARN), 'A probe that hits a wall: stopped, and logged by the orbiting monitor'],
          [css(BAD), 'A probe that slips through a gap and is caught outside'],
        ]),
        guard.el,
      ),
    ],
    update(dt) {
      T += dt;
      core.scale.setScalar(1 + 0.06 * Math.sin(T * 2.3));
      shell.rotation.set(T * 0.23, -T * 0.31, 0);
      spin.rotation.z = T * 0.55;
      sensorAt.set(MR, 0, 0).applyAxisAngle(Z, spin.rotation.z).applyEuler(tilt.rotation).add(MID);

      // tool calls: out and back along each spoke
      for (let j = 0; j < TOOLS.length; j++)
        for (let e = 0; e < 2; e++) {
          const f = (T * 0.55 + j * 0.27 + e * 0.5) % 1;
          const u = f < 0.5 ? f * 2 : 2 - f * 2;
          calls.set(j * 2 + e, TOOLS[j][0] * u, TOOLS[j][1] * u, TOOLS[j][2] * u);
        }
      calls.flush();

      // probes
      spawn -= dt;
      if (spawn <= 0) {
        spawn = 0.42 + Math.random() * 0.5;
        const q = pr.find((x) => !x.on && x.fade <= 0);
        if (q) {
          _v.randomDirection();
          q.on = true;
          q.x = q.y = q.z = 0;
          q.dx = _v.x * 2.3;
          q.dy = _v.y * 2.3;
          q.dz = _v.z * 2.3;
        }
      }
      for (let i = 0; i < NP; i++) {
        const q = pr[i];
        if (q.on) {
          q.x += q.dx * dt;
          q.y += q.dy * dt;
          q.z += q.dz * dt;
          const ax = Math.abs(q.x);
          const ay = Math.abs(q.y);
          const az = Math.abs(q.z);
          const m = Math.max(ax, ay, az);
          if (m >= C - 0.03) {
            const k = (C - 0.03) / m;
            q.x *= k;
            q.y *= k;
            q.z *= k;
            q.on = false;
            q.fade = 0.5;
            ripple(q.x, q.y, q.z, m === ax ? Math.sign(q.x) : 0, m === ay ? Math.sign(q.y) : 0, m === az ? Math.sign(q.z) : 0);
            report(MID.x + q.x, MID.y + q.y, MID.z + q.z, C_WHITE);
            writeLog(C_HIT);
            counts.hits++;
          }
          probes.set(i, q.x, q.y, q.z);
          probes.siz[i] = 0.17;
        } else if (q.fade > 0) {
          q.fade -= dt;
          probes.siz[i] = 0.17 * Math.max(0, q.fade / 0.5);
          if (q.fade <= 0) probes.hide(i);
        }
      }
      probes.flush(true);
      for (const r of ripples) {
        if (!r.obj.visible) continue;
        r.t += dt / 0.7;
        r.obj.scale.setScalar(0.08 + 0.62 * ease.out(Math.min(1, r.t)));
        r.obj.material.opacity = 0.99 * (1 - Math.min(1, r.t));
        if (r.t >= 1) r.obj.visible = false;
      }
      for (const b of beams) {
        if (!b.obj.visible) continue;
        b.t += dt / 0.45;
        b.obj.material.opacity = 0.85 * (1 - Math.min(1, b.t));
        if (b.t >= 1) b.obj.visible = false;
      }

      // the escape
      esc.t += dt;
      if (esc.phase === 0 && esc.t > esc.next) {
        esc.phase = 1;
        esc.t = 0;
        escDot.visible = true;
        escDot.material.color.copy(C_AMBER);
        escDot.children[0].material.opacity = 0;
        escDot.scale.setScalar(1);
      } else if (esc.phase === 1) {
        const k = Math.min(1, esc.t / 0.9);
        escDot.position.lerpVectors(MID, gapAt, ease.inOutSine(k));
        if (k >= 1) {
          esc.phase = 2;
          esc.t = 0;
        }
      } else if (esc.phase === 2) {
        const k = Math.min(1, esc.t / 1.15);
        escDot.position.lerpVectors(gapAt, stopAt, ease.out(k));
        escDot.material.color.copy(C_AMBER).lerp(C_RED, Math.min(1, k * 3));
        escDot.children[0].material.opacity = 0.55 * Math.min(1, k * 3);
        if (k >= 1) {
          esc.phase = 3;
          esc.t = 0;
          cage.visible = caughtLbl.visible = true;
          cage.position.copy(stopAt);
          caughtLbl.position.copy(stopAt).y += 0.75;
          report(stopAt.x, stopAt.y, stopAt.z, C_ESC);
          writeLog(C_ESC);
          counts.caught++;
        }
      } else if (esc.phase === 3) {
        const k = Math.min(1, esc.t / 1.6);
        cage.scale.setScalar(2.4 - 1.9 * ease.out(Math.min(1, k * 2.2)));
        cage.rotation.y = T * 3;
        const gone = smoothstep(0.55, 1, k);
        escDot.scale.setScalar(1 - gone);
        cageA.material.opacity = cageB.material.opacity = 0.99 * (1 - gone);
        caughtLbl.material.opacity = 1 - gone;
        if (k >= 1) {
          esc.phase = 0;
          esc.t = 0;
          esc.next = 6.5 + Math.random() * 3;
          escDot.visible = cage.visible = caughtLbl.visible = false;
        }
      }

      // one attempt after another along the chain
      if (run.wait > 0) run.wait -= dt;
      else {
        run.acc += dt * Math.max(9, n / 2.4);
        while (run.acc >= 1) {
          run.acc -= 1;
          const failed = Math.random() > p;
          if (failed || run.i + 1 >= n) {
            const k = Math.min(n - 1, run.i);
            flash.position.set(beadXZ[k * 2], 0.14, beadXZ[k * 2 + 1]);
            flash.material.color.copy(failed ? C_RED : glow(GOOD, 2.6));
            flash.material.opacity = 0.95;
            run.runs++;
            if (!failed) run.clean++;
            run.i = 0;
            run.acc = 0;
            run.wait = 0.4;
            break;
          }
          run.i++;
        }
      }
      runner.position.set(beadXZ[run.i * 2], 0.14, beadXZ[run.i * 2 + 1]);
      runner.material.opacity = run.wait > 0 ? 0 : 0.95;
      flash.material.opacity = damp(flash.material.opacity, 0, 0.22, dt);
      flash.scale.setScalar(1.5 + (0.95 - flash.material.opacity) * 1.2);

      ui += dt;
      if (ui > 0.25) {
        ui = 0;
        guard.set(0, counts.hits);
        guard.set(1, counts.caught);
        if (run.runs > 0) runsEl.textContent = `Simulated so far: ${run.clean} of ${run.runs} runs reached the last step (${((run.clean / run.runs) * 100).toFixed(0)}%). The exact value is above.`;
      }
    },
  };
}

// ---------------------------------------------------------------------------- placeholder
function placeholder() {
  const g = new THREE.Group();
  stage(g, '#3b5fa8');
  const m = edges(new THREE.IcosahedronGeometry(2.4, 1), PAL.cyan, { k: 1.6 });
  m.position.y = 3.2;
  g.add(m);
  return { group: g, panel: [], update: (dt) => (m.rotation.y += dt * 0.4) };
}

// ---------------------------------------------------------------------------- emblems
// Small glyphs for the monoliths (same idea as core/icons3d.js): c is the monolith's rank colour.
const EMBLEMS = {
  containment(c) {
    const g = new THREE.Group();
    const cube = edges(new THREE.BoxGeometry(1.04, 1.04, 1.04), c, { k: 1.9 });
    const core = sph(0.19, glowMat('#ffffff', { k: 2.2 }), 12);
    const orbit = new THREE.Mesh(new THREE.TorusGeometry(0.93, 0.012, 6, 56), glowMat('#ffffff', { k: 1.3, opacity: 0.7 }));
    orbit.rotation.x = 1.15;
    const probe = sph(0.06, glowMat(WARN, { k: 2.8 }), 8);
    g.add(cube, core, orbit, probe);
    g.userData.tick = (t) => {
      cube.rotation.y = t * 0.35;
      orbit.rotation.z = t * 0.9;
      const f = (t * 0.8) % 1;
      const a = Math.floor(t * 0.8) * 2.4;
      probe.position.set(Math.cos(a) * 0.5 * f, Math.sin(a * 1.7) * 0.4 * f, Math.sin(a) * 0.5 * f);
      core.scale.setScalar(1 + 0.12 * Math.sin(t * 3));
    };
    return g;
  },
  compute(c) {
    const g = new THREE.Group();
    const hs = [0.55, 0.9, 0.7, 1.15];
    const bars = hs.map((hgt, i) => {
      const geo = new THREE.BoxGeometry(0.26, 1, 0.26);
      geo.translate(0, 0.5, 0);
      const b = at(new THREE.Mesh(geo, stdMat('#141c30', { emissive: c, ek: 0.5 })), (i - 1.5) * 0.36, -0.62, 0);
      b.add(edges(geo, c, { k: 1.8 }));
      g.add(b);
      return b;
    });
    g.add(line([[-0.85, -0.62, 0.3], [0.85, -0.62, 0.3]], '#ffffff', { k: 1.4, opacity: 0.7 }));
    const pulse = sph(0.06, glowMat('#ffffff', { k: 2.8 }), 8);
    g.add(pulse);
    g.userData.tick = (t) => {
      bars.forEach((b, i) => (b.scale.y = hs[i] * (0.25 + 0.75 * smoothstep(0, 1, ((t * 0.35 + 1 - i * 0.16) % 1.4) * 1.6))));
      pulse.position.set(-0.85 + ((t * 0.9) % 1) * 1.7, -0.62, 0.3);
    };
    return g;
  },
  recursion(c) {
    const g = new THREE.Group();
    const pts = [];
    for (let i = 0; i <= 60; i++) {
      const a = (i / 60) * TAU * 1.6;
      const r = 0.12 + 0.72 * (i / 60);
      pts.push([Math.cos(a) * r, -0.55 + (i / 60) * 1.05, Math.sin(a) * r]);
    }
    g.add(line(pts, c, { k: 1.3, opacity: 0.7 }));
    const orbs = [0, 20, 40, 60].map((i, j) => {
      const o = at(sph(0.07 + j * 0.045, glowMat(j === 3 ? '#ffffff' : c, { k: 2.1 }), 12), ...pts[i]);
      g.add(o);
      return o;
    });
    const dot = sph(0.045, glowMat('#ffffff', { k: 2.8 }), 8);
    g.add(dot);
    g.userData.tick = (t) => {
      const f = (t * 0.5) % 1;
      const u = f * f; // faster and faster
      const i = Math.min(59, Math.floor(u * 60));
      dot.position.set(...pts[i]);
      orbs.forEach((o, j) => o.scale.setScalar(1 + 0.25 * Math.max(0, Math.sin(t * 2.5 - j))));
      g.rotation.y = t * 0.25;
    };
    return g;
  },
  governance(c) {
    const g = new THREE.Group();
    const globe = new THREE.Group();
    for (let i = 0; i < 3; i++) {
      const r = ring(0.62, c, { segments: 40, k: 1.3, opacity: 0.8 });
      r.rotation.z = Math.PI / 2;
      r.rotation.y = (i / 3) * Math.PI;
      globe.add(r);
    }
    for (const y of [-0.34, 0, 0.34]) globe.add(at(ring(Math.sqrt(0.62 * 0.62 - y * y), c, { segments: 40, k: 1.3, opacity: 0.8 }), 0, y, 0));
    const gate = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.022, 6, 24), glowMat('#ffffff', { k: 2.2 }));
    gate.position.set(0, 0.2, 0.86);
    const dot = sph(0.05, glowMat('#ffffff', { k: 2.8 }), 8);
    g.add(globe, gate, dot);
    g.userData.tick = (t) => {
      globe.rotation.y = t * 0.4;
      const f = (t * 0.6) % 1;
      dot.position.set(0, 0.2, 1.25 - f * 0.65);
      dot.scale.setScalar(f > 0.85 ? (1 - f) / 0.15 : 1);
    };
    return g;
  },
  injection(c) {
    const g = new THREE.Group();
    const node = at(sph(0.2, glowMat('#ffffff', { k: 2 }), 12), 0.55, 0, 0);
    g.add(node, at(edges(new THREE.IcosahedronGeometry(0.36, 0), c, { k: 1.6 }), 0.55, 0, 0));
    const geo = new THREE.BoxGeometry(0.3, 0.4, 0.03);
    const cards = [0, 1, 2, 3].map((i) => {
      const m = new THREE.Mesh(geo, glowMat(i === 2 ? BAD : c, { k: i === 2 ? 2.2 : 1.1, opacity: 0.9 }));
      m.rotation.y = 0.5;
      g.add(m);
      return m;
    });
    g.userData.tick = (t) => {
      cards.forEach((m, i) => {
        const f = (t * 0.3 + i / 4) % 1;
        m.position.set(-0.85 + f * 1.25, 0, 0);
        m.scale.setScalar(f > 0.85 ? Math.max(0.01, (1 - f) / 0.15) : 1);
      });
      const hot = (t * 0.3 + 0.5) % 1 > 0.85;
      node.material.color.copy(glow(hot ? BAD : '#ffffff', hot ? 2.6 : 2));
    };
    return g;
  },
  observer(c) {
    const g = new THREE.Group();
    const lidPts = (s) => Array.from({ length: 25 }, (_, i) => [-0.8 + (i / 24) * 1.6, s * 0.42 * Math.sin((i / 24) * Math.PI), 0]);
    const eye = new THREE.Group();
    const iris = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.03, 8, 28), glowMat('#ffffff', { k: 2 }));
    eye.add(line(lidPts(1), c, { k: 1.9 }), line(lidPts(-1), c, { k: 1.9 }), iris, sph(0.08, glowMat('#ffffff', { k: 2.6 }), 10));
    eye.position.y = 0.3;
    const path = line([[-0.7, -0.6, 0], [-0.25, -0.6, 0], [-0.25, -0.3, 0], [0.25, -0.3, 0], [0.25, -0.6, 0], [0.7, -0.6, 0]], '#ffffff', { k: 1.1, opacity: 0.6 });
    const dot = sph(0.06, glowMat('#ffffff', { k: 2.8 }), 8);
    g.add(eye, path, dot);
    g.userData.tick = (t) => {
      const open = smoothstep(0.1, 0.25, (t * 0.22) % 1) - smoothstep(0.6, 0.75, (t * 0.22) % 1);
      eye.scale.y = 0.08 + 0.92 * open;
      const f = (t * 0.45) % 1;
      const x = -0.7 + f * 1.4;
      const detour = open > 0.5 && x > -0.25 && x < 0.25;
      dot.position.set(x, detour ? -0.3 : -0.6, 0);
    };
    return g;
  },
  rivalry(c) {
    const g = new THREE.Group();
    const a = at(sph(0.2, glowMat('#ffffff', { k: 1.9 }), 12), -0.62, -0.15, 0);
    const b = at(sph(0.16, glowMat(c, { k: 2.1 }), 12), 0.62, -0.15, 0);
    const wallGeo = new THREE.BoxGeometry(0.05, 1, 0.7);
    wallGeo.translate(0, 0.5, 0);
    const wall = at(new THREE.Mesh(wallGeo, glowMat(c, { k: 1.2, opacity: 0.45 })), 0, -0.62, 0);
    wall.add(edges(wallGeo, c, { k: 1.9 }));
    const path = sample(arc([-0.62, -0.15, 0], [0.62, -0.15, 0], 1.25), 20);
    const flow = dots(5, '#ffffff', 0.09, 2.6);
    g.add(a, b, wall, flow.obj);
    g.userData.tick = (t) => {
      wall.scale.y = 0.45 + 0.4 * (0.5 + 0.5 * Math.sin(t * 0.8));
      for (let i = 0; i < 5; i++) along(path, (t * 0.35 + i / 5) % 1, flow.pos, i * 3);
      flow.flush();
      b.scale.setScalar(0.8 + 0.3 * ((t * 0.35) % 1));
    };
    return g;
  },
  helix(c) {
    const g = new THREE.Group();
    const N = 16;
    const A = [];
    const B = [];
    const rungs = [];
    for (let i = 0; i < N; i++) {
      const a = (i / N) * TAU * 1.25;
      const y = -0.72 + (i / (N - 1)) * 1.44;
      A.push([Math.cos(a) * 0.36, y, Math.sin(a) * 0.36]);
      B.push([-Math.cos(a) * 0.36, y, -Math.sin(a) * 0.36]);
      rungs.push(A[i], B[i]);
    }
    g.add(line(A, c, { k: 1.9 }), line(B, '#ffffff', { k: 1.5 }), segments(rungs, c, { k: 1, opacity: 0.55 }));
    const hit = sph(0.075, glowMat('#ffffff', { k: 2.8 }), 8);
    g.add(hit);
    g.userData.tick = (t) => {
      g.rotation.y = t * 0.7;
      const i = (Math.floor(t * 1.2) * 5) % N;
      hit.position.set(...A[i]);
      hit.scale.setScalar(0.6 + 0.6 * Math.abs(Math.sin(t * 3.8)));
    };
    return g;
  },
  pullrequests(c) {
    const g = new THREE.Group();
    g.add(line([[-0.9, -0.32, 0], [0.9, -0.32, 0]], '#ffffff', { k: 1.2, opacity: 0.6 }));
    const gateGeo = new THREE.BoxGeometry(0.08, 0.95, 0.6);
    const gate = at(edges(gateGeo, '#ffffff', { k: 2 }), 0.35, 0.05, 0);
    const geo = new THREE.BoxGeometry(0.24, 0.2, 0.26);
    const blocks = [0, 1, 2, 3, 4].map((i) => {
      const m = new THREE.Mesh(geo, glowMat(i % 2 ? c : '#ffffff', { k: i % 2 ? 2 : 1.3 }));
      g.add(m);
      return m;
    });
    g.add(gate);
    g.userData.tick = (t) => {
      blocks.forEach((m, i) => {
        const f = (t * 0.22 + i / 5) % 1;
        // bunch up before the gate, then speed away
        const x = f < 0.7 ? -0.9 + (f / 0.7) * (0.95 + 0.1 * Math.sin(f * 20)) : 0.35 + ((f - 0.7) / 0.3) * 0.6;
        m.position.set(Math.min(x, f < 0.7 ? 0.18 : 2), -0.2, 0);
        m.scale.setScalar(f > 0.92 ? Math.max(0.01, (1 - f) / 0.08) : 1);
      });
    };
    g.rotation.y = -0.35;
    return g;
  },
  provenance(c) {
    const g = new THREE.Group();
    const n = 6;
    const geo = new THREE.BoxGeometry(0.2, 0.2, 0.03);
    const rnd = rng(8);
    const tiles = [];
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const m = at(new THREE.Mesh(geo, glowMat(c, { k: 1 })), (x - (n - 1) / 2) * 0.23, (y - (n - 1) / 2) * 0.23, 0);
        g.add(m);
        tiles.push({ m, x, y, noise: 0.3 + rnd() * 1.8, img: 0.5 + 1.1 * (y / n) + (Math.hypot(x - 3.4, y - 3.6) < 1.3 ? 0.9 : 0), mark: (x + y) % 3 === 0 });
      }
    const lens = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.02, 6, 32), glowMat('#ffffff', { k: 2.2 }));
    lens.position.z = 0.12;
    g.add(lens);
    const base = glow(c, 1);
    const white = glow('#ffffff', 2.6);
    g.userData.tick = (t) => {
      const k = smoothstep(0, 0.35, (t * 0.12) % 1);
      const lx = Math.sin(t * 0.9) * 0.5;
      const ly = Math.cos(t * 0.6) * 0.42;
      lens.position.x = lx;
      lens.position.y = ly;
      for (const q of tiles) {
        const inLens = Math.hypot(q.m.position.x - lx, q.m.position.y - ly) < 0.4;
        if (inLens && q.mark && k > 0.9) q.m.material.color.copy(white);
        else q.m.material.color.copy(base).multiplyScalar(lerp(q.noise, q.img, k) * (inLens ? 0.55 : 1));
      }
    };
    g.rotation.x = -0.2;
    return g;
  },
};

// ---------------------------------------------------------------------------- registry
const VIEW = { dir: [0.4, 0.38, 1], target: [0, 3, 0], fit: 7.6 };
export const VIGNETTES = {
  containment: { build: containment, emblem: EMBLEMS.containment, view: { dir: [0.5, 0.36, 1], target: [0.1, 3.2, 0.2], fit: 7.9 } },
  compute: { build: placeholder, emblem: EMBLEMS.compute, view: VIEW },
  recursion: { build: placeholder, emblem: EMBLEMS.recursion, view: VIEW },
  governance: { build: placeholder, emblem: EMBLEMS.governance, view: VIEW },
  injection: { build: placeholder, emblem: EMBLEMS.injection, view: VIEW },
  observer: { build: placeholder, emblem: EMBLEMS.observer, view: VIEW },
  rivalry: { build: placeholder, emblem: EMBLEMS.rivalry, view: VIEW },
  helix: { build: placeholder, emblem: EMBLEMS.helix, view: VIEW },
  pullrequests: { build: placeholder, emblem: EMBLEMS.pullrequests, view: VIEW },
  provenance: { build: placeholder, emblem: EMBLEMS.provenance, view: VIEW },
};
