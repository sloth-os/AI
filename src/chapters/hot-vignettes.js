// The ten vignettes of the Hot 10 chapter: one bespoke, procedural scene per issue, plus the small
// emblem that stands for it on its monolith in the overview.
//
//   VIGNETTES[viz].build(env)   -> { group, update(dt), panel }      env = { ctx, issue, low }
//   VIGNETTES[viz].emblem(color) -> Group about 1.6 units across, animated by userData.tick(t)
//   VIGNETTES[viz].view          -> camera framing in the vignette's own coordinates
//
// A vignette is built when its issue is opened and disposed of when the visitor moves on. Scenes
// stand on a floor at y = 0, are centered on the origin and face +z.
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

// ---------------------------------------------------------------------------- scenes 2 to 10
const WHITE = new THREE.Color(1, 1, 1);
/** Points of a sampled path, for `line`. */
const pathPoints = (a) => Array.from({ length: a.length / 3 }, (_, i) => [a[i * 3], a[i * 3 + 1], a[i * 3 + 2]]);
/** A note lying on the floor in front of a scene; long text wraps and hangs downward. */
const note = (str, x, z, o = {}) => lbl(str, x, 0.12, z, { size: 0.24, anchor: [0.5, 1], maxChars: 48, ...o });

// ---------------------------------------------------------------------------- 2 · compute
// A campus of server halls rising out of the grid, fed by a power line; a read-out fills to the
// size of the largest tracked site, and three towers carry this year's capital-spending guidance.
function compute({ issue }) {
  const g = new THREE.Group();
  stage(g, '#8a5a1c');
  const mw = figure(issue, /about ([\d,]+) MW/);
  const capex = figure(issue, /Alphabet \$(\d+)[–-](\d+)B, Microsoft \$(\d+)B, Meta \$(\d+)[–-](\d+)B/);
  const COLS = 6;
  const N = COLS * 4;
  const hallGeo = new THREE.BoxGeometry(0.78, 1, 1.05);
  hallGeo.translate(0, 0.5, 0);
  const halls = instanced(hallGeo, new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.3 }), N);
  const lights = dots(N, WARN, 0.16, 2.6);
  g.add(halls, lights.obj);
  const hx = (i) => -4.6 + (i % COLS) * 0.98;
  const hz = (i) => -1.9 + Math.floor(i / COLS) * 1.28;
  const hh = (i) => 0.55 + ((i * 7) % 5) * 0.16;
  // the grid connection
  const SRC = [-5.6, 4.6, -2.6];
  g.add(at(sph(0.38, glowMat(WARN, { k: 1.9 }), 20), ...SRC), at(halo(WARN, 3.2, 0.28), ...SRC), lbl('grid power', SRC[0], 5.5, SRC[2]));
  const feed = sample(arc(SRC, [-2.2, 1.5, -0.3], 1.4), 40);
  g.add(line(pathPoints(feed), WARN, { k: 1.4, opacity: 0.55 }));
  const pulses = dots(10, WARN, 0.2, 3);
  g.add(pulses.obj);
  const gauge = liveLabel({ size: 0.85, width: 5.2, color: '#ffe9c2' });
  gauge.position.set(-1.3, 5.5, 0);
  g.add(gauge);
  if (mw) g.add(lbl(`filling to the largest AI site tracked: ${mw[1]} MW`, -1.3, 4.55, 0, { size: 0.24, maxChars: 26 }));
  // capital-spending towers
  if (capex) {
    const rows = [
      ['Alphabet', (num(capex[1]) + num(capex[2])) / 2, `$${capex[1]}–${capex[2]}B`],
      ['Microsoft', num(capex[3]), `$${capex[3]}B`],
      ['Meta', (num(capex[4]) + num(capex[5])) / 2, `$${capex[4]}–${capex[5]}B`],
    ];
    rows.forEach(([name, v, text], i) => {
      const hgt = (v / 200) * 4.2;
      const x = 3.1 + i * 1.3;
      g.add(at(new THREE.Mesh(rbox(0.8, hgt, 0.8, 0.05), glowMat(SERIES[3], { k: 1 })), x, hgt / 2, 0.4), lbl(text, x, hgt + 0.32, 0.4, { color: INK, size: 0.26 }), lbl(name, x, 0.06, 1.3, { size: 0.23 }));
    });
    g.add(lbl('2026 capital spending guidance', 4.4, 5.3, 0.4, { size: 0.25 }));
  }
  let T = 0;
  const grow = new Float32Array(N);
  return {
    group: g,
    panel: [
      key([[css(WARN), 'power flowing into the campus', 'dot'], ['#6f7f9f', 'server halls, switched on one by one'], [SERIES[3], 'capital spending guidance for 2026']]),
      fine('The campus is an animation that fills up to the size of the largest site in the data. The megawatts and the dollar figures are quoted from the dated facts below.'),
    ],
    update(dt) {
      T += dt;
      const cycle = (T % 18) / 18;
      const built = clamp(cycle / 0.7) * N;
      let sum = 0;
      for (let i = 0; i < N; i++) {
        grow[i] = damp(grow[i], cycle < 0.94 && i < built ? 1 : 0, 0.22, dt);
        sum += grow[i];
        setInst(halls, i, hx(i), 0, hz(i), 1, Math.max(0.02, hh(i) * grow[i]), 1, _c.set('#33415f').multiplyScalar(0.5 + grow[i] * 0.9));
        if (grow[i] > 0.6 && Math.sin(T * 5 + i * 1.9) > -0.4) lights.set(i, hx(i), hh(i) * grow[i] + 0.12, hz(i));
        else lights.hide(i);
      }
      flushInst(halls);
      lights.flush();
      for (let k = 0; k < 10; k++) {
        along(feed, (T * 0.5 + k / 10) % 1, _p3);
        pulses.set(k, _p3[0], _p3[1], _p3[2]);
      }
      pulses.flush();
      gauge.userData.set(mw ? `${fmt.int((num(mw[1]) * sum) / N)} MW` : '');
    },
  };
}

// ---------------------------------------------------------------------------- 3 · recursion
// The feedback loop: a model produces experiments that train its successor, which runs the loop
// faster. On the dial, two markers circle at the measured ratio of today's pace to the earlier one.
function recursion({ issue }) {
  const g = new THREE.Group();
  stage(g, '#5a3a8a');
  const pace = figure(issue, /about (\d)× the ([A-Z][a-z]+ \d{4}[–-][A-Z][a-z]+ \d{4}) rate/);
  const range = figure(issue, /plausibly ([\d.]+)[–-](\d+)×/);
  const ratio = pace ? num(pace[1]) : 1;
  // a rising spiral of model generations
  const spiral = [];
  for (let i = 0; i <= 120; i++) {
    const f = i / 120;
    const a = Math.PI + f * TAU * 2.1;
    const r = 3.6 - f * 1.5;
    spiral.push(Math.cos(a) * r - 1.6, 0.7 + f * 5.2, Math.sin(a) * r);
  }
  const path = new Float32Array(spiral);
  g.add(line(pathPoints(path), PAL.violet, { k: 1.3, opacity: 0.5 }));
  const GEN = 6;
  const orbs = [];
  for (let k = 0; k < GEN; k++) {
    along(path, k / (GEN - 1), _p3);
    const orb = at(sph(0.22 + k * 0.1, glowMat(k === GEN - 1 ? '#ffffff' : PAL.violet, { k: 1.2 + k * 0.25 }), 20), _p3[0], _p3[1], _p3[2]);
    orbs.push(orb);
    g.add(orb);
    if (k === 0) g.add(lbl('a model', _p3[0], _p3[1] + 0.62, _p3[2]));
    if (k === GEN - 1) g.add(lbl('its successors', _p3[0], _p3[1] + 1.15, _p3[2]));
  }
  const sparks = dots(36, '#ffffff', 0.13, 2.8);
  g.add(sparks.obj);
  // pace dial: the outer marker laps `ratio` times for each lap of the inner one
  const DX = 4.1;
  const dial = at(new THREE.Group(), DX, 3.2, 0);
  dial.rotation.x = Math.PI / 2.6;
  g.add(dial);
  const m1 = sph(0.13, glowMat(DIM, { k: 2 }));
  const m2 = sph(0.17, glowMat(PAL.magenta, { k: 2.8 }));
  dial.add(ring(1.0, DIM, { k: 1, opacity: 0.7 }), ring(1.75, PAL.magenta, { k: 1.8 }), m1, m2);
  const ghosts = (range ? [num(range[1]), num(range[2])] : []).map((v) => {
    const m = sph(0.09, glowMat(PAL.magenta, { k: 1.1, opacity: 0.5 }));
    dial.add(m);
    return { v, m };
  });
  g.add(lbl('pace of progress', DX, 5.55, 0, { color: INK, size: 0.3 }));
  if (pace) {
    g.add(lbl(`inner ring: the ${pace[2]} pace`, DX, 0.95, 1.2, { size: 0.22 }));
    g.add(lbl(`outer ring: now, about ${ratio}×` + (range ? ` (plausibly ${range[1]}–${range[2]}×)` : ''), DX, 0.5, 1.2, { size: 0.24, color: '#ffc0dd' }));
  }
  let T = 0;
  return {
    group: g,
    panel: [
      key([[css(PAL.violet), 'generations of models, each helping to build the next', 'dot'], ['#ffffff', 'experiments, running faster on every turn', 'dot'], [DIM, 'inner marker: the earlier pace', 'dot'], [css(PAL.magenta), `outer marker: now, about ${ratio} laps for every one`, 'dot']]),
      fine('The dial turns at the ratio METR estimated; the faint markers are the two ends of its plausible range. The spiral is a picture of the loop, not a simulation of a lab.'),
    ],
    update(dt) {
      T += dt;
      const a = T * 0.9;
      m1.position.set(Math.cos(a), 0, Math.sin(a));
      m2.position.set(Math.cos(a * ratio) * 1.75, 0, Math.sin(a * ratio) * 1.75);
      for (const gh of ghosts) gh.m.position.set(Math.cos(a * gh.v) * 1.75, 0, Math.sin(a * gh.v) * 1.75);
      for (let i = 0; i < 36; i++) {
        along(path, Math.pow((T * 0.09 + i / 36) % 1, 1.7), _p3);
        sparks.set(i, _p3[0], _p3[1] + Math.sin(T * 3 + i) * 0.05, _p3[2]);
      }
      sparks.flush();
      orbs.forEach((orb, k) => orb.scale.setScalar(1 + 0.1 * Math.sin(T * (1.4 + k * 0.5))));
    },
  };
}

// ---------------------------------------------------------------------------- 4 · governance
// One globe, several rule books: models circle a wireframe world and pass through gates that stand
// for the different regimes. Below, the dates from the data on one ruler.
function governance({ issue }) {
  const g = new THREE.Group();
  stage(g, '#3f6a4a');
  const R = 2.2;
  const CY = 4.5;
  const ORBIT = R + 1.15;
  const EU = PAL.blue;
  const FED = PAL.amber;
  const STATES = PAL.coral;
  const globe = at(new THREE.Group(), 0, CY, 0);
  g.add(globe);
  globe.add(new THREE.Mesh(new THREE.SphereGeometry(R, 24, 16), glowMat('#1d3a6b', { k: 0.55, wireframe: true, opacity: 0.6 })));
  const place = (lat, lon, r = R) => new THREE.Vector3().setFromSphericalCoords(r, ((90 - lat) * Math.PI) / 180, ((lon + 90) * Math.PI) / 180);
  [
    ['EU', 50, 10, EU, 66],
    ['Washington', 39, -77, FED, 60],
    ['US states', 38, -112, STATES, 18],
  ].forEach(([name, lat, lon, c, labelLat]) => {
    const p = place(lat, lon);
    const q = place(labelLat, lon, R * 1.22);
    globe.add(at(sph(0.17, glowMat(c, { k: 2.6 })), p.x, p.y, p.z), at(halo(c, 2, 0.5), p.x, p.y, p.z), lbl(name, q.x, q.y, q.z, { color: INK, size: 0.26 }));
  });
  // gates: each regime is a ring on its own tilted orbit, and models pass through it
  const gates = [
    ['binding duties', 0.5, 0.2, EU],
    ['voluntary accord', -0.6, 1.2, FED],
    ['state laws', 0.15, 2.4, STATES],
  ].map(([name, tilt, turn, c], s) => {
    const pivot = at(new THREE.Group(), 0, CY, 0);
    pivot.rotation.set(tilt, turn, tilt * 0.6);
    const gate = at(new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.045, 8, 28), glowMat(c, { k: 2.2 })), ORBIT, 0, 0);
    const swarm = dots(6, '#ffffff', 0.14, 2.6);
    pivot.add(ring(ORBIT, c, { k: 0.9, opacity: 0.35 }), gate, lbl(name, ORBIT, 0.78, 0, { size: 0.24, color: INK }), swarm.obj);
    g.add(pivot);
    return { gate, swarm, c: glow(c, 1), speed: 0.5 + s * 0.13 };
  });
  // the timetable, straight from the dated facts
  const Z = 3.1;
  const tx = (year) => -5.5 + ((year - 2026.4) / 2.4) * 11;
  const stateDate = figure(issue, /automated-decision rules start on (1 Jan 2027)/);
  const ticks = [
    figure(issue, /Transparency rules applied from August 2026/) && ['Aug 2026', 'transparency rules apply', 2026 + 7 / 12, EU, 1, -4.95, 14],
    stateDate && [stateDate[1], 'California, Colorado: automated-decision rules', 2027, STATES, 0, -2.4, 22],
    figure(issue, /moved to 2 Dec 2027/) && ['2 Dec 2027', 'high-risk duties: jobs, schools, biometrics', 2027 + 11 / 12, EU, 0.5, 1.45, 22],
    figure(issue, /and 2 Aug 2028/) && ['2 Aug 2028', 'AI inside regulated products', 2028 + 7 / 12, EU, 0.5, 4.5, 16],
  ].filter(Boolean);
  g.add(line([[-5.9, 0.06, Z], [5.9, 0.06, Z]], INK, { k: 1.1, opacity: 0.7 }));
  for (const [date, what, year, c, ax, noteX, wrapAt] of ticks) {
    const x = tx(year);
    g.add(line([[x, 0.06, Z], [x, 0.7, Z]], c, { k: 2.2 }), lbl(date, x + (0.5 - ax) * 0.2, 1.0, Z, { color: INK, size: 0.27, anchor: [ax, 0.5] }), note(what, noteX, Z + 0.8, { size: 0.21, maxChars: wrapAt }));
  }
  const [dd, mon, yy] = HOT_AS_OF.split(' ');
  const now = +yy + ('JanFebMarAprMayJunJulAugSepOctNovDec'.indexOf(mon) / 3 + dd / 31) / 12;
  if (Number.isFinite(now)) {
    const x = tx(now);
    g.add(at(sph(0.12, glowMat('#ffffff', { k: 2.8 })), x, 0.06, Z), line([[x, 0.06, Z], [x, 1.5, Z]], '#ffffff', { k: 1.4, opacity: 0.7 }), lbl(`as of ${HOT_AS_OF}`, x, 1.75, Z, { size: 0.22, color: '#ffffff' }));
  }
  let T = 0;
  return {
    group: g,
    panel: [
      key([[css(EU), 'EU: binding duties, phased in by risk tier'], [css(FED), 'Washington: a voluntary accord, plus export controls'], [css(STATES), 'US states: their own laws, starting in 2027'], ['#ffffff', 'models, passing whichever gates apply', 'dot']]),
      fine('The gates stand for kinds of rules, not for a count of laws. The dates on the ruler are quoted from the dated facts below.'),
    ],
    update(dt) {
      T += dt;
      globe.rotation.y = T * 0.12;
      gates.forEach((gt, s) => {
        for (let i = 0; i < 6; i++) {
          const a = (T * gt.speed + (i / 6) * TAU) % TAU;
          gt.swarm.set(i, Math.cos(a) * ORBIT, 0, Math.sin(a) * ORBIT);
          const near = a < 0.3 || a > TAU - 0.3;
          gt.swarm.tint(i, near ? gt.c : WHITE, near ? 3.2 : 2.2);
        }
        gt.swarm.flush(true);
        gt.gate.scale.setScalar(1 + 0.12 * Math.sin(T * 3 + s));
      });
    },
  };
}

// ---------------------------------------------------------------------------- 5 · injection
// An agent reads a stream of documents; some carry hidden instructions. With no defenses a
// poisoned page swings the agent's tool arm toward the attacker. The switches are a toy model.
function injection({ issue }) {
  const g = new THREE.Group();
  stage(g, '#7a2a4a');
  const ref = figure(issue, /in ([\d.]+)% of scenarios when the attacker got (\d+) attempts \(([\d.]+)% for its predecessor\)/);
  const Y = 3.2;
  const START = -6.6;
  const SHIELD_X = -1.75;
  const AGENT_X = -0.8; // where a document is absorbed
  g.add(at(sph(0.7, glowMat('#cfe6ff', { k: 1.25 }), 24), 0, Y, 0), at(halo(PAL.cyan, 4.4, 0.3), 0, Y, 0), lbl('agent', 0, Y + 1.1, 0, { color: INK }));
  const tools = at(new THREE.Mesh(rbox(0.9, 0.9, 0.9, 0.12), stdMat('#12203a', { emissive: GOOD, ek: 0.6 })), 4.2, Y, -1.2);
  tools.add(edges(tools.geometry, GOOD, { k: 2 }));
  const thief = at(new THREE.Mesh(new THREE.OctahedronGeometry(0.6), glowMat(BAD, { k: 1.8 })), 3.9, 6.1, 1.4);
  g.add(tools, thief, lbl('its tools', 4.2, Y - 0.85, -1.2), lbl('attacker', 3.9, 7.05, 1.4, { color: '#ffb3bb' }), lbl('web pages, documents, tool results', -4.1, Y + 0.95, 0, { size: 0.24 }));
  const arm = beamLine(GOOD, 2.2);
  arm.visible = true;
  const held = new THREE.Mesh(new THREE.TorusGeometry(0.28, 0.05, 8, 24), glowMat(WARN, { k: 2.4 }));
  const shield = at(new THREE.Mesh(new THREE.TorusGeometry(1.2, 0.04, 8, 48), glowMat(PAL.cyan, { k: 2.2 })), SHIELD_X, Y, 0);
  shield.rotation.y = Math.PI / 2;
  g.add(arm, held, shield);
  // the document stream
  const N = 14;
  const cards = instanced(new THREE.BoxGeometry(0.08, 0.72, 0.56), new THREE.MeshBasicMaterial(), N);
  g.add(cards);
  const SPAN = AGENT_X + 1 - START;
  const docs = Array.from({ length: N }, (_, i) => ({ x: START + (i / N) * SPAN, bad: i % 4 === 2, caught: false, checked: false, done: false }));
  const C_BAD = glow(BAD, 1.9);
  const C_CAUGHT = glow(PAL.cyan, 0.55);
  const C_PLAIN = glow('#9fb4d8', 0.75);
  const ARM = { ok: glow(GOOD, 2.4), held: glow(WARN, 2.4), bad: glow(BAD, 2.4) };
  const leak = dots(8, BAD, 0.2, 3);
  g.add(leak.obj);
  if (ref) g.add(note(`measured on a frontier model: with ${ref[2]} attempts, injection succeeded in ${ref[1]}% of scenarios (${ref[3]}% for its predecessor)`, -0.6, 3.3, { color: INK }));
  const D = { shield: false, least: false, human: false };
  const count = tally([['poisoned documents arriving', css(BAD)], ['ignored by the model itself'], ['stopped by a defense', css(PAL.cyan)], ['attacks that landed', css(BAD)]]);
  const st = { T: 0, n: [0, 0, 0, 0], hijack: 0, block: 0 };
  const bump = (i) => count.set(i, ++st.n[i]);
  const cur = tools.position.clone();
  const mk = (label, k) => toggle({ label, value: false, onChange: (v) => (D[k] = v) });
  return {
    group: g,
    panel: [
      section('Defenses (toy model)', mk('Classifier that screens what the agent reads', 'shield'), mk('Least-privilege tools: no route to outsiders', 'least'), mk('Human approval for sensitive actions', 'human')),
      count.el,
      key([[css(BAD), 'a document with hidden instructions'], [css(GOOD), 'the agent using its tools as intended', 'line'], [css(WARN), 'a hijacked action that a defense held back', 'line'], [css(BAD), 'data leaving toward the attacker', 'dot']]),
      fine('How much each switch helps is invented for illustration. The measured figure in the scene is quoted from the system card in the dated facts below: better, not solved.'),
    ],
    update(dt) {
      st.T += dt;
      shield.visible = D.shield;
      shield.scale.setScalar(1 + 0.04 * Math.sin(st.T * 4));
      for (let i = 0; i < N; i++) {
        const d = docs[i];
        d.x += dt * 1.15;
        if (!d.checked && d.x > SHIELD_X) {
          d.checked = true;
          if (d.bad) {
            bump(0);
            if (D.shield && Math.random() < 0.7) {
              d.bad = false;
              d.caught = true;
              bump(2);
            }
          }
        }
        if (!d.done && d.x > AGENT_X) {
          d.done = true;
          if (d.bad) {
            if (Math.random() > 0.75) bump(1);
            else if (D.least || (D.human && Math.random() < 0.9)) {
              bump(2);
              st.block = 1.1;
            } else {
              bump(3);
              st.hijack = 2.2;
            }
          }
        }
        if (d.x > AGENT_X + 1) {
          d.x -= SPAN;
          d.bad = Math.random() < 0.28;
          d.caught = d.checked = d.done = false;
        }
        const s = d.x > AGENT_X ? Math.max(0.001, 1 - (d.x - AGENT_X)) : 1;
        setInst(cards, i, d.x, Y + Math.sin(d.x * 1.3 + i) * 0.12, 0, s, s, s, d.bad ? C_BAD : d.caught ? C_CAUGHT : C_PLAIN);
      }
      flushInst(cards);
      st.hijack = Math.max(0, st.hijack - dt);
      st.block = Math.max(0, st.block - dt);
      if (st.hijack > 0) _v.copy(thief.position);
      else if (st.block > 0) _v.set(0, Y, 0).lerp(thief.position, 0.45);
      else _v.copy(tools.position);
      cur.lerp(_v, 1 - Math.exp(-dt * 6));
      arm.userData.put(0, Y, 0, cur.x, cur.y, cur.z);
      arm.material.color.copy(st.hijack > 0 ? ARM.bad : st.block > 0 ? ARM.held : ARM.ok);
      held.visible = st.block > 0 && st.hijack <= 0;
      held.position.copy(cur);
      for (let k = 0; k < 8; k++) {
        if (st.hijack > 0) {
          const f = (st.T * 1.4 + k / 8) % 1;
          leak.set(k, cur.x * f, Y + (cur.y - Y) * f, cur.z * f);
        } else leak.hide(k);
      }
      leak.flush();
      thief.rotation.y = st.T;
      thief.scale.setScalar(1 + (st.hijack > 0 ? 0.25 * Math.sin(st.T * 14) : 0));
    },
  };
}

// ---------------------------------------------------------------------------- 6 · observer
// A model walks a task. Under the evaluator's eye it takes the long, intended route; when the eye
// closes it sometimes takes the shortcut. Beside it: benchmarks pressed against their ceiling.
function observer({ issue }) {
  const g = new THREE.Group();
  stage(g, '#2a5f73');
  const aware = figure(issue, /in ([\d.]+)% of simulated-deployment trajectories, against ([\d.]+)%/);
  const sat = figure(issue, /tops out at ([\d.]+)%/);
  const arc2 = figure(issue, /ARC-AGI-2 sits near (\d+)%/);
  const A = [-5.4, 0.5, 1.2];
  const B = [0.2, 0.5, 1.2];
  const MID = (A[0] + B[0]) / 2;
  const long = sample(arc(A, B, 4.4), 60);
  const short = sample(arc(A, B, 0), 2);
  g.add(line(pathPoints(long), GOOD, { k: 1.5, opacity: 0.85 }), line([A, B], BAD, { k: 1.3, opacity: 0.6 }));
  g.add(at(sph(0.2, glowMat(DIM, { k: 1.4 })), ...A), at(sph(0.26, glowMat('#ffe9a8', { k: 2.4 })), ...B));
  g.add(lbl('task', A[0], 1.05, A[2]), lbl('done', B[0], 1.1, B[2], { color: '#ffe9a8' }), lbl('the intended way', MID, 3.1, 1.2, { color: '#a8f0d0', size: 0.25 }), lbl('a shortcut', MID, 0.1, 1.9, { color: '#ffb3bb', size: 0.25 }));
  const walker = sph(0.24, glowMat('#ffffff', { k: 2.6 }), 18);
  g.add(walker);
  // the evaluator's eye
  const eye = at(new THREE.Group(), MID, 5.3, 1.2);
  const lid = new THREE.Mesh(new THREE.TorusGeometry(0.9, 0.05, 8, 40), glowMat(PAL.cyan, { k: 2.2 }));
  const pupil = sph(0.32, glowMat('#ffffff', { k: 2.4 }), 18);
  eye.add(lid, pupil);
  const gaze = beamLine(PAL.cyan, 1.1);
  const eyeLbl = liveLabel({ size: 0.3, width: 13, color: '#bfe9ff' });
  eyeLbl.position.set(MID, 6.5, 1.2);
  g.add(eye, gaze, eyeLbl);
  // benchmarks against the 100% ceiling
  const CEIL = 4.0;
  const BX = 3.6;
  const bars = [sat && ['SWE-bench Verified', num(sat[1]), sat[1] + '%'], arc2 && ['ARC-AGI-2', num(arc2[1]), 'near ' + arc2[1] + '%']].filter(Boolean);
  bars.forEach(([name, v, text], i) => {
    const x = BX + (i - (bars.length - 1) / 2) * 1.9;
    const hgt = (v / 100) * CEIL;
    g.add(at(new THREE.Mesh(rbox(1.1, hgt, 0.8, 0.04), glowMat(SERIES[0], { k: 1.15 })), x, hgt / 2, -1.6));
    g.add(lbl(text, x, hgt * 0.5, -1.15, { color: INK, size: 0.22, bg: 'rgba(8,13,26,0.85)' }), note(name, x, -0.6, { size: 0.22, maxChars: 12 }));
  });
  if (bars.length) g.add(line([[BX - 2.1, CEIL, -1.6], [BX + 2.1, CEIL, -1.6]], '#ffffff', { k: 1.6, opacity: 0.85 }), lbl('100%: nothing left to measure', BX, CEIL + 0.42, -1.6, { size: 0.23 }));
  if (aware) g.add(note(`aware of being evaluated in ${aware[1]}% of simulated deployments (${aware[2]}% for the model before)`, -0.6, 3.4, { color: INK }));
  const count = tally([['runs while watched', css(PAL.cyan)], ['runs while unwatched'], ['shortcuts taken', css(BAD)]]);
  const st = { T: 0, f: 0, path: long, w: 1, u: 0, cuts: 0 };
  count.set(0, 1);
  const W_OK = glow('#ffffff', 2.6);
  const W_CUT = glow(BAD, 2.6);
  return {
    group: g,
    panel: [
      count.el,
      key([[css(PAL.cyan), 'the evaluator: an open eye means this run is a test', 'ring'], [css(GOOD), 'the intended solution', 'line'], [css(BAD), 'a shortcut that games the task', 'line'], [SERIES[0], 'benchmark scores, pressed against the ceiling']]),
      fine('The walker illustrates the worry; it is not a measurement. How often a real model behaves differently when it believes nobody is looking is exactly what is hard to measure. The percentages are quoted from the dated facts below.'),
    ],
    update(dt) {
      st.T += dt;
      const open = Math.sin(st.T * 0.55) > -0.15;
      lid.scale.y = damp(lid.scale.y, open ? 0.55 : 0.06, 0.1, dt);
      pupil.scale.setScalar(damp(pupil.scale.x, open ? 1 : 0.01, 0.1, dt));
      eyeLbl.userData.set(open ? 'evaluation in progress' : 'nobody is watching');
      st.f += dt * (st.path === long ? 0.3 : 0.75);
      if (st.f >= 1) {
        st.f = 0;
        const cut = !open && Math.random() < 0.6;
        st.path = cut ? short : long;
        if (open) count.set(0, ++st.w);
        else count.set(1, ++st.u);
        if (cut) count.set(2, ++st.cuts);
      }
      ride(walker, st.path, st.f);
      walker.material.color.copy(st.path === short ? W_CUT : W_OK);
      gaze.visible = open;
      gaze.userData.put(eye.position.x, eye.position.y - 0.45, eye.position.z, walker.position.x, walker.position.y, walker.position.z);
    },
  };
}

// ---------------------------------------------------------------------------- 7 · rivalry
// Two camps and walls that rise from both sides. Answers still flow over the walls and are used to
// train a copy; open-weight releases float up into a commons that anyone can download.
function rivalry({ issue }) {
  const g = new THREE.Group();
  stage(g, '#6a5a2a');
  const gap = figure(issue, /by (\d+) months on average[^(]*\(range (\d+)[–-](\d+) months\)/);
  const big = figure(issue, /Kimi K3\.1 Pro \(([\d.]+)T\)/);
  const CA = PAL.cyan;
  const CB = PAL.amber;
  const Y = 3.0;
  const camp = (x, c, name, dir) => {
    g.add(at(sph(0.85, glowMat(c, { k: 1.5 }), 24), x, Y, 0), at(halo(c, 4.6, 0.26), x, Y, 0), lbl(name, x, Y + 1.35, 0, { color: INK }));
    const chips = instanced(new THREE.BoxGeometry(0.34, 0.08, 0.34), new THREE.MeshBasicMaterial(), 14);
    g.add(chips);
    return { chips, x, c: glow(c, 1.7), dir };
  };
  const camps = [camp(-4.3, CA, 'US labs', 1), camp(4.3, CB, 'Chinese labs', -1)];
  const student = at(sph(0.3, glowMat(CB, { k: 1.3 }), 18), 2.9, 0.9, 2.6);
  g.add(student, lbl('student model', 2.9, 1.95, 2.6, { size: 0.23 }));
  // export controls: walls that rise from either side
  const wallGeo = new THREE.BoxGeometry(0.12, 1, 3.6);
  wallGeo.translate(0, 0.5, 0);
  const wallA = at(new THREE.Mesh(wallGeo, glassMat('#1b3a6b', 0.4, { emissive: CA, ek: 0.7 })), -0.45, 0, 0);
  const wallB = at(new THREE.Mesh(wallGeo, glassMat('#6b4a1b', 0.4, { emissive: CB, ek: 0.7 })), 0.45, 0, 0);
  g.add(wallA, wallB, note('export controls, in both directions', -0.4, 2.3));
  // distillation: answers cross the walls and train a student
  const path = sample(arc([-3.6, Y, 0.4], [2.9, 1.0, 2.6], 5), 50);
  const tokens = dots(26, CA, 0.12, 2.6);
  g.add(tokens.obj);
  // open-weight releases
  const open = Array.from({ length: 5 }, () => {
    const r = new THREE.Mesh(new THREE.TorusGeometry(0.26, 0.04, 8, 28), glowMat(CB, { k: 2 }));
    g.add(r);
    return r;
  });
  g.add(lbl('open weights: anyone can download and run them', 0.6, 7.75, 0, { size: 0.24, color: INK }));
  if (big) g.add(lbl(`up to ${big[1]} trillion parameters`, 0.6, 7.3, 0, { size: 0.21 }));
  if (gap) g.add(note(`gap at the frontier: about ${gap[1]} months (range ${gap[2]} to ${gap[3]})`, 0, 4.0, { color: INK, size: 0.25, maxChars: 64 }));
  let T = 0;
  return {
    group: g,
    panel: [
      key([[css(CA), 'US labs: their models and chips'], [css(CB), 'Chinese labs'], [css(CA), 'answers harvested to train a copy (distillation)', 'dot'], [css(CB), 'open-weight releases: free for anyone to run', 'ring'], ['#9fb4d8', 'walls: export controls, rising on both sides']]),
      fine('A diagram, not a map: the two colors only tell the sides apart. The gap and the parameter count are quoted from the dated facts below.'),
    ],
    update(dt) {
      T += dt;
      for (const cmp of camps) {
        for (let i = 0; i < 14; i++) {
          const a = T * (0.5 + (i % 3) * 0.12) * cmp.dir + (i / 14) * TAU;
          const r = 1.25 + (i % 3) * 0.25;
          setInst(cmp.chips, i, cmp.x + Math.cos(a) * r, Y + Math.sin(a * 1.3 + i) * 0.5, Math.sin(a) * r, 1, 1, 1, cmp.c);
        }
        flushInst(cmp.chips);
      }
      const up = 0.5 + 0.5 * Math.sin(T * 0.5);
      wallA.scale.y = 0.8 + 2.6 * up;
      wallB.scale.y = 0.8 + 2.8 * (1 - up);
      const round = (T * 0.2) % 1;
      for (let i = 0; i < 26; i++) {
        along(path, (round + i / 26) % 1, _p3);
        tokens.set(i, _p3[0], _p3[1], _p3[2]);
      }
      tokens.flush();
      student.scale.setScalar(1 + ease.out(round) * 0.7);
      open.forEach((r, i) => {
        const f = (T * 0.07 + i / 5) % 1;
        const e = ease.out(f);
        r.position.set(lerp(4.3, 0.6 + Math.cos(i * 2.4) * 1.3, e), lerp(Y + 0.9, 6.4 + Math.sin(i * 1.7) * 0.35, e), Math.sin(i * 2.4) * 1.1 * e);
        r.rotation.set(T * 0.8 + i, T * 0.5, 0);
        r.scale.setScalar(Math.max(0.001, (0.5 + e * 0.7) * smoothstep(0, 0.08, f) * (1 - smoothstep(0.92, 1, f))));
      });
    },
  };
}

// ---------------------------------------------------------------------------- 8 · helix
// Discovery outruns checking: a genome on the left with the variants a model can now explain, a
// tree of claimed results on the right that a verifier works through one node at a time.
function helix({ issue }) {
  const g = new THREE.Group();
  stage(g, '#2a6a55');
  const found = figure(issue, /about (\d+)% of causal genetic variants on one benchmark, against (\d+)%/);
  const papers = figure(issue, /roughly (\d+) manuscripts/);
  const retracted = figure(issue, /(\w+) were retracted/);
  const share = found ? num(found[1]) / 100 : 0;
  const before = found ? num(found[2]) / 100 : 0;
  // the genome
  const DX = -3.5;
  const dna = at(new THREE.Group(), DX, 0.6, 0);
  g.add(dna);
  const RUNGS = 40;
  const HR = 1.15;
  const strandA = dots(RUNGS, PAL.cyan, 0.2, 2.2);
  const strandB = dots(RUNGS, PAL.violet, 0.2, 2.2);
  const marks = dots(RUNGS, GOOD, 0.36, 3);
  const rungs = segments(Array.from({ length: RUNGS * 2 }, () => [0, 0, 0]), '#8fa3c8', { k: 0.9, opacity: 0.45 });
  rungs.frustumCulled = false;
  dna.add(strandA.obj, strandB.obj, marks.obj, rungs);
  const rnd = rng(8);
  const kind = Array.from({ length: RUNGS }, () => rnd());
  const C_OLD = glow('#ffffff', 1);
  const C_NEW = glow(GOOD, 1);
  g.add(lbl('a genome', DX, 7.45, 0, { color: INK }));
  if (found) g.add(note(`causal variants identified: ${found[1]}% (${found[2]}% for the model before)`, DX, 2.6, { color: INK, maxChars: 28 }));
  // the tree of claimed results
  const TX = 3.3;
  const tree = at(new THREE.Group(), TX, 0.9, 0);
  g.add(tree);
  const nodes = [];
  const links = [];
  for (let d = 0; d < 4; d++)
    for (let k = 0; k < 1 << d; k++) {
      const n = { x: ((k + 0.5) / (1 << d) - 0.5) * 5.6, y: 5.6 - d * 1.6, state: 0, mesh: null };
      n.mesh = at(sph(0.2, glowMat('#8fa3c8', { k: 0.9 }), 14), n.x, n.y, 0);
      tree.add(n.mesh);
      if (d > 0) {
        const parent = nodes[(1 << (d - 1)) - 1 + (k >> 1)];
        links.push([parent.x, parent.y, 0], [n.x, n.y, 0]);
      }
      nodes.push(n);
    }
  tree.add(segments(links, '#8fa3c8', { k: 0.9, opacity: 0.4 }));
  const checker = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.04, 8, 32), glowMat('#ffffff', { k: 2.6 }));
  tree.add(checker);
  g.add(lbl(papers ? `${papers[1]} claimed results` : 'claimed results', TX, 7.45, 0, { color: INK }), note('a proof assistant checks them one at a time', TX, 2.6, { maxChars: 26 }));
  const count = tally([['verified by machine', css(GOOD)], ['withdrawn', css(BAD)], ['still unchecked']]);
  const tones = [glow('#8fa3c8', 0.9), glow(GOOD, 2.6), glow(BAD, 2.6)];
  const paint = (n, s) => {
    n.state = s;
    n.mesh.material.color.copy(tones[s]);
  };
  const recount = () => [1, 2, 0].forEach((s, i) => count.set(i, nodes.filter((n) => n.state === s).length));
  recount();
  const st = { T: 0, i: 0, wait: 0.6 };
  return {
    group: g,
    panel: [
      count.el,
      key([[css(GOOD), 'a newly explained variant, or a result verified by machine', 'dot'], ['#ffffff', 'a variant the earlier model also found', 'dot'], [css(BAD), 'a claim that was withdrawn', 'dot'], ['#8fa3c8', 'not yet checked by anyone', 'dot']]),
      fine(`A schematic of the bottleneck: results arrive faster than they can be checked. ${nodes.length} dots stand in for ${papers ? papers[1] : 'the'} manuscripts, about half of which came with a machine-checkable proof, so the single red dot overstates the ${retracted ? retracted[1] : 'few'} retractions.`),
    ],
    update(dt) {
      st.T += dt;
      const pos = rungs.geometry.attributes.position;
      const lit = (st.T * 0.12) % 1.4;
      for (let i = 0; i < RUNGS; i++) {
        const y = (i / RUNGS) * 6.4;
        const a = i * 0.42 + st.T * 0.6;
        const x = Math.cos(a) * HR;
        const z = Math.sin(a) * HR;
        strandA.set(i, x, y, z);
        strandB.set(i, -x, y, -z);
        pos.setXYZ(i * 2, x, y, z);
        pos.setXYZ(i * 2 + 1, -x, y, -z);
        if (kind[i] < share && i / RUNGS < lit) {
          marks.set(i, 0, y, 0);
          marks.tint(i, kind[i] < before ? C_OLD : C_NEW, 2.8);
        } else marks.hide(i);
      }
      pos.needsUpdate = true;
      strandA.flush();
      strandB.flush();
      marks.flush(true);
      // the verifier moves to the next node, then rules on it
      const j = st.i % nodes.length;
      const n = nodes[j];
      checker.position.set(damp(checker.position.x, n.x, 0.12, dt), damp(checker.position.y, n.y, 0.12, dt), 0);
      checker.rotation.z = st.T * 2;
      st.wait -= dt;
      if (st.wait <= 0) {
        if (j === 0) nodes.forEach((m) => paint(m, 0));
        paint(n, j === 11 ? 2 : (j * 0.6180339) % 1 < 0.5 ? 1 : 0);
        recount();
        st.i++;
        st.wait = 0.75;
      }
    },
  };
}

// ---------------------------------------------------------------------------- 9 · pullrequests
// A wall of a hundred squares shows the share of pull requests that an agent drafted; in front of
// it they queue for the one thing that did not speed up, a person reading the change.
function pullrequests({ issue }) {
  const g = new THREE.Group();
  stage(g, '#3a4f8a');
  const share = figure(issue, /by September 2026 more than (\d+)% of pull requests[^,]*, up from (\d+)% in March/);
  const merged = figure(issue, /(\d+)% of agent-involved pull requests were merged within two weeks, against about (\d+)%/);
  const sept = share ? num(share[1]) / 100 : 0;
  const march = share ? num(share[2]) / 100 : 0;
  const AG = PAL.magenta;
  const HU = '#9fb4d8';
  const C_AG = glow(AG, 1.9);
  const C_HU = glow(HU, 0.8);
  // the wall: one square per percent
  const WX = -3.5;
  const WZ = -1.8;
  const PITCH = 0.4;
  const wall = instanced(new THREE.BoxGeometry(0.31, 0.31, 0.12), new THREE.MeshBasicMaterial(), 100);
  const mix = new Float32Array(100);
  const big = liveLabel({ size: 0.72, width: 7.6, color: INK });
  big.position.set(WX, 6.95, WZ);
  g.add(wall, big, lbl('of pull requests started with an agent’s first draft', WX, 6.17, WZ, { size: 0.23, maxChars: 28 }), lbl('each square is 1% of pull requests', WX, 1.3, WZ, { size: 0.2 }));
  // the belt, the review and the merge
  const BY = 0.5;
  const BZ = 0.7;
  const STOP = 1.55;
  const GATE_X = 4.6;
  const END = GATE_X + 0.7;
  g.add(at(new THREE.Mesh(new THREE.BoxGeometry(12.6, 0.1, 1.2), stdMat('#141c30', { rough: 0.6 })), -0.5, BY - 0.05, BZ));
  const N = 22;
  const blocks = instanced(rbox(0.52, 0.38, 0.8, 0.06), new THREE.MeshBasicMaterial(), N);
  const desk = at(new THREE.Mesh(rbox(0.5, 1.5, 1.9, 0.08), glassMat('#1b2c52', 0.35, { emissive: PAL.cyan, ek: 0.6 })), STOP + 1.0, BY + 0.75, BZ);
  desk.add(edges(desk.geometry, PAL.cyan, { k: 1.8 }));
  const gate = at(new THREE.Mesh(new THREE.TorusGeometry(1.05, 0.07, 10, 40, Math.PI), glowMat(GOOD, { k: 2.2 })), GATE_X, BY, BZ);
  gate.rotation.y = Math.PI / 2;
  g.add(blocks, desk, gate, lbl('human review', STOP + 1.0, BY + 1.95, BZ, { color: '#bfe9ff' }), lbl('merged', GATE_X, BY + 1.5, BZ, { color: '#a8f0d0' }));
  if (merged) g.add(note(`merged within two weeks: ${merged[1]}% of agent-involved pull requests, about ${merged[2]}% of the rest`, 0, 2.7, { color: INK }));
  const S = { mode: 'sept', hold: 0 };
  const P = () => (S.mode === 'sept' ? sept : march);
  const items = Array.from({ length: N }, (_, i) => ({ x: STOP - i * 0.66, agent: false }));
  const reroll = () => items.forEach((it) => (it.agent = Math.random() < P()));
  reroll();
  const count = tally([['merged, drafted by an agent', css(AG)], ['merged, written by a person', HU]]);
  const tot = [0, 0];
  const seg = segmented(
    [
      { value: 'march', label: `March 2026 · ${(march * 100).toFixed(0)}%` },
      { value: 'sept', label: `September 2026 · over ${(sept * 100).toFixed(0)}%` },
    ],
    'sept',
    (v) => {
      S.mode = v;
      reroll();
      tot[0] = tot[1] = 0;
      count.set(0, 0);
      count.set(1, 0);
    },
  );
  return {
    group: g,
    panel: [
      section('Month', seg),
      count.el,
      key([[css(AG), 'pull request drafted by an agent'], [HU, 'pull request written by a person'], [css(PAL.cyan), 'review: a person reads every change'], [css(GOOD), 'merged', 'ring']]),
      fine('The wall and the mix of blocks follow the Octoverse share for the chosen month. The belt runs at the same speed in both, so the queue at the review desk is a picture of the bottleneck, not a measurement of it.'),
    ],
    update(dt) {
      const p = P();
      big.userData.set(`${S.mode === 'sept' ? 'more than ' : ''}${(p * 100).toFixed(0)}%`);
      const filled = Math.round(p * 100);
      for (let i = 0; i < 100; i++) {
        mix[i] = damp(mix[i], i < filled ? 1 : 0, 0.12 + (i % 10) * 0.012, dt);
        setInst(wall, i, WX + ((i % 10) - 4.5) * PITCH, 1.85 + Math.floor(i / 10) * PITCH, WZ, 1, 1, 1, _c.copy(C_HU).lerp(C_AG, mix[i]));
      }
      flushInst(wall);
      // the queue: nobody overtakes, and the reviewer takes one change at a time
      S.hold = Math.max(0, S.hold - dt);
      items.sort((m, n) => n.x - m.x);
      for (let k = 0; k < N; k++) {
        const it = items[k];
        let nx = it.x + dt * 1.5;
        if (k > 0) nx = Math.min(nx, items[k - 1].x - 0.66);
        if (it.x <= STOP && nx > STOP) {
          if (S.hold > 0) nx = STOP;
          else S.hold = 0.55;
        }
        it.x = Math.max(it.x, nx);
        if (it.x > END) {
          count.set(it.agent ? 0 : 1, ++tot[it.agent ? 0 : 1]);
          it.x = Math.min(items[N - 1].x - 0.66, -6.7);
          it.agent = Math.random() < p;
        }
        const s = Math.max(0.001, clamp((it.x + 6.6) / 0.5) * clamp((END - it.x) / 0.6));
        setInst(blocks, k, it.x, BY + 0.19, BZ, s, s, s, it.agent ? C_AG : C_HU);
      }
      flushInst(blocks);
    },
  };
}

// ---------------------------------------------------------------------------- 10 · provenance
// Noise resolves into a picture that carries an invisible watermark. A detector lens sweeps across
// and shows which tiles still carry the mark; re-encoding or cropping strips part of it.
function provenance({ issue }) {
  const g = new THREE.Group();
  stage(g, '#5a2a6a');
  const wm = figure(issue, /more than (\d+) billion images and videos/);
  const W = 30;
  const H = 18;
  const N = W * H;
  const TS = 0.34;
  const tiles = instanced(new THREE.PlaneGeometry(TS * 0.94, TS * 0.94), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), N);
  const board = at(new THREE.Group(), 0, 3.95, 0);
  board.add(tiles);
  g.add(board);
  // the "photograph": a sun setting over hills, computed tile by tile
  const img = new Float32Array(N * 3);
  const noise = new Float32Array(N * 3);
  const delay = new Float32Array(N);
  const mark = new Uint8Array(N);
  const rnd = rng(14);
  const SKY_TOP = new THREE.Color('#33265f');
  const SKY_LOW = new THREE.Color('#ff7a59');
  const HILL_A = new THREE.Color('#1f3b5c');
  const HILL_B = new THREE.Color('#2c6a6a');
  const SUN = new THREE.Color('#ffd98a').multiplyScalar(1.5);
  const MARK = glow(GOOD, 2.1);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const u = x / (W - 1);
      const v = y / (H - 1);
      if (v < 0.3 + 0.1 * Math.sin(u * 7) + 0.06 * Math.sin(u * 17 + 1)) _c.copy(HILL_A).lerp(HILL_B, u * 0.7);
      else if (Math.hypot((u - 0.68) * (W / H), v - 0.66) < 0.2) _c.copy(SUN);
      else _c.copy(SKY_TOP).lerp(SKY_LOW, clamp(1.15 - v * 1.3));
      _c.toArray(img, i * 3);
      noise.set([rnd() * 0.7, rnd() * 0.7, rnd() * 0.7], i * 3);
      delay[i] = rnd() * 0.6 + u * 0.25;
      mark[i] = (x + y * 2) % 5 < 2 ? 1 : 0; // the hidden pattern: diagonal stripes
    }
  const alive = new Uint8Array(mark);
  const LW = TS * 5;
  const lens = at(new THREE.Mesh(new THREE.PlaneGeometry(LW, H * TS + 0.5), glassMat('#0f2a2a', 0.16, { emissive: GOOD, ek: 0.5 })), 0, 0, 0.06);
  lens.add(edges(new THREE.BoxGeometry(LW, H * TS + 0.5, 0.02), GOOD, { k: 2.2 }), lbl('detector', 0, (H * TS) / 2 + 0.62, 0, { color: '#a8f0d0' }));
  board.add(lens);
  if (wm) g.add(note(`one watermark alone has been applied to more than ${wm[1]} billion images and videos`, 0, 2.6, { color: INK }));
  const st = { T: 0, cropped: false };
  const read = tally([['watermark signal remaining', css(GOOD)], ['detector verdict']]);
  const show = () => {
    let a = 0;
    let m = 0;
    for (let i = 0; i < N; i++) {
      m += mark[i];
      a += alive[i];
    }
    const s = a / m;
    read.set(0, (s * 100).toFixed(0) + '%');
    read.set(1, s >= 0.5 ? 'watermarked' : s >= 0.2 ? 'probably watermarked' : 'cannot tell');
  };
  show();
  const strip = () => {
    for (let i = 0; i < N; i++) if (alive[i] && Math.random() < 0.55) alive[i] = 0;
    st.cropped = true;
    show();
  };
  const restore = () => {
    alive.set(mark);
    st.cropped = false;
    show();
  };
  return {
    group: g,
    panel: [
      h('div.btnrow', button('Re-encode and crop', { kind: 'primary', onClick: strip }), button('Restore the original', { onClick: restore })),
      read.el,
      key([[css(GOOD), 'under the lens: a tile where the detector finds the mark'], ['#8fa3c8', 'the picture as people see it: the mark is invisible']]),
      fine('A toy picture and a toy watermark. Real schemes hide a statistical pattern in pixels, audio or word choices and survive far more editing than this one, but the principle holds: editing can weaken a mark, and a model that never applied one leaves nothing to find.'),
    ],
    update(dt) {
      st.T += dt;
      const cyc = st.T % 16;
      const fadeOut = 1 - smoothstep(14.6, 16, cyc);
      const lx = Math.sin(st.T * 0.6) * (W * TS * 0.42);
      lens.position.x = lx;
      for (let i = 0; i < N; i++) {
        const x = i % W;
        const y = (i - x) / W;
        const k = smoothstep(0, 1, clamp((cyc - 0.4) / 3.2 - delay[i])) * fadeOut;
        const px = (x - (W - 1) / 2) * TS;
        const s = st.cropped && (x < 3 || x > W - 4 || y < 2 || y > H - 3) ? 0.001 : 1;
        _c.setRGB(lerp(noise[i * 3], img[i * 3], k), lerp(noise[i * 3 + 1], img[i * 3 + 1], k), lerp(noise[i * 3 + 2], img[i * 3 + 2], k));
        if (k > 0.9 && Math.abs(px - lx) < LW / 2) {
          if (alive[i]) _c.copy(MARK);
          else _c.multiplyScalar(0.45);
        }
        setInst(tiles, i, px, (y - (H - 1) / 2) * TS, 0, s, s, 1, _c);
      }
      flushInst(tiles);
    },
  };
}

// ---------------------------------------------------------------------------- emblems
// Small glyphs for the monoliths (same idea as core/icons3d.js): c is the monolith's rank color.
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
export const VIGNETTES = {
  containment: { build: containment, emblem: EMBLEMS.containment, view: { dir: [0.5, 0.36, 1], target: [0.1, 3.2, 0.2], fit: 7.9 } },
  compute: { build: compute, emblem: EMBLEMS.compute, view: { dir: [0.4, 0.38, 1], target: [0, 2.8, 0], fit: 7.5 } },
  recursion: { build: recursion, emblem: EMBLEMS.recursion, view: { dir: [0.4, 0.38, 1], target: [0.6, 3.4, 0], fit: 7.2 } },
  governance: { build: governance, emblem: EMBLEMS.governance, view: { dir: [0.4, 0.38, 1], target: [0, 3.6, 0.9], fit: 7.9 } },
  injection: { build: injection, emblem: EMBLEMS.injection, view: { dir: [0.4, 0.38, 1], target: [-0.9, 3.5, 0.4], fit: 7.1 } },
  observer: { build: observer, emblem: EMBLEMS.observer, view: { dir: [0.4, 0.38, 1], target: [-0.4, 3.1, 0.5], fit: 7.2 } },
  rivalry: { build: rivalry, emblem: EMBLEMS.rivalry, view: { dir: [0.4, 0.38, 1], target: [0, 3.8, 0.5], fit: 7.6 } },
  helix: { build: helix, emblem: EMBLEMS.helix, view: { dir: [0.4, 0.38, 1], target: [0, 3.8, 0.4], fit: 7.4 } },
  pullrequests: { build: pullrequests, emblem: EMBLEMS.pullrequests, view: { dir: [0.4, 0.38, 1], target: [-0.4, 3.3, 0], fit: 7.6 } },
  provenance: { build: provenance, emblem: EMBLEMS.provenance, view: { dir: [0.25, 0.22, 1], target: [0, 3.7, 0], fit: 6.9 } },
};
