// Procedural 3D glyphs (about 1.6 units across) used on portals and gallery pods, so every
// chapter and lab has a recognizable object instead of a flat icon.
// icon(name, color) returns a Group; if it animates, group.userData.tick(t, dt) drives it.
import * as THREE from 'three';
import { glowMat, stdMat, glassMat, line, segments, edges, rbox, points, rng, arc, halo } from './geom.js';
import { glow, SERIES } from './colors.js';

const sphere = (r, mat, seg = 14) => new THREE.Mesh(new THREE.SphereGeometry(r, seg, Math.max(8, seg - 4)), mat);
const box = (w, h, d, mat) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
const at = (obj, x, y, z) => {
  obj.position.set(x, y, z);
  return obj;
};

export const ICONS = {
  /** families: a small tree of related nodes */
  constellation(c) {
    const g = new THREE.Group();
    const rnd = rng(7);
    const nodes = [[0, 0, 0, 0.16]];
    const pairs = [];
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2 + 0.3;
      const p = [Math.cos(a) * 0.55, (rnd() - 0.5) * 0.5, Math.sin(a) * 0.55, 0.1];
      nodes.push(p);
      pairs.push([0, 0, 0], p);
      for (let j = 0; j < 2; j++) {
        const b = a + (j - 0.5) * 0.7;
        const q = [Math.cos(b) * 0.95, p[1] + (rnd() - 0.5) * 0.5, Math.sin(b) * 0.95, 0.055];
        nodes.push(q);
        pairs.push(p, q);
      }
    }
    g.add(segments(pairs, c, { k: 1.2, opacity: 0.55 }));
    const mat = glowMat(c, { k: 2.4 });
    nodes.forEach(([x, y, z, r]) => g.add(at(sphere(r, mat), x, y, z)));
    return g;
  },

  /** neural network: layers of units with signals pulsing left to right */
  network(c) {
    const g = new THREE.Group();
    const layers = [3, 5, 5, 2];
    const pos = layers.map((n, li) => Array.from({ length: n }, (_, i) => [(li - 1.5) * 0.48, (i - (n - 1) / 2) * 0.3, 0]));
    const pairs = [];
    for (let l = 0; l < layers.length - 1; l++) for (const a of pos[l]) for (const b of pos[l + 1]) pairs.push(a, b);
    g.add(segments(pairs, c, { k: 0.9, opacity: 0.28 }));
    const mats = layers.map(() => glowMat(c, { k: 2 }));
    pos.forEach((layer, li) => layer.forEach((p) => g.add(at(sphere(0.07, mats[li]), ...p))));
    const base = glow(c, 1);
    g.userData.tick = (t) => {
      mats.forEach((m, li) => m.color.copy(base).multiplyScalar(1.3 + 2.2 * Math.max(0, Math.sin(t * 2.6 - li * 1.1))));
    };
    return g;
  },

  /** problem space: a stack of levels */
  tower(c) {
    const g = new THREE.Group();
    const cols = ['#ffb84d', '#ff7a59', '#ff6fb1', '#9d8cff', '#5ad7ff'];
    cols.forEach((col, i) => {
      const w = 1.15 - i * 0.14;
      const slab = new THREE.Mesh(rbox(w, 0.13, w, 0.03), stdMat('#111a2e', { emissive: col, ek: 0.55, rough: 0.35 }));
      slab.position.y = -0.55 + i * 0.27;
      slab.add(edges(slab.geometry, col, { k: 1.8, opacity: 0.9 }));
      g.add(slab);
    });
    return g;
  },

  /** data flow: stations joined by a pipe with packets moving through */
  pipeline(c) {
    const g = new THREE.Group();
    const xs = [-0.72, 0, 0.72];
    xs.forEach((x, i) => {
      const b = new THREE.Mesh(rbox(0.36, 0.36, 0.36, 0.06), stdMat('#111a2e', { emissive: c, ek: 0.35 }));
      b.position.set(x, i === 1 ? 0.18 : -0.12, 0);
      b.add(edges(b.geometry, c, { k: 1.8 }));
      g.add(b);
    });
    const curve = new THREE.CatmullRomCurve3([new THREE.Vector3(-0.72, -0.12, 0), new THREE.Vector3(0, 0.18, 0), new THREE.Vector3(0.72, -0.12, 0)]);
    g.add(line(curve.getPoints(24), c, { k: 1.1, opacity: 0.6 }));
    const dots = [0, 1, 2].map(() => g.add(sphere(0.055, glowMat('#ffffff', { k: 2.6 }))) && g.children[g.children.length - 1]);
    g.userData.tick = (t) => dots.forEach((d, i) => d.position.copy(curve.getPoint((t * 0.35 + i / 3) % 1)));
    return g;
  },

  /** performance: bars of different heights */
  bars() {
    const g = new THREE.Group();
    const hs = [0.5, 0.95, 0.68, 1.25];
    hs.forEach((hgt, i) => {
      const b = new THREE.Mesh(rbox(0.22, hgt, 0.22, 0.04), glowMat(SERIES[i], { k: 1.7 }));
      b.position.set((i - 1.5) * 0.34, -0.62 + hgt / 2, 0);
      g.add(b);
    });
    g.add(line([[-0.8, -0.63, 0], [0.8, -0.63, 0]], '#8fa3c8', { k: 1, opacity: 0.7 }));
    return g;
  },

  /** ecosystem: a little spiral galaxy */
  galaxy(c) {
    const n = 420;
    const rnd = rng(11);
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const base = glow(c, 1);
    const hot = glow('#ffffff', 1);
    for (let i = 0; i < n; i++) {
      const r = Math.pow(rnd(), 0.6) * 0.95;
      const a = r * 5.2 + (i % 2) * Math.PI + rnd.normal() * 0.22;
      pos[i * 3] = Math.cos(a) * r;
      pos[i * 3 + 1] = rnd.normal() * 0.05 * (1.2 - r);
      pos[i * 3 + 2] = Math.sin(a) * r;
      const cc = base.clone().lerp(hot, Math.max(0, 0.9 - r * 1.3)).multiplyScalar(1.2 + rnd() * 1.6);
      col.set([cc.r, cc.g, cc.b], i * 3);
      size[i] = 0.035 + rnd() * 0.05;
    }
    const g = new THREE.Group();
    const p = points(pos, col, size);
    p.rotation.x = 0.5;
    g.add(p);
    g.userData.tick = (t) => (p.rotation.y = t * 0.25);
    return g;
  },

  /** deployment: a cloud above devices */
  cloud(c) {
    const g = new THREE.Group();
    const cm = glassMat('#8fb8ff', 0.3, { emissive: c, ek: 0.5 });
    [[-0.3, 0.42, 0, 0.26], [0, 0.52, 0, 0.34], [0.32, 0.42, 0, 0.24]].forEach(([x, y, z, r]) => g.add(at(sphere(r, cm, 18), x, y, z)));
    const dm = stdMat('#111a2e', { emissive: c, ek: 0.4 });
    const devs = [[-0.6, -0.5, 0.1], [0, -0.58, 0.25], [0.6, -0.5, 0.1]];
    const pairs = [];
    devs.forEach(([x, y, z]) => {
      const d = new THREE.Mesh(rbox(0.3, 0.2, 0.22, 0.04), dm);
      d.position.set(x, y, z);
      d.add(edges(d.geometry, c, { k: 1.8 }));
      g.add(d);
      pairs.push([x, y + 0.1, z], [x * 0.4, 0.28, 0]);
    });
    g.add(segments(pairs, c, { k: 1.3, opacity: 0.7 }));
    return g;
  },

  /** hot issues: a pulsing spiked core */
  flame(c) {
    const g = new THREE.Group();
    const shell = edges(new THREE.IcosahedronGeometry(0.78, 0), c, { k: 2 });
    const core = new THREE.Mesh(new THREE.OctahedronGeometry(0.36, 0), glowMat('#ffd9a0', { k: 2.4 }));
    const mid = edges(new THREE.IcosahedronGeometry(0.52, 1), '#ffb84d', { k: 1.4, opacity: 0.6 });
    g.add(shell, mid, core, halo(c, 2.2, 0.35));
    g.userData.tick = (t) => {
      const s = 1 + 0.14 * Math.sin(t * 3.1);
      core.scale.setScalar(s);
      core.rotation.y = t * 1.3;
      mid.rotation.set(t * 0.4, -t * 0.6, 0);
    };
    return g;
  },

  /** GPT: a row of tokens with attention arcs */
  gpt(c) {
    const g = new THREE.Group();
    const xs = [-0.7, -0.35, 0, 0.35, 0.7];
    const mats = xs.map((_, i) => stdMat('#111a2e', { emissive: i === 4 ? '#ffffff' : c, ek: i === 4 ? 1.2 : 0.5 }));
    xs.forEach((x, i) => {
      const chip = new THREE.Mesh(rbox(0.26, 0.2, 0.16, 0.04), mats[i]);
      chip.position.set(x, -0.42, 0);
      chip.add(edges(chip.geometry, c, { k: 1.6 }));
      g.add(chip);
    });
    const arcs = [0, 1, 2, 3].map((i) => {
      const a = arc([xs[4], -0.3, 0], [xs[i], -0.3, 0], 0.35 + (4 - i) * 0.2);
      const l = line(a.getPoints(20), c, { k: 1.6, opacity: 0.9 });
      g.add(l);
      return l;
    });
    g.userData.tick = (t) => arcs.forEach((l, i) => (l.material.opacity = 0.2 + 0.8 * Math.pow(0.5 + 0.5 * Math.sin(t * 1.7 + i * 1.9), 2)));
    return g;
  },

  /** CNN: a kernel window sliding over an image grid */
  conv(c) {
    const g = new THREE.Group();
    const n = 6;
    const tile = new THREE.BoxGeometry(0.2, 0.2, 0.03);
    const rnd = rng(5);
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const v = (x + y) % 3 === 0 || rnd() > 0.7 ? 1.5 : 0.35;
        const m = new THREE.Mesh(tile, glowMat(c, { k: v }));
        m.position.set((x - (n - 1) / 2) * 0.23, (y - (n - 1) / 2) * 0.23, 0);
        g.add(m);
      }
    const frame = edges(new THREE.BoxGeometry(0.69, 0.69, 0.12), '#ffffff', { k: 2.6 });
    g.add(frame);
    g.userData.tick = (t) => {
      const s = Math.floor(t * 2.2) % 16;
      frame.position.set(((s % 4) - 1.5) * 0.23, (Math.floor(s / 4) - 1.5) * 0.23, 0.05);
    };
    g.rotation.x = -0.35;
    return g;
  },

  /** diffusion: noise that condenses into a shape and dissolves again */
  diffusion(c) {
    const n = 260;
    const rnd = rng(3);
    const noise = new Float32Array(n * 3);
    const shape = new Float32Array(n * 3);
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n).fill(0.06);
    const cc = glow(c, 2);
    for (let i = 0; i < n; i++) {
      noise.set([rnd.normal() * 0.5, rnd.normal() * 0.5, rnd.normal() * 0.5], i * 3);
      const a = (i / n) * Math.PI * 2;
      const b = a * 3;
      shape.set([Math.cos(a) * (0.6 + 0.2 * Math.cos(b)), Math.sin(b) * 0.2, Math.sin(a) * (0.6 + 0.2 * Math.cos(b))], i * 3);
      col.set([cc.r, cc.g, cc.b], i * 3);
    }
    const p = points(pos, col, size);
    const g = new THREE.Group();
    g.add(p);
    g.userData.tick = (t) => {
      const k = 0.5 - 0.5 * Math.cos(t * 0.9);
      const e = k * k * (3 - 2 * k);
      for (let i = 0; i < n * 3; i++) pos[i] = noise[i] + (shape[i] - noise[i]) * e;
      p.geometry.attributes.position.needsUpdate = true;
    };
    return g;
  },

  /** grid world: an agent walking tiles towards a goal */
  world(c) {
    const g = new THREE.Group();
    const n = 4;
    for (let z = 0; z < n; z++)
      for (let x = 0; x < n; x++) {
        const wall = (x === 1 && z === 1) || (x === 2 && z === 2);
        const m = new THREE.Mesh(rbox(0.3, wall ? 0.3 : 0.07, 0.3, 0.03), stdMat(wall ? '#2a3350' : '#111a2e', { emissive: c, ek: wall ? 0.1 : 0.35 }));
        m.position.set((x - 1.5) * 0.34, wall ? 0.12 : 0, (z - 1.5) * 0.34);
        g.add(m);
      }
    const goal = new THREE.Mesh(new THREE.OctahedronGeometry(0.11), glowMat('#ffd166', { k: 2.6 }));
    goal.position.set(1.5 * 0.34, 0.26, -1.5 * 0.34);
    const agent = new THREE.Mesh(rbox(0.18, 0.18, 0.18, 0.04), glowMat('#ffffff', { k: 2.2 }));
    g.add(goal, agent);
    const path = [[0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0]];
    g.userData.tick = (t) => {
      const f = (t * 1.6) % (path.length + 1);
      const i = Math.min(path.length - 1, Math.floor(f));
      const j = Math.min(path.length - 1, i + 1);
      const k = Math.min(1, f - i);
      agent.position.set((path[i][0] + (path[j][0] - path[i][0]) * k - 1.5) * 0.34, 0.16, (path[i][1] + (path[j][1] - path[i][1]) * k - 1.5) * 0.34);
      goal.rotation.y = t * 2;
    };
    g.rotation.x = 0.5;
    g.position.y = -0.1;
    return g;
  },

  /** embedding space: clusters of meaning */
  embed() {
    const rnd = rng(19);
    const n = 240;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const centers = [[-0.5, 0.25, 0.1], [0.5, 0.3, -0.2], [0.05, -0.45, 0.3], [0.1, 0.1, -0.6]];
    for (let i = 0; i < n; i++) {
      const k = i % centers.length;
      const cc = glow(SERIES[[0, 1, 2, 6][k]], 2.2);
      pos.set([centers[k][0] + rnd.normal() * 0.17, centers[k][1] + rnd.normal() * 0.17, centers[k][2] + rnd.normal() * 0.17], i * 3);
      col.set([cc.r, cc.g, cc.b], i * 3);
      size[i] = 0.05 + rnd() * 0.04;
    }
    const g = new THREE.Group();
    g.add(points(pos, col, size));
    return g;
  },

  /** optimizer: a ball spiralling down a loss bowl */
  optim(c) {
    const g = new THREE.Group();
    for (let i = 1; i <= 5; i++) {
      const r = i * 0.17;
      const pts = [];
      for (let a = 0; a <= 48; a++) pts.push([Math.cos((a / 48) * Math.PI * 2) * r, r * r * 0.9 - 0.45, Math.sin((a / 48) * Math.PI * 2) * r]);
      g.add(line(pts, c, { k: 1.4 - i * 0.12, opacity: 0.85 - i * 0.1 }));
    }
    const ball = sphere(0.09, glowMat('#ffffff', { k: 2.8 }));
    g.add(ball);
    g.userData.tick = (t) => {
      const f = (t * 0.35) % 1;
      const r = 0.85 * Math.pow(1 - f, 1.5);
      const a = f * 14;
      ball.position.set(Math.cos(a) * r, r * r * 0.9 - 0.42, Math.sin(a) * r);
    };
    return g;
  },

  /** quantization: a smooth curve next to its staircase */
  quant(c) {
    const g = new THREE.Group();
    const smooth = [];
    const steps = [];
    for (let i = 0; i <= 60; i++) {
      const x = (i / 60) * 1.6 - 0.8;
      const y = Math.sin(x * 3.2) * 0.45;
      smooth.push([x, y, 0.12]);
      const q = Math.round(y / 0.22) * 0.22;
      if (steps.length) steps.push([x, steps[steps.length - 1][1], -0.12]);
      steps.push([x, q, -0.12]);
    }
    g.add(line(smooth, '#ffffff', { k: 1.6, opacity: 0.9 }), line(steps, c, { k: 2.2 }));
    return g;
  },

  /** KV cache: stacked memory rows, the newest one lit */
  cache(c) {
    const g = new THREE.Group();
    const rows = [];
    for (let i = 0; i < 6; i++) {
      const m = new THREE.Mesh(rbox(1.3, 0.12, 0.5, 0.03), glowMat(c, { k: 0.5 }));
      m.position.y = -0.5 + i * 0.2;
      rows.push(m);
      g.add(m);
    }
    const base = glow(c, 1);
    g.userData.tick = (t) => {
      const cur = Math.floor(t * 1.5) % 7;
      rows.forEach((m, i) => m.material.color.copy(base).multiplyScalar(i < cur ? 0.75 : i === cur ? 2.8 : 0.16));
    };
    return g;
  },

  /** batching: requests queueing into a GPU */
  batch(c) {
    const g = new THREE.Group();
    const gpu = new THREE.Mesh(rbox(0.6, 0.7, 0.6, 0.06), stdMat('#111a2e', { emissive: c, ek: 0.4 }));
    gpu.position.x = 0.5;
    gpu.add(edges(gpu.geometry, c, { k: 2 }));
    g.add(gpu);
    const blocks = Array.from({ length: 6 }, (_, i) => {
      const b = new THREE.Mesh(rbox(0.14, 0.14, 0.14, 0.03), glowMat(SERIES[i % 4], { k: 1.9 }));
      g.add(b);
      return b;
    });
    g.userData.tick = (t) =>
      blocks.forEach((b, i) => {
        const f = (t * 0.45 + i / 6) % 1;
        b.position.set(-0.95 + f * 1.3, ((i % 3) - 1) * 0.2, 0);
        b.scale.setScalar(f > 0.85 ? Math.max(0.01, (1 - f) / 0.15) : 1);
      });
    return g;
  },

  /** bake-off: a podium */
  podium() {
    const g = new THREE.Group();
    [[0, 0.8, 0], [-0.42, 0.55, 1], [0.42, 0.38, 2]].forEach(([x, hgt, i]) => {
      const b = new THREE.Mesh(rbox(0.38, hgt, 0.38, 0.04), stdMat('#111a2e', { emissive: SERIES[i], ek: 0.6 }));
      b.position.set(x, -0.6 + hgt / 2, 0);
      b.add(edges(b.geometry, SERIES[i], { k: 1.8 }));
      g.add(b, at(sphere(0.12, glowMat(SERIES[i], { k: 2.2 })), x, -0.6 + hgt + 0.16, 0));
    });
    return g;
  },

  /** records: a rising curve over a bar timeline */
  records(c) {
    const g = new THREE.Group();
    const pts = [];
    for (let i = 0; i < 7; i++) {
      const hgt = 0.12 + Math.pow(i / 6, 2.2) * 1.1;
      const b = new THREE.Mesh(new THREE.BoxGeometry(0.13, hgt, 0.13), glowMat(c, { k: 0.8 + i * 0.22 }));
      b.position.set((i - 3) * 0.24, -0.6 + hgt / 2, 0);
      g.add(b);
      pts.push([(i - 3) * 0.24, -0.6 + hgt + 0.08, 0]);
    }
    g.add(line(pts, '#ffffff', { k: 2 }));
    return g;
  },

  /** generic glowing orb */
  orb(c) {
    const g = new THREE.Group();
    g.add(sphere(0.4, glowMat(c, { k: 1.8 }), 20), edges(new THREE.IcosahedronGeometry(0.75, 1), c, { k: 1.2, opacity: 0.5 }));
    return g;
  },
};

export function icon(name, color) {
  return (ICONS[name] || ICONS.orb)(color);
}
