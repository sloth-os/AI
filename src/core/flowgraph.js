// A reusable 3D node-link diagram: typed nodes (rounded boxes with a small glyph), directed edges
// with arrowheads and moving packets, floor zones, and "exhibits": small 3D depictions of what the
// data looks like at one point of a pipeline (token chips, vectors, tensors, images, JSON ...).
// Used by the Data Flow and Deploy chapters.
//
// Everything a graph draws lives under `graph.group` and is batched: one instanced mesh for the node
// bodies, one mesh for all glyphs / edge tubes / arrowheads, one line set for the outlines, and one
// instanced mesh for the packets. Highlighting re-tints vertex-color ranges, so a graph costs about
// five draw calls plus its labels no matter how many nodes it has.
import * as THREE from 'three';
import { SERIES, glow, seq, div, withAlpha } from './colors.js';
import { rbox, instanced, setInst, flushInst, disposeDeep, halo, ring, rng, v3 } from './geom.js';
import { textSprite, FONTS } from './labels.js';
import { damp, clamp } from './tween.js';

// ------------------------------------------------------------------ vocabulary

/** Node roles. The color identifies the role everywhere: in the scene and in the panel legend. */
export const ROLES = {
  client: { label: 'People and devices', color: SERIES[0] },
  service: { label: 'Services and logic', color: SERIES[6] },
  compute: { label: 'Models and accelerators', color: SERIES[1] },
  storage: { label: 'Data at rest', color: SERIES[3] },
  external: { label: 'Tools and outside systems', color: SERIES[2] },
};

/** Node type -> role. The type also selects the glyph drawn on top of the box. */
export const NODE_TYPES = {
  user: 'client', client: 'client', device: 'client', laptop: 'client', browser: 'client', mic: 'client',
  app: 'service', service: 'service', gateway: 'service', queue: 'service', op: 'service', sampler: 'service', agent: 'service', scaler: 'service', lock: 'service',
  model: 'compute', gpu: 'compute', layers: 'compute',
  database: 'storage', store: 'storage', file: 'storage', table: 'storage', image: 'storage',
  tool: 'external', cloud: 'external', globe: 'external',
};

const BW = 1.7; // node body: width, height, depth, corner radius
const BH = 0.6;
const BD = 1.12;
const BR = 0.16;
const K = 24; // polyline segments per edge
const NODE = { dim: 0.2, normal: 1, focus: 1.6 };
const EDGE = { dim: 0.16, normal: 0.55, active: 2.3 };
const BODY = new THREE.Color('#3a4868');
const UP = new THREE.Vector3(0, 1, 0);
const ONE = new THREE.Vector3(1, 1, 1);
const LIGHT = new THREE.Vector3(0.3, 0.85, 0.45).normalize();

const _v = new THREE.Vector3();
const _n = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _t = new THREE.Vector3();
const _c = new THREE.Color();
const _c2 = new THREE.Color();
const _m = new THREE.Matrix4();
const _m3 = new THREE.Matrix3();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();

/** Growable vertex-colored geometry whose ranges can be re-tinted after it is built. */
class ColorBatch {
  constructor() {
    this.pos = [];
    this.col = [];
  }

  get count() {
    return this.pos.length / 3;
  }

  /** Append a geometry (it is consumed). `shade` bakes a soft directional light into the colors. */
  addGeometry(geo, matrix, color, shade = 0.5) {
    const g = geo.index ? geo.toNonIndexed() : geo;
    const p = g.attributes.position;
    const nrm = g.attributes.normal;
    if (matrix) _m3.getNormalMatrix(matrix);
    for (let i = 0; i < p.count; i++) {
      _v.fromBufferAttribute(p, i);
      if (matrix) _v.applyMatrix4(matrix);
      this.pos.push(_v.x, _v.y, _v.z);
      let k = 1;
      if (shade && nrm) {
        _n.fromBufferAttribute(nrm, i);
        if (matrix) _n.applyMatrix3(_m3).normalize();
        k = 1 - shade + shade * 1.3 * Math.max(0, _n.dot(LIGHT));
      }
      this.col.push(color.r * k, color.g * k, color.b * k);
    }
    if (g !== geo) g.dispose();
    geo.dispose();
  }

  addSegment(a, b, color) {
    this.pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
    this.col.push(color.r, color.g, color.b, color.r, color.g, color.b);
  }

  finish() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.pos), 3));
    this.base = new Float32Array(this.col);
    this.attr = new THREE.BufferAttribute(new Float32Array(this.col), 3);
    this.attr.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('color', this.attr);
    this.pos = this.col = null;
    return g;
  }

  /** Multiply the color of `n` vertices starting at `start` by k (relative to their base color). */
  tint(start, n, k) {
    const a = this.attr.array;
    const b = this.base;
    for (let i = start * 3, e = (start + n) * 3; i < e; i++) a[i] = b[i] * k;
    this.attr.needsUpdate = true;
  }
}

/** Points of a rounded rectangle centerd on the origin, counter-clockwise, as [x, y] pairs. */
function roundRectPts(w, h, r, seg = 5) {
  const pts = [];
  const hw = w / 2 - r;
  const hh = h / 2 - r;
  const corners = [[hw, hh, 0], [-hw, hh, Math.PI / 2], [-hw, -hh, Math.PI], [hw, -hh, Math.PI * 1.5]];
  for (const [cx, cy, a0] of corners)
    for (let i = 0; i <= seg; i++) {
      const a = a0 + (i / seg) * (Math.PI / 2);
      pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
  return pts;
}

/** Tube around a polyline (Float32Array of xyz), written straight into a batch. */
function addTube(batch, pts, count, i0, i1, radius, color, dashed) {
  const R = 6;
  const ringAt = (i, out) => {
    const a = Math.max(0, i - 1) * 3;
    const b = Math.min(count - 1, i + 1) * 3;
    _t.set(pts[b] - pts[a], pts[b + 1] - pts[a + 1], pts[b + 2] - pts[a + 2]).normalize();
    _a.crossVectors(UP, _t);
    if (_a.lengthSq() < 1e-6) _a.set(1, 0, 0);
    _a.normalize();
    _b.crossVectors(_t, _a);
    for (let j = 0; j < R; j++) {
      const ang = (j / R) * Math.PI * 2;
      const cx = Math.cos(ang) * radius;
      const sx = Math.sin(ang) * radius;
      out[j * 3] = pts[i * 3] + _a.x * cx + _b.x * sx;
      out[j * 3 + 1] = pts[i * 3 + 1] + _a.y * cx + _b.y * sx;
      out[j * 3 + 2] = pts[i * 3 + 2] + _a.z * cx + _b.z * sx;
    }
  };
  let prev = new Float32Array(R * 3);
  let next = new Float32Array(R * 3);
  ringAt(i0, prev);
  for (let i = i0; i < i1; i++) {
    ringAt(i + 1, next);
    if (!dashed || i % 3 !== 2) {
      for (let j = 0; j < R; j++) {
        const j2 = (j + 1) % R;
        const quad = [prev, j, next, j, next, j2, prev, j, next, j2, prev, j2];
        for (let q = 0; q < 12; q += 2) {
          const src = quad[q];
          const o = quad[q + 1] * 3;
          batch.pos.push(src[o], src[o + 1], src[o + 2]);
          batch.col.push(color.r, color.g, color.b);
        }
      }
    }
    const tmp = prev;
    prev = next;
    next = tmp;
  }
}

// ------------------------------------------------------------------ glyphs

/** Tiny modeling kit: every primitive is merged into the graph's triangle batch. */
function makeKit(batch, origin, scale, color) {
  const base = new THREE.Matrix4().compose(origin, new THREE.Quaternion(), new THREE.Vector3(scale, scale, scale));
  const frame = new THREE.Matrix4();
  const local = new THREE.Matrix4();
  const p = new THREE.Vector3();
  const main = glow(color, 1.45);
  const dim = glow(color, 0.62);
  const hot = glow(new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.72), 1.75);
  const put = (geo, x, y, z, c = main, rx = 0, ry = 0, rz = 0) => {
    local.compose(p.set(x, y, z), _q.setFromEuler(_e.set(rx, ry, rz)), ONE);
    local.premultiply(frame).premultiply(base);
    batch.addGeometry(geo, local, c);
  };
  return {
    main,
    dim,
    hot,
    frame,
    box: (w, h, d, x, y, z, c, rx, ry, rz) => put(new THREE.BoxGeometry(w, h, d), x, y, z, c, rx, ry, rz),
    cyl: (r1, r2, h, x, y, z, c, rx, ry, rz, seg = 14) => put(new THREE.CylinderGeometry(r1, r2, h, seg), x, y, z, c, rx, ry, rz),
    sph: (r, x, y, z, c) => put(new THREE.SphereGeometry(r, 12, 9), x, y, z, c),
    cone: (r, h, x, y, z, c, rx, ry, rz, seg = 12) => put(new THREE.ConeGeometry(r, h, seg), x, y, z, c, rx, ry, rz),
    torus: (r, tube, x, y, z, c, rx, ry, rz, arc = Math.PI * 2) => put(new THREE.TorusGeometry(r, tube, 6, 22, arc), x, y, z, c, rx, ry, rz),
    oct: (r, x, y, z, c, ry) => put(new THREE.OctahedronGeometry(r, 0), x, y, z, c, 0, ry, 0),
    rod(a, b, r, c = dim) {
      const A = new THREE.Vector3(...a);
      const B = new THREE.Vector3(...b);
      const d = B.clone().sub(A);
      local.compose(A.clone().add(B).multiplyScalar(0.5), _q.setFromUnitVectors(UP, d.clone().normalize()), ONE);
      local.premultiply(frame).premultiply(base);
      batch.addGeometry(new THREE.CylinderGeometry(r, r, d.length(), 6), local, c);
    },
  };
}

/** One glyph per node type, about 0.6 units tall, standing on y = 0. */
const GLYPHS = {
  user(g) {
    g.cyl(0.085, 0.2, 0.3, 0, 0.15, 0);
    g.sph(0.125, 0, 0.45, 0, g.hot);
  },
  device(g) {
    g.frame.makeRotationX(-0.2);
    g.box(0.3, 0.54, 0.05, 0, 0.3, 0, g.dim);
    g.box(0.24, 0.42, 0.02, 0, 0.31, 0.03, g.hot);
    g.box(0.08, 0.016, 0.02, 0, 0.065, 0.033, g.main);
  },
  laptop(g) {
    g.box(0.6, 0.035, 0.4, 0, 0.02, 0.06, g.main);
    g.box(0.6, 0.38, 0.03, 0, 0.22, -0.16, g.dim, -0.22);
    g.box(0.52, 0.3, 0.012, 0, 0.225, -0.14, g.hot, -0.22);
  },
  browser(g) {
    g.frame.makeRotationX(-0.16);
    g.box(0.64, 0.46, 0.04, 0, 0.29, 0, g.dim);
    g.box(0.64, 0.09, 0.05, 0, 0.475, 0, g.main);
    g.box(0.4, 0.035, 0.02, -0.07, 0.35, 0.03, g.hot);
    g.box(0.5, 0.035, 0.02, -0.02, 0.27, 0.03, g.hot);
    g.box(0.3, 0.035, 0.02, -0.12, 0.19, 0.03, g.hot);
  },
  app(g) {
    g.frame.makeRotationX(-0.16);
    g.box(0.62, 0.48, 0.04, 0, 0.3, 0, g.dim);
    g.box(0.3, 0.1, 0.03, -0.11, 0.42, 0.03, g.hot);
    g.box(0.36, 0.1, 0.03, 0.08, 0.28, 0.03, g.main);
    g.box(0.24, 0.1, 0.03, -0.14, 0.14, 0.03, g.hot);
  },
  service(g) {
    for (let i = 0; i < 3; i++) {
      const y = 0.075 + i * 0.17;
      g.box(0.54, 0.13, 0.36, 0, y, 0, i === 2 ? g.main : g.dim);
      g.sph(0.027, -0.2, y, 0.185, g.hot);
      g.box(0.2, 0.022, 0.012, 0.1, y, 0.185, g.hot);
    }
  },
  model(g) {
    const cols = [2, 3, 2];
    const at = (l, i) => [(l - 1) * 0.24, 0.3 + (i - (cols[l] - 1) / 2) * 0.18, 0];
    for (let l = 0; l < 2; l++) for (let i = 0; i < cols[l]; i++) for (let j = 0; j < cols[l + 1]; j++) g.rod(at(l, i), at(l + 1, j), 0.011);
    for (let l = 0; l < 3; l++) for (let i = 0; i < cols[l]; i++) g.sph(0.058, ...at(l, i), l === 1 ? g.hot : g.main);
  },
  gpu(g) {
    g.frame.makeRotationX(-0.3);
    g.box(0.46, 0.46, 0.07, 0, 0.33, 0, g.dim);
    g.box(0.24, 0.24, 0.03, 0, 0.33, 0.045, g.hot);
    for (let i = -1; i <= 1; i++) {
      g.box(0.05, 0.07, 0.03, i * 0.13, 0.595, 0);
      g.box(0.05, 0.07, 0.03, i * 0.13, 0.065, 0);
      g.box(0.07, 0.05, 0.03, 0.265, 0.33 + i * 0.13, 0);
      g.box(0.07, 0.05, 0.03, -0.265, 0.33 + i * 0.13, 0);
    }
  },
  database(g) {
    for (let i = 0; i < 3; i++) {
      g.cyl(0.23, 0.23, 0.13, 0, 0.075 + i * 0.17, 0, i === 2 ? g.main : g.dim, 0, 0, 0, 22);
      g.torus(0.23, 0.012, 0, 0.14 + i * 0.17, 0, g.hot, Math.PI / 2);
    }
  },
  file(g) {
    g.frame.makeRotationX(-0.18);
    g.box(0.38, 0.5, 0.04, 0, 0.29, 0, g.main);
    for (let i = 0; i < 4; i++) g.box(i === 3 ? 0.14 : 0.25, 0.03, 0.012, i === 3 ? -0.085 : -0.03, 0.43 - i * 0.09, 0.025, g.hot);
  },
  table(g) {
    g.frame.makeRotationX(-0.18);
    const tones = [g.dim, g.main, g.hot];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) g.box(0.1, 0.1, 0.03, (c - 2) * 0.122, 0.11 + r * 0.122, 0, tones[(r * 7 + c * 5 + ((r * c) % 2)) % 3]);
  },
  image(g) {
    g.frame.makeRotationX(-0.18);
    g.box(0.58, 0.44, 0.03, 0, 0.3, 0, g.dim);
    g.sph(0.05, 0.15, 0.4, 0.03, g.hot);
    g.cone(0.17, 0.2, -0.09, 0.2, 0.03, g.main, 0, Math.PI / 4, 0, 4);
    g.cone(0.12, 0.14, 0.11, 0.17, 0.03, g.main, 0, Math.PI / 4, 0, 4);
  },
  tool(g) {
    g.frame.makeRotationX(-0.18);
    g.cyl(0.17, 0.17, 0.1, 0, 0.32, 0, g.main, Math.PI / 2, 0, 0, 18);
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      g.box(0.085, 0.085, 0.1, Math.cos(a) * 0.2, 0.32 + Math.sin(a) * 0.2, 0, g.main, 0, 0, a);
    }
    g.cyl(0.065, 0.065, 0.12, 0, 0.32, 0, g.hot, Math.PI / 2, 0, 0, 12);
  },
  gateway(g) {
    g.oct(0.19, 0, 0.33, 0, g.main, Math.PI / 4);
    g.cone(0.06, 0.13, 0.31, 0.33, 0, g.hot, 0, 0, -Math.PI / 2);
    g.cone(0.06, 0.13, -0.31, 0.33, 0, g.hot, 0, 0, Math.PI / 2);
    g.cone(0.06, 0.13, 0, 0.33, 0.31, g.hot, Math.PI / 2);
    g.cone(0.06, 0.13, 0, 0.33, -0.31, g.hot, -Math.PI / 2);
  },
  cloud(g) {
    g.sph(0.15, -0.17, 0.3, 0);
    g.sph(0.2, 0, 0.38, 0, g.hot);
    g.sph(0.14, 0.18, 0.3, 0);
    g.cyl(0.11, 0.11, 0.36, 0, 0.23, 0, g.main, 0, 0, Math.PI / 2);
  },
  globe(g) {
    g.sph(0.2, 0, 0.3, 0, g.dim);
    g.torus(0.205, 0.012, 0, 0.3, 0, g.hot, Math.PI / 2);
    g.torus(0.205, 0.012, 0, 0.3, 0, g.hot);
    g.torus(0.205, 0.012, 0, 0.3, 0, g.main, 0, Math.PI / 2);
  },
  layers(g) {
    for (let i = 0; i < 4; i++) g.box(0.54, 0.055, 0.36, 0, 0.06 + i * 0.13, 0, i === 3 ? g.hot : i % 2 ? g.main : g.dim);
  },
  sampler(g) {
    g.frame.makeRotationFromEuler(_e.set(0.42, 0.6, 0)).setPosition(0, 0.32, 0);
    g.box(0.32, 0.32, 0.32, 0, 0, 0, g.main);
    for (const [x, y] of [[-0.08, 0.08], [0, 0], [0.08, -0.08]]) g.sph(0.034, x, y, 0.16, g.hot);
    for (const [x, z] of [[-0.07, -0.07], [0.07, 0.07]]) g.sph(0.034, x, 0.16, z, g.hot);
    g.sph(0.034, 0.16, 0, 0, g.hot);
  },
  queue(g) {
    const tones = [g.dim, g.dim, g.main, g.hot];
    for (let i = 0; i < 4; i++) g.box(0.1, 0.24, 0.2, -0.27 + i * 0.14, 0.2, 0, tones[i]);
    g.cone(0.07, 0.13, 0.33, 0.2, 0, g.hot, 0, 0, -Math.PI / 2);
  },
  op(g) {
    g.frame.makeRotationX(-0.18);
    g.cyl(0.22, 0.22, 0.1, 0, 0.31, 0, g.dim, Math.PI / 2, 0, 0, 6);
    g.cone(0.1, 0.17, 0.02, 0.31, 0.07, g.hot, 0, 0, -Math.PI / 2, 3);
  },
  agent(g) {
    g.box(0.4, 0.28, 0.28, 0, 0.2, 0, g.main);
    g.sph(0.042, -0.09, 0.22, 0.14, g.hot);
    g.sph(0.042, 0.09, 0.22, 0.14, g.hot);
    g.box(0.04, 0.1, 0.1, 0.22, 0.2, 0, g.dim);
    g.box(0.04, 0.1, 0.1, -0.22, 0.2, 0, g.dim);
    g.cyl(0.012, 0.012, 0.14, 0, 0.41, 0, g.dim);
    g.sph(0.036, 0, 0.5, 0, g.hot);
  },
  mic(g) {
    g.cyl(0.085, 0.085, 0.16, 0, 0.43, 0, g.main);
    g.sph(0.085, 0, 0.51, 0, g.hot);
    g.sph(0.085, 0, 0.35, 0, g.main);
    g.torus(0.125, 0.014, 0, 0.37, 0, g.hot, 0, 0, Math.PI, Math.PI);
    g.cyl(0.014, 0.014, 0.21, 0, 0.14, 0, g.dim);
    g.cyl(0.11, 0.11, 0.03, 0, 0.025, 0, g.dim);
  },
  scaler(g) {
    g.box(0.06, 0.22, 0.06, -0.12, 0.17, 0, g.main);
    g.cone(0.11, 0.17, -0.12, 0.36, 0, g.hot);
    g.box(0.06, 0.22, 0.06, 0.12, 0.34, 0, g.dim);
    g.cone(0.11, 0.17, 0.12, 0.15, 0, g.main, Math.PI);
  },
  lock(g) {
    g.box(0.32, 0.24, 0.14, 0, 0.14, 0, g.main);
    g.torus(0.1, 0.028, 0, 0.26, 0, g.hot, 0, 0, 0, Math.PI);
    g.sph(0.035, 0, 0.15, 0.07, g.hot);
  },
};
GLYPHS.client = GLYPHS.device;
GLYPHS.store = GLYPHS.database;

// ------------------------------------------------------------------ layout helpers

/** Position helpers. Each returns a map id -> [x, z] that can be merged into node specs with `place`. */
export const layout = {
  line(ids, { from = [0, 0], step = [4.4, 0] } = {}) {
    return Object.fromEntries(ids.map((id, i) => [id, [from[0] + step[0] * i, from[1] + step[1] * i]]));
  },
  /** Rows that alternate direction, so consecutive ids stay neighbors. */
  snake(ids, { perRow = 4, dx = 4.4, dz = 5, center = [0, 0] } = {}) {
    const rows = Math.ceil(ids.length / perRow);
    return Object.fromEntries(
      ids.map((id, i) => {
        const r = Math.floor(i / perRow);
        const c = r % 2 ? perRow - 1 - (i % perRow) : i % perRow;
        return [id, [center[0] + (c - (perRow - 1) / 2) * dx, center[1] + (r - (rows - 1) / 2) * dz]];
      }),
    );
  },
  ring(ids, { radius = 6, start = -Math.PI / 2, sweep = Math.PI * 2, center = [0, 0] } = {}) {
    const n = sweep >= Math.PI * 2 - 1e-6 ? ids.length : Math.max(1, ids.length - 1);
    return Object.fromEntries(ids.map((id, i) => [id, [center[0] + Math.cos(start + (i / n) * sweep) * radius, center[1] + Math.sin(start + (i / n) * sweep) * radius]]));
  },
  grid(ids, { cols = 3, dx = 4.4, dz = 4.4, center = [0, 0] } = {}) {
    const rows = Math.ceil(ids.length / cols);
    return Object.fromEntries(ids.map((id, i) => [id, [center[0] + ((i % cols) - (cols - 1) / 2) * dx, center[1] + (Math.floor(i / cols) - (rows - 1) / 2) * dz]]));
  },
  /** Copy positions from a layout map into node specs that have no `pos` yet. */
  place(nodes, map) {
    return nodes.map((n) => (n.pos || !map[n.id] ? n : { ...n, pos: map[n.id] }));
  },
};

/** Legend entries (for ui.legend) for the roles a graph spec actually uses. */
export function roleLegend(spec) {
  const used = new Set(spec.nodes.map((n) => n.role ?? NODE_TYPES[n.type] ?? 'service'));
  return Object.entries(ROLES)
    .filter(([id]) => used.has(id))
    .map(([, r]) => ({ color: r.color, label: r.label }));
}

/**
 * Animated version of app.setView({fit}): the engine works out the camera distance that frames a
 * sphere of radius `view.fit` between the panels, then the camera glides there. The view also
 * becomes the "home" pose, so Reset view and window resizes keep the current stage in frame.
 */
export function flyFit(app, view, dur = 1.2) {
  const cam = app.camera.position.clone();
  const tgt = app.controls.target.clone();
  app.setView(view);
  const dest = app.camera.position.clone();
  app.camera.position.copy(cam);
  app.controls.target.copy(tgt);
  return app.flyTo(dest, view.target, dur);
}

const waveValue = (w, t) => {
  // 1 inside the [from, to] window of each period, `low` outside, with short ramps
  const ph = (((t + (w.phase ?? 0)) % w.period) + w.period) % w.period / w.period;
  const r = w.ramp ?? 0.06;
  const up = clamp((ph - w.from) / r);
  const down = clamp((w.to - ph) / r);
  return (w.low ?? 0.25) + (1 - (w.low ?? 0.25)) * Math.min(up, down);
};

// ------------------------------------------------------------------ the graph

/**
 * @typedef {{id:string, type:string, label:string, pos:number[], size?:number, role?:string, color?:string, tip?:string,
 *            wave?:{period:number, from:number, to:number, low?:number, phase?:number}}} NodeSpec
 * @typedef {{id:string, from:string, to:string, label?:string, bend?:number, lift?:number, dashed?:boolean, both?:boolean, color?:string}} EdgeSpec
 * @typedef {{id:string, label?:string, nodes?:string[], rect?:number[], pad?:number, color?:string}} ZoneSpec
 * @typedef {{path:string|string[], color?:string, size?:number, count?:number, speed?:number, gap?:number, cycle?:number[], phase?:number}} FlowSpec
 */
export class FlowGraph {
  /**
   * @param {object} ctx chapter context (for picking)
   * @param {{nodes: NodeSpec[], edges?: EdgeSpec[], zones?: ZoneSpec[]}} spec
   * @param {{accent?:string, labels?:boolean, labelSize?:number, packets?:number, pick?:boolean,
   *          onClick?:(node:object)=>void, onHover?:(node:object|null)=>void}} [o]
   */
  constructor(ctx, spec, o = {}) {
    this.ctx = ctx;
    this.o = { accent: '#4de3b0', labels: true, labelSize: 0.34, packets: 96, pick: true, ...o };
    this.group = new THREE.Group();
    this.time = 0;
    this.nodes = new Map();
    this.edges = new Map();
    this.nodeList = [];
    this.edgeList = [];
    this.zones = [];
    this.flows = [];
    this.sends = [];
    this.exhibits = new Map();
    this.exList = [];
    this.intensity = 1;
    this._edgeLabels = 'active';
    this._labelsOn = false;
    this._used = 0;
    this._build(spec);
    if (this.o.labels) this.showLabels(true);
  }

  _build(spec) {
    const tri = (this.tri = new ColorBatch());
    const lin = (this.lin = new ColorBatch());

    for (const s of spec.nodes) {
      const role = s.role ?? NODE_TYPES[s.type] ?? 'service';
      const col = new THREE.Color(s.color ?? ROLES[role].color);
      const size = s.size ?? 1;
      const n = { ...s, i: this.nodeList.length, role, col, css: s.color ?? ROLES[role].color, size, x: s.pos[0], z: s.pos[s.pos.length - 1], hw: (BW / 2) * size, hd: (BD / 2) * size, top: BH * size, k: 1, kT: 1, applied: -1, hover: 0, label3d: null };
      this.nodes.set(n.id, n);
      this.nodeList.push(n);
      // outline: top and bottom rims plus four corner posts
      const oc = glow(col, 1.7);
      const oc2 = glow(col, 0.8);
      const rp = roundRectPts(BW * size + 0.02, BD * size + 0.02, BR * size, 4);
      n.l0 = lin.count;
      for (let j = 0; j < rp.length; j++) {
        const p = rp[j];
        const q = rp[(j + 1) % rp.length];
        lin.addSegment(_a.set(n.x + p[0], n.top + 0.004, n.z + p[1]), _b.set(n.x + q[0], n.top + 0.004, n.z + q[1]), oc);
        lin.addSegment(_a.set(n.x + p[0], 0.012, n.z + p[1]), _b.set(n.x + q[0], 0.012, n.z + q[1]), oc2);
      }
      const cx = n.hw - BR * size * 0.29;
      const cz = n.hd - BR * size * 0.29;
      for (const [sx, sz] of [[1, 1], [-1, 1], [-1, -1], [1, -1]]) lin.addSegment(_a.set(n.x + sx * cx, 0.012, n.z + sz * cz), _b.set(n.x + sx * cx, n.top, n.z + sz * cz), oc2);
      n.lN = lin.count - n.l0;
      // glyph
      n.g0 = tri.count;
      (GLYPHS[s.type] || GLYPHS.op)(makeKit(tri, new THREE.Vector3(n.x, n.top, n.z), 1.12 * size, col));
      n.gN = tri.count - n.g0;
    }

    const edgeCol = glow(this.o.accent, 1);
    for (const s of spec.edges ?? []) {
      const from = this.nodes.get(s.from);
      const to = this.nodes.get(s.to);
      if (!from || !to) {
        console.warn('[flowgraph] edge', s.id, 'references a missing node');
        continue;
      }
      const e = { ...s, i: this.edgeList.length, from, to, k: EDGE.normal, kT: EDGE.normal, applied: -1, lo: 0, loT: 0, active: false, label3d: null };
      this._edgeGeometry(e, tri, s.color ? glow(s.color, 1) : edgeCol);
      this.edges.set(e.id, e);
      this.edgeList.push(e);
    }

    for (const s of spec.zones ?? []) {
      let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      const pad = s.pad ?? 0.85;
      if (s.rect) [x0, z0, x1, z1] = s.rect;
      else
        for (const id of s.nodes) {
          const n = this.nodes.get(id);
          if (!n) continue;
          x0 = Math.min(x0, n.x - n.hw - pad);
          x1 = Math.max(x1, n.x + n.hw + pad);
          z0 = Math.min(z0, n.z - n.hd - pad);
          z1 = Math.max(z1, n.z + n.hd + pad + 0.5);
        }
      const zc = new THREE.Color(s.color ?? '#8fa3c8');
      const w = x1 - x0;
      const d = z1 - z0;
      const rp = roundRectPts(w, d, Math.min(0.6, w / 4, d / 4), 6);
      const z = { ...s, x0, x1, z0, z1, l0: lin.count, label3d: null };
      const lc = glow(zc, s.strong ? 1.5 : 0.85);
      for (let j = 0; j < rp.length; j++) {
        if (s.dashed && j % 2) continue;
        const p = rp[j];
        const q = rp[(j + 1) % rp.length];
        lin.addSegment(_a.set(x0 + w / 2 + p[0], 0.02, z0 + d / 2 + p[1]), _b.set(x0 + w / 2 + q[0], 0.02, z0 + d / 2 + q[1]), lc);
      }
      z.lN = lin.count - z.l0;
      const shape = new THREE.Shape(rp.map(([px, py]) => new THREE.Vector2(px, py)));
      const fill = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({ color: zc, transparent: true, opacity: s.fill ?? 0.07, depthWrite: false, side: THREE.DoubleSide }));
      fill.rotation.x = -Math.PI / 2;
      fill.position.set(x0 + w / 2, 0.008, z0 + d / 2);
      fill.scale.y = -1; // the shape was built in (x, z); rotating flips z
      fill.renderOrder = -2;
      z.fill = fill;
      this.group.add(fill);
      this.zones.push(z);
    }

    const bodies = (this.bodies = instanced(rbox(BW, BH, BD, BR, 3), new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4, metalness: 0.2 }), Math.max(1, this.nodeList.length)));
    for (const n of this.nodeList) setInst(bodies, n.i, n.x, n.top / 2, n.z, n.size);
    flushInst(bodies);
    this.group.add(bodies);

    if (tri.count) {
      this.triMesh = new THREE.Mesh(tri.finish(), new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
      this.group.add(this.triMesh);
    }
    if (lin.count) {
      this.linMesh = new THREE.LineSegments(lin.finish(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
      this.group.add(this.linMesh);
    }

    this.packets = instanced(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, fog: false }), this.o.packets);
    for (let i = 0; i < this.o.packets; i++) setInst(this.packets, i, 0, -50, 0, 0);
    flushInst(this.packets);
    this.group.add(this.packets);

    this.focusRing = ring(1, this.o.accent, { k: 2.4, opacity: 0.95, segments: 72 });
    this.focusRing.material.transparent = true;
    this.focusRing.visible = false;
    this.focusHalo = halo(this.o.accent, 4.4, 0.0);
    this.focusHalo.visible = false;
    this.exGroup = new THREE.Group();
    this.group.add(this.focusRing, this.focusHalo, this.exGroup);

    if (this.o.pick && this.nodeList.length) {
      const handlers = {
        hover: (hit) => this._setHover(this.nodeList[hit.instanceId] ?? null),
        out: () => this._setHover(null),
        tip: (hit) => {
          const n = this.nodeList[hit.instanceId];
          if (!n) return null;
          const name = n.label.replace(/\n/g, ' ');
          return n.tip ? `${name}\n${n.tip}` : name;
        },
      };
      if (this.o.onClick) handlers.click = (hit) => this.nodeList[hit.instanceId] && this.o.onClick(this.nodeList[hit.instanceId]);
      this._unpick = this.ctx.pick.add(bodies, handlers);
    }
  }

  _edgeGeometry(e, tri, color) {
    const A = e.from;
    const B = e.to;
    const y = 0.26;
    const dx = B.x - A.x;
    const dz = B.z - A.z;
    const len = Math.hypot(dx, dz) || 1;
    const px = dz / len; // right-hand side of the direction of travel
    const pz = -dx / len;
    const bend = (e.bend ?? 0) * len * 2;
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(A.x, y, A.z),
      new THREE.Vector3((A.x + B.x) / 2 + px * bend, y + (e.lift ?? 0) * 2, (A.z + B.z) / 2 + pz * bend),
      new THREE.Vector3(B.x, y, B.z),
    );
    const inside = (n, p, m) => Math.abs(p.x - n.x) < n.hw + m && Math.abs(p.z - n.z) < n.hd + m && p.y < n.top + m;
    const N = 120;
    let t0 = 0;
    let t1 = 1;
    for (let i = 0; i <= N; i++)
      if (!inside(A, curve.getPoint(i / N, _v), 0.05)) {
        t0 = i / N;
        break;
      }
    for (let i = N; i >= 0; i--)
      if (!inside(B, curve.getPoint(i / N, _v), 0.08)) {
        t1 = i / N;
        break;
      }
    if (t1 <= t0) {
      t0 = 0.3;
      t1 = 0.7;
    }
    // resample by arc length so packets move at constant speed
    const D = 72;
    const dense = new Float32Array((D + 1) * 3);
    const cum = new Float32Array(D + 1);
    for (let i = 0; i <= D; i++) {
      curve.getPoint(t0 + (t1 - t0) * (i / D), _v).toArray(dense, i * 3);
      if (i) cum[i] = cum[i - 1] + Math.hypot(dense[i * 3] - dense[i * 3 - 3], dense[i * 3 + 1] - dense[i * 3 - 2], dense[i * 3 + 2] - dense[i * 3 - 1]);
    }
    const L = cum[D];
    const pts = new Float32Array((K + 1) * 3);
    let j = 0;
    for (let i = 0; i <= K; i++) {
      const d = (i / K) * L;
      while (j < D - 1 && cum[j + 1] < d) j++;
      const a = (d - cum[j]) / Math.max(1e-6, cum[j + 1] - cum[j]);
      for (let c = 0; c < 3; c++) pts[i * 3 + c] = dense[j * 3 + c] + (dense[j * 3 + 3 + c] - dense[j * 3 + c]) * a;
    }
    e.pts = pts;
    e.len = L;
    e.mid = new THREE.Vector3(pts[(K / 2) * 3], pts[(K / 2) * 3 + 1], pts[(K / 2) * 3 + 2]);

    e.g0 = tri.count;
    const hl = Math.min(0.3, L * 0.3);
    const skip = Math.max(1, Math.ceil(((hl * 0.8) / L) * K));
    addTube(tri, pts, K + 1, e.both ? skip : 0, K - skip, 0.024, color, e.dashed);
    const head = (iTip, iBack) => {
      _a.fromArray(pts, iTip * 3);
      _t.fromArray(pts, iBack * 3).sub(_a).negate().normalize();
      _m.compose(_b.copy(_a).addScaledVector(_t, -hl / 2), _q.setFromUnitVectors(UP, _t), ONE);
      tri.addGeometry(new THREE.ConeGeometry(0.095, hl, 10), _m, color, 0.3);
    };
    head(K, K - 1);
    if (e.both) head(0, 1);
    e.gN = tri.count - e.g0;
  }

  _makeLabels() {
    if (this._labelsBuilt) return;
    this._labelsBuilt = true;
    this.labelGroup = new THREE.Group();
    for (const n of this.nodeList) {
      const s = textSprite(n.label, { size: this.o.labelSize * (0.8 + 0.2 * n.size), font: 'display', weight: 650, color: '#eef3ff', bg: 'rgba(6,9,18,0.74)', anchor: [0.5, 1], lineHeight: 1.12 });
      s.position.set(n.x, 0.1, n.z + n.hd + 0.14);
      n.label3d = s;
      this.labelGroup.add(s);
    }
    for (const e of this.edgeList) {
      if (!e.label) continue;
      const s = textSprite(e.label, { size: this.o.labelSize * 0.68, font: 'mono', weight: 600, color: '#dbe6fb', bg: 'rgba(6,9,18,0.9)', border: withAlpha(this.o.accent, 0.6) });
      s.position.copy(e.mid).y += 0.5 + (e.labelLift ?? 0);
      s.visible = false;
      s.material.opacity = 0;
      e.label3d = s;
      this.labelGroup.add(s);
    }
    for (const z of this.zones) {
      if (!z.label) continue;
      const s = textSprite(z.label.toUpperCase(), { size: this.o.labelSize * 0.6, font: 'mono', weight: 600, color: '#a9b6d3', anchor: [0, 1] });
      s.position.set(z.x0 + 0.3, 0.05, z.z1 + 0.12);
      z.label3d = s;
      this.labelGroup.add(s);
    }
    this.group.add(this.labelGroup);
    for (const n of this.nodeList) n.applied = -1;
  }

  /** Show or hide every text label of the graph (labels are created on first use). */
  showLabels(on = true) {
    if (on) this._makeLabels();
    this._labelsOn = on;
    if (this.labelGroup) this.labelGroup.visible = on;
    this._syncEdgeLabels();
  }

  /** Which edge labels are visible: 'active' (default), 'all' or 'none'. */
  showEdgeLabels(mode = 'active') {
    this._edgeLabels = mode;
    this._syncEdgeLabels();
  }

  _syncEdgeLabels() {
    for (const e of this.edgeList) e.loT = this._labelsOn && e.label && (this._edgeLabels === 'all' || (this._edgeLabels === 'active' && e.active)) ? 1 : 0;
  }

  _setHover(n) {
    if (this._hover === n) return;
    if (this._hover) this._hover.hover = 0;
    this._hover = n;
    if (n) n.hover = 1;
    this.o.onHover?.(n);
  }

  /**
   * Highlight some nodes. `ids` = null restores every node to normal.
   * @param {string[]|null} ids
   * @param {{dim?: boolean, ring?: string|false}} [o] dim: fade the other nodes (default true); ring: node that gets the pulsing ring (default: the first id)
   */
  focus(ids, o = {}) {
    const set = ids ? new Set(ids) : null;
    for (const n of this.nodeList) n.kT = !set ? NODE.normal : set.has(n.id) ? NODE.focus : o.dim === false ? NODE.normal : NODE.dim;
    const r = o.ring === false ? null : this.nodes.get(o.ring ?? ids?.[0]);
    this._ringNode = r ?? null;
    this.focusRing.visible = this.focusHalo.visible = !!r;
    if (r) {
      this.focusRing.position.set(r.x, 0.03, r.z);
      this.focusHalo.position.set(r.x, 0.5, r.z);
    }
  }

  /** Highlight some edges (their wire-format labels appear). `ids` = null restores all of them. */
  setActiveEdges(ids, o = {}) {
    const set = ids ? new Set(ids) : null;
    for (const e of this.edgeList) {
      e.active = !!set && set.has(e.id);
      e.kT = !set ? EDGE.normal : e.active ? EDGE.active : o.dim === false ? EDGE.normal : EDGE.dim;
    }
    this._syncEdgeLabels();
  }

  /** Overall brightness of the graph (1 = normal). Used to push inactive dioramas into the background. */
  setIntensity(k) {
    this.intensity = k;
    for (const n of this.nodeList) n.applied = -1;
    for (const e of this.edgeList) e.applied = -1;
  }

  _path(path) {
    const segs = [];
    let L = 0;
    for (const raw of Array.isArray(path) ? path : [path]) {
      const rev = raw[0] === '-';
      const e = this.edges.get(rev ? raw.slice(1) : raw);
      if (!e) continue;
      segs.push({ e, rev, len: e.len });
      L += e.len;
    }
    return { segs, L };
  }

  _packetColor(c) {
    const css = c === 'response' || c == null ? this.o.accent : c === 'request' ? '#ffffff' : c;
    return { color: glow(css, c === 'request' ? 2.3 : 2.6), color2: glow(css, 1.1) };
  }

  /**
   * Continuous packet streams. path: edge id or list of ids (prefix '-' to travel an edge backward).
   * Without `cycle` the packets loop forever; with cycle [period, from, to] a train of `count` packets
   * crosses the path once per period, during that fraction of it (so a response can follow a request).
   * @param {FlowSpec[]} list
   */
  setFlows(list) {
    this.flows = [];
    for (const f of list ?? []) {
      const p = this._path(f.path);
      if (!p.segs.length) continue;
      this.flows.push({ ...p, ...this._packetColor(f.color), size: f.size ?? 0.085, count: f.count ?? 3, speed: f.speed ?? 2.4, gap: f.gap ?? 0.5, cycle: f.cycle ?? null, phase: f.phase ?? 0 });
    }
  }

  /**
   * Fire one train of packets along an edge or a path of edges. Resolves when it has arrived.
   * @param {string|string[]} path
   * @param {{color?:string, size?:number, count?:number, gap?:number, speed?:number}} [o]
   */
  send(path, o = {}) {
    const p = this._path(path);
    if (!p.segs.length) return Promise.resolve(false);
    const count = o.count ?? 1;
    const gap = o.gap ?? 0.45;
    const dur = (p.L + (count - 1) * gap) / (o.speed ?? 5);
    return new Promise((resolve) => this.sends.push({ ...p, ...this._packetColor(o.color ?? 'request'), size: o.size ?? 0.11, count, gap, t0: this.time, dur, resolve }));
  }

  /** Alias of send for a multi-edge path. */
  play(path, o) {
    return this.send(path, o);
  }

  _pathPoint(f, d, out) {
    const segs = f.segs;
    for (let i = 0; i < segs.length; i++) {
      const s = segs[i];
      if (d <= s.len || i === segs.length - 1) {
        let u = clamp(d / s.len);
        if (s.rev) u = 1 - u;
        const pts = s.e.pts;
        const x = u * K;
        const k = Math.min(K - 1, Math.floor(x));
        const a = x - k;
        const o = k * 3;
        return out.set(pts[o] + (pts[o + 3] - pts[o]) * a, pts[o + 1] + (pts[o + 4] - pts[o + 1]) * a, pts[o + 2] + (pts[o + 5] - pts[o + 2]) * a);
      }
      d -= s.len;
    }
    return out;
  }

  _place(f, d, idx) {
    const max = this.o.packets;
    const fade = Math.max(0, Math.min(1, d / 0.3, (f.L - d) / 0.3));
    const s = f.size * (0.3 + 0.7 * fade) * (0.55 + 0.45 * Math.min(1, this.intensity));
    for (let h = 0; h < 3; h++) {
      const dd = d - h * f.size * 2.2;
      if (dd < 0 || idx >= max) break;
      this._pathPoint(f, dd, _v);
      const sc = s * (h === 0 ? 1 : h === 1 ? 0.66 : 0.4);
      setInst(this.packets, idx++, _v.x, _v.y, _v.z, sc, sc, sc, h === 0 ? f.color : f.color2);
    }
    return idx;
  }

  _train(f, prog, idx) {
    const head = prog * (f.L + (f.count - 1) * f.gap);
    for (let j = 0; j < f.count; j++) {
      const d = head - j * f.gap;
      if (d >= 0 && d <= f.L) idx = this._place(f, d, idx);
    }
    return idx;
  }

  _applyNode(n) {
    const k = n.k * this.intensity;
    _c.copy(BODY).multiplyScalar(0.4 + 0.6 * Math.min(1, k)).add(_c2.copy(n.col).multiplyScalar(0.05 + 0.13 * k));
    this.bodies.setColorAt(n.i, _c);
    this.tri.tint(n.g0, n.gN, 0.1 + 0.9 * Math.min(k, 1) + Math.max(0, k - 1) * 0.45);
    this.lin.tint(n.l0, n.lN, 0.12 + 0.88 * k);
    if (n.label3d) n.label3d.material.opacity = clamp(0.25 + 0.75 * n.k);
    n.applied = n.k;
  }

  /** Call every frame. */
  update(dt, t, camera = this.ctx.app.camera) {
    this.time += dt;
    const T = this.time;
    const nodes = this.nodeList;
    let dirty = false;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      let target = n.kT + n.hover * 0.5;
      if (n.wave) target *= waveValue(n.wave, T);
      n.k = Math.abs(n.k - target) < 0.004 ? target : damp(n.k, target, 0.11, dt);
      if (Math.abs(n.k - n.applied) > 0.003) {
        this._applyNode(n);
        dirty = true;
      }
    }
    if (dirty && this.bodies.instanceColor) this.bodies.instanceColor.needsUpdate = true;

    const edges = this.edgeList;
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      e.k = Math.abs(e.k - e.kT) < 0.004 ? e.kT : damp(e.k, e.kT, 0.11, dt);
      if (Math.abs(e.k - e.applied) > 0.003) {
        this.tri.tint(e.g0, e.gN, e.k * (0.35 + 0.65 * Math.min(1, this.intensity)));
        e.applied = e.k;
      }
      if (e.label3d && Math.abs(e.lo - e.loT) > 0.004) {
        e.lo = damp(e.lo, e.loT, 0.1, dt);
        if (Math.abs(e.lo - e.loT) < 0.01) e.lo = e.loT;
        e.label3d.material.opacity = e.lo;
        e.label3d.visible = e.lo > 0.02;
      }
    }

    const r = this._ringNode;
    if (r) {
      const pulse = 0.5 + 0.5 * Math.sin(T * 2.6);
      this.focusRing.scale.set(r.hw * (1.38 + 0.1 * pulse), 1, r.hd * (1.62 + 0.14 * pulse));
      this.focusRing.material.opacity = 0.45 + 0.5 * (1 - pulse);
      this.focusHalo.material.opacity = 0.16 + 0.08 * pulse;
    }

    // packets
    let idx = 0;
    const flows = this.flows;
    for (let i = 0; i < flows.length; i++) {
      const f = flows[i];
      if (f.cycle) {
        const ph = ((T + f.phase) % f.cycle[0]) / f.cycle[0];
        if (ph >= f.cycle[1] && ph <= f.cycle[2]) idx = this._train(f, (ph - f.cycle[1]) / (f.cycle[2] - f.cycle[1]), idx);
      } else {
        const spacing = f.L / f.count;
        const head = (T * f.speed + f.phase * f.L) % f.L;
        for (let j = 0; j < f.count; j++) {
          let d = head + j * spacing;
          if (d >= f.L) d -= f.L;
          idx = this._place(f, d, idx);
        }
      }
    }
    const sends = this.sends;
    for (let i = sends.length - 1; i >= 0; i--) {
      const s = sends[i];
      const prog = (T - s.t0) / s.dur;
      if (prog >= 1) {
        sends.splice(i, 1);
        s.resolve(true);
      } else idx = this._train(s, prog, idx);
    }
    for (let i = idx; i < this._used; i++) setInst(this.packets, i, 0, -50, 0, 0);
    if (idx || this._used) flushInst(this.packets);
    this._used = idx;

    // exhibits: turn toward the camera, ease in and out
    const list = this.exList;
    if (list.length) {
      this.group.worldToLocal(_a.copy(camera.position));
      for (let i = list.length - 1; i >= 0; i--) {
        const ex = list[i];
        ex.s = damp(ex.s, ex.sT, 0.1, dt);
        ex.inner.scale.setScalar(Math.max(0.001, ex.s));
        ex.inner.position.y = this.ctx.reduceMotion ? 0 : Math.sin(T * 1.1 + ex.phase) * 0.035;
        const yaw = Math.atan2(_a.x - ex.group.position.x, _a.z - ex.group.position.z);
        if (ex.fresh) {
          ex.group.rotation.y = yaw;
          ex.fresh = false;
        } else {
          const d = ((((yaw - ex.group.rotation.y + Math.PI) % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2)) - Math.PI;
          ex.group.rotation.y += d * (1 - Math.exp(-dt / 0.18));
        }
        if (Math.abs(ex.op - ex.opT) > 0.004) {
          ex.op = damp(ex.op, ex.opT, 0.1, dt);
          if (Math.abs(ex.op - ex.opT) < 0.01) ex.op = ex.opT;
          for (let m = 0; m < ex.mats.length; m++) ex.mats[m].opacity = ex.mats[m].userData.o * ex.op;
        }
        ex.update?.(dt, T);
        if (ex.dying && ex.s < 0.04) {
          list.splice(i, 1);
          this.exGroup.remove(ex.group);
          disposeDeep(ex.group);
        }
      }
    }
  }

  /**
   * Declare which exhibits float above which nodes. Exhibits whose node and spec are unchanged are
   * kept (so a main exhibit can shrink into a "before" ghost without being rebuilt).
   * @param {{node:string, spec:object, ghost?:boolean, lift?:number}[]} list
   */
  setExhibits(list) {
    const keep = new Set();
    for (const it of list ?? []) {
      if (!it.spec || !this.nodes.has(it.node)) continue;
      keep.add(it.node);
      let ex = this.exhibits.get(it.node);
      if (!ex || ex.spec !== it.spec) {
        if (ex) this._retire(ex);
        ex = this._mount(it);
      }
      ex.ghost = !!it.ghost;
      ex.sT = it.ghost ? 0.52 : 1;
      ex.opT = it.ghost ? 0.42 : 1;
    }
    for (const [id, ex] of this.exhibits)
      if (!keep.has(id)) {
        this._retire(ex);
        this.exhibits.delete(id);
      }
  }

  /** Show one exhibit above a node (spec = null removes it). */
  exhibit(node, spec, o = {}) {
    const rest = [...this.exhibits.values()].filter((ex) => ex.nodeId !== node).map((ex) => ({ node: ex.nodeId, spec: ex.spec, ghost: ex.ghost }));
    this.setExhibits(spec ? [...rest, { node, spec, ...o }] : rest);
    return this.exhibits.get(node) ?? null;
  }

  clearExhibits() {
    this.setExhibits([]);
  }

  _retire(ex) {
    ex.dying = true;
    ex.sT = 0;
  }

  _mount(it) {
    const n = this.nodes.get(it.node);
    const ex = buildExhibit(it.spec, { accent: this.o.accent });
    ex.spec = it.spec;
    ex.nodeId = it.node;
    ex.y0 = n.top + (it.lift ?? 1.35);
    ex.group.position.set(n.x, ex.y0, n.z);
    ex.s = 0.2;
    ex.sT = 1;
    ex.op = 1;
    ex.opT = 1;
    ex.fresh = true;
    ex.phase = n.i * 1.7;
    const tether = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -(ex.y0 - n.top - 0.72), 0), new THREE.Vector3(0, 0, 0)]), new THREE.LineBasicMaterial({ color: glow(this.o.accent, 1.5), transparent: true, opacity: 0.55, depthWrite: false }));
    ex.group.add(tether);
    ex.mats = [];
    ex.group.traverse((obj) => {
      if (!obj.material) return;
      obj.material.transparent = true;
      obj.material.userData.o = obj.material.opacity;
      ex.mats.push(obj.material);
    });
    this.exGroup.add(ex.group);
    this.exhibits.set(it.node, ex);
    this.exList.push(ex);
    return ex;
  }

  /** World position of a node (center of its top face). */
  nodeWorld(id, out = new THREE.Vector3()) {
    const n = this.nodes.get(id);
    return n ? this.group.localToWorld(out.set(n.x, n.top, n.z)) : out;
  }

  /** Bounding sphere (world space) of some nodes (default: all), including the exhibits above them. */
  bounds(ids) {
    const list = ids ? ids.map((id) => this.nodes.get(id)).filter(Boolean) : this.nodeList;
    const box = new THREE.Box3();
    for (const n of list) {
      box.expandByPoint(_v.set(n.x - n.hw, 0, n.z - n.hd));
      box.expandByPoint(_v.set(n.x + n.hw, n.top + 0.75, n.z + n.hd + 0.5));
      const ex = this.exhibits.get(n.id);
      if (ex && !ex.dying) {
        const hw = (ex.W / 2) * ex.sT;
        box.expandByPoint(_v.set(n.x - hw, ex.y0, n.z));
        box.expandByPoint(_v.set(n.x + hw, ex.y0 + ex.H * ex.sT, n.z));
      }
    }
    if (!ids) for (const z of this.zones) box.expandByPoint(_v.set(z.x0, 0, z.z0)).expandByPoint(_v.set(z.x1, 0, z.z1));
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(_v).length() / 2;
    this.group.updateWorldMatrix(true, false);
    this.group.localToWorld(center);
    return { center, radius };
  }

  /**
   * A camera view (for app.setView / flyFit) that frames some nodes.
   * @param {string[]} [ids] @param {{dir?:number[], pad?:number, minFit?:number, local?:boolean}} [o]
   *        dir: viewing direction (from the target toward the camera); local: dir is given in the graph's own frame
   */
  view(ids, o = {}) {
    const { center, radius } = this.bounds(ids);
    const d = v3(o.dir ?? [0.26, 0.58, 1]).normalize();
    if (o.local) d.applyQuaternion(this.group.getWorldQuaternion(_q));
    return { pos: center.clone().addScaledVector(d, 10).toArray(), target: center.toArray(), fit: Math.max(o.minFit ?? 2.4, radius * (o.pad ?? 1)) };
  }

  dispose() {
    this._unpick?.();
    this.group.parent?.remove(this.group);
    disposeDeep(this.group);
    for (const s of this.sends) s.resolve(false);
    this.sends.length = 0;
  }
}

export function createFlowGraph(ctx, spec, o) {
  return new FlowGraph(ctx, spec, o);
}

// ------------------------------------------------------------------ exhibits

const U = 150; // canvas pixels per world unit on exhibit cards
const INK = '#e9eefb';
const INK2 = '#b4bfd6';
const INK3 = '#8390ac';
const CODE = { key: '#9ec5f4', str: '#a8e0c4', num: '#f0c27a', bool: '#d9a6f0', special: '#8fb7ff', cite: '#f0c27a' };
/** Decorative tints that tell neighboring tokens / rows / requests apart (text stays neutral). */
export const TINTS = [SERIES[0], SERIES[2], SERIES[6], SERIES[3], SERIES[4]];
const CHIP = new THREE.Color('#141c30');
const EMPTY = new THREE.Color(0.028, 0.04, 0.075);

const F = {
  title: (px) => `650 ${px}px ${FONTS.display}`,
  body: (px, w = 560) => `${w} ${px}px ${FONTS.body}`,
  mono: (px, w = 520) => `${w} ${px}px ${FONTS.mono}`,
};

let _mctx = null;
function measure(text, font) {
  _mctx ??= document.createElement('canvas').getContext('2d');
  _mctx.font = font;
  return _mctx.measureText(text).width;
}

function clip(text, font, maxPx) {
  if (measure(text, font) <= maxPx) return text;
  let s = text;
  while (s.length > 1 && measure(s + '…', font) > maxPx) s = s.slice(0, -1);
  return s + '…';
}

function canvasPlane(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = Math.max(2, Math.round(w * U));
  c.height = Math.max(2, Math.round(h * U));
  const g2 = c.getContext('2d');
  // a hair of letter spacing switches off font ligatures, so "<|eot_id|>" and "->" are drawn literally
  if ('letterSpacing' in g2) g2.letterSpacing = '0.01px';
  draw(g2, c.width, c.height);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, fog: false }));
}

/** Unit box whose faces carry a baked shade, so unlit data cells still read as solid tiles. */
function cellGeo() {
  const g = new THREE.BoxGeometry(1, 1, 1);
  const n = g.attributes.normal;
  const col = new Float32Array(n.count * 3);
  for (let i = 0; i < n.count; i++) {
    const k = n.getZ(i) > 0.5 ? 1 : n.getY(i) > 0.5 ? 1.08 : n.getY(i) < -0.5 ? 0.42 : n.getZ(i) < -0.5 ? 0.4 : 0.66;
    col.fill(k, i * 3, i * 3 + 3);
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}

function cells(count, geo = cellGeo()) {
  const m = instanced(geo, new THREE.MeshBasicMaterial({ vertexColors: geo.attributes.color != null, transparent: true, fog: false }), Math.max(1, count));
  m.count = count;
  m.renderOrder = 6;
  return m;
}

/**
 * The card every exhibit sits on: dark glass, a title on the left, the shape / dtype on the right.
 * Content coordinates: origin at the center of the content box, x right, y up, world units.
 */
function exFrame(spec, env, cw, ch, draw) {
  const titleFont = F.title(0.2 * U);
  const capFont = F.mono(0.15 * U);
  const tw = spec.title ? measure(spec.title, titleFont) / U : 0;
  const cpw = spec.caption ? measure(spec.caption, capFont) / U : 0;
  const padX = 0.22;
  const padB = 0.2;
  const head = spec.title || spec.caption ? 0.52 : 0.2;
  const W = Math.max(cw + padX * 2, tw + cpw + padX * 2 + 0.5, 2.3);
  const H = ch + head + padB;
  const cy = padB + ch / 2;
  const P = { x: (x) => (x + W / 2) * U, y: (y) => (H - cy - y) * U, u: U };
  const group = new THREE.Group();
  const inner = new THREE.Group();
  group.add(inner);
  const card = canvasPlane(W, H, (g, w, h) => {
    g.beginPath();
    g.roundRect(1.5, 1.5, w - 3, h - 3, 0.13 * U);
    g.fillStyle = 'rgba(7,11,21,0.9)';
    g.fill();
    g.lineWidth = 2;
    g.strokeStyle = withAlpha(env.accent, 0.5);
    g.stroke();
    if (head > 0.3) {
      g.textBaseline = 'middle';
      g.fillStyle = env.accent;
      g.fillRect(padX * U, 0.175 * U, 0.04 * U, 0.2 * U);
      g.font = titleFont;
      g.fillStyle = INK;
      g.textAlign = 'left';
      g.fillText(spec.title ?? '', (padX + 0.11) * U, 0.285 * U);
      g.font = capFont;
      g.fillStyle = INK2;
      g.textAlign = 'right';
      g.fillText(spec.caption ?? '', w - padX * U, 0.29 * U);
      g.fillStyle = 'rgba(255,255,255,0.1)';
      g.fillRect(padX * U, (head - 0.08) * U, w - 2 * padX * U, 1.5);
    }
    draw?.(g, P);
  });
  card.position.y = H / 2;
  card.renderOrder = 5;
  const loop = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(roundRectPts(W, H, 0.13, 6).map(([x, y]) => new THREE.Vector3(x, y + H / 2, 0.004))),
    new THREE.LineBasicMaterial({ color: glow(env.accent, 1.6), transparent: true, opacity: 0.9, depthWrite: false }),
  );
  const content = new THREE.Group();
  content.position.set(0, cy, 0);
  inner.add(card, loop, content);
  return { group, inner, content, W, H, cw, ch };
}

/** Transparent text layer in front of the 3D content (same content coordinates as the card). */
function overlay(fr, z, draw) {
  const m = 0.14;
  const w = fr.cw + m * 2;
  const h = fr.ch + m * 2;
  const P = { x: (x) => (x + w / 2) * U, y: (y) => (h / 2 - y) * U, u: U };
  const pl = canvasPlane(w, h, (g) => draw(g, P));
  pl.position.z = z;
  pl.renderOrder = 7;
  return pl;
}

/** Split text into colored atoms and wrap them to a pixel width. Returns lines of [text, color]. */
function wrapRich(text, font, maxPx) {
  const lines = [];
  for (const para of String(text).split('\n')) {
    let line = [];
    let w = 0;
    for (const atom of para.match(/<\|[^|>]+\|>|\\n|\[\d+\]|\s+|[^\s<\\[]+|[<\\[]/g) ?? ['']) {
      const aw = measure(atom, font);
      if (w + aw > maxPx && line.length && atom.trim()) {
        lines.push(line);
        line = [];
        w = 0;
      }
      if (!line.length && !atom.trim()) continue;
      const color = /^<\|/.test(atom) ? CODE.special : atom === '\\n' ? INK3 : /^\[\d+\]$/.test(atom) ? CODE.cite : null;
      line.push([atom, color, aw]);
      w += aw;
    }
    lines.push(line);
  }
  return lines;
}

function drawCodeLine(g, line, x, y) {
  const put = (s, color) => {
    g.fillStyle = color;
    g.fillText(s, x, y);
    x += g.measureText(s).width;
  };
  let rest = line;
  const http = /^(GET|POST|PUT|DELETE|HTTP\/[\d.]+) /.exec(rest);
  const field = !http && /^([A-Za-z][\w-]*):(?= |$)/.exec(rest);
  if (http) {
    put(http[1], CODE.bool);
    rest = rest.slice(http[1].length);
  } else if (field && !/^\s*["{[]/.test(rest)) {
    put(field[1] + ':', CODE.key);
    rest = rest.slice(field[0].length);
  }
  const re = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?![\w-])|\b(true|false|null)\b|([{}[\],:])|(\s+)|([^\s"{}[\],:]+)/gi;
  let m;
  while ((m = re.exec(rest))) {
    if (m[1]) {
      put(m[1], m[2] ? CODE.key : CODE.str);
      if (m[2]) put(m[2], INK3);
    } else if (m[3]) put(m[3], CODE.num);
    else if (m[4]) put(m[4], CODE.bool);
    else if (m[5]) put(m[5], INK3);
    else put(m[6] ?? m[7], INK2);
  }
}

/** Small procedural pictures, sampled per pixel: (u, v) in [0,1], v = 0 at the top. Returns sRGB. */
const PAINTERS = {
  balloon(u, v, out, warm) {
    const sky = warm ? [[0.2, 0.25, 0.55], [0.98, 0.72, 0.45]] : [[0.16, 0.42, 0.82], [0.66, 0.84, 0.97]];
    const k = Math.pow(v, 1.3);
    let r = sky[0][0] + (sky[1][0] - sky[0][0]) * k;
    let g = sky[0][1] + (sky[1][1] - sky[0][1]) * k;
    let b = sky[0][2] + (sky[1][2] - sky[0][2]) * k;
    const ridge = 0.17 + 0.07 * Math.sin(u * 7 + 1) + 0.035 * Math.sin(u * 17 + 2.2);
    if (v > 1 - ridge) {
      const snow = v < 1 - ridge + 0.045;
      [r, g, b] = snow ? [0.9, 0.93, 0.98] : warm ? [0.33, 0.29, 0.42] : [0.3, 0.4, 0.55];
    }
    const cx = 0.5;
    const cy = 0.36;
    const R = 0.27;
    let half = 0;
    if (v <= cy) half = Math.sqrt(Math.max(0, R * R - (v - cy) * (v - cy)));
    else if (v < cy + 0.3) half = R * (1 - Math.pow((v - cy) / 0.3, 1.7));
    if (half > 0 && Math.abs(u - cx) < half) {
      const stripe = Math.floor(((u - cx) / half + 1) * 3);
      [r, g, b] = stripe % 2 ? [0.96, 0.78, 0.22] : [0.88, 0.25, 0.2];
    }
    if (Math.abs(u - cx) < 0.05 && v > 0.71 && v < 0.79) [r, g, b] = [0.42, 0.28, 0.16];
    return out.set(r, g, b);
  },
};
const _px = new THREE.Vector3();

/** Envelope of a short utterance followed by silence (the padding of a 30 s window). */
function speechEnv(seed = 5) {
  const rnd = rng(seed);
  const bursts = [];
  let x = 0.03;
  while (x < 0.36) {
    const w = 0.012 + rnd() * 0.02;
    bursts.push([x + w, w, 0.45 + rnd() * 0.55]);
    x += w * 2 + 0.004 + (rnd() < 0.25 ? 0.03 : 0);
  }
  return (u) => {
    let e = 0;
    for (let i = 0; i < bursts.length; i++) {
      const d = (u - bursts[i][0]) / bursts[i][1];
      e += bursts[i][2] * Math.exp(-d * d * 1.6);
    }
    return Math.min(1, e);
  };
}

/** Exhibit builders. Each returns {group, inner, content, W, H, update?}. */
export const EXHIBITS = {
  /** A block of text. Special tokens (<|...|>), literal "\n" and citations ([1]) are tinted. */
  text(spec, env) {
    const fs = (spec.size ?? 0.2) * U;
    const font = spec.mono ? F.mono(fs * 0.9, 520) : F.body(fs, 560);
    const lines = wrapRich(spec.text, font, (spec.width ?? 4.2) * U);
    const lh = (spec.size ?? 0.2) * 1.5;
    const cw = Math.max(2, ...lines.map((l) => l.reduce((s, a) => s + a[2], 0) / U));
    const ch = lines.length * lh + 0.06;
    return exFrame(spec, env, cw, ch, (g, P) => {
      g.font = font;
      g.textBaseline = 'middle';
      g.textAlign = 'left';
      lines.forEach((line, i) => {
        let x = P.x(-cw / 2);
        for (const [s, color, w] of line) {
          g.fillStyle = color ?? INK;
          g.fillText(s, x, P.y(ch / 2 - (i + 0.5) * lh - 0.03));
          x += w;
        }
      });
    });
  },

  /** Token chips. spec.tokens: [{t, id?, special?, tint?}], spec.show: 'text' | 'id' | 'both'. */
  tokens(spec, env) {
    const show = spec.show ?? 'text';
    const fT = F.body(0.185 * U, 600);
    const fS = F.mono(0.135 * U, 520);
    const fI = F.mono(0.165 * U, 560);
    const fSub = F.mono(0.115 * U, 500);
    const maxW = spec.width ?? 4.5;
    const chipH = show === 'both' ? 0.6 : 0.38;
    const gap = 0.07;
    const lay = [];
    let x = 0;
    let y = 0;
    let wMax = 0;
    for (const it of spec.tokens) {
      const main = show === 'id' ? String(it.id) : it.t;
      const font = show === 'id' ? fI : it.special ? fS : fT;
      let w = measure(main, font) / U + 0.2;
      if (show === 'both' && it.id != null) w = Math.max(w, measure(String(it.id), fSub) / U + 0.2);
      w = Math.max(w, 0.34);
      if (x > 0 && x + w > maxW) {
        x = 0;
        y += chipH + 0.1;
      }
      lay.push({ it, x, y, w, main, font });
      x += w + gap;
      wMax = Math.max(wMax, x - gap);
    }
    const cw = wMax;
    const ch = y + chipH;
    const fr = exFrame(spec, env, cw, ch);
    const mesh = cells(lay.length * 2);
    lay.forEach((L, i) => {
      L.cx = -cw / 2 + L.x + L.w / 2;
      L.cy = ch / 2 - L.y - chipH / 2;
      const tint = _c2.set(TINTS[(L.it.tint ?? i) % TINTS.length]);
      setInst(mesh, i * 2, L.cx, L.cy, 0.07, L.w, chipH, 0.1, _c.copy(CHIP).lerp(tint, L.it.special ? 0.12 : 0.34));
      setInst(mesh, i * 2 + 1, L.cx, L.cy - chipH / 2 + 0.028, 0.08, L.w, 0.056, 0.11, _c.copy(tint).multiplyScalar(L.it.special ? 0.55 : 1.2));
    });
    flushInst(mesh);
    fr.content.add(
      mesh,
      overlay(fr, 0.14, (g, P) => {
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        for (const L of lay) {
          g.font = L.font;
          g.fillStyle = L.it.special ? '#a9b9dc' : INK;
          g.fillText(L.main, P.x(L.cx), P.y(L.cy + (show === 'both' ? 0.11 : 0.025)));
          if (show === 'both' && L.it.id != null) {
            g.font = fSub;
            g.fillStyle = INK2;
            g.fillText(String(L.it.id), P.x(L.cx), P.y(L.cy - 0.12));
          }
        }
      }),
    );
    return fr;
  },

  /**
   * Rows of colored cells: one vector per row. spec: rows, cols, labels?, tints?, more? ('… 4,096'),
   * mode 'div' | 'seq', shared (0..1: how alike the rows are), result? {label} adds their mean, dimRows? [i].
   */
  vectors(spec, env) {
    const rows = spec.rows;
    const cols = spec.cols ?? 24;
    const cell = spec.cell ?? 0.105;
    const pitch = cell + 0.03;
    const rowPitch = spec.rowPitch ?? pitch;
    const fL = spec.monoLabels ? F.mono(0.12 * U, 520) : F.body(0.135 * U, 560);
    const fM = F.mono(0.13 * U, 520);
    const labels = spec.labels;
    const lw = labels ? Math.min(spec.labelWidth ?? 1.3, Math.max(...labels.map((l) => measure(l, fL) / U)) + 0.12) : 0;
    const sw = spec.tints === false ? 0 : 0.17;
    const mw = spec.more ? measure(spec.more, fM) / U + 0.16 : 0;
    const res = spec.result ? 1 : 0;
    const resGap = res ? 0.2 : 0;
    const cw = lw + sw + cols * pitch + mw;
    const ch = (rows + res) * rowPitch + resGap;
    const rndS = rng((spec.seed ?? 3) * 7919);
    const signal = Array.from({ length: cols }, () => rndS.normal() * 0.55);
    const V = [];
    for (let r = 0; r < rows; r++) {
      const rnd = rng((spec.rowSeeds?.[r] ?? (spec.seed ?? 3) * 131 + r) * 977 + 11);
      const sh = spec.shared ?? 0;
      let prev = 0;
      V.push(
        signal.map((s) => {
          prev = 0.35 * prev + 0.65 * rnd.normal() * 0.6;
          return clamp(sh * s + (1 - sh * 0.6) * prev, -1, 1);
        }),
      );
    }
    if (res) V.push(signal.map((_, c) => V.reduce((s, row) => s + row[c], 0) / rows));
    const rowY = (r) => ch / 2 - r * rowPitch - rowPitch / 2 - (r >= rows ? resGap : 0);
    const x0 = -cw / 2 + lw + sw;
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.textBaseline = 'middle';
      if (labels) {
        g.font = fL;
        g.textAlign = 'right';
        labels.forEach((l, r) => {
          g.fillStyle = spec.dimRows?.includes(r) ? INK3 : INK2;
          g.fillText(clip(l, fL, (lw - 0.08) * U), P.x(-cw / 2 + lw - 0.07), P.y(rowY(r)));
        });
      }
      if (res) {
        g.font = fL;
        g.textAlign = 'right';
        g.fillStyle = INK;
        if (lw) g.fillText(clip(spec.result.label, fL, (lw - 0.08) * U), P.x(-cw / 2 + lw - 0.07), P.y(rowY(rows)));
        g.strokeStyle = withAlpha(env.accent, 0.8);
        g.lineWidth = 2;
        g.strokeRect(P.x(x0 - 0.05), P.y(rowY(rows) + rowPitch / 2 + 0.03), (cols * pitch + 0.07) * U, (rowPitch + 0.06) * U);
      }
      if (spec.more) {
        g.font = fM;
        g.textAlign = 'left';
        g.fillStyle = INK3;
        g.fillText(spec.more, P.x(x0 + cols * pitch + 0.06), P.y(0));
      }
    });
    const mesh = cells((rows + res) * (cols + (sw ? 1 : 0)));
    let k = 0;
    for (let r = 0; r < rows + res; r++) {
      const dimmed = spec.dimRows?.includes(r) ? 0.28 : 1;
      if (sw) {
        const tint = r >= rows ? _c.set(env.accent) : _c.set(TINTS[(spec.rowTints?.[r] ?? r) % TINTS.length]);
        setInst(mesh, k++, x0 - 0.1, rowY(r), 0.05, 0.07, cell, 0.06, tint.multiplyScalar(dimmed));
      }
      for (let c = 0; c < cols; c++) {
        const v = V[r][c];
        const col = spec.mode === 'seq' ? seq(Math.abs(v), _c) : div(v, _c);
        setInst(mesh, k++, x0 + c * pitch + cell / 2, rowY(r), 0.05, cell, Math.min(cell, rowPitch - 0.02), 0.06, col.multiplyScalar(dimmed));
      }
    }
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /**
   * Heat tiles with relief. spec: rows, cols, pattern 'causal' | 'full' | 'signed' | 'blob', mode 'seq' | 'div',
   * xLabel?, yLabel?, rowLabels?, relief? (default true).
   */
  matrix(spec, env) {
    const { rows, cols } = spec;
    const cell = spec.cell ?? Math.min(0.26, 3.6 / cols, 2.7 / rows);
    const pitch = cell * 1.15;
    const fA = F.body(0.135 * U, 560);
    const fL = F.body(Math.min(0.135, cell * 0.8) * U, 560);
    const rl = spec.rowLabels;
    const lw = rl ? Math.min(1.3, Math.max(...rl.map((l) => measure(l, fL) / U)) + 0.12) : 0;
    const yl = spec.yLabel ? 0.26 : 0;
    const xl = spec.xLabel ? 0.3 : 0;
    const cw = lw + yl + cols * pitch;
    const ch = rows * pitch + xl;
    const rnd = rng((spec.seed ?? 4) * 613 + 5);
    const val = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      let mx = 1e-6;
      for (let c = 0; c < cols; c++) {
        let v;
        if (spec.pattern === 'causal') v = c > r ? null : Math.exp(-(r - c) * 0.5) * (0.35 + 0.65 * rnd()) + (c === 0 ? 0.22 : 0) + (rnd() < 0.1 ? 0.55 : 0);
        else if (spec.pattern === 'signed') v = clamp(rnd.normal() * 0.5, -1, 1);
        else if (spec.pattern === 'blob') {
          PAINTERS.balloon((c + 0.5) / cols, (r + 0.5) / rows, _px, false);
          v = (_px.x > 0.8 && _px.z < 0.4 ? 0.85 : 0.06) + rnd() * 0.14;
        } else v = 0.1 * rnd() + (r === c ? 0.5 : 0) + Math.exp(-Math.abs(r - c) * 0.7) * 0.3 * rnd() + (rnd() < 0.08 ? 0.6 : 0);
        row.push(v);
        if (v != null) mx = Math.max(mx, Math.abs(v));
      }
      val.push(spec.pattern === 'signed' ? row : row.map((v) => (v == null ? null : v / mx)));
    }
    const x0 = -cw / 2 + lw + yl;
    const yTop = ch / 2;
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.textBaseline = 'middle';
      if (rl) {
        g.font = fL;
        g.textAlign = 'right';
        g.fillStyle = INK2;
        rl.forEach((l, r) => g.fillText(clip(l, fL, (lw - 0.08) * U), P.x(x0 - 0.07), P.y(yTop - (r + 0.5) * pitch)));
      }
      g.font = fA;
      g.fillStyle = INK3;
      g.textAlign = 'center';
      if (spec.xLabel) g.fillText(spec.xLabel, P.x(x0 + (cols * pitch) / 2), P.y(-ch / 2 + 0.12));
      if (spec.yLabel) {
        g.save();
        g.translate(P.x(-cw / 2 + 0.11), P.y(yTop - (rows * pitch) / 2));
        g.rotate(-Math.PI / 2);
        g.fillText(spec.yLabel, 0, 0);
        g.restore();
      }
    });
    const mesh = cells(rows * cols);
    let k = 0;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const v = val[r][c];
        const depth = v == null || spec.relief === false ? 0.03 : 0.03 + Math.abs(v) * 0.2;
        const col = v == null ? _c.copy(EMPTY) : spec.pattern === 'signed' || spec.mode === 'div' ? div(v, _c) : seq(v, _c);
        setInst(mesh, k++, x0 + c * pitch + cell / 2, yTop - r * pitch - cell / 2 - (pitch - cell) / 2, 0.02 + depth / 2, cell, cell, depth, col);
      }
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /**
   * A tensor as stacked slabs. spec: slabs, rows, cols, mode 'noise' | 'div' | 'seq' | 'image', axes {depth, x, y},
   * painter? ('balloon' when mode is 'image').
   */
  tensor(spec, env) {
    const slabs = spec.slabs ?? 4;
    const { rows, cols } = spec;
    const cell = spec.cell ?? Math.min(0.14, 2.6 / cols, 1.9 / rows);
    const pitch = cell * 1.12;
    const dx = spec.dx ?? 0.2;
    const dy = spec.dy ?? 0.15;
    const dz = 0.24;
    const ax = spec.axes ?? {};
    const yl = ax.y ? 0.26 : 0;
    const xl = ax.x ? 0.28 : 0;
    const gw = cols * pitch + (slabs - 1) * dx;
    const gh = rows * pitch + (slabs - 1) * dy;
    const cw = yl + gw + (ax.depth ? 0.1 : 0);
    const ch = xl + gh + (ax.depth ? 0.24 : 0);
    const fA = F.body(0.135 * U, 560);
    const x0 = -cw / 2 + yl;
    const y0 = -ch / 2 + xl;
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.font = fA;
      g.fillStyle = INK3;
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      if (ax.x) g.fillText(ax.x, P.x(x0 + (cols * pitch) / 2), P.y(-ch / 2 + 0.11));
      if (ax.y) {
        g.save();
        g.translate(P.x(-cw / 2 + 0.11), P.y(y0 + (rows * pitch) / 2));
        g.rotate(-Math.PI / 2);
        g.fillText(ax.y, 0, 0);
        g.restore();
      }
      if (ax.depth) {
        g.textAlign = 'right';
        g.fillText(ax.depth, P.x(x0 + gw), P.y(y0 + gh + 0.13));
      }
    });
    const mesh = cells(slabs * rows * cols);
    const rnd = rng((spec.seed ?? 9) * 271 + 3);
    let k = 0;
    for (let s = slabs - 1; s >= 0; s--) {
      const shade = 1 - (s / Math.max(1, slabs)) * 0.5;
      let prev = 0;
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) {
          let col;
          if (spec.mode === 'image') {
            PAINTERS[spec.painter ?? 'balloon']((c + 0.5) / cols, (r + 0.5) / rows, _px, spec.warm);
            // three slabs = the red, green and blue planes; any other count shows the picture itself
            col = slabs === 3 ? _c.setRGB(s === 0 ? _px.x : 0, s === 1 ? _px.y : 0, s === 2 ? _px.z : 0, THREE.SRGBColorSpace) : _c.setRGB(_px.x, _px.y, _px.z, THREE.SRGBColorSpace);
          } else if (spec.mode === 'seq') col = seq(clamp(Math.abs(rnd.normal()) * 0.55), _c);
          else if (spec.mode === 'noise') col = div(clamp(rnd.normal() * 0.62, -1, 1), _c);
          else {
            prev = 0.5 * prev + 0.5 * rnd.normal() * 0.7;
            col = div(clamp(prev, -1, 1), _c);
          }
          setInst(mesh, k++, x0 + s * dx + c * pitch + cell / 2, y0 + s * dy + (rows - 1 - r) * pitch + cell / 2, 0.06 + (slabs - 1 - s) * dz, cell, cell, 0.05, col.multiplyScalar(shade));
        }
    }
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /** Probability (or score) bars. spec.items: [{label, value, chosen?}], tail? (count of tiny bars), tailLabel?, format?. */
  bars(spec, env) {
    const items = spec.items;
    const bw = spec.barWidth ?? 0.4;
    const gap = 0.16;
    const maxH = spec.height ?? 1.4;
    const fL = F.mono(0.14 * U, 560);
    const fV = F.mono(0.125 * U, 500);
    const tail = spec.tail ?? 0;
    const tw = tail ? tail * 0.075 + 0.2 : 0;
    const cw = items.length * (bw + gap) - gap + tw;
    const ch = maxH + 0.28 + 0.36 + (spec.tailLabel ? 0.2 : 0);
    const max = spec.max ?? Math.max(...items.map((i) => Math.abs(i.value)));
    const base = -ch / 2 + 0.36 + (spec.tailLabel ? 0.2 : 0);
    const fmtV = spec.format ?? ((v) => (v * 100).toFixed(v < 0.1 ? 1 : 0) + '%');
    const bx = (i) => -cw / 2 + i * (bw + gap) + bw / 2;
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      items.forEach((it, i) => {
        const h = (Math.abs(it.value) / max) * maxH;
        g.font = fL;
        g.fillStyle = INK;
        g.fillText(clip(it.label, fL, (bw + gap - 0.02) * U), P.x(bx(i)), P.y(base - 0.16));
        g.font = fV;
        g.fillStyle = INK2;
        g.fillText(fmtV(it.value), P.x(bx(i)), P.y(base + h + 0.13));
      });
      g.fillStyle = 'rgba(255,255,255,0.18)';
      g.fillRect(P.x(-cw / 2), P.y(base), cw * U, 1.5);
      if (spec.tailLabel) {
        g.font = F.body(0.125 * U, 520);
        g.fillStyle = INK3;
        g.textAlign = 'right';
        g.fillText(spec.tailLabel, P.x(cw / 2), P.y(-ch / 2 + 0.1));
      }
    });
    const mesh = cells(items.length + tail);
    items.forEach((it, i) => {
      const h = Math.max(0.02, (Math.abs(it.value) / max) * maxH);
      const col = it.chosen ? _c.copy(glow(env.accent, 1.7)) : seq(0.3 + 0.5 * (Math.abs(it.value) / max), _c);
      setInst(mesh, i, bx(i), base + h / 2, 0.14, bw, h, 0.24, col);
    });
    for (let j = 0; j < tail; j++) {
      const h = Math.max(0.015, 0.12 * Math.exp(-j * 0.16));
      setInst(mesh, items.length + j, -cw / 2 + items.length * (bw + gap) + 0.06 + j * 0.075, base + h / 2, 0.1, 0.045, h, 0.16, seq(0.3, _c));
    }
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /** A pixel grid. spec: n (pixels per side), size, patch? (pixels per patch: draws gaps), noise 0..1, warm?. */
  image(spec, env) {
    const n = spec.n ?? 28;
    const size = spec.size ?? 2.5;
    const pitch = size / n;
    const patch = spec.patch ?? 0;
    const pgap = patch ? (spec.patchGap ?? 0.07) : 0;
    const span = size + (patch ? (n / patch - 1) * pgap : 0);
    const fA = F.body(0.135 * U, 560);
    const cw = span;
    const ch = span + (spec.note ? 0.26 : 0);
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      if (!spec.note) return;
      g.font = fA;
      g.fillStyle = INK3;
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      g.fillText(spec.note, P.x(0), P.y(-ch / 2 + 0.1));
    });
    const mesh = cells(n * n);
    const rnd = rng((spec.seed ?? 2) * 151 + 7);
    const a = spec.noise ?? 0;
    const top = ch / 2;
    let k = 0;
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        PAINTERS[spec.painter ?? 'balloon']((i + 0.5) / n, (j + 0.5) / n, _px, spec.warm);
        const nz = () => clamp(0.5 + rnd.normal() * 0.3);
        _c.setRGB(_px.x + (nz() - _px.x) * a, _px.y + (nz() - _px.y) * a, _px.z + (nz() - _px.z) * a, THREE.SRGBColorSpace);
        const gx = patch ? Math.floor(i / patch) * pgap : 0;
        const gy = patch ? Math.floor(j / patch) * pgap : 0;
        setInst(mesh, k++, -span / 2 + i * pitch + gx + pitch / 2, top - j * pitch - gy - pitch / 2, 0.05, pitch * 0.96, pitch * 0.96, 0.06, _c);
      }
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /** A row of small pictures at decreasing noise levels. spec.frames: [{noise, label}]. */
  frames(spec, env) {
    const fs = spec.frames;
    const n = spec.n ?? 14;
    const size = spec.size ?? 1.02;
    const gap = 0.3;
    const pitch = size / n;
    const cw = fs.length * size + (fs.length - 1) * gap;
    const ch = size + 0.3;
    const fL = F.mono(0.125 * U, 520);
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      fs.forEach((f, i) => {
        const x = -cw / 2 + i * (size + gap) + size / 2;
        g.font = fL;
        g.fillStyle = INK2;
        g.fillText(f.label, P.x(x), P.y(-ch / 2 + 0.1));
        if (i < fs.length - 1) {
          g.fillStyle = INK3;
          g.font = F.body(0.2 * U, 600);
          g.fillText('→', P.x(x + size / 2 + gap / 2), P.y(-ch / 2 + 0.3 + size / 2));
        }
      });
    });
    const mesh = cells(fs.length * n * n);
    let k = 0;
    fs.forEach((f, fi) => {
      const rnd = rng(fi * 31 + 5);
      const ox = -cw / 2 + fi * (size + gap);
      for (let j = 0; j < n; j++)
        for (let i = 0; i < n; i++) {
          PAINTERS[spec.painter ?? 'balloon']((i + 0.5) / n, (j + 0.5) / n, _px, spec.warm);
          const nz = () => clamp(0.5 + rnd.normal() * 0.3);
          _c.setRGB(_px.x + (nz() - _px.x) * f.noise, _px.y + (nz() - _px.y) * f.noise, _px.z + (nz() - _px.z) * f.noise, THREE.SRGBColorSpace);
          setInst(mesh, k++, ox + i * pitch + pitch / 2, ch / 2 - j * pitch - pitch / 2, 0.05, pitch * 0.96, pitch * 0.96, 0.06, _c);
        }
    });
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /** An audio waveform: a short utterance, then the silence that pads the window. spec: left, right (axis labels). */
  waveform(spec, env) {
    const n = spec.n ?? 150;
    const cw = spec.width ?? 4.3;
    const hgt = 1.25;
    const ch = hgt + 0.28;
    const pitch = cw / n;
    const envf = speechEnv(spec.seed);
    const fA = F.mono(0.125 * U, 520);
    const mid = -ch / 2 + 0.28 + hgt / 2;
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.fillStyle = 'rgba(255,255,255,0.16)';
      g.fillRect(P.x(-cw / 2), P.y(mid), cw * U, 1.5);
      g.font = fA;
      g.fillStyle = INK3;
      g.textBaseline = 'middle';
      g.textAlign = 'left';
      g.fillText(spec.left ?? '0 s', P.x(-cw / 2), P.y(-ch / 2 + 0.1));
      g.textAlign = 'right';
      g.fillText(spec.right ?? '30 s', P.x(cw / 2), P.y(-ch / 2 + 0.1));
      if (spec.note) {
        g.textAlign = 'center';
        g.fillText(spec.note, P.x(cw * 0.18), P.y(-ch / 2 + 0.1));
      }
    });
    const mesh = cells(n);
    const rnd = rng(17);
    for (let i = 0; i < n; i++) {
      const e = envf((i + 0.5) / n);
      const h = Math.max(0.012, e * (0.35 + 0.65 * rnd()) * hgt);
      setInst(mesh, i, -cw / 2 + i * pitch + pitch / 2, mid, 0.05, pitch * 0.62, h, 0.06, seq(0.4 + 0.55 * Math.min(1, h / hgt), _c));
    }
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /** A log-mel spectrogram matching the waveform exhibit. spec: cols, rows, xLabel, yLabel. */
  spectrogram(spec, env) {
    const cols = spec.cols ?? 72;
    const rows = spec.rows ?? 20;
    const gw = spec.width ?? 4.1;
    const cell = gw / cols;
    const ch0 = rows * cell * 1.25;
    const yl = spec.yLabel ? 0.26 : 0;
    const cw = gw + yl;
    const ch = ch0 + 0.28;
    const envf = speechEnv(spec.seed);
    const fA = F.body(0.13 * U, 560);
    const x0 = -cw / 2 + yl;
    const y0 = -ch / 2 + 0.28;
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.font = fA;
      g.fillStyle = INK3;
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      g.fillText(spec.xLabel ?? 'time →', P.x(x0 + gw / 2), P.y(-ch / 2 + 0.1));
      if (spec.yLabel) {
        g.save();
        g.translate(P.x(-cw / 2 + 0.11), P.y(y0 + ch0 / 2));
        g.rotate(-Math.PI / 2);
        g.fillText(spec.yLabel, 0, 0);
        g.restore();
      }
    });
    const mesh = cells(cols * rows);
    const rnd = rng(23);
    let k = 0;
    for (let c = 0; c < cols; c++) {
      const u = (c + 0.5) / cols;
      const e = envf(u);
      const f1 = 0.13 + 0.06 * Math.sin(u * 60);
      const f2 = 0.4 + 0.13 * Math.sin(u * 37 + 1);
      const f3 = 0.66 + 0.06 * Math.sin(u * 49 + 2);
      const hiss = e > 0.2 && Math.sin(u * 211) > 0.6 ? 0.5 : 0;
      for (let r = 0; r < rows; r++) {
        const f = (r + 0.5) / rows;
        const b = (fc, w) => Math.exp(-(((f - fc) / w) ** 2));
        const v = e * (b(f1, 0.07) + 0.8 * b(f2, 0.08) + 0.5 * b(f3, 0.07) + hiss * b(0.85, 0.16)) * (1 - f * 0.35) + rnd() * 0.035;
        setInst(mesh, k++, x0 + c * cell + cell / 2, y0 + r * cell * 1.25 + cell * 0.62, 0.04, cell * 0.94, cell * 1.18, 0.05, seq(clamp(v), _c));
      }
    }
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /** A code card with light syntax coloring (JSON, HTTP head lines, SSE "data:" lines). spec: code, bar?, cols?, maxLines?. */
  json(spec, env) {
    const fs = spec.size ?? 0.145;
    const font = F.mono(fs * U, 500);
    const chW = measure('M', font) / U;
    const maxCols = spec.cols ?? 46;
    let lines = String(spec.code).split('\n');
    const maxLines = spec.maxLines ?? 15;
    if (lines.length > maxLines) lines = [...lines.slice(0, maxLines - 1), '  …'];
    lines = lines.map((l) => (l.length > maxCols ? l.slice(0, maxCols - 1) + '…' : l));
    const lh = fs * 1.5;
    const barH = spec.bar ? 0.34 : 0;
    const cw = Math.max(2.4, Math.max(...lines.map((l) => l.length), spec.bar ? spec.bar.length + 2 : 0) * chW + 0.06);
    const ch = lines.length * lh + barH + 0.04;
    return exFrame(spec, env, cw, ch, (g, P) => {
      g.textBaseline = 'middle';
      g.textAlign = 'left';
      if (spec.bar) {
        g.fillStyle = withAlpha(env.accent, 0.16);
        g.beginPath();
        g.roundRect(P.x(-cw / 2 - 0.05), P.y(ch / 2), (cw + 0.1) * U, 0.27 * U, 0.06 * U);
        g.fill();
        g.font = F.mono(fs * U, 620);
        g.fillStyle = INK;
        g.fillText(spec.bar, P.x(-cw / 2 + 0.03), P.y(ch / 2 - 0.14));
      }
      g.font = font;
      lines.forEach((l, i) => drawCodeLine(g, l, P.x(-cw / 2), P.y(ch / 2 - barH - (i + 0.5) * lh - 0.02)));
    });
  },

  /** Documents as page stacks. spec.docs: [{name, sub?, tint?}]. */
  docs(spec, env) {
    const docs = spec.docs;
    const pw = 0.82;
    const ph = 1.06;
    const gap = 0.5;
    const cw = docs.length * pw + (docs.length - 1) * gap + 0.14;
    const ch = ph + 0.12 + 0.46;
    const fN = F.mono(0.125 * U, 560);
    const fS = F.body(0.115 * U, 520);
    const px = (i) => -cw / 2 + i * (pw + gap) + pw / 2;
    const py = ch / 2 - ph / 2 - 0.1;
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      docs.forEach((d, i) => {
        g.font = fN;
        g.fillStyle = INK;
        g.fillText(clip(d.name, fN, (pw + gap - 0.06) * U), P.x(px(i)), P.y(-ch / 2 + 0.3));
        if (d.sub) {
          g.font = fS;
          g.fillStyle = INK3;
          g.fillText(clip(d.sub, fS, (pw + gap - 0.04) * U), P.x(px(i)), P.y(-ch / 2 + 0.1));
        }
      });
    });
    const mesh = cells(docs.length * 3);
    docs.forEach((d, i) => {
      for (let s = 2; s >= 0; s--) setInst(mesh, i * 3 + s, px(i) + s * 0.055, py + s * 0.045, 0.16 - s * 0.05, pw, ph, 0.03, _c.set('#c3cfe6').multiplyScalar(1 - s * 0.3));
    });
    flushInst(mesh);
    fr.content.add(
      mesh,
      overlay(fr, 0.185, (g, P) => {
        const rnd = rng(41);
        docs.forEach((d, i) => {
          g.fillStyle = TINTS[(d.tint ?? i) % TINTS.length];
          g.fillRect(P.x(px(i) - pw / 2 + 0.08), P.y(py + ph / 2 - 0.09), (pw - 0.36) * U, 0.07 * U);
          g.fillStyle = 'rgba(40,52,82,0.75)';
          for (let l = 0; l < 8; l++) g.fillRect(P.x(px(i) - pw / 2 + 0.08), P.y(py + ph / 2 - 0.26 - l * 0.095), (pw - 0.16) * (l === 7 ? 0.45 : 0.7 + rnd() * 0.3) * U, 0.035 * U);
        });
      }),
    );
    return fr;
  },

  /** Chunks grouped by source document. spec.docs: [{name, n, tint?}], pick? [chunk index...] highlights, start? numbering. */
  chunks(spec, env) {
    const docs = spec.docs;
    const tw = spec.tileWidth ?? 0.9;
    const th = 0.36;
    const gap = 0.09;
    const colGap = 0.3;
    const maxN = Math.max(...docs.map((d) => d.n));
    const cw = docs.length * tw + (docs.length - 1) * colGap;
    const ch = maxN * (th + gap) - gap + 0.3;
    const fN = F.mono(0.12 * U, 560);
    const fI = F.mono(0.12 * U, 560);
    const pick = new Set(spec.pick ?? []);
    const tiles = [];
    let id = spec.start ?? 0;
    docs.forEach((d, di) => {
      for (let j = 0; j < d.n; j++) tiles.push({ di, j, id: id++, x: -cw / 2 + di * (tw + colGap) + tw / 2, y: ch / 2 - 0.3 - j * (th + gap) - th / 2, tint: TINTS[(d.tint ?? di) % TINTS.length] });
    });
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.font = fN;
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      g.fillStyle = INK2;
      docs.forEach((d, di) => g.fillText(clip(d.name, fN, (tw + colGap - 0.04) * U), P.x(-cw / 2 + di * (tw + colGap) + tw / 2), P.y(ch / 2 - 0.1)));
    });
    const mesh = cells(tiles.length);
    tiles.forEach((t, i) => {
      const on = !pick.size || pick.has(t.id);
      const z = pick.has(t.id) ? 0.14 : 0.06;
      setInst(mesh, i, t.x, t.y, z, tw, th, 0.08, _c.copy(CHIP).lerp(_c2.set(t.tint), on ? 0.5 : 0.14).multiplyScalar(pick.has(t.id) ? 1.5 : 1));
      t.z = z;
    });
    flushInst(mesh);
    fr.content.add(
      mesh,
      overlay(fr, 0.2, (g, P) => {
        g.textBaseline = 'middle';
        for (const t of tiles) {
          const on = !pick.size || pick.has(t.id);
          g.globalAlpha = on ? 1 : 0.4;
          g.font = fI;
          g.textAlign = 'left';
          g.fillStyle = INK;
          g.fillText('#' + t.id, P.x(t.x - tw / 2 + 0.07), P.y(t.y));
          g.fillStyle = 'rgba(233,238,251,0.5)';
          g.fillRect(P.x(t.x - tw / 2 + 0.36), P.y(t.y + 0.07), (tw - 0.46) * U, 0.035 * U);
          g.fillRect(P.x(t.x - tw / 2 + 0.36), P.y(t.y - 0.035), (tw - 0.6) * U, 0.035 * U);
        }
        g.globalAlpha = 1;
      }),
    );
    return fr;
  },

  /**
   * A point cloud: vectors in embedding space. spec: n, k (nearest neighbors to highlight, 0 = none),
   * links? (draw the proximity graph), note?.
   */
  scatter(spec, env) {
    const n = spec.n ?? 240;
    const cw = spec.width ?? 3.6;
    const gh = 2.1;
    const ch = gh + (spec.note ? 0.26 : 0);
    const fA = F.body(0.135 * U, 560);
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      if (!spec.note) return;
      g.font = fA;
      g.fillStyle = INK3;
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      g.fillText(spec.note, P.x(0), P.y(-ch / 2 + 0.1));
    });
    const rnd = rng((spec.seed ?? 6) * 97 + 1);
    const centers = Array.from({ length: 6 }, () => [rnd.range(-1.3, 1.3), rnd.range(-0.7, 0.7), rnd.range(-0.45, 0.45)]);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const c = i % centers.length;
      pts.push([centers[c][0] + rnd.normal() * 0.24, centers[c][1] + rnd.normal() * 0.2, centers[c][2] + rnd.normal() * 0.2, c]);
    }
    const q = [centers[2][0] + 0.12, centers[2][1] - 0.05, centers[2][2] + 0.1];
    const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
    const kNear = spec.k ?? 0;
    const near = kNear ? pts.map((p, i) => [d2(p, q), i]).sort((a, b) => a[0] - b[0]).slice(0, kNear).map((x) => x[1]) : [];
    const nearSet = new Set(near);
    const cloud = new THREE.Group();
    cloud.position.set(0, ch / 2 - gh / 2, 0.75);
    const mesh = cells(n + (kNear ? 1 : 0), new THREE.IcosahedronGeometry(1, 1));
    pts.forEach((p, i) => {
      const hit = nearSet.has(i);
      const col = hit ? _c.copy(glow('#ffffff', 2.2)) : _c.set(TINTS[p[3] % TINTS.length]).multiplyScalar(kNear ? 0.5 : 1.15);
      setInst(mesh, i, p[0], p[1], p[2], hit ? 0.055 : 0.036, undefined, undefined, col);
    });
    if (kNear) setInst(mesh, n, q[0], q[1], q[2], 0.085, undefined, undefined, glow(env.accent, 2.6));
    flushInst(mesh);
    cloud.add(mesh);
    const seg = [];
    if (kNear) for (const i of near) seg.push(new THREE.Vector3(...q), new THREE.Vector3(pts[i][0], pts[i][1], pts[i][2]));
    if (spec.links)
      pts.forEach((p, i) => {
        const nn = pts.map((o, j) => [j === i ? Infinity : d2(p, o), j]).sort((a, b) => a[0] - b[0]).slice(0, 2);
        for (const [, j] of nn) if (j > i || !kNear) seg.push(new THREE.Vector3(p[0], p[1], p[2]), new THREE.Vector3(pts[j][0], pts[j][1], pts[j][2]));
      });
    if (seg.length) {
      const lines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), new THREE.LineBasicMaterial({ color: glow(kNear ? env.accent : '#8fa3c8', kNear ? 1.8 : 0.9), transparent: true, opacity: kNear ? 0.95 : 0.3, depthWrite: false }));
      cloud.add(lines);
    }
    fr.content.add(cloud);
    fr.update = (dt, t) => (cloud.rotation.y = Math.sin(t * 0.35) * 0.4);
    return fr;
  },

  /** A key-value cache: one K and one V strip per layer, filled up to the current token. spec: layers, slots, filled, first, last. */
  cache(spec, env) {
    const L = spec.layers ?? 8;
    const S = spec.slots ?? 26;
    const F0 = spec.filled ?? 16;
    const cell = 0.115;
    const pitch = cell + 0.022;
    const rowP = 0.25;
    const lw = 0.92;
    const cw = lw + S * pitch;
    const ch = L * rowP + 0.5;
    const fA = F.body(0.13 * U, 560);
    const fM = F.mono(0.115 * U, 520);
    const x0 = -cw / 2 + lw;
    const yTop = ch / 2 - 0.24;
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.textBaseline = 'middle';
      g.font = fA;
      g.fillStyle = INK2;
      g.textAlign = 'right';
      for (let l = 0; l < L; l++) {
        const name = l === 0 ? (spec.first ?? 'layer 1') : l === L - 1 ? (spec.last ?? 'layer 32') : l === Math.floor(L / 2) ? '…' : '';
        if (name) g.fillText(name, P.x(x0 - 0.3), P.y(yTop - l * rowP - rowP / 2 + 0.01));
        g.font = fM;
        g.fillStyle = INK3;
        g.fillText('K', P.x(x0 - 0.08), P.y(yTop - l * rowP - 0.065));
        g.fillText('V', P.x(x0 - 0.08), P.y(yTop - l * rowP - 0.18));
        g.font = fA;
        g.fillStyle = INK2;
      }
      g.fillStyle = INK3;
      g.textAlign = 'left';
      g.fillText(spec.filledLabel ?? `${F0} tokens cached`, P.x(x0), P.y(ch / 2 - 0.1));
      g.textAlign = 'right';
      g.fillText(spec.freeLabel ?? 'room for the tokens still to come →', P.x(cw / 2), P.y(-ch / 2 + 0.1));
    });
    const mesh = cells(L * S * 2);
    const newest = [];
    let k = 0;
    for (let l = 0; l < L; l++)
      for (let s = 0; s < S; s++)
        for (let kv = 0; kv < 2; kv++) {
          const filled = s < F0;
          const col = !filled ? _c.copy(EMPTY) : s === F0 - 1 ? _c.copy(glow('#ffffff', 1.9)) : seq(kv ? 0.72 : 0.5, _c);
          if (filled && s === F0 - 1) newest.push(k);
          setInst(mesh, k++, x0 + s * pitch + cell / 2, yTop - l * rowP - 0.065 - kv * 0.115, 0.05, cell, 0.095, filled ? 0.07 : 0.03, col);
        }
    flushInst(mesh);
    fr.content.add(mesh);
    const white = glow('#ffffff', 1);
    fr.update = (dt, t) => {
      _c.copy(white).multiplyScalar(1.3 + 0.9 * (0.5 + 0.5 * Math.sin(t * 4)));
      for (let i = 0; i < newest.length; i++) mesh.setColorAt(newest[i], _c);
      mesh.instanceColor.needsUpdate = true;
    };
    return fr;
  },

  /** Continuous batching as a chart: rows = batch slots, columns = decoding steps, one tint per request. */
  batch(spec, env) {
    const slots = spec.slots ?? 6;
    const steps = spec.steps ?? 28;
    const cell = 0.125;
    const pitch = cell + 0.02;
    const rowP = 0.25;
    const lw = 0.58;
    const cw = lw + steps * pitch;
    const ch = slots * rowP + 0.5;
    const fA = F.body(0.13 * U, 560);
    const x0 = -cw / 2 + lw;
    const yTop = ch / 2 - 0.22;
    const now = spec.now ?? Math.round(steps * 0.68);
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.font = fA;
      g.textBaseline = 'middle';
      g.textAlign = 'right';
      g.fillStyle = INK2;
      for (let s = 0; s < slots; s++) g.fillText('slot ' + (s + 1), P.x(x0 - 0.1), P.y(yTop - s * rowP - rowP / 2 + 0.02));
      g.fillStyle = INK3;
      g.textAlign = 'left';
      g.fillText(spec.xLabel ?? 'decoding steps →', P.x(x0), P.y(-ch / 2 + 0.1));
      g.textAlign = 'center';
      g.fillStyle = INK;
      g.fillText('now', P.x(x0 + now * pitch - 0.01), P.y(ch / 2 - 0.09));
      g.fillStyle = 'rgba(255,255,255,0.7)';
      g.fillRect(P.x(x0 + now * pitch - 0.012), P.y(yTop + 0.03), 2, (slots * rowP + 0.04) * U);
    });
    const mesh = cells(slots * steps);
    const rnd = rng((spec.seed ?? 8) * 53 + 9);
    let req = 0;
    let k = 0;
    for (let s = 0; s < slots; s++) {
      let c = -rnd.int(5);
      while (c < steps) {
        const len = 4 + rnd.int(11);
        const tint = TINTS[req++ % TINTS.length];
        for (let j = 0; j < len; j++, c++) {
          if (c < 0 || c >= steps) continue;
          const first = j === 0;
          _c.copy(CHIP).lerp(_c2.set(tint), first ? 1 : 0.62).multiplyScalar((first ? 1.5 : 1) * (c >= now ? 0.3 : 1));
          setInst(mesh, k++, x0 + c * pitch + cell / 2, yTop - s * rowP - rowP / 2 + 0.02, 0.05, cell, rowP - 0.07, first ? 0.11 : 0.06, _c);
        }
      }
    }
    mesh.count = k;
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /** A horizontal timeline. spec.segments: [{label, w, tone (0..1), ticks?}], marks: [{at (0..1), label}]. */
  timeline(spec, env) {
    const cw = spec.width ?? 4.3;
    const bh = 0.36;
    const ch = bh + 0.34 + 0.4;
    const total = spec.segments.reduce((s, x) => s + x.w, 0);
    const fA = F.body(0.13 * U, 560);
    const yb = -ch / 2 + 0.36;
    let acc = 0;
    const segs = spec.segments.map((s) => {
      const x = -cw / 2 + (acc / total) * cw;
      acc += s.w;
      return { ...s, x, wd: (s.w / total) * cw };
    });
    const fr = exFrame(spec, env, cw, ch, (g, P) => {
      g.font = fA;
      g.textBaseline = 'middle';
      g.textAlign = 'center';
      g.fillStyle = INK2;
      for (const s of segs) g.fillText(clip(s.label, fA, s.wd * U + 20), P.x(s.x + s.wd / 2), P.y(-ch / 2 + 0.12));
      for (const m of spec.marks ?? []) {
        const x = -cw / 2 + m.at * cw;
        g.fillStyle = 'rgba(255,255,255,0.75)';
        g.fillRect(P.x(x) - 1, P.y(yb + bh + 0.22), 2, (bh + 0.3) * U);
        g.fillStyle = INK;
        g.textAlign = m.at > 0.75 ? 'right' : m.at < 0.2 ? 'left' : 'center';
        g.fillText(m.label, P.x(x), P.y(ch / 2 - 0.1));
        g.textAlign = 'center';
      }
    });
    const nTicks = segs.reduce((s, x) => s + (x.ticks ?? 0), 0);
    const mesh = cells(segs.length + nTicks);
    let k = 0;
    for (const s of segs) {
      if (s.ticks) for (let i = 0; i < s.ticks; i++) setInst(mesh, k++, s.x + ((i + 0.5) / s.ticks) * s.wd, yb + bh / 2, 0.06, (s.wd / s.ticks) * 0.55, bh, 0.08, i === 0 ? _c.copy(glow(env.accent, 1.7)) : seq(s.tone ?? 0.6, _c));
      else setInst(mesh, k++, s.x + s.wd / 2, yb + bh / 2, 0.05, s.wd - 0.04, bh, 0.06, seq(s.tone ?? 0.4, _c));
    }
    mesh.count = k;
    flushInst(mesh);
    fr.content.add(mesh);
    return fr;
  },

  /** One big number. spec: value, label?, sub?. */
  scalar(spec, env) {
    const fV = F.title(0.62 * U);
    const fL = F.body(0.16 * U, 560);
    const cw = Math.max(2.3, measure(spec.value, fV) / U + 0.2, spec.sub ? measure(spec.sub, fL) / U : 0, spec.label ? measure(spec.label, fL) / U : 0);
    const ch = 0.86 + (spec.label ? 0.26 : 0) + (spec.sub ? 0.26 : 0);
    return exFrame(spec, env, cw, ch, (g, P) => {
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      let y = ch / 2;
      if (spec.label) {
        g.font = fL;
        g.fillStyle = INK2;
        g.fillText(spec.label, P.x(0), P.y(y - 0.13));
        y -= 0.26;
      }
      g.font = fV;
      g.fillStyle = INK;
      g.fillText(spec.value, P.x(0), P.y(y - 0.43));
      if (spec.sub) {
        g.font = fL;
        g.fillStyle = INK3;
        g.fillText(spec.sub, P.x(0), P.y(-ch / 2 + 0.13));
      }
    });
  },
};
EXHIBITS.ids = (spec, env) => EXHIBITS.tokens({ ...spec, show: spec.show ?? 'id' }, env);
EXHIBITS.noise = (spec, env) => EXHIBITS.tensor({ ...spec, mode: 'noise' }, env);

/**
 * Build one exhibit. spec.kind selects the depiction; spec.title and spec.caption fill the card header.
 * @param {{kind:string, title?:string, caption?:string}} spec
 * @param {{accent?:string}} [env]
 */
export function buildExhibit(spec, env = {}) {
  const make = EXHIBITS[spec.kind] ?? EXHIBITS.text;
  return make(spec, { accent: '#4de3b0', ...env });
}
