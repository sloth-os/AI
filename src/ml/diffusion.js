// Diffusion with the exact denoiser. For a finite set of target points y_i the best possible
// estimate of the clean sample behind a noisy point x_t is known in closed form:
//   E[x0 | x_t] = sum_i w_i y_i,   w_i proportional to exp(-|x_t - a_t y_i|^2 / (2 s_t^2))
// so no network has to be trained: the samplers below run on the real mathematics.
// Pure JS, no rendering code.

/** Deterministic pseudo-random generator with a normal() helper. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  const f = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  f.normal = () => {
    let u = 0;
    while (u === 0) u = f();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * f());
  };
  return f;
}

/** Target shapes: each returns n points (Float32Array n*3) that fit roughly in the unit ball. */
export const SHAPES = [
  {
    id: 'knot',
    name: 'Torus knot',
    make(n, rnd) {
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const t = (i / n) * Math.PI * 2;
        const r = 0.62 + 0.26 * Math.cos(3 * t);
        out.set([r * Math.cos(2 * t) + rnd.normal() * 0.012, 0.3 * Math.sin(3 * t) + rnd.normal() * 0.012, r * Math.sin(2 * t) + rnd.normal() * 0.012], i * 3);
      }
      return out;
    },
  },
  {
    id: 'ai',
    name: 'The letters AI',
    make(n, rnd) {
      // strokes of "A" and "I" as line segments in the x-y plane
      const segs = [
        [-0.85, -0.6, -0.5, 0.6], [-0.5, 0.6, -0.15, -0.6], [-0.69, -0.1, -0.31, -0.1],
        [0.5, -0.6, 0.5, 0.6], [0.25, 0.6, 0.75, 0.6], [0.25, -0.6, 0.75, -0.6],
      ];
      const len = segs.map(([a, b, c, d]) => Math.hypot(c - a, d - b));
      const total = len.reduce((s, v) => s + v, 0);
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        let u = rnd() * total;
        let k = 0;
        while (u > len[k] && k < segs.length - 1) u -= len[k++];
        const f = u / len[k];
        const [a, b, c, d] = segs[k];
        out.set([a + (c - a) * f + rnd.normal() * 0.012, b + (d - b) * f + rnd.normal() * 0.012, (rnd() - 0.5) * 0.16], i * 3);
      }
      return out;
    },
  },
  {
    id: 'galaxy',
    name: 'Spiral galaxy',
    make(n, rnd) {
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const r = 0.08 + 0.85 * Math.sqrt(rnd());
        const a = r * 5.5 + (i % 2) * Math.PI + rnd.normal() * 0.12;
        out.set([r * Math.cos(a), rnd.normal() * 0.03 * (1.2 - r), r * Math.sin(a)], i * 3);
      }
      return out;
    },
  },
  {
    id: 'heart',
    name: 'Heart',
    make(n, rnd) {
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const t = rnd() * Math.PI * 2;
        const x = 16 * Math.sin(t) ** 3;
        const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
        const k = 0.05 * Math.sqrt(rnd()) + 0.0005;
        out.set([x * k * 0.98, y * k * 0.98 + 0.08, (rnd() - 0.5) * 0.22], i * 3);
      }
      return out;
    },
  },
  {
    id: 'helix',
    name: 'Double helix',
    make(n, rnd) {
      const out = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const u = rnd();
        const a = u * Math.PI * 5;
        const strand = i % 5 === 0 ? rnd() * 2 - 1 : i % 2 ? 1 : -1; // every fifth point is a rung
        out.set([0.36 * Math.cos(a) * strand + rnd.normal() * 0.01, (u - 0.5) * 1.8, 0.36 * Math.sin(a) * strand + rnd.normal() * 0.01], i * 3);
      }
      return out;
    },
  },
];

/** Cosine noise schedule: how much signal (a) and noise (s) are left at time t in [0, 1]. a² + s² = 1. */
export function schedule(t) {
  const f = Math.cos(((t + 0.008) / 1.008) * (Math.PI / 2));
  const f0 = Math.cos((0.008 / 1.008) * (Math.PI / 2));
  const ab = Math.min(1, Math.max(1e-5, (f * f) / (f0 * f0)));
  return { a: Math.sqrt(ab), s: Math.sqrt(1 - ab), ab };
}

/**
 * The exact denoiser: writes E[x0 | x] for every particle into `out`.
 * `scale` multiplies the targets and `sigma` is the noise level, so the same routine serves both
 * the diffusion path (scale a_t, sigma s_t) and the flow-matching path (scale 1 - t, sigma t).
 */
export function denoise(x, n, targets, m, scale, sigma, out) {
  const inv = 1 / (2 * sigma * sigma);
  for (let p = 0; p < n; p++) {
    const px = x[p * 3];
    const py = x[p * 3 + 1];
    const pz = x[p * 3 + 2];
    // two passes for numerical safety: find the closest target, then weight relative to it
    let best = Infinity;
    for (let i = 0; i < m; i++) {
      const dx = px - scale * targets[i * 3];
      const dy = py - scale * targets[i * 3 + 1];
      const dz = pz - scale * targets[i * 3 + 2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < best) best = d;
    }
    let sw = 0;
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let i = 0; i < m; i++) {
      const tx = targets[i * 3];
      const ty = targets[i * 3 + 1];
      const tz = targets[i * 3 + 2];
      const dx = px - scale * tx;
      const dy = py - scale * ty;
      const dz = pz - scale * tz;
      const e = (dx * dx + dy * dy + dz * dz - best) * inv;
      if (e > 30) continue;
      const w = Math.exp(-e);
      sw += w;
      sx += w * tx;
      sy += w * ty;
      sz += w * tz;
    }
    out[p * 3] = sx / sw;
    out[p * 3 + 1] = sy / sw;
    out[p * 3 + 2] = sz / sw;
  }
}

export const SAMPLERS = [
  { id: 'ddpm', name: 'DDPM', blurb: 'The original sampler: every step removes some noise and then adds a little fresh noise back. Robust, and needs many steps.' },
  { id: 'ddim', name: 'DDIM', blurb: 'The same model, but deterministic: no fresh noise, so the steps can be far larger.' },
  { id: 'flow', name: 'Flow matching', blurb: 'Noise and data are joined by straight lines and the sampler follows the learned velocity. Straighter paths survive big steps best.' },
];

/**
 * One generation run. Call step() until it returns false.
 * guidance w: prediction = uncond + w (cond - uncond). w = 1 is the plain conditional model;
 * `pool` (all shapes mixed) is the unconditional model and is only evaluated when w != 1.
 */
export class Sampler {
  constructor({ sampler = 'ddim', steps = 30, n = 2000, targets, pool = null, guidance = 1, seed = 1 }) {
    this.kind = sampler;
    this.steps = steps;
    this.n = n;
    this.targets = targets;
    this.m = targets.length / 3;
    this.pool = pool;
    this.guidance = guidance;
    this.rnd = rng(seed);
    this.x = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) this.x[i] = this.rnd.normal();
    this.x0 = new Float32Array(n * 3);
    this.u0 = new Float32Array(n * 3);
    this.i = 0; // steps done
    this.evals = 0; // denoiser evaluations
    this.t = 1;
  }

  _predict(scale, sigma) {
    denoise(this.x, this.n, this.targets, this.m, scale, sigma, this.x0);
    this.evals++;
    if (this.pool && this.guidance !== 1) {
      denoise(this.x, this.n, this.pool, this.pool.length / 3, scale, sigma, this.u0);
      this.evals++;
      const w = this.guidance;
      for (let i = 0; i < this.n * 3; i++) this.x0[i] = this.u0[i] + w * (this.x0[i] - this.u0[i]);
    }
  }

  /** Advance one step. Returns false when the run is complete. */
  step() {
    if (this.i >= this.steps) return false;
    const t = 1 - this.i / this.steps;
    const t2 = 1 - (this.i + 1) / this.steps;
    const { x, x0, n } = this;
    if (this.kind === 'flow') {
      this._predict(1 - t, Math.max(t, 1e-4));
      const k = (t2 - t) / Math.max(t, 1e-4); // Euler step along v = (x - x0) / t
      for (let i = 0; i < n * 3; i++) x[i] += k * (x[i] - x0[i]);
    } else {
      const c = schedule(t);
      const d = schedule(t2);
      this._predict(c.a, c.s);
      let sig = 0;
      if (this.kind === 'ddpm' && t2 > 0) sig = Math.sqrt(Math.max(0, ((d.s * d.s) / (c.s * c.s)) * (1 - c.ab / d.ab)));
      const dir = Math.sqrt(Math.max(0, d.s * d.s - sig * sig));
      for (let i = 0; i < n * 3; i++) {
        const eps = (x[i] - c.a * x0[i]) / c.s;
        x[i] = d.a * x0[i] + dir * eps + (sig ? sig * this.rnd.normal() : 0);
      }
    }
    this.i++;
    this.t = t2;
    return this.i < this.steps;
  }

  run() {
    while (this.step());
    return this;
  }
}

/**
 * Quality of a particle cloud against the target shape.
 * sharpness = mean distance from a particle to its nearest target point (lower is sharper);
 * coverage = share of target points that have a particle within `radius` (higher is more complete).
 */
export function quality(x, n, targets, radius = 0.07) {
  const m = targets.length / 3;
  const near = new Float32Array(m).fill(Infinity);
  let sum = 0;
  for (let p = 0; p < n; p++) {
    let best = Infinity;
    for (let i = 0; i < m; i++) {
      const dx = x[p * 3] - targets[i * 3];
      const dy = x[p * 3 + 1] - targets[i * 3 + 1];
      const dz = x[p * 3 + 2] - targets[i * 3 + 2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < best) best = d;
      if (d < near[i]) near[i] = d;
    }
    sum += Math.sqrt(best);
  }
  let covered = 0;
  for (let i = 0; i < m; i++) if (near[i] < radius * radius) covered++;
  return { sharpness: sum / n, coverage: covered / m };
}
