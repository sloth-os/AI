// Gradient-descent optimisers and 2-parameter test landscapes for the Optimizer Arena.
// Everything is plain math on [x, y] so the 3D scene can show exactly what is computed.

/** @typedef {{id:string, name:string, formula:string, init:()=>object, step:(s:object, p:number[], grad:(p:number[])=>number[], lr:number)=>void}} Optimizer */

/** @type {Optimizer[]} */
export const OPTIMIZERS = [
  {
    id: 'sgd',
    name: 'SGD',
    formula: 'w ← w − η·g',
    init: () => ({}),
    step(_s, p, grad, lr) {
      const g = grad(p);
      p[0] -= lr * g[0];
      p[1] -= lr * g[1];
    },
  },
  {
    id: 'momentum',
    name: 'Momentum',
    formula: 'v ← 0.9·v + g\nw ← w − η·v',
    init: () => ({ v: [0, 0] }),
    step(s, p, grad, lr) {
      const g = grad(p);
      s.v[0] = 0.9 * s.v[0] + g[0];
      s.v[1] = 0.9 * s.v[1] + g[1];
      p[0] -= lr * s.v[0];
      p[1] -= lr * s.v[1];
    },
  },
  {
    id: 'nesterov',
    name: 'Nesterov',
    formula: 'g ← ∇L(w − η·0.9·v)   (look ahead)\nv ← 0.9·v + g\nw ← w − η·v',
    init: () => ({ v: [0, 0] }),
    step(s, p, grad, lr) {
      const g = grad([p[0] - lr * 0.9 * s.v[0], p[1] - lr * 0.9 * s.v[1]]);
      s.v[0] = 0.9 * s.v[0] + g[0];
      s.v[1] = 0.9 * s.v[1] + g[1];
      p[0] -= lr * s.v[0];
      p[1] -= lr * s.v[1];
    },
  },
  {
    id: 'rmsprop',
    name: 'RMSProp',
    formula: 's ← 0.9·s + 0.1·g²\nw ← w − η·g / (√s + ε)',
    init: () => ({ s: [0, 0] }),
    step(st, p, grad, lr) {
      const g = grad(p);
      for (let i = 0; i < 2; i++) {
        st.s[i] = 0.9 * st.s[i] + 0.1 * g[i] * g[i];
        p[i] -= (lr * g[i]) / (Math.sqrt(st.s[i]) + 1e-8);
      }
    },
  },
  {
    id: 'adam',
    name: 'Adam',
    formula: 'm ← 0.9·m + 0.1·g\nv ← 0.999·v + 0.001·g²\nw ← w − η·m̂ / (√v̂ + ε)',
    init: () => ({ m: [0, 0], v: [0, 0], t: 0 }),
    step(st, p, grad, lr) {
      const g = grad(p);
      st.t++;
      for (let i = 0; i < 2; i++) {
        st.m[i] = 0.9 * st.m[i] + 0.1 * g[i];
        st.v[i] = 0.999 * st.v[i] + 0.001 * g[i] * g[i];
        const mh = st.m[i] / (1 - Math.pow(0.9, st.t));
        const vh = st.v[i] / (1 - Math.pow(0.999, st.t));
        p[i] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
      }
    },
  },
];

/**
 * Test landscapes. `lr` holds each optimiser's default step size for that surface: about 60% of the
 * best value found by a sweep over step sizes (a realistic safety margin, the way a practitioner
 * tunes). `tol` is how close to the best loss counts as solved; `curve` shapes the height mapping.
 */
export const SURFACES = [
  {
    id: 'ravine',
    name: 'Ravine',
    blurb: 'A long narrow valley: 100 times steeper across than along. One step size cannot suit both directions.',
    f: ([x, y]) => 0.5 * (x * x + 100 * y * y),
    grad: ([x, y]) => [x, 100 * y],
    domain: [-4, 4, -1, 1],
    start: [-3.6, 0.8],
    minima: [[0, 0]],
    fmin: 0,
    tol: 1e-3,
    curve: 12,
    lr: { sgd: 0.011, momentum: 0.0045, nesterov: 0.006, rmsprop: 0.025, adam: 0.08 },
  },
  {
    id: 'banana',
    name: 'Banana valley',
    blurb: 'The Rosenbrock function: finding the curved valley is easy, following it to the minimum is slow.',
    f: ([x, y]) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    grad: ([x, y]) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    domain: [-2, 2, -1, 3],
    start: [-1.5, 2.2],
    minima: [[1, 1]],
    fmin: 0,
    tol: 1e-3,
    curve: 400,
    lr: { sgd: 0.0011, momentum: 0.0011, nesterov: 0.0006, rmsprop: 0.004, adam: 0.1 },
  },
  {
    id: 'saddle',
    name: 'Saddle',
    blurb: 'Two valleys separated by a saddle point where the gradient vanishes. High-dimensional loss surfaces are full of these.',
    f: ([x, y]) => 0.1 * (x * x - 1) ** 2 + 2 * y * y,
    grad: ([x, y]) => [0.4 * x * (x * x - 1), 4 * y],
    domain: [-1.8, 1.8, -1.6, 1.6],
    start: [0.002, 1.4],
    minima: [[-1, 0], [1, 0]],
    fmin: 0,
    tol: 1e-4,
    curve: 8,
    lr: { sgd: 0.25, momentum: 0.1, nesterov: 0.18, rmsprop: 0.03, adam: 0.045 },
  },
  {
    id: 'four',
    name: 'Four minima',
    blurb: 'Himmelblau’s function has four equally good minima. Where you end up depends on the optimiser’s dynamics, not only on where you start.',
    f: ([x, y]) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    grad: ([x, y]) => [4 * x * (x * x + y - 11) + 2 * (x + y * y - 7), 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)],
    domain: [-5, 5, -5, 5],
    start: [-0.3, -0.9],
    minima: [[3, 2], [-2.805118, 3.131312], [-3.77931, -3.283186], [3.584428, -1.848126]],
    fmin: 0,
    tol: 1e-3,
    curve: 60,
    lr: { sgd: 0.008, momentum: 0.0034, nesterov: 0.0034, rmsprop: 0.02, adam: 0.14 },
  },
  {
    id: 'bumpy',
    name: 'Bumpy bowl',
    blurb: 'A bowl covered in small dips. Plain descent falls into the first one; momentum and gradient noise can carry a ball out again.',
    f: ([x, y]) => 0.35 * (x * x + y * y) + 1.1 * (2 - Math.cos(2.6 * x) - Math.cos(2.6 * y)),
    grad: ([x, y]) => [0.7 * x + 2.86 * Math.sin(2.6 * x), 0.7 * y + 2.86 * Math.sin(2.6 * y)],
    domain: [-5, 5, -5, 5],
    start: [-4.3, 3.9],
    minima: [[0, 0]],
    fmin: 0,
    tol: 5e-3,
    curve: 4,
    lr: { sgd: 0.05, momentum: 0.025, nesterov: 0.08, rmsprop: 0.03, adam: 0.19 },
  },
];

/** Standard normal sample. */
export function gauss() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

/**
 * One optimiser run on one surface.
 * `noise` models mini-batch noise: each gradient is perturbed by noise · (|g| + small floor).
 */
export class Run {
  constructor(surface, optimizer, { start = surface.start, lrScale = 1, noise = 0 } = {}) {
    this.surface = surface;
    this.optimizer = optimizer;
    this.lr = surface.lr[optimizer.id] * lrScale;
    this.noise = noise;
    this.p = [...start];
    this.state = optimizer.init();
    this.steps = 0;
    this.loss = surface.f(this.p);
    this.best = this.loss;
    this.history = [[0, this.loss]];
    this.path = [[...this.p, this.loss]];
    this.solvedAt = this.loss - surface.fmin < surface.tol ? 0 : null;
    this.diverged = false;
    const [x0, x1, y0, y1] = surface.domain;
    this.floor = 0.02 * Math.hypot(...surface.grad([x0 + (x1 - x0) * 0.8, y0 + (y1 - y0) * 0.8]));
    this.grad = (q) => {
      const g = surface.grad(q);
      if (this.noise > 0) {
        const s = this.noise * (Math.hypot(g[0], g[1]) + this.floor);
        g[0] += s * gauss();
        g[1] += s * gauss();
      }
      return g;
    };
  }

  step() {
    if (this.diverged) return;
    this.optimizer.step(this.state, this.p, this.grad, this.lr);
    this.steps++;
    const [x0, x1, y0, y1] = this.surface.domain;
    const w = x1 - x0;
    const hgt = y1 - y0;
    if (!Number.isFinite(this.p[0]) || !Number.isFinite(this.p[1]) || this.p[0] < x0 - w || this.p[0] > x1 + w || this.p[1] < y0 - hgt || this.p[1] > y1 + hgt) {
      this.diverged = true;
      return;
    }
    this.loss = this.surface.f(this.p);
    this.best = Math.min(this.best, this.loss);
    this.history.push([this.steps, this.loss]);
    this.path.push([this.p[0], this.p[1], this.loss]);
    if (this.solvedAt == null && this.loss - this.surface.fmin < this.surface.tol) this.solvedAt = this.steps;
  }
}
