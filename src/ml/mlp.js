// A small multilayer perceptron with backpropagation, written from scratch, and the 2D toy datasets
// shared by the Neural Network Lab and the Model Bake-off. Typed arrays only, no dependencies, so the
// same file runs in the browser and in Node.

/** Deterministic pseudo-random generator (mulberry32). `const rnd = rng(42); rnd()` -> [0, 1). */
export function rng(seed = 1) {
  let a = seed >>> 0;
  const f = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  f.range = (lo, hi) => lo + (hi - lo) * f();
  f.int = (n) => Math.floor(f() * n);
  f.normal = () => {
    let u = 0;
    while (u === 0) u = f();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * f());
  };
  return f;
}

// ------------------------------------------------------------------------------------ datasets

/** Every generated point lies inside [-DATA_LIMIT, DATA_LIMIT]²; plot the square [-DOMAIN, DOMAIN]². */
export const DATA_LIMIT = 1.25;
export const DOMAIN = 1.3;

const SPIRAL_TURNS = 1.5;

/**
 * Two-class toy problems. `gen(c, rnd, out)` writes one clean point of class c (0 or 1) into out[0..1];
 * `jitter` is the standard deviation of the Gaussian position noise at noise level 1.
 */
export const DATASETS = [
  {
    id: 'linear',
    name: 'Linear',
    blurb: 'Two groups on either side of a tilted straight line.',
    jitter: 0.45,
    gen(c, rnd, out) {
      for (;;) {
        const x = rnd.range(-1, 1);
        const y = rnd.range(-1, 1);
        const s = 0.8 * x + 0.6 * y;
        if (Math.abs(s) < 0.1 || (s > 0 ? 1 : 0) !== c) continue;
        out[0] = x;
        out[1] = y;
        return;
      }
    },
  },
  {
    id: 'moons',
    name: 'Two moons',
    blurb: 'Two interlocking half rings: no straight line separates them.',
    jitter: 0.35,
    gen(c, rnd, out) {
      const t = rnd() * Math.PI;
      const x = c ? 1 - Math.cos(t) : Math.cos(t);
      const y = c ? 0.5 - Math.sin(t) : Math.sin(t);
      out[0] = (x - 0.5) * 0.72;
      out[1] = (y - 0.25) * 0.72;
    },
  },
  {
    id: 'circles',
    name: 'Circles',
    blurb: 'A disc of one class inside a ring of the other.',
    jitter: 0.3,
    gen(c, rnd, out) {
      const r = c ? 0.4 * Math.sqrt(rnd()) : Math.sqrt(0.5184 + rnd() * 0.4816);
      const a = rnd() * Math.PI * 2;
      out[0] = r * Math.cos(a);
      out[1] = r * Math.sin(a);
    },
  },
  {
    id: 'spiral',
    name: 'Spiral',
    blurb: 'Two arms wound around each other one and a half times.',
    jitter: 0.12,
    gen(c, rnd, out) {
      const t = Math.pow(rnd(), 0.65);
      const r = 0.1 + 0.9 * t;
      const a = SPIRAL_TURNS * 2 * Math.PI * t + c * Math.PI;
      out[0] = r * Math.cos(a);
      out[1] = r * Math.sin(a);
    },
  },
  {
    id: 'xor',
    name: 'XOR',
    blurb: 'Four squares in a checkerboard: the class depends on both inputs together.',
    jitter: 0.4,
    gen(c, rnd, out) {
      for (;;) {
        const x = rnd.range(-1, 1);
        const y = rnd.range(-1, 1);
        if (Math.abs(x) < 0.08 || Math.abs(y) < 0.08 || (x * y > 0 ? 1 : 0) !== c) continue;
        out[0] = x;
        out[1] = y;
        return;
      }
    },
  },
];

/**
 * Sample n points (classes alternate, so every prefix is balanced). `noise` in [0, 1] scales the
 * position jitter of the dataset; points pushed outside the plotting limit are drawn again.
 * @returns {{X: Float64Array, y: Uint8Array, n: number, d: 2}}  X is row-major: point i is X[2i], X[2i+1].
 */
export function makeDataset(id, n, noise = 0, seed = 1) {
  const def = DATASETS.find((s) => s.id === id);
  if (!def) throw new Error('unknown dataset ' + id);
  const rnd = rng(seed);
  const X = new Float64Array(n * 2);
  const y = new Uint8Array(n);
  const p = [0, 0];
  const sd = def.jitter * noise;
  for (let i = 0; i < n; i++) {
    const c = i & 1;
    let px, py;
    do {
      def.gen(c, rnd, p);
      px = p[0] + (sd > 0 ? sd * rnd.normal() : 0);
      py = p[1] + (sd > 0 ? sd * rnd.normal() : 0);
    } while (Math.abs(px) > DATA_LIMIT || Math.abs(py) > DATA_LIMIT);
    X[i * 2] = px;
    X[i * 2 + 1] = py;
    y[i] = c;
  }
  return { X, y, n, d: 2 };
}

/** Split a dataset into its first nTrain points and the rest (views, no copy). */
export function splitDataset(data, nTrain) {
  const d = data.d ?? 2;
  const k = Math.max(0, Math.min(data.n, nTrain));
  return {
    train: { X: data.X.subarray(0, k * d), y: data.y.subarray(0, k), n: k, d },
    test: { X: data.X.subarray(k * d), y: data.y.subarray(k), n: data.n - k, d },
  };
}

// ------------------------------------------------------------------------------------ the network

/**
 * Fully connected network: inputs -> hidden layers (tanh or ReLU) -> output.
 *   outputs = 1  : one sigmoid unit, binary cross-entropy, labels 0 / 1
 *   outputs >= 2 : softmax, categorical cross-entropy, labels 0 .. outputs-1
 * Trained by mini-batch gradient descent (plain SGD, SGD with momentum, or Adam) with optional L2.
 *
 * Everything the visualization needs is public:
 *   sizes[l]            units in layer l (layer 0 is the input)
 *   W[l][j * nin + i]   weight from unit i of layer l to unit j of layer l+1;  b[l][j] its bias
 *   a[l][i]             activation of unit i in layer l for the last forward pass (a[0] = the input)
 *   z[l][j]             weighted sum (before the squash) of unit j in layer l+1
 *   d[l][j]             error signal dLoss/dz of unit j in layer l+1 for the last backward pass
 *   gW[l], gb[l]        gradient used by the last update (mean over the mini-batch, L2 included)
 */
export class MLP {
  /**
   * @param {{inputs?: number, hidden?: number[], outputs?: number, activation?: 'tanh'|'relu',
   *   optimizer?: 'adam'|'sgd', lr?: number, momentum?: number, l2?: number, batch?: number, seed?: number}} [o]
   */
  constructor(o = {}) {
    this.inputs = o.inputs ?? 2;
    this.hidden = [...(o.hidden ?? [8, 8])];
    this.outputs = o.outputs ?? 1;
    this.activation = o.activation ?? 'tanh';
    this.optimizer = o.optimizer ?? 'adam';
    this.lr = o.lr ?? 0.03;
    this.momentum = o.momentum ?? 0;
    this.l2 = o.l2 ?? 0;
    this.batch = o.batch ?? 16;
    this.seed = o.seed ?? 1;
    this.beta1 = 0.9;
    this.beta2 = 0.999;

    this.sizes = [this.inputs, ...this.hidden, this.outputs];
    this.L = this.sizes.length - 1; // number of weight layers
    this.W = [];
    this.b = [];
    this.gW = [];
    this.gb = [];
    this.z = [];
    this.d = [];
    this.a = [new Float64Array(this.inputs)];
    // optimizer state: first moment / velocity and second moment
    this.mW = [];
    this.mb = [];
    this.vW = [];
    this.vb = [];
    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l];
      const nout = this.sizes[l + 1];
      this.W.push(new Float64Array(nin * nout));
      this.b.push(new Float64Array(nout));
      this.gW.push(new Float64Array(nin * nout));
      this.gb.push(new Float64Array(nout));
      this.mW.push(new Float64Array(nin * nout));
      this.mb.push(new Float64Array(nout));
      this.vW.push(new Float64Array(nin * nout));
      this.vb.push(new Float64Array(nout));
      this.z.push(new Float64Array(nout));
      this.d.push(new Float64Array(nout));
      this.a.push(new Float64Array(nout));
    }
    this.out = this.a[this.L];
    this.paramCount = this.W.reduce((s, w, l) => s + w.length + this.b[l].length, 0);
    this.reset(this.seed);
  }

  /**
   * (Re)initialize from a seed. Weights: Glorot uniform before tanh / sigmoid / softmax units, He
   * normal before ReLU units. Biases: uniform in ±1/√fan-in, which spreads the units' "folds" over
   * the input square instead of stacking them all on the origin.
   */
  reset(seed = this.seed) {
    this.seed = seed;
    const rnd = (this.rnd = rng(seed));
    for (let l = 0; l < this.L; l++) {
      const nin = this.sizes[l];
      const nout = this.sizes[l + 1];
      const W = this.W[l];
      const relu = this.activation === 'relu' && l < this.L - 1;
      const lim = Math.sqrt(6 / (nin + nout));
      const sd = Math.sqrt(2 / nin);
      for (let k = 0; k < W.length; k++) W[k] = relu ? sd * rnd.normal() : rnd.range(-lim, lim);
      const bl = 1 / Math.sqrt(nin);
      for (let j = 0; j < nout; j++) this.b[l][j] = rnd.range(-bl, bl);
      this.gW[l].fill(0);
      this.gb[l].fill(0);
      this.mW[l].fill(0);
      this.mb[l].fill(0);
      this.vW[l].fill(0);
      this.vb[l].fill(0);
      this.d[l].fill(0);
    }
    this.steps = 0; // mini-batch updates so far
    this.seen = 0; // training examples processed so far
    this.pending = 0; // examples in the gradient that has not been applied yet
    this.batchLoss = NaN;
    this.order = null;
    this.cursor = 0;
    this.from = 0;
    this.to = 0;
    return this;
  }

  /**
   * Forward pass for one example read from x[off .. off+inputs). Fills a[] and z[].
   * @returns {Float64Array} the output layer: [p] for a sigmoid output, class probabilities for softmax.
   */
  forward(x, off = 0) {
    const L = this.L;
    const relu = this.activation === 'relu';
    const a0 = this.a[0];
    for (let i = 0; i < a0.length; i++) a0[i] = x[off + i];
    for (let l = 0; l < L; l++) {
      const W = this.W[l];
      const b = this.b[l];
      const ain = this.a[l];
      const aout = this.a[l + 1];
      const z = this.z[l];
      const nin = ain.length;
      const nout = aout.length;
      const hiddenLayer = l < L - 1;
      for (let j = 0, k = 0; j < nout; j++) {
        let s = b[j];
        for (let i = 0; i < nin; i++, k++) s += W[k] * ain[i];
        z[j] = s;
        if (hiddenLayer) aout[j] = relu ? (s > 0 ? s : 0) : Math.tanh(s);
      }
    }
    const out = this.out;
    const zo = this.z[L - 1];
    if (out.length === 1) out[0] = 1 / (1 + Math.exp(-zo[0]));
    else {
      let m = -Infinity;
      for (let k = 0; k < out.length; k++) if (zo[k] > m) m = zo[k];
      let sum = 0;
      for (let k = 0; k < out.length; k++) sum += out[k] = Math.exp(zo[k] - m);
      for (let k = 0; k < out.length; k++) out[k] /= sum;
    }
    return out;
  }

  /** P(class 1) for one example (the height of the decision surface). */
  prob1(x, off = 0) {
    const out = this.forward(x, off);
    return out.length === 1 ? out[0] : out[1];
  }

  /**
   * Backward pass for the example of the last forward(): adds its gradient to gW / gb and leaves the
   * error signals in d[]. Call zeroGrad() first when starting a new batch.
   * @returns {number} the cross-entropy loss of this example
   */
  backward(target) {
    const L = this.L;
    const relu = this.activation === 'relu';
    const out = this.out;
    const dOut = this.d[L - 1];
    let loss;
    // sigmoid + binary cross-entropy and softmax + cross-entropy share the same simple form: dL/dz = p - y
    if (out.length === 1) {
      dOut[0] = out[0] - target;
      loss = -Math.log(Math.max(1e-12, target ? out[0] : 1 - out[0]));
    } else {
      for (let k = 0; k < out.length; k++) dOut[k] = out[k] - (k === target ? 1 : 0);
      loss = -Math.log(Math.max(1e-12, out[target]));
    }
    for (let l = L - 1; l >= 0; l--) {
      const W = this.W[l];
      const gW = this.gW[l];
      const gb = this.gb[l];
      const ain = this.a[l];
      const d = this.d[l];
      const nin = ain.length;
      const nout = d.length;
      if (l === 0) {
        for (let j = 0, k = 0; j < nout; j++) {
          const dj = d[j];
          gb[j] += dj;
          for (let i = 0; i < nin; i++, k++) gW[k] += dj * ain[i];
        }
      } else {
        const dPrev = this.d[l - 1];
        dPrev.fill(0);
        for (let j = 0, k = 0; j < nout; j++) {
          const dj = d[j];
          gb[j] += dj;
          for (let i = 0; i < nin; i++, k++) {
            gW[k] += dj * ain[i]; // blame for the weight: error at its target times activity at its source
            dPrev[i] += dj * W[k]; // blame passed back to the source unit
          }
        }
        // through the squash of the source layer: tanh' = 1 - a², relu' = 1 where the unit is active
        for (let i = 0; i < nin; i++) dPrev[i] *= relu ? (ain[i] > 0 ? 1 : 0) : 1 - ain[i] * ain[i];
      }
    }
    return loss;
  }

  zeroGrad() {
    for (let l = 0; l < this.L; l++) {
      this.gW[l].fill(0);
      this.gb[l].fill(0);
    }
  }

  /**
   * Gradient of the mean loss over the examples idx[from .. to) of (X, y), L2 penalty included, left
   * in gW / gb without touching the weights. idx = null uses rows from .. to directly.
   * @returns {number} mean loss of the batch
   */
  batchGradient(X, y, idx, from, to) {
    const m = to - from;
    if (m <= 0) return this.batchLoss;
    const nIn = this.inputs;
    this.zeroGrad();
    let loss = 0;
    for (let s = from; s < to; s++) {
      const i = idx ? idx[s] : s;
      this.forward(X, i * nIn);
      loss += this.backward(y[i]);
    }
    const inv = 1 / m;
    const l2 = this.l2;
    for (let l = 0; l < this.L; l++) {
      const gW = this.gW[l];
      const gb = this.gb[l];
      const W = this.W[l];
      for (let k = 0; k < gW.length; k++) gW[k] = gW[k] * inv + l2 * W[k];
      for (let j = 0; j < gb.length; j++) gb[j] *= inv;
    }
    this.pending = m;
    this.batchLoss = loss * inv;
    return this.batchLoss;
  }

  /** Move every parameter against its gradient (gW / gb) with the configured optimizer. */
  applyGradient() {
    const lr = this.lr;
    this.steps++;
    this.seen += this.pending;
    this.pending = 0;
    if (this.optimizer === 'adam') {
      const b1 = this.beta1;
      const b2 = this.beta2;
      const c1 = 1 - Math.pow(b1, this.steps);
      const c2 = 1 - Math.pow(b2, this.steps);
      const stepSize = (lr * Math.sqrt(c2)) / c1;
      for (let l = 0; l < this.L; l++) {
        adam(this.W[l], this.gW[l], this.mW[l], this.vW[l], stepSize, b1, b2);
        adam(this.b[l], this.gb[l], this.mb[l], this.vb[l], stepSize, b1, b2);
      }
    } else {
      const mu = this.momentum;
      for (let l = 0; l < this.L; l++) {
        sgd(this.W[l], this.gW[l], this.mW[l], lr, mu);
        sgd(this.b[l], this.gb[l], this.mb[l], lr, mu);
      }
    }
  }

  /**
   * One update from the examples idx[from .. to): batchGradient() then applyGradient().
   * @returns {number} mean loss of the batch before the update
   */
  trainBatch(X, y, idx, from, to) {
    if (to - from <= 0) return this.batchLoss;
    const loss = this.batchGradient(X, y, idx, from, to);
    this.applyGradient();
    return loss;
  }

  /**
   * Choose the next mini-batch of a data set with n rows: afterward it is order[from .. to). The
   * network walks through a shuffled order and reshuffles at the end of every epoch.
   */
  nextBatch(n) {
    if (!this.order || this.order.length !== n) {
      this.order = new Uint32Array(n);
      for (let i = 0; i < n; i++) this.order[i] = i;
      this.cursor = n;
    }
    if (this.cursor >= n) {
      const o = this.order;
      for (let i = n - 1; i > 0; i--) {
        const j = this.rnd.int(i + 1);
        const t = o[i];
        o[i] = o[j];
        o[j] = t;
      }
      this.cursor = 0;
    }
    this.from = this.cursor;
    this.to = Math.min(n, this.cursor + Math.max(1, this.batch | 0));
    this.cursor = this.to;
  }

  /**
   * One mini-batch step over (X, y) with n rows.
   * @returns {number} mean loss of the mini-batch
   */
  step(X, y, n) {
    this.nextBatch(n);
    return this.trainBatch(X, y, this.order, this.from, this.to);
  }

  /** Train for whole epochs (convenience for scripts and tests). */
  fit(X, y, n, epochs = 100) {
    const stepsPerEpoch = Math.ceil(n / Math.max(1, this.batch | 0));
    for (let s = 0; s < epochs * stepsPerEpoch; s++) this.step(X, y, n);
    return this;
  }

  /** Mean cross-entropy and accuracy over n rows of (X, y). `out` receives {loss, acc}. */
  evaluate(X, y, n, out = {}) {
    const nIn = this.inputs;
    let loss = 0;
    let hit = 0;
    for (let i = 0; i < n; i++) {
      const o = this.forward(X, i * nIn);
      const t = y[i];
      if (o.length === 1) {
        const p = o[0];
        loss -= Math.log(Math.max(1e-12, t ? p : 1 - p));
        if (p >= 0.5 === (t === 1)) hit++;
      } else {
        loss -= Math.log(Math.max(1e-12, o[t]));
        let best = 0;
        for (let k = 1; k < o.length; k++) if (o[k] > o[best]) best = k;
        if (best === t) hit++;
      }
    }
    out.loss = n ? loss / n : NaN;
    out.acc = n ? hit / n : NaN;
    return out;
  }

  /** P(class 1) for n rows of X, written to out. */
  predictProba(X, n, out = new Float64Array(n)) {
    const nIn = this.inputs;
    for (let i = 0; i < n; i++) out[i] = this.prob1(X, i * nIn);
    return out;
  }

  /** Euclidean length of the last gradient (all layers). */
  gradNorm() {
    let s = 0;
    for (let l = 0; l < this.L; l++) {
      for (const g of this.gW[l]) s += g * g;
      for (const g of this.gb[l]) s += g * g;
    }
    return Math.sqrt(s);
  }

  /** Largest |weight| in the network (biases excluded). */
  maxAbsWeight() {
    let m = 0;
    for (let l = 0; l < this.L; l++) for (const w of this.W[l]) if (Math.abs(w) > m) m = Math.abs(w);
    return m;
  }
}

function adam(p, g, m, v, stepSize, b1, b2) {
  for (let k = 0; k < p.length; k++) {
    const gk = g[k];
    m[k] = b1 * m[k] + (1 - b1) * gk;
    v[k] = b2 * v[k] + (1 - b2) * gk * gk;
    p[k] -= (stepSize * m[k]) / (Math.sqrt(v[k]) + 1e-8);
  }
}

function sgd(p, g, vel, lr, mu) {
  if (mu > 0) {
    for (let k = 0; k < p.length; k++) {
      vel[k] = mu * vel[k] + g[k];
      p[k] -= lr * vel[k];
    }
  } else for (let k = 0; k < p.length; k++) p[k] -= lr * g[k];
}
