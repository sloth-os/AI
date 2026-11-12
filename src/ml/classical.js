// Six classic classifiers written from scratch, with one shared interface, and the benchmark loop of
// the Model Bake-off. Binary labels (0 / 1), any number of features d, rows stored flat:
// example i is X[i*d .. i*d+d). No dependencies beyond the MLP next door; runs in the browser and in Node.
//
//   model.fit(X, y, n)                 train on n rows
//   model.fitSteps(X, y, n)            the same as a generator that yields between chunks of work
//   model.predictProba(X, n, out)      P(class 1) for n rows

import { MLP, rng, makeDataset, DATASETS } from './mlp.js';

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/** Shared plumbing: a model that trains in one go still offers the generator form. */
class Classifier {
  *fitSteps(X, y, n) {
    this.fit(X, y, n);
  }

  /** Fraction of n rows whose predicted class (P >= 0.5) matches y. */
  score(X, y, n) {
    const p = this.predictProba(X, n);
    let hit = 0;
    for (let i = 0; i < n; i++) if (p[i] >= 0.5 === (y[i] === 1)) hit++;
    return hit / n;
  }
}

// ------------------------------------------------------------------------ logistic regression

/**
 * Logistic regression: P(1 | x) = sigmoid(w·x + b). Fitted by Newton's method (iteratively
 * reweighted least squares) on the cross-entropy plus an L2 penalty (l2/2)·|w|², with step halving.
 */
export class LogisticRegression extends Classifier {
  constructor({ d = 2, l2 = 1, maxIter = 50, tol = 1e-9 } = {}) {
    super();
    this.d = d;
    this.l2 = l2;
    this.maxIter = maxIter;
    this.tol = tol;
    this.w = new Float64Array(d + 1); // w[d] is the bias
    this.iterations = 0;
  }

  _objective(X, y, n, w) {
    const d = this.d;
    let s = 0;
    for (let i = 0; i < n; i++) {
      let z = w[d];
      for (let f = 0; f < d; f++) z += w[f] * X[i * d + f];
      // log(1 + e^z) - y·z, written to stay finite for large |z|
      s += (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) - y[i] * z;
    }
    for (let f = 0; f < d; f++) s += 0.5 * this.l2 * w[f] * w[f];
    return s;
  }

  fit(X, y, n) {
    const d = this.d;
    const m = d + 1;
    const w = this.w.fill(0);
    const g = new Float64Array(m);
    const H = new Float64Array(m * m);
    const step = new Float64Array(m);
    const trial = new Float64Array(m);
    const row = new Float64Array(m);
    row[d] = 1;
    let obj = this._objective(X, y, n, w);
    this.iterations = 0;
    for (let it = 0; it < this.maxIter; it++) {
      g.fill(0);
      H.fill(0);
      for (let i = 0; i < n; i++) {
        let z = w[d];
        for (let f = 0; f < d; f++) z += w[f] * (row[f] = X[i * d + f]);
        const p = sigmoid(z);
        const r = p - y[i];
        const s = Math.max(p * (1 - p), 1e-10);
        for (let a = 0; a < m; a++) {
          g[a] += r * row[a];
          for (let b = 0; b <= a; b++) H[a * m + b] += s * row[a] * row[b];
        }
      }
      for (let f = 0; f < d; f++) {
        g[f] += this.l2 * w[f];
        H[f * m + f] += this.l2;
      }
      H[d * m + d] += 1e-9;
      for (let a = 0; a < m; a++) for (let b = 0; b < a; b++) H[b * m + a] = H[a * m + b];
      solve(H, g, step, m);
      // Newton step, halved until the objective goes down
      let scale = 1;
      let next = obj;
      for (let h = 0; h < 12; h++) {
        for (let a = 0; a < m; a++) trial[a] = w[a] - scale * step[a];
        next = this._objective(X, y, n, trial);
        if (next <= obj + 1e-12) break;
        scale *= 0.5;
      }
      let moved = 0;
      for (let a = 0; a < m; a++) {
        moved = Math.max(moved, Math.abs(trial[a] - w[a]));
        w[a] = trial[a];
      }
      obj = next;
      this.iterations = it + 1;
      if (moved < this.tol) break;
    }
    return this;
  }

  predictProba(X, n, out = new Float64Array(n)) {
    const d = this.d;
    const w = this.w;
    for (let i = 0; i < n; i++) {
      let z = w[d];
      for (let f = 0; f < d; f++) z += w[f] * X[i * d + f];
      out[i] = sigmoid(z);
    }
    return out;
  }
}

/** Solve A·x = b for a small dense system by Gaussian elimination with partial pivoting (A and b are overwritten). */
function solve(A, b, x, m) {
  for (let c = 0; c < m; c++) {
    let piv = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(A[r * m + c]) > Math.abs(A[piv * m + c])) piv = r;
    if (piv !== c) {
      for (let k = 0; k < m; k++) {
        const t = A[c * m + k];
        A[c * m + k] = A[piv * m + k];
        A[piv * m + k] = t;
      }
      const t = b[c];
      b[c] = b[piv];
      b[piv] = t;
    }
    const diag = A[c * m + c] || 1e-12;
    for (let r = c + 1; r < m; r++) {
      const f = A[r * m + c] / diag;
      if (f === 0) continue;
      for (let k = c; k < m; k++) A[r * m + k] -= f * A[c * m + k];
      b[r] -= f * b[c];
    }
  }
  for (let r = m - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < m; k++) s -= A[r * m + k] * x[k];
    x[r] = s / (A[r * m + r] || 1e-12);
  }
  return x;
}

// ------------------------------------------------------------------------ k-nearest neighbors

/**
 * k-nearest neighbors: no training at all, the model is the stored data. A prediction scans every
 * stored point (brute force), keeps the k closest and reports the share of class 1 among them.
 */
export class KNearestNeighbors extends Classifier {
  constructor({ d = 2, k = 5 } = {}) {
    super();
    this.d = d;
    this.k = k;
    this.n = 0;
  }

  fit(X, y, n) {
    this.X = X.slice(0, n * this.d);
    this.y = y.slice(0, n);
    this.n = n;
    return this;
  }

  predictProba(Xq, nq, out = new Float64Array(nq)) {
    const d = this.d;
    const n = this.n;
    const X = this.X;
    const y = this.y;
    const k = Math.min(this.k, n);
    const bestD = new Float64Array(k);
    const bestY = new Uint8Array(k);
    for (let q = 0; q < nq; q++) {
      bestD.fill(Infinity);
      let worst = Infinity;
      for (let i = 0; i < n; i++) {
        let dist = 0;
        if (d === 2) {
          const dx = X[i * 2] - Xq[q * 2];
          const dy = X[i * 2 + 1] - Xq[q * 2 + 1];
          dist = dx * dx + dy * dy;
        } else
          for (let f = 0; f < d; f++) {
            const df = X[i * d + f] - Xq[q * d + f];
            dist += df * df;
          }
        if (dist >= worst) continue;
        // insert into the sorted list of the k best so far
        let s = k - 1;
        while (s > 0 && bestD[s - 1] > dist) {
          bestD[s] = bestD[s - 1];
          bestY[s] = bestY[s - 1];
          s--;
        }
        bestD[s] = dist;
        bestY[s] = y[i];
        worst = bestD[k - 1];
      }
      let votes = 0;
      for (let s = 0; s < k; s++) votes += bestY[s];
      out[q] = votes / k;
    }
    return out;
  }
}

// ------------------------------------------------------------------------ Gaussian naive Bayes

/**
 * Gaussian naive Bayes: each class is one bell curve per feature, and the features are assumed
 * independent given the class. Training is a single pass that collects means and variances.
 */
export class GaussianNaiveBayes extends Classifier {
  constructor({ d = 2, smoothing = 1e-9 } = {}) {
    super();
    this.d = d;
    this.smoothing = smoothing;
    this.mean = [new Float64Array(d), new Float64Array(d)];
    this.variance = [new Float64Array(d).fill(1), new Float64Array(d).fill(1)];
    this.prior = [0.5, 0.5];
  }

  fit(X, y, n) {
    const d = this.d;
    const count = [0, 0];
    const sum = [new Float64Array(d), new Float64Array(d)];
    const sq = [new Float64Array(d), new Float64Array(d)];
    const all = new Float64Array(d);
    const allSq = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      const c = y[i];
      count[c]++;
      for (let f = 0; f < d; f++) {
        const v = X[i * d + f];
        sum[c][f] += v;
        sq[c][f] += v * v;
        all[f] += v;
        allSq[f] += v * v;
      }
    }
    let maxVar = 0;
    for (let f = 0; f < d; f++) maxVar = Math.max(maxVar, allSq[f] / n - (all[f] / n) ** 2);
    const eps = this.smoothing * maxVar + 1e-12;
    for (let c = 0; c < 2; c++) {
      const m = Math.max(1, count[c]);
      for (let f = 0; f < d; f++) {
        this.mean[c][f] = sum[c][f] / m;
        this.variance[c][f] = Math.max(0, sq[c][f] / m - this.mean[c][f] ** 2) + eps;
      }
      this.prior[c] = (count[c] + 1e-9) / (n + 2e-9);
    }
    return this;
  }

  predictProba(X, n, out = new Float64Array(n)) {
    const d = this.d;
    const [m0, m1] = this.mean;
    const [v0, v1] = this.variance;
    let base = Math.log(this.prior[1] / this.prior[0]);
    for (let f = 0; f < d; f++) base -= 0.5 * Math.log(v1[f] / v0[f]);
    for (let i = 0; i < n; i++) {
      let z = base; // log P(x, 1) - log P(x, 0)
      for (let f = 0; f < d; f++) {
        const x = X[i * d + f];
        z += ((x - m0[f]) * (x - m0[f])) / (2 * v0[f]) - ((x - m1[f]) * (x - m1[f])) / (2 * v1[f]);
      }
      out[i] = sigmoid(z);
    }
    return out;
  }
}

// ------------------------------------------------------------------------ CART decision tree

/** Row indices sorted by each feature: sorted[f] lists the rows in increasing order of feature f. */
export function argsortFeatures(X, n, d) {
  const sorted = [];
  for (let f = 0; f < d; f++) {
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    idx.sort((a, b) => X[a * d + f] - X[b * d + f]);
    sorted.push(idx);
  }
  return sorted;
}

/**
 * CART classification tree. Every node tries every threshold on every candidate feature and keeps
 * the cut with the lowest Gini impurity; cuts are therefore always parallel to an axis. Grown until
 * the leaves are pure, the depth limit is reached or a leaf would get fewer than minLeaf examples.
 *
 * The rows are sorted once per feature; a split then partitions those sorted lists in place, so
 * growing the whole tree costs about d · n · depth steps.
 *   maxFeatures > 0 : look at only that many randomly chosen features per node (for random forests)
 *   weights         : how many times each row counts (a bootstrap sample), or null
 */
export class DecisionTree extends Classifier {
  constructor({ d = 2, maxDepth = 12, minLeaf = 1, maxFeatures = 0, seed = 1 } = {}) {
    super();
    this.d = d;
    this.maxDepth = maxDepth;
    this.minLeaf = minLeaf;
    this.maxFeatures = maxFeatures;
    this.seed = seed;
    this.count = 0; // nodes
    this.leaves = 0;
    this.depth = 0;
  }

  fit(X, y, n, weights = null, presorted = null) {
    const d = this.d;
    const sorted = presorted ?? argsortFeatures(X, n, d);
    let m = 0;
    const lists = [];
    for (let f = 0; f < d; f++) {
      const src = sorted[f];
      const dst = new Int32Array(n);
      let k = 0;
      for (let i = 0; i < n; i++) if (!weights || weights[src[i]] > 0) dst[k++] = src[i];
      m = k;
      lists.push(dst);
    }
    let total = 0;
    let total1 = 0;
    for (let s = 0; s < m; s++) {
      const i = lists[0][s];
      const w = weights ? weights[i] : 1;
      total += w;
      if (y[i]) total1 += w;
    }
    const cap = Math.max(1, 2 * m);
    this.feature = new Int8Array(cap).fill(-1); // -1 marks a leaf
    this.threshold = new Float64Array(cap);
    this.left = new Int32Array(cap);
    this.right = new Int32Array(cap);
    this.prob = new Float64Array(cap); // share of class 1 among the training rows in the node
    this.count = 0;
    this.leaves = 0;
    this.depth = 0;
    this._t = { X, y, weights, lists, scratch: new Int32Array(m), side: new Uint8Array(n), order: Int32Array.from({ length: d }, (_, f) => f), rnd: rng(this.seed) };
    this._grow(0, m, 0, total, total1);
    this._t = null;
    // keep only the nodes that exist
    this.feature = this.feature.slice(0, this.count);
    this.threshold = this.threshold.slice(0, this.count);
    this.left = this.left.slice(0, this.count);
    this.right = this.right.slice(0, this.count);
    this.prob = this.prob.slice(0, this.count);
    return this;
  }

  _grow(lo, hi, depth, total, total1) {
    const { X, y, weights, lists, scratch, side, order, rnd } = this._t;
    const d = this.d;
    const id = this.count++;
    this.prob[id] = total > 0 ? total1 / total : 0.5;
    if (depth > this.depth) this.depth = depth;
    const minLeaf = this.minLeaf;
    if (depth >= this.maxDepth || total1 === 0 || total1 === total || total < 2 * minLeaf || hi - lo < 2) {
      this.leaves++;
      return id;
    }
    // candidate features: all of them, or a random subset (extended if the subset cannot split the node)
    const mtry = this.maxFeatures > 0 ? Math.min(d, this.maxFeatures) : d;
    if (mtry < d)
      for (let i = d - 1; i > 0; i--) {
        const j = rnd.int(i + 1);
        const t = order[i];
        order[i] = order[j];
        order[j] = t;
      }
    let bestF = -1;
    let bestPos = -1;
    let bestScore = Infinity;
    let bestThr = 0;
    let bestL = 0;
    let bestL1 = 0;
    for (let c = 0; c < d; c++) {
      if (c >= mtry && bestF >= 0) break;
      const f = mtry < d ? order[c] : c;
      const list = lists[f];
      let wl = 0;
      let wl1 = 0;
      for (let s = lo; s < hi - 1; s++) {
        const i = list[s];
        const w = weights ? weights[i] : 1;
        wl += w;
        if (y[i]) wl1 += w;
        const v = X[i * d + f];
        const vn = X[list[s + 1] * d + f];
        if (!(vn > v)) continue; // a threshold only fits between two different values
        const wr = total - wl;
        if (wl < minLeaf || wr < minLeaf) continue;
        const wr1 = total1 - wl1;
        // weighted Gini impurity of the two children (up to a constant factor)
        const score = (wl1 * (wl - wl1)) / wl + (wr1 * (wr - wr1)) / wr;
        if (score < bestScore) {
          bestScore = score;
          bestF = f;
          bestPos = s + 1;
          const mid = (v + vn) / 2;
          bestThr = mid < vn ? mid : v;
          bestL = wl;
          bestL1 = wl1;
        }
      }
    }
    if (bestF < 0) {
      this.leaves++;
      return id;
    }
    // rows left of the cut keep their place in the winning feature's list; the other lists are
    // partitioned so that each child again owns one contiguous, sorted range in every list
    const win = lists[bestF];
    for (let s = lo; s < bestPos; s++) side[win[s]] = 1;
    for (let s = bestPos; s < hi; s++) side[win[s]] = 0;
    for (let f = 0; f < d; f++) {
      if (f === bestF) continue;
      const list = lists[f];
      let a = lo;
      let b = 0;
      for (let s = lo; s < hi; s++) {
        const i = list[s];
        if (side[i]) list[a++] = i;
        else scratch[b++] = i;
      }
      for (let s = 0; s < b; s++) list[a + s] = scratch[s];
    }
    this.feature[id] = bestF;
    this.threshold[id] = bestThr;
    this.left[id] = this._grow(lo, bestPos, depth + 1, bestL, bestL1);
    this.right[id] = this._grow(bestPos, hi, depth + 1, total - bestL, total1 - bestL1);
    return id;
  }

  /** P(class 1) for the single row starting at X[off]. */
  probAt(X, off) {
    const feature = this.feature;
    let node = 0;
    let f = feature[0];
    while (f >= 0) {
      node = X[off + f] <= this.threshold[node] ? this.left[node] : this.right[node];
      f = feature[node];
    }
    return this.prob[node];
  }

  predictProba(X, n, out = new Float64Array(n)) {
    const d = this.d;
    for (let i = 0; i < n; i++) out[i] = this.probAt(X, i * d);
    return out;
  }
}

// ------------------------------------------------------------------------ random forest

/**
 * Random forest: many CART trees, each grown on a bootstrap sample of the rows (drawn with
 * replacement) and allowed to look at only a random subset of the features at every node. The
 * forest's answer is the average of the trees' answers, which cancels much of a single tree's variance.
 */
export class RandomForest extends Classifier {
  constructor({ d = 2, trees = 30, maxDepth = 12, minLeaf = 1, maxFeatures = 0, seed = 1 } = {}) {
    super();
    this.d = d;
    this.size = trees;
    this.maxDepth = maxDepth;
    this.minLeaf = minLeaf;
    this.maxFeatures = maxFeatures || Math.max(1, Math.round(Math.sqrt(d)));
    this.seed = seed;
    this.trees = [];
  }

  *fitSteps(X, y, n) {
    const rnd = rng(this.seed);
    const sorted = argsortFeatures(X, n, this.d);
    const weights = new Uint16Array(n);
    this.trees = [];
    for (let t = 0; t < this.size; t++) {
      weights.fill(0);
      for (let i = 0; i < n; i++) weights[rnd.int(n)]++;
      const tree = new DecisionTree({ d: this.d, maxDepth: this.maxDepth, minLeaf: this.minLeaf, maxFeatures: this.maxFeatures, seed: rnd.int(0x7fffffff) });
      this.trees.push(tree.fit(X, y, n, weights, sorted));
      yield;
    }
  }

  fit(X, y, n) {
    for (const _ of this.fitSteps(X, y, n));
    return this;
  }

  predictProba(X, n, out = new Float64Array(n)) {
    const d = this.d;
    const trees = this.trees;
    const T = trees.length;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += trees[t].probAt(X, i * d);
      out[i] = s / T;
    }
    return out;
  }
}

// ------------------------------------------------------------------------ the MLP as a classifier

/** The multilayer perceptron from mlp.js behind the same interface: a fixed number of epochs of Adam. */
export class MLPClassifier extends Classifier {
  constructor({ d = 2, hidden = [16, 16], activation = 'tanh', lr = 0.03, batch = 16, epochs = 100, l2 = 0, seed = 1 } = {}) {
    super();
    this.d = d;
    this.epochs = epochs;
    this.config = { inputs: d, hidden, activation, lr, batch, l2, outputs: 1, optimizer: 'adam' };
    this.seed = seed;
    this.net = null;
  }

  *fitSteps(X, y, n) {
    const net = (this.net = new MLP({ ...this.config, seed: this.seed }));
    const perEpoch = Math.ceil(n / net.batch);
    for (let e = 0; e < this.epochs; e++) {
      for (let s = 0; s < perEpoch; s++) net.step(X, y, n);
      yield;
    }
  }

  fit(X, y, n) {
    for (const _ of this.fitSteps(X, y, n));
    return this;
  }

  predictProba(X, n, out = new Float64Array(n)) {
    return this.net.predictProba(X, n, out);
  }
}

// ------------------------------------------------------------------------ the bake-off

/**
 * The six contestants with the fixed settings used for every dataset (no per-dataset tuning).
 * `make(seed)` returns a fresh, untrained model.
 */
export const MODELS = [
  { id: 'logreg', name: 'Logistic regression', short: 'LogReg', settings: 'L2 penalty 1, Newton’s method', make: () => new LogisticRegression() },
  { id: 'knn', name: 'k-nearest neighbors', short: 'k-NN', settings: 'k = 5, brute-force search', make: () => new KNearestNeighbors({ k: 5 }) },
  { id: 'nb', name: 'Gaussian naive Bayes', short: 'Naive Bayes', settings: 'one bell curve per class and input', make: () => new GaussianNaiveBayes() },
  { id: 'tree', name: 'Decision tree', short: 'Tree', settings: 'CART, Gini, depth ≤ 12, grown until pure', make: (seed) => new DecisionTree({ seed }) },
  { id: 'forest', name: 'Random forest', short: 'Forest', settings: '30 bootstrapped trees, 1 of 2 inputs per split', make: (seed) => new RandomForest({ trees: 30, seed }) },
  { id: 'mlp', name: 'Neural network (MLP)', short: 'MLP', settings: '2 × 16 tanh units, Adam, 100 epochs', make: (seed) => new MLPClassifier({ seed }) },
];

const clock = () => performance.now();

/** Run fn() often enough for the clock to resolve it; returns milliseconds per call. Yields between rounds. */
function* timeRepeated(fn, now, minMs, maxReps) {
  let reps = 1;
  for (;;) {
    const t0 = now();
    for (let r = 0; r < reps; r++) fn();
    const dt = now() - t0;
    if (dt >= minMs || reps >= maxReps) return dt / reps;
    reps = Math.min(maxReps, Math.max(reps * 2, Math.ceil((reps * minMs * 1.2) / Math.max(dt, minMs / 16))));
    yield;
  }
}

/**
 * Train one model and measure it. A generator, so the caller can spread the work over animation
 * frames: it yields between chunks, and only time spent inside the chunks is counted.
 * @returns {{acc: number, trainMs: number, predictMs: number}} test accuracy, training time, and
 *          prediction time per 1,000 points
 */
export function* measure(model, train, test, o = {}) {
  const now = o.now ?? clock;
  const minMs = o.minMs ?? 2;
  const maxReps = o.maxReps ?? 4096;
  let trainMs = 0;
  let t0 = now();
  for (const _ of model.fitSteps(train.X, train.y, train.n)) {
    trainMs += now() - t0;
    yield;
    t0 = now();
  }
  trainMs += now() - t0;
  yield;
  // faster than the clock can resolve: train again several times and average
  if (trainMs < minMs) trainMs = yield* timeRepeated(() => model.fit(train.X, train.y, train.n), now, minMs, maxReps);
  yield;
  const p = new Float64Array(test.n);
  const perCall = yield* timeRepeated(() => model.predictProba(test.X, test.n, p), now, minMs, maxReps);
  let hit = 0;
  for (let i = 0; i < test.n; i++) if (p[i] >= 0.5 === (test.y[i] === 1)) hit++;
  return { acc: hit / test.n, trainMs, predictMs: (perCall / test.n) * 1000 };
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
const median = (a) => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

/** One cell of the results table: every seed's measurements for one model on one dataset. */
export class Cell {
  constructor() {
    this.acc = [];
    this.trainMs = [];
    this.predictMs = [];
  }

  add(r) {
    this.acc.push(r.acc);
    this.trainMs.push(r.trainMs);
    this.predictMs.push(r.predictMs);
  }

  get runs() {
    return this.acc.length;
  }

  /** Mean test accuracy over the seeds run so far. */
  get accuracy() {
    return this.acc.length ? mean(this.acc) : NaN;
  }

  /** Standard deviation of the accuracy between seeds. */
  get spread() {
    if (this.acc.length < 2) return 0;
    const m = mean(this.acc);
    return Math.sqrt(this.acc.reduce((s, v) => s + (v - m) ** 2, 0) / (this.acc.length - 1));
  }

  /** Typical (median) training time in ms; the median ignores a cold first run. */
  get train() {
    return median(this.trainMs);
  }

  /** Typical (median) prediction time per 1,000 points in ms. */
  get predict() {
    return median(this.predictMs);
  }
}

/**
 * The whole experiment as a generator: for every seed, every dataset and every model, draw fresh
 * training and test data, train, measure. It yields `undefined` between chunks of work and an event
 * `{seed, m, d, model, train, test, result, cell, done, total}` whenever one model has been measured.
 * `cells[m][d]` (a Cell) accumulates the results; it is also the generator's return value.
 *
 * @param {{trainSize?: number, testSize?: number, noise?: number, seeds?: number, baseSeed?: number,
 *          models?: typeof MODELS, datasets?: typeof DATASETS, now?: () => number, minMs?: number}} [o]
 */
export function* bakeoff(o = {}) {
  const models = o.models ?? MODELS;
  const datasets = o.datasets ?? DATASETS;
  const trainSize = o.trainSize ?? 300;
  const testSize = o.testSize ?? 1000;
  const noise = o.noise ?? 0.3;
  const seeds = o.seeds ?? 5;
  const baseSeed = o.baseSeed ?? 1;
  const cells = (o.cells ?? models.map(() => datasets.map(() => new Cell())));
  const total = seeds * models.length * datasets.length;
  let done = 0;
  for (let s = 0; s < seeds; s++) {
    for (let d = 0; d < datasets.length; d++) {
      const seed = baseSeed + s * 7919 + d * 104729;
      const train = makeDataset(datasets[d].id, trainSize, noise, seed);
      const test = makeDataset(datasets[d].id, testSize, noise, seed + 15485863);
      yield;
      for (let m = 0; m < models.length; m++) {
        const model = models[m].make(seed + m + 1);
        const result = yield* measure(model, train, test, o);
        cells[m][d].add(result);
        done++;
        yield { seed: s, m, d, model, train, test, result, cell: cells[m][d], done, total };
      }
    }
  }
  return cells;
}

/** Index of the best model for dataset d (highest mean accuracy), or -1 before any result. */
export function winnerOf(cells, d) {
  let best = -1;
  for (let m = 0; m < cells.length; m++) {
    if (!cells[m][d].runs) continue;
    if (best < 0 || cells[m][d].accuracy > cells[best][d].accuracy) best = m;
  }
  return best;
}
