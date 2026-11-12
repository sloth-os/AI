// Nearest-neighbor search four ways, for the "index race" of the Embedding Space lab:
//   FlatIndex   exact search: compare the query with every vector
//   IVFIndex    inverted file: k-means cells, scan only the `nprobe` closest cells
//   HNSWIndex   hierarchical navigable small world graph: greedy walk with a candidate list of `ef`
//   LSHIndex    random-hyperplane hashing: `bits` sign bits per table, look only in matching buckets
// All of them are real implementations working on Float32Array rows; the lab measures recall@k,
// distance computations and time per query with `benchmark` below. Pure JS, no rendering.

/** Deterministic pseudo-random generator (mulberry32) with a normal() sampler. */
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

/** Squared Euclidean distance between row i of `data` and the vector at offset qo of `q`. */
export function dist2(data, i, q, qo, dim) {
  let s = 0;
  const o = i * dim;
  for (let j = 0; j < dim; j++) {
    const d = data[o + j] - q[qo + j];
    s += d * d;
  }
  return s;
}

/**
 * Synthetic "embedding-like" data: unit vectors in `dim` dimensions that form clusters of unequal
 * size. Cluster centers vary most along the first three axes (as real embeddings vary most along
 * a few directions), and each cluster is spread along a handful of its own directions.
 * @returns {{n:number, dim:number, data:Float32Array, label:Uint16Array, clusters:number, sample:(rand:()=>number, out:Float32Array, o:number)=>number}}
 */
export function makeDataset({ n = 20000, dim = 32, clusters = 48, spread = 0.55, intrinsic = 6, seed = 1 } = {}) {
  const rand = rng(seed);
  const centers = new Float32Array(clusters * dim);
  const basis = new Float32Array(clusters * intrinsic * dim);
  const weight = new Float32Array(clusters);
  let wsum = 0;
  for (let c = 0; c < clusters; c++) {
    let norm = 0;
    for (let j = 0; j < dim; j++) {
      const v = rand.normal() * (j < 3 ? 2.6 : 1);
      centers[c * dim + j] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    for (let j = 0; j < dim; j++) centers[c * dim + j] /= norm;
    for (let b = 0; b < intrinsic; b++) {
      // direction b of this cluster, shorter for later directions
      const len = (spread / Math.sqrt(dim)) * Math.pow(0.82, b);
      for (let j = 0; j < dim; j++) basis[(c * intrinsic + b) * dim + j] = rand.normal() * len;
    }
    weight[c] = 0.35 + rand() * rand() * 2.2; // unequal cluster sizes
    wsum += weight[c];
  }
  const cdf = new Float32Array(clusters);
  let acc = 0;
  for (let c = 0; c < clusters; c++) {
    acc += weight[c] / wsum;
    cdf[c] = acc;
  }
  /** Draw one point into out[o .. o + dim); returns its cluster. */
  const sample = (r, out, o) => {
    const u = r();
    let c = 0;
    while (c < clusters - 1 && cdf[c] < u) c++;
    for (let j = 0; j < dim; j++) out[o + j] = centers[c * dim + j] + r.normal() * (0.12 * spread) / Math.sqrt(dim);
    for (let b = 0; b < intrinsic; b++) {
      const z = r.normal();
      const bo = (c * intrinsic + b) * dim;
      for (let j = 0; j < dim; j++) out[o + j] += z * basis[bo + j];
    }
    let norm = 0;
    for (let j = 0; j < dim; j++) norm += out[o + j] * out[o + j];
    norm = Math.sqrt(norm);
    for (let j = 0; j < dim; j++) out[o + j] /= norm;
    return c;
  };
  const data = new Float32Array(n * dim);
  const label = new Uint16Array(n);
  for (let i = 0; i < n; i++) label[i] = sample(rand, data, i * dim);
  return { n, dim, data, label, clusters, sample };
}

/** Fresh query vectors from the same distribution as the data (they are not in the data set). */
export function makeQueries(ds, count, seed = 99) {
  const rand = rng(seed);
  const q = new Float32Array(count * ds.dim);
  for (let i = 0; i < count; i++) ds.sample(rand, q, i * ds.dim);
  return q;
}

/** The k best (smallest distance) candidates seen so far, kept sorted. */
export class TopK {
  constructor(k) {
    this.k = k;
    this.ids = new Int32Array(k).fill(-1);
    this.d = new Float32Array(k).fill(Infinity);
    this.size = 0;
  }

  reset() {
    this.size = 0;
    this.ids.fill(-1);
    this.d.fill(Infinity);
  }

  /** Largest distance that would still be accepted. */
  bound() {
    return this.size < this.k ? Infinity : this.d[this.k - 1];
  }

  push(id, dist) {
    if (this.size === this.k && dist >= this.d[this.k - 1]) return;
    let i = this.size < this.k ? this.size++ : this.k - 1;
    while (i > 0 && this.d[i - 1] > dist) {
      this.d[i] = this.d[i - 1];
      this.ids[i] = this.ids[i - 1];
      i--;
    }
    this.d[i] = dist;
    this.ids[i] = id;
  }
}

/**
 * Optional record of what a search touched, for the animation. `ids` = vectors whose distance
 * was computed, in order; `from` = the graph node whose links were being followed (-1 otherwise);
 * `group` = cell / table / graph layer the touch belongs to.
 */
export class Trace {
  constructor(capacity = 4096) {
    this.ids = new Int32Array(capacity);
    this.from = new Int32Array(capacity);
    this.group = new Int16Array(capacity);
    this.n = 0;
    this.groups = []; // IVF: probed cells in order; LSH: table numbers; graph: layers
  }

  reset() {
    this.n = 0;
    this.groups.length = 0;
  }

  add(id, from, group) {
    if (this.n === this.ids.length) {
      const grow = (arr, Type) => {
        const next = new Type(arr.length * 2);
        next.set(arr);
        return next;
      };
      this.ids = grow(this.ids, Int32Array);
      this.from = grow(this.from, Int32Array);
      this.group = grow(this.group, Int16Array);
    }
    this.ids[this.n] = id;
    this.from[this.n] = from;
    this.group[this.n] = group;
    this.n++;
  }
}

// ------------------------------------------------------------------------------------ flat

export class FlatIndex {
  constructor(ds) {
    this.ds = ds;
    this.name = 'Flat';
  }

  /**
   * @param {Float32Array} q query vectors, @param {number} qo offset of the query in q
   * @param {TopK} top receives the result, @returns {number} distance computations
   */
  search(q, qo, top, _opts, trace) {
    const { data, n, dim } = this.ds;
    top.reset();
    for (let i = 0; i < n; i++) top.push(i, dist2(data, i, q, qo, dim));
    if (trace) for (let i = 0; i < n; i++) trace.add(i, -1, 0);
    return n;
  }
}

// ------------------------------------------------------------------------------------ IVF

export class IVFIndex {
  constructor(ds, { cells = 128, iters = 8, sample = 6000, seed = 3 } = {}) {
    this.ds = ds;
    this.name = 'IVF';
    this.cells = cells;
    this.iters = iters;
    this.sampleSize = Math.min(sample, ds.n);
    this.seed = seed;
    this.centroids = new Float32Array(cells * ds.dim);
    this.assign = new Int32Array(ds.n); // cell of every vector
    this.start = new Int32Array(cells + 1); // cell c owns order[start[c] .. start[c + 1])
    this.order = new Int32Array(ds.n);
    this.cellDist = new Float32Array(cells);
    this.cellOrder = new Int32Array(cells);
    this.ready = false;
    this.progress = 0;
  }

  nearestCell(data, i) {
    const { dim } = this.ds;
    let best = 0;
    let bd = Infinity;
    for (let c = 0; c < this.cells; c++) {
      const d = dist2(this.centroids, c, data, i * dim, dim);
      if (d < bd) {
        bd = d;
        best = c;
      }
    }
    return best;
  }

  /** k-means on a sample, then assign every vector to its nearest centroid. Yields between chunks of work. */
  *build() {
    const { data, n, dim } = this.ds;
    const rand = rng(this.seed);
    const m = this.sampleSize;
    const pick = new Int32Array(m);
    for (let i = 0; i < m; i++) pick[i] = (rand() * n) | 0;
    for (let c = 0; c < this.cells; c++) {
      const src = pick[(rand() * m) | 0] * dim;
      for (let j = 0; j < dim; j++) this.centroids[c * dim + j] = data[src + j];
    }
    const sums = new Float64Array(this.cells * dim);
    const counts = new Int32Array(this.cells);
    const total = this.iters + 2;
    for (let it = 0; it < this.iters; it++) {
      sums.fill(0);
      counts.fill(0);
      for (let i = 0; i < m; i++) {
        const c = this.nearestCell(data, pick[i]);
        counts[c]++;
        const src = pick[i] * dim;
        for (let j = 0; j < dim; j++) sums[c * dim + j] += data[src + j];
      }
      for (let c = 0; c < this.cells; c++) {
        if (counts[c] === 0) {
          // an empty cell is restarted on a random vector
          const src = pick[(rand() * m) | 0] * dim;
          for (let j = 0; j < dim; j++) this.centroids[c * dim + j] = data[src + j];
        } else for (let j = 0; j < dim; j++) this.centroids[c * dim + j] = sums[c * dim + j] / counts[c];
      }
      this.progress = (it + 1) / total;
      yield;
    }
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      const c = this.nearestCell(data, i);
      this.assign[i] = c;
      counts[c]++;
      if (i % 4000 === 3999) {
        this.progress = (this.iters + (i + 1) / n) / total;
        yield;
      }
    }
    this.start[0] = 0;
    for (let c = 0; c < this.cells; c++) this.start[c + 1] = this.start[c] + counts[c];
    const fill = this.start.slice(0, this.cells);
    for (let i = 0; i < n; i++) this.order[fill[this.assign[i]]++] = i;
    this.progress = 1;
    this.ready = true;
  }

  search(q, qo, top, { nprobe = 4 } = {}, trace) {
    const { data, dim } = this.ds;
    const cells = this.cells;
    top.reset();
    for (let c = 0; c < cells; c++) {
      this.cellDist[c] = dist2(this.centroids, c, q, qo, dim);
      this.cellOrder[c] = c;
    }
    let comps = cells;
    const probes = Math.min(nprobe, cells);
    for (let p = 0; p < probes; p++) {
      // selection sort step: bring the p-th closest cell to position p
      let m = p;
      for (let c = p + 1; c < cells; c++) if (this.cellDist[this.cellOrder[c]] < this.cellDist[this.cellOrder[m]]) m = c;
      const cell = this.cellOrder[m];
      this.cellOrder[m] = this.cellOrder[p];
      this.cellOrder[p] = cell;
      if (trace) trace.groups.push(cell);
      for (let s = this.start[cell]; s < this.start[cell + 1]; s++) {
        const id = this.order[s];
        top.push(id, dist2(data, id, q, qo, dim));
        if (trace) trace.add(id, -1, p);
      }
      comps += this.start[cell + 1] - this.start[cell];
    }
    return comps;
  }
}

// ------------------------------------------------------------------------------------ HNSW

/** Binary heap of (distance, id) pairs on typed arrays. `max` = true keeps the largest on top. */
class Heap {
  constructor(max, capacity = 256) {
    this.sign = max ? -1 : 1;
    this.d = new Float32Array(capacity);
    this.id = new Int32Array(capacity);
    this.size = 0;
  }

  clear() {
    this.size = 0;
  }

  push(dist, id) {
    if (this.size === this.d.length) {
      const d = new Float32Array(this.size * 2);
      d.set(this.d);
      this.d = d;
      const ids = new Int32Array(this.size * 2);
      ids.set(this.id);
      this.id = ids;
    }
    const key = dist * this.sign;
    let i = this.size++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.d[p] * this.sign <= key) break;
      this.d[i] = this.d[p];
      this.id[i] = this.id[p];
      i = p;
    }
    this.d[i] = dist;
    this.id[i] = id;
  }

  topDist() {
    return this.d[0];
  }

  topId() {
    return this.id[0];
  }

  /** Remove the top entry. */
  pop() {
    const n = --this.size;
    if (n === 0) return;
    const dist = this.d[n];
    const id = this.id[n];
    const key = dist * this.sign;
    let i = 0;
    for (;;) {
      let c = i * 2 + 1;
      if (c >= n) break;
      if (c + 1 < n && this.d[c + 1] * this.sign < this.d[c] * this.sign) c++;
      if (this.d[c] * this.sign >= key) break;
      this.d[i] = this.d[c];
      this.id[i] = this.id[c];
      i = c;
    }
    this.d[i] = dist;
    this.id[i] = id;
  }
}

/**
 * HNSW (Malkov and Yashunin). Every vector is a node; node i lives on layers 0..level[i], where
 * the level is drawn at random so that each layer holds about 1/M of the layer below. Links are
 * chosen with the paper's diversity heuristic, which keeps clusters connected to each other.
 */
export class HNSWIndex {
  constructor(ds, { M = 8, efConstruction = 48, seed = 5 } = {}) {
    this.ds = ds;
    this.name = 'HNSW graph';
    this.M = M;
    this.M0 = M * 2;
    this.efConstruction = efConstruction;
    this.mL = 1 / Math.log(M);
    this.rand = rng(seed);
    this.level = new Int8Array(ds.n).fill(-1);
    this.links = [new Int32Array(ds.n * this.M0)]; // links[layer][node * cap + j]
    this.deg = [new Uint8Array(ds.n)];
    this.entry = -1;
    this.maxLevel = -1;
    this.count = 0;
    this.visited = new Uint32Array(ds.n);
    this.stamp = 0;
    this.cand = new Heap(false, 1024);
    this.best = new Heap(true, 1024);
    this.tmpIds = new Int32Array(1024);
    this.tmpD = new Float32Array(1024);
    this.sel = new Int32Array(this.M0 + 1);
    this.comps = 0;
    this.ready = false;
    this.progress = 0;
  }

  cap(layer) {
    return layer === 0 ? this.M0 : this.M;
  }

  /**
   * Best-first search inside one layer. Leaves the `ef` closest nodes found in this.best (a max-heap).
   * `entry` / `entryDist` is where the walk starts.
   */
  searchLayer(q, qo, entry, entryDist, ef, layer, trace) {
    const { data, dim } = this.ds;
    const cap = this.cap(layer);
    const links = this.links[layer];
    const deg = this.deg[layer];
    const visited = this.visited;
    const stamp = ++this.stamp;
    const cand = this.cand;
    const best = this.best;
    cand.clear();
    best.clear();
    cand.push(entryDist, entry);
    best.push(entryDist, entry);
    visited[entry] = stamp;
    while (cand.size) {
      const cd = cand.topDist();
      if (cd > best.topDist() && best.size >= ef) break; // nothing left that could improve the list
      const c = cand.topId();
      cand.pop();
      const base = c * cap;
      for (let j = 0, m = deg[c]; j < m; j++) {
        const e = links[base + j];
        if (visited[e] === stamp) continue;
        visited[e] = stamp;
        const d = dist2(data, e, q, qo, dim);
        this.comps++;
        if (trace) trace.add(e, c, layer);
        if (best.size < ef || d < best.topDist()) {
          cand.push(d, e);
          best.push(d, e);
          if (best.size > ef) best.pop();
        }
      }
    }
  }

  /** Copy this.best into tmpIds / tmpD sorted by increasing distance; returns the count. */
  drainBest() {
    const best = this.best;
    const n = best.size;
    if (n > this.tmpIds.length) {
      this.tmpIds = new Int32Array(n * 2);
      this.tmpD = new Float32Array(n * 2);
    }
    for (let i = n - 1; i >= 0; i--) {
      this.tmpIds[i] = best.topId();
      this.tmpD[i] = best.topDist();
      best.pop();
    }
    return n;
  }

  /**
   * The paper's neighbor selection heuristic: walk the candidates from nearest to farthest and
   * keep one only if it is closer to the new node than to every neighbor already kept.
   * Reads tmpIds / tmpD[0..count), writes this.sel, returns how many were kept.
   */
  selectNeighbors(count, m) {
    const { data, dim } = this.ds;
    let kept = 0;
    for (let i = 0; i < count && kept < m; i++) {
      const e = this.tmpIds[i];
      const de = this.tmpD[i];
      let ok = true;
      for (let j = 0; j < kept; j++) {
        if (dist2(data, e, data, this.sel[j] * dim, dim) < de) {
          ok = false;
          break;
        }
      }
      if (ok) this.sel[kept++] = e;
    }
    return kept;
  }

  /** Add vector i to the graph. */
  insert(i) {
    const { data, dim } = this.ds;
    const qo = i * dim;
    const lvl = Math.min(12, Math.floor(-Math.log(1 - this.rand()) * this.mL));
    this.level[i] = lvl;
    while (this.links.length <= lvl) {
      this.links.push(new Int32Array(this.ds.n * this.M));
      this.deg.push(new Uint8Array(this.ds.n));
    }
    this.count++;
    if (this.entry < 0) {
      this.entry = i;
      this.maxLevel = lvl;
      return;
    }
    let ep = this.entry;
    let epd = dist2(data, ep, data, qo, dim);
    for (let layer = this.maxLevel; layer > lvl; layer--) {
      this.searchLayer(data, qo, ep, epd, 1, layer, null);
      ep = this.best.topId();
      epd = this.best.topDist();
    }
    for (let layer = Math.min(lvl, this.maxLevel); layer >= 0; layer--) {
      this.searchLayer(data, qo, ep, epd, this.efConstruction, layer, null);
      const count = this.drainBest();
      ep = this.tmpIds[0];
      epd = this.tmpD[0];
      const kept = this.selectNeighbors(count, this.M);
      const cap = this.cap(layer);
      const links = this.links[layer];
      const deg = this.deg[layer];
      for (let j = 0; j < kept; j++) links[i * cap + j] = this.sel[j];
      deg[i] = kept;
      // link back; a neighbor that is already full re-selects its links with the same heuristic
      for (let j = 0; j < kept; j++) {
        const nb = links[i * cap + j];
        if (deg[nb] < cap) {
          links[nb * cap + deg[nb]++] = i;
          continue;
        }
        const total = cap + 1;
        for (let t = 0; t < cap; t++) {
          this.tmpIds[t] = links[nb * cap + t];
          this.tmpD[t] = dist2(data, this.tmpIds[t], data, nb * dim, dim);
        }
        this.tmpIds[cap] = i;
        this.tmpD[cap] = dist2(data, i, data, nb * dim, dim);
        // insertion sort by distance to nb
        for (let a = 1; a < total; a++) {
          const id = this.tmpIds[a];
          const d = this.tmpD[a];
          let b = a - 1;
          while (b >= 0 && this.tmpD[b] > d) {
            this.tmpIds[b + 1] = this.tmpIds[b];
            this.tmpD[b + 1] = this.tmpD[b];
            b--;
          }
          this.tmpIds[b + 1] = id;
          this.tmpD[b + 1] = d;
        }
        const k2 = this.selectNeighbors(total, cap);
        for (let t = 0; t < k2; t++) links[nb * cap + t] = this.sel[t];
        deg[nb] = k2;
      }
    }
    if (lvl > this.maxLevel) {
      this.maxLevel = lvl;
      this.entry = i;
    }
  }

  /** Insert every vector, yielding now and then so the page stays responsive. */
  *build(chunk = 200) {
    const n = this.ds.n;
    for (let i = 0; i < n; i++) {
      this.insert(i);
      if (i % chunk === chunk - 1) {
        this.progress = (i + 1) / n;
        yield;
      }
    }
    this.progress = 1;
    this.ready = true;
  }

  search(q, qo, top, { ef = 32 } = {}, trace) {
    const { data, dim } = this.ds;
    top.reset();
    if (this.entry < 0) return 0;
    this.comps = 1;
    let ep = this.entry;
    let epd = dist2(data, ep, q, qo, dim);
    if (trace) trace.add(ep, -1, this.maxLevel);
    for (let layer = this.maxLevel; layer > 0; layer--) {
      this.searchLayer(q, qo, ep, epd, 1, layer, trace);
      ep = this.best.topId();
      epd = this.best.topDist();
    }
    this.searchLayer(q, qo, ep, epd, Math.max(ef, top.k), 0, trace);
    const count = this.drainBest();
    for (let i = 0; i < Math.min(count, top.k); i++) top.push(this.tmpIds[i], this.tmpD[i]);
    return this.comps;
  }

  /** Number of links per layer, for the read-out. */
  stats() {
    const layers = [];
    for (let l = 0; l <= this.maxLevel; l++) {
      let nodes = 0;
      let edges = 0;
      for (let i = 0; i < this.ds.n; i++)
        if (this.level[i] >= l) {
          nodes++;
          edges += this.deg[l][i];
        }
      layers.push({ layer: l, nodes, links: edges });
    }
    return layers;
  }
}

// ------------------------------------------------------------------------------------ LSH

/**
 * Random-hyperplane LSH. Each of `tables` hash tables draws `maxBits` random hyperplanes; a vector's
 * key is the pattern of sides it falls on. Keys are stored sorted with the first hyperplane as the
 * most significant bit, so "the first b bits match" is one contiguous range: the number of bits
 * can be changed at query time without rebuilding anything.
 */
export class LSHIndex {
  constructor(ds, { tables = 8, maxBits = 18, seed = 11 } = {}) {
    this.ds = ds;
    this.name = 'LSH';
    this.tables = tables;
    this.maxBits = maxBits;
    this.seed = seed;
    this.planes = new Float32Array(tables * maxBits * ds.dim);
    this.keys = []; // per table: Uint32Array of sorted keys
    this.ids = []; // per table: Int32Array of vector ids in key order
    this.visited = new Uint32Array(ds.n);
    this.stamp = 0;
    this.ready = false;
    this.progress = 0;
  }

  key(table, vec, o) {
    const { dim } = this.ds;
    let key = 0;
    let p = table * this.maxBits * dim;
    for (let b = 0; b < this.maxBits; b++) {
      let s = 0;
      for (let j = 0; j < dim; j++) s += this.planes[p + j] * vec[o + j];
      p += dim;
      key = (key << 1) | (s >= 0 ? 1 : 0);
    }
    return key >>> 0;
  }

  *build() {
    const { data, n, dim } = this.ds;
    const rand = rng(this.seed);
    for (let i = 0; i < this.planes.length; i++) this.planes[i] = rand.normal();
    for (let t = 0; t < this.tables; t++) {
      const raw = new Uint32Array(n);
      for (let i = 0; i < n; i++) raw[i] = this.key(t, data, i * dim);
      const ids = Int32Array.from({ length: n }, (_, i) => i).sort((a, b) => raw[a] - raw[b]);
      const keys = new Uint32Array(n);
      for (let i = 0; i < n; i++) keys[i] = raw[ids[i]];
      this.keys.push(keys);
      this.ids.push(ids);
      this.progress = (t + 1) / this.tables;
      yield;
    }
    this.ready = true;
  }

  /** First position in keys whose value is >= target. */
  lowerBound(keys, target) {
    let lo = 0;
    let hi = keys.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (keys[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  search(q, qo, top, { bits = 10 } = {}, trace) {
    const { data, dim } = this.ds;
    const b = Math.max(1, Math.min(this.maxBits, bits | 0));
    const shift = this.maxBits - b;
    const stamp = ++this.stamp;
    top.reset();
    let comps = this.tables * b; // one projection per hyperplane actually used
    for (let t = 0; t < this.tables; t++) {
      const prefix = this.key(t, q, qo) >>> shift;
      const keys = this.keys[t];
      const ids = this.ids[t];
      // the bucket is every key that starts with the same b bits
      const lo = this.lowerBound(keys, (prefix << shift) >>> 0);
      const hi = prefix + 1 >= 2 ** b ? keys.length : this.lowerBound(keys, ((prefix + 1) << shift) >>> 0);
      if (trace) trace.groups.push(hi - lo);
      for (let s = lo; s < hi; s++) {
        const id = ids[s];
        if (this.visited[id] === stamp) continue;
        this.visited[id] = stamp;
        top.push(id, dist2(data, id, q, qo, dim));
        comps++;
        if (trace) trace.add(id, -1, t);
      }
    }
    return comps;
  }
}

// ------------------------------------------------------------------------------------ measuring

/** Exact k nearest neighbors of every query: the ground truth. Returns Int32Array(count * k). */
export function groundTruth(ds, queries, count, k) {
  const flat = new FlatIndex(ds);
  const top = new TopK(k);
  const out = new Int32Array(count * k);
  for (let i = 0; i < count; i++) {
    flat.search(queries, i * ds.dim, top);
    out.set(top.ids, i * k);
  }
  return out;
}

/** Share of the true k nearest neighbors that a result contains. */
export function recall(found, truth, to, k) {
  let hit = 0;
  for (let i = 0; i < k; i++) {
    const id = found[i];
    if (id < 0) continue;
    for (let j = 0; j < k; j++)
      if (truth[to + j] === id) {
        hit++;
        break;
      }
  }
  return hit / k;
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/**
 * Run a batch of queries through one index.
 * @returns {{recall:number, comps:number, ms:number}} mean recall@k, mean distance computations
 *          and mean wall-clock milliseconds per query
 */
export function benchmark(index, queries, count, truth, k, opts) {
  const dim = index.ds.dim;
  const top = new TopK(k);
  let rec = 0;
  let comps = 0;
  // first pass: results and counts
  for (let i = 0; i < count; i++) {
    comps += index.search(queries, i * dim, top, opts);
    rec += recall(top.ids, truth, i * k, k);
  }
  // timing pass(es): search only, repeated until the clock has something to measure
  let reps = 0;
  const t0 = now();
  let elapsed = 0;
  do {
    for (let i = 0; i < count; i++) index.search(queries, i * dim, top, opts);
    reps++;
    elapsed = now() - t0;
  } while (elapsed < 12 && reps < 20);
  return { recall: rec / count, comps: comps / count, ms: elapsed / (reps * count) };
}

/**
 * The three directions of largest variance of the data (principal components), by power iteration
 * on the covariance matrix of a sample. Used only to draw the vectors in 3D.
 * @returns {{mean: Float32Array, axes: Float32Array}} axes = 3 rows of `dim` values
 */
export function pca3(ds, sample = 3000, seed = 21) {
  const { data, n, dim } = ds;
  const rand = rng(seed);
  const m = Math.min(sample, n);
  const mean = new Float32Array(dim);
  const pick = new Int32Array(m);
  for (let i = 0; i < m; i++) {
    pick[i] = (rand() * n) | 0;
    for (let j = 0; j < dim; j++) mean[j] += data[pick[i] * dim + j] / m;
  }
  const cov = new Float64Array(dim * dim);
  for (let i = 0; i < m; i++) {
    const o = pick[i] * dim;
    for (let a = 0; a < dim; a++) {
      const va = data[o + a] - mean[a];
      for (let b = a; b < dim; b++) cov[a * dim + b] += (va * (data[o + b] - mean[b])) / m;
    }
  }
  for (let a = 0; a < dim; a++) for (let b = 0; b < a; b++) cov[a * dim + b] = cov[b * dim + a];
  const axes = new Float32Array(3 * dim);
  const v = new Float64Array(dim);
  const w = new Float64Array(dim);
  for (let c = 0; c < 3; c++) {
    for (let j = 0; j < dim; j++) v[j] = rand.normal();
    for (let it = 0; it < 120; it++) {
      for (let a = 0; a < dim; a++) {
        let s = 0;
        for (let b = 0; b < dim; b++) s += cov[a * dim + b] * v[b];
        w[a] = s;
      }
      // remove the components already found, then normalize
      for (let p = 0; p < c; p++) {
        let dot = 0;
        for (let j = 0; j < dim; j++) dot += w[j] * axes[p * dim + j];
        for (let j = 0; j < dim; j++) w[j] -= dot * axes[p * dim + j];
      }
      let norm = 0;
      for (let j = 0; j < dim; j++) norm += w[j] * w[j];
      norm = Math.sqrt(norm) || 1;
      for (let j = 0; j < dim; j++) v[j] = w[j] / norm;
    }
    for (let j = 0; j < dim; j++) axes[c * dim + j] = v[j];
  }
  return { mean, axes };
}

/** Project row i of `vec` (offset o) onto the three axes; writes x, y, z into out[oo..oo+3). */
export function project3(pca, vec, o, dim, out, oo) {
  for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let j = 0; j < dim; j++) s += (vec[o + j] - pca.mean[j]) * pca.axes[c * dim + j];
    out[oo + c] = s;
  }
}
