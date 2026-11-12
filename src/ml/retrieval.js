// Text retrieval for the Embedding Space lab: BM25 keyword search, dense cosine search over
// precomputed embeddings, reciprocal rank fusion, and the metrics used to compare them.
// Pure JS (runs in the browser and in Node).

const STOP = new Set(
  (
    'a an the and or but if then else of to in on at by for with from into onto over under about as is are was were be been being ' +
    'it its this that these those there here which who whom what when where why how do does did done can could should would will ' +
    'may might must shall not no nor so than too very just also only i me my we our you your he she they them their his her us ' +
    'has have had having each every any all some such own same other more most much many few both either neither up down out off ' +
    'again once while during before after above below between through because until against am s t'
  ).split(' '),
);

/** Very light stemming: plural endings only, so "tokens" matches "token". */
function stem(w) {
  if (w.length <= 3 || !/^[a-z]+$/.test(w)) return w;
  if (w.endsWith('ies')) return w.slice(0, -3) + 'y';
  if (w.endsWith('sses') || w.endsWith('xes') || w.endsWith('ches') || w.endsWith('shes')) return w.slice(0, -2);
  if (w.endsWith('s') && !w.endsWith('ss') && !w.endsWith('us') && !w.endsWith('is')) return w.slice(0, -1);
  return w;
}

/**
 * Lowercase, split into runs of letters, digits and underscores (so `top_p` and `q4_k_m` stay
 * whole), drop stop words, strip plural endings.
 * @returns {string[]}
 */
export function tokenize(text) {
  const out = [];
  for (const raw of String(text).toLowerCase().match(/[a-z0-9_]+/g) ?? []) {
    if (STOP.has(raw)) continue;
    out.push(stem(raw));
  }
  return out;
}

/**
 * Okapi BM25 over a small in-memory corpus.
 *   score(q, d) = sum over query terms t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgLen))
 *   idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
 */
export class BM25 {
  /** @param {string[]} docs */
  constructor(docs, { k1 = 1.2, b = 0.75 } = {}) {
    this.k1 = k1;
    this.b = b;
    this.n = docs.length;
    this.tf = docs.map((d) => {
      const m = new Map();
      for (const t of tokenize(d)) m.set(t, (m.get(t) || 0) + 1);
      return m;
    });
    this.len = this.tf.map((m) => {
      let s = 0;
      for (const v of m.values()) s += v;
      return s;
    });
    this.avgLen = this.len.reduce((a, c) => a + c, 0) / Math.max(1, this.n);
    this.df = new Map();
    for (const m of this.tf) for (const t of m.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
  }

  idf(term) {
    const n = this.df.get(term) || 0;
    return Math.log(1 + (this.n - n + 0.5) / (n + 0.5));
  }

  /** Contribution of one term to one document's score. */
  termScore(term, doc) {
    const tf = this.tf[doc].get(term) || 0;
    if (!tf) return 0;
    return (this.idf(term) * tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + (this.b * this.len[doc]) / this.avgLen));
  }

  /**
   * Rank the documents that share at least one term with the query.
   * @returns {{doc: number, score: number}[]} best first
   */
  search(query) {
    const terms = [...new Set(tokenize(query))];
    const out = [];
    for (let d = 0; d < this.n; d++) {
      let s = 0;
      for (const t of terms) s += this.termScore(t, d);
      if (s > 0) out.push({ doc: d, score: s });
    }
    return out.sort((a, b) => b.score - a.score || a.doc - b.doc);
  }

  /** Which query terms hit document `doc`, with their share of the score (for the "why" read-out). */
  explain(query, doc) {
    return [...new Set(tokenize(query))]
      .map((term) => ({ term, score: this.termScore(term, doc), df: this.df.get(term) || 0 }))
      .filter((t) => t.score > 0)
      .sort((a, b) => b.score - a.score);
  }
}

/** base64 -> Int8Array (rows * dim values). Works in browsers and in Node. */
export function decodeInt8(b64) {
  if (typeof atob === 'function') {
    const bin = atob(b64);
    const out = new Int8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  const buf = Buffer.from(b64, 'base64');
  return new Int8Array(buf.buffer, buf.byteOffset, buf.length).slice();
}

/** Int8 rows -> Float32 rows of unit length, so that a dot product is the cosine similarity. */
export function unitRows(int8, dim) {
  const rows = int8.length / dim;
  const out = new Float32Array(int8.length);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let j = 0; j < dim; j++) s += int8[r * dim + j] * int8[r * dim + j];
    const inv = 1 / Math.sqrt(s || 1);
    for (let j = 0; j < dim; j++) out[r * dim + j] = int8[r * dim + j] * inv;
  }
  return out;
}

/** Dot product of row a of matrix A with row b of matrix B. */
export function dotRows(A, a, B, b, dim) {
  let s = 0;
  const ao = a * dim;
  const bo = b * dim;
  for (let j = 0; j < dim; j++) s += A[ao + j] * B[bo + j];
  return s;
}

/**
 * Exact dense search: cosine similarity of one query row against every row of `matrix`.
 * @returns {{doc: number, score: number}[]} best first; `skip` leaves one row out (the query itself).
 */
export function denseSearch(queries, q, matrix, dim, { skip = -1, top = Infinity } = {}) {
  const rows = matrix.length / dim;
  const out = [];
  for (let d = 0; d < rows; d++) if (d !== skip) out.push({ doc: d, score: dotRows(queries, q, matrix, d, dim) });
  out.sort((a, b) => b.score - a.score || a.doc - b.doc);
  return top < out.length ? out.slice(0, top) : out;
}

/**
 * Reciprocal rank fusion: score(d) = sum over the lists of 1 / (k + rank of d in that list).
 * A document missing from a list simply gets nothing from it.
 * @param {{doc: number}[][]} lists ranked lists, best first
 * @returns {{doc: number, score: number, ranks: (number|null)[]}[]} ranks = position in each input list
 */
export function rrf(lists, k = 60) {
  const acc = new Map();
  lists.forEach((list, li) => {
    list.forEach((item, i) => {
      let e = acc.get(item.doc);
      if (!e) acc.set(item.doc, (e = { doc: item.doc, score: 0, ranks: lists.map(() => null) }));
      e.score += 1 / (k + i + 1);
      e.ranks[li] = i + 1;
    });
  });
  const best = (e) => Math.min(...e.ranks.map((r) => r ?? Infinity));
  return [...acc.values()].sort((a, b) => b.score - a.score || best(a) - best(b) || a.doc - b.doc);
}

/** 1-based rank of the first relevant document in a ranked list, or 0 when none is in it. */
export function firstRelevantRank(list, relevant) {
  for (let i = 0; i < list.length; i++) if (relevant.includes(list[i].doc)) return i + 1;
  return 0;
}

/**
 * Recall@k for one query: the share of its relevant documents found in the top k
 * (out of min(k, number relevant), so a perfect top-1 scores 1 even with two relevant passages).
 */
export function recallAt(list, relevant, k) {
  let hit = 0;
  for (let i = 0; i < Math.min(k, list.length); i++) if (relevant.includes(list[i].doc)) hit++;
  return hit / Math.min(k, relevant.length);
}

/**
 * Mean recall@k and mean reciprocal rank over a set of queries.
 * @param {{doc:number}[][]} rankings one ranked list per query
 * @param {number[][]} relevant relevant document indices per query
 */
export function evaluate(rankings, relevant, ks = [1, 3]) {
  const out = { mrr: 0, ranks: [] };
  for (const k of ks) out['recall@' + k] = 0;
  rankings.forEach((list, q) => {
    const r = firstRelevantRank(list, relevant[q]);
    out.ranks.push(r);
    out.mrr += r ? 1 / r : 0;
    for (const k of ks) out['recall@' + k] += recallAt(list, relevant[q], k);
  });
  const n = Math.max(1, rankings.length);
  out.mrr /= n;
  for (const k of ks) out['recall@' + k] /= n;
  return out;
}
