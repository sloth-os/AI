// A complete GPT in plain JavaScript: BPE tokenizer, transformer forward pass (with every
// intermediate activation exposed for visualization), KV-cached generation, sampling and
// post-training weight quantization. It runs the ~1M-parameter model trained by
// tools/train/train_tinygpt.py and is checked against PyTorch in tools/test-ml.mjs.

const EOT = '<|endoftext|>';
const NORMALIZE = { '“': '"', '”': '"', '‘': "'", '’': "'", '–': '-', '—': '-', '…': '...', ' ': ' ', '\x92': "'", '\x93': '"', '\x94': '"', '\t': ' ', '\r': '' };
const NORMALIZE_RE = /[“”‘’–—… \x92\x93\x94\t\r]/g;
const DROP_RE = /[^\x20-\x7e\n]/g;
const PRETOKEN_RE = /'(?:s|t|re|ve|m|ll|d)(?![A-Za-z])| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]|\s/g;
const SEP = '\u0000';

/** Byte-pair-encoding tokenizer. Mirrors tools/train/bpe.py rule for rule. */
export class Tokenizer {
  constructor({ vocab, merges }) {
    this.vocab = vocab;
    this.merges = merges;
    this.tokenToId = new Map(vocab.map((t, i) => [t, i]));
    this.rank = new Map(merges.map(([a, b], i) => [a + SEP + b, i]));
    this.cache = new Map();
    this.eot = this.tokenToId.get(EOT) ?? 0;
  }

  normalize(text) {
    return String(text).replace(NORMALIZE_RE, (c) => NORMALIZE[c]).replace(DROP_RE, '');
  }

  /** Merge one pre-token; returns every intermediate symbol list (first = characters, last = final tokens). */
  bpeSteps(word) {
    let sym = [...word];
    const steps = [sym];
    while (sym.length > 1) {
      let best = Infinity;
      let bi = -1;
      for (let i = 0; i < sym.length - 1; i++) {
        const r = this.rank.get(sym[i] + SEP + sym[i + 1]);
        if (r !== undefined && r < best) {
          best = r;
          bi = i;
        }
      }
      if (bi < 0) break;
      const a = sym[bi];
      const b = sym[bi + 1];
      const out = [];
      for (let i = 0; i < sym.length; ) {
        if (i + 1 < sym.length && sym[i] === a && sym[i + 1] === b) {
          out.push(a + b);
          i += 2;
        } else out.push(sym[i++]);
      }
      sym = out;
      steps.push(sym);
    }
    return steps;
  }

  _bpe(word) {
    let ids = this.cache.get(word);
    if (!ids) {
      const steps = this.bpeSteps(word);
      ids = steps[steps.length - 1].filter((s) => this.tokenToId.has(s)).map((s) => this.tokenToId.get(s));
      this.cache.set(word, ids);
    }
    return ids;
  }

  /** @returns {number[]} token ids (without the leading end-of-text marker) */
  encode(text) {
    const ids = [];
    for (const word of this.normalize(text).match(PRETOKEN_RE) || []) ids.push(...this._bpe(word));
    return ids;
  }

  /** Like encode, but keeps the story of how each word was merged (for the tokenization view). */
  encodeDetailed(text) {
    const words = (this.normalize(text).match(PRETOKEN_RE) || []).map((w) => ({ text: w, steps: this.bpeSteps(w), ids: this._bpe(w) }));
    return { words, ids: words.flatMap((w) => w.ids) };
  }

  decode(ids) {
    let s = '';
    for (const id of ids) if (id !== this.eot) s += this.vocab[id];
    return s;
  }

  /** Human-readable form of one token: spaces and newlines made visible. */
  show(id) {
    if (id === this.eot) return '⟨start⟩';
    return this.vocab[id].replace(/ /g, '·').replace(/\n/g, '↵');
  }
}

function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const hbits = u16[i];
    const e = (hbits >> 10) & 0x1f;
    const f = hbits & 0x3ff;
    let v;
    if (e === 0) v = (f / 1024) * 6.103515625e-5;
    else if (e === 31) v = f ? NaN : Infinity;
    else v = (1 + f / 1024) * Math.pow(2, e - 15);
    out[i] = hbits & 0x8000 ? -v : v;
  }
  return out;
}

const C_GELU = Math.sqrt(2 / Math.PI);
const gelu = (x) => 0.5 * x * (1 + Math.tanh(C_GELU * (x + 0.044715 * x * x * x)));

/** y[o] = sum_i W[o, i] * x[xOff + i] + b[o]; W is row-major [nOut, nIn]. */
function linear(y, yOff, x, xOff, W, b, nIn, nOut) {
  for (let o = 0, w = 0; o < nOut; o++, w += nIn) {
    let s = b ? b[o] : 0;
    let i = 0;
    for (; i + 3 < nIn; i += 4) s += W[w + i] * x[xOff + i] + W[w + i + 1] * x[xOff + i + 1] + W[w + i + 2] * x[xOff + i + 2] + W[w + i + 3] * x[xOff + i + 3];
    for (; i < nIn; i++) s += W[w + i] * x[xOff + i];
    y[yOff + o] = s;
  }
}

function layerNorm(y, yOff, x, xOff, w, b, d) {
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[xOff + i];
  mean /= d;
  let v = 0;
  for (let i = 0; i < d; i++) {
    const c = x[xOff + i] - mean;
    v += c * c;
  }
  const inv = 1 / Math.sqrt(v / d + 1e-5);
  for (let i = 0; i < d; i++) y[yOff + i] = (x[xOff + i] - mean) * inv * w[i] + b[i];
}

export class TinyGPT {
  /**
   * @param {{vocab:number,d:number,layers:number,heads:number,ctx:number}} config
   * @param {Map<string, Float32Array>} tensors  names as written by the training script (wte, wpe, h0.attn.w, ...)
   * @param {Map<string, number[]>} shapes
   */
  constructor(config, tensors, shapes, meta = {}) {
    this.config = config;
    this.t = tensors;
    this.shapes = shapes;
    this.meta = meta;
    /** multiply-adds performed since the counter was last reset: the honest unit of "work" */
    this.ops = 0;
    const { d, vocab } = config;
    this._x = new Float32Array(d);
    this._h = new Float32Array(d);
    this._qkv = new Float32Array(3 * d);
    this._att = new Float32Array(d);
    this._proj = new Float32Array(d);
    this._fc = new Float32Array(4 * d);
    this._mlp = new Float32Array(d);
    this._logits = new Float32Array(vocab);
    this.blocks = Array.from({ length: config.layers }, (_, i) => {
      const g = (n) => tensors.get(`h${i}.${n}`);
      return { ln1w: g('ln1.w'), ln1b: g('ln1.b'), attnW: g('attn.w'), attnB: g('attn.b'), projW: g('proj.w'), projB: g('proj.b'), ln2w: g('ln2.w'), ln2b: g('ln2.b'), fcW: g('fc.w'), fcB: g('fc.b'), outW: g('out.w'), outB: g('out.b') };
    });
    this.wte = tensors.get('wte');
    this.wpe = tensors.get('wpe');
    this.lnfw = tensors.get('lnf.w');
    this.lnfb = tensors.get('lnf.b');
  }

  /** Parse the tinygpt.bin container: "TGPT", uint32 header length, JSON header, float16 data. */
  static fromBuffer(buf) {
    const dv = new DataView(buf);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== 'TGPT') throw new Error('not a tinygpt.bin file');
    const hlen = dv.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
    const base = 8 + hlen;
    const tensors = new Map();
    const shapes = new Map();
    for (const t of header.tensors) {
      tensors.set(t.name, halfToFloat(new Uint16Array(buf.slice(base + t.offset, base + t.offset + t.length * 2))));
      shapes.set(t.name, t.shape);
    }
    return new TinyGPT(header.config, tensors, shapes, { params: header.params, valLoss: header.val_loss, trainSteps: header.train_steps, history: header.history });
  }

  get paramCount() {
    let n = 0;
    for (const a of this.t.values()) n += a.length;
    return n;
  }

  // ------------------------------------------------------------------ full forward pass
  /**
   * Run the whole sequence at once (no cache).
   * @param {number[]} ids  at most config.ctx tokens
   * @param {{capture?: boolean, allLogits?: boolean}} [opts]
   *   capture: keep the activations (residual stream after every layer, attention maps, update sizes)
   *   allLogits: return logits at every position (needed for the loss), otherwise only the last
   * @returns {{n:number, logits:Float32Array, allLogits?:Float32Array, acts?:object}}
   */
  forward(ids, opts = {}) {
    const { d, layers, heads, vocab } = this.config;
    const n = ids.length;
    const dh = d / heads;
    const scale = 1 / Math.sqrt(dh);
    let x = new Float32Array(n * d);
    for (let p = 0; p < n; p++) for (let i = 0; i < d; i++) x[p * d + i] = this.wte[ids[p] * d + i] + this.wpe[p * d + i];
    const acts = opts.capture ? { tok: new Float32Array(n * d), pos: new Float32Array(n * d), resid: [x.slice()], attn: [], attnNorm: [], mlpNorm: [] } : null;
    if (acts)
      for (let p = 0; p < n; p++)
        for (let i = 0; i < d; i++) {
          acts.tok[p * d + i] = this.wte[ids[p] * d + i];
          acts.pos[p * d + i] = this.wpe[p * d + i];
        }
    const h = new Float32Array(n * d);
    const qkv = new Float32Array(n * 3 * d);
    const y = new Float32Array(d);
    const yp = new Float32Array(d);
    const fc = new Float32Array(4 * d);
    const sc = new Float32Array(n);
    for (let l = 0; l < layers; l++) {
      const b = this.blocks[l];
      const attnMap = acts ? new Float32Array(heads * n * n) : null;
      const aNorm = acts ? new Float32Array(n) : null;
      const mNorm = acts ? new Float32Array(n) : null;
      for (let p = 0; p < n; p++) {
        layerNorm(h, p * d, x, p * d, b.ln1w, b.ln1b, d);
        linear(qkv, p * 3 * d, h, p * d, b.attnW, b.attnB, d, 3 * d);
      }
      this.ops += n * 3 * d * d;
      for (let p = 0; p < n; p++) {
        for (let hd = 0; hd < heads; hd++) {
          const qo = p * 3 * d + hd * dh;
          let max = -Infinity;
          for (let j = 0; j <= p; j++) {
            const ko = j * 3 * d + d + hd * dh;
            let s = 0;
            for (let i = 0; i < dh; i++) s += qkv[qo + i] * qkv[ko + i];
            s *= scale;
            sc[j] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let j = 0; j <= p; j++) sum += sc[j] = Math.exp(sc[j] - max);
          for (let i = 0; i < dh; i++) y[hd * dh + i] = 0;
          for (let j = 0; j <= p; j++) {
            const a = sc[j] / sum;
            if (attnMap) attnMap[(hd * n + p) * n + j] = a;
            const vo = j * 3 * d + 2 * d + hd * dh;
            for (let i = 0; i < dh; i++) y[hd * dh + i] += a * qkv[vo + i];
          }
          this.ops += (p + 1) * dh * 2;
        }
        linear(yp, 0, y, 0, b.projW, b.projB, d, d);
        let nn = 0;
        for (let i = 0; i < d; i++) {
          x[p * d + i] += yp[i];
          nn += yp[i] * yp[i];
        }
        if (aNorm) aNorm[p] = Math.sqrt(nn);
      }
      this.ops += n * d * d;
      for (let p = 0; p < n; p++) {
        layerNorm(h, p * d, x, p * d, b.ln2w, b.ln2b, d);
        linear(fc, 0, h, p * d, b.fcW, b.fcB, d, 4 * d);
        for (let i = 0; i < 4 * d; i++) fc[i] = gelu(fc[i]);
        linear(yp, 0, fc, 0, b.outW, b.outB, 4 * d, d);
        let nn = 0;
        for (let i = 0; i < d; i++) {
          x[p * d + i] += yp[i];
          nn += yp[i] * yp[i];
        }
        if (mNorm) mNorm[p] = Math.sqrt(nn);
      }
      this.ops += n * 8 * d * d;
      if (acts) {
        acts.resid.push(x.slice());
        acts.attn.push(attnMap);
        acts.attnNorm.push(aNorm);
        acts.mlpNorm.push(mNorm);
      }
    }
    const out = { n, acts };
    if (opts.allLogits) {
      out.allLogits = new Float32Array(n * vocab);
      for (let p = 0; p < n; p++) {
        layerNorm(h, 0, x, p * d, this.lnfw, this.lnfb, d);
        linear(out.allLogits, p * vocab, h, 0, this.wte, null, d, vocab);
      }
      this.ops += n * vocab * d;
      out.logits = out.allLogits.subarray((n - 1) * vocab, n * vocab);
    } else {
      out.logits = this.unembed(x, (n - 1) * d);
    }
    return out;
  }

  /** Final layer norm + multiplication by the (tied) embedding matrix: vector -> one logit per token. */
  unembed(x, off = 0, out = new Float32Array(this.config.vocab)) {
    const { d, vocab } = this.config;
    layerNorm(this._h, 0, x, off, this.lnfw, this.lnfb, d);
    linear(out, 0, this._h, 0, this.wte, null, d, vocab);
    this.ops += vocab * d;
    return out;
  }

  /** Mean next-token cross-entropy (nats) over a window of ctx + 1 token ids. */
  windowLoss(window) {
    const ids = window.slice(0, -1);
    const { allLogits, n } = this.forward(ids, { allLogits: true });
    const V = this.config.vocab;
    let total = 0;
    for (let p = 0; p < n; p++) {
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (allLogits[p * V + i] > max) max = allLogits[p * V + i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(allLogits[p * V + i] - max);
      total += Math.log(sum) + max - allLogits[p * V + window[p + 1]];
    }
    return total / n;
  }

  // ------------------------------------------------------------------ incremental (KV-cached) generation
  newCache() {
    const { d, layers, ctx } = this.config;
    return { n: 0, k: Array.from({ length: layers }, () => new Float32Array(ctx * d)), v: Array.from({ length: layers }, () => new Float32Array(ctx * d)) };
  }

  /**
   * Feed ONE new token. Keys and values of earlier tokens come from the cache, so the cost does not
   * depend on how many tokens have already been processed (apart from the cheap attention sums).
   * @returns {Float32Array} logits for the next token (a reused buffer: copy it if you keep it)
   */
  step(id, cache) {
    const { d, layers, heads, ctx } = this.config;
    const p = cache.n;
    if (p >= ctx) throw new Error('context window is full');
    const dh = d / heads;
    const scale = 1 / Math.sqrt(dh);
    const x = this._x;
    for (let i = 0; i < d; i++) x[i] = this.wte[id * d + i] + this.wpe[p * d + i];
    for (let l = 0; l < layers; l++) {
      const b = this.blocks[l];
      const K = cache.k[l];
      const Vv = cache.v[l];
      layerNorm(this._h, 0, x, 0, b.ln1w, b.ln1b, d);
      linear(this._qkv, 0, this._h, 0, b.attnW, b.attnB, d, 3 * d);
      for (let i = 0; i < d; i++) {
        K[p * d + i] = this._qkv[d + i];
        Vv[p * d + i] = this._qkv[2 * d + i];
      }
      for (let hd = 0; hd < heads; hd++) {
        let max = -Infinity;
        const sc = this._fc; // scratch (at least ctx long because 4d >= ctx)
        for (let j = 0; j <= p; j++) {
          let s = 0;
          for (let i = 0; i < dh; i++) s += this._qkv[hd * dh + i] * K[j * d + hd * dh + i];
          sc[j] = s * scale;
          if (sc[j] > max) max = sc[j];
        }
        let sum = 0;
        for (let j = 0; j <= p; j++) sum += sc[j] = Math.exp(sc[j] - max);
        for (let i = 0; i < dh; i++) this._att[hd * dh + i] = 0;
        for (let j = 0; j <= p; j++) {
          const a = sc[j] / sum;
          for (let i = 0; i < dh; i++) this._att[hd * dh + i] += a * Vv[j * d + hd * dh + i];
        }
      }
      linear(this._proj, 0, this._att, 0, b.projW, b.projB, d, d);
      for (let i = 0; i < d; i++) x[i] += this._proj[i];
      layerNorm(this._h, 0, x, 0, b.ln2w, b.ln2b, d);
      linear(this._fc, 0, this._h, 0, b.fcW, b.fcB, d, 4 * d);
      for (let i = 0; i < 4 * d; i++) this._fc[i] = gelu(this._fc[i]);
      linear(this._mlp, 0, this._fc, 0, b.outW, b.outB, 4 * d, d);
      for (let i = 0; i < d; i++) x[i] += this._mlp[i];
      this.ops += 12 * d * d + (p + 1) * d * 2;
    }
    cache.n = p + 1;
    return this.unembed(x, 0, this._logits);
  }

  // ------------------------------------------------------------------ quantization
  /** Names of the weight matrices (what real quantizers compress); biases and norms stay in full precision. */
  get matrixNames() {
    return [...this.t.keys()].filter((k) => this.shapes.get(k).length === 2);
  }

  /** Bytes needed to store the model with `bits` per matrix weight (16-bit everything else, one 16-bit scale per row). */
  sizeBytes(bits) {
    let bytes = 0;
    for (const [name, a] of this.t) {
      const shape = this.shapes.get(name);
      if (shape.length === 2 && bits < 16) bytes += (a.length * bits) / 8 + shape[0] * 2;
      else bytes += a.length * 2;
    }
    return bytes;
  }

  /**
   * Post-training quantization of every weight matrix to `bits` bits (symmetric, one scale per output row).
   * method 'minmax' scales each row to its largest weight; 'mse' searches for the clipping range that
   * minimizes the squared rounding error (it sacrifices rare large weights to resolve the many small ones).
   * The result is a normal TinyGPT whose weights have been rounded: it measures the QUALITY effect.
   * @returns {TinyGPT}
   */
  quantized(bits, { method = 'mse' } = {}) {
    if (bits >= 16) return this;
    const tensors = new Map();
    let err = 0;
    let count = 0;
    for (const [name, a] of this.t) {
      const shape = this.shapes.get(name);
      if (shape.length !== 2) {
        tensors.set(name, a);
        continue;
      }
      const r = quantizeMatrix(a, shape[0], shape[1], bits, method);
      tensors.set(name, r.data);
      err += r.sse;
      count += a.length;
    }
    const m = new TinyGPT(this.config, tensors, this.shapes, { ...this.meta, bits, method, rmse: Math.sqrt(err / count) });
    return m;
  }
}

/** Number of distinct values a weight can take at a given bit width (symmetric scheme). */
export const quantLevels = (bits) => 2 * (Math.pow(2, bits - 1) - 1) + 1;

/**
 * Round one matrix to a symmetric uniform grid, row by row.
 * @returns {{data: Float32Array, sse: number}} de-quantized weights and the summed squared error
 */
export function quantizeMatrix(W, rows, cols, bits, method = 'mse') {
  const qmax = Math.pow(2, bits - 1) - 1; // 8 bits -> 127, 4 -> 7, 2 -> 1 (ternary)
  const out = new Float32Array(W.length);
  let sse = 0;
  for (let r = 0; r < rows; r++) {
    const o = r * cols;
    let maxAbs = 0;
    for (let i = 0; i < cols; i++) maxAbs = Math.max(maxAbs, Math.abs(W[o + i]));
    if (maxAbs === 0) continue;
    let bestClip = maxAbs;
    if (method === 'mse') {
      let bestErr = Infinity;
      for (let c = 0; c <= 20; c++) {
        const clip = maxAbs * (1 - c * 0.04);
        const s = clip / qmax;
        let e = 0;
        for (let i = 0; i < cols; i++) {
          const q = Math.max(-qmax, Math.min(qmax, Math.round(W[o + i] / s)));
          const dlt = W[o + i] - q * s;
          e += dlt * dlt;
        }
        if (e < bestErr) {
          bestErr = e;
          bestClip = clip;
        }
      }
    }
    const s = bestClip / qmax;
    for (let i = 0; i < cols; i++) {
      const q = Math.max(-qmax, Math.min(qmax, Math.round(W[o + i] / s)));
      out[o + i] = q * s;
      const dlt = W[o + i] - out[o + i];
      sse += dlt * dlt;
    }
  }
  return { data: out, sse };
}

// ---------------------------------------------------------------------- sampling
/** Probabilities from logits at a temperature (temperature -> 0 approaches arg-max). */
export function softmax(logits, temperature = 1, out = new Float32Array(logits.length)) {
  const t = Math.max(temperature, 1e-6);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += out[i] = Math.exp((logits[i] - max) / t);
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

/** The k most probable entries, sorted: [{id, p}]. */
export function topK(probs, k) {
  const idx = [];
  for (let i = 0; i < probs.length; i++) {
    if (idx.length < k) {
      idx.push(i);
      if (idx.length === k) idx.sort((a, b) => probs[b] - probs[a]);
    } else if (probs[i] > probs[idx[k - 1]]) {
      let j = k - 1;
      while (j > 0 && probs[idx[j - 1]] < probs[i]) {
        idx[j] = idx[j - 1];
        j--;
      }
      idx[j] = i;
    }
  }
  if (idx.length < k) idx.sort((a, b) => probs[b] - probs[a]);
  return idx.map((id) => ({ id, p: probs[id] }));
}

/**
 * Choose the next token.
 * @param {Float32Array} logits
 * @param {{temperature?:number, topK?:number, topP?:number, rnd?:()=>number}} o
 *   temperature 0 = greedy; topK keeps the k most likely tokens; topP keeps the smallest set whose
 *   probabilities add up to p ("nucleus"). Both filters are applied, then the survivors are renormalized.
 * @returns {{id:number, p:number, candidates:{id:number,p:number,kept:boolean}[], keptMass:number}}
 */
export function sample(logits, o = {}) {
  const temperature = o.temperature ?? 1;
  const probs = softmax(logits, temperature === 0 ? 1 : temperature);
  const show = Math.max(o.show ?? 12, 1);
  const k = Math.min(o.topK && o.topK > 0 ? o.topK : logits.length, logits.length);
  const ranked = topK(probs, Math.max(Math.min(k, 400), show));
  const topP = o.topP ?? 1;
  let cum = 0;
  let keep = 0;
  for (let i = 0; i < ranked.length && i < k; i++) {
    keep = i + 1;
    cum += ranked[i].p;
    if (cum >= topP) break;
  }
  if (temperature === 0) keep = 1;
  const keptMass = ranked.slice(0, keep).reduce((s, r) => s + r.p, 0);
  let r = (o.rnd ?? Math.random)() * keptMass;
  let pick = ranked[0];
  for (let i = 0; i < keep; i++) {
    r -= ranked[i].p;
    if (r <= 0) {
      pick = ranked[i];
      break;
    }
    pick = ranked[i];
  }
  return {
    id: pick.id,
    p: pick.p / keptMass,
    keptMass,
    candidates: ranked.slice(0, show).map((c, i) => ({ id: c.id, p: c.p, kept: i < keep })),
  };
}

let loaded = null;
/** Fetch and assemble the model, its tokenizer and the held-out evaluation set (once per page load). */
export function loadTinyGPT(asset) {
  loaded ??= fetchTinyGPT(asset).catch((err) => {
    loaded = null;
    throw err;
  });
  return loaded;
}

async function fetchTinyGPT(asset) {
  const [bin, vocab, evalSet] = await Promise.all([
    fetch(asset('models/tinygpt.bin')).then((r) => {
      if (!r.ok) throw new Error('could not load the model weights');
      return r.arrayBuffer();
    }),
    fetch(asset('models/tinygpt-vocab.json')).then((r) => r.json()),
    fetch(asset('models/tinygpt-eval.json')).then((r) => r.json()),
  ]);
  return { model: TinyGPT.fromBuffer(bin), tokenizer: new Tokenizer(vocab), evalSet, bytes: bin.byteLength };
}
