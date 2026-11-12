// A small convolutional network in plain JS: the forward pass of the model trained by
// tools/train/train_mnist_cnn.py, with every intermediate activation kept, plus the MNIST-style
// pre-processing that turns a free-hand drawing into the 28x28 image the network expects.
// No three.js and no DOM in here, so everything can be checked in Node (see selfTest).
//
//   input 1x28x28 -> conv 5x5 (8 maps, 24x24) -> ReLU -> max-pool 2 (12x12)
//                 -> conv 5x5 (16 maps, 8x8)  -> ReLU -> max-pool 2 (4x4)
//                 -> flatten (256) -> linear -> 10 logits -> softmax

export const SIZE = 28; // input side
export const K = 5; // kernel side of both convolutions
export const C1 = 8; // maps after conv1
export const C2 = 16; // maps after conv2
export const S1 = 24; // conv1 output side
export const P1 = 12; // after pooling
export const S2 = 8; // conv2 output side
export const P2 = 4; // after pooling
export const FLAT = C2 * P2 * P2; // 256
export const CLASSES = 10;

// ------------------------------------------------------------------------------- loading
function bytesFromBase64(b64) {
  if (typeof atob === 'function') {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  return new Uint8Array(Buffer.from(b64, 'base64'));
}

function float32FromBase64(b64) {
  const bytes = bytesFromBase64(b64);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength >> 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/**
 * Turn the exported JSON into typed arrays.
 * Weight layouts follow PyTorch: conv [out, in, ky, kx], linear [out, in].
 */
export function loadModel(json) {
  if (json.format !== 'mnist-cnn-v1') throw new Error('unexpected model format: ' + json.format);
  const w = (name) => float32FromBase64(json.weights[name].data);
  const pixels = bytesFromBase64(json.samples.pixels);
  const samples = json.samples.labels.map((label, i) => {
    const px = new Float32Array(SIZE * SIZE);
    for (let k = 0; k < px.length; k++) px[k] = pixels[i * SIZE * SIZE + k] / 255;
    return { pixels: px, label, kind: json.samples.kinds[i], testIndex: json.samples.test_index[i] };
  });
  const model = {
    conv1: { w: w('conv1.w'), b: w('conv1.b') },
    conv2: { w: w('conv2.w'), b: w('conv2.b') },
    fc: { w: w('fc.w'), b: w('fc.b') },
    params: json.params,
    testAccuracy: json.test_accuracy,
    testErrors: json.test_errors,
    testCount: json.test_count,
    train: json.train,
    samples,
    refLogits: json.ref_logits,
    refSums: json.ref_sums,
  };
  const counted = model.conv1.w.length + model.conv1.b.length + model.conv2.w.length + model.conv2.b.length + model.fc.w.length + model.fc.b.length;
  if (counted !== json.params) throw new Error(`parameter count mismatch: ${counted} vs ${json.params}`);
  return model;
}

/** Parameters per layer, computed from the weight arrays. */
export function paramTable(model) {
  return [
    { name: 'conv1', what: '8 kernels of 5×5×1, plus 8 biases', count: model.conv1.w.length + model.conv1.b.length },
    { name: 'conv2', what: '16 kernels of 5×5×8, plus 16 biases', count: model.conv2.w.length + model.conv2.b.length },
    { name: 'linear', what: '256 × 10 weights, plus 10 biases', count: model.fc.w.length + model.fc.b.length },
  ];
}

// ------------------------------------------------------------------------------- layers
/**
 * "Valid" 5x5 convolution (really cross-correlation, as in every deep-learning library), stride 1.
 * x: [inC, H, W] flat, w: [outC, inC, 5, 5] flat, out: [outC, H-4, W-4] flat (pre-activation).
 */
export function conv2d(x, inC, H, W, w, b, outC, out) {
  const OH = H - K + 1;
  const OW = W - K + 1;
  for (let o = 0; o < outC; o++) {
    const bias = b[o];
    const oBase = o * OH * OW;
    for (let y = 0; y < OH; y++) {
      for (let xx = 0; xx < OW; xx++) {
        let s = bias;
        for (let c = 0; c < inC; c++) {
          let wi = (o * inC + c) * K * K;
          let xi = c * H * W + y * W + xx;
          for (let ky = 0; ky < K; ky++) {
            s += x[xi] * w[wi] + x[xi + 1] * w[wi + 1] + x[xi + 2] * w[wi + 2] + x[xi + 3] * w[wi + 3] + x[xi + 4] * w[wi + 4];
            wi += K;
            xi += W;
          }
        }
        out[oBase + y * OW + xx] = s;
      }
    }
  }
  return out;
}

export function relu(x, out) {
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
  return out;
}

/** 2x2 max-pooling, stride 2. x: [C, H, W] flat -> out: [C, H/2, W/2] flat. */
export function maxPool2(x, C, H, W, out) {
  const OH = H >> 1;
  const OW = W >> 1;
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < OH; y++) {
      for (let xx = 0; xx < OW; xx++) {
        const i = c * H * W + 2 * y * W + 2 * xx;
        let m = x[i];
        if (x[i + 1] > m) m = x[i + 1];
        if (x[i + W] > m) m = x[i + W];
        if (x[i + W + 1] > m) m = x[i + W + 1];
        out[c * OH * OW + y * OW + xx] = m;
      }
    }
  }
  return out;
}

export function linear(x, w, b, out) {
  const n = x.length;
  for (let o = 0; o < out.length; o++) {
    let s = b[o];
    const base = o * n;
    for (let i = 0; i < n; i++) s += w[base + i] * x[i];
    out[o] = s;
  }
  return out;
}

export function softmax(logits, out) {
  let m = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += out[i] = Math.exp(logits[i] - m);
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

/** Buffers for one forward pass (allocate once, reuse for every stroke). */
export function createActs() {
  return {
    input: new Float32Array(SIZE * SIZE),
    z1: new Float32Array(C1 * S1 * S1), // conv1 before ReLU (signed)
    a1: new Float32Array(C1 * S1 * S1), // after ReLU
    p1: new Float32Array(C1 * P1 * P1), // after pooling
    z2: new Float32Array(C2 * S2 * S2),
    a2: new Float32Array(C2 * S2 * S2),
    p2: new Float32Array(C2 * P2 * P2), // this is also the flattened vector (channel, row, column order)
    logits: new Float32Array(CLASSES),
    probs: new Float32Array(CLASSES),
    pred: 0,
  };
}

/**
 * Run the network on a 28x28 image (Float32Array(784), 0 = background, 1 = ink).
 * Returns `acts` with every intermediate activation filled in.
 */
export function forward(model, input, acts = createActs()) {
  if (acts.input !== input) acts.input.set(input);
  conv2d(acts.input, 1, SIZE, SIZE, model.conv1.w, model.conv1.b, C1, acts.z1);
  relu(acts.z1, acts.a1);
  maxPool2(acts.a1, C1, S1, S1, acts.p1);
  conv2d(acts.p1, C1, P1, P1, model.conv2.w, model.conv2.b, C2, acts.z2);
  relu(acts.z2, acts.a2);
  maxPool2(acts.a2, C2, S2, S2, acts.p2);
  linear(acts.p2, model.fc.w, model.fc.b, acts.logits);
  softmax(acts.logits, acts.probs);
  let best = 0;
  for (let i = 1; i < CLASSES; i++) if (acts.probs[i] > acts.probs[best]) best = i;
  acts.pred = best;
  return acts;
}

/**
 * The square of input pixels that one cell of a layer can see: {x, y, size} in input pixels.
 * layer: 'conv1' | 'pool1' | 'conv2' | 'pool2'
 */
export function receptiveField(layer, cx, cy) {
  switch (layer) {
    case 'conv1':
      return { x: cx, y: cy, size: 5 };
    case 'pool1':
      return { x: 2 * cx, y: 2 * cy, size: 6 };
    case 'conv2':
      return { x: 2 * cx, y: 2 * cy, size: 14 };
    case 'pool2':
      return { x: 4 * cx, y: 4 * cy, size: 16 };
    default:
      throw new Error('unknown layer ' + layer);
  }
}

// ------------------------------------------------------------------------------- pre-processing
const BOX = 20; // MNIST: the ink is scaled to fit a 20x20 box ...
const CENTRE = 14.5; // ... and its centre of mass is put on the centre of pixel 14 (measured on the training set)

/**
 * MNIST-style normalisation of a drawing.
 *   1. find the bounding box of the ink,
 *   2. scale it (keeping the aspect ratio) so that its longer side is 20 pixels,
 *   3. place it in a 28x28 frame so that the centre of mass of the ink is at the centre.
 * Resampling is an exact area average, so thin strokes do not alias when a large pad is shrunk.
 *
 * @param {ArrayLike<number>} gray  w*h ink values (any positive scale, background 0)
 * @param {Float32Array} [out]  784 values in [0, 1]
 * @returns {{pixels: Float32Array, empty: boolean, box?: number[], scale?: number, offset?: number[]}}
 *          box = [x0, y0, x1, y1] in source pixels (inclusive); a source point u lands on u * scale + offset.
 */
export function preprocessDrawing(gray, w, h, out = new Float32Array(SIZE * SIZE)) {
  out.fill(0);
  let max = 0;
  for (let i = 0; i < w * h; i++) if (gray[i] > max) max = gray[i];
  if (!(max > 0)) return { pixels: out, empty: true };
  const thr = max * 0.12;
  let x0 = w;
  let y0 = h;
  let x1 = -1;
  let y1 = -1;
  let m = 0;
  let mx = 0;
  let my = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = gray[y * w + x];
      if (v <= thr) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      m += v;
      mx += v * (x + 0.5);
      my += v * (y + 0.5);
    }
  }
  const bw = x1 - x0 + 1;
  const bh = y1 - y0 + 1;
  const s = BOX / Math.max(bw, bh); // destination pixels per source pixel
  // centre of mass on the centre of the frame, but never push the ink box out of the frame
  const ox = Math.min(SIZE - (x1 + 1) * s, Math.max(-x0 * s, CENTRE - (mx / m) * s));
  const oy = Math.min(SIZE - (y1 + 1) * s, Math.max(-y0 * s, CENTRE - (my / m) * s));

  // separable area average: columns first (into tmp: bh rows x 28 columns), then rows
  const tmp = new Float32Array(bh * SIZE);
  for (let j = 0; j < SIZE; j++) {
    const a = Math.max(x0, (j - ox) / s);
    const b = Math.min(x1 + 1, (j + 1 - ox) / s);
    if (b <= a) continue;
    const inv = s; // 1 / footprint width
    for (let x = Math.floor(a); x < b; x++) {
      const cover = (Math.min(b, x + 1) - Math.max(a, x)) * inv;
      if (cover <= 0) continue;
      for (let r = 0; r < bh; r++) tmp[r * SIZE + j] += gray[(y0 + r) * w + x] * cover;
    }
  }
  let peak = 0;
  for (let i = 0; i < SIZE; i++) {
    const a = Math.max(y0, (i - oy) / s);
    const b = Math.min(y1 + 1, (i + 1 - oy) / s);
    if (b <= a) continue;
    for (let y = Math.floor(a); y < b; y++) {
      const cover = (Math.min(b, y + 1) - Math.max(a, y)) * s;
      if (cover <= 0) continue;
      const row = (y - y0) * SIZE;
      for (let j = 0; j < SIZE; j++) out[i * SIZE + j] += tmp[row + j] * cover;
    }
    for (let j = 0; j < SIZE; j++) if (out[i * SIZE + j] > peak) peak = out[i * SIZE + j];
  }
  if (peak > 0) for (let i = 0; i < out.length; i++) out[i] = Math.min(1, out[i] / peak);
  return { pixels: out, empty: false, box: [x0, y0, x1, y1], scale: s, offset: [ox, oy] };
}

/**
 * Rotate (degrees, counter-clockwise on screen), scale and shift (pixels, +x right, +y down) a 28x28
 * image about its centre, with bilinear sampling. Used to test how brittle the network is.
 */
export function transformImage(src, out, { angle = 0, dx = 0, dy = 0, scale = 1 } = {}) {
  const t = (-angle * Math.PI) / 180; // screen y points down
  const cos = Math.cos(t) / scale;
  const sin = Math.sin(t) / scale;
  const c = 14; // pixel-centre coordinates of the frame centre
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const ux = x - c - dx;
      const uy = y - c - dy;
      const sx = cos * ux + sin * uy + c;
      const sy = -sin * ux + cos * uy + c;
      const ix = Math.floor(sx);
      const iy = Math.floor(sy);
      const fx = sx - ix;
      const fy = sy - iy;
      let v = 0;
      if (ix >= 0 && ix < SIZE && iy >= 0 && iy < SIZE) v += src[iy * SIZE + ix] * (1 - fx) * (1 - fy);
      if (ix + 1 >= 0 && ix + 1 < SIZE && iy >= 0 && iy < SIZE) v += src[iy * SIZE + ix + 1] * fx * (1 - fy);
      if (ix >= 0 && ix < SIZE && iy + 1 >= 0 && iy + 1 < SIZE) v += src[(iy + 1) * SIZE + ix] * (1 - fx) * fy;
      if (ix + 1 >= 0 && ix + 1 < SIZE && iy + 1 >= 0 && iy + 1 < SIZE) v += src[(iy + 1) * SIZE + ix + 1] * fx * fy;
      out[y * SIZE + x] = v;
    }
  }
  return out;
}

// ------------------------------------------------------------------------------- reading a kernel
// A kernel is described by measurement, not by eye: it is compared with a bank of synthetic 5x5
// test patterns (strokes and edges at 15-degree steps, a dot, solid ink) and the best match wins.
let PROBES = null;
function probes() {
  if (PROBES) return PROBES;
  const list = [];
  const add = (kind, angle, fn) => {
    const patch = new Float32Array(K * K);
    let n = 0;
    for (let y = 0; y < K; y++)
      for (let x = 0; x < K; x++) {
        const v = fn(x - 2, y - 2);
        patch[y * K + x] = v;
        n += v * v;
      }
    list.push({ kind, angle, patch, norm: Math.sqrt(n) });
  };
  for (let a = 0; a < 180; a += 15) {
    // a stroke through (or one pixel beside) the centre; `a` is its direction, counter-clockwise from "pointing right"
    const t = (a * Math.PI) / 180;
    for (const off of [-1, 0, 1]) add('stroke', a, (x, y) => Math.exp(-((x * Math.sin(t) + y * Math.cos(t) - off) ** 2) / (2 * 0.85 * 0.85)));
  }
  for (let a = 0; a < 360; a += 15) {
    // an edge: ink on the side that `a` points to (0 = right, 90 = up), background on the other side
    const t = (a * Math.PI) / 180;
    for (const off of [-1, 0, 1]) add('edge', a, (x, y) => 1 / (1 + Math.exp(-(x * Math.cos(t) - y * Math.sin(t) - off) / 0.5)));
  }
  add('dot', 0, (x, y) => Math.exp(-(x * x + y * y) / 2));
  add('ink', 0, () => 1);
  PROBES = list;
  return list;
}

/**
 * Which synthetic pattern does a 5x5 kernel respond to most, relative to the pattern's energy?
 * @param {ArrayLike<number>} k 25 weights, row by row
 * @returns {{kind: 'stroke'|'edge'|'dot'|'ink', angle: number, match: number, worst: {kind: string, angle: number, match: number}}}
 *          match = cosine similarity between kernel and pattern, in [-1, 1]
 */
export function probeKernel(k) {
  let kn = 0;
  for (let i = 0; i < K * K; i++) kn += k[i] * k[i];
  kn = Math.sqrt(kn) || 1;
  let best = null;
  let worst = null;
  for (const p of probes()) {
    let s = 0;
    for (let i = 0; i < K * K; i++) s += k[i] * p.patch[i];
    const match = s / (kn * p.norm);
    if (!best || match > best.match) best = { kind: p.kind, angle: p.angle, match };
    if (!worst || match < worst.match) worst = { kind: p.kind, angle: p.angle, match };
  }
  return { ...best, worst };
}

const STROKE_WORDS = [
  'horizontal strokes (—)',
  'nearly horizontal strokes that rise slightly to the right',
  'shallow strokes that rise to the right',
  'diagonal strokes that rise to the right (/)',
  'steep strokes that rise to the right',
  'nearly vertical strokes that lean right',
  'vertical strokes (|)',
  'nearly vertical strokes that lean left',
  'steep strokes that fall to the right',
  'diagonal strokes that fall to the right (\\)',
  'shallow strokes that fall to the right',
  'nearly horizontal strokes that fall slightly to the right',
];
const SIDE_WORDS = ['on the right', 'on the upper right', 'above', 'on the upper left', 'on the left', 'on the lower left', 'below', 'on the lower right'];

function patternWords(p) {
  if (p.kind === 'stroke') return STROKE_WORDS[Math.round(p.angle / 15) % 12];
  if (p.kind === 'edge') {
    const side = Math.round(p.angle / 45) % 8;
    return `edges with ink ${SIDE_WORDS[side]} and background ${SIDE_WORDS[(side + 4) % 8]}`;
  }
  if (p.kind === 'dot') return 'a dot of ink about three pixels wide';
  return 'solid ink (the inside of thick strokes)';
}

/** One or two measured sentences about a conv1 kernel. */
export function describeKernel(k) {
  const p = probeKernel(k);
  const n = probes().length;
  let s =
    p.match >= 0.3
      ? `Best match among ${n} test patterns: ${patternWords(p)} (similarity ${p.match.toFixed(2)}).`
      : `None of ${n} test patterns matches it well; the closest is ${patternWords(p)} (similarity ${p.match.toFixed(2)}), so it reacts to a mixture.`;
  const same = p.worst.kind === p.kind && p.worst.angle === p.angle;
  if (p.worst.match < -0.15) s += same ? ' The same stroke one pixel to the side drives it below zero, so it marks one border of the stroke.' : ` Most opposed: ${patternWords(p.worst)}.`;
  return s;
}

/**
 * A conv2 kernel is 5x5x8: how much does it listen to each of the 8 pooled maps?
 * Returns per input map the sum of positive weights, the sum of negative weights and the share of total |weight|.
 */
export function conv2Inputs(model, j) {
  const rows = [];
  let total = 0;
  for (let c = 0; c < C1; c++) {
    let pos = 0;
    let neg = 0;
    const base = (j * C1 + c) * K * K;
    for (let i = 0; i < K * K; i++) {
      const v = model.conv2.w[base + i];
      if (v > 0) pos += v;
      else neg += v;
    }
    total += pos - neg;
    rows.push({ map: c, pos, neg });
  }
  for (const r of rows) r.share = (r.pos - r.neg) / total;
  return rows;
}

/** One measured sentence about a conv2 kernel. */
export function describeConv2(model, j) {
  const rows = conv2Inputs(model, j);
  const net = rows.map((r) => ({ map: r.map, v: r.pos + r.neg }));
  const up = [...net].sort((a, b) => b.v - a.v).filter((r) => r.v > 0).slice(0, 2);
  const down = [...net].sort((a, b) => a.v - b.v).filter((r) => r.v < 0).slice(0, 1);
  const names = (list) => list.map((r) => r.map + 1).join(' and ');
  let s = `One kernel of 5×5×8 = 200 weights reads all eight pooled maps at once, so each cell sees a 14×14 patch of the image.`;
  if (up.length) s += ` Its weights add up most positively for map${up.length > 1 ? 's' : ''} ${names(up)}`;
  if (down.length) s += `${up.length ? ' and' : ' Its weights add up'} most negatively for map ${names(down)}`;
  return s + '.';
}

// ------------------------------------------------------------------------------- self-test
/**
 * Compare this JS implementation with the PyTorch reference stored in the model file.
 * @returns {{maxLogitError: number, sumErrors: Record<string, number>, agree: number, count: number}}
 */
export function selfTest(model) {
  const acts = createActs();
  let maxLogitError = 0;
  let agree = 0;
  let sumErrors = {};
  model.samples.forEach((s, i) => {
    forward(model, s.pixels, acts);
    const ref = model.refLogits[i];
    let refBest = 0;
    for (let c = 0; c < CLASSES; c++) {
      maxLogitError = Math.max(maxLogitError, Math.abs(acts.logits[c] - ref[c]));
      if (ref[c] > ref[refBest]) refBest = c;
    }
    if (refBest === acts.pred) agree++;
    if (i === 0 && model.refSums) {
      const sum = (a) => a.reduce((p, v) => p + v, 0);
      sumErrors = {
        conv1: Math.abs(sum(acts.a1) - model.refSums.conv1),
        pool1: Math.abs(sum(acts.p1) - model.refSums.pool1),
        conv2: Math.abs(sum(acts.a2) - model.refSums.conv2),
        pool2: Math.abs(sum(acts.p2) - model.refSums.pool2),
      };
    }
  });
  return { maxLogitError, sumErrors, agree, count: model.samples.length };
}
