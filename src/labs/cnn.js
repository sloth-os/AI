// How a CNN Sees: a real 5,994-parameter convolutional network (trained on MNIST by
// tools/train/train_mnist_cnn.py) laid out left to right. The front lane (z = 0) holds one path
// through the network: the input, one selected map per layer, the flattened vector and the ten
// outputs. Behind every selected map stands the deck of its sibling maps as see-through planes.
// All numbers come from src/ml/cnn.js; this file draws them and wires the controls.
import * as THREE from 'three';
import './cnn.css';
import {
  loadModel, forward, createActs, preprocessDrawing, transformImage, describeKernel, describeConv2, paramTable, receptiveField,
  SIZE, K, C1, C2, S1, P1, S2, P2, FLAT, CLASSES,
} from '../ml/cnn.js';
import { SERIES, PAL, glow, seq, div } from '../core/colors.js';
import { gridFloor, instanced, setInst, flushInst } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, segmented, slider, toggle, tour, table, fmt } from '../core/ui.js';
import { heatmap } from '../core/charts.js';
import { damp } from '../core/tween.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const CELL = 0.1; // world size of one cell, the same in every layer, so shrinking maps really shrink
const LX = { input: -6.6, conv1: -3.0, pool1: -0.55, conv2: 1.15, pool2: 2.3, flat: 3.4, out: 4.5 };
const DECK_Z = -0.9; // first deck slot behind the lane
const FLAT_H = 3.2;
const BAR_DY = 0.32;
const BAR_LEN = 1.5;
const PAD = 280; // drawing pad resolution
const SCAN_COLOR = PAL.amber;

// sRGB byte ramps for the data textures (same ramps as the charts: seq = magnitude, div = signed)
const SEQ_LUT = new Uint8Array(256 * 3);
const DIV_LUT = new Uint8Array(513 * 3);
{
  const c = new THREE.Color();
  for (let i = 0; i < 256; i++) {
    const hex = seq(i / 255, c).getHex();
    SEQ_LUT.set([hex >> 16, (hex >> 8) & 255, hex & 255], i * 3);
  }
  for (let i = 0; i <= 512; i++) {
    const hex = div(i / 256 - 1, c).getHex();
    DIV_LUT.set([hex >> 16, (hex >> 8) & 255, hex & 255], i * 3);
  }
}

const MAP_VS = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;
// One feature map: cells from a nearest-filtered data texture, hairline gaps once cells are large
// on screen, a thin frame, and either per-cell transparency (deck) or a solid plate (selected).
const MAP_FS = /* glsl */ `
  uniform sampler2D uMap;
  uniform vec2 uCells;
  uniform float uSolid;
  uniform float uDim;
  uniform vec3 uFrame;
  uniform float uFrameA;
  uniform float uGain;
  varying vec2 vUv;
  void main() {
    vec4 t = texture2D(uMap, vUv);
    vec2 p = vUv * uCells;
    vec2 fw = max(fwidth(p), vec2(1e-5));
    vec2 c = fract(p);
    vec2 g = smoothstep(vec2(0.0), fw * 1.3, c) * smoothstep(vec2(0.0), fw * 1.3, 1.0 - c);
    vec2 on = clamp((1.0 / fw - 5.0) / 8.0, 0.0, 1.0);
    float gap = mix(1.0, g.x, on.x * 0.6) * mix(1.0, g.y, on.y * 0.6);
    vec3 col = t.rgb * uGain * gap;
    float a = mix(t.a, 1.0, uSolid) * uDim;
    vec2 e = min(vUv, 1.0 - vUv) / max(fwidth(vUv), vec2(1e-6));
    float edge = (1.0 - smoothstep(0.7, 1.9, min(e.x, e.y))) * uFrameA;
    col = mix(col, uFrame, edge);
    a = max(a, edge * uDim);
    gl_FragColor = vec4(col, a);
    #include <colorspace_fragment>
  }`;

const TILE_VS = /* glsl */ `
  varying vec3 vColor;
  void main() {
    vec4 p = vec4(position, 1.0);
    #ifdef USE_INSTANCING
      p = instanceMatrix * p;
    #endif
    float shade = normal.z > 0.5 ? 1.0 : 0.5 + 0.2 * normal.y;
    #ifdef USE_INSTANCING_COLOR
      vColor = instanceColor * shade;
    #else
      vColor = vec3(shade);
    #endif
    gl_Position = projectionMatrix * modelViewMatrix * p;
  }`;
const TILE_FS = /* glsl */ `
  varying vec3 vColor;
  void main() {
    gl_FragColor = vec4(vColor, 1.0);
    #include <colorspace_fragment>
  }`;

// a few things that are not digits, as pen strokes in a unit square (y down)
const DOODLES = {
  A: [[[0.2, 0.88], [0.5, 0.12], [0.8, 0.88]], [[0.32, 0.6], [0.68, 0.6]]],
  X: [[[0.22, 0.14], [0.78, 0.86]], [[0.78, 0.14], [0.22, 0.86]]],
  face: [
    Array.from({ length: 33 }, (_, i) => [0.5 + 0.36 * Math.cos((i / 32) * Math.PI * 2), 0.5 + 0.36 * Math.sin((i / 32) * Math.PI * 2)]),
    [[0.38, 0.4], [0.38, 0.44]],
    [[0.62, 0.4], [0.62, 0.44]],
    Array.from({ length: 9 }, (_, i) => [0.36 + 0.035 * i, 0.6 + 0.09 * Math.sin((i / 8) * Math.PI)]),
  ],
};

export async function enter(ctx) {
  const { app, root } = ctx;
  ctx.ui.loading('Loading the CNN');
  const res = await fetch(ctx.asset('models/mnist-cnn.json'));
  if (!res.ok) throw new Error(`could not load models/mnist-cnn.json (${res.status})`);
  const model = loadModel(await res.json());
  const acts = createActs();
  // activations of the 24 sample digits, used to find the image patches that excite a map the most
  const sampleActs = model.samples.map((s) => {
    const a = forward(model, s.pixels);
    return { a1: a.a1, a2: a.a2, pred: a.pred };
  });

  const VIEW_DIR = [-0.3, 0.38, 1];
  const OVERVIEW = { target: [-0.75, 0.25, -1.4], dir: VIEW_DIR, fit: 8.0 };
  const viewOpts = { min: 1.2, max: 70, maxPolar: Math.PI * 0.62 };
  app.setView({ pos: OVERVIEW.target.map((v, i) => v + OVERVIEW.dir[i]), target: OVERVIEW.target, fit: OVERVIEW.fit, ...viewOpts });
  app.setBloom({ strength: 0.45, radius: 0.6 });
  root.add(gridFloor({ size: 70, y: -2.35, color: '#31508c', opacity: 0.3 }));

  /** Fly to a framing that fits a sphere of `fit` around `target` between the panels, and make it the home view. */
  function frame(v, dur = 1.5) {
    const cam = app.camera.position.clone();
    const tgt = app.controls.target.clone();
    app.setView({ pos: v.target.map((x, i) => x + v.dir[i]), target: v.target, fit: v.fit, ...viewOpts });
    const goal = app.camera.position.clone();
    app.camera.position.copy(cam);
    app.controls.target.copy(tgt);
    app.flyTo(goal, v.target, dur);
  }

  // ---------------------------------------------------------------- state
  const st = {
    sel1: 0, // selected map in conv1 / pool1
    sel2: 0, // selected map in conv2 / pool2
    selClass: -1, // class whose evidence the fan shows (-1 = the prediction)
    pre: 0, // 1 = show the selected conv1 map before ReLU (signed)
    preCur: 0,
    ext: 1, // extrusion of the input tiles
    extCur: 1,
    numbers: 0, // opacity of the pixel-value overlay
    numbersCur: 0,
    fan: 0.4,
    fanCur: 0.4,
    angle: 0,
    dx: 0,
    source: 'sample', // 'sample' | 'pad' | 'blank'
    sample: null,
    step: 0,
  };
  const scan = { mode: 'none', n: 1, pos: 0, f: 0, revealed: Infinity, playing: true, pinned: false };
  const base = new Float32Array(SIZE * SIZE); // the image before the rotate / shift sliders
  const maxOf = { a1: 1, z1: 1, a2: 1, p2: 1 };
  const disposables = [];

  // ---------------------------------------------------------------- input tiles
  const tileGeo = new THREE.BoxGeometry(CELL * 0.92, CELL * 0.92, 1);
  tileGeo.translate(0, 0, 0.5);
  const tiles = instanced(tileGeo, new THREE.ShaderMaterial({ vertexShader: TILE_VS, fragmentShader: TILE_FS }), SIZE * SIZE);
  root.add(tiles);
  const tileColor = new THREE.Color();
  const tileX = (x) => LX.input + (x - 13.5) * CELL;
  const tileY = (y) => (13.5 - y) * CELL;
  const tileTop = () => 0.02 + 0.3 * st.extCur;

  function paintInput() {
    const k = 0.3 * st.extCur;
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const v = acts.input[y * SIZE + x];
        setInst(tiles, y * SIZE + x, tileX(x), tileY(y), 0, 1, 1, 0.02 + k * v, seq(0.06 + 0.94 * v, tileColor));
      }
    flushInst(tiles);
  }

  // pixel values printed on the tiles (one canvas texture, only refreshed while it is visible)
  const numCanvas = document.createElement('canvas');
  numCanvas.width = numCanvas.height = 1400;
  const numG = numCanvas.getContext('2d');
  const numTex = new THREE.CanvasTexture(numCanvas);
  numTex.colorSpace = THREE.SRGBColorSpace;
  numTex.anisotropy = 4;
  const numbers = new THREE.Mesh(new THREE.PlaneGeometry(SIZE * CELL, SIZE * CELL), new THREE.MeshBasicMaterial({ map: numTex, transparent: true, depthWrite: false, opacity: 0, fog: false }));
  numbers.position.set(LX.input, 0, 0.06);
  numbers.renderOrder = 3;
  numbers.visible = false;
  root.add(numbers);
  let numbersDirty = true;
  function paintNumbers() {
    numbersDirty = false;
    const c = numCanvas.width / SIZE;
    numG.clearRect(0, 0, numCanvas.width, numCanvas.height);
    numG.font = `600 ${Math.round(c * 0.4)}px "JetBrains Mono Variable", ui-monospace, monospace`;
    numG.textAlign = 'center';
    numG.textBaseline = 'middle';
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const v = acts.input[y * SIZE + x];
        const n = Math.round(v * 255);
        numG.fillStyle = n === 0 ? 'rgba(180,191,214,0.3)' : v > 0.5 ? '#04121a' : '#e6ecf8';
        numG.fillText(String(n), (x + 0.5) * c, (y + 0.52) * c);
      }
    numTex.needsUpdate = true;
  }

  // ---------------------------------------------------------------- feature maps
  const planeGeo = new THREE.PlaneGeometry(1, 1);
  function makeMap(cols, rows, w, hgt) {
    const data = new Uint8Array(cols * rows * 4);
    const tex = new THREE.DataTexture(data, cols, rows, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    disposables.push(tex);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: tex },
        uCells: { value: new THREE.Vector2(cols, rows) },
        uSolid: { value: 0 },
        uDim: { value: 1 },
        uFrame: { value: glow('#8fa3c8', 0.8) },
        uFrameA: { value: 0.45 },
        uGain: { value: 1.18 },
      },
      vertexShader: MAP_VS,
      fragmentShader: MAP_FS,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(planeGeo, mat);
    mesh.scale.set(w, hgt, 1);
    return { mesh, tex, data, mat, cols, rows, solid: 0 };
  }

  const layerDefs = [
    { name: 'conv1', label: 'conv 1', n: S1, count: C1, dz: 0.5, src: () => acts.a1, max: () => maxOf.a1, sel: () => st.sel1 },
    { name: 'pool1', label: 'pool 1', n: P1, count: C1, dz: 0.5, src: () => acts.p1, max: () => maxOf.a1, sel: () => st.sel1 },
    { name: 'conv2', label: 'conv 2', n: S2, count: C2, dz: 0.22, src: () => acts.a2, max: () => maxOf.a2, sel: () => st.sel2 },
    { name: 'pool2', label: 'pool 2', n: P2, count: C2, dz: 0.22, src: () => acts.p2, max: () => maxOf.a2, sel: () => st.sel2 },
  ];
  const layers = {};
  const mapMeshes = [];
  for (const def of layerDefs) {
    const L = { ...def, x: LX[def.name], maps: [] };
    for (let i = 0; i < L.count; i++) {
      const m = makeMap(L.n, L.n, L.n * CELL, L.n * CELL);
      m.slotZ = DECK_Z - i * L.dz;
      m.mesh.position.set(L.x, 0, m.slotZ);
      m.mesh.userData = { layer: L.name, index: i };
      root.add(m.mesh);
      mapMeshes.push(m.mesh);
      L.maps.push(m);
    }
    layers[L.name] = L;
  }
  /** World position of the point (u, v) of a map, in cells from its top-left corner. */
  const cellX = (L, u) => L.x - (L.n * CELL) / 2 + u * CELL;
  const cellY = (L, v) => (L.n * CELL) / 2 - v * CELL;

  function paintMap(L, i) {
    const m = L.maps[i];
    const n = L.n;
    const src = L.src();
    const max = L.max() || 1;
    const off = i * n * n;
    const d = m.data;
    const selected = L.sel() === i;
    const revealed = selected && scan.mode === L.name ? scan.revealed : Infinity;
    const pre = selected && L.name === 'conv1' ? st.preCur : 0;
    for (let y = 0; y < n; y++)
      for (let x = 0; x < n; x++) {
        const k = y * n + x;
        const o = ((n - 1 - y) * n + x) * 4;
        if (k >= revealed) {
          d[o] = 9;
          d[o + 1] = 17;
          d[o + 2] = 34;
          d[o + 3] = 14;
          continue;
        }
        const t = Math.pow(src[off + k] / max, 0.6);
        const s = ((t * 255) | 0) * 3;
        let r = SEQ_LUT[s];
        let g = SEQ_LUT[s + 1];
        let b = SEQ_LUT[s + 2];
        if (pre > 0) {
          const z = acts.z1[off + k] / maxOf.z1;
          const q = Math.round((Math.sign(z) * Math.pow(Math.abs(z), 0.6) + 1) * 256) * 3;
          r += (DIV_LUT[q] - r) * pre;
          g += (DIV_LUT[q + 1] - g) * pre;
          b += (DIV_LUT[q + 2] - b) * pre;
        }
        d[o] = r;
        d[o + 1] = g;
        d[o + 2] = b;
        d[o + 3] = 40 + 215 * t;
      }
    m.tex.needsUpdate = true;
  }
  const paintLayer = (L) => {
    for (let i = 0; i < L.count; i++) paintMap(L, i);
  };

  // the flattened vector: 256 cells in one column, map 1 at the top
  const flat = makeMap(1, FLAT, 0.22, FLAT_H);
  flat.mesh.position.set(LX.flat, 0, 0);
  flat.mat.uniforms.uSolid.value = 1;
  flat.mat.uniforms.uFrameA.value = 0.7;
  flat.mesh.userData = { layer: 'flat', index: 0 };
  root.add(flat.mesh);
  const flatY = (i) => FLAT_H / 2 - ((i + 0.5) * FLAT_H) / FLAT;
  function paintFlat() {
    const max = maxOf.a2 || 1;
    for (let i = 0; i < FLAT; i++) {
      const s = ((Math.pow(acts.p2[i] / max, 0.6) * 255) | 0) * 3;
      const o = (FLAT - 1 - i) * 4;
      flat.data[o] = SEQ_LUT[s];
      flat.data[o + 1] = SEQ_LUT[s + 1];
      flat.data[o + 2] = SEQ_LUT[s + 2];
      flat.data[o + 3] = 255;
    }
    flat.tex.needsUpdate = true;
  }

  // thin guides: every pooled 4x4 map unrolls into 16 cells of the vector
  const unrollGeo = new THREE.BufferGeometry();
  const unrollPos = new Float32Array(C2 * 2 * 3);
  unrollGeo.setAttribute('position', new THREE.BufferAttribute(unrollPos, 3));
  const unroll = new THREE.LineSegments(unrollGeo, new THREE.LineBasicMaterial({ color: glow('#8fa3c8', 0.9), transparent: true, opacity: 0.3, depthWrite: false }));
  unroll.frustumCulled = false;
  root.add(unroll);

  // ---------------------------------------------------------------- output bars + classifier fan
  const barGeo = new THREE.BoxGeometry(1, 0.2, 0.2);
  barGeo.translate(0.5, 0, 0);
  const barY = (c) => 4.5 * BAR_DY - c * BAR_DY;
  const barBase = glow(SERIES[0], 1);
  const bars = Array.from({ length: CLASSES }, (_, c) => {
    const mesh = new THREE.Mesh(barGeo, new THREE.MeshBasicMaterial({ color: barBase.clone() }));
    mesh.position.set(LX.out, barY(c), 0);
    mesh.userData = { cls: c };
    const digit = textSprite(String(c), { size: 0.25, font: 'mono', weight: 700, color: '#e6ecf8' });
    digit.position.set(LX.out - 0.2, barY(c), 0);
    const pct = textSprite('0%', { size: 0.19, font: 'mono', weight: 600, color: '#b4bfd6', anchor: [0, 0.5] });
    pct.position.set(LX.out + 0.2, barY(c), 0);
    // generous invisible hit box so a thin bar is still easy to tap
    const hit = new THREE.Mesh(new THREE.BoxGeometry(BAR_LEN + 1.1, BAR_DY, 0.22), new THREE.MeshBasicMaterial({ visible: false }));
    hit.position.set(LX.out + BAR_LEN / 2 + 0.1, barY(c), 0);
    hit.userData = { cls: c };
    root.add(mesh, digit, pct, hit);
    return { mesh, digit, pct, hit, len: 0.02 };
  });

  const fanGeo = new THREE.BufferGeometry();
  const fanPos = new Float32Array(FLAT * 2 * 3);
  const fanCol = new Float32Array(FLAT * 2 * 3);
  fanGeo.setAttribute('position', new THREE.BufferAttribute(fanPos, 3));
  fanGeo.setAttribute('color', new THREE.BufferAttribute(fanCol, 3));
  const fanMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4, depthWrite: false, blending: THREE.AdditiveBlending });
  const fan = new THREE.LineSegments(fanGeo, fanMat);
  fan.frustumCulled = false;
  root.add(fan);
  const fanClass = () => (st.selClass >= 0 ? st.selClass : acts.pred);
  const fanTmp = new THREE.Color();
  function paintFan() {
    const c = fanClass();
    let max = 1e-9;
    for (let i = 0; i < FLAT; i++) max = Math.max(max, Math.abs(model.fc.w[c * FLAT + i] * acts.p2[i]));
    for (let i = 0; i < FLAT; i++) {
      const v = (model.fc.w[c * FLAT + i] * acts.p2[i]) / max;
      const mag = Math.pow(Math.abs(v), 0.7);
      div(Math.sign(v) * (0.35 + 0.65 * mag), fanTmp).multiplyScalar(mag * 1.5);
      fanPos.set([LX.flat + 0.13, flatY(i), 0, LX.out - 0.38, barY(c), 0], i * 6);
      fanCol.set([fanTmp.r, fanTmp.g, fanTmp.b, fanTmp.r, fanTmp.g, fanTmp.b], i * 6);
    }
    fanGeo.attributes.position.needsUpdate = true;
    fanGeo.attributes.color.needsUpdate = true;
  }

  // ---------------------------------------------------------------- captions
  // captions keep a constant size on screen, so they stay readable in the overview and modest in close-ups
  const caption = (str, x, y) => {
    const s = textSprite(str, { size: 0.0128, attenuate: false, color: '#a3b0cc', weight: 550, lineHeight: 1.3, anchor: [0.5, 1] });
    s.position.set(x, y, 0);
    root.add(s);
    return s;
  };
  caption('input\n28 × 28', LX.input, -1.52);
  caption('conv 5×5, ReLU\n8 maps of 24 × 24', LX.conv1, -1.32);
  caption('max-pool\n8 of 12 × 12', LX.pool1, -0.72);
  caption('conv 5×5, ReLU\n16 of 8 × 8', LX.conv2, -1.32);
  caption('max-pool\n16 of 4 × 4', LX.pool2, -0.32);
  caption('flatten\n256', LX.flat, -1.72);
  caption('linear, softmax\n10 classes', LX.out + 0.75, -1.72);

  // ---------------------------------------------------------------- scanner
  // drawn with screen-space ("fat") lines so the window stays visible at any zoom
  const MAX_SEG = 64;
  const scanGeo = new LineSegmentsGeometry();
  scanGeo.setPositions(new Float32Array(MAX_SEG * 6));
  const scanBuf = scanGeo.attributes.instanceStart.data;
  scanBuf.setUsage(THREE.DynamicDrawUsage);
  const scanPos = scanBuf.array;
  const scanMat = new LineMaterial({ color: glow(SCAN_COLOR, 1.9), linewidth: 2, transparent: true, opacity: 0.95, depthWrite: false, depthTest: false });
  const scanLines = new LineSegments2(scanGeo, scanMat);
  scanLines.frustumCulled = false;
  scanLines.renderOrder = 8;
  scanLines.visible = false;
  root.add(scanLines);
  // the sliding window shows the kernel itself: one 5x5 slice per input map
  const kernelWins = Array.from({ length: C1 }, () => {
    const m = makeMap(K, K, K * CELL, K * CELL);
    m.mat.uniforms.uSolid.value = 1;
    m.mat.uniforms.uDim.value = 0.8;
    m.mat.uniforms.uFrameA.value = 0;
    m.mat.depthTest = false;
    m.mesh.renderOrder = 7;
    m.mesh.visible = false;
    root.add(m.mesh);
    return m;
  });
  function paintKernelWins() {
    const paint = (m, w, off, max) => {
      for (let y = 0; y < K; y++)
        for (let x = 0; x < K; x++) {
          const q = Math.round((w[off + y * K + x] / max + 1) * 256) * 3;
          m.data.set([DIV_LUT[q], DIV_LUT[q + 1], DIV_LUT[q + 2], 255], ((K - 1 - y) * K + x) * 4);
        }
      m.tex.needsUpdate = true;
    };
    if (scan.mode === 'conv2') {
      const off = st.sel2 * C1 * K * K;
      let max = 1e-9;
      for (let i = 0; i < C1 * K * K; i++) max = Math.max(max, Math.abs(model.conv2.w[off + i]));
      kernelWins.forEach((m, c) => paint(m, model.conv2.w, off + c * K * K, max));
    } else {
      const off = st.sel1 * K * K;
      let max = 1e-9;
      for (let i = 0; i < K * K; i++) max = Math.max(max, Math.abs(model.conv1.w[off + i]));
      paint(kernelWins[0], model.conv1.w, off, max);
    }
  }

  let segCount = 0;
  const seg = (ax, ay, az, bx, by, bz) => {
    scanPos.set([ax, ay, az, bx, by, bz], segCount * 6);
    segCount++;
  };
  const rect = (x0, y0, x1, y1, z) => {
    seg(x0, y0, z, x1, y0, z);
    seg(x1, y0, z, x1, y1, z);
    seg(x1, y1, z, x0, y1, z);
    seg(x0, y1, z, x0, y0, z);
  };
  /** Connect the corners of two axis-aligned rectangles. */
  const link = (a, za, b, zb) => {
    seg(a[0], a[1], za, b[0], b[1], zb);
    seg(a[2], a[1], za, b[2], b[1], zb);
    seg(a[2], a[3], za, b[2], b[3], zb);
    seg(a[0], a[3], za, b[0], b[3], zb);
  };

  function layoutScanner() {
    const on = scan.mode !== 'none';
    scanLines.visible = on;
    kernelWins.forEach((m, c) => (m.mesh.visible = on && (scan.mode === 'conv2' || (scan.mode === 'conv1' && c === 0))));
    if (!on) return;
    segCount = 0;
    const n = scan.n;
    const row = Math.floor(scan.pos / n);
    const col = scan.pos % n;
    const u = col + (col < n - 1 ? scan.f : 0); // slide along the row, jump back at its end
    const v = row;
    if (scan.mode === 'conv1') {
      const L = layers.conv1;
      const zw = tileTop() + 0.05;
      const zc = L.maps[st.sel1].mesh.position.z + 0.012;
      const a = [tileX(u - 0.5), tileY(v - 0.5), tileX(u + 4.5), tileY(v + 4.5)];
      const b = [cellX(L, u), cellY(L, v), cellX(L, u + 1), cellY(L, v + 1)];
      rect(a[0], a[1], a[2], a[3], zw);
      rect(b[0], b[1], b[2], b[3], zc);
      link(a, zw, b, zc);
      kernelWins[0].mesh.position.set((a[0] + a[2]) / 2, (a[1] + a[3]) / 2, zw - 0.01);
    } else if (scan.mode === 'pool1') {
      const A = layers.conv1;
      const B = layers.pool1;
      const za = A.maps[st.sel1].mesh.position.z + 0.012;
      const zb = B.maps[st.sel1].mesh.position.z + 0.012;
      const a = [cellX(A, 2 * u), cellY(A, 2 * v), cellX(A, 2 * u + 2), cellY(A, 2 * v + 2)];
      const b = [cellX(B, u), cellY(B, v), cellX(B, u + 1), cellY(B, v + 1)];
      rect(a[0], a[1], a[2], a[3], za);
      rect(b[0], b[1], b[2], b[3], zb);
      link(a, za, b, zb);
    } else {
      const A = layers.pool1;
      const B = layers.conv2;
      const a = [cellX(A, u), cellY(A, v), cellX(A, u + 5), cellY(A, v + 5)];
      const b = [cellX(B, u), cellY(B, v), cellX(B, u + 1), cellY(B, v + 1)];
      let zMin = Infinity;
      let zMax = -Infinity;
      A.maps.forEach((m, c) => {
        const z = m.mesh.position.z + 0.012;
        zMin = Math.min(zMin, z);
        zMax = Math.max(zMax, z);
        rect(a[0], a[1], a[2], a[3], z);
        kernelWins[c].mesh.position.set((a[0] + a[2]) / 2, (a[1] + a[3]) / 2, z - 0.004);
      });
      link(a, zMin, a, zMax); // the kernel is a 5x5x8 block through the whole deck
      const zb = B.maps[st.sel2].mesh.position.z + 0.012;
      rect(b[0], b[1], b[2], b[3], zb);
      link(a, zMax, b, zb);
    }
    scanGeo.instanceCount = segCount;
    scanBuf.needsUpdate = true;
  }

  const scanLayer = () => (scan.mode === 'none' ? null : layers[scan.mode]);
  function setScan(mode, { restart = false } = {}) {
    const changed = scan.mode !== mode;
    const old = scanLayer();
    scan.mode = mode;
    const L = scanLayer();
    scan.n = L ? L.n : 1;
    if (changed || restart) {
      scan.pos = 0;
      scan.f = 0;
      scan.pinned = false;
      scan.playing = !ctx.reduceMotion;
      scan.revealed = restart && !ctx.reduceMotion ? 0 : Infinity;
      if (ctx.reduceMotion && L) scan.pos = bestCell(L);
    }
    if (old && old !== L) paintMap(old, old.sel());
    if (L) {
      paintMap(L, L.sel());
      paintKernelWins();
    }
    scanBtn.set(scan.playing && !scan.pinned);
    updateCalc();
  }
  /** Index of the strongest cell of the selected map of a layer. */
  function bestCell(L) {
    const src = L.src();
    const off = L.sel() * L.n * L.n;
    let best = 0;
    for (let k = 1; k < L.n * L.n; k++) if (src[off + k] > src[off + best]) best = k;
    return best;
  }

  // ---------------------------------------------------------------- right panel: pad + predictions
  const padCanvas = h('canvas.cnn-pad', { width: PAD, height: PAD, 'aria-label': 'Drawing pad: draw a digit' });
  const pg = padCanvas.getContext('2d', { willReadFrequently: true });
  const thumb = document.createElement('canvas');
  thumb.width = thumb.height = SIZE;
  const thumbG = thumb.getContext('2d');
  let padDirty = false;
  let drawing = null;
  const clearPad = () => {
    pg.fillStyle = '#000';
    pg.fillRect(0, 0, PAD, PAD);
  };
  const pen = () => {
    pg.strokeStyle = '#fff';
    pg.fillStyle = '#fff';
    pg.lineWidth = PAD * 0.078;
    pg.lineCap = 'round';
    pg.lineJoin = 'round';
  };
  const padPoint = (e) => {
    const r = padCanvas.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * PAD, ((e.clientY - r.top) / r.height) * PAD];
  };
  padCanvas.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    padCanvas.setPointerCapture?.(e.pointerId);
    if (st.source !== 'pad') clearPad();
    st.source = 'pad';
    st.sample = null;
    drawing = padPoint(e);
    pen();
    pg.beginPath();
    pg.arc(drawing[0], drawing[1], pg.lineWidth / 2, 0, Math.PI * 2);
    pg.fill();
    padDirty = true;
  });
  padCanvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    e.preventDefault();
    const p = padPoint(e);
    pen();
    pg.beginPath();
    pg.moveTo(drawing[0], drawing[1]);
    pg.lineTo(p[0], p[1]);
    pg.stroke();
    drawing = p;
    padDirty = true;
  });
  const endStroke = () => (drawing = null);
  padCanvas.addEventListener('pointerup', endStroke);
  padCanvas.addEventListener('pointercancel', endStroke);

  const gray = new Float32Array(PAD * PAD);
  function readPad() {
    padDirty = false;
    const img = pg.getImageData(0, 0, PAD, PAD).data;
    for (let i = 0; i < PAD * PAD; i++) gray[i] = img[i * 4];
    const pre = preprocessDrawing(gray, PAD, PAD, base);
    if (pre.empty) st.source = 'blank';
    applyInput();
  }
  function drawStrokes(strokes) {
    clearPad();
    pen();
    for (const s of strokes) {
      pg.beginPath();
      s.forEach(([x, y], i) => (i ? pg.lineTo(x * PAD, y * PAD) : pg.moveTo(x * PAD, y * PAD)));
      pg.stroke();
    }
    st.source = 'pad';
    st.sample = null;
    readPad();
  }
  function showOnPad(pixels) {
    const img = thumbG.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const v = Math.round(pixels[i] * 255);
      img.data.set([v, v, v, 255], i * 4);
    }
    thumbG.putImageData(img, 0, 0);
    clearPad();
    pg.imageSmoothingEnabled = true;
    pg.drawImage(thumb, 0, 0, PAD, PAD);
  }
  function loadSample(i) {
    const s = model.samples[i];
    st.source = 'sample';
    st.sample = i;
    base.set(s.pixels);
    showOnPad(s.pixels);
    applyInput();
  }
  const cursor = {}; // which of a digit's samples comes next
  function sampleOf(digit) {
    const list = model.samples.map((s, i) => i).filter((i) => model.samples[i].kind === 'typical' && model.samples[i].label === digit);
    cursor[digit] = ((cursor[digit] ?? -1) + 1) % list.length;
    loadSample(list[cursor[digit]]);
  }
  const errorList = model.samples.map((s, i) => i).filter((i) => model.samples[i].kind === 'error');
  let errorCursor = -1;

  const probRows = Array.from({ length: CLASSES }, (_, c) => {
    const bar = h('i');
    const val = h('span.pv', '');
    const row = h('button.cnn-prob', { type: 'button', title: `Show the evidence for ${c}` }, h('span.pd', String(c)), h('span.pb', bar), val);
    row.addEventListener('click', () => {
      st.selClass = st.selClass === c ? -1 : c;
      paintFan();
      updateProbs();
    });
    return { row, bar, val };
  });
  const status = h('div.cnn-status');
  function updateProbs() {
    const shown = fanClass();
    probRows.forEach((r, c) => {
      const p = acts.probs[c];
      r.bar.style.width = (p * 100).toFixed(1) + '%';
      r.val.textContent = p >= 0.995 ? '100%' : p >= 0.1 ? (p * 100).toFixed(0) + '%' : (p * 100).toFixed(1) + '%';
      r.row.classList.toggle('on', c === acts.pred);
      r.row.classList.toggle('pick', c === shown && st.selClass >= 0);
    });
    const conf = fmt.pct(acts.probs[acts.pred], 1);
    let src;
    if (st.source === 'blank') src = 'Empty pad. The network still answers';
    else if (st.source === 'sample') {
      const s = model.samples[st.sample];
      src = `MNIST test digit, labeled ${s.label}. The network says`;
    } else src = 'Your drawing. The network says';
    status.replaceChildren(h('span', src + ' '), h('b', String(acts.pred)), h('span', ` (${conf})`));
  }

  // ---------------------------------------------------------------- right panel: the selected map
  const kernelMap = heatmap({ rows: ['', '', '', '', ''], cols: ['', '', '', '', ''], values: [[0]], mode: 'div', cell: 24, colLabels: false, format: (v) => v.toFixed(3), rowLabel: 'row', colLabel: 'column' });
  const kernelNote = h('div.txt.small');
  const patchCanvas = h('canvas.cnn-patches', { width: 120, height: 120 });
  const patchNote = h('div.cnn-cap');
  const kernelBox = h('div.cnn-kernel', kernelMap.el, h('div.cnn-patchbox', patchCanvas, patchNote));
  const mapChips = h('div.cnn-chips');
  const layerSeg = segmented([{ value: 'conv1', label: 'Layer 1 · 8 maps' }, { value: 'conv2', label: 'Layer 2 · 16 maps' }], 'conv1', (v) => showKernel(v));
  let kernelLayer = 'conv1';

  function showKernel(which = kernelLayer) {
    kernelLayer = which;
    layerSeg.set(which);
    const count = which === 'conv1' ? C1 : C2;
    const sel = which === 'conv1' ? st.sel1 : st.sel2;
    mapChips.replaceChildren(
      ...Array.from({ length: count }, (_, i) => {
        const b = h('button' + (i === sel ? '.on' : ''), { type: 'button', 'aria-label': `Select map ${i + 1}` }, String(i + 1));
        b.addEventListener('click', () => select(which, i));
        return b;
      }),
    );
    kernelBox.classList.toggle('cnn-k2', which === 'conv2');
    if (which === 'conv1') {
      const w = model.conv1.w.subarray(sel * K * K, (sel + 1) * K * K);
      kernelMap.update({
        rows: ['1', '2', '3', '4', '5'],
        cols: ['1', '2', '3', '4', '5'],
        values: Array.from({ length: K }, (_, y) => Array.from(w.subarray(y * K, y * K + K))),
        max: w.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-9),
        cell: 24,
      });
      kernelNote.replaceChildren(text(`**Kernel ${sel + 1}**, bias ${model.conv1.b[sel].toFixed(2)}. Red weights add, blue subtract. ${describeKernel(w)}`).firstChild);
    } else {
      // eight 5x5 slices, laid out 4 by 2 with an empty row / column between them
      const off = sel * C1 * K * K;
      const values = Array.from({ length: 11 }, () => new Array(23).fill(NaN));
      let max = 1e-9;
      for (let c = 0; c < C1; c++)
        for (let y = 0; y < K; y++)
          for (let x = 0; x < K; x++) {
            const v = model.conv2.w[off + c * K * K + y * K + x];
            values[(c >> 2) * 6 + y][(c & 3) * 6 + x] = v;
            max = Math.max(max, Math.abs(v));
          }
      const rows = new Array(11).fill('');
      rows[2] = 'maps 1–4';
      rows[8] = 'maps 5–8';
      kernelMap.update({ rows, cols: new Array(23).fill(''), values, max, cell: 14 });
      kernelNote.replaceChildren(text(`**Kernel ${sel + 1}** of layer 2, bias ${model.conv2.b[sel].toFixed(2)}. ${describeConv2(model, sel)}`).firstChild);
    }
    drawPatches(which, sel);
  }

  /** The patches of the sample digits that give the selected map its largest values. */
  function drawPatches(which, sel) {
    const n = which === 'conv1' ? S1 : S2;
    const found = sampleActs
      .map((sa, s) => {
        const a = which === 'conv1' ? sa.a1 : sa.a2;
        let best = 0;
        for (let k = 1; k < n * n; k++) if (a[sel * n * n + k] > a[sel * n * n + best]) best = k;
        return { s, k: best, v: a[sel * n * n + best] };
      })
      .sort((a, b) => b.v - a.v)
      .slice(0, 4);
    const size = 56;
    const gapPx = 8;
    const perRow = which === 'conv1' ? 2 : 4;
    patchCanvas.width = perRow * size + (perRow - 1) * gapPx;
    patchCanvas.height = (4 / perRow) * size + (4 / perRow - 1) * gapPx;
    patchCanvas.style.width = patchCanvas.width + 'px';
    const g = patchCanvas.getContext('2d');
    found.forEach((f, j) => {
      const rf = receptiveField(which, f.k % n, Math.floor(f.k / n));
      const px = model.samples[f.s].pixels;
      const c = size / rf.size;
      const x0 = (j % perRow) * (size + gapPx);
      const y0 = Math.floor(j / perRow) * (size + gapPx);
      for (let y = 0; y < rf.size; y++)
        for (let x = 0; x < rf.size; x++) {
          const s = ((px[(rf.y + y) * SIZE + rf.x + x] * 255) | 0) * 3;
          g.fillStyle = `rgb(${SEQ_LUT[s]},${SEQ_LUT[s + 1]},${SEQ_LUT[s + 2]})`;
          g.fillRect(x0 + x * c, y0 + y * c, Math.ceil(c), Math.ceil(c));
        }
    });
    const rf = which === 'conv1' ? 5 : 14;
    patchNote.textContent = which === 'conv1' ? 'Where it peaks: 5 × 5 patches from the sample digits.' : 'Where it peaks: the 14 × 14 pixel patches, out of the 24 sample digits, that excite this map the most.';
  }

  function select(which, i) {
    if (which === 'conv1' || which === 'pool1') {
      const old = st.sel1;
      st.sel1 = i;
      if (old !== i) {
        paintMap(layers.conv1, old);
        paintMap(layers.conv1, i);
        paintMap(layers.pool1, old);
        paintMap(layers.pool1, i);
      }
      kernelLayer = 'conv1';
    } else {
      const old = st.sel2;
      st.sel2 = i;
      if (old !== i) {
        paintMap(layers.conv2, old);
        paintMap(layers.conv2, i);
        paintMap(layers.pool2, old);
        paintMap(layers.pool2, i);
      }
      kernelLayer = 'conv2';
    }
    if (scan.mode !== 'none') {
      scan.pinned = false;
      paintKernelWins();
    }
    showKernel();
    updateCalc();
  }

  // ---------------------------------------------------------------- left panel: live arithmetic
  const calcCanvas = h('canvas.cnn-calc', { width: 560, height: 220 });
  const calcText = h('div.cnn-cap');
  const calcBox = h('div.cnn-calcbox', calcCanvas, calcText);
  function updateCalc() {
    if (!calcBox.isConnected || scan.mode === 'none') return;
    const g = calcCanvas.getContext('2d');
    const W = calcCanvas.width;
    const H = calcCanvas.height;
    g.clearRect(0, 0, W, H);
    const n = scan.n;
    const cx = scan.pos % n;
    const cy = Math.floor(scan.pos / n);
    const grid = (x0, size, cells, value, color, label) => {
      const c = size / cells;
      for (let y = 0; y < cells; y++)
        for (let x = 0; x < cells; x++) {
          g.fillStyle = color(value(x, y));
          g.fillRect(x0 + x * c + 1, 30 + y * c + 1, c - 2, c - 2);
        }
      g.fillStyle = '#93a1bf';
      g.font = '500 20px "Inter Variable", system-ui, sans-serif';
      g.textAlign = 'center';
      g.fillText(label, x0 + size / 2, 20);
    };
    const seqCol = (t) => {
      const s = ((Math.max(0, Math.min(1, t)) * 255) | 0) * 3;
      return `rgb(${SEQ_LUT[s]},${SEQ_LUT[s + 1]},${SEQ_LUT[s + 2]})`;
    };
    const divCol = (t) => {
      const q = Math.round((Math.max(-1, Math.min(1, t)) + 1) * 256) * 3;
      return `rgb(${DIV_LUT[q]},${DIV_LUT[q + 1]},${DIV_LUT[q + 2]})`;
    };
    const big = (str, x) => {
      g.fillStyle = '#e6ecf8';
      g.font = '650 30px "JetBrains Mono Variable", ui-monospace, monospace';
      g.textAlign = 'center';
      g.fillText(str, x, 132);
    };
    if (scan.mode === 'conv1') {
      const w = model.conv1.w;
      const off = st.sel1 * K * K;
      let wmax = 1e-9;
      for (let i = 0; i < K * K; i++) wmax = Math.max(wmax, Math.abs(w[off + i]));
      grid(0, 170, K, (x, y) => acts.input[(cy + y) * SIZE + cx + x], seqCol, '25 pixels');
      big('×', 195);
      grid(220, 170, K, (x, y) => w[off + y * K + x] / wmax, divCol, '25 weights');
      const z = acts.z1[st.sel1 * S1 * S1 + scan.pos];
      big('=', 415);
      big(z.toFixed(2), 495);
      calcText.textContent = `Position (${cx + 1}, ${cy + 1}): sum of 25 products ${(z - model.conv1.b[st.sel1]).toFixed(2)}, plus bias ${model.conv1.b[st.sel1].toFixed(2)}, gives ${z.toFixed(2)}. After ReLU: ${Math.max(0, z).toFixed(2)}.`;
    } else if (scan.mode === 'pool1') {
      const a = acts.a1;
      const off = st.sel1 * S1 * S1;
      const max = maxOf.a1 || 1;
      const vals = [0, 1, 2, 3].map((q) => a[off + (2 * cy + (q >> 1)) * S1 + 2 * cx + (q & 1)]);
      grid(60, 170, 2, (x, y) => Math.pow(vals[y * 2 + x] / max, 0.6), seqCol, '4 neighbors');
      g.fillStyle = '#e6ecf8';
      g.font = '600 24px "JetBrains Mono Variable", ui-monospace, monospace';
      vals.forEach((v, q) => {
        g.fillStyle = Math.pow(v / max, 0.6) > 0.55 ? '#04121a' : '#e6ecf8';
        g.fillText(v.toFixed(2), 60 + 42.5 + (q & 1) * 85, 30 + 50 + (q >> 1) * 85);
      });
      big('max =', 330);
      big(Math.max(...vals).toFixed(2), 470);
      calcText.textContent = `Block (${cx + 1}, ${cy + 1}): the largest of the four values survives; the other three are dropped.`;
    } else {
      // conv2: eight small patches, one per pooled map, each with its own slice of the kernel
      const off = st.sel2 * C1 * K * K;
      const max = maxOf.a1 || 1;
      let total = model.conv2.b[st.sel2];
      const parts = [];
      for (let c = 0; c < C1; c++) {
        let s = 0;
        for (let y = 0; y < K; y++) for (let x = 0; x < K; x++) s += acts.p1[c * P1 * P1 + (cy + y) * P1 + cx + x] * model.conv2.w[off + c * K * K + y * K + x];
        parts.push(s);
        total += s;
        const c0 = 64 * c;
        const cell = 56 / K;
        for (let y = 0; y < K; y++)
          for (let x = 0; x < K; x++) {
            g.fillStyle = seqCol(Math.pow(acts.p1[c * P1 * P1 + (cy + y) * P1 + cx + x] / max, 0.6));
            g.fillRect(c0 + x * cell, 30 + y * cell, cell - 1, cell - 1);
          }
      }
      const pmax = Math.max(1e-9, ...parts.map(Math.abs));
      parts.forEach((s, c) => {
        g.fillStyle = divCol((s / pmax) * 0.9);
        const hgt = (Math.abs(s) / pmax) * 44;
        g.fillRect(64 * c + 14, s >= 0 ? 150 - hgt : 150, 28, hgt);
      });
      g.fillStyle = '#93a1bf';
      g.font = '500 20px "Inter Variable", system-ui, sans-serif';
      g.textAlign = 'left';
      g.fillText('the same 5 × 5 window on all 8 pooled maps', 0, 20);
      g.fillText('each map’s share of the sum (red adds, blue subtracts)', 0, 214);
      g.fillRect(0, 150, 504, 1);
      calcText.textContent = `Position (${cx + 1}, ${cy + 1}): 200 products plus bias ${model.conv2.b[st.sel2].toFixed(2)} give ${total.toFixed(2)}. After ReLU: ${Math.max(0, total).toFixed(2)}.`;
    }
  }

  // ---------------------------------------------------------------- running the network
  function applyInput() {
    if (st.angle !== 0 || st.dx !== 0) transformImage(base, acts.input, { angle: st.angle, dx: st.dx });
    else acts.input.set(base);
    run();
  }
  function run() {
    forward(model, acts.input, acts);
    const maxAbs = (a) => {
      let m = 1e-6;
      for (let i = 0; i < a.length; i++) {
        const v = Math.abs(a[i]);
        if (v > m) m = v;
      }
      return m;
    };
    maxOf.a1 = maxAbs(acts.a1);
    maxOf.z1 = maxAbs(acts.z1);
    maxOf.a2 = maxAbs(acts.a2);
    paintInput();
    for (const L of Object.values(layers)) paintLayer(L);
    paintFlat();
    paintFan();
    updateProbs();
    updateCalc();
    numbersDirty = true;
  }

  const rotSlider = slider({ label: 'Rotate', min: -180, max: 180, step: 1, value: 0, format: (v) => v.toFixed(0) + '°', onInput: (v) => { st.angle = v; applyInput(); } });
  const shiftSlider = slider({ label: 'Shift sideways', min: -10, max: 10, step: 1, value: 0, format: (v) => (v > 0 ? '+' : '') + v.toFixed(0) + ' px', onInput: (v) => { st.dx = v; applyInput(); } });
  function setPerturb(angle, dx) {
    st.angle = angle;
    st.dx = dx;
    rotSlider.set(angle);
    shiftSlider.set(dx);
    applyInput();
  }
  const scanBtn = toggle({ label: 'Scanner running', value: true, onChange: (v) => { scan.playing = v; scan.pinned = false; } });

  // ---------------------------------------------------------------- tour
  const views = {
    input: { target: [LX.input, 0, 0.1], dir: [0.02, 0.06, 1], fit: 1.72 },
    conv1: { target: [-4.75, 0.05, -0.3], dir: [-0.28, 0.3, 1], fit: 3.5 },
    relu: { target: [LX.conv1 - 0.2, 0.2, -0.9], dir: [-0.32, 0.34, 1], fit: 2.5 },
    pool: { target: [-1.9, 0.05, -0.4], dir: [-0.3, 0.3, 1], fit: 2.6 },
    conv2: { target: [0.45, 0.35, -1.9], dir: [-0.62, 0.48, 1], fit: 3.0 },
    head: { target: [3.55, 0, -0.9], dir: [-0.42, 0.3, 1], fit: 3.3 },
    all: OVERVIEW,
  };
  /** Configure the scene for a tour step. */
  function stage(o) {
    st.ext = o.ext ?? 1;
    st.numbers = o.numbers ?? 0;
    st.pre = o.pre ?? 0;
    st.fan = o.fan ?? 0.4;
    if (o.kernel) showKernel(o.kernel);
    setScan(o.scan ?? 'none', { restart: true });
    frame(views[o.view]);
  }
  const reluSeg = segmented([{ value: 1, label: 'Before ReLU' }, { value: 0, label: 'After ReLU' }], 1, (v) => (st.pre = v));

  const params = paramTable(model);
  const denseFirst = SIZE * SIZE * C1 * S1 * S1 + C1 * S1 * S1; // a fully connected layer with the same input and output sizes as conv1
  const steps = [
    {
      title: 'Pixels are numbers',
      text: `The network never sees a digit. It sees a grid of 28 × 28 = **784 numbers**: 0 for empty background, up to 255 for full ink (divided by 255 before they enter the network).\n\nEvery later layer is arithmetic on these numbers. Nothing else goes in.`,
      body: () => callout('Draw a digit in the pad, or pick a sample. A drawing is cropped, scaled to 20 × 20 and centered on its center of mass in the 28 × 28 frame: the same recipe that produced the MNIST training digits.', 'key'),
      enter: () => stage({ view: 'input', ext: 0.06, numbers: 1 }),
    },
    {
      title: 'One convolution kernel',
      text: `A **kernel** is a 5 × 5 grid of weights. Slide it over the image. At each position multiply the 25 pixels under it by the 25 weights, add them up and add a bias. The result is one cell of a **feature map**: large where the image looks like the kernel’s pattern.\n\nThe same 25 weights are used at all 24 × 24 positions.`,
      body: () => calcBox,
      enter: () => stage({ view: 'conv1', scan: 'conv1', pre: 1, kernel: 'conv1' }),
    },
    {
      title: 'ReLU keeps what matched',
      text: `A sum can be negative: the patch looked like the opposite of the pattern (blue). **ReLU** replaces every negative value with zero and keeps positive values as they are: \`max(0, x)\`.\n\nWhat remains is a map of where the pattern was found. Without a nonlinearity between them, two stacked convolutions would collapse into a single one.`,
      body: () => h('div', reluSeg.el),
      enter: () => {
        stage({ view: 'relu', scan: 'none', pre: 1, kernel: 'conv1' });
        reluSeg.set(1);
        ctx.after(1700, () => {
          if (st.step !== 2) return;
          st.pre = 0;
          reluSeg.set(0);
        });
      },
    },
    {
      title: 'Pooling shrinks',
      text: `**Max-pooling** looks at each 2 × 2 block and keeps only the largest value. The map shrinks from 24 × 24 to 12 × 12, a quarter of the numbers.\n\nA stroke that moves by one pixel often gives the same pooled value, so small shifts stop mattering, and the next layer covers twice the distance with the same kernel size.`,
      body: () => calcBox,
      enter: () => stage({ view: 'pool', scan: 'pool1', kernel: 'conv1' }),
    },
    {
      title: 'Strokes become parts',
      text: `The second convolution reads **all eight pooled maps at once**. Each of its 16 kernels is a 5 × 5 × 8 block of 200 weights that asks for a combination, such as one kind of stroke above another.\n\nBecause of the pooling in between, one cell now sees a 14 × 14 patch of the image: enough for corners, loops and line ends.`,
      body: () => calcBox,
      enter: () => stage({ view: 'conv2', scan: 'conv2', kernel: 'conv2' }),
    },
    {
      title: 'The classifier head',
      text: `After a second pooling, 16 maps of 4 × 4 remain. They are **flattened** into one list of 256 numbers.\n\nThe last layer is a weighted vote: each of the ten outputs adds up the 256 numbers with its own weights, and **softmax** turns the ten sums into probabilities that total 100%. The lines show which numbers push the chosen digit up (red) or down (blue).`,
      body: () => callout('Click an output bar, or a row of the prediction list, to see the evidence for a different digit.', 'key'),
      enter: () => stage({ view: 'head', fan: 1 }),
    },
    {
      title: 'Why convolution',
      text: `**Weight sharing.** A kernel is reused at every position, so the first layer needs only ${fmt.int(params[0].count)} parameters to compute ${fmt.int(C1 * S1 * S1)} numbers. A fully connected layer between the same sizes would need ${fmt.int(denseFirst)}.\n\nThis whole network has **${fmt.int(model.params)} parameters** and reads ${fmt.pct(model.testAccuracy, 2)} of the ${fmt.int(model.testCount)} MNIST test digits correctly. Sharing also builds in an assumption that suits images: a stroke is the same stroke wherever it appears.`,
      body: () =>
        h(
          'div',
          table(['Layer', 'What it holds', 'Parameters'], [...params.map((p) => [p.name, p.what, fmt.int(p.count)]), ['total', '', fmt.int(model.params)]], { compact: true }),
          h('div.cnn-cap', `Trained for ${model.train.epochs} passes over the 60,000 MNIST training digits (${model.train.seconds} s on a CPU), with random shifts, rotations and rescaling.`),
        ),
      enter: () => stage({ view: 'all' }),
    },
    {
      title: 'Try to fool it',
      text: `The network has only ever seen centered, upright digits. Shift or rotate the input, or give it something that is not a digit.\n\nIt cannot answer “none of these”: softmax always spreads 100% over the ten digits, often with high confidence. **Confidence is not correctness**, and that holds for much larger models too.`,
      body: () =>
        h(
          'div.btnrow',
          button('Letter A', { onClick: () => drawStrokes(DOODLES.A) }),
          button('Letter X', { onClick: () => drawStrokes(DOODLES.X) }),
          button('A face', { onClick: () => drawStrokes(DOODLES.face) }),
          button('Rotate 90°', { onClick: () => setPerturb(90, st.dx) }),
          button('Shift 7 px', { onClick: () => setPerturb(st.angle, 7) }),
          button('Undo shift and rotation', { kind: 'ghost', onClick: () => setPerturb(0, 0) }),
        ),
      enter: () => stage({ view: 'all', fan: 0.8 }),
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'Next lab →', onFinish: () => ctx.router.go('labs/diffusion'), onStep: (i) => (st.step = i) });

  ctx.ui.left(
    header({
      kicker: 'Lab · How a CNN Sees',
      title: 'A network that reads handwriting',
      lead: `A convolutional network with ${fmt.int(model.params)} parameters, trained on MNIST. Draw a digit and follow it through every layer.`,
    }),
    guide.body,
  );
  ctx.ui.right(
    section(
      'Draw a digit',
      h('div.cnn-draw', padCanvas, h('div.cnn-probs', probRows.map((r) => r.row))),
      status,
      h(
        'div.cnn-samples',
        button('Clear', { kind: 'small', onClick: () => { clearPad(); st.source = 'blank'; st.sample = null; base.fill(0); applyInput(); } }),
        h('span.cnn-digits', Array.from({ length: CLASSES }, (_, d) => h('button', { type: 'button', title: `Load a test digit ${d}`, onclick: () => sampleOf(d) }, String(d)))),
        errorList.length > 0 &&
          button('Misread', {
            kind: 'small',
            title: 'Test digits this network gets wrong',
            onClick: () => {
              errorCursor = (errorCursor + 1) % errorList.length;
              loadSample(errorList[errorCursor]);
            },
          }),
      ),
    ),
    section('One feature map', layerSeg, mapChips, kernelBox, kernelNote, scanBtn),
    section('Stress it', rotSlider, shiftSlider),
  );
  ctx.ui.dock(guide.dock);

  // ---------------------------------------------------------------- picking
  ctx.pick.add(tiles, {
    cursor: 'default',
    tip: (hit) => {
      const i = hit.instanceId;
      const v = acts.input[i];
      return `pixel (${(i % SIZE) + 1}, ${Math.floor(i / SIZE) + 1}) = ${Math.round(v * 255)} / 255 = ${v.toFixed(2)}`;
    },
  });
  const cellOf = (hit) => {
    const { layer, index } = hit.object.userData;
    const L = layers[layer];
    const x = Math.min(L.n - 1, Math.floor(hit.uv.x * L.n));
    const y = Math.min(L.n - 1, Math.floor((1 - hit.uv.y) * L.n));
    return { L, index, x, y, value: L.src()[index * L.n * L.n + y * L.n + x] };
  };
  ctx.pick.add(mapMeshes, {
    tip: (hit) => {
      const c = cellOf(hit);
      const picked = c.L.sel() === c.index;
      return `${c.L.label} · map ${c.index + 1} · cell (${c.x + 1}, ${c.y + 1}) = ${c.value.toFixed(2)}${picked ? '' : '\nclick to bring this map to the front'}`;
    },
    click: (hit) => {
      const c = cellOf(hit);
      if (c.L.sel() !== c.index) return select(c.L.name, c.index);
      if (scan.mode === c.L.name) {
        // park the scanner on the clicked cell
        scan.pos = c.y * c.L.n + c.x;
        scan.f = 0;
        scan.pinned = true;
        scan.revealed = Infinity;
        scanBtn.set(false);
        paintMap(c.L, c.index);
        updateCalc();
      }
    },
  });
  ctx.pick.add(flat.mesh, {
    cursor: 'default',
    tip: (hit) => {
      const i = Math.min(FLAT - 1, Math.floor((1 - hit.uv.y) * FLAT));
      const c = fanClass();
      return `value ${i + 1} of 256 (map ${Math.floor(i / 16) + 1}) = ${acts.p2[i].toFixed(2)}\n× weight ${model.fc.w[c * FLAT + i].toFixed(2)} for “${c}”`;
    },
  });
  ctx.pick.add(bars.map((b) => b.hit), {
    tip: (hit) => `P(${hit.object.userData.cls}) = ${fmt.pct(acts.probs[hit.object.userData.cls], 1)}   logit ${acts.logits[hit.object.userData.cls].toFixed(2)}`,
    click: (hit) => {
      st.selClass = st.selClass === hit.object.userData.cls ? -1 : hit.object.userData.cls;
      paintFan();
      updateProbs();
    },
  });
  ctx.key(' ', () => {
    scan.playing = !(scan.playing && !scan.pinned);
    scan.pinned = false;
    scanBtn.set(scan.playing);
  });

  // ---------------------------------------------------------------- start
  clearPad();
  showKernel('conv1');
  loadSample(model.samples.findIndex((s) => s.label === 7));

  const hot = new THREE.Color();
  return {
    update(dt) {
      if (padDirty) readPad();

      // tiles: extrusion and the number overlay follow the tour step
      if (Math.abs(st.extCur - st.ext) > 0.002) {
        st.extCur = damp(st.extCur, st.ext, 0.18, dt);
        paintInput();
      }
      st.numbersCur = damp(st.numbersCur, st.numbers, 0.2, dt);
      numbers.visible = st.numbersCur > 0.02;
      if (numbers.visible) {
        numbers.material.opacity = st.numbersCur;
        numbers.position.z = tileTop() + 0.012;
        if (numbersDirty) paintNumbers();
      }
      if (Math.abs(st.preCur - st.pre) > 0.004) {
        st.preCur = Math.abs(st.preCur - st.pre) < 0.01 ? st.pre : damp(st.preCur, st.pre, 0.22, dt);
        paintMap(layers.conv1, st.sel1);
      }
      st.fanCur = damp(st.fanCur, st.fan, 0.25, dt);
      fanMat.opacity = st.fanCur;

      // selected maps slide to the front lane, the others stay in their deck slot
      for (const L of Object.values(layers)) {
        const sel = L.sel();
        for (let i = 0; i < L.count; i++) {
          const m = L.maps[i];
          const on = i === sel ? 1 : 0;
          m.mesh.position.z = damp(m.mesh.position.z, on ? 0 : m.slotZ, 0.14, dt);
          m.solid = damp(m.solid, on, 0.1, dt);
          const u = m.mat.uniforms;
          u.uSolid.value = m.solid;
          u.uFrameA.value = 0.55 + 0.4 * m.solid;
          m.mat.depthWrite = m.solid > 0.9;
        }
      }
      const P = layers.pool2;
      for (let j = 0; j < C2; j++) {
        const m = P.maps[j].mesh.position;
        unrollPos.set([m.x + 0.2, 0, m.z, LX.flat - 0.11, flatY(j * 16 + 7.5), 0], j * 6);
      }
      unrollGeo.attributes.position.needsUpdate = true;

      // scanner
      const L = scanLayer();
      if (L) {
        const total = L.n * L.n;
        if (scan.playing && !scan.pinned) {
          const revealing = scan.revealed < total;
          const rate = revealing ? Math.min({ conv1: 110, pool1: 34, conv2: 13 }[scan.mode], 5 + scan.pos * { conv1: 1.4, pool1: 1, conv2: 0.6 }[scan.mode]) : { conv1: 16, pool1: 8, conv2: 5 }[scan.mode];
          scan.f += dt * rate;
          let moved = false;
          while (scan.f >= 1) {
            scan.f -= 1;
            scan.pos++;
            moved = true;
            if (scan.pos >= total) {
              scan.pos = 0;
              scan.revealed = Infinity;
            }
          }
          if (moved) {
            if (scan.revealed < total) scan.revealed = scan.pos + 1;
            paintMap(L, L.sel());
            updateCalc();
          }
        } else scan.f = 0;
      }
      layoutScanner();

      // output bars ease towards the probabilities
      const shown = fanClass();
      for (let c = 0; c < CLASSES; c++) {
        const b = bars[c];
        b.len = damp(b.len, 0.02 + BAR_LEN * acts.probs[c], 0.1, dt);
        b.mesh.scale.x = b.len;
        b.mesh.material.color.copy(hot.copy(barBase).multiplyScalar(c === acts.pred ? 2.3 : c === shown ? 1.5 : 0.85));
        b.pct.position.x = LX.out + b.len + 0.1;
        const p = acts.probs[c];
        b.pct.userData.setText(p >= 0.995 ? '100%' : Math.round(p * 100) + '%');
      }
    },
    dispose() {
      for (const d of disposables) d.dispose();
    },
    // handles for the headless tests (tools/shot.mjs --eval)
    lab: {
      st,
      scan,
      acts,
      select,
      loadSample,
      drawStrokes,
      setPerturb,
      finishScan() {
        const L = scanLayer();
        if (!L) return;
        scan.revealed = Infinity;
        paintMap(L, L.sel());
      },
    },
  };
}
