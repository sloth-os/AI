// Neural Network Lab: a real multilayer perceptron trained live on 2D toy data.
// Two linked parts: the network itself (weights, activations, forward and backward pulses) and its
// decision surface as a terrain. The math lives in ../ml/mlp.js; this file draws it and wires the controls.
import * as THREE from 'three';
import './neural.css';
import { MLP, DATASETS, DOMAIN, makeDataset, splitDataset } from '../ml/mlp.js';
import { PAL, SERIES, div, divCss, glow } from '../core/colors.js';
import { gridFloor, line, segments, ring, points, instanced, setInst, flushInst, glowMat, glassMat } from '../core/geom.js';
import { textSprite, textPlane, htmlLabel } from '../core/labels.js';
import { h, header, section, text, callout, button, segmented, slider, toggle, stat, statRow, tour, fmt } from '../core/ui.js';
import { lineChart } from '../core/charts.js';

const S = 2.9; // half-size of the surface in world units
const H = 2.4; // world height of probability 1
const NET_W = 5.4; // width of the widest network (five columns)
const NET_H = 4.2; // height of the tallest column (eight units)
const MAX_UNITS = 8;
const MAX_HIDDEN = 3;
const MAX_NODES = 2 + MAX_HIDDEN * MAX_UNITS + 1;
const MAX_EDGES = 2 * MAX_UNITS + (MAX_HIDDEN - 1) * MAX_UNITS * MAX_UNITS + MAX_UNITS;
const N_TEST = 150;
const MAX_POINTS = 400 + N_TEST;
const FOLD_SEG = 22;
const RAMP_GAIN = 0.82; // the surface uses the middle of the diverging ramp so that data points stay readable on it
const L2_STEPS = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1];
const BATCH_STEPS = [1, 2, 4, 8, 16, 32, 64, 0]; // 0 = the whole training set
const COLOR_FWD = PAL.mint;
const COLOR_BACK = PAL.amber;
const TRAIN_COLOR = SERIES[2];
const TEST_COLOR = SERIES[6];
// seconds per stage of the pulse animation
const T_GAP = 0.36;
const T_LINK = 0.32;
const T_TURN = 0.16;
const T_REST = 0.5;

const DEFAULTS = { dataset: 'moons', noise: 0.2, nTrain: 200, hidden: [6, 4], activation: 'tanh', optimizer: 'sgd', lr: 0.3, batch: 16, l2: 0, dataSeed: 3, netSeed: 3 };
const VIEW_DEFAULTS = { pulses: 'both', sticks: 'miss', folds: false, snap: false };

const toX = (x1) => (x1 / DOMAIN) * S;
const toZ = (x2) => (-x2 / DOMAIN) * S;
const f2 = (v) => v.toFixed(2).replace('-', '−');
const pct = (v) => (Number.isFinite(v) ? (v * 100).toFixed(v > 0.9995 || v < 0.0005 ? 0 : 1) + '%' : '–');
const _c = new THREE.Color();
const _v = new THREE.Vector3();
const _size = new THREE.Vector2();

/** Round data markers with a crisp outline: discs (training), rings (test), a white halo when misclassified. */
function glyphMaterial(opacity, depthTest) {
  const mat = new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 1 }, uOpacity: { value: opacity } },
    vertexShader: /* glsl */ `
      attribute vec3 aColor;
      attribute float aSize;
      attribute float aStyle;
      attribute float aMiss;
      varying vec3 vColor;
      varying float vStyle;
      varying float vMiss;
      uniform float uScale;
      void main() {
        vColor = aColor;
        vStyle = aStyle;
        vMiss = aMiss;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = max(4.0, aSize * (1.0 + 0.8 * aMiss) * uScale / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      varying float vStyle;
      varying float vMiss;
      uniform float uOpacity;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        float aa = fwidth(d) * 1.2;
        float R = vMiss > 0.5 ? 0.56 : 1.0;
        float body = 1.0 - smoothstep(R - aa, R, d);
        float hole = vStyle > 0.5 ? smoothstep(R * 0.4 - aa, R * 0.4, d) : 1.0;
        float rim = smoothstep(R * 0.78 - aa, R * 0.78, d);
        vec3 col = mix(vColor, vec3(0.015, 0.02, 0.04), rim * 0.9);
        float a = body * hole;
        if (vMiss > 0.5) {
          float halo = smoothstep(0.72 - aa, 0.72, d) * (1.0 - smoothstep(1.0 - aa, 1.0, d));
          col = mix(col, vec3(1.6), halo);
          a = max(a, halo);
        }
        if (a < 0.02) discard;
        gl_FragColor = vec4(col, a * uOpacity);
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
    depthTest,
  });
  return mat;
}

export async function enter(ctx) {
  const { app, root } = ctx;
  const N = ctx.quality === 'low' ? 48 : ctx.quality === 'medium' ? 64 : 80; // surface grid resolution
  const V = N + 1;
  app.setBloom({ strength: 0.5, radius: 0.6 });
  root.add(gridFloor({ size: 60, y: -0.08, color: '#27606a', opacity: 0.3 }));

  // ------------------------------------------------------------------ state
  const cfg = { ...DEFAULTS, hidden: [...DEFAULTS.hidden] };
  const view = { ...VIEW_DEFAULTS };
  const unitsMemory = [6, 4, 4]; // units per hidden layer remembered while a layer is switched off
  let net = null;
  let train = null;
  let test = null;
  let playing = false;
  let speed = 100; // mini-batch steps per second; below 1 = slow motion, one step per pulse cycle
  let stepAcc = 0;
  let pendingStep = false; // slow motion: a gradient has been computed and waits for the end of the backward sweep
  let backOnce = false; // show one backward sweep after a manual step
  let evalTimer = 0;
  let uiTimer = 0;
  let liveFn = null;
  const metrics = { train: { loss: NaN, acc: NaN }, test: { loss: NaN, acc: NaN } };
  const hist = { train: [], test: [] };
  const probe = new Float64Array([0.4, 0.3]);
  let probeLabel = -1; // class of the probed point when it is a training example, else -1
  let probeP = 0.5;
  const batchSize = () => (cfg.batch > 0 ? Math.min(cfg.batch, cfg.nTrain) : cfg.nTrain);

  // ------------------------------------------------------------------ the decision surface
  const terG = new THREE.Group();
  terG.position.set(2.55, 0, 1.25);
  root.add(terG);

  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(V * V * 3);
  const gx = new Float64Array(V); // data coordinate of grid column / row i
  for (let i = 0; i < V; i++) gx[i] = -DOMAIN + (i / N) * 2 * DOMAIN;
  for (let j = 0; j < V; j++)
    for (let i = 0; i < V; i++) {
      const k = (j * V + i) * 3;
      pos[k] = toX(gx[i]);
      pos[k + 1] = 0.5 * H;
      pos[k + 2] = toZ(gx[j]);
    }
  const index = new Uint16Array(N * N * 6);
  for (let j = 0, t = 0; j < N; j++)
    for (let i = 0; i < N; i++) {
      const a = j * V + i;
      index[t++] = a;
      index[t++] = a + 1;
      index[t++] = a + V;
      index[t++] = a + 1;
      index[t++] = a + V + 1;
      index[t++] = a + V;
    }
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  const posAttr = new THREE.BufferAttribute(pos, 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('position', posAttr);
  // the height changes every frame; fixed bounds keep picking correct without recomputing them
  geo.boundingBox = new THREE.Box3(new THREE.Vector3(-S, 0, -S), new THREE.Vector3(S, H, S));
  geo.boundingSphere = geo.boundingBox.getBoundingSphere(new THREE.Sphere());

  // the diverging ramp as a lookup texture: texel t holds div() for probability t
  const rampData = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    div(((i / 255) * 2 - 1) * RAMP_GAIN, _c).convertLinearToSRGB();
    rampData[i * 4] = Math.round(_c.r * 255);
    rampData[i * 4 + 1] = Math.round(_c.g * 255);
    rampData[i * 4 + 2] = Math.round(_c.b * 255);
    rampData[i * 4 + 3] = 255;
  }
  const rampTex = new THREE.DataTexture(rampData, 256, 1, THREE.RGBAFormat);
  rampTex.colorSpace = THREE.SRGBColorSpace;
  rampTex.minFilter = rampTex.magFilter = THREE.LinearFilter;
  rampTex.needsUpdate = true;
  ctx.onDispose(() => rampTex.dispose());

  const surfMat = new THREE.ShaderMaterial({
    uniforms: { uRamp: { value: rampTex }, uH: { value: H } },
    vertexShader: /* glsl */ `
      uniform float uH;
      varying float vP;
      varying vec3 vWorld;
      void main() {
        vP = position.y / uH;
        vec4 w = modelMatrix * vec4(position, 1.0);
        vWorld = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D uRamp;
      varying float vP;
      varying vec3 vWorld;
      void main() {
        vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
        float diff = 0.56 + 0.44 * abs(dot(n, normalize(vec3(0.3, 1.0, 0.35))));
        vec3 col = texture2D(uRamp, vec2(clamp(vP, 0.002, 0.998), 0.5)).rgb * diff;
        // a contour every 0.1 of probability, and a bright line where the two classes meet (p = 0.5)
        float v = vP * 10.0;
        float w = max(fwidth(v), 1e-4);
        float f = fract(v);
        float contour = (1.0 - smoothstep(0.0, w * 1.3, min(f, 1.0 - f))) * step(0.05, vP) * step(vP, 0.95);
        col += contour * 0.07;
        float bw = max(fwidth(vP), 1e-5);
        float boundary = 1.0 - smoothstep(bw * 0.9, bw * 2.2, abs(vP - 0.5));
        col = mix(col, vec3(1.5), boundary * 0.92);
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }`,
    side: THREE.DoubleSide,
  });
  const surface = new THREE.Mesh(geo, surfMat);
  terG.add(surface);

  // base plate, frame of the unit cube, axis labels
  const plate = new THREE.Mesh(new THREE.PlaneGeometry(2 * S, 2 * S), glassMat('#0b1b2c', 0.55));
  plate.rotation.x = -Math.PI / 2;
  plate.position.y = -0.04;
  terG.add(plate);
  const corners = [[-S, -S], [S, -S], [S, S], [-S, S]];
  terG.add(line(corners.map(([x, z]) => [x, 0, z]), PAL.steel, { k: 1.1, opacity: 0.85, closed: true }));
  terG.add(line(corners.map(([x, z]) => [x, H, z]), PAL.steel, { k: 0.9, opacity: 0.28, closed: true }));
  terG.add(segments(corners.flatMap(([x, z]) => [[x, 0, z], [x, H, z]]), PAL.steel, { k: 0.9, opacity: 0.28 }));
  const axisStyle = { size: 0.3, color: '#93a1bf', weight: 550 };
  const ax1 = textPlane('x₁ →', axisStyle);
  ax1.rotation.x = -Math.PI / 2;
  ax1.position.set(0, 0, S + 0.4);
  const ax2 = new THREE.Group();
  const ax2Text = textPlane('x₂ →', axisStyle);
  ax2Text.rotation.x = -Math.PI / 2;
  ax2.add(ax2Text);
  ax2.rotation.y = Math.PI / 2;
  ax2.position.set(S + 0.4, 0, 0);
  terG.add(ax1, ax2);
  for (const [label, y] of [['0', 0], ['0.5', H / 2], ['1', H]]) {
    const s = textSprite(label, { size: 0.22, color: '#7f8aa5', weight: 500, font: 'mono', anchor: [1, 0.5] });
    s.position.set(-S - 0.14, y, -S);
    terG.add(s);
  }
  const axP = textSprite('P(red)', { size: 0.26, color: '#93a1bf', weight: 550, anchor: [0.5, 0] });
  axP.position.set(-S, H + 0.2, -S);
  terG.add(axP);

  // data points: one buffer, training points first
  const gPos = new Float32Array(MAX_POINTS * 3);
  const gCol = new Float32Array(MAX_POINTS * 3);
  const gSize = new Float32Array(MAX_POINTS);
  const gStyle = new Float32Array(MAX_POINTS);
  const gMiss = new Float32Array(MAX_POINTS);
  const glyphGeo = new THREE.BufferGeometry();
  glyphGeo.setAttribute('position', new THREE.BufferAttribute(gPos, 3));
  glyphGeo.setAttribute('aColor', new THREE.BufferAttribute(gCol, 3));
  glyphGeo.setAttribute('aSize', new THREE.BufferAttribute(gSize, 1));
  glyphGeo.setAttribute('aStyle', new THREE.BufferAttribute(gStyle, 1));
  glyphGeo.setAttribute('aMiss', new THREE.BufferAttribute(gMiss, 1));
  const scaleHook = (mat) => (renderer, _scene, camera) => {
    mat.uniforms.uScale.value = renderer.getDrawingBufferSize(_size).y / (2 * Math.tan((camera.fov * Math.PI) / 360));
  };
  // drawn twice: faintly through the surface, then solid where nothing is in front
  for (const [opacity, depthTest, order] of [[0.26, false, 3], [1, true, 4]]) {
    const mat = glyphMaterial(opacity, depthTest);
    const pts = new THREE.Points(glyphGeo, mat);
    pts.frustumCulled = false;
    pts.renderOrder = order;
    pts.onBeforeRender = scaleHook(mat);
    terG.add(pts);
  }
  const CLASS = [div(-1), div(1)];
  const CLASS_CSS = [divCss(-0.8), divCss(0.8)];

  // error sticks: from each training point (at its label) to the surface (the prediction)
  const sPos = new Float32Array(400 * 6);
  const sCol = new Float32Array(400 * 6);
  const stickGeo = new THREE.BufferGeometry();
  stickGeo.setAttribute('position', new THREE.BufferAttribute(sPos, 3));
  stickGeo.setAttribute('color', new THREE.BufferAttribute(sCol, 3));
  const sticks = new THREE.LineSegments(stickGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
  sticks.frustumCulled = false;
  terG.add(sticks);

  // "folds": the line where each first-layer unit switches sign, draped on the surface
  const fPos = new Float32Array(MAX_UNITS * FOLD_SEG * 6);
  const foldGeo = new THREE.BufferGeometry();
  foldGeo.setAttribute('position', new THREE.BufferAttribute(fPos, 3));
  const folds = new THREE.LineSegments(foldGeo, new THREE.LineBasicMaterial({ color: glow('#d6fff0', 1.25), transparent: true, opacity: 0.8, depthWrite: false }));
  folds.frustumCulled = false;
  folds.visible = false;
  terG.add(folds);

  // probe: the point whose forward pass the network shows
  const probeG = new THREE.Group();
  const probeStem = line([[0, 0, 0], [0, H + 0.25, 0]], '#ffffff', { k: 1.2, opacity: 0.55 });
  const probeHead = new THREE.Mesh(new THREE.SphereGeometry(0.085, 16, 12), glowMat('#ffffff', { k: 2 }));
  const probeRing = ring(0.15, '#ffffff', { k: 1.6, segments: 32 });
  probeRing.position.y = 0.01;
  const probeLab = htmlLabel('', 'label3d nn-val');
  probeLab.center.set(0.5, 1);
  probeLab.position.y = H + 0.38;
  probeG.add(probeStem, probeHead, probeRing, probeLab);
  terG.add(probeG);

  // ------------------------------------------------------------------ the network
  const netG = new THREE.Group();
  netG.position.set(-2.75, 3.0, -2.3);
  netG.rotation.y = 0.14;
  root.add(netG);

  const nodeX = new Float32Array(MAX_NODES);
  const nodeY = new Float32Array(MAX_NODES);
  const nodeR = new Float32Array(MAX_NODES);
  const nodeLayer = new Int8Array(MAX_NODES);
  const nodeIdx = new Int8Array(MAX_NODES);
  const nodeVal = new Float32Array(MAX_NODES); // displayed value in [-1, 1]
  const nodeW = new Float32Array(MAX_NODES * 3); // world position
  const layerStart = new Int16Array(MAX_HIDDEN + 3);
  const flash = new Float32Array(MAX_HIDDEN + 2);
  const flashDir = new Int8Array(MAX_HIDDEN + 2);
  let nodeCount = 0;

  const eSrc = new Int16Array(MAX_EDGES);
  const eDst = new Int16Array(MAX_EDGES);
  const eLayer = new Int8Array(MAX_EDGES);
  const eW = new Int16Array(MAX_EDGES); // index into W[layer]
  const eUx = new Float32Array(MAX_EDGES);
  const eUy = new Float32Array(MAX_EDGES);
  const eLen = new Float32Array(MAX_EDGES);
  const eMx = new Float32Array(MAX_EDGES);
  const eMy = new Float32Array(MAX_EDGES);
  const eShown = new Float32Array(MAX_EDGES); // displayed (smoothed) weight
  let edgeCount = 0;

  const sphere = new THREE.SphereGeometry(1, 24, 16);
  const units = instanced(sphere, new THREE.MeshBasicMaterial(), MAX_NODES);
  const rims = instanced(sphere, new THREE.MeshBasicMaterial({ side: THREE.BackSide }), MAX_NODES);
  const edges = instanced(
    new THREE.CylinderGeometry(1, 1, 1, 6, 1, true),
    new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }),
    MAX_EDGES,
  );
  edges.renderOrder = 1;
  netG.add(edges, rims, units);

  const captions = Array.from({ length: MAX_HIDDEN + 2 }, () => {
    const s = textSprite(' ', { size: 0.2, color: '#93a1bf', weight: 500, anchor: [0.5, 1] });
    netG.add(s);
    return s;
  });
  const inLabs = [0, 1].map(() => {
    const l = htmlLabel('', 'label3d nn-val');
    l.center.set(1, 0.5);
    netG.add(l);
    return l;
  });
  const outLab = htmlLabel('', 'label3d nn-val');
  outLab.center.set(0, 0.5);
  netG.add(outLab);
  // weight and bias read-outs, only for the single neuron
  const wLabs = [0, 1, 2].map(() => {
    const l = htmlLabel('', 'label3d nn-val dim');
    netG.add(l);
    return l;
  });

  // link from the output unit to the probed point on the surface
  const LINK_N = 28;
  const linkPos = new Float32Array(LINK_N * 3);
  const linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute('position', new THREE.BufferAttribute(linkPos, 3));
  const link = new THREE.Line(linkGeo, new THREE.LineBasicMaterial({ color: glow('#ffffff', 1), transparent: true, opacity: 0.3, depthWrite: false }));
  link.frustumCulled = false;
  root.add(link);

  // pulses: one forward and one backward slot per edge, plus one for the link
  const P = 2 * MAX_EDGES + 1;
  const pPos = new Float32Array(P * 3);
  const pCol = new Float32Array(P * 3);
  const pSize = new Float32Array(P);
  const pulses = points(pPos, pCol, pSize);
  pulses.renderOrder = 6;
  root.add(pulses);
  const FWD = glow(COLOR_FWD, 2.3);
  const BACK = glow(COLOR_BACK, 2.5);
  const FWD_RIM = glow(COLOR_FWD, 1.5);
  const BACK_RIM = glow(COLOR_BACK, 1.6);
  const RIM = new THREE.Color('#5d7396');

  function layoutNetwork() {
    const sizes = net.sizes;
    const nl = sizes.length;
    const dx = Math.min(2.0, NET_W / (nl - 1));
    nodeCount = 0;
    for (let l = 0; l < nl; l++) {
      layerStart[l] = nodeCount;
      const n = sizes[l];
      const dy = Math.min(0.92, NET_H / Math.max(1, n - 1));
      for (let i = 0; i < n; i++) {
        const k = nodeCount++;
        nodeX[k] = (l - (nl - 1) / 2) * dx;
        nodeY[k] = ((n - 1) / 2 - i) * dy;
        nodeR[k] = l === 0 || l === nl - 1 ? 0.25 : n > 6 ? 0.19 : 0.22;
        nodeLayer[k] = l;
        nodeIdx[k] = i;
        nodeVal[k] = 0;
      }
    }
    layerStart[nl] = nodeCount;
    netG.updateMatrixWorld(true);
    for (let k = 0; k < nodeCount; k++) _v.set(nodeX[k], nodeY[k], 0).applyMatrix4(netG.matrixWorld).toArray(nodeW, k * 3);

    edgeCount = 0;
    edges.instanceMatrix.array.fill(0);
    for (let l = 0; l < nl - 1; l++)
      for (let j = 0; j < sizes[l + 1]; j++)
        for (let i = 0; i < sizes[l]; i++) {
          const e = edgeCount++;
          const a = layerStart[l] + i;
          const b = layerStart[l + 1] + j;
          const ex = nodeX[b] - nodeX[a];
          const ey = nodeY[b] - nodeY[a];
          const len = Math.hypot(ex, ey);
          eSrc[e] = a;
          eDst[e] = b;
          eLayer[e] = l;
          eW[e] = j * sizes[l] + i;
          eUx[e] = ex / len;
          eUy[e] = ey / len;
          eLen[e] = len;
          eMx[e] = (nodeX[a] + nodeX[b]) / 2;
          eMy[e] = (nodeY[a] + nodeY[b]) / 2;
          eShown[e] = net.W[l][eW[e]];
        }
    units.count = rims.count = nodeCount;
    edges.count = edgeCount;

    const act = cfg.activation === 'relu' ? 'ReLU' : 'tanh';
    captions.forEach((s, l) => {
      s.visible = l < nl;
      if (l >= nl) return;
      const name = l === 0 ? 'inputs' : l === nl - 1 ? 'output · sigmoid' : `hidden ${l} · ${sizes[l]} ${act}`;
      s.userData.setText(name);
      s.position.set(nodeX[layerStart[l]], -NET_H / 2 - 0.42, 0);
    });
    inLabs.forEach((lab, i) => lab.position.set(nodeX[i] - 0.4, nodeY[i], 0));
    const o = nodeCount - 1;
    outLab.position.set(nodeX[o] + 0.4, nodeY[o], 0);
    // the single neuron shows its two weights and its bias
    const single = net.L === 1;
    wLabs.forEach((lab) => (lab.visible = single));
    if (single) {
      wLabs[0].position.set(eMx[0], eMy[0] + 0.3, 0);
      wLabs[1].position.set(eMx[1], eMy[1] - 0.3, 0);
      wLabs[2].position.set(nodeX[o], nodeY[o] - 0.55, 0);
    }
    flash.fill(0);
    updateEdges(1);
    updateUnits(1);
    units.computeBoundingSphere();
  }

  function updateEdges(dt) {
    const f = 1 - Math.exp(-dt / 0.12);
    const m = edges.instanceMatrix.array;
    const col = edges.instanceColor.array;
    for (let e = 0; e < edgeCount; e++) {
      const w = (eShown[e] += (net.W[eLayer[e]][eW[e]] - eShown[e]) * f);
      const a = Math.abs(w);
      const mag = a / (a + 1.2); // saturating: a weight of 1.2 is drawn at half the maximum width
      const r = 0.007 + 0.052 * mag;
      const o = e * 16;
      m[o] = eUy[e] * r;
      m[o + 1] = -eUx[e] * r;
      m[o + 4] = eUx[e] * eLen[e];
      m[o + 5] = eUy[e] * eLen[e];
      m[o + 10] = r;
      m[o + 12] = eMx[e];
      m[o + 13] = eMy[e];
      m[o + 15] = 1;
      div((w < 0 ? -1 : 1) * (0.45 + 0.5 * mag), _c);
      const b = 0.14 + 1.25 * mag;
      col[e * 3] = _c.r * b;
      col[e * 3 + 1] = _c.g * b;
      col[e * 3 + 2] = _c.b * b;
    }
    edges.instanceMatrix.needsUpdate = true;
    edges.instanceColor.needsUpdate = true;
  }

  /** Forward pass for the probed point, then color every unit by its activation. */
  function updateUnits(dt) {
    const f = 1 - Math.exp(-dt / 0.09);
    const L = net.L;
    const relu = cfg.activation === 'relu';
    probeP = net.prob1(probe);
    let k = 0;
    for (let l = 0; l <= L; l++) {
      const a = net.a[l];
      flash[l] *= Math.exp(-dt / 0.28);
      const fl = flash[l];
      for (let i = 0; i < a.length; i++, k++) {
        const v = l === 0 ? a[i] / DOMAIN : l === L ? 2 * probeP - 1 : relu ? a[i] / (1 + a[i]) : a[i];
        const shown = (nodeVal[k] += (v - nodeVal[k]) * f);
        const mag = Math.min(1, Math.abs(shown));
        const r = nodeR[k] * (1 + 0.16 * fl);
        div(shown * 0.92, _c).multiplyScalar(0.5 + 0.85 * mag);
        setInst(units, k, nodeX[k], nodeY[k], 0, r, r, r, _c);
        _c.copy(RIM).multiplyScalar(0.75 + 0.5 * mag);
        if (fl > 0.01) _c.lerp(flashDir[l] < 0 ? BACK_RIM : FWD_RIM, Math.min(1, fl));
        setInst(rims, k, nodeX[k], nodeY[k], 0, r * 1.15 + 0.012, r * 1.15 + 0.012, r * 1.15 + 0.012, _c);
      }
    }
    flushInst(units);
    flushInst(rims);
  }

  // ------------------------------------------------------------------ data and model
  function buildData() {
    const total = cfg.nTrain + N_TEST;
    const data = makeDataset(cfg.dataset, total, cfg.noise, cfg.dataSeed * 7919 + 13);
    ({ train, test } = splitDataset(data, cfg.nTrain));
    for (let i = 0; i < total; i++) {
      const y = data.y[i];
      gPos[i * 3] = toX(data.X[i * 2]);
      gPos[i * 3 + 1] = y ? H : 0.015;
      gPos[i * 3 + 2] = toZ(data.X[i * 2 + 1]);
      CLASS[y].toArray(gCol, i * 3);
      gStyle[i] = i < cfg.nTrain ? 0 : 1;
      gSize[i] = i < cfg.nTrain ? 0.19 : 0.17;
      gMiss[i] = 0;
    }
    glyphGeo.setDrawRange(0, total);
    for (const name of ['position', 'aColor', 'aSize', 'aStyle']) glyphGeo.attributes[name].needsUpdate = true;
    stickGeo.setDrawRange(0, cfg.nTrain * 2);
  }

  function buildNet() {
    net = new MLP({ inputs: 2, hidden: cfg.hidden, outputs: 1, activation: cfg.activation, optimizer: cfg.optimizer, lr: cfg.lr, l2: cfg.l2, batch: batchSize(), seed: cfg.netSeed * 104729 + 7 });
    hist.train.length = 0;
    hist.test.length = 0;
    stepAcc = 0;
    pendingStep = false;
    backOnce = false;
    resetWave();
    layoutNetwork();
    surfaceDirty();
    evaluate(true);
  }

  const pTrain = new Float64Array(400);
  const pTest = new Float64Array(N_TEST);

  /** Loss and accuracy on both sets, plus the per-point marks (misclassified halo, error sticks). */
  function evaluate(record) {
    net.evaluate(train.X, train.y, train.n, metrics.train);
    net.evaluate(test.X, test.y, test.n, metrics.test);
    net.predictProba(train.X, train.n, pTrain);
    net.predictProba(test.X, test.n, pTest);
    const all = view.sticks === 'all';
    for (let i = 0; i < train.n; i++) {
      const p = pTrain[i];
      const y = train.y[i];
      const miss = p >= 0.5 !== (y === 1);
      gMiss[i] = miss ? 1 : 0;
      const o = i * 6;
      sPos[o] = sPos[o + 3] = gPos[i * 3];
      sPos[o + 2] = sPos[o + 5] = gPos[i * 3 + 2];
      sPos[o + 1] = y ? H : 0;
      sPos[o + 4] = p * H;
      const b = miss ? 0.95 : all ? 0.3 : 0;
      for (let c = 0; c < 6; c++) sCol[o + c] = b;
    }
    for (let i = 0; i < test.n; i++) gMiss[train.n + i] = pTest[i] >= 0.5 !== (test.y[i] === 1) ? 1 : 0;
    glyphGeo.attributes.aMiss.needsUpdate = true;
    stickGeo.attributes.position.needsUpdate = true;
    stickGeo.attributes.color.needsUpdate = true;
    if (view.folds) updateFolds();
    if (record) {
      const epoch = net.seen / train.n;
      const last = hist.train[hist.train.length - 1];
      if (!last || epoch > last[0]) {
        hist.train.push([epoch, metrics.train.loss]);
        hist.test.push([epoch, metrics.test.loss]);
        if (hist.train.length > 700)
          for (const arr of [hist.train, hist.test]) {
            let k = 0;
            for (let i = 0; i < arr.length; i += 2) arr[k++] = arr[i];
            arr.length = k;
          }
      }
    }
    evalTimer = 0;
  }

  // ---- surface: recomputed a few rows per frame, then eased towards the new heights
  const target = new Float32Array(V * V).fill(0.5);
  const shown = new Float32Array(V * V).fill(0.5);
  const q = new Float64Array(2);
  let sweepRow = 0;
  let rowsLeft = V;
  let settled = false;
  const surfaceDirty = () => (rowsLeft = V);

  function sweep(budgetMs) {
    if (rowsLeft <= 0) return;
    const t0 = performance.now();
    do {
      const j = sweepRow;
      q[1] = gx[j];
      for (let i = 0, k = j * V; i < V; i++, k++) {
        q[0] = gx[i];
        target[k] = net.prob1(q);
      }
      sweepRow = (sweepRow + 1) % V;
      rowsLeft--;
    } while (rowsLeft > 0 && performance.now() - t0 < budgetMs);
    settled = false;
  }

  function easeSurface(dt) {
    if (settled) return;
    const f = 1 - Math.exp(-dt / 0.1);
    let maxD = 0;
    for (let k = 0; k < shown.length; k++) {
      const d = target[k] - shown[k];
      if (d > maxD) maxD = d;
      else if (-d > maxD) maxD = -d;
      shown[k] += d * f;
      pos[k * 3 + 1] = shown[k] * H;
    }
    posAttr.needsUpdate = true;
    if (maxD < 0.0004 && rowsLeft <= 0) settled = true;
  }

  function updateFolds() {
    let n = 0;
    const D = DOMAIN;
    if (net.L >= 1) {
      const W = net.W[0];
      const b = net.b[0];
      const count = Math.min(MAX_UNITS, net.sizes[1]);
      for (let j = 0; j < count; j++) {
        const w1 = W[j * 2];
        const w2 = W[j * 2 + 1];
        // clip the line w1·x1 + w2·x2 + b = 0 to the square
        const ends = [];
        if (Math.abs(w2) > 1e-9) for (const x of [-D, D]) { const y = -(w1 * x + b[j]) / w2; if (Math.abs(y) <= D) ends.push([x, y]); }
        if (Math.abs(w1) > 1e-9) for (const y of [-D, D]) { const x = -(w2 * y + b[j]) / w1; if (Math.abs(x) < D) ends.push([x, y]); }
        if (ends.length < 2) continue;
        const [a, c] = ends;
        let px = 0;
        let py = 0;
        let pz = 0;
        for (let s = 0; s <= FOLD_SEG; s++) {
          q[0] = a[0] + ((c[0] - a[0]) * s) / FOLD_SEG;
          q[1] = a[1] + ((c[1] - a[1]) * s) / FOLD_SEG;
          const x = toX(q[0]);
          const y = net.prob1(q) * H + 0.035;
          const z = toZ(q[1]);
          if (s > 0) {
            fPos[n++] = px;
            fPos[n++] = py;
            fPos[n++] = pz;
            fPos[n++] = x;
            fPos[n++] = y;
            fPos[n++] = z;
          }
          px = x;
          py = y;
          pz = z;
        }
      }
    }
    foldGeo.setDrawRange(0, n / 3);
    foldGeo.attributes.position.needsUpdate = true;
  }

  function setProbe(x1, x2, label = -1) {
    const lim = DOMAIN * 0.995;
    probe[0] = Math.max(-lim, Math.min(lim, x1));
    probe[1] = Math.max(-lim, Math.min(lim, x2));
    probeLabel = label;
    probeG.position.set(toX(probe[0]), 0, toZ(probe[1]));
  }

  /** Move the probe to the training point nearest to (x1, x2). */
  function snapProbe(x1, x2) {
    let best = 0;
    let bd = Infinity;
    for (let i = 0; i < train.n; i++) {
      const d = (train.X[i * 2] - x1) ** 2 + (train.X[i * 2 + 1] - x2) ** 2;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    setProbe(train.X[best * 2], train.X[best * 2 + 1], train.y[best]);
  }

  // ------------------------------------------------------------------ training
  function trainSteps(n) {
    for (let s = 0; s < n; s++) net.step(train.X, train.y, train.n);
    if (n > 0) surfaceDirty();
  }

  /** Slow motion: pick the next mini-batch, compute its gradient, and show its last example. */
  function beginSlowStep() {
    net.nextBatch(train.n);
    net.batchGradient(train.X, train.y, net.order, net.from, net.to);
    const i = net.order[net.to - 1];
    setProbe(train.X[i * 2], train.X[i * 2 + 1], train.y[i]);
    pendingStep = true;
  }

  function finishSlowStep() {
    net.applyGradient();
    pendingStep = false;
    surfaceDirty();
    evaluate(true);
  }

  // ------------------------------------------------------------------ pulses
  const wave = { t: 0, seg: -1, begin: true };
  function resetWave() {
    wave.t = 0;
    wave.seg = -1;
    wave.begin = true;
  }
  const hidePulse = (i) => {
    pSize[i] = 0;
    pCol[i * 3] = pCol[i * 3 + 1] = pCol[i * 3 + 2] = 0;
  };
  const smooth = (u) => u * u * (3 - 2 * u);

  function updatePulses(dt) {
    const L = net.L;
    const slow = playing && speed < 1;
    const back = view.pulses === 'both' && (playing || backOnce);
    const F = L * T_GAP;
    const tLink = F;
    const tBack = F + T_LINK + T_TURN;
    const tApply = back ? tBack + F : F + T_LINK;
    const total = tApply + T_REST;
    if (slow && wave.begin) {
      beginSlowStep();
      wave.begin = false;
    }
    wave.t += dt;
    if (slow && pendingStep && wave.t >= tApply) finishSlowStep();
    if (wave.t >= total) {
      wave.t = 0;
      wave.seg = -1;
      wave.begin = true;
      backOnce = false;
    }
    const t = wave.t;
    // which stage is running: forward gap g (0..L-1), the link (L), backward gap g as 2L - g, or none (-1)
    let seg = -1;
    let u = 0;
    if (t < F) {
      seg = Math.min(L - 1, Math.floor(t / T_GAP));
      u = t / T_GAP - seg;
    } else if (t < tLink + T_LINK) {
      seg = L;
      u = (t - tLink) / T_LINK;
    } else if (back && t >= tBack && t < tBack + F) {
      const k = Math.min(L - 1, Math.floor((t - tBack) / T_GAP));
      seg = L + 1 + k;
      u = (t - tBack) / T_GAP - k;
    }
    if (seg !== wave.seg) {
      // a wave front arrives: flash the layer it has reached
      if (seg >= 0 && seg < L) {
        flash[seg] = 1;
        flashDir[seg] = 1;
      } else if (seg === L) {
        flash[L] = 1;
        flashDir[L] = 1;
      } else if (seg > L) {
        const layer = L - (seg - L - 1);
        flash[layer] = 1;
        flashDir[layer] = -1;
      } else if (wave.seg > L) {
        flash[0] = 1;
        flashDir[0] = -1;
      }
      wave.seg = seg;
    }
    const show = view.pulses !== 'none' && !ctx.reduceMotion;
    const fwdGap = show && seg >= 0 && seg < L ? seg : -1;
    const backGap = show && seg > L ? L - 1 - (seg - L - 1) : -1;
    const k = smooth(Math.min(1, Math.max(0, u)));
    const fade = Math.min(1, u * 6, (1 - u) * 6);
    let maxSig = 1e-9;
    let maxGrad = 1e-12;
    if (fwdGap >= 0) for (let e = 0; e < edgeCount; e++) if (eLayer[e] === fwdGap) maxSig = Math.max(maxSig, Math.abs(net.W[fwdGap][eW[e]] * net.a[fwdGap][nodeIdx[eSrc[e]]]));
    if (backGap >= 0) for (let e = 0; e < edgeCount; e++) maxGrad = Math.max(maxGrad, Math.abs(net.gW[eLayer[e]][eW[e]]));
    for (let e = 0; e < edgeCount; e++) {
      const l = eLayer[e];
      const a = eSrc[e] * 3;
      const b = eDst[e] * 3;
      if (l === fwdGap) {
        // forward: the signal this edge carries for the probed point, weight × source activation
        const s = Math.abs(net.W[l][eW[e]] * net.a[l][nodeIdx[eSrc[e]]]) / maxSig;
        pPos[e * 3] = nodeW[a] + (nodeW[b] - nodeW[a]) * k;
        pPos[e * 3 + 1] = nodeW[a + 1] + (nodeW[b + 1] - nodeW[a + 1]) * k;
        pPos[e * 3 + 2] = nodeW[a + 2] + (nodeW[b + 2] - nodeW[a + 2]) * k + 0.03;
        pSize[e] = (0.09 + 0.26 * s) * fade;
        const br = 0.3 + 0.7 * s;
        pCol[e * 3] = FWD.r * br;
        pCol[e * 3 + 1] = FWD.g * br;
        pCol[e * 3 + 2] = FWD.b * br;
      } else hidePulse(e);
      const o = MAX_EDGES + e;
      if (l === backGap) {
        // backward: the size of this weight's gradient
        const s = Math.sqrt(Math.abs(net.gW[l][eW[e]]) / maxGrad);
        pPos[o * 3] = nodeW[b] + (nodeW[a] - nodeW[b]) * k;
        pPos[o * 3 + 1] = nodeW[b + 1] + (nodeW[a + 1] - nodeW[b + 1]) * k;
        pPos[o * 3 + 2] = nodeW[b + 2] + (nodeW[a + 2] - nodeW[b + 2]) * k + 0.03;
        pSize[o] = (0.09 + 0.28 * s) * fade;
        const br = 0.3 + 0.7 * s;
        pCol[o * 3] = BACK.r * br;
        pCol[o * 3 + 1] = BACK.g * br;
        pCol[o * 3 + 2] = BACK.b * br;
      } else hidePulse(o);
    }
    for (let e = edgeCount; e < MAX_EDGES; e++) {
      hidePulse(e);
      hidePulse(MAX_EDGES + e);
    }
    // the link: output unit -> the probed point on the surface
    const o = (nodeCount - 1) * 3;
    probeHead.getWorldPosition(_v);
    const mx = (nodeW[o] + _v.x) / 2;
    const my = Math.max(nodeW[o + 1], _v.y) + 1.1;
    const mz = (nodeW[o + 2] + _v.z) / 2;
    for (let i = 0; i < LINK_N; i++) {
      const s = i / (LINK_N - 1);
      const w0 = (1 - s) * (1 - s);
      const w1 = 2 * s * (1 - s);
      const w2 = s * s;
      linkPos[i * 3] = w0 * nodeW[o] + w1 * mx + w2 * _v.x;
      linkPos[i * 3 + 1] = w0 * nodeW[o + 1] + w1 * my + w2 * _v.y;
      linkPos[i * 3 + 2] = w0 * nodeW[o + 2] + w1 * mz + w2 * _v.z;
    }
    linkGeo.attributes.position.needsUpdate = true;
    const lp = 2 * MAX_EDGES;
    if (show && seg === L) {
      const w0 = (1 - k) * (1 - k);
      const w1 = 2 * k * (1 - k);
      const w2 = k * k;
      pPos[lp * 3] = w0 * nodeW[o] + w1 * mx + w2 * _v.x;
      pPos[lp * 3 + 1] = w0 * nodeW[o + 1] + w1 * my + w2 * _v.y;
      pPos[lp * 3 + 2] = w0 * nodeW[o + 2] + w1 * mz + w2 * _v.z;
      pSize[lp] = 0.3 * fade;
      FWD.toArray(pCol, lp * 3);
    } else hidePulse(lp);
    pulses.geometry.attributes.position.needsUpdate = true;
    pulses.geometry.attributes.color.needsUpdate = true;
    pulses.geometry.attributes.size.needsUpdate = true;
  }

  // ------------------------------------------------------------------ HUD: controls and read-outs
  const chartSeries = [
    { name: 'training', color: TRAIN_COLOR, data: hist.train },
    { name: 'test', color: TEST_COLOR, data: hist.test },
  ];
  const chart = lineChart({ height: 112, xLabel: 'epoch', yLabel: 'cross-entropy loss', series: chartSeries, yFormat: (v) => String(+v.toFixed(v < 0.1 ? 3 : 2)) });
  const key = (color) => h('i.nn-stat-key', { style: { '--c': color } });
  const statTrain = stat({ label: 'Training accuracy', value: '–' });
  const statTest = stat({ label: 'Test accuracy', value: '–' });
  const statEpoch = stat({ label: 'Epoch', value: '0' });
  statTrain.el.querySelector('.l').prepend(key(TRAIN_COLOR));
  statTest.el.querySelector('.l').prepend(key(TEST_COLOR));

  const playBtn = button('▶ Run', { kind: 'primary', onClick: () => setPlaying(!playing) });
  const dataSeg = segmented(DATASETS.map((d) => ({ value: d.id, label: d.name, title: d.blurb })), cfg.dataset, (id) => change({ dataset: id }), { wrap: true });
  const noiseSlider = slider({ label: 'Noise', min: 0, max: 1, step: 0.05, value: cfg.noise, format: (v) => v.toFixed(2), onInput: (v) => change({ noise: v }) });
  const sizeSlider = slider({ label: 'Training points', min: 20, max: 400, step: 20, value: cfg.nTrain, format: (v) => String(v), onInput: (v) => change({ nTrain: v }) });
  const layersSeg = segmented([0, 1, 2, 3].map((n) => ({ value: n, label: String(n), title: n === 0 ? 'No hidden layer: a single neuron' : `${n} hidden layer${n > 1 ? 's' : ''}` })), cfg.hidden.length, (n) => change({ hidden: unitsMemory.slice(0, n) }));
  const unitSliders = [0, 1, 2].map((i) =>
    slider({
      label: `Layer ${i + 1}`,
      min: 2,
      max: MAX_UNITS,
      step: 1,
      value: unitsMemory[i],
      format: (v) => `${v} units`,
      onInput: (v) => {
        unitsMemory[i] = v;
        change({ hidden: unitsMemory.slice(0, cfg.hidden.length) });
      },
    }),
  );
  const actSeg = segmented([{ value: 'tanh', label: 'tanh' }, { value: 'relu', label: 'ReLU' }], cfg.activation, (v) => change({ activation: v }));
  const optSeg = segmented([{ value: 'sgd', label: 'SGD', title: 'Plain stochastic gradient descent' }, { value: 'adam', label: 'Adam', title: 'Adam: adapts the step for every weight' }], cfg.optimizer, (v) => change({ optimizer: v }));
  const lrSlider = slider({ label: 'Learning rate', min: -3, max: 0.5, step: 0.05, value: Math.log10(cfg.lr), format: (v) => String(+Math.pow(10, v).toPrecision(2)), onInput: (v) => change({ lr: +Math.pow(10, v).toPrecision(2) }) });
  const batchSlider = slider({ label: 'Batch size', min: 0, max: BATCH_STEPS.length - 1, step: 1, value: BATCH_STEPS.indexOf(cfg.batch), format: (i) => (BATCH_STEPS[i] ? String(BATCH_STEPS[i]) : 'all'), onInput: (i) => change({ batch: BATCH_STEPS[i] }) });
  const l2Slider = slider({ label: 'L2 penalty', min: 0, max: L2_STEPS.length - 1, step: 1, value: L2_STEPS.indexOf(cfg.l2), format: (i) => String(L2_STEPS[i]), onInput: (i) => change({ l2: L2_STEPS[i] }) });
  const speedSlider = slider({
    label: 'Speed',
    min: -0.25,
    max: 3.3,
    step: 0.05,
    value: Math.log10(speed),
    format: (v) => (v < 0 ? 'slow motion' : `${fmt.int(+Math.pow(10, v).toPrecision(2))} steps/s`),
    onInput: (v) => setSpeed(v < 0 ? 0.5 : +Math.pow(10, v).toPrecision(2)),
  });
  const foldToggle = toggle({ label: 'First-layer folds', value: view.folds, onChange: (v) => setView({ folds: v }) });
  const stickToggle = toggle({ label: 'All error sticks', value: view.sticks === 'all', onChange: (v) => setView({ sticks: v ? 'all' : 'miss' }) });
  const field = (label, control) => h('div.nn-field', h('span.lab', label), control);
  const unitGrid = h('div.nn-grid3', unitSliders);

  function syncControls() {
    dataSeg.set(cfg.dataset);
    noiseSlider.set(cfg.noise);
    sizeSlider.set(cfg.nTrain);
    layersSeg.set(cfg.hidden.length);
    cfg.hidden.forEach((n, i) => (unitsMemory[i] = n));
    unitSliders.forEach((s, i) => {
      s.set(unitsMemory[i]);
      s.el.hidden = i >= cfg.hidden.length;
    });
    unitGrid.hidden = cfg.hidden.length === 0;
    actSeg.set(cfg.activation);
    optSeg.set(cfg.optimizer);
    lrSlider.set(Math.log10(cfg.lr));
    batchSlider.set(Math.max(0, BATCH_STEPS.indexOf(cfg.batch)));
    l2Slider.set(Math.max(0, L2_STEPS.indexOf(cfg.l2)));
    speedSlider.set(speed < 1 ? -0.25 : Math.log10(speed));
    foldToggle.set(view.folds);
    stickToggle.set(view.sticks === 'all');
    playBtn.textContent = playing ? '❚❚ Pause' : '▶ Run';
  }

  function refreshUI() {
    statTrain.set(pct(metrics.train.acc), `loss ${metrics.train.loss.toFixed(3)}`);
    statTest.set(pct(metrics.test.acc), `loss ${metrics.test.loss.toFixed(3)}`);
    const epoch = net.seen / train.n;
    statEpoch.set(epoch < 10 ? epoch.toFixed(1) : fmt.int(epoch), `${fmt.int(net.steps)} step${net.steps === 1 ? '' : 's'}`);
    chart.update(chartSeries);
    liveFn?.();
    uiTimer = 0;
  }

  function updateLabels() {
    const set = (lab, str) => {
      const el = lab.userData.el;
      if (el.textContent !== str) el.textContent = str;
    };
    set(inLabs[0], `x₁ = ${f2(probe[0])}`);
    set(inLabs[1], `x₂ = ${f2(probe[1])}`);
    set(outLab, `P(red) = ${probeP.toFixed(2)}`);
    set(probeLab, probeLabel < 0 ? `P(red) = ${probeP.toFixed(2)}` : `P(red) = ${probeP.toFixed(2)} · is ${probeLabel ? 'red' : 'blue'}`);
    if (net.L === 1) {
      set(wLabs[0], `w₁ = ${f2(net.W[0][0])}`);
      set(wLabs[1], `w₂ = ${f2(net.W[0][1])}`);
      set(wLabs[2], `b = ${f2(net.b[0][0])}`);
    }
  }

  function setPlaying(v) {
    playing = v;
    playBtn.textContent = playing ? '❚❚ Pause' : '▶ Run';
    if (playing && speed < 1) {
      pendingStep = false;
      resetWave();
    }
  }

  function setSpeed(v) {
    const wasSlow = speed < 1;
    speed = v;
    if (speed < 1 !== wasSlow) {
      pendingStep = false;
      resetWave();
    }
  }

  function setView(patch) {
    Object.assign(view, patch);
    folds.visible = view.folds;
    evaluate(false);
  }

  /** A control changed: rebuild what depends on it. Data and architecture changes start a fresh network. */
  function change(patch) {
    const data = ['dataset', 'noise', 'nTrain', 'dataSeed'].some((k) => k in patch);
    const network = ['hidden', 'activation', 'optimizer', 'netSeed'].some((k) => k in patch);
    Object.assign(cfg, patch);
    if (data) buildData();
    if (data || network) buildNet();
    else {
      net.lr = cfg.lr;
      net.l2 = cfg.l2;
      net.batch = batchSize();
    }
    if (probeLabel >= 0) probeLabel = -1;
    syncControls();
    refreshUI();
  }

  /** Put the whole lab into a known state (used by the tour). */
  function setup(c = {}, o = {}) {
    Object.assign(cfg, DEFAULTS, c);
    cfg.hidden = [...cfg.hidden];
    Object.assign(view, VIEW_DEFAULTS, o.view);
    folds.visible = view.folds;
    speed = o.speed ?? 100;
    buildData();
    buildNet();
    const [px, py] = o.probe ?? [0.4, 0.3];
    if (view.snap) snapProbe(px, py);
    else setProbe(px, py);
    setPlaying(!!o.play);
    syncControls();
    refreshUI();
  }

  // ------------------------------------------------------------------ the tour
  const live = h('pre.nn-live');
  const dataWord = (y) => (y ? 'red' : 'blue');
  const steps = [
    {
      title: 'One neuron',
      text: 'A neuron does two things. It adds up its inputs, each multiplied by a **weight**, plus a bias. Then it **squashes** that sum into a number between 0 and 1.\n\nThe terrain shows that number for every possible input: its height is the neuron’s answer to “how likely is this point red?”. A single neuron can only make one straight ramp. It is already training: watch the ramp turn until the blue points lie low and the red points lie high.',
      body: () => [live, callout('Click anywhere on the surface to choose the point that is fed into the network.', 'key')],
      live: () => {
        if (net.L !== 1) return;
        const w = net.W[0];
        live.textContent = `sum    = w₁·x₁ + w₂·x₂ + b\n       = ${f2(w[0])}·(${f2(probe[0])}) + ${f2(w[1])}·(${f2(probe[1])}) + (${f2(net.b[0][0])})\n       = ${f2(net.z[0][0])}\nP(red) = 1 / (1 + e^(−sum)) = ${probeP.toFixed(2)}`;
      },
      enter: () => setup({ dataset: 'linear', hidden: [], lr: 0.3 }, { speed: 30, play: true, probe: [0.5, 0.45], view: { pulses: 'forward' } }),
    },
    {
      title: 'A layer',
      text: 'One ramp cannot wrap around the red disc in the middle. So put several neurons side by side in a **hidden layer** and let the output neuron mix their answers.\n\nEach hidden neuron still draws one straight fold, shown as a pale line on the surface. Three or four folds, added together, can fence a region in. That is the whole trick of a neural network: simple units, combined.',
      body: () => callout('Line color is the sign of a weight (blue negative, red positive); line thickness is its size.', 'info'),
      enter: () => setup({ dataset: 'circles', hidden: [4], lr: 0.3, netSeed: 1 }, { speed: 120, play: true, probe: [0.1, 0.1], view: { pulses: 'forward', folds: true } }),
    },
    {
      title: 'The forward pass',
      text: 'A prediction is numbers flowing from left to right. The two coordinates of the chosen point enter at the inputs. Every unit computes its weighted sum and its squash and hands the result on; a unit’s brightness is its **activation** for this point. The last number, P(red), is the height of the surface there.\n\nThis network was just created. Its weights are random, so the surface is arbitrary and it gets about half of the points wrong.',
      body: () => callout('Click different places on the surface and watch which units light up.', 'key'),
      enter: () => setup({}, { play: false, probe: [-0.45, 0.25], view: { pulses: 'forward' } }),
    },
    {
      title: 'The loss',
      text: 'How wrong is it? Every training point floats at its true answer: blue at height 0, red at height 1. The stick from a point to the surface is its error.\n\nThe **loss** turns all the sticks into one number. For each point take −ln of the probability the network gave to the correct class, then average. A confident wrong answer costs the most. Always answering 50% scores ln 2 ≈ 0.69, so anything near that is no better than a coin.',
      body: () => live,
      live: () => {
        const pRight = probeLabel === 1 ? probeP : 1 - probeP;
        const own = probeLabel < 0 ? 'click a training point' : `P(red) = ${probeP.toFixed(2)}, it is ${dataWord(probeLabel)}\n                its loss = −ln(${pRight.toFixed(2)}) = ${(-Math.log(Math.max(1e-12, pRight))).toFixed(2)}`;
        live.textContent = `chosen point    ${own}\ntraining loss   ${metrics.train.loss.toFixed(3)}   (mean of ${train.n} points)\ntest loss       ${metrics.test.loss.toFixed(3)}   (${test.n} points it never trains on)`;
      },
      enter: () => setup({}, { play: false, probe: [-0.45, 0.25], view: { pulses: 'forward', sticks: 'all', snap: true } }),
    },
    {
      title: 'Backpropagation',
      text: 'Which weights are to blame? In slow motion, one training point at a time:\n\n**1. Forward** (green). The point runs through the network.\n**2. Compare.** The output is checked against the label.\n**3. Backward** (amber). The error travels back along the same edges. Each weight receives the error at its far end times the activity at its near end. That product is its **gradient**: how much the loss would change if the weight were nudged. Pulse size shows the size of each gradient.\n\nOne backward sweep prices every weight at once, for about the cost of the forward pass. Then every weight moves a little and the surface shifts toward the point.',
      body: () => live,
      live: () => {
        if (probeLabel < 0) {
          live.textContent = 'waiting for the next training point…';
          return;
        }
        let g = 0;
        for (let l = 0; l < net.L; l++) for (const v of net.gW[l]) g = Math.max(g, Math.abs(v));
        live.textContent = `example      x = (${f2(probe[0])}, ${f2(probe[1])}), it is ${dataWord(probeLabel)}\nprediction   P(red) = ${probeP.toFixed(2)}\nerror        P − label = ${f2(probeP - probeLabel)}\nlargest gradient on a weight: ${g.toFixed(3)}`;
      },
      enter: () => setup({ batch: 1 }, { speed: 0.5, play: true, probe: [-0.45, 0.25], view: { pulses: 'both', sticks: 'all', snap: true } }),
    },
    {
      title: 'The gradient step',
      text: 'Now repeat that thousands of times. Each step moves every weight a little against its gradient:\n\n`w ← w − learning rate × gradient`\n\nOne step barely matters. Here each step averages the gradient over a random **mini-batch** of 16 points, and a few thousand steps bend the surface until it fits. The loss chart on the right falls as it happens.',
      body: () => callout('Try the learning rate while it runs: at 0.01 the surface crawls, at 3 it thrashes. **Reset** starts again from new random weights.', 'key'),
      enter: () => setup({}, { speed: 150, play: true, probe: [-0.45, 0.25], view: { pulses: 'both' } }),
    },
    {
      title: 'Capacity and overfitting',
      text: 'How many units? **Too few**, and the surface cannot bend enough: two hidden units cannot close a ring. **Too many**, with few and noisy points, and the network memorizes its training set, quirks included. Training accuracy climbs toward 100% while the **test points** (rings), which it never trains on, get worse. That gap is overfitting.\n\nRemedies: more data, a smaller network, stopping early, or an **L2 penalty** that keeps the weights small.',
      body: () => {
        const run = (id) => {
          if (id === 'small') setup({ dataset: 'circles', hidden: [2], nTrain: 200, optimizer: 'adam', lr: 0.03, netSeed: 3 }, { speed: 300, play: true, view: { folds: true, pulses: 'both' } });
          else setup({ dataset: 'moons', noise: 0.7, nTrain: 60, hidden: [8, 8, 8], optimizer: 'adam', lr: 0.03, l2: id === 'l2' ? 0.01 : 0, netSeed: 1, dataSeed: 1 }, { speed: 300, play: true, view: { pulses: 'both' } });
        };
        const seg = segmented(
          [
            { value: 'small', label: 'Too small' },
            { value: 'large', label: 'Too large' },
            { value: 'l2', label: 'Too large + L2' },
          ],
          'large',
          run,
        );
        return [seg, live];
      },
      live: () => {
        const gap = (metrics.train.acc - metrics.test.acc) * 100;
        live.textContent = `training   ${pct(metrics.train.acc).padStart(6)}   loss ${metrics.train.loss.toFixed(2)}\ntest       ${pct(metrics.test.acc).padStart(6)}   loss ${metrics.test.loss.toFixed(2)}\ngap        ${gap.toFixed(1).replace('-', '−')} points after ${fmt.int(net.seen / train.n)} epochs`;
      },
      enter: () => setup({ dataset: 'moons', noise: 0.7, nTrain: 60, hidden: [8, 8, 8], optimizer: 'adam', lr: 0.03, netSeed: 1, dataSeed: 1 }, { speed: 300, play: true, view: { pulses: 'both' } }),
    },
    {
      title: 'Your turn',
      text: 'Two spiral arms, wound around each other: the hardest shape here. Everything on the right is live, so change it and watch what happens.\n\nSome things to try: remove a hidden layer; switch tanh to ReLU and look at the creases; raise the noise; set the batch size to 1; compare SGD with Adam.',
      body: () => callout(`Nothing is precomputed. The network has at most a few hundred weights and trains in plain JavaScript on this page; the next lab looks at the step rule itself.`, 'info', 'Is this real?'),
      enter: () => setup({ dataset: 'spiral', noise: 0.15, hidden: [8, 8], optimizer: 'adam', lr: 0.03 }, { speed: 400, play: true, probe: [0.3, 0.3], view: { pulses: 'both' } }),
    },
  ];

  buildData();
  buildNet();
  setProbe(probe[0], probe[1]);

  const guide = tour(ctx, steps, {
    lastLabel: 'Next lab →',
    onFinish: () => ctx.router.go('labs/optim'),
    onStep: (i) => {
      liveFn = steps[i].live ?? null;
      liveFn?.();
    },
  });

  const swatch = (cls, color, label) => h('span', h('i' + (cls ? '.' + cls : ''), { style: { '--c': color } }), label);
  ctx.ui.left(
    header({ kicker: 'Lab · Neural network', title: 'Watch backpropagation bend a decision surface', lead: 'A real multilayer perceptron, trained live in your browser. On one side the network; on the other its answer for every point of the input plane.' }),
    guide.body,
    section(
      'Key',
      h(
        'div.nn-key',
        swatch('', CLASS_CSS[0], 'training point, blue class'),
        swatch('', CLASS_CSS[1], 'training point, red class'),
        swatch('ring', CLASS_CSS[1], 'test point (ring)'),
        swatch('miss', CLASS_CSS[0], 'misclassified'),
        swatch('bar', CLASS_CSS[0], 'negative weight'),
        swatch('bar', CLASS_CSS[1], 'positive weight'),
        swatch('pulse', COLOR_FWD, 'forward signal'),
        swatch('pulse', COLOR_BACK, 'gradient, backward'),
      ),
    ),
  );
  ctx.ui.right(
    section('Data', dataSeg, h('div.nn-grid2', noiseSlider, sizeSlider)),
    h('div.btnrow', playBtn, button('Step', { title: 'One mini-batch step', onClick: stepOnce }), button('Reset', { title: 'New random weights', onClick: resetNet })),
    statRow(statTrain, statTest, statEpoch),
    section('Loss', chart),
    section('Network', h('div.nn-row', h('span.lab', 'Hidden layers'), layersSeg), unitGrid, h('div.nn-grid2', field('Activation', actSeg), field('Optimizer', optSeg))),
    section('Training', h('div.nn-grid2', lrSlider, batchSlider, l2Slider, speedSlider)),
    section('Show', h('div.nn-toggles', foldToggle, stickToggle)),
  );
  ctx.ui.dock(guide.dock);

  function stepOnce() {
    setPlaying(false);
    pendingStep = false;
    trainSteps(1);
    resetWave();
    backOnce = true;
    evaluate(true);
    refreshUI();
  }

  function resetNet() {
    cfg.netSeed++;
    buildNet();
    setPlaying(false);
    refreshUI();
  }

  // ------------------------------------------------------------------ picking
  ctx.pick.add(surface, {
    cursor: 'crosshair',
    tip: (hit) => {
      terG.worldToLocal(_v.copy(hit.point));
      q[0] = (_v.x / S) * DOMAIN;
      q[1] = (-_v.z / S) * DOMAIN;
      const p = net.prob1(q);
      return `x₁ ${f2(q[0])}, x₂ ${f2(q[1])}\nP(red) = ${p.toFixed(2)}  ·  click to probe`;
    },
    click: (hit) => {
      terG.worldToLocal(_v.copy(hit.point));
      const x1 = (_v.x / S) * DOMAIN;
      const x2 = (-_v.z / S) * DOMAIN;
      if (view.snap) snapProbe(x1, x2);
      else setProbe(x1, x2);
      updateUnits(0.05);
      updateLabels();
      liveFn?.();
    },
  });
  ctx.pick.add(units, {
    tip: (hit) => {
      const k = hit.instanceId;
      if (k == null || k >= nodeCount) return null;
      const l = nodeLayer[k];
      const i = nodeIdx[k];
      net.forward(probe);
      if (l === 0) return `Input x${i === 0 ? '₁' : '₂'} = ${f2(net.a[0][i])}`;
      const z = net.z[l - 1][i];
      const bias = net.b[l - 1][i];
      if (l === net.L) return `Output unit\nweighted sum ${f2(z)} (bias ${f2(bias)})\nsigmoid → P(red) = ${net.out[0].toFixed(2)}`;
      return `Hidden layer ${l}, unit ${i + 1}\nweighted sum ${f2(z)} (bias ${f2(bias)})\n${cfg.activation === 'relu' ? 'ReLU' : 'tanh'} → activation ${f2(net.a[l][i])}`;
    },
  });
  ctx.key(' ', () => setPlaying(!playing));

  // ------------------------------------------------------------------ camera: frame both parts between the panels
  {
    const pts = [];
    for (const x of [-NET_W / 2 - 1.5, NET_W / 2 + 1.7]) for (const y of [-NET_H / 2 - 0.8, NET_H / 2 + 0.35]) pts.push(netG.localToWorld(new THREE.Vector3(x, y, 0)));
    for (const x of [-S - 0.5, S + 0.7]) for (const z of [-S, S + 0.7]) for (const y of [0, H + 0.75]) pts.push(terG.localToWorld(new THREE.Vector3(x, y, z)));
    const dir = new THREE.Vector3(0.2, 0.58, 1).normalize();
    const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dir).normalize();
    const up = new THREE.Vector3().crossVectors(dir, right);
    const center = new THREE.Box3().setFromPoints(pts).getCenter(new THREE.Vector3());
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const p of pts) {
      const d = p.clone().sub(center);
      u0 = Math.min(u0, d.dot(right));
      u1 = Math.max(u1, d.dot(right));
      v0 = Math.min(v0, d.dot(up));
      v1 = Math.max(v1, d.dot(up));
    }
    center.addScaledVector(right, (u0 + u1) / 2).addScaledVector(up, (v0 + v1) / 2);
    // the engine fits a sphere into the smaller side of the free area; convert the box to that radius
    const w = window.innerWidth;
    const hgt = window.innerHeight;
    const hud = document.getElementById('hud');
    const L = hud?.querySelector('.panel.left');
    const R = hud?.querySelector('.panel.right');
    let freeW = w;
    let freeH = hgt - 140;
    if (w > 860) freeW = w - (L && !L.hidden ? L.offsetWidth + 24 : 0) - (R && !R.hidden ? R.offsetWidth + 24 : 0);
    else {
      const sheet = hud?.dataset.sheet === 'left' ? L : hud?.dataset.sheet === 'right' ? R : null;
      freeH = hgt - 170 - (sheet && !sheet.hidden ? sheet.offsetHeight : 0);
    }
    freeW = Math.max(160, freeW);
    freeH = Math.max(160, freeH);
    const minDim = Math.min(freeW, freeH);
    const fit = Math.max(((u1 - u0) / 2) * (minDim / freeW), ((v1 - v0) / 2) * (minDim / freeH)) * 1.08;
    app.setView({ pos: center.clone().addScaledVector(dir, 20).toArray(), target: center.toArray(), fit, min: 4, max: 70, maxPolar: Math.PI * 0.56 });
  }

  syncControls();
  refreshUI();

  return {
    update(dt) {
      if (playing && speed >= 1) {
        stepAcc += dt * speed;
        let n = Math.min(600, Math.floor(stepAcc));
        if (n > 0) {
          stepAcc -= n;
          // stay inside the frame: stop after about 5 ms of training
          const t0 = performance.now();
          while (n > 0) {
            const chunk = Math.min(n, 8);
            trainSteps(chunk);
            n -= chunk;
            if (performance.now() - t0 > 5) {
              stepAcc = 0;
              break;
            }
          }
        }
      }
      updatePulses(dt);
      sweep(ctx.quality === 'high' ? 3 : 2.2);
      easeSurface(dt);
      evalTimer += dt;
      if (playing && speed >= 1 && evalTimer > 0.1) evaluate(true);
      updateEdges(dt);
      updateUnits(dt);
      probeHead.position.y = probeP * H;
      updateLabels();
      uiTimer += dt;
      if (uiTimer > 0.2 && (playing || pendingStep)) refreshUI();
    },
    onRoute(route) {
      const s = parseInt(route.query.s, 10);
      if (s) guide.go(s - 1, true);
    },
    // handles for the smoke tests (tools/shot.mjs --eval)
    api: {
      cfg,
      metrics,
      get net() {
        return net;
      },
      setup,
      change,
      train(n) {
        trainSteps(n);
        evaluate(true);
        refreshUI();
        return { steps: net.steps, train: { ...metrics.train }, test: { ...metrics.test } };
      },
      probe: (x1, x2) => (view.snap ? snapProbe(x1, x2) : setProbe(x1, x2)),
    },
  };
}
