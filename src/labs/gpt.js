// Inside a GPT: a real ~1M-parameter transformer running in the browser, laid out in 3D.
// Tokens stand in a row at the bottom; each token's vector (the "residual stream") rises through
// four blocks; attention arcs show which earlier tokens a token reads from; the last token's
// vector becomes a probability for every word in the vocabulary, one is sampled and appended.
// Every number comes from src/ml/tinygpt.js.
import * as THREE from 'three';
import './gpt.css';
import { loadTinyGPT, softmax, topK, sample } from '../ml/tinygpt.js';
import { PAL, SERIES, glow, div } from '../core/colors.js';
import { glowMat, glassMat, stdMat, edges, rbox, arc, instanced, setInst, flushInst, gridFloor, line } from '../core/geom.js';
import { textSprite, textPlane } from '../core/labels.js';
import { h, header, section, text, callout, button, segmented, slider, tour, stat, statRow, table, chips, fmt } from '../core/ui.js';
import { heatmap } from '../core/charts.js';
import { damp, ease } from '../core/tween.js';
import { FRONTIER } from '../data/landscape.js';

const DX = 1.34; // spacing between token columns
const COLS = 16;
const ROWS = 8; // a 128-number vector is drawn as a 16 x 8 tile
const TILE_W = 1.12;
const TILE_H = 0.56;
const LEVEL_Y = [1.25, 3.45, 5.65, 7.85, 10.05]; // vector tiles: after embedding, after block 1..4
const BLOCK_Y = [2.35, 4.55, 6.75, 8.95];
const SHELF_Y = 11.35;
const NBARS = 10;
const PRESETS = ['Once upon a time, there was a little girl named', 'Tom and Lily went to the park. They saw a big', 'The little dog was very happy because', 'One day, a cat found a', 'Mom said, "Do not touch the'];

/** Replace a container's children, skipping null / false and unwrapping {el} components. */
function fill(el, ...kids) {
  el.replaceChildren(...kids.flat().filter((k) => k != null && k !== false).map((k) => (k.el instanceof Node ? k.el : k)));
}

/** Text printed flat on a surface (it does not turn toward the camera), replaceable through set(). */
function faceText(parent, x, y, z, opts) {
  let mesh = null;
  let cur = null;
  return {
    set(str, maxWidth) {
      if (str === cur) return;
      cur = str;
      if (mesh) {
        parent.remove(mesh);
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
      mesh = textPlane(str, opts);
      mesh.position.set(x, y, z);
      mesh.geometry.computeBoundingBox();
      const w = mesh.geometry.boundingBox.max.x - mesh.geometry.boundingBox.min.x;
      if (maxWidth && w > maxWidth) mesh.scale.setScalar(maxWidth / w);
      parent.add(mesh);
    },
  };
}

export async function enter(ctx) {
  const { app, root } = ctx;
  ctx.ui.loading('Loading a 1M-parameter GPT');
  const { model, tokenizer, evalSet, bytes } = await loadTinyGPT(ctx.asset);
  const { d, layers: L, heads: H, ctx: CTX, vocab: V } = model.config;

  // how many token columns fit between the panels
  const freeW = window.innerWidth > 860 ? window.innerWidth - 800 : window.innerWidth;
  const MAXT = Math.max(6, Math.min(14, Math.round(freeW / 84)));
  const HALF_W = (MAXT * DX) / 2;
  app.setView({ pos: [1.6, 7.7, 10], target: [-0.9, 6.5, 0], fit: Math.max(HALF_W + 1.6, 8.3), min: 3, max: 70 });
  root.add(gridFloor({ size: 70, y: -0.45, color: '#25477f', opacity: 0.3 }));

  // ------------------------------------------------------------------ state
  const S = {
    prompt: PRESETS[0],
    ids: [],
    promptLen: 0,
    fw: null,
    query: 0, // index into ids of the token whose attention we show
    layer: 0,
    head: -1, // -1 = average of the heads
    temperature: 0.8,
    topK: 40,
    topP: 1,
    stage: 'generate',
    busy: false,
    auto: false,
    autoLeft: 0,
    embedView: 'sum',
    cands: [],
    stepMs: 0,
    lastPick: -1,
  };
  const kv = { cache: null, n: 0 };
  const vis = () => {
    const n = S.ids.length;
    const count = Math.min(n, MAXT);
    return { n, count, offset: n - count };
  };
  const slotX = (i, count) => (i - (count - 1) / 2) * DX;

  // ------------------------------------------------------------------ 3D: token chips
  const chipGeo = rbox(1.16, 0.46, 0.5, 0.09);
  const chipEdgeGeo = new THREE.EdgesGeometry(chipGeo, 30);
  const chips3d = Array.from({ length: MAXT }, (_, i) => {
    const mat = stdMat('#101a30', { emissive: PAL.cyan, ek: 0.25, rough: 0.4 });
    const mesh = new THREE.Mesh(chipGeo, mat);
    const edge = new THREE.LineSegments(chipEdgeGeo, new THREE.LineBasicMaterial({ color: glow(PAL.cyan, 1.5) }));
    mesh.add(edge);
    const label = faceText(mesh, 0, 0.01, 0.262, { size: 0.3, font: 'mono', weight: 650, color: '#eef3ff' });
    const idLabel = faceText(mesh, 0, -0.4, 0.262, { size: 0.17, font: 'mono', weight: 500, color: '#8fa0c0' });
    root.add(mesh);
    ctx.pick.add(mesh, {
      click: () => setQuery(vis().offset + i),
      tip: () => {
        const v = vis();
        const id = S.ids[v.offset + i];
        return id == null ? null : `token ${JSON.stringify(id === tokenizer.eot ? '<start of story>' : tokenizer.vocab[id])}\nid ${id} · position ${v.offset + i}\nclick: show what this token attends to`;
      },
    });
    return { mesh, edge, label, idLabel, mat };
  });
  const earlier = textSprite(' ', { size: 0.22, color: '#8fa0c0', weight: 500 });
  earlier.position.set(0, 0, 0);
  root.add(earlier);

  // ------------------------------------------------------------------ 3D: vector tiles (one InstancedMesh per level)
  const cw = TILE_W / COLS;
  const chh = TILE_H / ROWS;
  const cellGeo = new THREE.PlaneGeometry(cw * 0.9, chh * 0.86);
  const tiles = LEVEL_Y.map((y, lvl) => {
    const mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    const mesh = instanced(cellGeo, mat, MAXT * d);
    mesh.userData.level = lvl;
    root.add(mesh);
    ctx.pick.add(mesh, {
      cursor: 'default',
      tip: (hit) => {
        const v = vis();
        const slot = Math.floor(hit.instanceId / d);
        const dim = hit.instanceId % d;
        const p = v.offset + slot;
        if (slot >= v.count || !S.fw) return null;
        const val = levelData(lvl)[p * d + dim];
        return `${lvl === 0 ? 'input vector' : 'after block ' + lvl} · ${JSON.stringify(tokenizer.show(S.ids[p]))}\ndimension ${dim} of ${d} = ${val.toFixed(3)}`;
      },
    });
    return { mesh, mat, y, reveal: 1 };
  });
  const levelLabels = LEVEL_Y.map((y, lvl) => {
    const s = textSprite(lvl === 0 ? 'input vectors' : `after block ${lvl}`, { size: 0.2, color: '#8fa0c0', weight: 500, anchor: [1, 0.5] });
    s.position.set(-HALF_W - 0.45, y, 0);
    root.add(s);
    return s;
  });

  // ------------------------------------------------------------------ 3D: the four blocks
  const blockGroup = new THREE.Group();
  root.add(blockGroup);
  const slabGeo = rbox(MAXT * DX + 0.5, 0.86, 1.5, 0.12);
  const blocks = BLOCK_Y.map((y, l) => {
    const slab = new THREE.Mesh(slabGeo, glassMat('#1a2c52', 0.16, { emissive: PAL.blue, ek: 0.12 }));
    slab.position.y = y;
    const e = edges(slabGeo, PAL.blue, { k: 1.1, opacity: 0.5 });
    slab.add(e);
    const lab = textSprite(`BLOCK ${l + 1}`, { size: 0.21, font: 'mono', weight: 650, color: '#9fc3ff', anchor: [1, 0.5] });
    lab.position.set(-HALF_W - 0.45, y, 0);
    blockGroup.add(slab, lab);
    ctx.pick.add(slab, { click: () => setLayer(l), tip: `Block ${l + 1}: every token reads from earlier tokens (attention), then thinks on its own (MLP).\nClick to inspect its attention.` });
    return { slab, edge: e, lab };
  });
  // per token, per block: an "attention" bead and an "MLP" bead whose brightness is the size of the update
  const beadGeo = new THREE.SphereGeometry(0.11, 12, 8);
  const cubeGeo = new THREE.BoxGeometry(0.2, 0.2, 0.2);
  const attnBeads = instanced(beadGeo, new THREE.MeshBasicMaterial(), L * MAXT);
  const mlpBeads = instanced(cubeGeo, new THREE.MeshBasicMaterial(), L * MAXT);
  blockGroup.add(attnBeads, mlpBeads);

  // residual stream: one vertical line per token
  const streamGeo = new THREE.BufferGeometry();
  streamGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAXT * 6), 3));
  const streams = new THREE.LineSegments(streamGeo, new THREE.LineBasicMaterial({ color: glow(PAL.steel, 0.9), transparent: true, opacity: 0.35, depthWrite: false }));
  streams.frustumCulled = false;
  root.add(streams);

  // ------------------------------------------------------------------ 3D: attention arcs (tubes, rebuilt when the query changes)
  const arcGroup = new THREE.Group();
  root.add(arcGroup);
  const ARC_UP = new THREE.Vector3(0, 0.8, 0.6).normalize();
  function clearArcs() {
    for (const m of arcGroup.children) {
      m.geometry.dispose();
      m.material.dispose();
    }
    arcGroup.clear();
  }

  // ------------------------------------------------------------------ 3D: next-token probability bars
  const shelf = new THREE.Group();
  shelf.position.y = SHELF_Y;
  root.add(shelf);
  const BAR_DX = Math.min(1.3, (MAXT * DX) / NBARS);
  const shelfPlate = new THREE.Mesh(rbox(NBARS * BAR_DX + 0.5, 0.06, 0.9, 0.03), glassMat('#1a2c52', 0.3, { emissive: PAL.cyan, ek: 0.2 }));
  shelf.add(shelfPlate);
  const shelfLabel = textSprite('probability of each possible next token', { size: 0.21, color: '#8fa0c0', weight: 500 });
  shelfLabel.position.set(0, -0.82, 0.5);
  shelf.add(shelfLabel);
  const barGeo = new THREE.BoxGeometry(0.5, 1, 0.5);
  barGeo.translate(0, 0.5, 0);
  const bars = Array.from({ length: NBARS }, (_, i) => {
    const mat = new THREE.MeshBasicMaterial({ color: glow(PAL.cyan, 1.05) });
    const mesh = new THREE.Mesh(barGeo, mat);
    mesh.position.set((i - (NBARS - 1) / 2) * BAR_DX, 0.04, 0);
    mesh.scale.y = 0.02;
    const name = textSprite(' ', { size: 0.26, font: 'mono', weight: 650, color: '#eef3ff', bg: 'rgba(5,8,16,0.72)' });
    name.position.set(mesh.position.x, -0.36, 0.5);
    const pct = textSprite(' ', { size: 0.21, font: 'mono', weight: 600, color: '#c6d2ea' });
    shelf.add(mesh, name, pct);
    ctx.pick.add(mesh, {
      click: () => S.cands[i] && generate(S.cands[i].id),
      tip: () => (S.cands[i] ? `${JSON.stringify(tokenizer.vocab[S.cands[i].id])}  ${(S.cands[i].p * 100).toFixed(1)}%${S.cands[i].kept ? '' : '  (filtered out by top-k / top-p)'}\nclick to force this token` : null),
    });
    return { mesh, mat, name, pct, target: 0.02 };
  });
  const funnel = line([[0, 0, 0], [0, 1, 0]], PAL.cyan, { k: 1.6, opacity: 0.7 });
  root.add(funnel);
  const ghost = new THREE.Mesh(chipGeo, glowMat('#ffffff', { k: 2.2 }));
  ghost.visible = false;
  root.add(ghost);

  // ------------------------------------------------------------------ painting
  const levelData = (lvl) => {
    if (lvl > 0) return S.fw.acts.resid[lvl];
    return S.embedView === 'token' ? S.fw.acts.tok : S.embedView === 'position' ? S.fw.acts.pos : S.fw.acts.resid[0];
  };
  const col = new THREE.Color();
  const black = new THREE.Color(0, 0, 0);
  let chipY = 0; // the token row slides up next to the block being inspected

  function layout() {
    const v = vis();
    for (let i = 0; i < MAXT; i++) {
      const c = chips3d[i];
      const on = i < v.count;
      c.mesh.visible = on;
      if (!on) continue;
      const id = S.ids[v.offset + i];
      c.mesh.position.x = slotX(i, v.count);
      c.label.set(tokenizer.show(id), 1.04);
      c.idLabel.set('id ' + id);
    }
    earlier.visible = v.offset > 0;
    if (v.offset > 0) {
      earlier.userData.setText(`+${v.offset} earlier`);
      earlier.position.x = slotX(0, v.count) - DX * 0.95;
    }
    for (const t of tiles) {
      for (let i = 0; i < MAXT; i++) {
        const x0 = slotX(i, v.count);
        const s = i < v.count ? 1 : 0;
        for (let k = 0; k < d; k++) setInst(t.mesh, i * d + k, x0 + ((k % COLS) - (COLS - 1) / 2) * cw, t.y + ((ROWS - 1) / 2 - Math.floor(k / COLS)) * chh, 0, s, s, s);
      }
      t.mesh.instanceMatrix.needsUpdate = true;
      t.mesh.computeBoundingSphere();
    }
    const sp = streamGeo.attributes.position.array;
    for (let i = 0; i < MAXT; i++) {
      const x = i < v.count ? slotX(i, v.count) : 0;
      const top = i < v.count ? LEVEL_Y[L] + 0.4 : 0.3;
      sp.set([x, 0.3, -0.05, x, top, -0.05], i * 6);
    }
    streamGeo.attributes.position.needsUpdate = true;
    for (let l = 0; l < L; l++)
      for (let i = 0; i < MAXT; i++) {
        const s = i < v.count && (S.stage !== 'attention' || l === S.layer) ? 1 : 0;
        const x = slotX(i, v.count);
        setInst(attnBeads, l * MAXT + i, x - 0.24, BLOCK_Y[l], 0.78, s, s, s);
        setInst(mlpBeads, l * MAXT + i, x + 0.24, BLOCK_Y[l], 0.78, s, s, s);
      }
    attnBeads.instanceMatrix.needsUpdate = true;
    mlpBeads.instanceMatrix.needsUpdate = true;
  }

  function paint() {
    if (!S.fw) return;
    const v = vis();
    tiles.forEach((t, lvl) => {
      const data = levelData(lvl);
      // robust color scale per level: 2.2 x the root-mean-square of what is on screen
      let ss = 0;
      for (let i = v.offset * d; i < v.n * d; i++) ss += data[i] * data[i];
      const scale = 2.2 * Math.sqrt(ss / Math.max(1, v.count * d)) || 1;
      for (let i = 0; i < v.count; i++) {
        const o = (v.offset + i) * d;
        for (let k = 0; k < d; k++) {
          const u = Math.max(-1, Math.min(1, data[o + k] / scale));
          div(u, col).multiplyScalar(0.55 + Math.abs(u) * 1.25);
          t.mesh.setColorAt(i * d + k, col);
        }
      }
      for (let i = v.count * d; i < MAXT * d; i++) t.mesh.setColorAt(i, black);
      t.mesh.instanceColor.needsUpdate = true;
    });
    // update magnitudes inside the blocks
    let maxA = 1e-6;
    let maxM = 1e-6;
    for (let l = 0; l < L; l++)
      for (let p = v.offset; p < v.n; p++) {
        maxA = Math.max(maxA, S.fw.acts.attnNorm[l][p]);
        maxM = Math.max(maxM, S.fw.acts.mlpNorm[l][p]);
      }
    const ca = glow(PAL.cyan, 1);
    const cm = glow(PAL.amber, 1);
    for (let l = 0; l < L; l++)
      for (let i = 0; i < v.count; i++) {
        attnBeads.setColorAt(l * MAXT + i, col.copy(ca).multiplyScalar(0.25 + 2.4 * (S.fw.acts.attnNorm[l][v.offset + i] / maxA)));
        mlpBeads.setColorAt(l * MAXT + i, col.copy(cm).multiplyScalar(0.25 + 2.4 * (S.fw.acts.mlpNorm[l][v.offset + i] / maxM)));
      }
    flushInst(attnBeads);
    flushInst(mlpBeads);
    chips3d.forEach((c, i) => {
      const p = v.offset + i;
      const isQ = p === S.query;
      const isNew = p >= S.promptLen;
      c.edge.material.color.copy(glow(isQ ? '#ffffff' : isNew ? PAL.amber : PAL.cyan, isQ ? 2.6 : 1.5));
      c.mat.emissive.copy(glow(isNew ? PAL.amber : PAL.cyan, isQ ? 0.7 : 0.25));
    });
  }

  /** attention from the query token to key j in layer l (one head, or the mean over heads) */
  function attnWeight(l, q, j, head = S.head) {
    const A = S.fw.acts.attn[l];
    const n = S.fw.n;
    if (head >= 0) return A[(head * n + q) * n + j];
    let s = 0;
    for (let hd = 0; hd < H; hd++) s += A[(hd * n + q) * n + j];
    return s / H;
  }

  function updateArcs() {
    clearArcs();
    if (!S.fw) return;
    const st = STAGES[S.stage];
    const v = vis();
    const q = Math.min(S.query, v.n - 1);
    const qSlot = q - v.offset;
    if (qSlot < 0 || st.arcs === 'none') return;
    const xFrom = slotX(qSlot, v.count);
    for (let l = 0; l < L; l++) {
      if (st.arcs === 'one' && l !== S.layer) continue;
      const base = glow(S.head >= 0 && st.arcs === 'one' ? SERIES[S.head] : PAL.cyan, 1);
      const y = BLOCK_Y[l];
      for (let s = 0; s <= v.count; s++) {
        // s === v.count stands for all tokens that are off-screen on the left, bundled into one arc
        let w = 0;
        let xTo = 0;
        if (s < v.count) {
          if (v.offset + s > q) continue;
          w = attnWeight(l, q, v.offset + s);
          xTo = slotX(s, v.count);
        } else if (v.offset > 0) {
          for (let j = 0; j < v.offset; j++) w += attnWeight(l, q, j);
          xTo = slotX(0, v.count) - DX * 0.95;
        }
        if (w < 0.02) continue;
        const span = Math.abs(xFrom - xTo);
        let path;
        if (span < 1e-3) {
          // a token attending to itself: a small loop
          const pts = [];
          for (let k = 0; k < 10; k++) {
            const a = (k / 10) * Math.PI * 2;
            pts.push(new THREE.Vector3(xFrom + Math.sin(a) * 0.17, y + 0.12 * (1 - Math.cos(a)), 0.8 + 0.3 * (1 - Math.cos(a))));
          }
          path = new THREE.CatmullRomCurve3(pts, true);
        } else path = arc([xFrom, y, 0.8], [xTo, y, 0.8], 0.35 + span * 0.17, ARC_UP);
        const mesh = new THREE.Mesh(
          new THREE.TubeGeometry(path, span < 1e-3 ? 16 : 22, 0.012 + 0.06 * w, 6, span < 1e-3),
          new THREE.MeshBasicMaterial({ color: base.clone().multiplyScalar(0.7 + 2.2 * w), transparent: true, opacity: Math.min(1, 0.4 + w * 1.2), depthWrite: false }),
        );
        arcGroup.add(mesh);
      }
    }
  }

  function updateBars() {
    const st = STAGES[S.stage];
    shelf.visible = funnel.visible = st.bars;
    const v = vis();
    const xLast = slotX(v.count - 1, v.count);
    const fp = funnel.geometry.attributes.position.array;
    fp.set([xLast, LEVEL_Y[L] + 0.38, 0, 0, SHELF_Y - 0.06, 0]);
    funnel.geometry.attributes.position.needsUpdate = true;
    const maxP = Math.max(0.08, S.cands[0]?.p ?? 1);
    bars.forEach((b, i) => {
      const c = S.cands[i];
      if (!c) {
        b.target = 0.02;
        b.name.userData.setText(' ');
        b.pct.userData.setText(' ');
        return;
      }
      b.target = 0.05 + 2.3 * (c.p / maxP);
      b.name.userData.setText(tokenizer.show(c.id));
      b.name.scale.multiplyScalar(Math.min(1, (BAR_DX * 0.94) / b.name.scale.x));
      b.pct.userData.setText((c.p * 100).toFixed(c.p < 0.1 ? 1 : 0) + '%');
      b.mat.color.copy(c.id === S.lastPick ? glow('#ffffff', 1.9) : c.kept ? glow(PAL.cyan, 1.05) : glow(PAL.slate, 0.9));
    });
  }

  // ------------------------------------------------------------------ model plumbing
  function candidates() {
    const r = sample(S.fw.logits, { temperature: S.temperature, topK: S.topK, topP: S.topP, show: NBARS, rnd: () => 0 });
    S.cands = r.candidates;
  }

  function recompute() {
    S.fw = model.forward(S.ids, { capture: true });
    S.query = Math.min(Math.max(S.query, 0), S.ids.length - 1);
    candidates();
    layout();
    paint();
    updateArcs();
    updateBars();
    renderHud();
    stageBody?.();
  }

  function setPrompt(str) {
    S.prompt = str;
    const enc = tokenizer.encode(str).slice(0, CTX - 24);
    S.ids = [tokenizer.eot, ...enc];
    S.promptLen = S.ids.length;
    S.query = S.ids.length - 1;
    S.lastPick = -1;
    kv.cache = model.newCache();
    for (const id of S.ids) model.step(id, kv.cache);
    recompute();
    tiles.forEach((t, i) => {
      t.reveal = 0;
      ctx.tweens.add({ dur: 0.5, delay: i * 0.13, ease: ease.out, update: (k) => (t.reveal = k) });
    });
  }

  function setQuery(p) {
    S.query = Math.max(0, Math.min(S.ids.length - 1, p));
    paint();
    updateArcs();
    stageBody?.();
  }

  /** Show only what the current stage is about. In the attention stage that is one block and the vectors around it. */
  function focus() {
    const st = STAGES[S.stage];
    const one = S.stage === 'attention';
    tiles.forEach((t, i) => (t.mesh.visible = levelLabels[i].visible = i < st.levels && (!one || i === S.layer || i === S.layer + 1)));
    blockGroup.visible = st.blocks;
    blocks.forEach((b, i) => {
      b.slab.visible = b.lab.visible = !one || i === S.layer;
      b.edge.material.opacity = one ? 1 : 0.5;
    });
    streams.visible = st.levels > 1 && !one;
    chipY = one ? LEVEL_Y[S.layer] - 0.82 : 0;
    if (S.fw) layout();
  }

  function setLayer(l) {
    S.layer = l;
    focus();
    updateArcs();
    stageBody?.();
    if (S.stage === 'attention') app.frame({ target: [-0.9, BLOCK_Y[l] - 0.35, 0], dir: [0.2, 0.42, 1], fit: HALF_W + 1.5 }, 0.9);
  }

  /** Append one token (sampled, or forced when `forceId` is given) with a little flight animation. */
  function generate(forceId) {
    if (S.busy) return;
    if (S.ids.length >= CTX) {
      stopAuto();
      ctx.ui.toast(`Context window full: this model can only see ${CTX} tokens at once.`);
      return;
    }
    const r = sample(S.fw.logits, { temperature: S.temperature, topK: S.topK, topP: S.topP, show: NBARS });
    const id = forceId ?? r.id;
    S.lastPick = id;
    updateBars();
    const from = new THREE.Vector3();
    const barIdx = S.cands.findIndex((c) => c.id === id);
    if (barIdx >= 0 && shelf.visible) from.copy(bars[barIdx].mesh.position).add(shelf.position).setY(SHELF_Y + bars[barIdx].target);
    else from.set(0, SHELF_Y, 0);
    const v = vis();
    const count2 = Math.min(v.count + 1, MAXT);
    const to = new THREE.Vector3(slotX(count2 - 1, count2), 0, 0);
    S.busy = true;
    ghost.visible = true;
    ghost.scale.setScalar(0.4);
    const fast = S.auto && S.fastAuto;
    ctx.tweens
      .add({
        dur: ctx.reduceMotion || fast ? 0.01 : 0.42,
        ease: ease.inOut,
        update: (k) => {
          ghost.position.lerpVectors(from, to, k);
          ghost.position.z = Math.sin(k * Math.PI) * 1.6;
          ghost.scale.setScalar(0.4 + 0.6 * k);
        },
      })
      .promise.then(() => {
        ghost.visible = false;
        S.busy = false;
        if (ctx.dead) return;
        S.ids.push(id);
        S.query = S.ids.length - 1;
        const t0 = performance.now();
        model.step(id, kv.cache);
        const ms = performance.now() - t0;
        S.stepMs = S.stepMs ? S.stepMs * 0.8 + ms * 0.2 : ms;
        recompute();
        if (id === tokenizer.eot) stopAuto();
      });
  }

  function stopAuto() {
    S.auto = false;
    genBtn.textContent = '▶ Generate';
  }
  function toggleAuto() {
    S.auto = !S.auto;
    S.autoLeft = 40;
    genBtn.textContent = S.auto ? '❚❚ Stop' : '▶ Generate';
  }

  // ------------------------------------------------------------------ HUD (right panel: always there)
  const promptInput = h('textarea', { rows: 2, spellcheck: 'false', 'aria-label': 'Prompt' });
  promptInput.value = S.prompt;
  let debounce = 0;
  promptInput.addEventListener('input', () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => !ctx.dead && (stopAuto(), setPrompt(promptInput.value)), 260);
  });
  ctx.onDispose(() => clearTimeout(debounce));
  const genBtn = button('▶ Generate', { kind: 'primary', onClick: toggleAuto });
  const outEl = h('div.gpt-out');
  const candsEl = h('div.gpt-cands');
  const statTokens = stat({ label: 'Context used', value: '0', unit: `/ ${CTX}` });
  const statSpeed = stat({ label: 'Model speed', value: '–', unit: 'tok/s', sub: 'KV-cached step' });

  function renderHud() {
    const gen = tokenizer.decode(S.ids.slice(S.promptLen));
    outEl.replaceChildren(document.createTextNode(tokenizer.decode(S.ids.slice(1, S.promptLen))), h('b', gen));
    outEl.scrollTop = outEl.scrollHeight;
    statTokens.set(String(S.ids.length));
    if (S.stepMs) statSpeed.set(fmt.int(1000 / S.stepMs), `${S.stepMs.toFixed(1)} ms per token, KV-cached`);
    const maxP = Math.max(0.05, S.cands[0]?.p ?? 1);
    candsEl.replaceChildren(
      ...S.cands.map((c) =>
        h(
          'div.gpt-cand' + (c.kept ? '' : '.cut') + (c.id === S.lastPick ? '.pick' : ''),
          { title: c.kept ? 'Click to force this token' : 'Filtered out by top-k / top-p', onclick: () => generate(c.id), style: { cursor: 'pointer' } },
          h('span.tk', tokenizer.show(c.id)),
          h('span', h('div.bar', { style: { width: (c.p / maxP) * 100 + '%' } })),
          h('span.pv', (c.p * 100).toFixed(1) + '%'),
        ),
      ),
    );
  }
  const resample = () => {
    if (!S.fw) return;
    candidates();
    updateBars();
    renderHud();
  };

  ctx.ui.right(
    section(
      'Prompt',
      promptInput,
      chips(PRESETS.map((p, i) => ({ label: 'Story ' + (i + 1), title: p, onClick: () => ((promptInput.value = p), stopAuto(), setPrompt(p)) }))),
    ),
    h('div.btnrow', genBtn, button('+1 token', { onClick: () => (stopAuto(), generate()), title: 'Sample a single token' }), button('Reset', { onClick: () => (stopAuto(), setPrompt(promptInput.value)) })),
    section('Output', outEl),
    statRow(statTokens, statSpeed),
    section('Next token', candsEl),
    slider({ label: 'Temperature', min: 0, max: 2, step: 0.05, value: S.temperature, format: (v) => (v === 0 ? '0 (greedy)' : v.toFixed(2)), onInput: (v) => ((S.temperature = v), resample()) }),
    slider({ label: 'Top-k', min: 1, max: 100, step: 1, value: S.topK, format: (v) => String(v), onInput: (v) => ((S.topK = v), resample()) }),
    slider({ label: 'Top-p (nucleus)', min: 0.05, max: 1, step: 0.01, value: S.topP, format: (v) => v.toFixed(2), onInput: (v) => ((S.topP = v), resample()) }),
  );

  // ------------------------------------------------------------------ stage-specific explanations (left panel)
  let stageBody = null; // re-renders the dynamic part of the current tour step
  const dyn = h('div.tourbody');

  function tokenView() {
    const det = tokenizer.encodeDetailed(S.prompt);
    let pick = det.words.reduce((best, w, i) => (w.steps.length > det.words[best].steps.length ? i : best), 0);
    const merge = h('div.gpt-merge');
    const row = h('div.gpt-toks');
    const show = (i) => {
      pick = i;
      [...row.children].forEach((c, j) => c.classList.toggle('on', j === i));
      const w = det.words[i];
      merge.replaceChildren(
        ...w.steps.map((syms, si) => {
          const prev = si > 0 ? new Set(w.steps[si - 1]) : null;
          return h('div.row', h('span.n', String(si)), syms.map((s) => h('span.sym' + (prev && !prev.has(s) ? '.new' : ''), s.replace(/ /g, '·').replace(/\n/g, '↵'))));
        }),
      );
    };
    det.words.forEach((w, i) => row.appendChild(h('span.gpt-tok', { onclick: () => show(i), title: 'Show how this piece was merged' }, w.text.replace(/ /g, '·').replace(/\n/g, '↵'), h('small', w.ids.length === 1 ? 'id ' + w.ids[0] : w.ids.length + ' tokens'))));
    show(pick);
    const chars = tokenizer.normalize(S.prompt).length;
    fill(
      dyn,
      statRow(stat({ label: 'Characters', value: String(chars) }), stat({ label: 'Tokens', value: String(det.ids.length) }), stat({ label: 'Vocabulary', value: fmt.int(V) })),
      section('Pieces of your prompt (click one)', row),
      section('Its merges, step by step', merge),
      text(`Start from single characters (step 0) and repeatedly glue together the pair that was most frequent in the training text. This tokenizer learned **${fmt.int(tokenizer.merges.length)} merges**; common words end up as one token, rare ones as several pieces. The dot · marks a leading space: it is part of the token.`, 'small'),
    );
  }

  function embedView() {
    const p = S.query;
    const id = S.ids[p];
    // nearest neighbors of this token in embedding space (cosine similarity over the real matrix)
    const W = model.wte;
    let na = 0;
    for (let i = 0; i < d; i++) na += W[id * d + i] ** 2;
    const sims = [];
    for (let t = 1; t < V; t++) {
      if (t === id) continue;
      let dot = 0;
      let nb = 0;
      for (let i = 0; i < d; i++) {
        dot += W[id * d + i] * W[t * d + i];
        nb += W[t * d + i] ** 2;
      }
      sims.push([dot / Math.sqrt(na * nb + 1e-12), t]);
    }
    sims.sort((a, b) => b[0] - a[0]);
    fill(
      dyn,
      section(
        'What the bottom row of tiles shows',
        segmented(
          [
            { value: 'token', label: 'Token vector' },
            { value: 'position', label: 'Position vector' },
            { value: 'sum', label: 'Sum' },
          ],
          S.embedView,
          (v) => {
            S.embedView = v;
            paint();
          },
        ),
      ),
      section(
        `Tokens whose vectors are closest to ${JSON.stringify(tokenizer.show(id))}`,
        table(['Token', 'Cosine similarity'], sims.slice(0, 6).map(([s, t]) => [h('code', tokenizer.show(t)), s.toFixed(2)]), { compact: true }),
        text('Nobody told the model which words are alike. Tokens that appear in the same contexts were pushed to similar vectors by training. Click another token chip in the scene to change the query.', 'small'),
      ),
    );
  }

  let attnChart = null;
  function attentionView() {
    const v = vis();
    const labels = Array.from({ length: v.count }, (_, i) => tokenizer.show(S.ids[v.offset + i]));
    const values = labels.map((_, r) => labels.map((_, c) => (c <= r ? attnWeight(S.layer, v.offset + r, v.offset + c) : NaN)));
    const data = { rows: labels, cols: labels, values, max: 1, cell: Math.max(12, Math.min(24, 300 / v.count)), rowLabel: 'from', colLabel: 'to', format: (x) => (x * 100).toFixed(0) + '%', onHover: (r) => r >= 0 && r + v.offset !== S.query && ((S.query = r + v.offset), paint(), updateArcs()) };
    if (!attnChart || !dyn.contains(attnChart.el)) {
      attnChart = heatmap({ ...data, title: 'Who reads from whom (rows read from columns)' });
      fill(
      dyn,
        section('Block', segmented(Array.from({ length: L }, (_, l) => ({ value: l, label: String(l + 1) })), S.layer, setLayer)),
        section(
          'Head',
          segmented([{ value: -1, label: 'average' }, ...Array.from({ length: H }, (_, k) => ({ value: k, label: String(k + 1) }))], S.head, (k) => {
            S.head = k;
            updateArcs();
            attentionView();
          }),
        ),
        attnChart.el,
        text('Each row is one token deciding where to look. The upper-right half is empty on purpose: a token may only attend to itself and to earlier tokens (the **causal mask**), otherwise it could read the answer it is supposed to predict.\n\nHover a row, or click a token chip in the scene, to draw its arcs. Different heads specialize: look for one that mostly reads the previous token.', 'small'),
      );
    } else attnChart.update(data);
  }

  function lensView() {
    const p = S.query;
    const rows = [];
    for (let lvl = 0; lvl <= L; lvl++) {
      const probs = softmax(model.unembed(S.fw.acts.resid[lvl], p * d));
      const top = topK(probs, 3);
      rows.push([lvl === 0 ? 'input' : 'after block ' + lvl, h('span', top.map((t, i) => h('span', i ? ', ' : '', i === 0 ? h('b', tokenizer.show(t.id)) : tokenizer.show(t.id), ` ${(t.p * 100).toFixed(0)}%`)))]);
    }
    const next = S.ids[p + 1];
    fill(
      dyn,
      section(`Reading the stream early, at ${JSON.stringify(tokenizer.show(S.ids[p]))}`, h('div.gpt-lens', table(['Depth', 'Top guesses for the next token'], rows, { compact: true }))),
      next != null && text(`The token that actually follows here is **${tokenizer.show(next)}**.`, 'small'),
      text('This is the **logit lens**, a basic interpretability tool: take the vector at an intermediate depth and push it through the output layer as if the model stopped there. The guess usually starts vague and sharpens block by block. Click any token chip to move the lens.', 'small'),
    );
  }

  function scaleView() {
    const mine = model.paramCount;
    const rows = [
      ['This model', mine, '2026, trained for this page'],
      ['GPT-2 small', 124e6, '2019'],
      ['GPT-3', 175e9, '2020'],
      ...FRONTIER.filter((f) => /T-parameter|T MoE|parameters/.test(f.note) && /(\d\.\d+)T/.test(f.note)).map((f) => [f.model, parseFloat(f.note.match(/(\d\.\d+)T/)[1]) * 1e12, f.date + ', open weights']),
    ];
    fill(
      dyn,
      table(
        ['Model', 'Parameters', 'Times larger'],
        rows.map(([name, p, when]) => [h('span', name, h('small', { style: { display: 'block', color: 'var(--ink-3)' } }, when)), fmt.compact(p), p === mine ? '1×' : fmt.compact(p / mine) + '×']),
        { compact: true },
      ),
      text(`The weights you are running take **${fmt.bytes(bytes)}** as 16-bit floats and were trained for ${fmt.int(model.meta.trainSteps || 0)} steps on a laptop-class CPU, on short children’s stories (TinyStories). That is why it can continue a fairy tale and knows nothing else.\n\nEvery step you just walked through (tokens, vectors, attention, MLP, sampling) is the same in the largest models. What changes is the count: more blocks, wider vectors, a longer context, and about a million times more parameters and data.`),
      h('div.btnrow', button('Measure the KV cache →', { onClick: () => ctx.router.go('perf/kvcache') }), button('Quantize this model →', { onClick: () => ctx.router.go('perf/quant') })),
    );
  }

  // which parts of the scene each stage shows, and where the camera goes
  const STAGES = {
    tokens: { levels: 0, blocks: false, arcs: 'none', bars: false, cam: { target: [-0.4, 0.2, 0], dir: [0, 0.22, 1], fit: HALF_W + 1 } },
    embed: { levels: 1, blocks: false, arcs: 'none', bars: false, cam: { target: [-0.9, 0.75, 0], dir: [0, 0.25, 1], fit: HALF_W + 1.4 } },
    attention: { levels: L + 1, blocks: true, arcs: 'one', bars: false, cam: null },
    stream: { levels: L + 1, blocks: true, arcs: 'all', bars: false, cam: { target: [-0.9, 5.6, 0], dir: [0.3, 0.14, 1], fit: Math.max(HALF_W + 1.6, 6.8) } },
    predict: { levels: L + 1, blocks: true, arcs: 'all', bars: true, cam: { target: [-0.3, 10.9, 0], dir: [0.12, 0.14, 1], fit: Math.max(HALF_W + 0.8, 4.8) } },
    generate: { levels: L + 1, blocks: true, arcs: 'all', bars: true, cam: { target: [-0.9, 6.5, 0], dir: [0.25, 0.12, 1], fit: Math.max(HALF_W + 1.6, 8.3) } },
    scale: { levels: L + 1, blocks: true, arcs: 'all', bars: true, cam: { target: [-0.9, 6.5, 0], dir: [-0.5, 0.2, 1], fit: Math.max(HALF_W + 1.6, 8.6) } },
  };

  function setStage(name, body) {
    S.stage = name;
    stageBody = body ?? null;
    const st = STAGES[name];
    focus();
    if (name !== 'generate') stopAuto();
    if (S.fw) {
      updateArcs();
      updateBars();
      paint();
    }
    fill(dyn);
    stageBody?.();
    if (st.cam) app.frame(st.cam, 1.1);
    else setLayer(S.layer);
  }

  const steps = [
    {
      title: 'Text becomes tokens',
      text: 'A language model never sees letters or words. Your text is first cut into **tokens**, pieces from a fixed vocabulary, and each token is replaced by its number. Type your own prompt on the right and watch the row change.',
      enter: () => setStage('tokens', tokenView),
    },
    {
      title: 'Tokens become vectors',
      text: `Each token id looks up a row of **${d} numbers** in the embedding table: its vector. A second vector says where in the sequence it stands. Their sum is what enters the network. Each tile in the scene is one such vector: red cells are positive, blue negative.`,
      enter: () => setStage('embed', embedView),
    },
    {
      title: 'Attention: tokens read each other',
      text: 'Inside every block, each token asks a question (its **query**) and every earlier token offers a **key**. Where they match, information flows: the arcs. This is how “she” finds out who “she” is.',
      enter: () => setStage('attention', attentionView),
    },
    {
      title: 'The residual stream',
      text: `Nothing is replaced on the way up. Each block **adds** two small updates to every token’s vector: what attention gathered (cyan beads) and what the block’s MLP computed from it (amber cubes; brighter means a bigger update). After ${L} blocks the last vector has to contain the answer.`,
      enter: () => setStage('stream', lensView),
    },
    {
      title: 'A probability for every token',
      text: `The top vector of the last token is compared with all **${fmt.int(V)}** token vectors in the vocabulary. The result is one score per token, turned into probabilities. The model’s whole output is this distribution, nothing more.\n\nNow reshape it: **temperature** flattens or sharpens it, **top-k** and **top-p** cut off the unlikely tail (the gray bars).`,
      enter: () => setStage('predict'),
    },
    {
      title: 'Sample, append, repeat',
      text: 'One token is drawn from the distribution, appended to the text, and the whole process runs again on the longer sequence. That loop is all “generation” is. Press **Generate**, or click a bar to force a different token and watch the story fork.',
      body: () => callout('Generated tokens are amber. The model reads its own output exactly like your prompt, which is why one odd token can derail everything after it.', 'key'),
      enter: () => setStage('generate'),
    },
    {
      title: 'Now scale it up',
      text: 'You have seen the complete algorithm.',
      enter: () => setStage('scale', scaleView),
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'Next lab →', onFinish: () => ctx.router.go('labs/neural') });
  ctx.ui.left(
    header({ kicker: 'Lab · Inside a GPT', title: 'A real language model, taken apart', lead: `**${fmt.int(model.paramCount)} parameters**, ${L} blocks, ${H} attention heads each, trained on children’s stories. It is running on this page right now.` }),
    guide.body,
    dyn,
  );
  ctx.ui.dock(guide.dock);
  ctx.key(' ', toggleAuto);

  setPrompt(S.prompt);
  // evalSet is only used by the performance experiments; keep the reference for the stat below
  void evalSet;

  let autoTimer = 0;
  return {
    update(dt, t) {
      for (const b of bars) {
        b.mesh.scale.y = damp(b.mesh.scale.y, b.target, 0.09, dt);
        b.pct.position.set(b.mesh.position.x, b.mesh.scale.y + 0.22, 0);
      }
      tiles.forEach((tl) => tl.mat.color.setScalar(tl.reveal));
      for (const c of chips3d) c.mesh.position.y = damp(c.mesh.position.y, chipY, 0.12, dt);
      earlier.position.y = chips3d[0].mesh.position.y;
      if (S.auto && !S.busy) {
        autoTimer += dt;
        if (autoTimer > 0.16) {
          autoTimer = 0;
          if (S.autoLeft-- > 0) generate();
          else stopAuto();
        }
      }
      if (S.stage === 'scale' && !ctx.reduceMotion) root.rotation.y = Math.sin(t * 0.25) * 0.25;
      else root.rotation.y = damp(root.rotation.y, 0, 0.2, dt);
    },
  };
}
