// Embedding Space: meaning as geometry. Three scenes share the lab:
//  1. a map of 400 real sentence embeddings (all-MiniLM-L6-v2) with exact nearest neighbors,
//  2. a race between vector indexes (flat, IVF, HNSW, LSH) on 12,000 synthetic vectors,
//  3. a retrieval bench: BM25 vs dense vs hybrid on 12 questions over 55 passages.
// The algorithms are in ../ml/ann.js and ../ml/retrieval.js.
import * as THREE from 'three';
import './embed.css';
import { makeDataset, makeQueries, FlatIndex, IVFIndex, HNSWIndex, LSHIndex, TopK, Trace, groundTruth, benchmark, pca3, project3 } from '../ml/ann.js';
import { BM25, decodeInt8, unitRows, denseSearch, rrf, firstRelevantRank, recallAt } from '../ml/retrieval.js';
import { PAL, SERIES, glow } from '../core/colors.js';
import { instanced, flushInst, gridFloor, points, glowMat, rbox, edges } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, segmented, slider, tour, table, chips, fmt } from '../core/ui.js';
import { barChart } from '../core/charts.js';

const K = 10; // neighbors asked for in the index race
const NQ = 60; // queries per benchmark
const METHODS = [
  { id: 'bm25', name: 'Keywords (BM25)', color: SERIES[0] },
  { id: 'dense', name: 'Vectors (dense)', color: SERIES[1] },
  { id: 'hybrid', name: 'Hybrid (rank fusion)', color: SERIES[2] },
];

export async function enter(ctx) {
  const { app, root } = ctx;
  ctx.ui.loading('Loading embeddings');
  const E = await fetch(ctx.asset('data/embeddings.json')).then((r) => r.json());
  const dim = E.dim;
  app.setView({ pos: [0, 6, 22], target: [0, 0, 0], fit: 11.5, min: 3, max: 80 });
  root.add(gridFloor({ size: 90, y: -9.5, color: '#3a2f66', opacity: 0.22 }));

  // =================================================================== scene 1: the semantic map
  const mapG = new THREE.Group();
  root.add(mapG);
  const W = E.words.text.length;
  const wordVec = unitRows(decodeInt8(E.words.vec), dim);
  const catColor = E.categories.map((_, i) => '#' + new THREE.Color().setHSL(i / E.categories.length, 0.72, 0.62).getHexString());
  let maxAbs = 0;
  for (const v of E.words.xyz) maxAbs = Math.max(maxAbs, Math.abs(v));
  const wpos = Array.from({ length: W }, (_, i) => new THREE.Vector3(E.words.xyz[i * 3], E.words.xyz[i * 3 + 1], E.words.xyz[i * 3 + 2]).multiplyScalar(9 / maxAbs));
  const dots = instanced(new THREE.SphereGeometry(0.13, 10, 8), new THREE.MeshBasicMaterial(), W);
  mapG.add(dots);
  const baseLabels = [];
  for (let i = 0; i < W; i += 5) {
    const s = textSprite(E.words.text[i], { size: 0.3, color: '#c9d4ea', weight: 500, anchor: [0.5, 0] });
    s.position.copy(wpos[i]).y += 0.2;
    s.userData.i = i;
    mapG.add(s);
    baseLabels.push(s);
  }
  const hotLabels = Array.from({ length: 10 }, () => {
    const s = textSprite(' ', { size: 0.3, color: '#ffffff', weight: 700, anchor: [0.5, 0], bg: 'rgba(5,8,16,0.7)', onTop: true });
    s.visible = false;
    mapG.add(s);
    return s;
  });
  const nbGeo = new THREE.BufferGeometry();
  nbGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9 * 6), 3));
  const nbLines = new THREE.LineSegments(nbGeo, new THREE.LineBasicMaterial({ color: glow('#ffffff', 2) }));
  nbLines.frustumCulled = false;
  mapG.add(nbLines);
  const M = { sel: -1, hover: -1, offCats: new Set(), neighbors: [] };
  const m4 = new THREE.Matrix4();
  const col = new THREE.Color();
  function paintMap() {
    const near = new Set(M.neighbors.map((n) => n.doc));
    for (let i = 0; i < W; i++) {
      const off = M.offCats.has(E.words.cat[i]);
      const s = i === M.sel ? 2.4 : near.has(i) ? 1.7 : i === M.hover ? 1.6 : off ? 0.5 : 1;
      m4.makeScale(s, s, s).setPosition(wpos[i]);
      dots.setMatrixAt(i, m4);
      dots.setColorAt(i, col.copy(glow(i === M.sel ? '#ffffff' : catColor[E.words.cat[i]], off ? 0.25 : near.has(i) || i === M.sel ? 2.6 : M.sel >= 0 ? 0.7 : 1.5)));
    }
    flushInst(dots);
    dots.computeBoundingSphere();
    for (const s of baseLabels) s.visible = M.sel < 0 && !M.offCats.has(E.words.cat[s.userData.i]);
    const lp = nbGeo.attributes.position.array;
    hotLabels.forEach((s) => (s.visible = false));
    if (M.sel >= 0) {
      const all = [{ doc: M.sel, score: 1 }, ...M.neighbors];
      all.forEach((n, k) => {
        if (k >= hotLabels.length) return;
        const s = hotLabels[k];
        s.userData.setText(E.words.text[n.doc]);
        s.position.copy(wpos[n.doc]).y += 0.26 + (k % 3) * 0.3;
        s.visible = true;
        if (k > 0) {
          wpos[M.sel].toArray(lp, (k - 1) * 6);
          wpos[n.doc].toArray(lp, (k - 1) * 6 + 3);
        }
      });
    }
    nbGeo.setDrawRange(0, M.sel >= 0 ? M.neighbors.length * 2 : 0);
    nbGeo.attributes.position.needsUpdate = true;
  }
  function selectWord(i) {
    M.sel = i;
    M.neighbors = i >= 0 ? denseSearch(wordVec, i, wordVec, dim, { skip: i, top: 8 }) : [];
    paintMap();
    renderMapPanel();
    if (i >= 0 && stage === 'map') app.frame({ target: wpos[i].toArray(), dir: [0, 0.25, 1], fit: 4.6 }, 1);
  }
  ctx.pick.add(dots, {
    hover: (hit) => ((M.hover = hit.instanceId), paintMap()),
    out: () => ((M.hover = -1), paintMap()),
    click: (hit) => selectWord(hit.instanceId),
    tip: (hit) => `${E.words.text[hit.instanceId]}\n${E.categories[E.words.cat[hit.instanceId]]} · click for its nearest neighbors`,
  });

  // =================================================================== scene 2: the index race
  const raceG = new THREE.Group();
  root.add(raceG);
  const ds = makeDataset({ n: 12000, dim: 32, clusters: 40, seed: 5 });
  const pca = pca3(ds);
  const rpos = new Float32Array(ds.n * 3);
  for (let i = 0; i < ds.n; i++) project3(pca, ds.data, i * ds.dim, ds.dim, rpos, i * 3);
  let rmax = 0;
  for (const v of rpos) rmax = Math.max(rmax, Math.abs(v));
  for (let i = 0; i < rpos.length; i++) rpos[i] *= 9.5 / rmax;
  const rcol = new Float32Array(ds.n * 3);
  const rsize = new Float32Array(ds.n).fill(0.11);
  const baseCol = (i, out) => out.setHSL((ds.label[i] * 0.618) % 1, 0.55, 0.5).multiplyScalar(0.55);
  const cloud = points(rpos, rcol, rsize, { additive: true });
  raceG.add(cloud);
  const qMark = new THREE.Mesh(new THREE.OctahedronGeometry(0.35), glowMat('#ffd166', { k: 2.8 }));
  raceG.add(qMark);
  const queries = makeQueries(ds, NQ);
  const truth = groundTruth(ds, queries, NQ, K);
  const INDEXES = [
    { id: 'flat', name: 'Flat (exact)', color: SERIES[0], index: new FlatIndex(ds), opts: () => ({}), ready: true, note: 'compares the query with every vector' },
    { id: 'ivf', name: 'IVF', color: SERIES[1], index: new IVFIndex(ds, { cells: 96 }), opts: () => ({ nprobe: R.nprobe }), note: 'k-means cells; search only the nearest few' },
    { id: 'hnsw', name: 'HNSW graph', color: SERIES[2], index: new HNSWIndex(ds, { M: 8 }), opts: () => ({ ef: R.ef }), note: 'greedy walk through a layered neighbor graph' },
    { id: 'lsh', name: 'LSH', color: SERIES[3], index: new LSHIndex(ds, { tables: 8 }), opts: () => ({ bits: R.bits }), note: 'random hyperplanes hash similar vectors together' },
  ];
  const R = { nprobe: 4, ef: 32, bits: 10, sel: 'ivf', query: 0, trace: null, shown: 0, results: {}, builders: INDEXES.filter((x) => !x.ready).map((x) => ({ x, gen: x.index.build() })), done: null };
  function resetCloud() {
    for (let i = 0; i < ds.n; i++) {
      baseCol(i, col);
      rcol[i * 3] = col.r;
      rcol[i * 3 + 1] = col.g;
      rcol[i * 3 + 2] = col.b;
      rsize[i] = 0.11;
    }
    cloud.geometry.attributes.color.needsUpdate = true;
    cloud.geometry.attributes.size.needsUpdate = true;
  }
  function runQuery(next = false) {
    if (next) R.query = (R.query + 1) % NQ;
    const x = INDEXES.find((i) => i.id === R.sel);
    resetCloud();
    project3(pca, queries, R.query * ds.dim, ds.dim, qMark.position, 0);
    const p = [0, 0, 0];
    project3(pca, queries, R.query * ds.dim, ds.dim, p, 0);
    qMark.position.set(p[0], p[1], p[2]).multiplyScalar(9.5 / rmax);
    if (!x.ready) {
      R.trace = null;
      return;
    }
    const top = new TopK(K);
    const trace = new Trace();
    const comps = x.index.search(queries, R.query * ds.dim, top, x.opts(), trace);
    R.trace = { trace, top, comps, x };
    R.shown = 0;
    renderRacePanel();
  }
  function bench(only) {
    for (const x of INDEXES) if (x.ready && (!only || only === x.id)) R.results[x.id] = benchmark(x.index, queries, NQ, truth, K, x.opts());
    renderRacePanel();
  }

  // =================================================================== scene 3: the retrieval bench
  const benchG = new THREE.Group();
  root.add(benchG);
  const P = E.passages.items;
  const Q = E.questions.items;
  const passVec = unitRows(decodeInt8(E.passages.vec), dim);
  const quesVec = unitRows(decodeInt8(E.questions.vec), dim);
  const bm25 = new BM25(P.map((p) => p.title + '. ' + p.text));
  const idOf = Object.fromEntries(P.map((p, i) => [p.id, i]));
  const relevant = Q.map((q) => q.relevant.map((id) => idOf[id]).filter((v) => v != null));
  const rankings = { bm25: [], dense: [], hybrid: [] };
  Q.forEach((q, qi) => {
    const a = bm25.search(q.text);
    const b = denseSearch(quesVec, qi, passVec, dim);
    rankings.bm25.push(a);
    rankings.dense.push(b);
    rankings.hybrid.push(rrf([a.slice(0, 20), b.slice(0, 20)]));
  });
  const metrics = Object.fromEntries(
    METHODS.map((m) => {
      const lists = rankings[m.id];
      const mean = (f) => lists.reduce((s, l, qi) => s + f(l, relevant[qi]), 0) / lists.length;
      return [m.id, { r1: mean((l, rel) => recallAt(l, rel, 1)), r3: mean((l, rel) => recallAt(l, rel, 3)), mrr: mean((l, rel) => { const r = firstRelevantRank(l, rel); return r ? 1 / r : 0; }) }];
    }),
  );
  const B = { q: 0 };
  const cardGeo = rbox(5.2, 0.86, 0.3, 0.08);
  const cards = METHODS.map((m, c) =>
    Array.from({ length: 5 }, (_, r) => {
      const mesh = new THREE.Mesh(cardGeo, new THREE.MeshStandardMaterial({ color: '#141f38', roughness: 0.5, emissive: new THREE.Color(0, 0, 0) }));
      mesh.position.set((c - 1) * 6.4, 2.6 - r * 1.25, 0);
      const outline = edges(cardGeo, m.color, { k: 1.4, opacity: 0.7 });
      mesh.add(outline);
      const lab = textSprite(' ', { size: 0.34, color: '#eef3ff', weight: 600 });
      lab.position.set(0, 0, 0.25);
      mesh.add(lab);
      benchG.add(mesh);
      return { mesh, lab, outline };
    }),
  );
  METHODS.forEach((m, c) => {
    const t = textSprite(m.name, { size: 0.5, font: 'display', weight: 700, color: '#eef3ff' });
    t.position.set((c - 1) * 6.4, 3.9, 0);
    benchG.add(t);
  });
  const qLabel = textSprite(' ', { size: 0.46, color: '#ffe9a8', weight: 600, maxChars: 60 });
  qLabel.position.set(0, 5.3, 0);
  benchG.add(qLabel);
  const linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(10 * 6), 3));
  const rankLinks = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: glow('#8fa3c8', 1.2), transparent: true, opacity: 0.7 }));
  benchG.add(rankLinks);
  function paintBench() {
    const q = Q[B.q];
    qLabel.userData.setText('“' + q.text + '”');
    const rel = new Set(relevant[B.q]);
    const tops = METHODS.map((m) => rankings[m.id][B.q].slice(0, 5));
    tops.forEach((list, c) =>
      cards[c].forEach((card, r) => {
        const d = list[r];
        card.mesh.visible = !!d;
        if (!d) return;
        const hit = rel.has(d.doc);
        card.lab.userData.setText(`${r + 1}. ${P[d.doc].title}`);
        card.lab.scale.multiplyScalar(Math.min(1, 4.9 / card.lab.scale.x));
        card.mesh.material.emissive.copy(hit ? glow('#0ca30c', 0.9) : new THREE.Color(0, 0, 0));
        card.outline.material.color.copy(glow(hit ? '#7dffa0' : METHODS[c].color, hit ? 2.4 : 1.2));
      }),
    );
    // connect the same passage across neighboring columns: how its rank moves
    const lp = linkGeo.attributes.position.array;
    let n = 0;
    for (let c = 0; c < 2; c++)
      tops[c].forEach((d, r) => {
        const r2 = tops[c + 1].findIndex((e) => e.doc === d.doc);
        if (r2 < 0 || n >= 10) return;
        lp.set([(c - 1) * 6.4 + 2.6, 2.6 - r * 1.25, 0, c * 6.4 - 2.6, 2.6 - r2 * 1.25, 0], n * 6);
        n++;
      });
    linkGeo.setDrawRange(0, n * 2);
    linkGeo.attributes.position.needsUpdate = true;
  }

  // =================================================================== panels
  let stage = 'map';
  function renderMapPanel() {
    if (stage !== 'map') return;
    const catChips = chips(
      E.categories.map((c, i) => ({
        label: c,
        color: catColor[i],
        onClick: () => {
          if (M.offCats.has(i)) M.offCats.delete(i);
          else M.offCats.add(i);
          paintMap();
          renderMapPanel();
        },
      })),
    );
    [...catChips.children].forEach((el, i) => (el.style.opacity = M.offCats.has(i) ? 0.35 : 1));
    const search = h('input', { type: 'text', placeholder: 'Find a word, e.g. violin', 'aria-label': 'Find a word' });
    search.addEventListener('change', () => {
      const i = E.words.text.indexOf(search.value.trim().toLowerCase());
      if (i >= 0) selectWord(i);
      else ctx.ui.toast('That word is not on the map. Try one you can see.');
    });
    ctx.ui.right(
      section('Find a word', search),
      M.sel >= 0
        ? section(
            `Nearest to “${E.words.text[M.sel]}” in ${dim} dimensions`,
            table(['Word', 'Category', 'Cosine'], M.neighbors.map((n) => [h('a', { href: '#', onclick: (e) => (e.preventDefault(), selectWord(n.doc)) }, E.words.text[n.doc]), E.categories[E.words.cat[n.doc]], n.score.toFixed(2)]), { compact: true }),
            text('These neighbors are computed on the full vectors, not on the 3D picture. If a neighbor looks far away on the map, that is the projection: 384 dimensions cannot be flattened into three without distortion.', 'small'),
            h('div.btnrow', button('Clear selection', { onClick: () => (selectWord(-1), app.home()) })),
          )
        : text('Click any dot to see its nearest neighbors.', 'small'),
      section('Categories (click to dim)', catChips),
      text(`Model: ${E.model} · ${W} words · layout: ${E.layout}.`, 'small'),
    );
  }
  function renderRacePanel() {
    if (stage !== 'race') return;
    const building = INDEXES.filter((x) => !x.ready);
    const rows = INDEXES.map((x) => {
      const r = R.results[x.id];
      return [h('span', h('i', { style: { display: 'inline-block', width: '9px', height: '9px', borderRadius: '3px', background: x.color, marginRight: '7px' } }), x.name), r ? (r.recall * 100).toFixed(1) + '%' : '…', r ? fmt.int(r.comps) : '…', r ? (r.ms < 0.1 ? (r.ms * 1000).toFixed(0) + ' µs' : r.ms.toFixed(2) + ' ms') : '…'];
    });
    const cur = INDEXES.find((x) => x.id === R.sel);
    ctx.ui.right(
      section('Index to watch', segmented(INDEXES.map((x) => ({ value: x.id, label: x.name.split(' ')[0] })), R.sel, (v) => ((R.sel = v), runQuery()))),
      text(`**${cur.name}**: ${cur.note}.` + (R.trace && R.trace.x === cur ? ` This query touched **${fmt.int(R.trace.comps)}** of ${fmt.int(ds.n)} vectors.` : cur.ready ? '' : ' Still building…'), 'small'),
      h('div.btnrow', button('▶ New query', { kind: 'primary', onClick: () => runQuery(true) }), button('Replay', { onClick: () => runQuery() })),
      slider({ label: 'IVF: cells probed (nprobe)', min: 1, max: 32, step: 1, value: R.nprobe, format: String, onInput: (v) => (R.nprobe = v), onChange: () => (bench('ivf'), R.sel === 'ivf' && runQuery()) }),
      slider({ label: 'HNSW: candidates kept (ef)', min: 10, max: 200, step: 2, value: R.ef, format: String, onInput: (v) => (R.ef = v), onChange: () => (bench('hnsw'), R.sel === 'hnsw' && runQuery()) }),
      slider({ label: 'LSH: hash bits', min: 4, max: 18, step: 1, value: R.bits, format: String, onInput: (v) => (R.bits = v), onChange: () => (bench('lsh'), R.sel === 'lsh' && runQuery()) }),
      section(`Measured over ${NQ} queries, top ${K}`, table(['Index', 'Recall', 'Distances', 'Time / query'], rows, { compact: true })),
      barChart({ title: 'Distance computations per query (fewer is faster)', rows: INDEXES.filter((x) => R.results[x.id]).map((x) => ({ label: x.name.split(' ')[0], value: R.results[x.id].comps, color: x.color })), format: fmt.int }),
      building.length ? h('div.hint', 'building: ' + building.map((x) => x.name).join(', ') + ' …') : text(`${fmt.int(ds.n)} synthetic vectors in ${ds.dim} dimensions, drawn through their three main directions. Recall = share of the true ${K} nearest neighbors that the index returned.`, 'small'),
    );
  }
  function renderBenchPanel() {
    if (stage !== 'bench') return;
    const qList = h(
      'div.emb-q',
      Q.map((q, i) => {
        const b = h('button' + (i === B.q ? '.on' : ''), { type: 'button' }, h('small', q.kind), q.text);
        b.addEventListener('click', () => ((B.q = i), paintBench(), renderBenchPanel()));
        return b;
      }),
    );
    const rel = new Set(relevant[B.q]);
    const lists = METHODS.map((m) =>
      section(
        m.name,
        h('ol.emb-rank', rankings[m.id][B.q].slice(0, 5).map((d, r) => h('li' + (rel.has(d.doc) ? '.hit' : ''), h('span.n', String(r + 1)), P[d.doc].title, h('span.s', d.score.toFixed(m.id === 'dense' ? 2 : m.id === 'bm25' ? 1 : 3))))),
      ),
    );
    ctx.ui.right(
      barChart({ title: `Recall@1 over the ${Q.length} questions`, rows: METHODS.map((m) => ({ label: m.name.split(' ')[0], value: metrics[m.id].r1, color: m.color })), max: 1, format: (v) => (v * 100).toFixed(0) + '%' }),
      table(['Method', 'Recall@1', 'Recall@3', 'MRR'], METHODS.map((m) => [m.name, (metrics[m.id].r1 * 100).toFixed(0) + '%', (metrics[m.id].r3 * 100).toFixed(0) + '%', metrics[m.id].mrr.toFixed(2)]), { compact: true }),
      section('Question', qList),
      ...lists,
      text(`${P.length} passages and ${Q.length} questions written for this lab. Green marks the passage that actually answers the question. “Paraphrase” questions share no keywords with their answer; “exact” ones hinge on a rare term.`, 'small'),
    );
  }

  function setStage(s) {
    stage = s;
    mapG.visible = s === 'map';
    raceG.visible = s === 'race';
    benchG.visible = s === 'bench';
    if (s === 'map') {
      app.frame({ target: [0, 0, 0], dir: [0, 0.25, 1], fit: 11.5 }, 1.1);
      renderMapPanel();
    } else if (s === 'race') {
      app.frame({ target: [0, 0, 0], dir: [0.3, 0.3, 1], fit: 12 }, 1.1);
      runQuery();
      renderRacePanel();
    } else {
      app.frame({ target: [0, 1.4, 0], dir: [0, 0.05, 1], fit: 10.6 }, 1.1);
      paintBench();
      renderBenchPanel();
    }
  }

  const steps = [
    {
      title: 'Text becomes a vector',
      text: `An **embedding model** turns a word or a sentence into a list of numbers: here ${dim} of them, from the small open model ${E.model.split('/').pop()}. Every dot is one of ${W} everyday words, placed by its real vector (squeezed into 3D for drawing) and colored by a category the model was never told about.`,
      enter: () => setStage('map'),
    },
    {
      title: 'Near means similar',
      text: 'Nothing sorted these words. The model learned from billions of sentences that words used in similar ways should get similar vectors, so animals drift together, and so do instruments and feelings. **Click a dot**: its nearest neighbors are its closest meanings.',
      enter: () => (setStage('map'), M.sel < 0 && selectWord(E.words.text.indexOf('violin') >= 0 ? E.words.text.indexOf('violin') : 0)),
    },
    {
      title: 'Search is finding neighbors',
      text: `Semantic search, recommendation and RAG are the same operation: embed the query, then find the vectors closest to it. The honest way compares the query with **every** stored vector. Here that is ${fmt.int(ds.n)} distance computations per query; with a billion vectors it is hopeless.`,
      enter: () => ((R.sel = 'flat'), setStage('race')),
    },
    {
      title: 'Indexes trade recall for speed',
      text: 'An index organizes the vectors in advance so a query can skip most of them. **IVF** sorts them into cells and looks only in the nearest few. **HNSW** links each vector to its neighbors and walks the graph. **LSH** hashes similar vectors into the same bucket. They touch a small fraction of the data and sometimes miss a true neighbor. Move the sliders to trade one for the other.',
      enter: () => ((R.sel = 'hnsw'), setStage('race')),
    },
    {
      title: 'Keywords, vectors, or both',
      text: 'For retrieval-augmented generation the question is which passages to hand the model. **BM25** matches rare words exactly, so it finds identifiers and jargon but fails on paraphrases. **Dense** vectors match meaning and miss exact terms. **Hybrid** search fuses the two rankings and is rarely the worst of the three.',
      body: () => callout('Each column is one method’s top five for the selected question; green is the right passage, and the lines show how a passage moves between rankings.', 'key'),
      enter: () => setStage('bench'),
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'All labs →', onFinish: () => ctx.router.go('labs') });
  ctx.ui.left(header({ kicker: 'Lab · Embedding space', title: 'Meaning as geometry', lead: 'Real embeddings you can fly through, the indexes that make searching them fast, and what that means for retrieval.' }), guide.body);
  ctx.ui.dock(guide.dock);

  paintMap();
  resetCloud();
  bench('flat');
  return {
    update(dt, t) {
      qMark.rotation.y = t * 2;
      // build the indexes a little every frame, then benchmark each as it becomes ready
      if (R.builders.length) {
        const t0 = performance.now();
        while (R.builders.length && performance.now() - t0 < 6) {
          const b = R.builders[0];
          if (b.gen.next().done) {
            b.x.ready = true;
            R.builders.shift();
            bench(b.x.id);
            if (stage === 'race' && R.sel === b.x.id) runQuery();
          }
        }
      }
      // reveal what the watched index touched, in order
      if (stage === 'race' && R.trace && R.shown < R.trace.trace.n) {
        const tr = R.trace.trace;
        const per = Math.max(4, Math.ceil(tr.n / 110));
        const c = glow(R.trace.x.color, 2.6);
        for (let k = 0; k < per && R.shown < tr.n; k++, R.shown++) {
          const id = tr.ids[R.shown];
          rcol.set([c.r, c.g, c.b], id * 3);
          rsize[id] = 0.2;
        }
        if (R.shown >= tr.n) {
          const g = glow('#7dffa0', 3);
          for (const id of R.trace.top.ids) {
            if (id < 0) continue;
            rcol.set([g.r, g.g, g.b], id * 3);
            rsize[id] = 0.42;
          }
        }
        cloud.geometry.attributes.color.needsUpdate = true;
        cloud.geometry.attributes.size.needsUpdate = true;
      }
      if (stage === 'map' && M.sel < 0 && !ctx.reduceMotion) mapG.rotation.y = t * 0.03;
      else mapG.rotation.y = 0;
    },
  };
}
