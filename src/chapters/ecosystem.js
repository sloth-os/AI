// Project galaxy: about a hundred projects as stars. Clusters are categories (color + name),
// size follows GitHub stars, shape says how open a project is. A second arrangement stacks the
// same stars into the layers of the AI stack. All content lives in ../data/projects.js.
import * as THREE from 'three';
import './ecosystem.css';
import { PROJECTS, CATEGORIES, LAYERS, KINDS, DEPLOY_MODES, STARS_AS_OF, starCount } from '../data/projects.js';
import { PAL } from '../core/colors.js';
import { points, instanced, halo, line, edges, rng } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, segmented, kv } from '../core/ui.js';
import { damp, ease, clamp } from '../core/tween.js';

// Decorative cluster colors: always shown together with the cluster name.
const CAT_COLOR = {
  frameworks: PAL.amber,
  libraries: '#ffe27a',
  'open-models': PAL.mint,
  'closed-models': PAL.steel,
  eval: '#d9c3a0',
  training: PAL.coral,
  serving: PAL.cyan,
  local: PAL.blue,
  edge: PAL.lime,
  retrieval: PAL.violet,
  appfw: '#d98cff',
  agents: PAL.red,
  coding: PAL.magenta,
};
// Short names for the 3D name plates and the chips; the full name is in the card and the tooltip.
const SHORT = {
  frameworks: 'Frameworks',
  libraries: 'Model libraries',
  'open-models': 'Open models',
  'closed-models': 'Closed models',
  eval: 'Evaluation',
  training: 'Training',
  serving: 'GPU serving',
  local: 'Local runners',
  edge: 'Browser and edge',
  retrieval: 'Retrieval',
  appfw: 'App frameworks',
  agents: 'Agent frameworks',
  coding: 'Coding agents',
};
const LAYER_SHORT = { training: 'Training', serving: 'Serving', apps: 'App frameworks' };
const KIND_NAME = Object.fromEntries(KINDS.map((k) => [k.id, k.name]));
const DEPLOY_NAME = Object.fromEntries(DEPLOY_MODES.map((d) => [d.id, d.name]));
const SPACING = 1.0; // distance scale between stars inside a cluster
const SPIN = 0.028; // ambient rotation, rad/s
const Y = new THREE.Vector3(0, 1, 0);

export async function enter(ctx) {
  const { app, root } = ctx;
  const N = PROJECTS.length;
  const NC = CATEGORIES.length;
  const catIdx = Object.fromEntries(CATEGORIES.map((c, i) => [c.id, i]));

  // ---------------------------------------------------------------- per-project state
  const P = PROJECTS.map((p, i) => {
    const n = starCount(p.stars);
    const cat = catIdx[p.category];
    return {
      p, i, n, cat,
      r: n ? 0.26 + 0.44 * clamp((Math.log10(n) - 3.4) / 2.2) : 0.4, // log(stars); fixed size without a repo
      color: new THREE.Color(CAT_COLOR[p.category]),
      order: cat / (NC - 1),
      hay: [p.name, p.org, p.repo ?? '', CATEGORIES[cat].name, KIND_NAME[p.kind], p.oneLine, p.coreAlgo, p.process.join(' '), p.deploy.map((d) => DEPLOY_NAME[d]).join(' ')].join(' ').toLowerCase(),
      g: new THREE.Vector3(), s: new THREE.Vector3(), pos: new THREE.Vector3(),
      match: true, shown: false, label: null, side: 1, mesh: null, slot: 0, top: false,
    };
  });
  const byId = Object.fromEntries(P.map((s) => [s.p.id, s]));
  const byStars = [...P].sort((a, b) => b.n - a.n || a.p.name.localeCompare(b.p.name));
  byStars.slice(0, 18).forEach((s) => (s.top = true));
  const members = CATEGORIES.map((c) => byStars.filter((s) => s.p.category === c.id));
  const R = members.map((m) => SPACING * Math.sqrt(m.length)); // cluster radius

  // ---------------------------------------------------------------- galaxy layout: a core and three arms
  const rnd = rng(7);
  const centers = CATEGORIES.map(() => new THREE.Vector3());
  const core = CATEGORIES.findIndex((c) => c.arm === 'core');
  const armIds = [...new Set(CATEGORIES.filter((c) => c.arm !== 'core').map((c) => c.arm))];
  const armCats = armIds.map((arm) => CATEGORIES.map((c, ci) => ci).filter((ci) => CATEGORIES[ci].arm === arm));
  armCats.forEach((list, a) => {
    let r = 0;
    let th = 0.35 + (a / armCats.length) * Math.PI * 2;
    list.forEach((ci, k) => {
      if (k === 0) r = R[core] + R[ci] + 1.0;
      else {
        const d = R[list[k - 1]] + R[ci] + 0.9;
        const dr = 0.36 * d;
        th += Math.sqrt(d * d - dr * dr) / (r + dr / 2);
        r += dr;
      }
      centers[ci].set(Math.cos(th) * r, rnd.range(-0.45, 0.45), Math.sin(th) * r);
    });
  });
  for (let it = 0; it < 80; it++) {
    // push apart any clusters that still overlap (the core stays put)
    for (let a = 0; a < NC; a++)
      for (let b = a + 1; b < NC; b++) {
        const dx = centers[b].x - centers[a].x;
        const dz = centers[b].z - centers[a].z;
        const d = Math.hypot(dx, dz) || 1e-3;
        const need = R[a] + R[b] + 0.9;
        if (d >= need) continue;
        const push = (need - d) / d;
        const wa = a === core ? 0 : b === core ? 1 : 0.5;
        centers[a].x -= dx * push * wa;
        centers[a].z -= dz * push * wa;
        centers[b].x += dx * push * (1 - wa);
        centers[b].z += dz * push * (1 - wa);
      }
  }
  members.forEach((m, ci) => {
    const phase = rnd() * Math.PI * 2;
    m.forEach((s, k) => {
      // sunflower packing: the most-starred project sits in the middle
      const rr = SPACING * 0.92 * Math.sqrt(k + 0.5);
      const a = phase + k * 2.39996;
      s.g.set(centers[ci].x + Math.cos(a) * rr, centers[ci].y + rnd.range(-0.5, 0.5), centers[ci].z + Math.sin(a) * rr);
    });
  });
  const galaxyR = Math.max(...centers.map((c, ci) => Math.hypot(c.x, c.z) + R[ci]));

  // ---------------------------------------------------------------- stack layout: eight plates, bottom to top
  const DY = 4.0;
  const SX = 1.9;
  const SZ = 1.6;
  const PD = 3 * SZ + 0.8;
  const catStack = CATEGORIES.map(() => new THREE.Vector3()); // name plates sit to the right of their layer
  const catStackCY = CATEGORIES.map(() => 0.5);
  let PW = 0;
  const layerY = LAYERS.map((_, li) => (li - (LAYERS.length - 1) / 2) * DY);
  LAYERS.forEach((L, li) => {
    const cats = CATEGORIES.map((c, ci) => ci).filter((ci) => CATEGORIES[ci].layer === L.id);
    const total = cats.reduce((sum, ci) => sum + members[ci].length, 0);
    const rows = total > 16 ? 3 : total > 6 ? 2 : 1;
    const cols = cats.map((ci) => Math.ceil(members[ci].length / rows));
    const gap = 1.0;
    const width = cols.reduce((sum, c) => sum + c * SX, 0) + gap * (cats.length - 1);
    PW = Math.max(PW, width);
    let x = -width / 2;
    cats.forEach((ci, k) => {
      const w = cols[k] * SX;
      members[ci].forEach((s, j) => {
        const col = Math.floor(j / rows);
        const row = j % rows;
        s.s.set(x + (col + 0.5) * SX, layerY[li] + 0.6, ((rows - 1) / 2 - row) * SZ);
      });
      catStack[ci].set(0, layerY[li], PD / 2);
      catStackCY[ci] = 0.5 + (k - (cats.length - 1) / 2) * 1.15;
      x += w + gap;
    });
  });
  PW += 1.8;
  catStack.forEach((v) => (v.x = PW / 2 + 0.5));
  const catStackCenter = members.map((m) => m.reduce((acc, s) => acc.add(s.s), new THREE.Vector3()).multiplyScalar(1 / m.length));
  const stackFit = Math.hypot(PW / 2, (DY * (LAYERS.length - 1)) / 2 + 1.2, PD / 2) + 0.6;
  const VIEW = {
    galaxy: { dir: [0, 1.15, 1], fit: galaxyR + 0.6 },
    stack: { dir: [0.42, 0.55, 0.9], fit: stackFit },
  };
  app.setView({ pos: VIEW.galaxy.dir, target: [0, 0, 0], fit: VIEW.galaxy.fit, min: 3, max: 160 });

  // name plates have a fixed size on screen; the engine widens the field of view on narrow screens, so compensate
  const labelK = Math.tan((app.camera.fov * Math.PI) / 360) / Math.tan((25 * Math.PI) / 180);

  // ---------------------------------------------------------------- scene
  const galaxy = new THREE.Group();
  root.add(galaxy);
  root.add(new THREE.AmbientLight(0xffffff, 2.0)); // stars are self-lit; the scene lights only add a little shading

  // background dust that follows the three arms
  const dustN = { low: 900, medium: 2200, high: 3400 }[ctx.quality] ?? 2200;
  const dust = (() => {
    const rd = rng(11);
    const pos = new Float32Array(dustN * 3);
    const col = new Float32Array(dustN * 3);
    const siz = new Float32Array(dustN);
    const curves = armCats.map((list) => {
      const cs = list.map((ci) => centers[ci].clone().setY(0));
      const last = cs[cs.length - 1];
      const prev = cs[cs.length - 2] ?? new THREE.Vector3();
      return new THREE.CatmullRomCurve3([cs[0].clone().multiplyScalar(0.3), ...cs, last.clone().add(last.clone().sub(prev).multiplyScalar(0.7))]);
    });
    const p = new THREE.Vector3();
    for (let i = 0; i < dustN; i++) {
      const u = rd();
      let x;
      let y;
      let z;
      if (u < 0.2) {
        const r = Math.abs(rd.normal()) * 3.2;
        const a = rd() * Math.PI * 2;
        x = Math.cos(a) * r;
        z = Math.sin(a) * r;
        y = rd.normal() * 0.7;
      } else if (u < 0.32) {
        const r = Math.sqrt(rd()) * galaxyR * 1.15;
        const a = rd() * Math.PI * 2;
        x = Math.cos(a) * r;
        z = Math.sin(a) * r;
        y = rd.normal() * 0.5;
      } else {
        const v = rd();
        curves[rd.int(curves.length)].getPoint(v, p);
        const sp = 1.0 + 1.9 * v;
        x = p.x + rd.normal() * sp;
        z = p.z + rd.normal() * sp;
        y = rd.normal() * 0.45;
      }
      pos.set([x, y, z], i * 3);
      const b = 0.14 + 0.5 * rd() ** 2;
      col.set([0.62 * b, 0.74 * b, b], i * 3);
      siz[i] = 0.1 + 0.26 * rd();
    }
    return points(pos, col, siz);
  })();
  galaxy.add(dust);

  // a soft cloud, a stem and a name plate per cluster
  const nebulas = [];
  const stems = [];
  const catLabels = [];
  const catGalaxy = centers.map((c, ci) => new THREE.Vector3(c.x, c.y + 1.9 + R[ci] * 0.7, c.z));
  CATEGORIES.forEach((c, ci) => {
    const color = CAT_COLOR[c.id];
    const neb = halo(color, R[ci] * 3.6, 0.13, 1.0);
    neb.position.copy(centers[ci]);
    neb.renderOrder = -2;
    nebulas.push(neb);
    const stem = line([centers[ci], catGalaxy[ci]], color, { k: 1.2, opacity: 0.45 });
    stems.push(stem);
    const lab = textSprite(SHORT[c.id], { size: 0.0132 * labelK, attenuate: false, weight: 700, color: '#eef3ff', bg: 'rgba(6,10,20,0.84)', border: color, anchor: [0.5, 0], onTop: true });
    lab.renderOrder = 22;
    lab.userData.s0 = lab.scale.clone();
    catLabels.push(lab);
    galaxy.add(neb, stem, lab);
  });

  // plates and layer names of the stack arrangement
  const stackGroup = new THREE.Group();
  stackGroup.visible = false;
  galaxy.add(stackGroup);
  const plateGeo = new THREE.BoxGeometry(PW, 0.06, PD);
  const plateMat = new THREE.MeshBasicMaterial({ color: '#0e1830', transparent: true, opacity: 0, depthWrite: false, fog: false });
  const plateEdges = [];
  const layerLabels = LAYERS.map((L, li) => {
    const plate = new THREE.Mesh(plateGeo, plateMat);
    plate.position.y = layerY[li];
    plate.renderOrder = -3;
    const edge = edges(plateGeo, '#5f7fb8', { k: 1.0, opacity: 0.6 });
    edge.position.y = layerY[li];
    plateEdges.push(edge);
    const lab = textSprite(`${li + 1} · ${LAYER_SHORT[L.id] ?? L.name}`, { size: 0.0138 * labelK, attenuate: false, weight: 650, font: 'display', color: '#c3cee6', bg: 'rgba(6,10,20,0.7)', anchor: [1, 0.5], onTop: true });
    lab.position.set(-PW / 2 - 0.5, layerY[li], PD / 2);
    stackGroup.add(plate, edge, lab);
    return lab;
  });

  // stars: one instanced mesh per kind, so the kind is a shape and not only a color
  const geoms = {
    'open-source': new THREE.SphereGeometry(1, 20, 14),
    'open-weights': new THREE.TorusGeometry(0.86, 0.25, 10, 30),
    closed: new THREE.OctahedronGeometry(1.2, 0),
  };
  const starMeshes = KINDS.map((k) => {
    const list = P.filter((s) => s.p.kind === k.id);
    const mesh = instanced(geoms[k.id], new THREE.MeshLambertMaterial({ color: 0xffffff, fog: false }), list.length);
    list.forEach((s, j) => {
      s.mesh = mesh;
      s.slot = j;
    });
    galaxy.add(mesh);
    return mesh;
  });
  const haloPos = new Float32Array(N * 3);
  const haloCol = new Float32Array(N * 3);
  const haloSize = new Float32Array(N);
  const halos = points(haloPos, haloCol, haloSize);
  galaxy.add(halos);
  // invisible, generous hit targets (good for touch; rings are hollow)
  const pickMesh = instanced(new THREE.SphereGeometry(1, 10, 8), new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false, transparent: true, opacity: 0 }), N);
  galaxy.add(pickMesh);

  const ringGeo = new THREE.TorusGeometry(1, 0.045, 8, 56);
  const selRing = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: new THREE.Color(2.2, 2.2, 2.2), fog: false }));
  const hovRing = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: new THREE.Color(1.2, 1.2, 1.2), transparent: true, opacity: 0.75, fog: false }));
  selRing.visible = hovRing.visible = false;
  galaxy.add(selRing, hovRing);

  // ---------------------------------------------------------------- state
  let arrangement = 'galaxy';
  let mix = 0; // 0 = galaxy, 1 = stack
  let mixTween = null;
  let spin = 0;
  let spinSpeed = 0;
  let selected = null;
  let hovered = null;
  let active = false; // any filter set
  let labelsDirty = true;
  let labelTimer = 0;
  let pointerDown = false;
  let lastTouch = -1e9;
  let pickDirty = true;
  const filter = { q: '', cat: null, kind: 'all', deploy: null };
  const tmpC = new THREE.Color();

  function recolor() {
    for (const s of P) {
      const dim = active && !s.match;
      s.mesh.setColorAt(s.slot, tmpC.copy(s.color).multiplyScalar(dim ? 0.09 : 1.0));
      tmpC.copy(s.color).multiplyScalar(dim ? 0.02 : 0.24).toArray(haloCol, s.i * 3);
      haloSize[s.i] = s.r * (dim ? 3 : 5);
    }
    for (const m of starMeshes) m.instanceColor.needsUpdate = true;
    halos.geometry.attributes.color.needsUpdate = true;
    halos.geometry.attributes.size.needsUpdate = true;
  }

  const visibleRanked = () => byStars.filter((s) => s.match);

  // ---------------------------------------------------------------- HUD: left panel
  const glyph = (s, cls = '') => h('i.eco-glyph.' + s.p.kind + (cls ? '.' + cls : ''), { style: { '--c': CAT_COLOR[s.p.category] } });
  const kindGlyph = (kind) => h('i.eco-glyph.' + kind);

  const search = h('input', {
    type: 'text',
    placeholder: 'Search name, org, algorithm…',
    'aria-label': 'Search projects',
    autocomplete: 'off',
    spellcheck: 'false',
    oninput: () => {
      filter.q = search.value;
      applyFilter();
    },
    onkeydown: (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') {
        const first = visibleRanked()[0];
        if (first) select(first.i);
      } else if (e.key === 'Escape') {
        search.value = filter.q = '';
        applyFilter();
        search.blur();
      }
    },
  });
  const countEl = h('span');
  const clearBtn = h('button.btn.small.ghost', { type: 'button', onclick: () => resetFilters() }, 'Clear filters');
  const catChips = h('div.chips.eco-chips');
  const catChipEls = CATEGORIES.map((c, ci) => {
    const el = h('span.chip', { role: 'button', tabindex: '0', title: `${c.name}. ${c.blurb}`, style: { '--c': CAT_COLOR[c.id] } }, h('i.eco-dot'), SHORT[c.id], h('small', String(members[ci].length)));
    const toggle = () => focusCategory(filter.cat === c.id ? null : c.id);
    el.addEventListener('click', toggle);
    el.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), toggle()));
    catChips.appendChild(el);
    return el;
  });
  const catNote = h('div.txt.small');
  const kindSeg = segmented(
    [{ value: 'all', label: 'All' }, ...KINDS.map((k) => ({ value: k.id, label: h('span', kindGlyph(k.id), k.name), title: k.note }))],
    'all',
    (v) => {
      filter.kind = v;
      applyFilter();
    },
    { wrap: true },
  );
  const depChips = h('div.chips.eco-chips');
  const depChipEls = DEPLOY_MODES.map((d) => {
    const count = PROJECTS.filter((p) => p.deploy.includes(d.id)).length;
    const el = h('span.chip', { role: 'button', tabindex: '0', style: { '--c': 'var(--accent)' } }, d.name, h('small', String(count)));
    const toggle = () => {
      filter.deploy = filter.deploy === d.id ? null : d.id;
      applyFilter();
    };
    el.addEventListener('click', toggle);
    el.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), toggle()));
    depChips.appendChild(el);
    return el;
  });
  const listEl = h('div.eco-list');
  let rowEls = new Map();

  function renderList() {
    const vis = visibleRanked();
    rowEls = new Map();
    listEl.replaceChildren(
      ...vis.map((s, k) => {
        const row = h(
          'button.eco-row' + (s.i === selected ? '.on' : ''),
          { type: 'button', title: s.p.oneLine, onclick: () => select(s.i) },
          h('span.eco-rank', String(k + 1)),
          glyph(s),
          h('span.eco-name', s.p.name, h('small', s.p.org)),
          h('span.eco-stars', { title: s.p.stars ? `GitHub stars, ${STARS_AS_OF}` : 'No public repository' }, s.p.stars ?? '–'),
        );
        rowEls.set(s.i, row);
        return row;
      }),
    );
    if (!vis.length) listEl.appendChild(h('div.eco-empty', 'Nothing matches. Clear a filter to see more.'));
    countEl.textContent = active ? `${vis.length} of ${N} projects match` : `All ${N} projects`;
    clearBtn.hidden = !active;
  }

  function applyFilter() {
    const tokens = filter.q.toLowerCase().split(/\s+/).filter(Boolean);
    active = !!(tokens.length || filter.cat || filter.kind !== 'all' || filter.deploy);
    for (const s of P) {
      s.match = (!filter.cat || s.p.category === filter.cat) && (filter.kind === 'all' || s.p.kind === filter.kind) && (!filter.deploy || s.p.deploy.includes(filter.deploy)) && tokens.every((tk) => s.hay.includes(tk));
    }
    catChipEls.forEach((el, ci) => el.classList.toggle('on', CATEGORIES[ci].id === filter.cat));
    catChips.classList.toggle('has-on', !!filter.cat);
    depChipEls.forEach((el, k) => el.classList.toggle('on', DEPLOY_MODES[k].id === filter.deploy));
    depChips.classList.toggle('has-on', !!filter.deploy);
    kindSeg.set(filter.kind);
    const cat = filter.cat && CATEGORIES[catIdx[filter.cat]];
    catNote.hidden = !cat;
    catNote.replaceChildren(cat ? h('p', h('strong', cat.name + '. '), cat.blurb) : '');
    recolor();
    renderList();
    labelsDirty = true;
  }

  function resetFilters() {
    const hadCat = !!filter.cat;
    Object.assign(filter, { q: '', cat: null, kind: 'all', deploy: null });
    search.value = '';
    applyFilter();
    if (hadCat && selected == null) app.home();
  }

  function focusCategory(id) {
    filter.cat = id;
    applyFilter();
    if (!id) {
      if (selected == null) app.home();
      return;
    }
    const ci = catIdx[id];
    const target = arrangement === 'stack' ? catStackCenter[ci].clone() : centers[ci].clone().applyAxisAngle(Y, spin);
    spinSpeed = 0;
    app.frame({ target, dir: app.camera.position.clone().sub(app.controls.target), fit: R[ci] + 2.6 }, 1.2);
  }

  ctx.ui.left(
    header({
      kicker: 'Ecosystem · Project galaxy',
      title: 'What people actually build with',
      lead: `**${N} projects** in ${NC} clusters. A star’s size follows its GitHub stars; its shape says how open it is.`,
    }),
    section('Find', h('div.eco-search', search), h('div.eco-count', countEl, clearBtn)),
    section('Cluster', catChips, catNote),
    section('Openness', kindSeg),
    section('Deployment mode', depChips),
    section('Ranked by GitHub stars', listEl),
  );

  // ---------------------------------------------------------------- HUD: right panel and dock
  const viewOpts = [
    { value: 'galaxy', label: 'Galaxy', title: 'Clusters by category' },
    { value: 'stack', label: 'Stack', title: 'Layers from frameworks up to user-facing tools' },
  ];
  const segA = segmented(viewOpts, 'galaxy', (v) => setArrangement(v));
  const segB = segmented(viewOpts, 'galaxy', (v) => setArrangement(v));
  const viewNote = h('div.txt.small');
  const detail = h('div.eco-detail');
  const VIEW_NOTE = {
    galaxy: 'Thirteen clusters on three arms around the frameworks at the core: models, running them, and building with them.',
    stack: 'The same projects as eight layers. Each layer builds on the ones below it; read from the bottom up.',
  };

  function guide() {
    const counts = Object.fromEntries(KINDS.map((k) => [k.id, PROJECTS.filter((p) => p.kind === k.id).length]));
    const withStars = PROJECTS.filter((p) => p.stars).length;
    const ask = (label, fn) => {
      const el = h('span.chip.click', { role: 'button', tabindex: '0' }, label);
      el.addEventListener('click', fn);
      return el;
    };
    const setFilter = (patch) => () => {
      Object.assign(filter, { q: '', cat: null, kind: 'all', deploy: null }, patch);
      search.value = '';
      applyFilter();
      if (window.innerWidth <= 860) app.hud.setSheet?.('left');
    };
    return [
      section(
        'How to read the map',
        h(
          'div.eco-legend',
          KINDS.map((k) => h('div', kindGlyph(k.id), h('span', h('b', `${k.name} (${counts[k.id]})`), ': ' + k.note))),
          h('div', h('span'), h('span', h('b', 'Size'), `: GitHub stars on a log scale, as displayed on ${STARS_AS_OF} (${withStars} of ${N} projects). Projects without a public repository get one fixed size.`)),
          h('div', h('span'), h('span', h('b', 'Color'), ': the cluster. Filters dim what does not match; nothing changes color.')),
        ),
      ),
      section(
        'Try a question',
        h(
          'div.chips',
          ask('What runs in the browser?', setFilter({ deploy: 'browser' })),
          ask('What can run air-gapped?', setFilter({ deploy: 'air-gapped' })),
          ask('Which products are closed?', setFilter({ kind: 'closed' })),
          ask('Who uses PagedAttention?', setFilter({ q: 'PagedAttention' })),
        ),
        h('div.txt.small', h('p', 'Click a star, or a row in the ranked list, to open its card: core algorithm, major process, deployment modes.')),
      ),
    ];
  }

  function card(s) {
    const p = s.p;
    const cat = CATEGORIES[s.cat];
    const vis = visibleRanked();
    const at = vis.indexOf(s);
    const step = (d) => () => {
      const list = vis.length ? vis : byStars;
      const k = at < 0 ? 0 : (at + d + list.length) % list.length;
      select(list[k].i);
    };
    const host = p.repo ? 'github.com/' + p.repo : p.url.replace(/^https?:\/\/(www\.)?/, '').split('/')[0];
    return h(
      'div.card.eco-card',
      h(
        'div.eco-head',
        glyph(s, 'big'),
        h('h2', p.name),
        h(
          'div.eco-nav',
          h('button.btn.small.ghost', { type: 'button', title: 'Previous in the list (←)', 'aria-label': 'Previous project', onclick: step(-1) }, '←'),
          h('button.btn.small.ghost', { type: 'button', title: 'Next in the list (→)', 'aria-label': 'Next project', onclick: step(1) }, '→'),
          h('button.btn.small.ghost', { type: 'button', title: 'Close (Esc)', 'aria-label': 'Close', onclick: () => clearSel(true) }, '✕'),
        ),
      ),
      h('p.eco-one', p.oneLine),
      kv([
        ['Organization', h('span', p.org)],
        ['Cluster', h('span.eco-inline', h('i.eco-dot', { style: { '--c': CAT_COLOR[cat.id] } }), cat.name)],
        ['Openness', h('span.eco-inline', kindGlyph(p.kind), KIND_NAME[p.kind])],
        ['First release', h('span', p.year ? String(p.year) : 'not confirmed')],
        ['GitHub stars', h('span', p.stars ? `${p.stars} (as of ${STARS_AS_OF})` : 'no public repository')],
        p.flagship && [p.kind === 'closed' ? 'Current flagship' : 'Lab’s flagship', h('span', p.flagship)],
        p.breakout && ['Note', h('span', 'One of the open-source breakouts of 2026')],
      ]),
      h('h4', 'Core algorithm'),
      h('p.eco-algo', p.coreAlgo),
      h('h4', 'Major process'),
      h('ol.eco-steps', p.process.map((st) => h('li', st))),
      h('h4', 'Deployment modes'),
      h('div.chips', p.deploy.map((id) => h('a.chip.click', { href: '#/deploy/' + id, title: 'Open this mode in the Deploy chapter' }, DEPLOY_NAME[id] + ' →'))),
      h('div.eco-foot', h('a.btn.small.eco-link', { href: p.url, target: '_blank', rel: 'noopener' }, host + ' ↗')),
    );
  }

  function renderDetail() {
    detail.replaceChildren(...(selected != null ? [card(P[selected])] : guide()));
  }

  ctx.ui.right(section('Arrangement', segA, viewNote), detail);
  ctx.ui.dock(h('div.eco-dock', segB, h('span.eco-hint', 'Click a star for its card · drag to orbit · scroll to zoom')));

  // ---------------------------------------------------------------- selection, arrangement, camera
  const worldTarget = (s) => (arrangement === 'stack' ? s.s.clone() : s.g.clone().applyAxisAngle(Y, spin));

  function select(i, { fly = true, url = true } = {}) {
    selected = i;
    const s = P[i];
    spinSpeed = 0;
    if (url) ctx.router.replace('ecosystem/' + s.p.id);
    renderDetail();
    for (const [k, row] of rowEls) row.classList.toggle('on', k === i);
    labelsDirty = true;
    if (window.innerWidth <= 860) app.hud.setSheet?.('right');
    if (fly) app.frame({ target: worldTarget(s), dir: app.camera.position.clone().sub(app.controls.target), fit: Math.max(7.5, R[s.cat] * 1.3 + 3.5) }, 1.2);
  }

  function clearSel(goHome) {
    if (selected == null) return;
    selected = null;
    ctx.router.replace('ecosystem');
    renderDetail();
    for (const row of rowEls.values()) row.classList.remove('on');
    labelsDirty = true;
    if (goHome) app.home();
  }

  function setArrangement(mode) {
    if (mode === arrangement) return;
    arrangement = mode;
    segA.set(mode);
    segB.set(mode);
    viewNote.replaceChildren(h('p', VIEW_NOTE[mode]));
    spin = Math.atan2(Math.sin(spin), Math.cos(spin));
    spinSpeed = 0;
    mixTween?.cancel();
    const from = mix;
    const to = mode === 'stack' ? 1 : 0;
    mixTween = ctx.tweens.add({ dur: ctx.reduceMotion ? 0.01 : 1.6, ease: (k) => k, update: (k) => (mix = from + (to - from) * k) });
    const v = VIEW[mode];
    app.frame({ target: [0, 0, 0], dir: v.dir, fit: v.fit }, 1.5).then((ok) => ok && !ctx.dead && app.setHome(app.camera.position, app.controls.target));
  }

  ctx.pick.add(pickMesh, {
    tip: (hit) => tipFor(hit.instanceId),
    hover: (hit) => {
      hovered = hit.instanceId;
      labelsDirty = true;
    },
    out: () => {
      hovered = null;
      labelsDirty = true;
    },
    click: (hit) => select(hit.instanceId),
  });
  ctx.pick.onMiss(() => clearSel(false));
  const tips = new Map();
  function tipFor(i) {
    if (!tips.has(i)) {
      const p = P[i].p;
      tips.set(i, h('div', h('b', p.name), h('small', p.oneLine), h('small', p.stars ? `${p.stars} GitHub stars · ${KIND_NAME[p.kind]}` : `${KIND_NAME[p.kind]} · no public repository`)));
    }
    return tips.get(i);
  }

  const typing = () => /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName ?? '');
  const stepSel = (d) => {
    if (typing()) return;
    const list = visibleRanked();
    if (!list.length) return;
    const at = selected == null ? -1 : list.indexOf(P[selected]);
    select(list[(at + d + list.length) % list.length].i);
  };
  ctx.key('Escape', () => clearSel(true));
  ctx.key('ArrowRight', () => stepSel(1));
  ctx.key('ArrowLeft', () => stepSel(-1));
  ctx.key('/', () => {
    if (typing()) return;
    search.focus();
  });
  const canvas = app.renderer.domElement;
  ctx.listen(canvas, 'pointerdown', () => {
    pointerDown = true;
    lastTouch = performance.now();
  });
  ctx.listen(window, 'pointerup', () => (pointerDown = false));
  ctx.listen(canvas, 'wheel', () => (lastTouch = performance.now()), { passive: true });

  // ---------------------------------------------------------------- labels: names for the biggest, decluttered on screen
  const labelFor = (s) => {
    if (!s.label) {
      s.label = textSprite(s.p.name, { size: 0.0117 * labelK, attenuate: false, weight: 600, color: '#e3eaf8', bg: 'rgba(5,8,16,0.62)', anchor: [0.5, 0], onTop: true });
      s.label.visible = false;
      galaxy.add(s.label);
    }
    return s.label;
  };
  const _v = new THREE.Vector3();
  function rectOf(sprite, W, H, pxs) {
    sprite.getWorldPosition(_v).project(app.camera);
    if (_v.z > 1 || _v.z < -1) return null;
    const x = (_v.x * 0.5 + 0.5) * W;
    const y = (0.5 - _v.y * 0.5) * H;
    const w = sprite.scale.x * pxs;
    const hh = sprite.scale.y * pxs;
    const x0 = x - w * sprite.center.x;
    const y0 = y - hh * (1 - sprite.center.y);
    if (x0 > W || x0 + w < 0 || y0 > H || y0 + hh < 0) return null;
    return [x0 - 2, y0 - 1, x0 + w + 2, y0 + hh + 1];
  }
  const hits = (a, list) => list.some((b) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]);

  function layoutLabels() {
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    const pxs = H / (2 * Math.tan((app.camera.fov * Math.PI) / 360));
    const placed = [];
    // cluster name plates first: when two collide, one steps up or down by its own height
    const km = ease.inOut(mix);
    catLabels.forEach((lab, ci) => {
      const cy0 = km * catStackCY[ci]; // galaxy: sits above its stem; stack: listed to the right of its layer
      lab.center.set(0.5 * (1 - km), cy0);
      let r = rectOf(lab, W, H, pxs);
      if (!r) return;
      if (hits(r, placed)) {
        const hgt = r[3] - r[1];
        for (const k of [1, -1, 2, -2, 3, -3]) {
          const moved = [r[0], r[1] - k * hgt, r[2], r[3] - k * hgt];
          if (hits(moved, placed)) continue;
          lab.center.y = cy0 - (k * hgt) / (lab.scale.y * pxs);
          r = moved;
          break;
        }
      }
      placed.push(r);
    });
    if (mix > 0.5)
      for (const lab of layerLabels) {
        const r = rectOf(lab, W, H, pxs);
        if (r) placed.push(r);
      }
    const score = (s) => (s.i === selected ? 8e9 : 0) + (s.i === hovered ? 4e9 : 0) + (s.top ? 2e9 : 0) + (s.shown ? 1e9 : 0) + s.n;
    const cand = P.filter((s) => !active || s.match || s.i === selected || s.i === hovered).sort((a, b) => score(b) - score(a));
    const budget = active ? 60 : 44;
    let count = 0;
    const keep = new Set();
    for (const s of cand) {
      const forced = s.i === selected || s.i === hovered;
      if (!forced && count >= budget) break;
      const lab = labelFor(s);
      // a name goes above its star, or below it when that spot is taken
      let r = null;
      for (const side of [s.side, -s.side]) {
        lab.center.y = side > 0 ? 0 : 1;
        lab.position.copy(s.pos).addScaledVector(_up, side * (s.r + 0.1));
        const rr = rectOf(lab, W, H, pxs);
        if (!rr) break;
        if (forced || !hits(rr, placed)) {
          r = rr;
          s.side = side;
          break;
        }
      }
      if (!r) {
        lab.center.y = s.side > 0 ? 0 : 1;
        continue;
      }
      placed.push(r);
      keep.add(s.i);
      count++;
    }
    for (const s of P) {
      s.shown = keep.has(s.i);
      if (s.label) s.label.visible = s.shown;
    }
  }

  // ---------------------------------------------------------------- start
  viewNote.replaceChildren(h('p', VIEW_NOTE.galaxy));
  applyFilter();
  renderDetail();
  const openFromRoute = (route, fly) => {
    const s = byId[route?.rest?.[0]];
    if (s) {
      if (s.i !== selected) select(s.i, { fly, url: false });
    } else if (selected != null && !route?.rest?.[0]) clearSel(false);
  };
  if (byId[ctx.route.rest?.[0]]) ctx.after(400, () => openFromRoute(ctx.route, true));

  const _q = new THREE.Quaternion();
  const _up = new THREE.Vector3(0, 1, 0);
  const _qi = new THREE.Quaternion();
  const _qs = new THREE.Quaternion();
  const _m = new THREE.Matrix4();
  const _sc = new THREE.Vector3();
  let lastMix = -1;

  return {
    onRoute(route) {
      openFromRoute(route, true);
    },
    update(dt, t) {
      // ambient rotation, paused by any interaction
      const paused = ctx.reduceMotion || arrangement === 'stack' || selected != null || hovered != null || pointerDown || active || performance.now() - lastTouch < 4000;
      spinSpeed = paused ? 0 : damp(spinSpeed, SPIN, 0.8, dt);
      spin += spinSpeed * dt;
      const km = ease.inOut(mix);
      galaxy.rotation.y = spin * (1 - km);
      galaxy.updateMatrixWorld();
      galaxy.getWorldQuaternion(_q).invert().multiply(app.camera.quaternion); // faces the camera, in galaxy space
      _up.set(0, 1, 0).applyQuaternion(_q); // screen-up, in galaxy space
      _qs.setFromAxisAngle(Y, ctx.reduceMotion ? 0.6 : t * 0.5);

      for (const s of P) {
        const k = ease.inOut(clamp(mix * 1.35 - 0.35 * s.order));
        s.pos.lerpVectors(s.g, s.s, k);
        const sc = s.r * (s.i === selected ? 1.22 : s.i === hovered ? 1.15 : 1) * (1 - 0.15 * km);
        _m.compose(s.pos, s.p.kind === 'open-weights' ? _q : s.p.kind === 'closed' ? _qs : _qi, _sc.setScalar(sc));
        s.mesh.setMatrixAt(s.slot, _m);
        _m.compose(s.pos, _qi, _sc.setScalar(Math.max(s.r * 1.5, 0.55)));
        pickMesh.setMatrixAt(s.i, _m);
        s.pos.toArray(haloPos, s.i * 3);
        if (s.label) s.label.position.copy(s.pos).addScaledVector(_up, s.side * (sc + 0.1));
      }
      for (const m of starMeshes) m.instanceMatrix.needsUpdate = true;
      pickMesh.instanceMatrix.needsUpdate = true;
      halos.geometry.attributes.position.needsUpdate = true;
      if (mix !== lastMix || pickDirty) {
        pickMesh.computeBoundingSphere?.();
        pickDirty = false;
        labelsDirty = true;
      }

      // cluster furniture follows the arrangement
      if (mix !== lastMix) {
        CATEGORIES.forEach((c, ci) => {
          catLabels[ci].position.lerpVectors(catGalaxy[ci], catStack[ci], km);
          catLabels[ci].scale.copy(catLabels[ci].userData.s0).multiplyScalar(1 - 0.15 * km); // the stack lists up to three per layer
          nebulas[ci].material.opacity = 0.13 * (1 - km);
          nebulas[ci].visible = km < 0.98;
          stems[ci].material.opacity = 0.45 * (1 - km);
          stems[ci].visible = km < 0.98;
        });
        stackGroup.visible = km > 0.01;
        plateMat.opacity = 0.5 * km;
        for (const e of plateEdges) e.material.opacity = 0.6 * km;
        for (const lab of layerLabels) lab.material.opacity = km;
        dust.material.uniforms.uOpacity.value = 1 - 0.85 * km;
        lastMix = mix;
      }

      const ring = (mesh, i, k) => {
        mesh.visible = i != null;
        if (i == null) return;
        mesh.position.copy(P[i].pos);
        mesh.quaternion.copy(_q);
        mesh.scale.setScalar(P[i].r * k);
      };
      ring(selRing, selected, 1.85 + (ctx.reduceMotion ? 0 : 0.08 * Math.sin(t * 4)));
      ring(hovRing, hovered !== selected ? hovered : null, 1.6);

      labelTimer += dt;
      if (labelTimer > (labelsDirty ? 0.03 : 0.14)) {
        labelTimer = 0;
        labelsDirty = false;
        layoutLabels();
      }
    },
  };
}
