// Families: every kind of AI on one map. The same 70 methods can be arranged as a family tree,
// along a timeline, or by the kind of signal they learn from; positions glide between the three.
import * as THREE from 'three';
import './families.css';
import { FAMILIES, SIGNALS, ERAS, METHODS, LAB_NAMES, MILESTONES, YEAR_MIN, YEAR_MAX } from '../data/families.js';
import { glow } from '../core/colors.js';
import { instanced, flushInst, line, gridFloor, glowMat, edges } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, button, segmented, callout } from '../core/ui.js';

const byId = Object.fromEntries(METHODS.map((m) => [m.id, m]));
const famIndex = Object.fromEntries(FAMILIES.map((f, i) => [f.id, i]));
const SIZE = [0, 0.34, 0.46, 0.62];
const MODES = [
  { value: 'tree', label: 'Family tree' },
  { value: 'time', label: 'Timeline' },
  { value: 'signal', label: 'Learning signal' },
];
const MODE_TEXT = {
  tree: 'Eleven families around one center. Lines show which method builds directly on which, often across families: follow them and you can trace today’s agents back to a 1958 neuron.',
  time: 'The same methods by the year of their founding paper, one lane per family. Recent years are stretched: half of this map was published after 2012. Press **Play history** to watch the field assemble.',
  signal: 'A different cut: grouped by where a method’s knowledge comes from. Hand-written rules, labeled examples, raw data that labels itself, reward, or evolution.',
};

// ---- layout helpers
const X0 = -18;
const X1 = 18;
const LANE = 2.5;
const yearX = (year) => {
  if (year < YEAR_MIN) return X0 - 2.2;
  const t = (year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN);
  return X0 + Math.pow(t, 2.4) * (X1 - X0);
};

function layouts() {
  const pos = { tree: {}, time: {}, signal: {} };
  const hubs = {};
  FAMILIES.forEach((f, i) => {
    const a = (i / FAMILIES.length) * Math.PI * 2 - Math.PI / 2;
    hubs[f.id] = new THREE.Vector3(Math.cos(a) * 9.5, 0, Math.sin(a) * 9.5);
    const mine = METHODS.filter((m) => m.family === f.id).sort((p, q) => p.year - q.year);
    mine.forEach((m, k) => {
      const row = Math.floor(k / 3);
      const b = a + ((k % 3) - 1) * 0.19 * (1 - row * 0.08);
      const r = 13.6 + row * 2.9;
      pos.tree[m.id] = new THREE.Vector3(Math.cos(b) * r, ((k % 2) * 2 - 1) * (0.5 + (row % 2) * 0.7), Math.sin(b) * r);
    });
    // timeline: one lane per family, stacking upward when two methods are close in time
    // (seen from above: methods that are close in time are spread across the width of the lane)
    const z = (i - (FAMILIES.length - 1) / 2) * LANE;
    const levels = [];
    mine.forEach((m) => {
      const x = yearX(m.year);
      let lvl = 0;
      while (levels[lvl] != null && x - levels[lvl] < 1.5) lvl++;
      levels[lvl] = x;
      pos.time[m.id] = new THREE.Vector3(x, 0.3, z + [0, 0.75, -0.75, 0.38, -0.38][lvl % 5]);
    });
  });
  SIGNALS.forEach((s, i) => {
    const a = (i / SIGNALS.length) * Math.PI * 2 - Math.PI / 2;
    const c = new THREE.Vector3(Math.cos(a) * 13, 0, Math.sin(a) * 13);
    hubs['signal:' + s.id] = c;
    METHODS.filter((m) => m.signal === s.id)
      .sort((p, q) => famIndex[p.family] - famIndex[q.family] || p.year - q.year)
      .forEach((m, k) => {
        const r = 1.5 + Math.sqrt(k) * 1.75;
        const b = k * 2.39996;
        pos.signal[m.id] = new THREE.Vector3(c.x + Math.cos(b) * r, ((k % 3) - 1) * 0.6, c.z + Math.sin(b) * r);
      });
  });
  return { pos, hubs };
}

export async function enter(ctx) {
  const { app, root } = ctx;
  const { pos, hubs } = layouts();
  const VIEW = {
    tree: { target: [0, 0, 0], dir: [0, 1.15, 1], fit: 24 },
    time: { target: [-1.2, 0, 0.6], dir: [0, 2.2, 1], fit: 22 },
    signal: { target: [0, 0, 0], dir: [0, 1.15, 1], fit: 22 },
  };
  app.setView({ pos: [0, 28, 24], target: [0, 0, 0], fit: 24, min: 5, max: 130, maxPolar: Math.PI * 0.6 });
  root.add(gridFloor({ size: 150, y: -2.4, color: '#3a2f66', opacity: 0.22 }));

  const S = { mode: 'tree', sel: null, hover: -1, off: new Set(), query: '', cursor: YEAR_MAX, playing: false };

  // ------------------------------------------------------------------ nodes
  const N = METHODS.length;
  const nodes = instanced(new THREE.SphereGeometry(1, 20, 14), new THREE.MeshBasicMaterial(), N);
  root.add(nodes);
  const P = METHODS.map((m) => ({ m, cur: pos.tree[m.id].clone().multiplyScalar(0.01), target: pos.tree[m.id], scale: 0, color: glow(FAMILIES[famIndex[m.family]].color, 1) }));
  const labels = P.map((p) => {
    const s = textSprite(p.m.name.replace(/ \(.*\)$/, ''), { size: 0.52 + p.m.importance * 0.08, color: '#dfe7f7', weight: 600, anchor: [0.5, 0], maxChars: 16, lineHeight: 1.06 });
    root.add(s);
    return s;
  });
  const matches = (m) => !S.off.has(m.family) && (!S.query || (m.name + ' ' + m.oneLine).toLowerCase().includes(S.query));
  const visible = (m) => S.mode !== 'time' || !S.playing || m.year <= S.cursor;

  // ------------------------------------------------------------------ links
  const pairs = METHODS.flatMap((m, i) => m.links.map((l) => [i, METHODS.indexOf(byId[l])]));
  const linkGeo = new THREE.BufferGeometry();
  linkGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs.length * 6), 3));
  const links = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: glow('#8fa3c8', 1), transparent: true, opacity: 0.2, depthWrite: false }));
  const hiGeo = new THREE.BufferGeometry();
  hiGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs.length * 6), 3));
  const hiLinks = new THREE.LineSegments(hiGeo, new THREE.LineBasicMaterial({ color: glow('#ffffff', 2.2), transparent: true, opacity: 0.95, depthWrite: false }));
  links.frustumCulled = hiLinks.frustumCulled = false;
  root.add(links, hiLinks);

  // ------------------------------------------------------------------ mode furniture
  const treeGroup = new THREE.Group();
  const timeGroup = new THREE.Group();
  const signalGroup = new THREE.Group();
  root.add(treeGroup, timeGroup, signalGroup);
  const core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.5, 1), glowMat('#cfe6ff', { k: 1.3 }));
  core.add(edges(new THREE.IcosahedronGeometry(2.3, 1), '#9d8cff', { k: 1.4, opacity: 0.7 }));
  const coreLabel = textSprite('Artificial\nIntelligence', { size: 0.62, font: 'display', weight: 700, color: '#eef3ff', lineHeight: 1.05 });
  coreLabel.position.y = 3.6;
  treeGroup.add(core, coreLabel);
  FAMILIES.forEach((f) => {
    const hub = hubs[f.id];
    const gem = new THREE.Mesh(new THREE.OctahedronGeometry(0.55), glowMat(f.color, { k: 2 }));
    gem.position.copy(hub);
    const lab = textSprite(f.name, { size: 0.62, font: 'display', weight: 700, color: f.color, anchor: [0.5, 0], maxChars: 16, lineHeight: 1.05, k: 1.05 });
    lab.position.copy(hub).setY(1.0);
    treeGroup.add(gem, lab, line([[0, 0, 0], hub], f.color, { k: 1, opacity: 0.35 }));
    ctx.pick.add(gem, { click: () => toggleOnly(f.id), tip: `${f.name}\n${f.blurb}` });
    // timeline lane
    const z = (famIndex[f.id] - (FAMILIES.length - 1) / 2) * LANE;
    timeGroup.add(line([[X0 - 3.2, 0, z], [X1 + 0.8, 0, z]], f.color, { k: 0.9, opacity: 0.28 }));
    const lane = textSprite(f.short, { size: 0.62, font: 'display', weight: 700, color: f.color, anchor: [0, 0.5], k: 1.05 });
    lane.position.set(X0 - 3.2, 0.2, z - 0.95);
    timeGroup.add(lane);
  });
  const zEdge = ((FAMILIES.length - 1) / 2) * LANE + 1.7;
  for (let y = 1950; y <= YEAR_MAX; y += y < 2010 ? 10 : 5) {
    const x = yearX(y);
    timeGroup.add(line([[x, -0.05, -zEdge], [x, -0.05, zEdge]], '#8fa3c8', { k: 0.8, opacity: 0.16 }));
    const t = textSprite(String(y), { size: 0.62, font: 'mono', weight: 600, color: '#93a1bf' });
    t.position.set(x, 0, zEdge + 0.8);
    timeGroup.add(t);
  }
  const before = textSprite('earlier', { size: 0.5, font: 'mono', weight: 600, color: '#93a1bf' });
  before.position.set(X0 - 2.2, 0, zEdge + 0.8);
  timeGroup.add(before);
  ERAS.forEach((e, i) => {
    const xa = yearX(e.from);
    const xb = yearX(e.to);
    const zz = -zEdge - 0.8 - (i % 2) * 1.3 - (e.id === 'foundation' ? 1.3 : 0);
    timeGroup.add(line([[xa, 0, zz], [xb, 0, zz]], '#ffe9a8', { k: 1.5, opacity: 0.8 }));
    const lab = textSprite(e.name, { size: 0.58, color: '#ffe9a8', weight: 600, anchor: [0, 0] });
    lab.position.set(xa, 0.15, zz);
    timeGroup.add(lab);
  });
  MILESTONES.forEach((ms) => {
    const lab = textSprite(`${ms.year} · ${ms.name}`, { size: 0.5, color: '#c3cde2', weight: 500, anchor: [0, 0.5] });
    lab.position.set(yearX(ms.year) - 1.5, 0.15, zEdge + 2.1);
    timeGroup.add(lab);
  });
  const cursorLine = line([[0, 0, -zEdge], [0, 6, -zEdge], [0, 6, zEdge], [0, 0, zEdge]], '#ffffff', { k: 2, opacity: 0.8 });
  const cursorLabel = textSprite('1943', { size: 1.6, font: 'display', weight: 700, color: '#eef3ff', onTop: true });
  timeGroup.add(cursorLine, cursorLabel);
  SIGNALS.forEach((s) => {
    const c = hubs['signal:' + s.id];
    const lab = textSprite(s.name, { size: 0.8, font: 'display', weight: 700, color: '#eef3ff', anchor: [0.5, 0] });
    lab.position.copy(c).setY(2.6);
    signalGroup.add(lab);
  });

  // ------------------------------------------------------------------ HUD
  const modeText = h('div');
  const chipsEl = h('div.chips.fam-chips');
  const search = h('input.fam-search', { type: 'text', placeholder: 'Search 70 methods…', 'aria-label': 'Search methods' });
  search.addEventListener('input', () => {
    S.query = search.value.trim().toLowerCase();
    renderMatches();
  });
  const matchEl = h('div.chips');
  const playBtn = button('▶ Play history', { onClick: () => togglePlay() });
  const modeSeg = segmented(MODES, S.mode, (v) => setMode(v));

  function renderChips() {
    chipsEl.replaceChildren(
      ...FAMILIES.map((f) => {
        const c = h('span.chip' + (S.off.has(f.id) ? '.off' : ''), { style: { '--c': f.color }, title: f.blurb }, h('i'), f.short);
        c.addEventListener('click', () => {
          if (S.off.has(f.id)) S.off.delete(f.id);
          else S.off.add(f.id);
          renderChips();
          renderMatches();
        });
        return c;
      }),
    );
  }
  function toggleOnly(fid) {
    const only = S.off.size === FAMILIES.length - 1 && !S.off.has(fid);
    S.off = new Set(only ? [] : FAMILIES.filter((f) => f.id !== fid).map((f) => f.id));
    renderChips();
    renderMatches();
  }
  function renderMatches() {
    const list = METHODS.filter(matches);
    const show = S.query || S.off.size ? list : [];
    matchEl.replaceChildren(
      ...show.slice(0, 40).map((m) => {
        const c = h('span.chip.click', { style: { '--c': FAMILIES[famIndex[m.family]].color } }, m.name.replace(/ \(.*\)$/, ''));
        c.addEventListener('click', () => select(m.id));
        return c;
      }),
    );
  }

  function card(m) {
    const f = FAMILIES[famIndex[m.family]];
    const sig = SIGNALS.find((s) => s.id === m.signal);
    const chipFor = (id) => {
      const o = byId[id];
      const c = h('span.chip.click', { style: { '--c': FAMILIES[famIndex[o.family]].color } }, o.name.replace(/ \(.*\)$/, ''));
      c.addEventListener('click', () => select(id));
      return c;
    };
    const later = METHODS.filter((o) => o.links.includes(m.id));
    const labs = [].concat(m.lab || []);
    return [
      h(
        'div.fam-card',
        h('h2', m.name),
        h('div.meta', h('span', h('span.sw', { style: { '--c': f.color } }), ' ', h('b', f.name)), h('span', 'first published ', h('b', m.year < 0 ? 'antiquity' : String(m.year))), h('span', 'learns from ', h('b', sig.name.toLowerCase()))),
        h('p.one', m.oneLine),
      ),
      section('How it works', text(m.how)),
      section('Strengths', text(m.strengths)),
      section('Limits', text(m.limits)),
      m.signalNote && callout(m.signalNote, 'info'),
      section('Where it comes from', text(m.origin, 'small'), h('ul', m.famous.map((x) => h('li', x)))),
      m.links.length ? section('Builds on', h('div.chips', m.links.map(chipFor))) : null,
      later.length ? section('Led to', h('div.chips', later.map((o) => chipFor(o.id)))) : null,
      labs.length ? h('div.btnrow', labs.map((l) => button(`Try it: ${LAB_NAMES[l]} →`, { kind: 'primary', onClick: () => ctx.router.go(l) }))) : null,
    ];
  }
  function overview() {
    return [
      section(
        'The eleven families',
        h(
          'div.fam-legend',
          FAMILIES.map((f) => {
            const row = h('div.row', { style: { '--c': f.color } }, h('i'), h('span', h('b', f.name), ' · ', f.blurb));
            row.addEventListener('click', () => toggleOnly(f.id));
            return row;
          }),
        ),
      ),
      text('Click a sphere for the method behind it. Sphere size is an editorial judgment of how central the method is today.', 'small'),
    ];
  }

  function select(id, fly = true) {
    S.sel = id && byId[id] ? id : null;
    ctx.router.replace('families' + (S.sel ? '/' + S.sel : ''));
    ctx.ui.right(S.sel ? card(byId[S.sel]) : overview());
    if (S.sel && fly) {
      const p = P[METHODS.indexOf(byId[S.sel])].target;
      app.frame({ target: [p.x, p.y, p.z], dir: VIEW[S.mode].dir, fit: 9 }, 1.1);
    }
  }

  function setMode(mode) {
    S.mode = mode;
    S.playing = false;
    S.cursor = YEAR_MAX;
    modeSeg.set(mode);
    P.forEach((p) => (p.target = pos[mode][p.m.id]));
    modeText.replaceChildren(text(MODE_TEXT[mode]).firstChild);
    playBtn.hidden = mode !== 'time';
    playBtn.textContent = '▶ Play history';
    if (S.sel) {
      const p = P[METHODS.indexOf(byId[S.sel])].target;
      app.frame({ target: [p.x, p.y, p.z], dir: VIEW[mode].dir, fit: mode === 'time' ? 12 : 9 }, 1.4);
    } else app.frame(VIEW[mode], 1.4);
    app.setHome(app._fitPos({ ...app._home, ...VIEW[mode], pos: [VIEW[mode].target[0] + VIEW[mode].dir[0], VIEW[mode].target[1] + VIEW[mode].dir[1], VIEW[mode].target[2] + VIEW[mode].dir[2]] }), VIEW[mode].target);
  }
  function togglePlay() {
    S.playing = !S.playing;
    if (S.playing) {
      S.cursor = YEAR_MIN - 6;
      select(null, false);
      app.frame(VIEW.time, 1);
    } else S.cursor = YEAR_MAX;
    playBtn.textContent = S.playing ? '❚❚ Stop' : '▶ Play history';
  }

  ctx.ui.left(
    header({ kicker: 'Chapter 01 · Families', title: 'Every kind of AI, on one map', lead: `**${N} methods** in ${FAMILIES.length} families, from 1950s search to 2026 agents. Three ways to arrange them.` }),
    modeText,
    search,
    section('Families (click to filter)', chipsEl),
    matchEl,
    h('div.btnrow', button('Next: open the machines →', { kind: 'ghost', onClick: () => ctx.router.go('labs') })),
  );
  ctx.ui.dock(h('div.dockrow', modeSeg, playBtn));
  ctx.pick.add(nodes, {
    filter: (hit) => P[hit.instanceId].scale > 0.05,
    hover: (hit) => (S.hover = hit.instanceId),
    move: (hit) => (S.hover = hit.instanceId),
    out: () => (S.hover = -1),
    click: (hit) => select(METHODS[hit.instanceId].id),
    tip: (hit) => {
      const m = METHODS[hit.instanceId];
      return `${m.name} · ${m.year}\n${m.oneLine}`;
    },
  });
  ctx.pick.onMiss(() => S.sel && select(null, false));
  ctx.key('Escape', () => select(null, false));

  renderChips();
  renderMatches();
  setMode('tree');
  select(ctx.route.rest[0] || null);

  const m4 = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const sv = new THREE.Vector3();
  const col = new THREE.Color();
  return {
    onRoute: (route) => (route.rest[0] || null) !== S.sel && select(route.rest[0] || null),
    update(dt, t) {
      const k = 1 - Math.exp(-dt * 4.5);
      if (S.playing) {
        S.cursor += dt * 6.2;
        if (S.cursor > YEAR_MAX + 3) togglePlay();
      }
      const selIdx = S.sel ? METHODS.indexOf(byId[S.sel]) : -1;
      const near = new Set(selIdx >= 0 ? [selIdx, ...pairs.filter((p) => p[0] === selIdx || p[1] === selIdx).flat()] : []);
      // level of detail for names: the big ones always, the rest as you zoom in or filter
      const dist = app.camera.position.distanceTo(app.controls.target);
      const lod = S.query || S.off.size ? 1 : dist < 34 ? 1 : dist < 52 ? 2 : 3;
      P.forEach((p, i) => {
        p.cur.lerp(p.target, k);
        const on = matches(p.m);
        const want = visible(p.m) ? SIZE[p.m.importance] * (i === S.hover || i === selIdx ? 1.35 : 1) : 0;
        p.scale += (want - p.scale) * Math.min(1, dt * 9);
        m4.compose(p.cur, q, sv.setScalar(Math.max(p.scale, 1e-4)));
        nodes.setMatrixAt(i, m4);
        const dim = !on || (selIdx >= 0 && !near.has(i));
        nodes.setColorAt(i, col.copy(p.color).multiplyScalar(i === selIdx ? 3 : dim ? 0.22 : 1.5 + 0.25 * Math.sin(t * 1.3 + i)));
        const lab = labels[i];
        lab.position.copy(p.cur).y += SIZE[p.m.importance] + 0.18;
        lab.visible = p.scale > 0.1 && (i === S.hover || (!dim && (p.m.importance >= lod || near.has(i))));
        lab.material.opacity = dim ? 0.45 : 1;
      });
      flushInst(nodes);
      const lp = linkGeo.attributes.position.array;
      const hp = hiGeo.attributes.position.array;
      let hn = 0;
      pairs.forEach(([a, b], i) => {
        const show = P[a].scale > 0.05 && P[b].scale > 0.05 && matches(P[a].m) && matches(P[b].m);
        (show ? P[a].cur : P[b].cur).toArray(lp, i * 6);
        P[b].cur.toArray(lp, i * 6 + 3);
        if (show && (a === selIdx || b === selIdx)) {
          P[a].cur.toArray(hp, hn * 6);
          P[b].cur.toArray(hp, hn * 6 + 3);
          hn++;
        }
      });
      linkGeo.attributes.position.needsUpdate = true;
      hiGeo.attributes.position.needsUpdate = true;
      hiGeo.setDrawRange(0, hn * 2);
      // fade the furniture of each arrangement in and out
      for (const [g, mode] of [[treeGroup, 'tree'], [timeGroup, 'time'], [signalGroup, 'signal']]) {
        const want = S.mode === mode ? 1 : 0;
        g.userData.k = (g.userData.k ?? 0) + (want - (g.userData.k ?? 0)) * Math.min(1, dt * 5);
        g.visible = g.userData.k > 0.02;
        g.scale.setScalar(0.4 + 0.6 * g.userData.k);
      }
      core.rotation.y = t * 0.2;
      cursorLine.visible = cursorLabel.visible = S.playing;
      if (S.playing) {
        const x = yearX(Math.max(YEAR_MIN, Math.min(YEAR_MAX, S.cursor)));
        cursorLine.position.x = x;
        cursorLabel.position.set(x, 1.5, -zEdge - 4.6);
        cursorLabel.userData.setText(String(Math.round(Math.max(YEAR_MIN, Math.min(YEAR_MAX, S.cursor)))));
      }
    },
  };
}
