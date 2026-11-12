// Hot 10: the ten hottest issues in AI as of HOT_AS_OF (content: ../data/hot.js).
// Overview: a spiral ramp that climbs to rank 1, with one monolith per issue. Opening an issue
// (#/hot/<id>) flies the camera to its monolith, where the issue's vignette is built and played;
// the vignettes themselves live in ./hot-vignettes.js.
import * as THREE from 'three';
import './hot.css';
import { HOT, HOT_AS_OF, ALSO_WATCHING } from '../data/hot.js';
import { glow } from '../core/colors.js';
import { glowMat, stdMat, segments, edges, rbox, halo, points, gridFloor, disposeDeep } from '../core/geom.js';
import { textPlane, FONTS } from '../core/labels.js';
import { h, header, section, text, callout, rich, button } from '../core/ui.js';
import { damp, clamp, ease } from '../core/tween.js';
import { VIGNETTES } from './hot-vignettes.js';

// ---------------------------------------------------------------------------- text helpers
// The data file is written in British English and the site speaks American English. This changes
// spelling only: no fact or figure is touched, and none of these words occurs inside a quotation.
const US_SPELLING = {
  organisation: 'organization', centre: 'center', weaponise: 'weaponize', reorganised: 'reorganized',
  recognise: 'recognize', quantisation: 'quantization', 'pre-emption': 'preemption', practise: 'practice',
  medallists: 'medalists', licence: 'license', labelled: 'labeled', labelling: 'labeling', defence: 'defense',
  offence: 'offense', criticising: 'criticizing', catalogue: 'catalog', authorised: 'authorized',
  authorisation: 'authorization',
};
const US_RE = new RegExp(Object.keys(US_SPELLING).join('|'), 'gi');
const us = (s) =>
  String(s).replace(US_RE, (m) => {
    const r = US_SPELLING[m.toLowerCase()];
    return m[0] === m[0].toUpperCase() ? r[0].toUpperCase() + r.slice(1) : r;
  });

const host = (url) => {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
};
const srcLink = (url) => h('a', { href: url, target: '_blank', rel: 'noopener' }, host(url), h('span.hot-ext', ' ↗'));
const two = (n) => String(n).padStart(2, '0');

const measure = document.createElement('canvas').getContext('2d');
/** Break `str` into the fewest lines no wider than `maxEm`, then make the lines as even as possible. */
function wrapEven(str, font, weight, maxEm) {
  measure.font = `${weight} 100px ${FONTS[font]}`;
  const words = str.split(' ');
  const w = words.map((x) => measure.measureText(x).width / 100);
  const space = measure.measureText(' ').width / 100;
  const n = words.length;
  const best = new Array(n + 1);
  best[n] = { lines: 0, widest: 0, next: n };
  for (let i = n - 1; i >= 0; i--) {
    let run = 0;
    for (let j = i + 1; j <= n; j++) {
      run += w[j - 1] + (j > i + 1 ? space : 0);
      if (run > maxEm && j > i + 1) break;
      const rest = best[j];
      const cand = { lines: rest.lines + 1, widest: Math.max(run, rest.widest), next: j };
      const cur = best[i];
      if (!cur || cand.lines < cur.lines || (cand.lines === cur.lines && cand.widest < cur.widest)) best[i] = cand;
    }
  }
  const out = [];
  for (let i = 0; i < n; i = best[i].next) out.push(words.slice(i, best[i].next).join(' '));
  return out.join('\n');
}

// ---------------------------------------------------------------------------- the tower
// Rank colours run from white-hot gold (1) to cool violet (10). They are decoration: the rank
// itself is always written out.
const HEAT = ['#ffd98a', '#ffbe55', '#ff9f52', '#ff8259', '#ff6f78', '#ff6fa6', '#f273cf', '#cf82f2', '#ad8cff', '#8f9cff'];

const TAU = Math.PI * 2;
const TURNS = 3;
const PHI_END = TURNS * TAU + 1.03; // angle at which the ramp reaches the ground
const R_MAX = 16.2; // radius of the ramp at the ground
const DEPTH = 0.42; // the spiral is an ellipse in plan, so that the top is not far behind the front row
const HEIGHT = 10.6; // height of the summit
const PITCH = 4.55; // horizontal distance between neighbouring monoliths, seen from the front
const SLAB_W = 3.6;
const SLAB_H = 5.05;
// where each rank stands, seen from the front: 1 / 2 3 / 4 5 6 / 7 8 9 10  -> [turn of the ramp, column]
const SLOTS = [[0, 0], [1, -1], [1, 1], [2, -2], [2, 0], [2, 2], [3, -3], [3, -1], [3, 1], [3, 3]];

const radiusAt = (phi) => R_MAX * Math.pow(Math.max(phi, 0) / PHI_END, 0.78);
const heightAt = (phi) => HEIGHT * (1 - phi / PHI_END);
const onRamp = (phi, out = new THREE.Vector3(), side = 0) => {
  const r = radiusAt(phi) + side;
  return out.set(r * Math.sin(phi), heightAt(phi), DEPTH * r * Math.cos(phi));
};
/** Angle on turn `turn` of the ramp whose point lies `column * PITCH` to the right of the axis. */
function slotAngle(turn, column) {
  if (turn === 0) return 0;
  let lo = -1.5;
  let hi = 1.5;
  for (let i = 0; i < 40; i++) {
    const a = (lo + hi) / 2;
    if (radiusAt(turn * TAU + a) * Math.sin(a) < column * PITCH) lo = a;
    else hi = a;
  }
  return turn * TAU + (lo + hi) / 2;
}
const heatAt = (u, out = new THREE.Color()) => {
  const x = clamp(u) * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(x));
  return out.set(HEAT[i]).lerp(_c.set(HEAT[i + 1]), x - i);
};
const _c = new THREE.Color();

function buildOverview(ctx, { onOpen, onHover }) {
  const group = new THREE.Group();
  const small = window.innerWidth < 700;

  const floor = gridFloor({ size: 150, y: -0.7, color: '#6a4a9c', opacity: 0.3, fade: 46, step: 2 });
  group.add(floor);

  // ---- the ramp: a ribbon with two bright edges, coloured by height
  const N = 420;
  const phi0 = 0.9;
  const phi1 = PHI_END + 0.55;
  const strip = new Float32Array((N + 1) * 6);
  const stripCol = new Float32Array((N + 1) * 6);
  const edgeA = new Float32Array((N + 1) * 3);
  const edgeB = new Float32Array((N + 1) * 3);
  const edgeCol = new Float32Array((N + 1) * 3);
  const v = new THREE.Vector3();
  const col = new THREE.Color();
  for (let i = 0; i <= N; i++) {
    const phi = phi0 + ((phi1 - phi0) * i) / N;
    const half = Math.min(0.85, 0.2 + radiusAt(phi) * 0.07);
    onRamp(phi, v, -half).toArray(strip, i * 6);
    v.toArray(edgeA, i * 3);
    onRamp(phi, v, half).toArray(strip, i * 6 + 3);
    v.toArray(edgeB, i * 3);
    heatAt(phi / PHI_END, col);
    stripCol.set([col.r, col.g, col.b, col.r, col.g, col.b], i * 6);
    const e = 1.25 - 0.45 * (phi / PHI_END);
    edgeCol.set([col.r * e, col.g * e, col.b * e], i * 3);
  }
  const idx = [];
  for (let i = 0; i < N; i++) idx.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
  const stripGeo = new THREE.BufferGeometry();
  stripGeo.setIndex(idx);
  stripGeo.setAttribute('position', new THREE.BufferAttribute(strip, 3));
  stripGeo.setAttribute('color', new THREE.BufferAttribute(stripCol, 3));
  const stripMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.17, side: THREE.DoubleSide, depthWrite: false });
  group.add(new THREE.Mesh(stripGeo, stripMat));
  const edgeMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85, depthWrite: false });
  for (const arr of [edgeA, edgeB]) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(edgeCol, 3));
    group.add(new THREE.Line(geo, edgeMat));
  }

  // ---- sparks climbing the ramp towards rank 1
  const P = ctx.quality === 'low' ? 40 : 84;
  const ppos = new Float32Array(P * 3);
  const pcol = new Float32Array(P * 3);
  const psize = new Float32Array(P);
  const sparks = Array.from({ length: P }, (_, i) => ({ u: i / P, side: (Math.random() - 0.5) * 0.9, speed: 0.016 + Math.random() * 0.012 }));
  const sparkPts = points(ppos, pcol, psize);
  group.add(sparkPts);

  // ---- monoliths
  const slabGeo = rbox(SLAB_W, SLAB_H, 0.26, 0.09);
  const rimGeo = new THREE.BoxGeometry(SLAB_W + 0.02, SLAB_H + 0.02, 0.28);
  const hitGeo = new THREE.BoxGeometry(SLAB_W + 0.3, SLAB_H + 2.3, 1.2);
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });
  const stilts = [];
  const steles = HOT.map((issue, i) => {
    const rank = i + 1;
    const color = HEAT[i];
    const size = rank === 1 ? 1.28 : rank <= 3 ? 1.08 : 1;
    const phi = slotAngle(...SLOTS[i]);
    const g = new THREE.Group(); // turns to face the camera
    onRamp(phi, g.position);
    const body = new THREE.Group(); // grows and shrinks
    g.add(body);
    stilts.push(g.position.clone(), g.position.clone().setY(-0.7));

    const slab = new THREE.Mesh(slabGeo, stdMat('#0a101e', { rough: 0.42, metal: 0.4, emissive: color, ek: 0.06 }));
    slab.position.y = SLAB_H / 2;
    const rim = edges(rimGeo, color, { k: 1.5 });
    rim.position.y = SLAB_H / 2;
    const z = 0.15;
    const numeral = textPlane(String(rank), { size: 2.25, font: 'display', weight: 700, color: '#f7f9ff' });
    numeral.position.set(0, SLAB_H - 1.2, z);
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.05, 0.02), glowMat(color, { k: 2.1 }));
    bar.position.set(0, SLAB_H - 2.28, z);
    const inner = SLAB_W - 0.36;
    const title = textPlane(wrapEven(us(issue.title), 'display', 650, inner / 0.4), { size: 0.4, font: 'display', weight: 650, color: '#eef3ff', anchor: [0.5, 1], lineHeight: 1.14 });
    title.position.set(0, SLAB_H - 2.4, z);
    const titleH = title.geometry.parameters.height;
    const tag = textPlane(wrapEven(us(issue.tag), 'body', 500, inner / 0.285), { size: 0.285, weight: 500, color: '#a7b4cf', anchor: [0.5, 1], lineHeight: 1.18 });
    tag.position.set(0, SLAB_H - 2.4 - titleH + 0.02, z);
    for (const t of [title, tag]) {
      const wide = t.geometry.parameters.width;
      if (wide > SLAB_W - 0.12) t.scale.setScalar((SLAB_W - 0.12) / wide);
    }
    if (small) title.visible = tag.visible = false; // unreadable at phone size: the list in the sheet carries them

    const emblem = VIGNETTES[issue.viz].emblem(color);
    emblem.position.y = SLAB_H + 1.18;
    emblem.scale.setScalar(0.94);
    const plate = new THREE.Mesh(new THREE.TorusGeometry(0.86, 0.016, 6, 56), glowMat(color, { k: 1.5 }));
    plate.rotation.x = Math.PI / 2;
    plate.position.y = SLAB_H + 0.17;
    const aura = halo(color, 3.6, 0.13);
    aura.position.y = SLAB_H + 1.1;
    body.add(slab, rim, numeral, bar, title, tag, emblem, plate, aura);

    const hit = new THREE.Mesh(hitGeo, hitMat);
    hit.position.y = (SLAB_H + 2.3) / 2;
    g.add(hit);
    const tip = h('div', h('b', `${rank} · ${us(issue.title)}`), h('small', us(issue.hook)));
    ctx.pick.add(hit, { hover: () => onHover(i), out: () => onHover(-1), click: () => onOpen(i), tip });

    group.add(g);
    return { g, body, rim, emblem, aura, size, color: glow(color, 1), k: 1, hover: 0, wait: 0, phase: i * 1.7 };
  });
  group.add(segments(stilts, '#8f7bd0', { k: 0.8, opacity: 0.22 }));

  // summit: a small landing under rank 1 and a beacon behind it
  const summit = new THREE.Mesh(new THREE.CylinderGeometry(2.5, 2.9, 0.18, 48), stdMat('#141a2e', { emissive: HEAT[0], ek: 0.18, rough: 0.5 }));
  summit.position.y = HEIGHT - 0.1;
  const beacon = halo(HEAT[0], 15, 0.2);
  beacon.position.set(0, HEIGHT + 4.2, -1.2);
  group.add(summit, beacon);

  let shown = true;
  let fade = 1; // 0..1 for everything that is not a monolith
  let hovered = -1;
  const anchors = steles.map((s) => s.g.position.clone());

  return {
    group,
    /** Where the vignette of issue i is staged. */
    anchor: (i) => anchors[i],
    setHover(i) {
      hovered = i;
    },
    setShown(on, snap, focus = -1) {
      shown = on;
      steles.forEach((s, i) => {
        s.wait = snap ? 0 : on ? 0.25 + i * 0.035 : i === focus ? 0.1 : 0;
        if (snap) s.k = on ? 1 : 0;
      });
      if (snap) fade = on ? 1 : 0;
    },
    update(dt, t, camera) {
      fade = damp(fade, shown ? 1 : 0, shown ? 0.3 : 0.12, dt);
      const live = fade > 0.01 || steles.some((s) => s.k > 0.01);
      group.visible = live;
      if (!live) return;
      stripMat.opacity = 0.17 * fade;
      edgeMat.opacity = 0.85 * fade;
      floor.material.uniforms.uOpacity.value = 0.3 * fade;
      sparkPts.material.uniforms.uOpacity.value = fade;
      beacon.material.opacity = (0.17 + 0.05 * Math.sin(t * 1.3)) * fade;
      summit.scale.setScalar(Math.max(0.001, fade));

      for (let i = 0; i < P; i++) {
        const s = sparks[i];
        s.u -= dt * s.speed;
        if (s.u < 0) s.u += 1;
        const phi = phi0 + (phi1 - phi0) * s.u;
        onRamp(phi, v, s.side * Math.min(0.85, 0.2 + radiusAt(phi) * 0.07));
        ppos[i * 3] = v.x;
        ppos[i * 3 + 1] = v.y + 0.08;
        ppos[i * 3 + 2] = v.z;
        heatAt(phi / PHI_END, col).multiplyScalar(2.4);
        pcol[i * 3] = col.r;
        pcol[i * 3 + 1] = col.g;
        pcol[i * 3 + 2] = col.b;
        psize[i] = 0.12 + 0.1 * (1 - s.u);
      }
      const geo = sparkPts.geometry;
      geo.attributes.position.needsUpdate = geo.attributes.color.needsUpdate = geo.attributes.size.needsUpdate = true;

      steles.forEach((s, i) => {
        if (s.wait > 0) s.wait -= dt;
        else s.k = damp(s.k, shown ? 1 : 0, shown ? 0.16 : 0.09, dt);
        s.g.visible = s.k > 0.01;
        if (!s.g.visible) return;
        s.hover = damp(s.hover, hovered === i ? 1 : 0, 0.1, dt);
        s.body.scale.setScalar(s.size * ease.out(clamp(s.k)) * (1 + 0.05 * s.hover));
        s.g.rotation.y = Math.atan2(camera.position.x - s.g.position.x, camera.position.z - s.g.position.z);
        s.rim.material.color.copy(s.color).multiplyScalar(1.45 + 1.5 * s.hover);
        s.aura.material.opacity = 0.12 + 0.2 * s.hover;
        s.emblem.position.y = SLAB_H + 1.18 + Math.sin(t * 0.9 + s.phase) * 0.06;
        s.emblem.rotation.y = Math.sin(t * 0.45 + s.phase) * 0.55;
        s.emblem.userData.tick?.(t + s.phase, dt);
      });
    },
  };
}

// camera framing of the whole tower: looking straight at it, slightly from above
const TOWER_VIEW = { dir: [0, 0.2, 1], target: [0, 8.1, 2.2], fit: 10.9, wide: 1.36 };

/** Width / height of the part of the viewport that the panels leave free (the engine frames a sphere, the tower is wider than tall). */
function freeAspect() {
  const w = window.innerWidth;
  const hgt = window.innerHeight;
  const hud = document.getElementById('hud');
  const open = (el) => el && !el.hidden && !el.classList.contains('collapsed') && !hud.classList.contains('no-panels');
  const L = hud?.querySelector('.panel.left');
  const R = hud?.querySelector('.panel.right');
  if (w > 860) return (w - (open(L) ? L.offsetWidth + 24 : 0) - (open(R) ? R.offsetWidth + 24 : 0)) / (hgt - 140);
  const sheet = hud?.dataset.sheet === 'left' ? L : hud?.dataset.sheet === 'right' ? R : null;
  return w / Math.max(120, hgt - 170 - (open(sheet) ? sheet.offsetHeight : 0));
}

// ---------------------------------------------------------------------------- chapter
export async function enter(ctx) {
  const { app, root } = ctx;
  app.setBloom({ strength: 0.56, radius: 0.7 });
  app.setFog(0.0032);

  const indexOf = (route) => {
    const key = route.rest[0];
    if (!key) return -1;
    const byId = HOT.findIndex((x) => x.id === key);
    if (byId >= 0) return byId;
    const n = Number(key);
    return Number.isInteger(n) && n >= 1 && n <= HOT.length ? n - 1 : -1;
  };
  const pathOf = (i) => (i < 0 ? 'hot' : 'hot/' + HOT[i].id);
  const open = (i) => ctx.router.go(pathOf(i));

  let cards = [];
  const overview = buildOverview(ctx, {
    onOpen: open,
    onHover: (i) => {
      overview.setHover(i);
      cards.forEach((c, j) => c.classList.toggle('on', j === i));
    },
  });
  root.add(overview.group);

  let current = -2; // -1 = overview, 0..9 = issue
  let active = null; // { index, vig, k }
  const leaving = [];

  // ---- camera: let the engine frame the view (it knows the panels), then fly there
  function frame(view, at, pose) {
    const target = [at.x + view.target[0], at.y + view.target[1], at.z + view.target[2]];
    const pos = [target[0] + view.dir[0], target[1] + view.dir[1], target[2] + view.dir[2]];
    const fit = view.wide ? view.fit * Math.max(1, view.wide / freeAspect()) : view.fit;
    app.setView({ pos, target, fit, min: 2.5, max: 190, maxPolar: Math.PI * (view.maxPolar ?? 0.56) });
    if (!pose || ctx.reduceMotion) return;
    const dest = app.camera.position.clone();
    app.camera.position.copy(pose.position);
    app.controls.target.copy(pose.target);
    app.flyTo(dest, target, pose.seconds);
  }

  // ---- panels
  function overviewPanels() {
    cards = HOT.map((issue, i) =>
      h(
        'a.card.click.hot-card',
        {
          href: '#/' + pathOf(i),
          style: { '--c': HEAT[i] },
          onmouseenter: () => overview.setHover(i),
          onmouseleave: () => overview.setHover(-1),
          onfocus: () => overview.setHover(i),
          onblur: () => overview.setHover(-1),
        },
        h('span.rk', String(i + 1)),
        h('h4', us(issue.title)),
        h('span.tag', us(issue.tag)),
      ),
    );
    ctx.ui.left(
      header({
        kicker: `Chapter 08 · Hot 10 · as of ${HOT_AS_OF}`,
        title: 'The ten hottest issues in AI',
        lead: `Where the field is under the most pressure, as of **${HOT_AS_OF}**. Each issue has its own live 3D scene: pick one to fly in.`,
      }),
      h(
        'div.hot-note',
        rich(
          `**How the ranking was made.** Issues are ordered by how much attention and activity they drew in the weeks before ${HOT_AS_OF}. The order is an editorial judgment. The facts behind it are not: every one carries its date and a link to its source, with figures quoted as the source reported them.`,
        ),
      ),
      section('The ranking', h('ol.hot-list', cards.map((c) => h('li', c)))),
      section(
        'Also watching',
        text('Fundamental problems that were quieter in the news this fall but have not gone away.', 'small'),
        ALSO_WATCHING.map((a) => h('div.hot-also', h('h4', us(a.title)), h('p', us(a.text)), h('div.src', 'Source: ', srcLink(a.src)))),
      ),
    );
    ctx.ui.right();
    ctx.ui.dock(
      h(
        'div.hot-start',
        h('span.hint', 'Click a monolith or a card to fly in · ← → step through the ranking'),
        button('Start at number 1 →', { kind: 'primary', onClick: () => open(0) }),
      ),
    );
  }

  function issuePanels(i, vig) {
    cards = [];
    const issue = HOT[i];
    const rank = i + 1;
    ctx.ui.left(
      h(
        'header.ph.hot-head',
        h('div.kicker', h('a', { href: '#/hot' }, '← Hot 10'), h('span', `as of ${HOT_AS_OF}`)),
        h('div.hot-rank', h('span.n', two(rank)), h('span.of', '/ 10')),
        h('h1', us(issue.title)),
        h('span.hot-tag', { style: { '--c': HEAT[i] } }, us(issue.tag)),
        h('div.lead', rich(us(issue.hook))),
      ),
      section('What is at stake', text(us(issue.what))),
      callout(us(issue.mech), 'key', 'The mechanism'),
      h('a.btn.primary.hot-link', { href: issue.link.href }, issue.link.label + ' →'),
    );
    ctx.ui.right(
      vig.panel,
      section(
        'Dated facts',
        issue.facts.map((f) => h('div.hot-fact', h('span.when', f.when), h('p', us(f.text)), h('div.src', 'Source: ', srcLink(f.src)))),
      ),
      section('Approaches', h('ul.hot-ul', issue.approaches.map((a) => h('li', us(a))))),
      section('Open questions', h('ul.hot-ul.q', issue.open.map((q) => h('li', us(q))))),
    );

    const nav = (j, label, kind) =>
      h('button.btn.nav' + kind, { type: 'button', title: j < 0 || j >= HOT.length ? 'Back to all ten' : `${j + 1}. ${us(HOT[j].title)}`, onclick: () => open(j >= HOT.length ? -1 : j), disabled: j < 0 }, label);
    ctx.ui.dock(
      h(
        'div.tour.hot-dock',
        nav(i - 1, ['←', i > 0 && h('b', String(rank - 1))], '.ghost'),
        h(
          'button.btn.ghost.all',
          { type: 'button', title: 'Back to all ten (Esc)', 'aria-label': 'Back to all ten', onclick: () => open(-1) },
          h('span.ico', { html: '<svg viewBox="0 0 16 16"><rect x="1" y="1" width="6" height="6" rx="1.4"/><rect x="9" y="1" width="6" height="6" rx="1.4"/><rect x="1" y="9" width="6" height="6" rx="1.4"/><rect x="9" y="9" width="6" height="6" rx="1.4"/></svg>' }),
          h('span', 'All ten'),
        ),
        h(
          'div.mid',
          h('span.t', `${rank} / ${HOT.length} · ${us(issue.title)}`),
          h(
            'div.dots',
            HOT.map((x, j) =>
              h('button.dot' + (j === i ? '.on' : j < i ? '.done' : ''), { type: 'button', title: `${j + 1}. ${us(x.title)}`, 'aria-label': `Issue ${j + 1}: ${us(x.title)}`, onclick: () => open(j) }),
            ),
          ),
        ),
        nav(i + 1, i + 1 < HOT.length ? [h('b', String(rank + 1)), '→'] : ['All ten', '→'], '.primary'),
      ),
    );
  }

  // ---- switching between the overview and an issue
  function show(i, snap = false) {
    if (i === current) return;
    const far = current < 0 || i < 0;
    const pose = snap ? null : { position: app.camera.position.clone(), target: app.controls.target.clone(), seconds: far ? 1.7 : 1.35 };
    if (active) {
      leaving.push(active);
      active = null;
    }
    current = i;
    overview.setHover(-1);
    if (i < 0) {
      overview.setShown(true, snap);
      overviewPanels();
      frame(TOWER_VIEW, new THREE.Vector3(), pose);
      document.title = 'Hot 10 · AI Atlas 3D';
      return;
    }
    const issue = HOT[i];
    const def = VIGNETTES[issue.viz];
    overview.setShown(false, snap, i);
    const vig = def.build({ ctx, issue, low: ctx.quality === 'low' });
    vig.group.position.copy(overview.anchor(i));
    vig.group.scale.setScalar(snap ? 1 : 0.001);
    root.add(vig.group);
    active = { index: i, vig, k: snap ? 1 : 0, wait: snap ? 0 : 0.3 };
    issuePanels(i, vig);
    frame(def.view, overview.anchor(i), pose);
    document.title = `${i + 1}. ${us(issue.title)} · Hot 10 · AI Atlas 3D`;
  }

  function discard(item) {
    root.remove(item.vig.group);
    item.vig.dispose?.();
    disposeDeep(item.vig.group);
  }

  const step = (d) => {
    const next = current + d;
    if (current < 0) open(d > 0 ? 0 : HOT.length - 1);
    else open(next >= HOT.length ? -1 : next);
  };
  ctx.key('ArrowRight', () => step(1));
  ctx.key('ArrowLeft', () => step(-1));
  ctx.key('ArrowUp', () => open(-1));
  ctx.key('Escape', () => {
    const help = document.querySelector('#hud .help');
    if (help && !help.hidden) help.hidden = true;
    else if (current >= 0) open(-1);
  });
  ctx.onDispose(() => {
    for (const item of leaving) item.vig.dispose?.();
    active?.vig.dispose?.();
  });

  const first = indexOf(ctx.route);
  if (ctx.route.rest[0] && first < 0) ctx.router.replace('hot');
  show(first, true);

  return {
    update(dt, t) {
      overview.update(dt, t, app.camera);
      for (let j = leaving.length - 1; j >= 0; j--) {
        const item = leaving[j];
        item.k = damp(item.k, 0, 0.07, dt);
        item.vig.group.scale.setScalar(Math.max(0.001, item.k));
        if (item.k < 0.03) {
          discard(item);
          leaving.splice(j, 1);
        }
      }
      if (active) {
        if (active.wait > 0) active.wait -= dt;
        else if (active.k < 0.9995) {
          active.k = damp(active.k, 1, 0.2, dt);
          active.vig.group.scale.setScalar(ease.out(clamp(active.k)));
        } else if (active.k !== 1) {
          active.k = 1;
          active.vig.group.scale.setScalar(1);
        }
        active.vig.update(dt, t);
      }
    },
    onRoute(route) {
      const i = indexOf(route);
      if (route.rest[0] && i < 0) ctx.router.replace('hot');
      show(i);
    },
    /** For tests and screenshots: the vignette that is playing, if any. */
    get vignette() {
      return active?.vig ?? null;
    },
  };
}
