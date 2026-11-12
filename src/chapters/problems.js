// Problems and solutions: a tower of the eight levels of the AI stack. Each floor holds the classic
// problems of its level. Opening a problem shows its competing solutions as profile shapes: the six
// spokes are the editorial 1-5 ratings on the six requirement angles (same directions as the labels
// around the floor), the height is the score under the visitor's requirement lens, and the crown
// marks the current winner. Content and scoring live in ../data/problems.js.
import * as THREE from 'three';
import './problems.css';
import { LEVELS, ANGLES, PRESETS, WEIGHT_WORDS, PROBLEMS_AS_OF, rankSolutions, findLevel, findProblem } from '../data/problems.js';
import { SERIES, PAL, glow } from '../core/colors.js';
import { glowMat, stdMat, glassMat, edges, ring, halo, gridFloor, instanced, setInst, flushInst } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, callout, button, segmented, slider, meterList } from '../core/ui.js';
import { radar } from '../core/charts.js';
import { damp, clamp } from '../core/tween.js';

const NA = ANGLES.length;
const R = 5.9; // circumradius of a floor (corners on the x axis, flat edges front and back)
const APO = (R * Math.sqrt(3)) / 2;
const T = 0.14; // thickness of a floor
const GAP = 1.4; // spacing of the closed tower
const LIFT = 10; // how far the floors above an open level rise
const RING = 3.3; // radius of the horseshoe of solutions around the open problem
const U = 0.24; // world length of one rating point
const GOLD = '#ffd27a'; // the current winner, everywhere
const PICK = '#8fdcff'; // the solution the visitor selected, everywhere
const STEEL = '#8497bd';
const ANGLE_COLOR = ANGLES.map((_, i) => SERIES[i]);
// Direction of angle i on the floor: Quality points away from the viewer, then clockwise seen from
// above, exactly like the radar chart in the panel.
const DIRS = ANGLES.map((_, i) => [Math.sin((i * Math.PI) / 3), -Math.cos((i * Math.PI) / 3)]);
const LEVEL_COLOR = LEVELS.map((_, i) => `hsl(${Math.round(205 + (i * 115) / (LEVELS.length - 1))}, 85%, 64%)`);
const MAX_PROB = Math.max(...LEVELS.map((l) => l.problems.length));
const MAX_SOL = Math.max(...LEVELS.flatMap((l) => l.problems.map((p) => p.solutions.length)));
const N_PROB = LEVELS.reduce((a, l) => a + l.problems.length, 0);
const N_SOL = LEVELS.reduce((a, l) => a + l.problems.reduce((b, p) => b + p.solutions.length, 0), 0);
const levelIndex = new Map(LEVELS.map((l, i) => [l, i]));
const BASE_WINNER = new Map(); // best fit under the default lens, to count how many a lens changes
for (const l of LEVELS) for (const p of l.problems) BASE_WINNER.set(p, rankSolutions(p, PRESETS[0].w)[0].sol);

/** Where problem j of n stands on its floor before one of them is opened. */
const levelSpot = (j, n) => [(j - (n - 1) / 2) * 3, n > 3 ? (j % 2 ? 1.8 : 0.3) : 1];

export async function enter(ctx) {
  const { app, root } = ctx;
  const narrow = window.innerWidth <= 860;
  const LS = narrow ? 1.25 : 1; // larger 3D text on small screens
  const TAU = ctx.reduceMotion ? 0.001 : 0.2;
  const tap = ctx.isTouch ? 'Tap' : 'Click';

  const sel = { level: undefined, problem: null, solution: null };
  let weights = [...PRESETS[0].w];
  let presetId = PRESETS[0].id;
  let ranked = [];
  let focus = null; // the solution in the detail card: the selected one, else the winner
  let crownSlot = null;
  let crownSnap = true;

  root.add(gridFloor({ size: 90, y: -1.1, color: '#31508c', opacity: 0.3 }));

  // ---------------------------------------------------------------- the tower
  const tower = new THREE.Group();
  root.add(tower);
  const hexGeo = new THREE.CylinderGeometry(R, R, T, 6).rotateY(Math.PI / 6);
  const pipGeo = new THREE.CylinderGeometry(0.2, 0.27, 0.3, 6);
  const floors = LEVELS.map((level, i) => {
    const color = LEVEL_COLOR[i];
    const g = new THREE.Group();
    const plate = new THREE.Mesh(hexGeo, glassMat('#0f1b36', 0.3, { emissive: color, ek: 0.07 }));
    const rim = edges(hexGeo, color, { k: 1.4, opacity: 0.95 });
    const title = textSprite(`${i + 1} · ${level.name}`, { size: 0.4 * LS, font: 'display', weight: 650, color: '#eef3ff', bg: 'rgba(5,8,16,0.62)' });
    title.position.set(0, 0.5, APO + 0.2);
    const n = level.problems.length;
    const pips = instanced(pipGeo, glowMat(color, { k: 1.2 }), n);
    for (let j = 0; j < n; j++) {
      const [x, z] = levelSpot(j, n);
      setInst(pips, j, x * 0.85, T / 2 + 0.15, z);
    }
    flushInst(pips);
    g.add(plate, rim, title, pips);
    g.position.y = i * GAP;
    tower.add(g);
    const f = {
      level, i, g, plate, rim, title, pips,
      base: glow(color, 1.3), bright: glow(color, 2.7),
      y: i * GAP, ty: i * GAP, op: 1, top: 1, pl: 0.3, tpl: 0.3, tt: 1, ttt: 1, hi: 0, thi: 0, on: 0,
      tip: `Level ${i + 1} · ${level.name}\n${level.desc}\n${n} problems: ${level.problems.map((p) => p.short).join(', ')}`,
    };
    ctx.pick.add(plate, {
      tip: () => (sel.level === level ? null : f.tip),
      hover: () => (f.thi = 1),
      out: () => (f.thi = 0),
      click: () => (sel.level !== level ? go(level) : sel.solution && pickSolution(null)),
    });
    return f;
  });

  // ---------------------------------------------------------------- the open floor
  const stage = new THREE.Group(); // origin: the top surface of the open floor
  stage.visible = false;
  root.add(stage);

  // six angles around the rim: a bar in the angle's color (brightness = its weight) and its name
  const angleMats = ANGLES.map(() => new THREE.MeshBasicMaterial());
  const rimGeo = new THREE.BoxGeometry(2.3, 0.07, 0.1);
  const rimLabels = ANGLES.map((a, i) => {
    const [dx, dz] = DIRS[i];
    const bar = new THREE.Mesh(rimGeo, angleMats[i]);
    bar.position.set(dx * APO, 0.04, dz * APO);
    bar.rotation.y = (-i * Math.PI) / 3;
    const lab = textSprite(a.short, { size: 0.28 * LS, weight: 600, color: '#dfe7f7' });
    lab.position.set(dx * (APO + 0.95), 0.34, dz * (APO + 0.95));
    stage.add(bar, lab);
    return lab;
  });

  // problems: pedestals
  const pedGeo = new THREE.CylinderGeometry(0.4, 0.52, 0.5, 6);
  const gemGeo = new THREE.OctahedronGeometry(0.24);
  const poleGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)]);
  const poleMat = new THREE.LineBasicMaterial({ color: glow(PAL.amber, 1.1), transparent: true, opacity: 0.55, depthWrite: false });
  const peds = Array.from({ length: MAX_PROB }, () => {
    const g = new THREE.Group();
    const col = new THREE.Mesh(pedGeo, stdMat('#1b2a4b', { rough: 0.4, metal: 0.3, emissive: PAL.amber, ek: 0.14, flat: true }));
    const wire = edges(pedGeo, PAL.amber, { k: 1.5 });
    col.position.y = wire.position.y = 0.25;
    const gem = new THREE.Mesh(gemGeo, glowMat(PAL.amber, { k: 2 }));
    gem.position.y = 0.84;
    const name = textSprite(' ', { size: 0.26 * LS, weight: 650, color: '#f2f5ff', bg: 'rgba(6,10,20,0.82)', border: 'rgba(255,255,255,0.24)', maxChars: 15, anchor: [0.5, 0] });
    const best = textSprite(' ', { size: 0.215 * LS, weight: 600, color: '#ffeec4', bg: 'rgba(40,27,4,0.85)', border: GOLD, maxChars: 17, anchor: [0.5, 0] });
    best.position.y = 1.2;
    const pole = new THREE.Line(poleGeo, poleMat);
    pole.position.y = 1.1;
    g.add(col, wire, gem, name, best, pole);
    g.visible = false;
    stage.add(g);
    const p = { g, gem, name, best, pole, problem: null, x: 0, z: 0, s: 1, ly: 1.86, tx: 0, tz: 0, ts: 1, tly: 1.86 };
    ctx.pick.add([col, gem, name], {
      tip: () => (p.problem && p.problem !== sel.problem ? `${p.problem.name}\n${p.problem.statement}` : null),
      click: () => p.problem && (p.problem === sel.problem ? pickSolution(null) : go(sel.level, p.problem)),
    });
    return p;
  });

  // solutions: profile shapes
  const guidePts = [];
  for (let i = 0; i < NA; i++) {
    const a = DIRS[i];
    const b = DIRS[(i + 1) % NA];
    const A = new THREE.Vector3(a[0] * 5 * U, 0, a[1] * 5 * U);
    guidePts.push(A, new THREE.Vector3(b[0] * 5 * U, 0, b[1] * 5 * U), new THREE.Vector3(), A.clone());
  }
  const guideGeo = new THREE.BufferGeometry().setFromPoints(guidePts); // rating 5 on every angle, and the six axes
  const guideMat = new THREE.LineBasicMaterial({ color: glow('#7d93c4', 0.9), transparent: true, opacity: 0.42, depthWrite: false });
  const barGeo = new THREE.BoxGeometry(1, 0.05, 0.075).translate(0.5, 0, 0);
  const slots = Array.from({ length: MAX_SOL }, () => {
    const g = new THREE.Group();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NA * 27), 3));
    const body = new THREE.Mesh(geo, stdMat('#1d2e55', { rough: 0.35, metal: 0.25, emissive: '#3b6fd0', ek: 0.2, flat: true }));
    const wgeo = new THREE.BufferGeometry();
    wgeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NA * 18), 3));
    const wire = new THREE.LineSegments(wgeo, new THREE.LineBasicMaterial({ color: 0xffffff, depthWrite: false }));
    wire.frustumCulled = false;
    const pillar = new THREE.Group(); // unit height, scaled to the score
    pillar.add(body, wire);
    const cap = new THREE.Group(); // rides on top of the pillar
    const spokes = ANGLES.map((_, i) => {
      const bar = new THREE.Mesh(barGeo, angleMats[i]);
      bar.rotation.y = Math.PI / 2 - (i * Math.PI) / 3;
      bar.position.y = 0.035;
      cap.add(bar);
      return bar;
    });
    const guide = new THREE.LineSegments(guideGeo, guideMat);
    guide.position.y = 0.015;
    const name = textSprite(' ', { size: 0.26 * LS, weight: 600, color: '#eef3ff', bg: 'rgba(6,10,20,0.8)', border: 'rgba(255,255,255,0.16)', maxChars: 15, anchor: [0.5, 0] });
    const score = textSprite('0.0', { size: 0.3 * LS, weight: 700, font: 'mono', color: '#ffffff' });
    g.add(pillar, cap, guide, name, score);
    g.visible = false;
    stage.add(g);
    const s = { g, body, wire, pillar, cap, spokes, name, score, sol: null, tip: '', above: true, H: 0.05, tH: 1, lift: 0, tLift: 0 };
    ctx.pick.add([body, name], { tip: () => (s.sol ? s.tip : null), click: () => s.sol && pickSolution(s.sol) });
    return s;
  });

  function writeShape(s, r) {
    const p = s.body.geometry.attributes.position.array;
    const w = s.wire.geometry.attributes.position.array;
    for (let i = 0; i < NA; i++) {
      const j = (i + 1) % NA;
      const ax = DIRS[i][0] * r[i] * U;
      const az = DIRS[i][1] * r[i] * U;
      const bx = DIRS[j][0] * r[j] * U;
      const bz = DIRS[j][1] * r[j] * U;
      // one triangle of the top fan, then the two triangles of the wall below it (facing outwards)
      p.set([0, 1, 0, bx, 1, bz, ax, 1, az, ax, 1, az, bx, 1, bz, bx, 0, bz, ax, 1, az, bx, 0, bz, ax, 0, az], i * 27);
      // top outline, base outline, vertical edge
      w.set([ax, 1, az, bx, 1, bz, ax, 0, az, bx, 0, bz, ax, 0, az, ax, 1, az], i * 18);
      s.spokes[i].scale.x = r[i] * U;
    }
    s.body.geometry.attributes.position.needsUpdate = true;
    s.body.geometry.computeVertexNormals();
    s.body.geometry.computeBoundingSphere();
    s.wire.geometry.attributes.position.needsUpdate = true;
  }

  // winner crown, rings, links from the problem to its solutions, rating digits of the focused shape
  const crown = new THREE.Group();
  const crownMat = glowMat(GOLD, { k: 2.5 });
  crown.add(new THREE.Mesh(new THREE.TorusGeometry(0.37, 0.028, 8, 40).rotateX(Math.PI / 2), crownMat));
  const spikeGeo = new THREE.ConeGeometry(0.05, 0.18, 8);
  for (let i = 0; i < 6; i++) {
    const c = new THREE.Mesh(spikeGeo, crownMat);
    c.position.set(Math.cos((i * Math.PI) / 3) * 0.37, 0.1, Math.sin((i * Math.PI) / 3) * 0.37);
    crown.add(c);
  }
  const glowSpot = halo(GOLD, 3, 0.28, 1.3);
  const winRing = ring(1.3, GOLD, { k: 2.2, segments: 64 });
  const pickRing = ring(1.42, PICK, { k: 2, segments: 64 });
  const fanGeo = new THREE.BufferGeometry();
  fanGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_SOL * 6), 3));
  const fan = new THREE.LineSegments(fanGeo, new THREE.LineBasicMaterial({ color: glow(PAL.amber, 1), transparent: true, opacity: 0.4, depthWrite: false }));
  fan.frustumCulled = false;
  fan.position.y = 0.02;
  const solFx = new THREE.Group();
  const digits = new THREE.Group(); // re-parented to the cap of the focused shape
  solFx.add(crown, glowSpot, winRing, pickRing, fan, digits);
  solFx.visible = false;
  stage.add(solFx);
  const digitSprites = ANGLES.map(() => {
    const d = textSprite('0', { size: 0.25 * LS, weight: 700, font: 'mono', color: '#ffffff' });
    digits.add(d);
    return d;
  });

  const GRAY = new THREE.Color('#5b6781');
  const BASE_EM = new THREE.Color('#3b6fd0');
  const WIN_EM = new THREE.Color(GOLD);
  const BASE_WIRE = glow('#9fc0ff', 1);
  const WIN_WIRE = glow(GOLD, 2.3);
  const PICK_WIRE = glow(PICK, 2);

  /** The lens in the scene: spokes and rim bars of an angle glow with its weight. */
  function paintLens() {
    for (let i = 0; i < NA; i++) {
      const w = weights[i];
      if (w === 0) angleMats[i].color.copy(GRAY).multiplyScalar(0.5);
      else angleMats[i].color.set(ANGLE_COLOR[i]).multiplyScalar([0, 0.8, 1.45, 2.3][w]);
      rimLabels[i].material.opacity = w === 0 ? 0.45 : 1;
      for (const s of slots) s.spokes[i].scale.z = 0.6 + 0.45 * w;
    }
  }

  // ---------------------------------------------------------------- HUD
  const leftBox = h('div.pr-col');
  const dynBox = h('div.pr-col');
  const presetNote = h('div.pr-foot');
  const changedEl = h('b');
  let bestSpans = []; // [problem, element] pairs in the panels that name the current best fit

  const presetSeg = segmented(PRESETS.map((p) => ({ value: p.id, label: p.name, title: p.note })), presetId, (id) => setLens(PRESETS.find((p) => p.id === id).w), { wrap: true });
  presetSeg.el.classList.add('pr-presets');
  const sliders = ANGLES.map((a, i) => {
    const s = slider({
      label: a.short, min: 0, max: 3, step: 1, value: weights[i],
      format: (v) => WEIGHT_WORDS[v],
      onInput: (v) => {
        weights[i] = v;
        lensChanged();
      },
    });
    s.el.title = `${a.name}: ${a.desc}`;
    s.el.querySelector('.lab').prepend(h('i.pr-sw', { style: { '--c': ANGLE_COLOR[i] } }));
    return s;
  });
  const rankList = meterList([], { max: 5 });
  rankList.el.classList.add('pr-rank');
  const profile = radar({ axes: ANGLES.map((a) => a.short), size: 230 });
  const detail = h('div.card.pr-detail');
  const anglesList = h('div.pr-angles', ANGLES.map((a, i) => h('div.pr-angle', h('i.pr-sw', { style: { '--c': ANGLE_COLOR[i] } }), h('b', a.name), h('span', a.desc))));
  const changedNote = h('div.callout', h('p', 'Under this lens ', changedEl, ' problems have a different best fit than under the General purpose lens.'));

  const winnerOf = (p) => rankSolutions(p, weights)[0].sol;

  function refreshWinners() {
    let changed = 0;
    for (const l of LEVELS) for (const p of l.problems) if (winnerOf(p) !== BASE_WINNER.get(p)) changed++;
    changedEl.textContent = `${changed} of ${N_PROB}`;
    for (const [p, el] of bestSpans) el.textContent = winnerOf(p).name;
    for (const ped of peds) if (ped.problem) ped.best.userData.setText(winnerOf(ped.problem).name);
  }

  function setLens(w) {
    weights = [...w];
    sliders.forEach((s, i) => s.set(weights[i]));
    lensChanged();
  }

  function lensChanged(url = true) {
    const match = PRESETS.find((p) => p.w.every((v, i) => v === weights[i]));
    presetId = match ? match.id : null;
    presetSeg.set(presetId);
    presetNote.textContent = match ? match.note : weights.some((v) => v > 0) ? 'Custom lens: your own mix of weights.' : 'Everything ignored: all six angles count the same.';
    paintLens();
    refreshWinners();
    if (sel.problem) rerank();
    if (url) syncUrl();
  }

  function crumbs() {
    const sep = () => h('span.pr-sep', '›');
    const kids = [sel.level ? h('button.pr-crumb', { type: 'button', onclick: () => go(null) }, 'All levels') : h('span.pr-crumb.cur', 'All levels')];
    if (sel.level) kids.push(sep(), sel.problem ? h('button.pr-crumb', { type: 'button', onclick: () => go(sel.level) }, sel.level.name) : h('span.pr-crumb.cur', sel.level.name));
    if (sel.problem) kids.push(sep(), h('span.pr-crumb.cur', sel.problem.short));
    return h('nav.pr-crumbs', { 'aria-label': 'Breadcrumb' }, kids);
  }

  function problemItem(level, p, compact) {
    const best = h('b');
    bestSpans.push([p, best]);
    return h(
      'button.pr-item',
      { type: 'button', onclick: () => go(level, p) },
      h('span.pr-it', h('b', p.name), !compact && h('small', p.statement), h('span.pr-best', h('i.pr-dot'), 'Best fit now:', best)),
    );
  }

  function renderLeft() {
    bestSpans = [];
    const { level, problem } = sel;
    const kids = [crumbs()];
    if (!level) {
      kids.push(
        header({
          kicker: `Problems · ${LEVELS.length} levels, ${N_PROB} problems, ${N_SOL} solutions`,
          title: 'Every level of AI has its own hard problems',
          lead: 'From raw data at the bottom of the tower to trust at the top, each floor holds the classic problems of its level and the competing ways to solve them. **No solution wins everywhere**: it depends on what you require. Set the requirement lens and watch the winners change.',
        }),
        section(
          'The eight levels',
          h('div.pr-list', [...LEVELS].reverse().map((l) => {
            const i = levelIndex.get(l);
            return h(
              'button.pr-item',
              { type: 'button', onclick: () => go(l) },
              h('span.pr-num', { style: { '--c': LEVEL_COLOR[i] } }, String(i + 1)),
              h('span.pr-it', h('b', l.name), h('small', l.problems.map((p) => p.short).join(' · '))),
            );
          })),
        ),
        callout(`The 1-5 ratings behind every shape and score are **editorial judgments** made for this atlas to build intuition about trade-offs. They are not measurements. Each published fact names its source and links to it (checked ${PROBLEMS_AS_OF}).`, 'warn', 'Judged, not measured'),
      );
    } else if (!problem) {
      const i = levelIndex.get(level);
      kids.push(
        header({ kicker: `Level ${i + 1} of ${LEVELS.length}`, title: level.name, lead: level.desc }),
        section('Problems on this level', h('div.pr-list', level.problems.map((p) => problemItem(level, p)))),
        h(
          'div.btnrow',
          i < LEVELS.length - 1 && button(`↑ ${LEVELS[i + 1].name}`, { kind: 'small', title: 'One level up', onClick: () => go(LEVELS[i + 1]) }),
          i > 0 && button(`↓ ${LEVELS[i - 1].name}`, { kind: 'small', title: 'One level down', onClick: () => go(LEVELS[i - 1]) }),
        ),
        h('div.pr-foot', 'Ratings are editorial judgments for intuition; published facts carry their sources.'),
      );
    } else {
      const i = levelIndex.get(level);
      kids.push(
        header({ kicker: `Level ${i + 1} · ${level.name}`, title: problem.name, lead: problem.statement }),
        h('div.pr-block', h('h5', 'Why it is hard'), h('p', problem.hard)),
        h('div.pr-block', h('h5', 'What the ratings mean here'), h('p', problem.yardstick)),
        callout('Each shape is one solution. Its six spokes are its ratings on the six angles named around the floor, and its height is its score under your lens. The crown marks the current winner; the numbers on the spokes belong to the solution in the detail card.', 'key', 'Reading the floor'),
        section('Other problems on this level', h('div.pr-list', level.problems.filter((p) => p !== problem).map((p) => problemItem(level, p, true)))),
        h('div.pr-foot', `Ratings (1-5) are editorial judgments for intuition, not measurements. Facts are quoted from the sources named beside them, checked ${PROBLEMS_AS_OF}.`),
      );
    }
    leftBox.replaceChildren(...kids.filter(Boolean));
    if (leftBox.parentElement) leftBox.parentElement.scrollTop = 0;
  }

  function renderRight() {
    const { level, problem } = sel;
    const how = h('div.pr-foot', 'A score is the weighted geometric mean of the six ratings, so one weak angle that you care about cannot be bought back elsewhere.');
    if (!level) dynBox.replaceChildren(section('Six angles of a requirement', anglesList), changedNote, how);
    else if (!problem) {
      const rows = level.problems.map((p) => {
        const best = h('b');
        bestSpans.push([p, best]);
        return h('button.pr-item.pr-win', { type: 'button', onclick: () => go(level, p) }, h('i.pr-dot'), h('span.pr-it', h('small', p.short), best));
      });
      dynBox.replaceChildren(section('Best fit on this level', h('div.pr-list', rows)), changedNote, how);
    } else {
      dynBox.replaceChildren(
        section('Ranking under this lens', rankList.el, h('div.pr-foot', 'Score out of 5: weighted geometric mean of six editorial ratings. A judgment, not a measurement.')),
        section('Profile', profile.el),
        detail,
      );
    }
  }

  function renderDock() {
    const { level, problem } = sel;
    const back = problem
      ? button(`← ${level.name}`, { kind: 'small', onClick: () => go(level) })
      : level
        ? button('← All levels', { kind: 'small', onClick: () => go(null) })
        : null;
    const hint = problem
      ? 'Spokes are ratings, height is the score. Move a slider and watch the crown.'
      : level
        ? `${tap} a pedestal to compare its solutions. Gold tags name the best fit under your lens.`
        : `${tap} a floor to open it. Level 1, Data, is at the bottom.`;
    // on small screens the dock sits over the bottom sheet: keep it to one row
    ctx.ui.dock(h('div.dockrow', back, !(narrow && back) && h('span.hint', hint)));
  }

  function renderDetail() {
    const s = focus;
    const k = ranked.findIndex((e) => e.sol === s);
    let src = null;
    if (s.src) {
      let host = s.src;
      try {
        host = new URL(s.src).hostname.replace(/^www\./, '');
      } catch {
        /* keep the raw string */
      }
      src = h('a.pr-src', { href: s.src, target: '_blank', rel: 'noopener' }, `Source: ${host}`);
    }
    detail.replaceChildren(
      h('div.pr-dhead', h('h4', s.name), h('span.chip', { style: { '--c': k === 0 ? GOLD : PICK } }, k === 0 ? 'current winner' : `rank ${k + 1} of ${ranked.length}`)),
      h('div.pr-ratings', ANGLES.map((a, i) => h('div.pr-rt', { title: `${a.name}: ${a.desc}` }, h('i.pr-sw', { style: { '--c': ANGLE_COLOR[i] } }), h('span', a.short), h('b', String(s.r[i]))))),
      h('div.pr-block', h('h5', 'How it works'), h('p', s.how)),
      h('div.pr-block', h('h5', 'Published fact'), s.fact ? h('p', s.fact, s.asOf ? ` (as of ${PROBLEMS_AS_OF})` : '', src && ' ', src) : h('p.pr-muted', 'No single published figure is quoted for this one.')),
      h('div.pr-block', h('h5', 'Choose it when'), h('p', s.when)),
      ...(s.link ? [h('a.btn.primary', { href: s.link.href }, `${s.link.label} →`)] : []),
    );
  }

  // ---------------------------------------------------------------- scene state
  function layoutFloors() {
    const s = sel.level ? levelIndex.get(sel.level) : -1;
    for (const f of floors) {
      const d = f.i - s;
      f.on = s >= 0 && d === 0 ? 1 : 0;
      if (s < 0) [f.ty, f.top, f.tpl] = [f.i * GAP, 1, 0.3];
      else if (d === 0) [f.ty, f.top, f.tpl] = [s * GAP, 1, 0.78];
      else if (d < 0) [f.ty, f.top, f.tpl] = [s * GAP - 0.32 + d * 0.34, 0.4, 0.2]; // packed underneath
      else [f.ty, f.top, f.tpl] = [s * GAP + LIFT + (d - 1) * 0.34, 0.3, 0.1]; // lifted out of the way
      f.ttt = s < 0 ? 1 : 0;
      f.pips.visible = s < 0;
    }
    stage.visible = s >= 0;
  }

  function layoutPeds(snap) {
    const level = sel.level;
    if (!level) return;
    const n = level.problems.length;
    let k = 0;
    peds.forEach((p, j) => {
      const prob = level.problems[j] || null;
      if (prob && p.problem !== prob) p.name.userData.setText(prob.name);
      p.problem = prob;
      p.g.visible = !!prob;
      if (!prob) return;
      if (!sel.problem) [p.tx, p.tz, p.ts, p.tly] = [...levelSpot(j, n), 1.2, 1.96];
      else if (prob === sel.problem) [p.tx, p.tz, p.ts, p.tly] = [0, 0, 1.05, 1.22];
      else [p.tx, p.tz, p.ts, p.tly] = [(k++ - (n - 2) / 2) * 2.75, -4.5, 0.9, 3]; // signposts at the back
      p.best.visible = !sel.problem;
      p.pole.visible = !!sel.problem && prob !== sel.problem;
      if (snap) [p.x, p.z, p.ly, p.s] = [p.tx, p.tz, p.tly, 0.05];
    });
  }

  function assignShapes() {
    const p = sel.problem;
    solFx.visible = !!p;
    const n = p ? p.solutions.length : 0;
    // a horseshoe that opens to the back, so the problem in the middle and the signposts stay in view
    const step = n > 1 ? Math.min(Math.PI / 3, (Math.PI * 4) / 3 / (n - 1)) : 0;
    const fp = fanGeo.attributes.position.array;
    slots.forEach((s, k) => {
      const sol = p ? p.solutions[k] || null : null;
      s.sol = sol;
      s.g.visible = !!sol;
      if (!sol) return;
      const a = -Math.PI / 2 + (k - (n - 1) / 2) * step;
      const x = Math.cos(a) * RING;
      const z = -Math.sin(a) * RING;
      s.g.position.set(x, 0, z);
      writeShape(s, sol.r);
      s.name.userData.setText(sol.name);
      s.H = 0.05;
      s.lift = 0;
      fp.set([0, 0, 0, x, 0, z], k * 6);
      // name tags point away from the middle: above the far shapes, beside the outer ones, in front of the near ones
      const side = x < 0 ? -1 : 1;
      let outX = 0;
      let outZ = 0;
      for (let i = 0; i < NA; i++) {
        outX = Math.max(outX, DIRS[i][0] * side * sol.r[i] * U);
        outZ = Math.max(outZ, DIRS[i][1] * sol.r[i] * U);
      }
      s.above = z < -1.2;
      if (s.above) s.name.center.set(0.5, 0);
      else if (z > 1.9) {
        s.name.center.set(0.5, 1);
        s.name.position.set(side * 0.15, 0.4, outZ + 0.32);
      } else {
        s.name.center.set(side > 0 ? 0 : 1, 0.5);
        s.name.position.set(side * (outX + 0.3), 0.5, 0.25);
      }
      if (s.above) s.name.position.set(0, 0.8, 0);
    });
    fanGeo.setDrawRange(0, n * 2);
    fanGeo.attributes.position.needsUpdate = true;
    crownSnap = true;
  }

  /** Re-rank the open problem under the lens and update shapes, crown and the right panel. */
  function rerank() {
    const p = sel.problem;
    ranked = rankSolutions(p, weights);
    const win = ranked[0].sol;
    focus = sel.solution || win;
    let mean = 0;
    for (const e of ranked) mean += e.score / ranked.length;
    const lo = ranked[ranked.length - 1].score;
    const span = Math.max(1e-6, ranked[0].score - lo);
    ranked.forEach((e, k) => {
      const s = slots[e.index];
      const t = (e.score - lo) / span;
      const picked = e.sol === sel.solution;
      s.tH = clamp(0.9 + (e.score - mean), 0.2, 1.8);
      s.tLift = e.sol === focus ? 0.42 : 0;
      s.score.userData.setText(e.score.toFixed(1));
      s.body.material.color.set(k === 0 ? '#4b3a15' : '#1d2e55');
      s.body.material.emissive.copy(k === 0 ? WIN_EM : BASE_EM).multiplyScalar(k === 0 ? 0.45 : 0.1 + 0.3 * t);
      s.wire.material.color.copy(k === 0 ? WIN_WIRE : picked ? PICK_WIRE : BASE_WIRE).multiplyScalar(k === 0 || picked ? 1 : 0.7 + 0.8 * t);
      s.tip = `${e.sol.name}\nScore ${e.score.toFixed(2)} of 5 under this lens, rank ${k + 1} of ${ranked.length}\n${ANGLES.map((a, i) => `${a.short} ${e.sol.r[i]}`).join(' · ')}`;
    });
    crownSlot = slots[ranked[0].index];
    const fs = slots[p.solutions.indexOf(focus)];
    fs.cap.add(digits);
    digitSprites.forEach((d, i) => {
      const rr = focus.r[i] * U + 0.22;
      d.userData.setText(String(focus.r[i]));
      d.position.set(DIRS[i][0] * rr, 0.12, DIRS[i][1] * rr);
    });
    pickRing.visible = !!sel.solution;
    pickRing.position.set(fs.g.position.x, 0.03, fs.g.position.z);

    rankList.update(
      ranked.map((e, k) => ({
        label: h('span.pr-rl', h('span.pr-rk' + (k === 0 ? '.win' : ''), String(k + 1)), e.sol.name),
        value: e.score,
        note: e.score.toFixed(2),
        color: k === 0 ? GOLD : STEEL,
        active: e.sol === focus,
        title: ANGLES.map((a, i) => `${a.short} ${e.sol.r[i]}`).join(' · '),
        onClick: () => pickSolution(e.sol),
      })),
    );
    profile.update(
      focus === win
        ? [ranked[1] && { name: `${ranked[1].sol.name} (runner-up)`, color: STEEL, values: ranked[1].sol.r, dim: true }, { name: `${win.name} (winner)`, color: GOLD, values: win.r }].filter(Boolean)
        : [{ name: `${win.name} (winner)`, color: GOLD, values: win.r }, { name: `${focus.name} (selected)`, color: PICK, values: focus.r }],
    );
    renderDetail();
  }

  function pickSolution(sol) {
    if (!sel.problem) return;
    sel.solution = sol && sol !== sel.solution ? sol : null;
    rerank();
    syncUrl();
  }

  // ---------------------------------------------------------------- camera, navigation, URL
  const VIEW = { min: 4, max: 110, maxPolar: Math.PI * 0.5 };
  function frameView(instant) {
    const open = !!sel.level;
    const target = open ? [0, levelIndex.get(sel.level) * GAP + 0.9, 0.1] : [0, ((LEVELS.length - 1) * GAP) / 2 + 0.15, 0];
    const dir = open ? [0.14, 0.84, 1] : [0.34, 0.3, 1];
    const v = { ...VIEW, target, pos: target.map((c, i) => c + dir[i]), fit: open ? (narrow ? 5.7 : 6.8) : narrow ? 6.5 : 8.2 };
    if (instant) return app.setView(v);
    // let the engine fit the view between the panels (it becomes "home"), then glide there
    const cam = app.camera.position.clone();
    const tgt = app.controls.target.clone();
    app.setView(v);
    const dest = app.camera.position.clone();
    app.camera.position.copy(cam);
    app.controls.target.copy(tgt);
    return app.flyTo(dest, target, 1.2);
  }

  function syncUrl() {
    const path = ['problems', sel.level?.id, sel.problem?.id].filter(Boolean).join('/');
    const q = {};
    if (presetId !== PRESETS[0].id) q.lens = presetId ?? weights.join('');
    if (sel.solution) q.sol = sel.solution.id;
    ctx.router.replace(path, q);
  }

  /** Show a level, a problem on it and optionally a selected solution (all null = the whole tower). */
  function go(level = null, problem = null, solution = null, o = {}) {
    const levelChanged = sel.level !== level;
    const problemChanged = levelChanged || sel.problem !== problem;
    sel.level = level;
    sel.problem = problem;
    sel.solution = solution;
    if (levelChanged) layoutFloors();
    if (problemChanged) {
      layoutPeds(levelChanged);
      assignShapes();
      renderLeft();
      renderRight();
      renderDock();
    }
    refreshWinners();
    if (problem) rerank();
    if (levelChanged) frameView(o.instant);
    if (o.url !== false) syncUrl();
  }

  function goUp() {
    if (sel.solution) pickSolution(null);
    else if (sel.problem) go(sel.level);
    else if (sel.level) go(null);
  }

  function applyRoute(route, initial) {
    const [lid, pid] = route.rest || [];
    const level = findLevel(lid);
    const problem = findProblem(level, pid);
    const q = route.query || {};
    const lens = PRESETS.find((p) => p.id === q.lens)?.w || (/^[0-3]{6}$/.test(q.lens || '') ? [...q.lens].map(Number) : null);
    if (lens && lens.some((v, i) => v !== weights[i])) {
      weights = [...lens];
      sliders.forEach((s, i) => s.set(weights[i]));
    }
    if (lens || initial) lensChanged(false);
    go(level, problem, problem?.solutions.find((s) => s.id === q.sol) || null, { url: false, instant: initial });
  }

  ctx.ui.left(leftBox);
  ctx.ui.right(
    section('Requirement lens', h('div.pr-foot', 'What do you require? Pick a scenario, or set how much each angle matters.'), presetSeg.el, presetNote, h('div.pr-lens', sliders.map((s) => s.el))),
    dynBox,
  );
  ctx.pick.onMiss(goUp);
  ctx.key('Escape', goUp);
  applyRoute(ctx.route, true);

  return {
    update(dt, t) {
      for (const f of floors) {
        f.y = damp(f.y, f.ty, TAU, dt);
        f.op = damp(f.op, f.top, TAU, dt);
        f.pl = damp(f.pl, f.tpl, TAU, dt);
        f.tt = damp(f.tt, f.ttt, TAU, dt);
        f.hi = damp(f.hi, f.thi, 0.08, dt);
        f.g.position.y = f.y;
        f.plate.material.opacity = f.pl;
        f.rim.material.opacity = 0.95 * f.op;
        f.rim.material.color.copy(f.base).lerp(f.bright, Math.max(f.hi, f.on));
        f.title.material.opacity = f.tt;
        f.title.visible = f.tt > 0.03;
      }
      if (!sel.level) return;
      stage.position.y = floors[levelIndex.get(sel.level)].y + T / 2;
      for (const p of peds) {
        if (!p.problem) continue;
        p.x = damp(p.x, p.tx, TAU, dt);
        p.z = damp(p.z, p.tz, TAU, dt);
        p.s = damp(p.s, p.ts, TAU, dt);
        p.ly = damp(p.ly, p.tly, TAU, dt);
        p.g.position.set(p.x, 0, p.z);
        p.g.scale.setScalar(p.s);
        p.name.position.y = p.ly;
        p.pole.scale.y = Math.max(0.01, p.ly - 1.1);
        if (!ctx.reduceMotion) p.gem.rotation.y += dt * 0.9;
      }
      if (!sel.problem || !crownSlot) return;
      for (const s of slots) {
        if (!s.sol) continue;
        s.H = damp(s.H, s.tH, TAU, dt);
        s.lift = damp(s.lift, s.tLift, TAU, dt);
        s.pillar.scale.y = s.H;
        s.cap.position.y = s.H;
        s.score.position.y = s.H + 0.36;
        if (s.above) s.name.position.y = s.H + 0.74 + s.lift;
      }
      const c = crown.position;
      const gx = crownSlot.g.position.x;
      const gz = crownSlot.g.position.z;
      const gy = crownSlot.H + 0.36;
      if (crownSnap) c.set(gx, gy, gz);
      else c.set(damp(c.x, gx, 0.12, dt), damp(c.y, gy, 0.12, dt), damp(c.z, gz, 0.12, dt));
      crownSnap = false;
      if (!ctx.reduceMotion) crown.rotation.y += dt * 0.9;
      glowSpot.position.set(c.x, c.y * 0.5, c.z);
      winRing.position.set(c.x, 0.03, c.z);
      winRing.scale.setScalar(ctx.reduceMotion ? 1 : 1 + 0.04 * Math.sin(t * 3));
    },
    onRoute(route) {
      applyRoute(route, false);
    },
  };
}
