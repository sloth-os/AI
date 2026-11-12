// Deployment Modes: ten ways to put a model in front of users. Each mode is a small animated
// topology (core/flowgraph.js) on its own tile of one floor, so the modes can be compared at a
// glance ("All ten") or studied one at a time. Content and diagrams live in data/deployments.js;
// this file builds the stages, runs the small live simulations and fills the panels.
import * as THREE from 'three';
import './deploy.css';
import { MODES, RATINGS } from '../data/deployments.js';
import { AS_OF, FRONTIER, SELF_HOSTED, SERVING_NOTES, TECHNIQUES_2026 } from '../data/landscape.js';
import { FlowGraph, roleLegend, flyFit } from '../core/flowgraph.js';
import { PAL, SERIES, STATUS, glow, seqCss, withAlpha } from '../core/colors.js';
import { gridFloor, line, ring, edges, instanced, setInst, flushInst, glowMat, glassMat } from '../core/geom.js';
import { textSprite, textPlane } from '../core/labels.js';
import { h, header, section, text, callout, button, legend, chips, meterList, tabs, rich } from '../core/ui.js';
import { damp } from '../core/tween.js';

const ACCENT = PAL.lime;
/** What a packet carries -> its color, in the scene and in the legend. */
const KIND = { request: '#ffffff', response: ACCENT, weights: SERIES[3], data: SERIES[3], update: SERIES[4], control: SERIES[6], blocked: STATUS.critical };
const flowColor = (kind) => (kind === 'request' || kind === 'response' ? kind : KIND[kind]);

const TILE_W = 17.4; // every mode stands on a tile of the same size
const TILE_D = 15;
const DX = 18.8; // tile pitch: three columns, rows of 3 + 3 + 3 + 1
const DZ = 16.6;
const cellOf = (i) => (i === 9 ? [0, 1.5 * DZ] : [((i % 3) - 1) * DX, (Math.floor(i / 3) - 1.5) * DZ]);
const MODE_DIR = [0.22, 1.15, 1]; // viewing directions (from the target toward the camera)
const ALL_DIR = [0, 1.3, 1];
const LIMITS = { min: 5, max: 340, maxPolar: Math.PI * 0.47 };
const EDGE_ON = 0.55; // flowgraph's normal edge brightness
const EDGE_OFF = 0.1;

// ------------------------------------------------------------------ live behaviors

/** Lets a node (a replica) appear and vanish: body scale, glyph brightness and its edges. */
function presence(g, id, edgeIds) {
  const n = g.nodes.get(id);
  const es = edgeIds.map((e) => g.edges.get(e)).filter(Boolean);
  return (p, lit, flash = 0) => {
    const s = Math.max(0.001, p);
    setInst(g.bodies, n.i, n.x, (n.top / 2) * s, n.z, n.size * s);
    n.kT = p < 0.02 ? 0 : lit ? 1 + flash * 1.6 : 0.3 + 0.4 * p;
    for (const e of es) e.kT = lit ? EDGE_ON : EDGE_OFF;
  };
}

/** A small tower of boxes above a node: one box per waiting request. */
function queueStack(parent, x, y, z, max = 8) {
  const mesh = instanced(new THREE.BoxGeometry(0.44, 0.11, 0.44), glowMat('#ffffff', { k: 1.25 }), max);
  mesh.position.set(x, y, z);
  parent.add(mesh);
  let shown = -1;
  return (q) => {
    const n = Math.min(max, q);
    if (n === shown) return;
    shown = n;
    for (let i = 0; i < max; i++) setInst(mesh, i, 0, i * 0.17, 0, i < n ? 1 : 0);
    flushInst(mesh);
  };
}

const SAY = {
  'self-hosted': (q) => (q > 1 ? `${q} requests waiting: capacity is fixed` : q === 1 ? '1 request waiting' : 'The GPU keeps up'),
  k8s: (q, warm, booting) => (booting ? `Scaling up: replica ${warm + 1} is starting` : `${warm} replicas ready · ${q > 5 ? 'long queue' : q > 0 ? 'short queue' : 'no queue'}`),
  serverless: (q, warm, booting) =>
    warm + booting === 0 ? 'Scaled to zero: nothing is running' : warm === 0 ? 'Cold start: the request waits' : booting ? `${warm} warm, 1 more starting` : `${warm} warm ${warm > 1 ? 'replicas' : 'replica'}`,
};

/**
 * Requests arrive at a queue in front of a pool of replicas. Each warm replica serves one request
 * per `service` seconds; the pool grows when the queue builds up (after a `boot` delay) and shrinks
 * when replicas sit idle. One server (min = max = 1), an autoscaled cluster and scale-to-zero are
 * the same loop with different limits.
 */
function poolSim(env, cfg) {
  const { g, fx, setStatus } = env;
  const qn = g.nodes.get(cfg.queueAt);
  const stack = queueStack(fx, qn.x, qn.top + 0.98, qn.z);
  const say = SAY[env.mode.id];
  const reps = cfg.replicas.map((r, i) => ({ ...r, state: i < cfg.min ? 'warm' : 'off', boot: 0, busy: 0, idle: 0, flash: 0, p: i < cfg.min ? 1 : 0, show: presence(g, r.id, [r.edge, r.ctl, r.wts].filter(Boolean)) }));
  const dynamic = cfg.max > cfg.min;
  let q = 0;
  let t = 0;
  let acc = 0;
  let next = 0;
  let lastLoop = -1;
  let cool = 0;

  const rateAt = (u) => {
    let r = 0;
    for (const [from, v] of cfg.rate) if (u >= from) r = v;
    return r;
  };
  const arrive = () =>
    g.send(cfg.inPath, { size: 0.11, speed: 6.5 }).then((ok) => {
      if (ok && !env.dead) q++;
    });
  const serve = (r) =>
    g.send(r.edge, { size: 0.11, speed: 6.5 }).then((ok) => {
      if (ok && !env.dead) g.send(['-' + r.edge, ...cfg.outPath], { color: 'response', count: 5, size: 0.065, gap: 0.4, speed: 6.5 });
    });
  const start = (r) => {
    r.state = 'boot';
    r.boot = 0;
    if (r.ctl) g.send(r.ctl, { color: KIND.control, size: 0.09, count: 2, gap: 0.4, speed: 7 });
    const w = g.edges.get(r.wts);
    if (w) g.send(r.wts, { color: KIND.weights, size: 0.085, count: 6, gap: 0.4, speed: (w.len + 2) / (cfg.boot * 0.85) });
  };

  return (dt) => {
    t += dt;
    const loop = Math.floor(t / cfg.period);
    const tt = t - loop * cfg.period;
    if (cfg.schedule) {
      if (loop !== lastLoop) {
        lastLoop = loop;
        next = 0;
      }
      while (next < cfg.schedule.length && tt >= cfg.schedule[next]) {
        next++;
        arrive();
      }
    } else {
      acc += rateAt(tt / cfg.period) * dt;
      while (acc >= 1) {
        acc -= 1;
        arrive();
      }
    }

    let warm = 0;
    let booting = 0;
    for (const r of reps) {
      if (r.state === 'boot') {
        r.boot += dt / cfg.boot;
        if (r.boot >= 1) {
          r.state = 'warm';
          r.busy = 0;
          r.idle = 0;
          r.flash = 0.6;
        }
      }
      if (r.state === 'warm') {
        r.busy -= dt;
        if (r.busy <= 0 && q > 0) {
          q--;
          r.busy = cfg.service;
          r.idle = 0;
          serve(r);
        } else if (r.busy <= 0) r.idle += dt;
        warm++;
      } else if (r.state === 'boot') booting++;
    }

    if (dynamic) {
      cool -= dt;
      const total = warm + booting;
      if (cool <= 0 && total < cfg.max && ((total === 0 && q > 0) || (booting === 0 && q >= cfg.upAt))) {
        start(reps.find((r) => r.state === 'off'));
        cool = 1;
        booting++;
      }
      for (let i = reps.length - 1; i >= cfg.min; i--) {
        const r = reps[i];
        if (r.state === 'warm' && r.idle > cfg.idleOff) {
          r.state = 'off';
          warm--;
          break;
        }
      }
      for (const r of reps) {
        r.flash = Math.max(0, r.flash - dt);
        r.p = damp(r.p, r.state === 'off' ? 0 : r.state === 'boot' ? 0.25 + 0.6 * r.boot : 1, 0.12, dt);
        r.show(r.p, r.state === 'warm', r.flash);
      }
      g.bodies.instanceMatrix.needsUpdate = true;
    }
    stack(q);
    setStatus(say(q, warm, booting));
  };
}

/** A router that answers most requests on the near tier and escalates the hard ones to the far tier. */
function cascadeSim(env, cfg) {
  const { g, setStatus } = env;
  const far = g.edges.get(cfg.far);
  let t = cfg.every;
  let k = 0;
  let local = 0;
  let cloud = 0;
  let hot = 0;
  const fire = (hard) =>
    g.send(cfg.inPath, { size: hard ? 0.18 : 0.1, speed: 6 }).then((ok) => {
      if (!ok || env.dead) return;
      if (hard) {
        cloud++;
        hot = 2.4;
      } else local++;
      const leg = hard ? cfg.far : cfg.near;
      g.send(leg, { size: hard ? 0.18 : 0.1, speed: hard ? 4.5 : 6 }).then((ok2) => {
        if (ok2 && !env.dead) g.send(['-' + leg, '-' + cfg.inPath], { color: 'response', count: hard ? 8 : 4, size: 0.065, gap: 0.4, speed: hard ? 5 : 6.5 });
      });
    });
  return (dt) => {
    t += dt;
    if (t >= cfg.every) {
      t -= cfg.every;
      if (local + cloud >= 20) local = cloud = 0;
      fire(cfg.pattern[k++ % cfg.pattern.length] === 'h');
    }
    hot = Math.max(0, hot - dt);
    far.kT = hot > 0 ? 2 : 0.3;
    setStatus(`Local ${local} · cloud ${cloud}`);
  };
}

/** The air gap: a glass wall, and outside traffic that runs into it and goes no further. */
function wallProp(env, cfg) {
  const { g, inner, fx } = env;
  const { z, x0, x1, label } = cfg.wall;
  const H = 1.7;
  const geo = new THREE.BoxGeometry(x1 - x0, H, 0.24);
  const slab = new THREE.Mesh(geo, glassMat('#4a2c12', 0.2, { emissive: PAL.amber, ek: 0.2 }));
  slab.position.set((x0 + x1) / 2, H / 2, z);
  const rim = edges(geo, PAL.amber, { k: 2 });
  rim.position.copy(slab.position);
  const sign = textPlane(label, { size: 0.5, font: 'mono', weight: 650, color: '#ffe2b0' });
  sign.position.set((x0 + x1) / 2 + 2.4, H * 0.55, z + 0.14);
  inner.add(slab, rim, sign);

  const N = 3;
  const col = glow(STATUS.critical, 2.4);
  const lanes = cfg.blocked.map((id) => {
    const n = g.nodes.get(id);
    const a = new THREE.Vector3(n.x, 0.5, n.z + n.hd + 0.15);
    const b = new THREE.Vector3(n.x, 0.5, z - 0.16);
    inner.add(line([a, b], STATUS.critical, { k: 1, opacity: 0.5 }));
    const flash = ring(0.32, STATUS.critical, { k: 2.6, segments: 32, opacity: 0.99 });
    flash.rotation.x = Math.PI / 2;
    flash.position.copy(b);
    fx.add(flash);
    return { a, b, flash };
  });
  const dots = instanced(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, fog: false }), N * lanes.length);
  fx.add(dots);
  let t = 0;
  return (dt) => {
    t += dt;
    lanes.forEach((ln, j) => {
      let hit = 1;
      for (let i = 0; i < N; i++) {
        const s = (t * 0.36 + i / N + j * 0.17) % 1;
        const p = Math.min(1, s / 0.86);
        const sc = s < 0.86 ? 0.1 : 0.1 * (1 - (s - 0.86) / 0.14);
        setInst(dots, j * N + i, ln.a.x, 0.5, ln.a.z + (ln.b.z - ln.a.z) * p, sc, sc, sc, col);
        if (s >= 0.86) hit = Math.min(hit, (s - 0.86) / 0.14);
      }
      ln.flash.scale.setScalar(0.5 + 2 * hit);
      ln.flash.material.opacity = hit >= 1 ? 0 : 1 - hit;
    });
    flushInst(dots);
  };
}

// ------------------------------------------------------------------ one stage per mode

function buildStage(ctx, mode, index, onNode) {
  const [cx, cz] = cellOf(index);
  const group = new THREE.Group();
  group.position.set(cx, 0, cz);
  ctx.root.add(group);
  const inner = new THREE.Group(); // the diagram, centerd on the tile
  const fx = new THREE.Group(); // bright extras that are hidden while the stage is in the background
  group.add(inner);
  inner.add(fx);

  const zoneSpecs = mode.graph.zones ?? [];
  const g = new FlowGraph(ctx, { ...mode.graph, zones: zoneSpecs.map(({ label, ...z }) => z) }, { accent: ACCENT, labels: false, labelSize: 0.4, packets: 96, onClick: (n) => onNode(index, n) });
  inner.add(g.group);

  const captions = new THREE.Group(); // zone names (larger than the graph's own, and behind each zone)
  zoneSpecs.forEach((z, i) => {
    if (!z.label) return;
    const s = textSprite(z.label.toUpperCase(), { size: 0.34, font: 'mono', weight: 600, color: '#b7c3de', anchor: [0, 0] });
    s.position.set(g.zones[i].x0 + 0.3, 0.06, g.zones[i].z0 - 0.08);
    captions.add(s);
  });
  captions.visible = false;
  inner.add(captions);

  let x0 = Infinity;
  let x1 = -Infinity;
  let z0 = Infinity;
  let z1 = -Infinity;
  const grow = (ax, az, bx, bz) => {
    x0 = Math.min(x0, ax);
    z0 = Math.min(z0, az);
    x1 = Math.max(x1, bx);
    z1 = Math.max(z1, bz);
  };
  for (const n of g.nodeList) grow(n.x - n.hw, n.z - n.hd, n.x + n.hw, n.z + n.hd + 0.9);
  for (const z of g.zones) grow(z.x0, z.z0, z.x1, z.z1 + 0.55);
  const wall = mode.props?.wall;
  if (wall) grow(wall.x0, wall.z, wall.x1, wall.z);
  inner.position.set(-(x0 + x1) / 2, 0, -(z0 + z1) / 2);
  const w = x1 - x0;
  const d = z1 - z0;

  const hw = TILE_W / 2;
  const hd = TILE_D / 2;
  const pad = new THREE.Mesh(new THREE.PlaneGeometry(TILE_W, TILE_D), new THREE.MeshBasicMaterial({ color: '#0a1428', transparent: true, opacity: 0.5, depthWrite: false }));
  pad.rotation.x = -Math.PI / 2;
  pad.position.y = -0.012;
  pad.renderOrder = -4;
  const rim = line([[-hw, 0, -hd], [hw, 0, -hd], [hw, 0, hd], [-hw, 0, hd]], PAL.steel, { closed: true, k: 1 });
  const title = textSprite(`${index + 1} · ${mode.short}`, { size: 1.25, font: 'display', weight: 650, color: '#eef3ff', bg: 'rgba(6,9,18,0.82)', border: withAlpha(ACCENT, 0.55), anchor: [0.5, 0], onTop: true });
  title.position.set(0, 0.25, -hd + 0.3);
  const titleBase = title.scale.clone();
  group.add(pad, rim, title);

  const st = mode.status ?? { at: [0, 0] };
  const status = textSprite(' ', { size: 0.4, weight: 600, color: '#f2f6ff', bg: 'rgba(8,12,22,0.88)', border: withAlpha(ACCENT, 0.75), anchor: [0.5, 0], onTop: true });
  status.position.set(st.at[0], st.y ?? 2.5, st.at[1]);
  status.visible = false;
  inner.add(status);
  let said = '';
  const setStatus = (s) => {
    if (s === said) return;
    said = s;
    status.userData.setText(s);
  };

  const env = { ctx, mode, g, inner, fx, setStatus, dead: false };
  const parts = [];
  if (mode.flows) g.setFlows(mode.flows.map((f) => ({ ...f, color: flowColor(f.kind) })));
  if (st.steps) {
    const P = st.period ?? 1;
    parts.push(() => {
      const u = (g.time % P) / P;
      let s = st.steps[0][1];
      for (const [from, txt] of st.steps) if (u >= from) s = txt;
      setStatus(s);
    });
  }
  if (mode.sim?.type === 'pool') parts.push(poolSim(env, mode.sim));
  if (mode.sim?.type === 'cascade') parts.push(cascadeSim(env, mode.sim));
  if (wall) parts.push(wallProp(env, mode.props));

  return {
    mode,
    g,
    pickables: [pad, title],
    center: new THREE.Vector3(cx, 0.4, cz),
    fit: Math.max(w * 0.52, d * 0.4) + 0.45,
    /** state: 'on' (the selected mode, in focus), 'off' (in the background) or 'all' (overview). */
    setState(state, selected) {
      const on = state === 'on';
      g.setIntensity(state === 'off' ? 0.3 : 1);
      g.showLabels(on);
      if (on) g.showEdgeLabels('all');
      else g.focus(null);
      fx.visible = state !== 'off';
      status.visible = on;
      captions.visible = on;
      title.visible = !on;
      title.scale.copy(titleBase).multiplyScalar(state === 'all' ? 1 : 0.32);
      title.material.opacity = state === 'off' ? 0.75 : 1;
      rim.material.color.copy(glow(selected ? ACCENT : PAL.steel, selected ? 2.1 : 0.55));
      pad.material.opacity = state === 'off' ? 0.3 : 0.5;
    },
    update(dt, t) {
      for (const p of parts) p(dt);
      g.update(dt, t);
    },
    dispose() {
      env.dead = true;
      g.dispose();
    },
  };
}

// ------------------------------------------------------------------ dated facts (data/landscape.js)

const FACTS = {
  closed() {
    const c = FRONTIER.filter((f) => f.weights === 'closed');
    return { text: `Closed-weight flagships, reachable only through an API: ${c.map((f) => f.model).join(', ')}.`, src: c[0].src };
  },
  open() {
    const o = FRONTIER.filter((f) => f.weights === 'open');
    return { text: `Open-weight flagships a cluster of your own can serve: ${o.map((f) => f.model.split(' / ')[0]).join(', ')}.`, src: o[0].src };
  },
  vram: () => TECHNIQUES_2026.find((t) => /VRAM/.test(t.text)),
  trillion: () => TECHNIQUES_2026.find((t) => /trillion/i.test(t.name)),
  ollama: () => ({ text: `Most-pulled models on Ollama, the usual single-machine fare: ${SELF_HOSTED.rows.slice(0, 5).map((r) => r[0]).join(', ')}.`, src: SELF_HOSTED.src }),
};

function datedNotes(mode) {
  const out = (mode.facts ?? []).map((k) => FACTS[k]?.()).filter(Boolean);
  if (mode.notes) {
    const re = new RegExp(mode.notes);
    out.push(...SERVING_NOTES.filter((n) => re.test(n.text)));
  }
  return out;
}

// ------------------------------------------------------------------ the chapter

export async function enter(ctx) {
  const { app, root } = ctx;
  app.setFog(0.003);
  root.add(gridFloor({ size: 320, y: -0.04, color: '#2a4676', opacity: 0.3, fade: 125 }));

  const fromRoute = (route) => {
    const i = MODES.findIndex((m) => m.id === route.rest?.[0]);
    return { i, view: i < 0 || route.query?.view === 'all' ? 'all' : 'mode' };
  };
  const first = fromRoute(ctx.route);
  let sel = Math.max(0, first.i);
  let view = first.view; // 'mode': one stage in focus; 'all': the ten side by side

  const stages = MODES.map((mode, i) => buildStage(ctx, mode, i, onNode));
  const inFocus = (i) => view === 'mode' && i === sel;

  function onNode(i, node) {
    if (!inFocus(i)) choose(i, 'mode');
    else stages[i].g.focus([node.id], { dim: false });
  }
  stages.forEach((s, i) =>
    ctx.pick.add(s.pickables, {
      filter: () => !inFocus(i),
      click: () => choose(i, 'mode'),
      tip: () => `${s.mode.name}\n${s.mode.tagline}`,
    }),
  );
  ctx.pick.onMiss(() => stages[sel].g.focus(null));

  // ---------------------------------------------------------------- panels
  const pickBtns = MODES.map((m, i) =>
    h('button.dep-mode', { type: 'button', role: 'tab', title: m.name, onclick: () => choose(i, 'mode') }, h('span.n', String(i + 1).padStart(2, '0')), h('span.t', m.short)),
  );
  const detail = h('div.dep-detail');
  const head = header({ kicker: 'Deployment modes', title: 'Ten ways to put a model in front of users', lead: 'The same model behaves very differently depending on the machines around it. Pick a mode and watch its requests flow.' });
  head.classList.add('dep-head');
  ctx.ui.left(head, h('div.dep-pick', { role: 'tablist' }, pickBtns), detail);

  const modeBody = h('div.dep-right');
  const allBody = h('div.dep-right');
  const tabsApi = tabs(
    [
      { id: 'mode', label: 'This mode', content: modeBody },
      { id: 'all', label: 'Compare all ten', content: allBody },
    ],
    view,
    (id) => setView(id),
  );
  ctx.ui.right(tabsApi);

  const dockTitle = h('div.dep-now');
  const allBtn = button('All ten', { title: 'All ten modes side by side (O)', onClick: () => setView(view === 'all' ? 'mode' : 'all') });
  ctx.ui.dock(
    h(
      'div.dep-dock',
      button('←', { kind: 'ghost', title: 'Previous mode (←)', onClick: () => choose(sel - 1) }),
      dockTitle,
      button('Next →', { kind: 'primary', title: 'Next mode (→)', onClick: () => choose(sel + 1) }),
      allBtn,
    ),
  );

  function renderDetail(m) {
    detail.replaceChildren(
      h('h2.steptitle', h('span.num', String(sel + 1).padStart(2, '0')), m.name),
      h('div.dep-tag', m.tagline),
      text(m.what),
      callout(m.watch, 'key', 'Watch for'),
      section(m.pathTitle ?? 'Request path', h('ol.dep-path', m.path.map((s) => h('li', rich(s))))),
      section('In the scene', legend(m.legend.map(([kind, label]) => ({ color: KIND[kind], label, shape: 'dot' }))), legend(roleLegend(m.graph))),
      section('Strengths', h('ul.dep-list.pro', m.pros.map((s) => h('li', s)))),
      section('Weaknesses', h('ul.dep-list.con', m.cons.map((s) => h('li', s)))),
    );
  }

  function renderMode(m) {
    const notes = datedNotes(m);
    const meters = meterList(
      RATINGS.map((r) => {
        const v = m.ratings[r.id];
        return { label: r.label, value: v ?? 0, note: v == null ? 'n/a' : v + ' / 5', title: r.hint };
      }),
      { max: 5 },
    );
    modeBody.replaceChildren(
      ...[
        section(
          'Ratings (editorial)',
          meters,
          h('div.hint', 'Editorial judgment for a typical setup, not a measurement. 5 is always the favorable end: fastest, cheapest, most private, least effort, largest model.'),
          m.ratingNote && h('div.hint', m.ratingNote),
        ),
        section('Typical stack', m.stack.map(([role, items]) => h('div.dep-stack', h('div.role', role), chips(items)))),
        section('Choose it when', h('ul.dep-list.pick', m.choose.map((s) => h('li', s)))),
        notes.length > 0 && section('Dated facts · ' + AS_OF, notes.map((n) => h('div.dep-note', n.text, ' ', h('a', { href: n.src, target: '_blank', rel: 'noopener' }, 'source')))),
      ].filter(Boolean),
    );
  }

  function renderAll() {
    allBody.replaceChildren(
      h(
        'table.tbl.compact.dep-cmp',
        h('thead', h('tr', h('th', 'Mode'), RATINGS.map((r) => h('th', { title: `${r.label} (${r.hint})` }, r.short)))),
        h(
          'tbody',
          MODES.map((m, i) =>
            h(
              'tr' + (i === sel ? '.on' : ''),
              { onclick: () => choose(i, i === sel ? 'mode' : undefined) },
              h('td', h('span.n', String(i + 1)), m.short),
              RATINGS.map((r) => {
                const v = m.ratings[r.id];
                return h('td', h('span.dep-cell', { style: v == null ? null : { background: seqCss(0.1 + 0.085 * v) }, title: `${m.name} · ${r.label}: ${v == null ? 'not applicable' : v + ' / 5'}` }, v == null ? '–' : String(v)));
              }),
            ),
          ),
        ),
      ),
      h('div.hint', RATINGS.map((r) => `${r.short} = ${r.label.toLowerCase()}`).join(' · ') + '.'),
      h('div.hint', 'Editorial ratings from 1 to 5, not measurements. 5 is always the favorable end: fastest, cheapest, most private, least effort, largest model. Select a row to highlight a mode, select it again to zoom in.'),
    );
  }

  // ---------------------------------------------------------------- state
  const viewOf = () => {
    if (view === 'all') return { pos: [ALL_DIR[0], ALL_DIR[1], 1.5 + ALL_DIR[2]], target: [0, 0, 1.5], fit: 31, ...LIMITS };
    const s = stages[sel];
    return { pos: [s.center.x + MODE_DIR[0], s.center.y + MODE_DIR[1], s.center.z + MODE_DIR[2]], target: s.center.toArray(), fit: s.fit, ...LIMITS };
  };

  let shown = -1;
  function sync(o = {}) {
    const m = MODES[sel];
    stages.forEach((s, i) => s.setState(view === 'all' ? 'all' : i === sel ? 'on' : 'off', i === sel));
    pickBtns.forEach((b, i) => {
      b.classList.toggle('on', i === sel);
      b.setAttribute('aria-selected', String(i === sel));
    });
    // on phones the picker is one scrolling row: keep the current mode in view
    const row = pickBtns[sel].parentElement;
    if (row) row.scrollLeft += pickBtns[sel].getBoundingClientRect().left - row.getBoundingClientRect().left - (row.clientWidth - pickBtns[sel].offsetWidth) / 2;
    if (shown !== sel) {
      shown = sel;
      renderDetail(m);
      renderMode(m);
      if (detail.parentElement) detail.parentElement.scrollTop = 0;
    }
    renderAll();
    if (tabsApi.get() !== view) tabsApi.set(view);
    dockTitle.replaceChildren(h('b', `${sel + 1} / ${MODES.length}`), m.name);
    allBtn.textContent = view === 'all' ? 'Zoom in' : 'All ten';
    allBtn.classList.toggle('on', view === 'all');
    if (!o.silent) ctx.router.replace('deploy/' + m.id, view === 'all' ? { view: 'all' } : undefined);
    if (o.instant) app.setView(viewOf());
    else flyFit(app, viewOf(), 1.2);
  }

  function choose(i, toView) {
    sel = (i + MODES.length) % MODES.length;
    if (toView) view = toView;
    sync();
  }
  function setView(v) {
    if (v === view) return;
    view = v;
    sync();
  }

  ctx.key('ArrowRight', () => choose(sel + 1));
  ctx.key('ArrowLeft', () => choose(sel - 1));
  ctx.key('o', () => setView(view === 'all' ? 'mode' : 'all'));
  ctx.key('Escape', () => setView('all'));

  sync({ instant: true, silent: true });

  return {
    update(dt, t) {
      for (const s of stages) s.update(dt, t);
    },
    onRoute(route) {
      const r = fromRoute(route);
      if (r.i >= 0) sel = r.i;
      view = r.view;
      sync({ silent: true });
    },
    dispose() {
      for (const s of stages) s.dispose();
    },
  };
}
