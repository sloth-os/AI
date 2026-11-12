// Data Flow: the major pipelines of AI systems, each one a guided walk along a 3D node-link diagram.
// The pipelines are plain data (../data/pipelines.js); this file draws the chosen one with the
// flowgraph engine, steps through its stages and fills the panels.
// Route: #/flow/<pipelineId>?s=<stage>
import './flow.css';
import { PIPELINES } from '../data/pipelines.js';
import { PROTOCOLS, SERVING_NOTES, AS_OF } from '../data/landscape.js';
import { createFlowGraph, flyFit, roleLegend } from '../core/flowgraph.js';
import { gridFloor } from '../core/geom.js';
import { h, header, section, text, button, select, kv, code, legend, tour } from '../core/ui.js';

const ACCENT = '#4de3b0';
const ELEV = 0.6; // tangent of the camera's elevation
const viewDir = (yaw) => [Math.sin(yaw), ELEV, Math.cos(yaw)]; // from the scene toward the camera
const DIR = viewDir(0.19);
const LIMITS = { min: 3, max: 110, maxPolar: Math.PI * 0.495 };

/** Dated facts a stage can cite with `ref`: they are read from landscape.js so the two never disagree. */
const protocolNote = (id) => {
  const p = PROTOCOLS.find((x) => x.id === id);
  return p && { title: p.name, text: p.status, src: p.src };
};
const servingNote = (re, title) => {
  const n = SERVING_NOTES.find((x) => re.test(x.text));
  return n && { title, text: n.text, src: n.src };
};
const REFS = {
  mcp: () => protocolNote('mcp'),
  a2a: () => protocolNote('a2a'),
  engines: () => servingNote(/vLLM and SGLang/, 'Serving engines'),
  throughput: () => servingNote(/ShareGPT/, 'One independent measurement'),
  webgpu: () => servingNote(/WebGPU now ships/, 'In the browser'),
};

const host = (url) => {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
};

export async function enter(ctx) {
  const { app, root } = ctx;
  root.add(gridFloor({ size: 140, y: -0.03, color: '#2a5f6e', opacity: 0.32 }));

  let pipe = null; // the pipeline on stage
  let graph = null; // its FlowGraph
  let guide = null; // its tour
  let overview = false;

  // ---------------------------------------------------------------- panels (built once)
  const pickSel = select({
    label: 'Pipeline',
    value: PIPELINES[0].id,
    options: PIPELINES.map((p, i) => ({ value: p.id, label: `${i + 1}. ${p.name}` })),
    onChange: (id) => choose(id),
  });
  const pickBtns = new Map();
  const pickGrid = h(
    'div.flow-pick',
    { role: 'tablist', 'aria-label': 'Pipelines' },
    PIPELINES.map((p, i) => {
      const b = h('button', { type: 'button', role: 'tab', title: `${p.name}: ${p.tagline}`, onclick: () => choose(p.id) }, h('span.n', String(i + 1).padStart(2, '0')), h('span.t', p.short));
      pickBtns.set(p.id, b);
      return b;
    }),
  );
  const pickWrap = h('div.flow-select', pickSel.el); // shown instead of the grid on small screens
  const blurb = h('div.flow-blurb');
  const tourSlot = h('div.flow-tour');
  const detail = h('div.flow-detail');
  const overviewBtn = button('Whole pipeline', { title: 'See every stage and every wire format at once', onClick: () => (overview ? showStage(guide.index) : showOverview()) });

  ctx.ui.left(
    h('div.flow-head', header({ kicker: 'Data Flow', title: 'How data is transformed and exchanged', lead: 'Ten pipelines, one stage at a time: what the data looks like, its shape and type, and the wire format on every hop.' })),
    pickWrap,
    pickGrid,
    blurb,
    tourSlot,
  );
  ctx.ui.right(detail);

  // ---------------------------------------------------------------- one stage
  const overviewView = () => ({ ...graph.view(null, { dir: pipe.dir ?? DIR, pad: pipe.pad ?? 0.9 }), ...LIMITS });

  /**
   * Viewing direction for a stage. When the data just moved toward or away from the default camera
   * (the previous node sits behind or in front of the active one), the camera swings part of the
   * way to the side, so that the two nodes stand next to each other and the exhibit does not hide
   * where the data came from.
   */
  function stageDir(st) {
    if (st.dir) return st.dir;
    const a = !st.frame && st.from ? graph.nodes.get(st.from) : null;
    const b = graph.nodes.get(st.node);
    if (a && b && Math.abs(b.z - a.z) > Math.abs(b.x - a.x) * 1.2) return viewDir(b.z > a.z ? -0.6 : 0.6);
    return pipe.dir ?? DIR;
  }

  function frameStage(st, dur = 1.3) {
    const ids = st.frame ?? [st.from, st.node].filter(Boolean);
    const o = { dir: stageDir(st), pad: st.pad ?? 0.95, minFit: 3.2 };
    const v = graph.view(ids, o);
    // Keep the exhibit legible: when the framed nodes are far apart, stay close to the active node
    // (its exhibit always remains in view) and show only as much of the way back as fits.
    const cap = st.maxFit ?? (st.frame ? 6.2 : 4.8);
    if (v.fit > cap) {
      const near = graph.view([st.node], o);
      const dist = Math.hypot(...near.target.map((x, i) => v.target[i] - x));
      const t = dist > 1e-6 ? Math.min(1, Math.max(0, (cap - near.fit) / dist)) : 0;
      const target = near.target.map((x, i) => x + (v.target[i] - x) * t);
      v.pos = target.map((x, i) => x + v.pos[i] - v.target[i]);
      v.target = target;
      v.fit = Math.max(cap, near.fit);
    }
    flyFit(app, { ...v, ...LIMITS }, dur);
  }

  /** Left panel, under the step text: the facts to remember about the data at this stage. */
  const glance = (st) =>
    h('div.flow-glance', kv([['Shape', h('code.flow-shape', st.shape)], ['Type', st.dtype], st.size && ['Size', st.size], ['Arrived as', st.wire.name]]));

  function renderDetail(st) {
    const ref = st.ref && REFS[st.ref]?.();
    const p = st.payload;
    detail.replaceChildren(
      ...[
        section('Example payload', p.label && h('div.flow-cap', p.label), code(p.code, p.lang ?? ''), p.note && text(p.note, 'small')),
        section('Edge just crossed', h('div.flow-wire', st.wire.name), text(st.wire.detail, 'small')),
        section('Where the time and cost go', text(st.cost)),
        st.links?.length && section('Go deeper', h('div.flow-links', st.links.map((l) => h('a', { href: l.href }, l.label + ' →')))),
        ref && section(`Dated fact · ${AS_OF}`, text(`**${ref.title}.** ${ref.text}`, 'small'), h('a.flow-src', { href: ref.src, target: '_blank', rel: 'noopener' }, 'Source: ' + host(ref.src))),
        section('This pipeline', text(pipe.model, 'small'), legend(roleLegend(pipe.graph)), h('div.btnrow', overviewBtn)),
      ].filter(Boolean),
    );
    const body = detail.parentElement;
    if (body) body.scrollTop = 0;
  }

  function showStage(k) {
    const st = pipe.stages[k];
    if (!st) return;
    overview = false;
    overviewBtn.textContent = 'Whole pipeline';
    const edges = st.edges ?? [];
    const frameIds = st.frame ?? [st.from, st.node].filter(Boolean);
    graph.showEdgeLabels('active');
    graph.focus([st.node, ...(st.from ? [st.from] : []), ...(st.also ?? [])], { ring: st.node });
    graph.setActiveEdges(edges.map((e) => e.replace(/^-/, '')));
    graph.setFlows(st.flows ?? edges.map((e, i) => ({ path: e, count: 3, speed: 2.4, phase: i * 0.21 })));
    // the data at this stage, plus a small ghost of what it was one stage earlier
    const shown = st.exhibit ? [{ node: st.node, spec: st.exhibit, lift: st.lift }] : [];
    const before = pipe.stages[k - 1];
    if (before?.exhibit && before.node !== st.node && frameIds.includes(before.node)) shown.push({ node: before.node, spec: before.exhibit, ghost: true, lift: before.lift });
    for (const x of st.extras ?? []) shown.push({ node: x.node, spec: x.spec, ghost: !!x.ghost });
    graph.setExhibits(shown);
    frameStage(st);
    renderDetail(st);
    if (window.innerWidth <= 860 && pickWrap.isConnected) pickWrap.scrollIntoView({ block: 'start' }); // keep the picker in reach
  }

  function showOverview() {
    overview = true;
    overviewBtn.textContent = 'Back to this stage';
    graph.focus(null);
    graph.setActiveEdges(null);
    graph.showEdgeLabels('all');
    graph.clearExhibits();
    graph.setFlows(pipe.graph.edges.map((e, i) => ({ path: e.id, count: 2, speed: 2.2, phase: i * 0.13 })));
    flyFit(app, overviewView(), 1.2);
  }

  function nodeClicked(n) {
    const at = pipe.stages.map((s, i) => (s.node === n.id ? i : -1)).filter((i) => i >= 0);
    if (!at.length) return;
    const target = at.find((i) => i > guide.index) ?? at[0];
    if (target === guide.index) showStage(target);
    else guide.go(target);
  }

  // ---------------------------------------------------------------- one pipeline
  const choose = (id) => {
    if (id !== pipe?.id) ctx.router.go('flow/' + id, { s: '1' });
  };

  function load(id, start = 0) {
    pipe = PIPELINES.find((p) => p.id === id) ?? PIPELINES[0];
    const mine = pipe;
    graph?.dispose();
    const stagesAt = (nodeId) => pipe.stages.map((s, i) => (s.node === nodeId ? `Stage ${i + 1}: ${s.title}` : null)).filter(Boolean);
    graph = createFlowGraph(
      ctx,
      { ...pipe.graph, nodes: pipe.graph.nodes.map((n) => ({ ...n, tip: n.tip ?? (stagesAt(n.id).join('\n') || undefined) })) },
      { accent: ACCENT, packets: 220, labelSize: 0.36, onClick: nodeClicked },
    );
    root.add(graph.group);
    overview = false;
    app.setView(overviewView());

    pickSel.set(pipe.id);
    for (const [pid, b] of pickBtns) {
      b.classList.toggle('on', pid === pipe.id);
      b.setAttribute('aria-selected', String(pid === pipe.id));
    }
    blurb.replaceChildren(text(`**${pipe.name}.** ${pipe.tagline}`, 'small'));

    const next = PIPELINES[PIPELINES.indexOf(pipe) + 1];
    start = Math.max(0, Math.min(pipe.stages.length - 1, start));
    const steps = pipe.stages.map((st, k) => ({
      title: st.title,
      text: st.what,
      body: () => glance(st),
      enter: () => pipe === mine && showStage(k),
    }));
    guide = tour(ctx, steps, {
      start,
      lastLabel: next ? `Next: ${next.short} →` : 'Next chapter →',
      onFinish: () => (next ? choose(next.id) : ctx.router.go(app.nextNav('flow').path)),
    });
    tourSlot.replaceChildren(guide.body);
    ctx.ui.dock(guide.dock);
    ctx.router.replace('flow/' + pipe.id, { s: String(start + 1) });
  }

  const stepOf = (route) => Math.max(1, parseInt(route.query.s, 10) || 1) - 1;
  load(ctx.route.rest?.[0], stepOf(ctx.route));

  return {
    update(dt, t) {
      graph?.update(dt, t);
    },
    onRoute(route) {
      const id = route.rest?.[0];
      const s = stepOf(route);
      if (id && id !== pipe.id && PIPELINES.some((p) => p.id === id)) load(id, s);
      else if (guide.index >= 0 && s !== guide.index) guide.go(s, true);
    },
  };
}
