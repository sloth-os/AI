// The hub: a living "AI core" with one portal per chapter in orbit around it.
import * as THREE from 'three';
import { NAV } from '../routes.js';
import { PAL, glow } from '../core/colors.js';
import { glowMat, segments, edges, ring, halo, points, rng, arc, line } from '../core/geom.js';
import { makePod, layoutRing } from '../core/gallery.js';
import { h, header, button, section, statRow, stat } from '../core/ui.js';
import { HOT_AS_OF } from '../data/hot.js';

const PORTALS = [
  { nav: 'families', icon: 'constellation', sub: 'Every kind of AI, on one map', tip: 'Symbolic AI, classical ML, deep learning, generative models, reinforcement learning, agents: how they relate and when each appeared.' },
  { nav: 'labs', icon: 'network', sub: 'Open the machines and run them', tip: 'Seven hands-on labs: a real GPT, a CNN you can draw for, diffusion, reinforcement learning, optimisers, embeddings.' },
  { nav: 'problems', icon: 'tower', sub: 'Eight levels, six requirements', tip: 'From data and chips up to agents and safety: the problems at each level, the competing solutions and their trade-offs.' },
  { nav: 'flow', icon: 'pipeline', sub: 'How data transforms and travels', tip: 'Follow one request through tokenisation, tensors, retrieval, tool calls and protocols such as MCP.' },
  { nav: 'perf', icon: 'bars', sub: 'Measured live on your machine', tip: 'Quantisation, KV caching, batching, model bake-offs and the record books.' },
  { nav: 'ecosystem', icon: 'galaxy', sub: 'The projects that run AI', tip: 'Frameworks, models, serving engines, vector databases, agent tools: what each does and the algorithm at its core.' },
  { nav: 'deploy', icon: 'cloud', sub: 'From data centre to browser tab', tip: 'Ten deployment modes compared: cloud API, self-hosted, Kubernetes, serverless, edge, in-browser, hybrid, federated.' },
  { nav: 'hot', icon: 'flame', sub: `Top 10, as of ${HOT_AS_OF}`, tip: 'The ten hottest issues in AI right now, each with its own 3D animation and dated sources.' },
];

function buildCore(ctx) {
  const g = new THREE.Group();
  const rnd = rng(42);
  const N = 150;
  const R = 2.6;
  const nodes = [];
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const a = i * Math.PI * (3 - Math.sqrt(5));
    nodes.push(new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r).multiplyScalar(R * (0.93 + rnd() * 0.14)));
  }
  // connect each node to its nearest neighbours: the "synapses"
  const pairs = [];
  const links = [];
  for (let i = 0; i < N; i++) {
    const near = nodes
      .map((p, j) => [p.distanceToSquared(nodes[i]), j])
      .sort((a, b) => a[0] - b[0])
      .slice(1, 4);
    for (const [, j] of near)
      if (j > i) {
        pairs.push(nodes[i], nodes[j]);
        links.push([i, j]);
      }
  }
  g.add(segments(pairs, PAL.cyan, { k: 0.9, opacity: 0.4 }));

  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const size = new Float32Array(N);
  const tones = [PAL.cyan, PAL.violet, PAL.magenta, PAL.blue];
  nodes.forEach((p, i) => {
    p.toArray(pos, i * 3);
    const c = glow(tones[i % tones.length], 1.6 + rnd() * 1.6);
    col.set([c.r, c.g, c.b], i * 3);
    size[i] = 0.08 + rnd() * 0.09;
  });
  g.add(points(pos, col, size));

  // signals travelling along the synapses
  const P = 70;
  const ppos = new Float32Array(P * 3);
  const pcol = new Float32Array(P * 3);
  const psize = new Float32Array(P).fill(0.12);
  const pulses = Array.from({ length: P }, () => ({ link: links[rnd.int(links.length)], t: rnd(), speed: 0.4 + rnd() * 0.9 }));
  const white = glow('#ffffff', 2.6);
  for (let i = 0; i < P; i++) pcol.set([white.r, white.g, white.b], i * 3);
  const pulsePts = points(ppos, pcol, psize);
  g.add(pulsePts);

  const inner = edges(new THREE.IcosahedronGeometry(1.25, 1), PAL.violet, { k: 1.5, opacity: 0.8 });
  const heart = new THREE.Mesh(new THREE.IcosahedronGeometry(0.5, 2), glowMat('#bfe6ff', { k: 1.25 }));
  g.add(inner, heart, halo(PAL.cyan, 12, 0.16), halo(PAL.violet, 6.5, 0.14));

  const tmp = new THREE.Vector3();
  return {
    group: g,
    update(dt, t) {
      g.rotation.y = t * 0.07;
      inner.rotation.set(t * 0.21, -t * 0.16, 0);
      heart.scale.setScalar(1 + 0.07 * Math.sin(t * 2.1));
      for (let i = 0; i < P; i++) {
        const p = pulses[i];
        p.t += dt * p.speed;
        if (p.t >= 1) {
          p.t = 0;
          p.link = links[rnd.int(links.length)];
        }
        tmp.lerpVectors(nodes[p.link[0]], nodes[p.link[1]], p.t).toArray(ppos, i * 3);
      }
      pulsePts.geometry.attributes.position.needsUpdate = true;
    },
  };
}

export async function enter(ctx) {
  const { app, root } = ctx;
  app.setView({ pos: [0, 8.5, 25], target: [0, -0.6, 0], min: 9, max: 60, maxPolar: Math.PI * 0.62, autoRotate: 0.3 });
  app.setBloom({ strength: 0.5, radius: 0.7 });
  app.setFog(0.004);

  const core = buildCore(ctx);
  root.add(core.group);

  const RADIUS = 12;
  const r1 = ring(RADIUS, PAL.steel, { k: 0.8, opacity: 0.22, segments: 160 });
  r1.position.y = -0.95;
  const r2 = ring(RADIUS * 0.62, PAL.cyan, { k: 0.8, opacity: 0.12, segments: 120 });
  r2.rotation.x = 0.35;
  const r3 = ring(RADIUS * 1.35, PAL.violet, { k: 0.7, opacity: 0.1, segments: 180 });
  r3.rotation.z = 0.2;
  root.add(r1, r2, r3);

  const pods = PORTALS.map((p, i) => {
    const nav = NAV.find((n) => n.id === p.nav);
    return makePod(ctx, {
      title: nav.label,
      sub: p.sub,
      kicker: String(i + 1).padStart(2, '0'),
      color: nav.accent,
      icon: p.icon,
      tip: p.tip,
      scale: 1.35,
      onClick: () => ctx.router.go(nav.path),
    });
  });
  layoutRing(pods, { radius: RADIUS, start: Math.PI / PORTALS.length });
  pods.forEach((p) => root.add(p.group));

  // streams of data between the core and every portal
  const streams = [];
  const S = 9;
  const spos = new Float32Array(pods.length * S * 3);
  const scol = new Float32Array(pods.length * S * 3);
  const ssize = new Float32Array(pods.length * S).fill(0.11);
  pods.forEach((p, pi) => {
    const end = p.group.position.clone().multiplyScalar(0.86);
    const start = end.clone().normalize().multiplyScalar(2.9);
    const curve = arc(start, end, 2.2 + (pi % 2) * 0.9);
    root.add(line(curve.getPoints(40), p.item.color, { k: 0.9, opacity: 0.2 }));
    const c = glow(p.item.color, 2.4);
    for (let j = 0; j < S; j++) {
      scol.set([c.r, c.g, c.b], (pi * S + j) * 3);
      streams.push({ curve, offset: j / S, dir: pi % 2 ? 1 : -1 });
    }
  });
  const streamPts = points(spos, scol, ssize);
  root.add(streamPts);

  ctx.ui.left(
    header({
      kicker: 'An immersive 3D atlas',
      title: 'How AI actually works',
      lead: 'Fly through the whole field: from a single neuron to trillion-parameter models, from one token to a fleet of agents. Nothing here is a video. **Every model on this site runs live in your browser**, so you can poke it and watch what changes.',
    }),
    h('div.btnrow', button('Start the tour →', { kind: 'primary', onClick: () => ctx.router.go('families') }), button('Open a real GPT', { onClick: () => ctx.router.go('labs/gpt') })),
    statRow(
      stat({ label: 'Live labs', value: '7' }),
      stat({ label: 'Benchmarks', value: '5', sub: 'measured locally' }),
      stat({ label: 'Hot issues', value: '10', sub: HOT_AS_OF }),
    ),
    section(
      'Chapters',
      PORTALS.map((p, i) => {
        const nav = NAV.find((n) => n.id === p.nav);
        return h(
          'a.card.click.chaplink',
          { href: '#/' + nav.path, style: { '--accent': nav.accent } },
          h('h4', h('span.n', String(i + 1).padStart(2, '0')), nav.label),
          h('p', p.tip),
        );
      }),
    ),
  );
  ctx.ui.dock(h('div.dockrow', h('span.hint', 'Drag to orbit · scroll to zoom · click a portal to dive in · press ? for help')));

  const tmp = new THREE.Vector3();
  return {
    update(dt, t) {
      core.update(dt, t);
      for (const p of pods) p.update(dt, t);
      r2.rotation.y = t * 0.05;
      r3.rotation.y = -t * 0.03;
      streams.forEach((s, i) => {
        let f = (s.offset + t * 0.07 * s.dir) % 1;
        if (f < 0) f += 1;
        s.curve.getPoint(f, tmp).toArray(spos, i * 3);
      });
      streamPts.geometry.attributes.position.needsUpdate = true;
    },
  };
}
