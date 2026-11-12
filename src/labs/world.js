// Plan vs Learn: one maze, two philosophies. With a map, search algorithms plan a route; without
// one, reinforcement learners discover it from reward alone. All algorithms are in ../ml/gridworld.js.
import * as THREE from 'three';
import { Grid, PRESETS, SEARCHERS, LEARNERS, DEFAULT_PARAMS, Agent, Trial, compareSearch, smooth, FLOOR, WALL, MUD, PIT, TILE_NAMES, ACTIONS, REWARD } from '../ml/gridworld.js';
import { PAL, SERIES, glow, seq } from '../core/colors.js';
import { instanced, flushInst, gridFloor, glowMat, ring, rbox } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, segmented, slider, select, tour, code, table, stat, statRow, legend, fmt } from '../core/ui.js';
import { lineChart, barChart } from '../core/charts.js';
import { damp } from '../core/tween.js';

const SCOLOR = Object.fromEntries(SEARCHERS.map((s, i) => [s.id, SERIES[i]]));
const LCOLOR = Object.fromEntries(LEARNERS.map((l, i) => [l.id, SERIES[i]]));
const TILE_COLOR = { [FLOOR]: '#16223d', [WALL]: '#3b4a6b', [MUD]: '#7a5a1e', [PIT]: '#5a0f18' };
const EPISODES = 250;

export async function enter(ctx) {
  const { app, root } = ctx;
  let grid = Grid.preset('terrain');
  const N = grid.n;
  const W = grid.w;
  const H = grid.h;
  app.setView({ pos: [0, 15, 12], target: [0, 0, 0.4], fit: 9.6, min: 4, max: 60, maxPolar: Math.PI * 0.47 });
  root.add(gridFloor({ size: 70, y: -0.55, color: '#3f5a2a', opacity: 0.25 }));
  const tx = (i) => grid.x(i) - (W - 1) / 2;
  const tz = (i) => grid.y(i) - (H - 1) / 2;

  const S = { mode: 'plan', algo: 'astar', learner: 'q', playing: true, speed: 30, k: 0, results: [], agent: null, trials: [], target: EPISODES, preset: 'terrain' };
  const params = { ...DEFAULT_PARAMS };

  // ------------------------------------------------------------------ 3D: board
  const tileGeo = rbox(0.94, 1, 0.94, 0.05);
  tileGeo.translate(0, 0.5, 0);
  const tiles = instanced(tileGeo, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.1 }), N);
  root.add(tiles);
  const arrowGeo = new THREE.ConeGeometry(0.1, 0.26, 4);
  arrowGeo.rotateX(Math.PI / 2); // points along +z
  const arrows = instanced(arrowGeo, new THREE.MeshBasicMaterial(), N * 4);
  root.add(arrows);
  const pathGeo = new THREE.BufferGeometry();
  pathGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  const pathLine = new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: glow('#ffffff', 2.4) }));
  pathLine.frustumCulled = false;
  root.add(pathLine);
  const agentMesh = new THREE.Mesh(rbox(0.46, 0.46, 0.46, 0.1), glowMat('#ffffff', { k: 2.2 }));
  const goal = new THREE.Mesh(new THREE.OctahedronGeometry(0.3), glowMat('#ffd166', { k: 2.6 }));
  const startRing = ring(0.4, '#ffffff', { k: 2, segments: 40 });
  const sLab = textSprite('START', { size: 0.26, font: 'mono', weight: 700, color: '#dfe7f7' });
  const gLab = textSprite('GOAL', { size: 0.26, font: 'mono', weight: 700, color: '#ffe9a8' });
  root.add(agentMesh, goal, startRing, sLab, gLab);
  const agentTarget = new THREE.Vector3();

  const m4 = new THREE.Matrix4();
  const q4 = new THREE.Quaternion();
  const e3 = new THREE.Euler();
  const p3 = new THREE.Vector3();
  const s3 = new THREE.Vector3();
  const col = new THREE.Color();
  const heightOf = (c) => (c === WALL ? 0.95 : c === PIT ? 0.03 : 0.14);

  /** Repaint every tile: terrain colors, plus the search wave or the learned values on top. */
  function paint() {
    const res = S.mode === 'plan' ? S.results.find((r) => r.algo === S.algo) : null;
    const ag = S.mode === 'learn' ? S.agent : null;
    let vmin = Infinity;
    let vmax = -Infinity;
    if (ag)
      for (let i = 0; i < N; i++) {
        const v = ag.value(i);
        if (!Number.isNaN(v) && grid.cells[i] !== PIT) {
          vmin = Math.min(vmin, v);
          vmax = Math.max(vmax, v);
        }
      }
    const onPath = new Set(res && S.k >= res.order.length ? res.path : []);
    for (let i = 0; i < N; i++) {
      const c = grid.cells[i];
      m4.compose(p3.set(tx(i), c === PIT ? -0.25 : 0, tz(i)), q4.identity(), s3.set(1, heightOf(c), 1));
      tiles.setMatrixAt(i, m4);
      col.set(TILE_COLOR[c]);
      if (c === PIT) col.multiplyScalar(1.6);
      if (res && c !== WALL && c !== PIT) {
        const e = res.expandedAt[i];
        const d = res.discoveredAt[i];
        if (onPath.has(i)) col.copy(glow('#ffffff', 0.85));
        else if (e >= 0 && e < S.k) seq(0.3 + 0.5 * (1 - e / Math.max(1, res.order.length)), col).multiplyScalar(c === MUD ? 0.75 : 1);
        else if (d >= 0 && d <= S.k) col.copy(glow(SCOLOR[S.algo], 1.5));
      }
      if (ag && c !== WALL && c !== PIT) {
        const v = ag.value(i);
        if (!Number.isNaN(v)) seq(0.18 + 0.72 * ((v - vmin) / Math.max(1e-9, vmax - vmin)), col).multiplyScalar(c === MUD ? 0.7 : 1);
      }
      tiles.setColorAt(i, col);
      // four action-value arrows per tile (learning mode only)
      for (let a = 0; a < 4; a++) {
        const k = i * 4 + a;
        let s = 0;
        if (ag && c !== WALL && c !== PIT && i !== grid.goal && ag.tried[k] > 0) {
          const u = (ag.q[k] - vmin) / Math.max(1e-9, vmax - vmin);
          s = 0.35 + 0.9 * Math.max(0, Math.min(1, u));
          arrows.setColorAt(k, col.copy(glow(a === bestTried(ag, i) ? '#ffffff' : PAL.lime, a === bestTried(ag, i) ? 1.6 : 0.5 + u)));
        }
        e3.set(0, Math.atan2(ACTIONS[a][0], ACTIONS[a][1]), 0);
        m4.compose(p3.set(tx(i) + ACTIONS[a][0] * 0.3, heightOf(c) + 0.06, tz(i) + ACTIONS[a][1] * 0.3), q4.setFromEuler(e3), s3.setScalar(Math.max(s, 1e-4)));
        arrows.setMatrixAt(k, m4);
      }
    }
    flushInst(tiles);
    flushInst(arrows);
    tiles.computeBoundingSphere();
    // route: the planned path, or the learner's current greedy policy
    let route = [];
    if (res && S.k >= res.order.length) route = res.path;
    if (ag) route = ag.greedyPath().path;
    const pp = pathGeo.attributes.position.array;
    route.forEach((i, n) => pp.set([tx(i), heightOf(grid.cells[i]) + 0.22, tz(i)], n * 3));
    pathGeo.setDrawRange(0, route.length);
    pathGeo.attributes.position.needsUpdate = true;
    pathLine.material.color.copy(glow(ag ? LCOLOR[S.learner] : '#ffffff', 2.4));
  }
  function bestTried(ag, i) {
    let b = -1;
    for (let a = 0; a < 4; a++) if (ag.tried[i * 4 + a] > 0 && (b < 0 || ag.q[i * 4 + a] > ag.q[i * 4 + b])) b = a;
    return b;
  }
  function placeMarkers() {
    goal.position.set(tx(grid.goal), 0.75, tz(grid.goal));
    startRing.position.set(tx(grid.start), 0.17, tz(grid.start));
    sLab.position.set(tx(grid.start), 0.75, tz(grid.start));
    gLab.position.set(tx(grid.goal), 1.35, tz(grid.goal));
    agentTarget.set(tx(grid.start), 0.42, tz(grid.start));
    agentMesh.position.copy(agentTarget);
  }

  // ------------------------------------------------------------------ logic
  function replan() {
    S.results = compareSearch(grid);
    S.k = 0;
    renderPlan();
  }
  function relearn() {
    S.agent = new Agent(grid, S.learner, params, 11);
    S.trials = LEARNERS.map((l) => new Trial(grid, l.id, params, { runs: 8, seed: 3 }));
    S.target = EPISODES;
    renderLearn();
  }
  function rebuild() {
    placeMarkers();
    replan();
    relearn();
    paint();
  }
  function setMode(mode) {
    S.mode = mode;
    modeSeg.set(mode);
    planBox.hidden = mode !== 'plan';
    learnBox.hidden = mode !== 'learn';
    S.k = 0;
    S.playing = true;
    playBtn.textContent = '❚❚ Pause';
    agentTarget.set(tx(grid.start), 0.42, tz(grid.start));
    paint();
  }

  // ------------------------------------------------------------------ HUD
  const modeSeg = segmented([{ value: 'plan', label: 'Plan (map known)' }, { value: 'learn', label: 'Learn (no map)' }], S.mode, setMode);
  const presetSel = select({ label: 'World', options: PRESETS.map((p) => ({ value: p.id, label: p.name })), value: S.preset, onChange: (id) => loadPreset(id) });
  const presetBlurb = h('div.txt.small');
  const playBtn = button('❚❚ Pause', { kind: 'primary', onClick: () => ((S.playing = !S.playing), (playBtn.textContent = S.playing ? '❚❚ Pause' : '▶ Run')) });
  const algoSeg = segmented(SEARCHERS.map((s) => ({ value: s.id, label: s.short })), S.algo, (v) => ((S.algo = v), (S.k = 0), renderPlan()));
  const planInfo = h('div');
  const expandBars = barChart({ rows: [], unit: 'tiles', title: 'Tiles expanded before the goal was found' });
  const planTable = h('div');
  const learnSeg = segmented(LEARNERS.map((l) => ({ value: l.id, label: l.name })), S.learner, (v) => ((S.learner = v), (S.agent = new Agent(grid, v, params, 11)), renderLearn()));
  const learnInfo = h('div');
  const stEp = stat({ label: 'Episodes (watched agent)', value: '0' });
  const stRet = stat({ label: 'Last return', value: '–' });
  const retChart = lineChart({ height: 140, xLabel: 'episode', yLabel: 'mean return of 8 runs (smoothed)', series: [] });
  const stepChart = lineChart({ height: 110, xLabel: 'episode', yLabel: 'steps per episode', series: [], legend: false });
  const planBox = h('div.tabbody', section('Algorithm to watch', algoSeg), planInfo, expandBars, planTable);
  const learnBox = h(
    'div.tabbody',
    section('Learner to watch', learnSeg),
    learnInfo,
    statRow(stEp, stRet),
    slider({ label: 'Exploration ε', min: 0, max: 0.5, step: 0.01, value: params.epsilon, format: (v) => v.toFixed(2), onChange: () => relearn(), onInput: (v) => (params.epsilon = v) }),
    slider({ label: 'Learning rate α', min: 0.05, max: 1, step: 0.05, value: params.alpha, format: (v) => v.toFixed(2), onChange: () => relearn(), onInput: (v) => (params.alpha = v) }),
    slider({ label: 'Discount γ', min: 0.8, max: 1, step: 0.01, value: params.gamma, format: (v) => v.toFixed(2), onChange: () => relearn(), onInput: (v) => (params.gamma = v) }),
    h('div.btnrow', button('+250 episodes', { onClick: () => (S.target += 250) }), button('Forget everything', { onClick: () => relearn() })),
    section('Return per episode', retChart),
    section(null, stepChart),
  );

  function renderPlan() {
    const cur = SEARCHERS.find((s) => s.id === S.algo);
    const res = S.results.find((r) => r.algo === S.algo);
    planInfo.replaceChildren(text(`**${cur.name}** expands the ${cur.rule}. ${cur.blurb}`, 'small'));
    expandBars.update(S.results.map((r) => ({ label: SEARCHERS.find((s) => s.id === r.algo).short, value: r.expanded, color: SCOLOR[r.algo], active: r.algo === S.algo, onClick: () => (algoSeg.set(r.algo), (S.algo = r.algo), (S.k = 0), renderPlan()) })));
    planTable.replaceChildren(
      table(
        ['', 'Expanded', 'Path cost', 'Cheapest?'],
        S.results.map((r) => [SEARCHERS.find((s) => s.id === r.algo).short, String(r.expanded), r.found ? String(r.cost) : 'no path', r.found ? (r.optimal ? 'yes' : 'no') : '–']),
        { compact: true },
      ),
      text(`A step costs 1, a step into mud ${-REWARD.mud}. Pits are never entered when planning. ${res && !res.found ? 'There is no route to the goal in this world.' : ''}`, 'small'),
    );
  }
  function renderLearn() {
    const cur = LEARNERS.find((l) => l.id === S.learner);
    learnInfo.replaceChildren(code(cur.formula), text(cur.blurb, 'small'));
  }
  function refreshLearn() {
    const ag = S.agent;
    stEp.set(String(ag.episode), ag.outcome ? `last ended: ${ag.outcome === 'limit' ? 'step limit' : ag.outcome}` : '');
    stRet.set(ag.episode ? fmt.int(ag.ret) : '–', `goal +${REWARD.goal}, pit ${REWARD.pit}, step ${REWARD.step}`);
    retChart.update(S.trials.map((t) => ({ name: LEARNERS.find((l) => l.id === t.algo).name, color: LCOLOR[t.algo], data: smooth(t.returns, 12) })));
    stepChart.update(S.trials.map((t) => ({ name: t.algo, color: LCOLOR[t.algo], data: smooth(t.steps, 12) })));
  }
  function loadPreset(id) {
    S.preset = id;
    grid = Grid.preset(id);
    presetBlurb.replaceChildren(text(PRESETS.find((p) => p.id === id).blurb, 'small').firstChild);
    rebuild();
  }

  const steps = [
    {
      title: 'Search with a map',
      text: 'The oldest kind of AI: you know the world exactly, so you can think before you move. A search algorithm grows a **frontier** of tiles it could visit next and expands one at a time until it reaches the goal. Breadth-first search expands them in the order found, ring after ring.',
      enter: () => (setMode('plan'), algoSeg.set('bfs'), (S.algo = 'bfs'), (S.k = 0), renderPlan()),
    },
    {
      title: 'A good guess focuses the search',
      text: '**A-star** (written A*) ranks every frontier tile by the cost paid so far plus an optimistic estimate of what is left (here the straight-line distance in tiles). It still finds the cheapest path, but stops wasting effort behind its back. Compare the “expanded” counts of the four algorithms, then try Greedy: fast, and fooled by the mud.',
      enter: () => (setMode('plan'), algoSeg.set('astar'), (S.algo = 'astar'), (S.k = 0), renderPlan()),
    },
    {
      title: 'No map: only reward',
      text: 'Now take the map away. The agent knows nothing: not where the walls are, not where the goal is. It can only act and see what happens: −1 per step, −5 in mud, −100 for a pit, +100 for the goal. This is **reinforcement learning**. Early episodes are a random stumble.',
      enter: () => (setMode('learn'), learnSeg.set('q'), (S.learner = 'q'), relearn()),
    },
    {
      title: 'The Q-table',
      text: 'The agent keeps a number for every action on every tile: its estimate of the total reward that follows. After each step it nudges that number toward what it just experienced. The four arrows on a tile are those numbers (the white one is currently best), and the tile color is the best of them. Watch value **flow backward from the goal**.',
      body: () => code(LEARNERS[0].formula),
      enter: () => (setMode('learn'), learnSeg.set('q'), (S.learner = 'q'), renderLearn()),
    },
    {
      title: 'Explore or exploit',
      text: 'With probability **ε** the agent ignores what it knows and tries something random. Without exploration it never finds better routes; with too much it keeps falling into pits it already knows. On the **Cliff walk** world Q-learning learns the shortest path along the edge while SARSA, which accounts for its own random slips, learns the safe detour.',
      enter: () => (presetSel.set('cliff'), loadPreset('cliff'), setMode('learn'), learnSeg.set('sarsa'), (S.learner = 'sarsa'), relearn()),
    },
    {
      title: 'Learn a model, then plan',
      text: '**Dyna-Q** closes the circle. While acting it memorizes what each action did, building its own map, and after every real step it replays twenty remembered steps in its head. In the chart it needs far fewer real episodes to reach the same return: planning and learning are not rivals.',
      body: () => callout('The same idea at scale: AlphaZero searches a game tree guided by a learned network; MuZero and Dreamer plan inside a learned model of the world; language-model agents plan in text before they act.', 'key'),
      enter: () => (presetSel.set('maze'), loadPreset('maze'), setMode('learn'), learnSeg.set('dyna'), (S.learner = 'dyna'), relearn()),
    },
    {
      title: 'Your world',
      text: 'Click any tile to change it: floor → wall → mud → pit. Both the planners and the learners start over on your map. Build a trap for the greedy search, or a world where the learner has to cross mud to reach the goal.',
      enter: () => {},
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'Next lab →', onFinish: () => ctx.router.go('labs/embed') });
  ctx.ui.left(header({ kicker: 'Lab · Plan vs Learn', title: 'One maze, two kinds of intelligence', lead: 'With a map you can **search**. Without one you have to **learn** from reward.' }), guide.body);
  ctx.ui.right(modeSeg, presetSel, presetBlurb, h('div.btnrow', playBtn, button('Restart', { onClick: () => ((S.k = 0), S.mode === 'learn' && relearn(), paint()) })), slider({ label: 'Speed', min: 0.5, max: 2.6, step: 0.01, value: Math.log10(S.speed), format: (v) => Math.round(Math.pow(10, v)) + ' steps/s', onInput: (v) => (S.speed = Math.pow(10, v)) }), planBox, learnBox, legend([{ color: TILE_COLOR[WALL], label: 'wall' }, { color: TILE_COLOR[MUD], label: 'mud (costly)' }, { color: '#a01c2c', label: 'pit' }]));
  ctx.ui.dock(guide.dock);
  ctx.pick.add(tiles, {
    cursor: 'cell',
    tip: (hit) => {
      const i = hit.instanceId;
      const base = `${TILE_NAMES[grid.cells[i]]} · click to change`;
      if (S.mode === 'learn' && S.agent) {
        const v = S.agent.value(i);
        return Number.isNaN(v) ? base : `${base}\nbest learned value ${v.toFixed(1)}`;
      }
      const res = S.results.find((r) => r.algo === S.algo);
      return res && res.expandedAt[i] >= 0 ? `${base}\nexpanded ${res.expandedAt[i] + 1}. of ${res.order.length}, cost so far ${res.g[i]}` : base;
    },
    click: (hit) => {
      const i = hit.instanceId;
      if (grid.set(i, (grid.cells[i] + 1) % 4)) rebuild();
    },
  });
  ctx.key(' ', () => playBtn.click());

  presetBlurb.replaceChildren(text(PRESETS[0].blurb, 'small').firstChild);
  planBox.hidden = false;
  learnBox.hidden = true;
  rebuild();

  let acc = 0;
  let uiTimer = 0;
  return {
    update(dt, t) {
      goal.rotation.y = t * 1.5;
      goal.position.y = 0.78 + Math.sin(t * 2) * 0.06;
      if (S.playing) acc += dt * S.speed;
      const n = Math.min(400, Math.floor(acc));
      acc -= n;
      let dirty = false;
      if (S.mode === 'plan') {
        const res = S.results.find((r) => r.algo === S.algo);
        if (res && n > 0 && S.k < res.order.length + res.path.length + 30) {
          S.k += n;
          dirty = true;
          const walk = S.k - res.order.length;
          const at = walk >= 0 && res.path.length ? res.path[Math.min(res.path.length - 1, walk)] : grid.start;
          agentTarget.set(tx(at), 0.42, tz(at));
        } else if (res && S.playing && S.k >= res.order.length + res.path.length + 30) S.k = 0; // loop the demonstration
      } else if (S.agent) {
        for (let i = 0; i < n; i++) S.agent.step();
        if (n > 0) {
          dirty = true;
          const s = S.agent.active ? S.agent.s : S.agent.last.s2;
          agentTarget.set(tx(s), 0.42 - (grid.cells[s] === PIT ? 0.35 : 0), tz(s));
        }
        // the three background trials fill in the learning curves a few episodes per frame
        const t0 = performance.now();
        while (S.trials[0].episodes < S.target && performance.now() - t0 < 5) for (const tr of S.trials) tr.completeEpisode();
      }
      if (dirty) paint();
      const kd = S.speed > 80 ? 0.02 : 0.07;
      agentMesh.position.x = damp(agentMesh.position.x, agentTarget.x, kd, dt);
      agentMesh.position.y = damp(agentMesh.position.y, agentTarget.y, kd, dt);
      agentMesh.position.z = damp(agentMesh.position.z, agentTarget.z, kd, dt);
      uiTimer += dt;
      if (S.mode === 'learn' && uiTimer > 0.25) {
        uiTimer = 0;
        refreshLearn();
      }
    },
  };
}
