// Grid world for the "Plan vs Learn" lab: one maze, two families of algorithms.
//   * planning with a known map: breadth-first search, Dijkstra, greedy best-first, A*
//   * learning from reward alone: tabular Q-learning, SARSA and Dyna-Q (all epsilon-greedy)
// Pure JS, no rendering. Everything the lab shows (nodes expanded, path cost, learning curves)
// is produced by the functions in this file.

export const FLOOR = 0;
export const WALL = 1;
export const MUD = 2;
export const PIT = 3;
export const TILE_NAMES = ['floor', 'wall', 'mud', 'pit'];

/** What the learner is paid. Planning uses the same numbers as costs: 1 per step, 5 for mud. */
export const REWARD = { step: -1, mud: -5, pit: -100, goal: 100 };

/** Actions: 0 = north (toward row 0), 1 = east, 2 = south, 3 = west. */
export const ACTIONS = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];
export const ACTION_NAMES = ['north', 'east', 'south', 'west'];
export const ACTION_ARROWS = ['↑', '→', '↓', '←'];

/** Deterministic pseudo-random generator (mulberry32), so runs can be repeated exactly. */
export function rng(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Maps are written as text, one string per row (row 0 is the far side of the board):
 *   .  floor   #  wall   ~  mud   x  pit   S  start   G  goal
 */
export const PRESETS = [
  {
    id: 'terrain',
    name: 'Mixed terrain',
    blurb: 'Walls, a mud field on the direct route and a few pits. The path with the fewest steps is not the cheapest one.',
    rows: [
      '.......#...G',
      '.###...#.~~.',
      '.x.#.....~~.',
      '...#...###~.',
      '##.###...#x.',
      '.....#...#..',
      '.~~~.#.....#',
      '.~~~...##...',
      '.~~~.#..#...',
      '.....#..#..x',
      '.###......#.',
      'S....x..#...',
    ],
  },
  {
    id: 'cliff',
    name: 'Cliff walk',
    blurb: 'The classic test from Sutton and Barto: the shortest path runs along the edge of a cliff.',
    rows: ['............', '............', '............', 'SxxxxxxxxxxG'],
  },
  {
    id: 'maze',
    name: 'Maze',
    blurb: 'Corridors and dead ends. News of a reward found at the far end has a long way to travel back.',
    rows: [
      '.#.#...#..G',
      '.#.#.#.#.#.',
      '.#...#...#.',
      '.#.#######.',
      '.....#.....',
      '####.#.####',
      '...#.#.....',
      '.#.#.###.#.',
      '.#.#.#...#.',
      '.#.#.#.#.#.',
      'S#.....#...',
    ],
  },
  {
    id: 'blank',
    name: 'Blank',
    blurb: 'An empty board. Click tiles to build your own world.',
    rows: [
      '............',
      '..........G.',
      '............',
      '............',
      '............',
      '............',
      '............',
      '............',
      '............',
      '............',
      '.S..........',
      '............',
    ],
  },
];

export class Grid {
  /** @param {string[]} rows */
  constructor(rows) {
    this.h = rows.length;
    this.w = rows[0].length;
    this.n = this.w * this.h;
    this.cells = new Uint8Array(this.n);
    this.start = 0;
    this.goal = this.n - 1;
    for (let y = 0; y < this.h; y++)
      for (let x = 0; x < this.w; x++) {
        const c = rows[y][x];
        const i = y * this.w + x;
        this.cells[i] = c === '#' ? WALL : c === '~' ? MUD : c === 'x' ? PIT : FLOOR;
        if (c === 'S') this.start = i;
        if (c === 'G') this.goal = i;
      }
    // transition tables for the learner: next state, reward and "episode over" per (state, action)
    this.next = new Int16Array(this.n * 4);
    this.reward = new Float32Array(this.n * 4);
    this.terminal = new Uint8Array(this.n * 4);
    this.compile();
  }

  static preset(id) {
    return new Grid((PRESETS.find((p) => p.id === id) ?? PRESETS[0]).rows);
  }

  x(i) {
    return i % this.w;
  }

  y(i) {
    return (i / this.w) | 0;
  }

  /** Change one tile (the start and goal tiles stay floor). Returns false if nothing changed. */
  set(i, type) {
    if (i === this.start || i === this.goal || this.cells[i] === type) return false;
    this.cells[i] = type;
    this.compile();
    return true;
  }

  /** Cost of stepping onto tile i when planning; Infinity for tiles a planner never enters. */
  cost(i) {
    const c = this.cells[i];
    return c === FLOOR ? 1 : c === MUD ? 5 : Infinity;
  }

  /** Neighbor of tile i in direction a, or -1 when that would leave the board. */
  neighbor(i, a) {
    const x = (i % this.w) + ACTIONS[a][0];
    const y = ((i / this.w) | 0) + ACTIONS[a][1];
    return x < 0 || y < 0 || x >= this.w || y >= this.h ? -1 : y * this.w + x;
  }

  /** Rebuild the transition tables after an edit. */
  compile() {
    for (let s = 0; s < this.n; s++)
      for (let a = 0; a < 4; a++) {
        const k = s * 4 + a;
        const t = this.neighbor(s, a);
        if (t < 0 || this.cells[t] === WALL) {
          // bumping into a wall or the edge: stay put, pay for the step
          this.next[k] = s;
          this.reward[k] = REWARD.step;
          this.terminal[k] = 0;
        } else if (this.cells[t] === PIT) {
          this.next[k] = t;
          this.reward[k] = REWARD.pit;
          this.terminal[k] = 1;
        } else {
          this.next[k] = t;
          this.reward[k] = (this.cells[t] === MUD ? REWARD.mud : REWARD.step) + (t === this.goal ? REWARD.goal : 0);
          this.terminal[k] = t === this.goal ? 1 : 0;
        }
      }
  }

  manhattan(i, j = this.goal) {
    return Math.abs((i % this.w) - (j % this.w)) + Math.abs(((i / this.w) | 0) - ((j / this.w) | 0));
  }
}

// ------------------------------------------------------------------------------------ planning

/** f = priority of a frontier node. g = cost so far, h = Manhattan estimate to the goal. */
export const SEARCHERS = [
  {
    id: 'bfs',
    name: 'Breadth-first',
    short: 'BFS',
    rule: 'oldest frontier tile first',
    blurb: 'Expands tiles in the order they were discovered: ring after ring. Finds the path with the fewest steps and is blind to what a step costs.',
  },
  {
    id: 'dijkstra',
    name: 'Dijkstra',
    short: 'Dijkstra',
    rule: 'lowest cost so far (g)',
    blurb: 'Always expands the cheapest tile reached so far. Guaranteed to find the cheapest path, and searches in every direction to be sure.',
  },
  {
    id: 'greedy',
    name: 'Greedy best-first',
    short: 'Greedy',
    rule: 'lowest estimate to the goal (h)',
    blurb: 'Always expands the tile that looks closest to the goal. Very fast when the guess is right, and easily led into detours and mud.',
  },
  {
    id: 'astar',
    name: 'A*',
    short: 'A*',
    rule: 'lowest g + h',
    blurb: 'Cost so far plus the estimate of what is left. Still guaranteed to find the cheapest path, while ignoring tiles that cannot be on it.',
  },
];

/** Minimal binary heap of [key, tie, seq, node, g] entries stored in parallel arrays. */
class Heap {
  constructor() {
    this.key = [];
    this.tie = [];
    this.seq = [];
    this.node = [];
    this.g = [];
  }

  get size() {
    return this.key.length;
  }

  less(i, j) {
    if (this.key[i] !== this.key[j]) return this.key[i] < this.key[j];
    if (this.tie[i] !== this.tie[j]) return this.tie[i] < this.tie[j];
    return this.seq[i] < this.seq[j];
  }

  swap(i, j) {
    for (const arr of [this.key, this.tie, this.seq, this.node, this.g]) {
      const t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
  }

  push(key, tie, seq, node, g) {
    let i = this.key.length;
    this.key.push(key);
    this.tie.push(tie);
    this.seq.push(seq);
    this.node.push(node);
    this.g.push(g);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      this.swap(i, p);
      i = p;
    }
  }

  /** Remove the smallest entry; returns [node, g]. */
  pop() {
    const out = [this.node[0], this.g[0]];
    const last = this.key.length - 1;
    this.swap(0, last);
    for (const arr of [this.key, this.tie, this.seq, this.node, this.g]) arr.pop();
    let i = 0;
    for (;;) {
      const l = i * 2 + 1;
      const r = l + 1;
      let m = i;
      if (l < last && this.less(l, m)) m = l;
      if (r < last && this.less(r, m)) m = r;
      if (m === i) break;
      this.swap(i, m);
      i = m;
    }
    return out;
  }
}

/**
 * Run one search from grid.start to grid.goal.
 * A tile is "expanded" when it is taken off the frontier and its neighbors are looked at; the
 * search stops when the goal itself is taken off the frontier.
 * @param {Grid} grid
 * @param {'bfs'|'dijkstra'|'greedy'|'astar'} algo
 * @returns {{
 *   algo: string, found: boolean, path: number[], cost: number, steps: number, expanded: number,
 *   order: number[], expandedAt: Int16Array, discoveredAt: Int16Array, g: Float64Array, parent: Int16Array
 * }} order = tiles in the order they were expanded; expandedAt[i] = position of tile i in that
 *    order (-1 = never); discoveredAt[i] = number of expansions finished when tile i first joined
 *    the frontier (-1 = never). path runs from start to goal (empty when there is none).
 */
export function search(grid, algo) {
  const n = grid.n;
  const g = new Float64Array(n).fill(Infinity);
  const parent = new Int16Array(n).fill(-1);
  const expandedAt = new Int16Array(n).fill(-1);
  const discoveredAt = new Int16Array(n).fill(-1);
  const order = [];
  const heap = new Heap();
  const usesCost = algo === 'dijkstra' || algo === 'astar';
  let seq = 0;
  const push = (node, gv) => {
    const h = grid.manhattan(node);
    if (algo === 'bfs') heap.push(seq, 0, seq, node, gv);
    else if (algo === 'dijkstra') heap.push(gv, 0, seq, node, gv);
    else if (algo === 'greedy') heap.push(h, 0, seq, node, gv);
    else heap.push(gv + h, h, seq, node, gv); // A*: among equal f prefer the tile nearer the goal
    seq++;
  };
  g[grid.start] = 0;
  discoveredAt[grid.start] = 0;
  push(grid.start, 0);
  let found = false;
  while (heap.size) {
    const [node, gv] = heap.pop();
    if (expandedAt[node] >= 0) continue; // an outdated copy of a tile that was already expanded
    if (usesCost && gv > g[node]) continue;
    expandedAt[node] = order.length;
    order.push(node);
    if (node === grid.goal) {
      found = true;
      break;
    }
    for (let a = 0; a < 4; a++) {
      const t = grid.neighbor(node, a);
      if (t < 0) continue;
      const c = grid.cost(t);
      if (c === Infinity || expandedAt[t] >= 0) continue;
      const ng = g[node] + c;
      if (usesCost) {
        if (ng >= g[t]) continue;
      } else if (discoveredAt[t] >= 0) continue; // BFS and greedy keep the first way they found
      g[t] = ng;
      parent[t] = node;
      if (discoveredAt[t] < 0) discoveredAt[t] = order.length;
      push(t, ng);
    }
  }
  const path = [];
  if (found) for (let i = grid.goal; i >= 0; i = parent[i]) path.unshift(i);
  return {
    algo,
    found,
    path,
    cost: found ? g[grid.goal] : Infinity,
    steps: found ? path.length - 1 : 0,
    expanded: order.length,
    order,
    expandedAt,
    discoveredAt,
    g,
    parent,
  };
}

/** Run all four searches; `optimal` compares each path cost with Dijkstra's (the true minimum). */
export function compareSearch(grid) {
  const results = SEARCHERS.map((s) => search(grid, s.id));
  const best = results.find((r) => r.algo === 'dijkstra').cost;
  for (const r of results) r.optimal = r.found && r.cost === best;
  return results;
}

/** Cost of walking a list of tiles (the first one is where you stand). */
export function pathCost(grid, path) {
  let c = 0;
  for (let i = 1; i < path.length; i++) c += grid.cost(path[i]);
  return c;
}

// ------------------------------------------------------------------------------------ learning

export const LEARNERS = [
  {
    id: 'q',
    name: 'Q-learning',
    formula: 'Q(s,a) ← Q(s,a) + α·[ r + γ·max Q(s′,·) − Q(s,a) ]',
    blurb: 'Learns the value of always acting best from the next tile on, whatever it actually does next (off-policy).',
  },
  {
    id: 'sarsa',
    name: 'SARSA',
    formula: 'Q(s,a) ← Q(s,a) + α·[ r + γ·Q(s′,a′) − Q(s,a) ]\na′ = the action it really takes next',
    blurb: 'Learns the value of the policy it is actually following, random exploratory moves included (on-policy).',
  },
  {
    id: 'dyna',
    name: 'Dyna-Q',
    formula: 'model(s,a) ← (r, s′)\nthen n times: pick a remembered (s,a),\n  read (r, s′) from the model, apply the Q-learning update',
    blurb: 'Q-learning plus a learned model of the world. After every real step it replays imagined steps from memory.',
  },
];

/**
 * q0 is the value every action starts with. The largest return there is (the goal reward) makes
 * the learner optimiztic: anything it has not tried yet looks better than what it has tried, so
 * it tries everything at least once.
 */
export const DEFAULT_PARAMS = { epsilon: 0.1, alpha: 0.5, gamma: 1, planning: 20, maxSteps: 400, q0: REWARD.goal };

/**
 * One tabular learner in one grid world. `params` is read live at every step, so sliders can
 * share one object between all learners.
 */
export class Agent {
  /**
   * @param {Grid} grid
   * @param {'q'|'sarsa'|'dyna'} algo
   * @param {{epsilon:number, alpha:number, gamma:number, planning:number, maxSteps:number}} params
   * @param {number} seed
   */
  constructor(grid, algo, params, seed = 1) {
    this.grid = grid;
    this.algo = algo;
    this.params = params;
    this.rand = rng(seed);
    const n = grid.n;
    this.q = new Float64Array(n * 4).fill(params.q0 ?? 0); // action values
    this.tried = new Uint32Array(n * 4); // how often each (tile, action) was really taken
    // Dyna-Q's model of the world: what happened the last time (s, a) was tried
    this.modelNext = new Int16Array(n * 4).fill(-1);
    this.modelReward = new Float32Array(n * 4);
    this.modelDone = new Uint8Array(n * 4);
    this.seen = new Int32Array(n * 4); // list of (s, a) pairs with a model entry
    this.seenCount = 0;
    this.planned = new Int32Array(64); // (s, a) pairs replayed after the last real step
    this.plannedCount = 0;
    this.s = grid.start;
    this.a = -1; // SARSA: the action already chosen for the current tile
    this.active = false; // an episode is in progress
    this.episode = 0; // finished episodes
    this.steps = 0; // steps in the current (or last finished) episode
    this.ret = 0; // return of the current (or last finished) episode
    this.totalSteps = 0;
    this.outcome = ''; // how the last episode ended: 'goal' | 'pit' | 'limit'
    this.last = { s: grid.start, a: 0, r: 0, s2: grid.start, explore: false, done: false };
  }

  /** Greedy action with random tie-breaking. */
  best(s) {
    const q = this.q;
    const k = s * 4;
    let m = q[k];
    let count = 1;
    let pick = 0;
    for (let a = 1; a < 4; a++) {
      const v = q[k + a];
      if (v > m) {
        m = v;
        count = 1;
        pick = a;
      } else if (v === m) {
        // reservoir sampling keeps every tied action equally likely
        count++;
        if (this.rand() * count < 1) pick = a;
      }
    }
    return pick;
  }

  maxQ(s) {
    const q = this.q;
    const k = s * 4;
    return Math.max(q[k], q[k + 1], q[k + 2], q[k + 3]);
  }

  /** Epsilon-greedy choice. Sets this.explored when the move was a random one. */
  choose(s) {
    if (this.rand() < this.params.epsilon) {
      this.explored = true;
      return (this.rand() * 4) | 0;
    }
    this.explored = false;
    return this.best(s);
  }

  startEpisode() {
    this.s = this.grid.start;
    this.steps = 0;
    this.ret = 0;
    this.active = true;
    this.a = this.algo === 'sarsa' ? this.choose(this.s) : -1;
    this.sarsaExplored = this.explored;
  }

  /** One real step in the world. Returns true when the episode has just ended. */
  step() {
    if (!this.active) this.startEpisode();
    const { grid, q, params } = this;
    const s = this.s;
    let a;
    let explore;
    if (this.algo === 'sarsa') {
      a = this.a;
      explore = this.sarsaExplored;
    } else {
      a = this.choose(s);
      explore = this.explored;
    }
    const k = s * 4 + a;
    const r = grid.reward[k];
    const s2 = grid.next[k];
    const done = grid.terminal[k] === 1;
    this.tried[k]++;

    if (this.algo === 'sarsa') {
      let target = r;
      if (!done) {
        const a2 = this.choose(s2);
        this.a = a2;
        this.sarsaExplored = this.explored;
        target += params.gamma * q[s2 * 4 + a2];
      }
      q[k] += params.alpha * (target - q[k]);
    } else {
      q[k] += params.alpha * (r + (done ? 0 : params.gamma * this.maxQ(s2)) - q[k]);
      if (this.algo === 'dyna') {
        if (this.modelNext[k] < 0) this.seen[this.seenCount++] = k;
        this.modelNext[k] = s2;
        this.modelReward[k] = r;
        this.modelDone[k] = done ? 1 : 0;
        const n = Math.min(params.planning | 0, this.planned.length);
        for (let i = 0; i < n; i++) {
          const j = this.seen[(this.rand() * this.seenCount) | 0];
          const tr = this.modelReward[j];
          const ts = this.modelNext[j];
          q[j] += params.alpha * (tr + (this.modelDone[j] ? 0 : params.gamma * this.maxQ(ts)) - q[j]);
          this.planned[i] = j;
        }
        this.plannedCount = n;
      }
    }

    this.s = s2;
    this.steps++;
    this.totalSteps++;
    this.ret += r;
    const last = this.last;
    last.s = s;
    last.a = a;
    last.r = r;
    last.s2 = s2;
    last.explore = explore;
    last.done = done;
    if (done || this.steps >= params.maxSteps) {
      this.active = false;
      this.episode++;
      this.outcome = !done ? 'limit' : s2 === grid.goal ? 'goal' : 'pit';
      return true;
    }
    return false;
  }

  /** Run until the current episode is over (starts one if none is in progress). */
  finishEpisode() {
    while (!this.step());
  }

  /** Best value learned for tile s, looking only at actions that were actually tried. NaN if none. */
  value(s) {
    let m = NaN;
    for (let a = 0; a < 4; a++) if (this.tried[s * 4 + a] > 0 && !(m >= this.q[s * 4 + a])) m = this.q[s * 4 + a];
    return m;
  }

  /**
   * Follow the greedy policy from the start tile (ties resolved by the lowest action index, no
   * exploration). `end` says how the walk finished: 'goal', 'pit', 'loop' (the policy ran in a
   * circle or into a wall) or 'unknown' (it reached a tile it has never acted from).
   * @returns {{path: number[], end: string, cost: number, ret: number}}
   */
  greedyPath() {
    const { grid, q } = this;
    const path = [grid.start];
    const seen = new Uint8Array(grid.n);
    let s = grid.start;
    let ret = 0;
    let end = 'loop';
    for (let i = 0; i < grid.n; i++) {
      seen[s] = 1;
      let a = -1;
      for (let b = 0; b < 4; b++) if (this.tried[s * 4 + b] > 0 && (a < 0 || q[s * 4 + b] > q[s * 4 + a])) a = b;
      if (a < 0) {
        end = 'unknown';
        break;
      }
      const k = s * 4 + a;
      const s2 = grid.next[k];
      ret += grid.reward[k];
      if (s2 !== s) path.push(s2);
      if (grid.terminal[k]) {
        end = s2 === grid.goal ? 'goal' : 'pit';
        break;
      }
      if (seen[s2]) break;
      s = s2;
    }
    return { path, end, cost: end === 'goal' ? pathCost(grid, path) : Infinity, ret };
  }
}

/**
 * Several independent runs of one algorithm on the same grid, kept in step episode by episode.
 * The curves the lab plots are the mean over these runs.
 */
export class Trial {
  constructor(grid, algo, params, { runs = 8, seed = 1 } = {}) {
    this.grid = grid;
    this.algo = algo;
    this.agents = Array.from({ length: runs }, (_, i) => new Agent(grid, algo, params, seed * 7919 + i * 104729 + 1));
    this.episodes = 0;
    this.returns = []; // mean return per episode
    this.steps = []; // mean steps per episode
    this.success = []; // share of runs that reached the goal
    this.realSteps = 0; // real environment steps, mean per run
  }

  /** Finish the current episode in every run and record the means. */
  completeEpisode() {
    let ret = 0;
    let steps = 0;
    let ok = 0;
    for (const ag of this.agents) {
      if (ag.episode <= this.episodes) ag.finishEpisode();
      ret += ag.ret;
      steps += ag.steps;
      if (ag.outcome === 'goal') ok++;
    }
    const r = this.agents.length;
    this.returns.push(ret / r);
    this.steps.push(steps / r);
    this.success.push(ok / r);
    this.realSteps += steps / r;
    this.episodes++;
  }
}

/** Trailing moving average (window w) of a series, as [episode, value] pairs for plotting. */
export function smooth(series, w = 10) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < series.length; i++) {
    sum += series[i];
    if (i >= w) sum -= series[i - w];
    out.push([i + 1, sum / Math.min(i + 1, w)]);
  }
  return out;
}

/**
 * Exact optimal values by value iteration (needs the map, so this is planning, not learning).
 * Used as the reference that the learners are compared with.
 * @returns {{v: Float64Array, q: Float64Array, sweeps: number}}
 */
export function valueIteration(grid, gamma = 1, tol = 1e-9, maxSweeps = 2000) {
  const n = grid.n;
  const v = new Float64Array(n);
  const q = new Float64Array(n * 4);
  let sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    let delta = 0;
    for (let s = 0; s < n; s++) {
      if (grid.cells[s] === WALL || grid.cells[s] === PIT || s === grid.goal) continue;
      let m = -Infinity;
      for (let a = 0; a < 4; a++) {
        const k = s * 4 + a;
        const val = grid.reward[k] + (grid.terminal[k] ? 0 : gamma * v[grid.next[k]]);
        q[k] = val;
        if (val > m) m = val;
      }
      // a tile that cannot reach the goal keeps sinking; clamp so the loop still ends
      if (m < -1e6) m = -1e6;
      delta = Math.max(delta, Math.abs(m - v[s]));
      v[s] = m;
    }
    if (delta < tol) break;
  }
  return { v, q, sweeps };
}
