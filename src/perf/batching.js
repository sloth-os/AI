// Batching: one GPU, many users. Four copies of the same simulated GPU serve the same arrival
// trace, each under a different scheduler. The simulation lives in ../ml/servingsim.js; this file
// draws its state and wires the controls.
import * as THREE from 'three';
import './batching.css';
import { SCHEDULERS, DEFAULTS, SLOT, Trace, GpuSim, windowStats, runFull, stepTime } from '../ml/servingsim.js';
import { SERIES, glow, mix } from '../core/colors.js';
import { gridFloor, rbox, edges, stdMat, glassMat, instanced, setInst, flushInst, points, segments } from '../core/geom.js';
import { textSprite, textPlane, htmlLabel } from '../core/labels.js';
import { h, header, section, callout, button, slider, tour, code, table } from '../core/ui.js';
import { lineChart } from '../core/charts.js';
import { damp } from '../core/tween.js';

const NS = SCHEDULERS.length;
const COLOR = SCHEDULERS.map((_, i) => SERIES[i]);
const PITCH = 3.6; // distance between stations
const SW = 3.0; // station width
const SLAB_Z0 = -1.0; // back edge of the GPU slab; it reaches forward to SLAB_Z0 + SLAB_D
const SLAB_D = 4.0;
const SLAB_H = 0.3;
const BAR_L = 2.6; // world length of a bar that still has `outMax` tokens to generate
const BAR_X0 = -BAR_L / 2;
const LANE_Z0 = SLAB_Z0 + 0.2;
const LANES_D = SLAB_D - 0.4;
const LANE_MAX = 0.36; // deepest a lane is drawn
const CONV_Z0 = SLAB_Z0 - 0.45; // the conveyor runs from here back to CONV_Z0 - CONV_L
const CONV_L = 3.3;
const QMAX = 44; // waiting requests drawn per station (the label carries the full count)
const MAX_LANES = 32;
const PMAX = 560; // token particles
const WINDOW = 20000; // live metrics look at the last 20 simulated seconds
const HORIZON = 1800000; // the live trace is 30 simulated minutes long, then starts over
const FULL_HORIZON = 120000; // the results table serves a fixed 2-minute trace to the end

const fmtT = (ms) => (!Number.isFinite(ms) ? '–' : ms < 1000 ? Math.round(ms) + ' ms' : ms < 60000 ? (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + ' s' : (ms / 60000).toFixed(1) + ' min');
const fmtN = (v) => (!Number.isFinite(v) ? '–' : v >= 100 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1));
const fmtPct = (v) => (!Number.isFinite(v) ? '–' : Math.round(v * 100) + '%');
const swatch = (color, label) => h('span.bt-sw', h('i', { style: { '--c': color } }), label);

export async function enter(ctx) {
  const { app, root } = ctx;
  app.setView({ pos: [0, 12.6, 9.4], target: [0, 0.1, -0.95], fit: 7.6, min: 4, max: 70, maxPolar: Math.PI * 0.49 });
  app.setBloom({ strength: 0.5, radius: 0.6 });
  root.add(gridFloor({ size: 70, y: -0.02, color: '#31508c', opacity: 0.3 }));

  // ---------------------------------------------------------------- simulation state
  const p = { ...DEFAULTS };
  let trace = null;
  let sims = [];
  let T = 0; // simulated time shown, ms
  let speed = 2;
  let playing = true;

  // ---------------------------------------------------------------- static scene
  const stations = SCHEDULERS.map((s, i) => {
    const x = (i - (NS - 1) / 2) * PITCH;
    const slab = new THREE.Mesh(rbox(SW, SLAB_H, SLAB_D, 0.06), stdMat('#0b1120', { rough: 0.38, metal: 0.55 }));
    slab.position.set(x, SLAB_H / 2, SLAB_Z0 + SLAB_D / 2);
    const rim = edges(new THREE.BoxGeometry(SW + 0.02, SLAB_H + 0.02, SLAB_D + 0.02), COLOR[i], { k: 1.3 });
    rim.position.copy(slab.position);
    const belt = new THREE.Mesh(new THREE.BoxGeometry(SW, 0.05, CONV_L + 0.25), glassMat('#101a30', 0.55));
    belt.position.set(x, 0.035, CONV_Z0 - CONV_L / 2 + 0.05);
    const zb = CONV_Z0 - CONV_L - 0.07;
    const beltEdge = segments(
      [[x - SW / 2, 0.07, CONV_Z0 + 0.17], [x - SW / 2, 0.07, zb], [x + SW / 2, 0.07, CONV_Z0 + 0.17], [x + SW / 2, 0.07, zb], [x - SW / 2, 0.07, zb], [x + SW / 2, 0.07, zb]],
      '#8fa3c8',
      { k: 0.7, opacity: 0.45 },
    );
    // ruler: quarter marks of the longest possible answer
    const ticks = [];
    for (let q = 1; q <= 4; q++) {
      const tx = x + BAR_X0 + (BAR_L * q) / 4;
      ticks.push([tx, SLAB_H + 0.012, LANE_Z0 - 0.06], [tx, SLAB_H + 0.012, LANE_Z0 + LANES_D + 0.06]);
    }
    const ruler = segments(ticks, '#8fa3c8', { k: 0.6, opacity: 0.22 });
    const name = h('b', s.short);
    const line1 = h('small', '');
    const line2 = h('small', '');
    const tag = htmlLabel(h('div', { style: { '--c': COLOR[i] }, title: s.name }, name, line1, line2), 'bt-tag');
    tag.position.set(x, 0.75, CONV_Z0 - CONV_L - 0.75);
    tag.center.set(0.5, 1);
    root.add(slab, rim, belt, beltEdge, ruler, tag);
    return {
      x,
      rim,
      line1,
      line2,
      rimBase: glow(COLOR[i], 1),
      cBright: glow(COLOR[i], 1.75),
      cDim: glow(COLOR[i], 0.5),
      cFlash: glow(mix(COLOR[i], '#ffffff', 0.55), 1.9),
      cPrefill: glow(mix(COLOR[i], '#ffffff', 0.8), 1.7),
      cToken: glow(mix(COLOR[i], '#ffffff', 0.25), 2.6),
      lastHead: 0,
      shift: 0,
      glowK: 0,
      budget: 0,
      rr: 0,
    };
  });

  // one inlet for all four conveyors: the same request reaches every GPU at the same instant
  const zIn = CONV_Z0 - CONV_L - 0.07;
  const xEdge = ((NS - 1) / 2) * PITCH + SW / 2;
  const inlet = segments([[-xEdge, 0.07, zIn - 0.28], [xEdge, 0.07, zIn - 0.28], ...stations.flatMap((st) => [[st.x, 0.07, zIn - 0.28], [st.x, 0.07, zIn]])], '#dfe7f7', { k: 1.1, opacity: 0.8 });
  root.add(inlet);
  const inletBase = glow('#dfe7f7', 1);
  let inletFlash = 0;
  const cap = (str, x, z, o = {}) => {
    const s = textSprite(str, { size: 0.27, color: '#93a1bf', weight: 550, ...o });
    s.position.set(x, 0.12, z);
    return s;
  };
  // row captions lie on the floor beside the first station and read from front to back
  const flat = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3(0, 0, -1), new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0)));
  const side = (str, z) => {
    const m = textPlane(str, { size: 0.3, color: '#a9b6d3', weight: 550 });
    m.quaternion.copy(flat);
    m.position.set(-xEdge - 0.32, 0.02, z);
    return m;
  };
  root.add(
    cap('the same requests reach all four GPUs at the same instant', 0, zIn - 0.8),
    cap('bar length = tokens still to generate', 0, SLAB_Z0 + SLAB_D + 0.55, { size: 0.25 }),
    side('waiting', CONV_Z0 - CONV_L / 2),
    side('batch slots', SLAB_Z0 + SLAB_D / 2),
  );

  // lanes (lit, dark) and bars (unlit, coloured; they glow when they are doing work)
  const unitBox = new THREE.BoxGeometry(1, 1, 1);
  unitBox.translate(0.5, 0.5, 0); // grows to +x and +y from its anchor
  const tracks = instanced(unitBox, stdMat('#ffffff', { rough: 0.6, metal: 0.2 }), NS * MAX_LANES);
  const BAR_CAP = NS * (MAX_LANES + QMAX);
  const bars = instanced(unitBox, new THREE.MeshBasicMaterial({ color: 0xffffff }), BAR_CAP);
  bars.count = 0;
  bars.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, -1), 14); // instances move: keep picking valid
  root.add(tracks, bars);
  const instKind = new Uint8Array(BAR_CAP); // SLOT state, or 9 for a waiting request
  const instSt = new Uint8Array(BAR_CAP);
  const instReq = new Int32Array(BAR_CAP);
  const cTrack = new THREE.Color('#1a2742');
  const cHeld = new THREE.Color('#5d6883');
  const tmpC = new THREE.Color();
  let lanesDrawn = 0;

  function layoutLanes(lanes) {
    lanesDrawn = lanes;
    const lp = LANES_D / lanes;
    const ld = Math.min(LANE_MAX, lp * 0.74);
    let n = 0;
    for (const st of stations) for (let j = 0; j < lanes; j++) setInst(tracks, n++, st.x + BAR_X0, SLAB_H, LANE_Z0 + (j + 0.5) * lp, BAR_L, 0.014, ld, cTrack);
    tracks.count = n;
    flushInst(tracks);
  }

  // token particles
  const ppos = new Float32Array(PMAX * 3).fill(-999);
  const pcol = new Float32Array(PMAX * 3);
  const psize = new Float32Array(PMAX).fill(0.11);
  const pvel = new Float32Array(PMAX * 3);
  const plife = new Float32Array(PMAX);
  const pown = new Uint8Array(PMAX);
  let pcur = 0;
  const sparks = points(ppos, pcol, psize);
  root.add(sparks);
  const sparkRate = ctx.reduceMotion ? 0 : ctx.quality === 'low' ? 70 : 130; // particles per second per GPU

  function spawn(i, x, y, z) {
    const k = pcur;
    pcur = (pcur + 1) % PMAX;
    ppos[k * 3] = x;
    ppos[k * 3 + 1] = y;
    ppos[k * 3 + 2] = z;
    pvel[k * 3] = 0.25 + Math.random() * 0.45;
    pvel[k * 3 + 1] = 1.0 + Math.random() * 0.8;
    pvel[k * 3 + 2] = (Math.random() - 0.5) * 0.25;
    plife[k] = 1;
    pown[k] = i;
  }

  function stepParticles(dt) {
    for (let k = 0; k < PMAX; k++) {
      if (plife[k] <= 0) continue;
      const l = (plife[k] -= dt / 0.8);
      if (l <= 0) {
        ppos[k * 3 + 1] = -999;
        pcol[k * 3] = pcol[k * 3 + 1] = pcol[k * 3 + 2] = 0;
        continue;
      }
      ppos[k * 3] += pvel[k * 3] * dt;
      ppos[k * 3 + 1] += pvel[k * 3 + 1] * dt;
      ppos[k * 3 + 2] += pvel[k * 3 + 2] * dt;
      const c = stations[pown[k]].cToken;
      const a = l * l;
      pcol[k * 3] = c.r * a;
      pcol[k * 3 + 1] = c.g * a;
      pcol[k * 3 + 2] = c.b * a;
    }
    sparks.geometry.attributes.position.needsUpdate = true;
    sparks.geometry.attributes.color.needsUpdate = true;
  }

  // ---------------------------------------------------------------- drawing the simulation
  function draw(dt) {
    let lanes = Math.max(1, Math.min(MAX_LANES, p.maxBatch | 0));
    for (const s of sims) if (s.topSlot > lanes) lanes = Math.min(MAX_LANES, s.topSlot);
    if (lanes !== lanesDrawn) layoutLanes(lanes);
    const lp = LANES_D / lanes;
    const ld = Math.min(LANE_MAX, lp * 0.74);
    const outMax = p.outMax;
    const realMs = 1000 * Math.max(speed, 0.01);
    let n = 0;
    let fresh = 0;
    for (let i = 0; i < NS; i++) {
      const s = sims[i];
      const st = stations[i];
      const x0 = st.x + BAR_X0;
      const f = s.progress(T);
      st.budget = Math.min(8, st.budget + sparkRate * dt);
      st.rr = (st.rr + 1) % lanes;
      let working = 0;
      for (let jj = 0; jj < lanes; jj++) {
        const j = (jj + st.rr) % lanes; // rotate so the particle budget is shared fairly
        const state = s.slotState[j];
        const z = LANE_Z0 + (j + 0.5) * lp;
        let len = 0;
        if (state === SLOT.DECODE) {
          len = Math.max(0.02, ((s.slotRem[j] - f) / outMax) * BAR_L);
          setInst(bars, n, x0, SLAB_H + 0.014, z, len, 0.11, ld, st.cBright);
          working++;
        } else if (state === SLOT.PREFILL) {
          // the prompt is being processed: the bar is whole and lights up as the step completes
          len = Math.max(0.02, (s.slotRem[j] / outMax) * BAR_L);
          setInst(bars, n, x0, SLAB_H + 0.014, z, len, 0.11, ld, tmpC.copy(st.cBright).lerp(st.cPrefill, 0.35 + 0.65 * f));
          working++;
        } else if (state === SLOT.HELD) {
          setInst(bars, n, x0, SLAB_H + 0.014, z, 0.1, 0.035, ld, cHeld);
        }
        if (state !== SLOT.IDLE) {
          instKind[n] = state;
          instSt[n] = i;
          instReq[n] = s.slotReq[j];
          n++;
        }
        if (s.emitted[j] > 0) {
          if (st.budget >= 1) {
            st.budget -= 1;
            spawn(i, x0 + len, SLAB_H + 0.15, z);
          }
          s.emitted[j] = 0;
        }
      }
      st.glowK = damp(st.glowK, working / lanes, 0.18, dt);
      st.rim.material.color.copy(st.rimBase).multiplyScalar(0.55 + 1.5 * st.glowK);

      // the queue: first come, first served, nearest the GPU is next
      const moved = s.head - st.lastHead;
      st.lastHead = s.head;
      if (moved > 0) st.shift = Math.min(3, st.shift + moved);
      st.shift = damp(st.shift, 0, 0.1, dt);
      const q = s.queueAt(T);
      const shown = Math.min(q, QMAX);
      const qp = Math.max(CONV_L / QMAX, Math.min(0.21, CONV_L / Math.max(1, shown)));
      for (let k = 0; k < shown; k++) {
        const r = s.head + k;
        const z = CONV_Z0 - (k + 0.5 + st.shift) * qp;
        if (z < CONV_Z0 - CONV_L) break;
        const age = (T - trace.t[r]) / realMs; // seconds on screen since it arrived
        const flash = age < 0.4 ? 1 - age / 0.4 : 0;
        if (flash > fresh) fresh = flash;
        setInst(bars, n, x0, 0.07, z, Math.max(0.02, (trace.out[r] / outMax) * BAR_L), 0.07, qp * 0.62, flash > 0 ? tmpC.copy(st.cDim).lerp(st.cFlash, flash) : st.cDim);
        instKind[n] = 9;
        instSt[n] = i;
        instReq[n] = r;
        n++;
      }
    }
    bars.count = n;
    flushInst(bars);
    inletFlash = Math.max(fresh, damp(inletFlash, 0, 0.15, dt));
    inlet.material.color.copy(inletBase).multiplyScalar(0.55 + 1.6 * inletFlash);
  }

  ctx.pick.add(bars, {
    filter: (hit) => hit.instanceId < bars.count,
    tip: (hit) => {
      const k = hit.instanceId;
      const s = sims[instSt[k]];
      const r = instReq[k];
      if (r < 0 || r >= trace.t.length) return null;
      const out = trace.out[r];
      const head = `Request ${r + 1} · ${SCHEDULERS[instSt[k]].name}\n${trace.prompt[r]} prompt tokens, ${out} to generate\n`;
      if (instKind[k] === 9) return head + `Waiting for ${fmtT(T - trace.t[r])}`;
      const slot = s.slotReq.indexOf(r);
      if (instKind[k] === SLOT.HELD) return head + 'Finished. Its slot is held until the batch ends.';
      if (instKind[k] === SLOT.PREFILL) return head + 'Prompt is being processed';
      return head + `${slot >= 0 ? s.slotRem[slot] : 0} tokens left`;
    },
  });

  // ---------------------------------------------------------------- HUD: live read-outs
  const clock = h('span.bt-clock', '');
  const playBtn = button('❚❚ Pause', { kind: 'primary', onClick: () => setPlaying(!playing) });
  const groups = [
    { key: 'throughput', title: 'Throughput, tokens/s', format: fmtN },
    { key: 'util', title: 'Slots doing useful work', format: fmtPct, max: 1 },
    { key: 'ttft50', title: 'First token, median', format: fmtT },
    { key: 'ttft95', title: 'First token, 95th percentile', format: fmtT },
    { key: 'e2e50', title: 'Whole answer, median', format: fmtT },
    { key: 'e2e95', title: 'Whole answer, 95th percentile', format: fmtT },
  ].map((g) => {
    const rows = SCHEDULERS.map((s, i) => {
      const bar = h('i', { style: { '--c': COLOR[i] } });
      const val = h('span.v', '–');
      return { bar, val, el: h('div.r', { title: s.name }, h('span.t', bar), val) };
    });
    return { ...g, rows, el: h('div.bt-m', h('h4', g.title), rows.map((r) => r.el)) };
  });
  const shown = SCHEDULERS.map(() => true);
  const chart = lineChart({ height: 138, legend: false, xLabel: 'simulated seconds', yLabel: 'requests waiting', series: [] });
  const chips = h('div.chips');
  function renderChips() {
    chips.replaceChildren(
      ...SCHEDULERS.map((s, i) => {
        const c = h('span.chip.click', { style: { '--c': COLOR[i], opacity: shown[i] ? 1 : 0.4 }, title: 'Show or hide this line', role: 'button', 'aria-pressed': String(shown[i]) }, h('i.bt-line', { style: { '--c': COLOR[i] } }), s.short);
        c.addEventListener('click', () => {
          shown[i] = !shown[i];
          renderChips();
          refresh();
        });
        return c;
      }),
    );
  }
  const resultBox = h('div');
  const modelBox = h('div');

  function refresh() {
    clock.textContent = `${(T / 1000).toFixed(1)} s simulated`;
    const stats = sims.map((s) => windowStats(s, T, WINDOW));
    for (const g of groups) {
      const vals = stats.map((w) => w[g.key]);
      const max = g.max ?? Math.max(1e-9, ...vals.filter(Number.isFinite));
      g.rows.forEach((r, i) => {
        const v = vals[i];
        r.bar.style.width = Number.isFinite(v) ? Math.max(1.5, Math.min(100, (v / max) * 100)) + '%' : '0%';
        r.val.textContent = g.format(v);
      });
    }
    stations.forEach((st, i) => {
      const s = sims[i];
      const w = stats[i];
      const waitingToFill = !s.busy && w.queue > 0;
      st.line1.textContent = s.busy ? `${s.iterUseful} running${s.iterHeld ? ` + ${s.iterHeld} held` : ''}` : waitingToFill ? 'idle, filling a batch' : 'idle';
      st.line2.textContent = `${s.busy ? Math.round(w.stepMs) + ' ms/step · ' : ''}${w.queue} waiting`;
    });
    // queue length over the last two simulated minutes
    const b1 = Math.floor(T / 250);
    const b0 = Math.max(0, b1 - 480);
    chart.update(
      SCHEDULERS.map((s, i) => {
        if (!shown[i]) return null;
        const q = sims[i].qSeries;
        const data = [];
        for (let b = b0; b <= b1 && b < q.length; b += 2) data.push([b / 4, q[b]]);
        return { name: s.name, color: COLOR[i], data };
      }).filter(Boolean),
    );
  }

  // ---------------------------------------------------------------- HUD: the full run
  let fullTimer = 0;
  ctx.onDispose(() => clearTimeout(fullTimer));
  function scheduleFull() {
    clearTimeout(fullTimer);
    fullTimer = setTimeout(() => !ctx.dead && renderFull(), 140);
  }
  function renderFull() {
    const res = runFull(p, { horizon: FULL_HORIZON });
    const best = (key, dir) => {
      const vals = res.map((r) => r[key]);
      const b = dir > 0 ? Math.max(...vals) : Math.min(...vals);
      return vals.map((v) => Math.abs(v - b) <= Math.abs(b) * 0.005);
    };
    const row = (label, key, format, dir) => {
      const top = dir ? best(key, dir) : [];
      return [label, ...res.map((r, i) => (top[i] ? h('b', format(r[key])) : format(r[key])))];
    };
    resultBox.replaceChildren(
      table(
        ['', ...SCHEDULERS.map((s, i) => swatch(COLOR[i], s.short))],
        [
          row('Tokens/s', 'throughput', fmtN, 1),
          row('First token, median', 'ttft50', fmtT, -1),
          row('First token, p95', 'ttft95', fmtT, -1),
          row('Whole answer, median', 'e2e50', fmtT, -1),
          row('Whole answer, p95', 'e2e95', fmtT, -1),
          row('Slots used', 'util', fmtPct, 1),
          row('Slots wasted', 'wasted', fmtPct, 0),
          row('Longest queue', 'queueMax', (v) => String(v), -1),
        ],
        { compact: true },
      ),
      h('div.txt.small', h('p', `The same settings on a fixed trace: ${res[0].arrivals} requests arriving over ${FULL_HORIZON / 1000} simulated seconds (${fmtN(res[0].offered)} tokens/s asked for), served to the last token. Best value in bold.`)),
    );
    modelBox.replaceChildren(
      code(
        [
          `one step, B sequences   ${p.t0} + ${p.t1} × B ms`,
          `                        B = 1: ${stepTime(p, 1).toFixed(1)} ms   B = ${p.maxBatch}: ${stepTime(p, p.maxBatch).toFixed(1)} ms`,
          `tokens per step         1 per running sequence`,
          `prompt processing       ${p.tp} ms per prompt token,`,
          `                        inside the sequence's first step`,
          `memory                  at most ${p.maxBatch} sequences at once`,
          `arrivals                Poisson, ${p.rate.toFixed(2)} per second`,
          `output length           log-normal, mean ${p.outMean}, cap ${p.outMax}`,
          `prompt length           log-normal, mean ${p.promptMean}`,
          `static batching         starts short after ${p.timeout} ms`,
        ].join('\n'),
      ),
    );
  }

  // ---------------------------------------------------------------- controls
  function reset(warm = 0) {
    trace = new Trace(p, { horizon: HORIZON });
    sims = SCHEDULERS.map((s) => new GpuSim(s.id, trace, p));
    T = 0;
    if (warm > 0) {
      T = warm;
      trace.ensure(T);
      for (const s of sims) {
        s.advance(T);
        s.emitted.fill(0);
      }
    }
    stations.forEach((st, i) => {
      st.lastHead = sims[i].head;
      st.shift = 0;
      st.budget = 0;
    });
    plife.fill(0);
    ppos.fill(-999);
    pcol.fill(0);
    refresh();
  }

  function setPlaying(v) {
    playing = v;
    playBtn.textContent = playing ? '❚❚ Pause' : '▶ Run';
  }

  const set = (key, v) => {
    p[key] = v;
    if (key === 'rate' && trace) trace.retime(T);
    scheduleFull();
  };
  const S = {
    rate: slider({ label: 'Arrival rate', min: 0.1, max: 6, step: 0.05, value: p.rate, format: (v) => v.toFixed(2) + ' requests/s', onInput: (v) => set('rate', v) }),
    maxBatch: slider({ label: 'Maximum batch size (memory)', min: 1, max: MAX_LANES, step: 1, value: p.maxBatch, format: (v) => v + (v === 1 ? ' sequence' : ' sequences'), onInput: (v) => set('maxBatch', v) }),
    outSpread: slider({ label: 'Spread of output lengths', min: 0, max: 1.6, step: 0.05, value: p.outSpread, format: (v) => (v === 0 ? 'all equal' : '± ' + Math.round(v * 100) + '%'), onInput: (v) => set('outSpread', v) }),
    speed: slider({ label: 'Simulation speed', min: -0.6, max: 1.7, step: 0.01, value: Math.log10(speed), format: (v) => '× ' + +Math.pow(10, v).toPrecision(2), onInput: (v) => (speed = Math.pow(10, v)) }),
    t0: slider({ label: 'Fixed cost of a step, t0', min: 1, max: 60, step: 1, value: p.t0, format: (v) => v + ' ms', onInput: (v) => set('t0', v) }),
    t1: slider({ label: 'Cost per sequence, t1', min: 0, max: 20, step: 0.1, value: p.t1, format: (v) => v.toFixed(1) + ' ms', onInput: (v) => set('t1', v) }),
    outMean: slider({ label: 'Mean output length', min: 16, max: 256, step: 8, value: p.outMean, format: (v) => v + ' tokens', onInput: (v) => set('outMean', v) }),
    promptMean: slider({ label: 'Mean prompt length', min: 0, max: 3000, step: 50, value: p.promptMean, format: (v) => v + ' tokens', onInput: (v) => set('promptMean', Math.max(1, v)) }),
    timeout: slider({ label: 'Static batching gives up waiting after', min: 100, max: 5000, step: 100, value: p.timeout, format: (v) => (v / 1000).toFixed(1) + ' s', onInput: (v) => set('timeout', v) }),
  };

  /** Tour steps load a scenario: parameters not named go back to their defaults. */
  function scenario(over = {}, o = {}) {
    Object.assign(p, DEFAULTS, over);
    for (const key of Object.keys(S)) if (key !== 'speed') S[key].set(p[key]);
    speed = o.speed ?? 2;
    S.speed.set(Math.log10(speed));
    reset(o.warm ?? 12000);
    renderFull();
    setPlaying(true);
  }

  const paper = '[Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)';
  const steps = [
    {
      title: 'One token per step',
      text: 'A language model writes one token per forward pass. Every request here is a bar whose length is the number of tokens it still has to generate. It waits on the conveyor, takes a slot on the GPU, and shrinks as tokens come out.\n\nAt this light load, one request every five seconds, all four schedulers cope. **No batching** runs a single sequence and leaves the rest of the GPU dark.',
      enter: () => scenario({ rate: 0.2 }),
    },
    {
      title: 'Why batching works',
      text: 'A decoding step has a large fixed cost: the weights are read from memory once per step, however many sequences share it. In this model a step for B sequences takes **t0 + t1 × B** milliseconds. One sequence gets a token every 21.5 ms; eight sequences get eight tokens every 32 ms, **5.4 times the throughput**.\n\nRequests now arrive five times as fast and, for the moment, all ask for the same number of tokens. One at a time cannot keep up: its queue only grows. The three batching schedulers perform alike.',
      enter: () => scenario({ rate: 1.2, outSpread: 0 }),
    },
    {
      title: 'Static batching wastes slots',
      text: 'Real answers differ in length. **Static batching** waits for a full batch (or a timeout) and runs it until its **longest** sequence ends. Short answers finish early, but their slots stay reserved, and are still computed as padding, until the whole batch is done: the gray stubs.\n\nWith these uneven lengths about half of the slot time is wasted. The GPU never rests, yet the queue builds up.',
      enter: () => scenario({ rate: 1.2 }),
    },
    {
      title: 'Dynamic batching: do not wait',
      text: 'At low load static batching has a second problem: requests sit on the conveyor while the GPU is idle, waiting for a batch to fill. **Dynamic batching** starts with whatever is queued as soon as the GPU is free, so the median time to first token falls from about a second to well under a tenth of a second.\n\nA batch still runs to completion, though. The wasted slots remain, and a request that arrives just after a batch has started waits for all of it.',
      enter: () => scenario({ rate: 0.3 }),
    },
    {
      title: 'Continuous batching',
      text: '**Continuous batching** (iteration-level scheduling) decides again at every decoding step: finished sequences leave, waiting requests join. No slot is held by a finished answer, and a new request waits for one step, not for one batch.\n\nThis is the load of step 3. The run-to-completion schedulers fall behind; this one serves everything and returns the first token within a few steps.',
      body: () => callout('Orca introduced iteration-level scheduling ([Yu et al., OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)). vLLM and SGLang, the default open-source serving engines in 2026, both list continuous batching as a core feature.', 'info'),
      enter: () => scenario({ rate: 1.2 }),
    },
    {
      title: 'What it costs',
      text: 'Batching is a trade. Each sequence in a batch of B gets its next token only after t0 + t1 × B ms, so every stream slows down as the batch fills: alone, 46 tokens per second; in a full batch of 32, under 15. A joining request also has its prompt processed inside a step, and everyone else in the batch waits for it.\n\nThe batch limit is now 32 and requests arrive faster. The GPU can deliver up to 470 tokens per second instead of 250, but each user reads slower text. Operators set the limit to balance **time to first token and per-stream speed against throughput**.',
      enter: () => scenario({ rate: 3, maxBatch: 32 }, { speed: 3 }),
    },
    {
      title: 'A simulation, not a benchmark',
      text: `Every number on this page follows from the cost model in the right panel. No GPU is being measured, and real engines add effects left out here: memory limits counted in tokens instead of sequences, prompt caching, prompts processed in chunks, and schedulers that are not first come, first served.\n\nFor measured results see the vLLM paper (${paper}). It combines iteration-level scheduling with paged memory for the key-value cache and reports that vLLM “improves the throughput of popular LLMs by 2-4× with the same level of latency” compared with the best earlier systems, FasterTransformer and Orca.`,
      body: () => callout('Your turn: move the sliders. Try t0 = 2 ms and t1 = 20 ms, a GPU without a fixed cost, and batching stops paying off.', 'key'),
      enter: () => scenario({}),
    },
  ];
  const guide = tour(ctx, steps, { lastLabel: 'Hall of Records →', onFinish: () => ctx.router.go('perf/records') });

  ctx.ui.left(
    header({ kicker: 'Experiment · Batching', title: 'One GPU, many users', lead: 'Four copies of one GPU serve **the same stream of requests**. Only the scheduler differs.' }),
    guide.body,
    callout('Simulated under a stated cost model, not measured on a GPU. The ratios between schedulers are the lesson; the absolute numbers are not.', 'warn'),
  );
  ctx.ui.right(
    h('div.btnrow.bt-top', playBtn, button('Reset', { title: 'Start the trace again from an empty system', onClick: () => { reset(0); setPlaying(true); } }), clock),
    S.rate,
    S.maxBatch,
    S.outSpread,
    S.speed,
    section(
      'In the scene',
      h(
        'div.bt-key',
        h('span', h('i.multi'), 'generating: length = tokens left'),
        h('span', h('i', { style: { '--c': '#f2f5fb' } }), 'processing its prompt'),
        h('span', h('i.stub', { style: { '--c': '#5d6883' } }), 'finished, slot still held (wasted)'),
        h('span', h('i.multi.dim'), 'waiting on the conveyor'),
      ),
    ),
    section(`Live · last ${WINDOW / 1000} simulated seconds`, h('div.bt-legend', SCHEDULERS.map((s, i) => swatch(COLOR[i], s.name))), h('div.bt-grid', groups.map((g) => g.el))),
    section('Queue length over time', chips, chart),
    section('Full run with these settings', resultBox),
    section('Cost model', modelBox, S.t0, S.t1, S.outMean, S.promptMean, S.timeout),
  );
  ctx.ui.dock(guide.dock);
  ctx.key(' ', () => setPlaying(!playing));

  renderChips();
  reset(12000);
  renderFull();

  let uiTimer = 0;
  return {
    update(dt) {
      if (playing) {
        T += dt * 1000 * speed;
        if (T >= HORIZON) {
          ctx.ui.toast('The 30-minute trace has ended. Starting over.');
          reset(0);
        }
        trace.ensure(T);
        for (const s of sims) s.advance(T);
      }
      draw(dt);
      stepParticles(dt);
      uiTimer += dt;
      if (playing && uiTimer > 0.2) {
        uiTimer = 0;
        refresh();
      }
    },
  };
}
