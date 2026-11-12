// Discrete-event simulation of one GPU serving LLM requests under four batching policies.
// Plain JS, seeded and deterministic, no three.js: the lab in ../perf/batching.js draws exactly
// what is computed here, and tools can run it in Node.
//
// Cost model (all times in simulated milliseconds):
//   one iteration of a batch of n sequences takes  t0 + t1·n  (+ tp per prompt token processed)
//   every running sequence gets one token per iteration; a sequence's first iteration also
//   processes its prompt and ends with its first token
//   at most `maxBatch` sequences fit in memory at once
// All four policies stream tokens as they are produced, so time to first token is comparable.

export const SCHEDULERS = [
  { id: 'none', name: 'No batching', short: 'None', rule: 'One request at a time. The other slots stay empty.' },
  { id: 'static', name: 'Static batching', short: 'Static', rule: 'Wait for a full batch (or a timeout), then run it until its longest sequence ends.' },
  { id: 'dynamic', name: 'Dynamic batching', short: 'Dynamic', rule: 'Take whatever is queued when the GPU is free, then run it until its longest sequence ends.' },
  { id: 'continuous', name: 'Continuous batching', short: 'Continuous', rule: 'Sequences join and leave the batch at every decoding step.' },
];

export const DEFAULTS = {
  rate: 1.2, // requests per second (Poisson arrivals)
  maxBatch: 8, // sequences that fit in memory at once
  t0: 20, // ms: fixed cost of one decoding step
  t1: 1.5, // ms: extra cost per sequence in the batch
  tp: 0.04, // ms per prompt token (prompt processing)
  outMean: 128, // mean output length, tokens
  outSpread: 0.8, // coefficient of variation of the output length (0 = every answer the same length)
  outMax: 512, // cap on the output length ("max_tokens")
  promptMean: 512, // mean prompt length, tokens
  promptSpread: 0.8,
  promptMax: 4096,
  timeout: 1000, // ms: how long static batching waits for a full batch before running a partial one
  seed: 7,
};

/** Slot states. HELD = the sequence has finished but its slot is not released until the batch ends. */
export const SLOT = { IDLE: 0, PREFILL: 1, DECODE: 2, HELD: 3 };
const { IDLE, PREFILL, DECODE, HELD } = SLOT;

export const MAX_SLOTS = 64;

/** Duration of one iteration with n sequences in the batch, `promptTokens` of them new. */
export function stepTime(p, n, promptTokens = 0) {
  return Math.max(0.01, p.t0 + p.t1 * n + p.tp * promptTokens);
}

/** Deterministic pseudo-random generator (mulberry32) -> [0, 1). */
export function mulberry32(seed = 1) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The arrival trace: a Poisson process whose requests carry a prompt length and an output length.
 * It is generated lazily and shared by every simulated GPU, so all of them serve the same requests.
 * `params` is read live: a new rate or length distribution applies to the arrivals generated next.
 */
export class Trace {
  /** @param {typeof DEFAULTS} params  @param {{horizon?: number}} [o] horizon: no arrivals after this time (ms) */
  constructor(params, o = {}) {
    this.p = params;
    this.rnd = mulberry32(params.seed ?? 1);
    this.horizon = o.horizon ?? Infinity;
    this.t = []; // arrival times, ms, ascending
    this.prompt = []; // prompt tokens
    this.out = []; // output tokens the request will generate
    this.next = this._gap(); // time of the next arrival that has not been materialised yet
    this._ptr = 0;
  }

  _gap() {
    return (-Math.log(1 - this.rnd()) / Math.max(1e-6, this.p.rate)) * 1000;
  }

  _normal() {
    let u = 0;
    while (u === 0) u = this.rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * this.rnd());
  }

  /** Log-normal sample with the given mean and coefficient of variation, rounded and clamped to [1, max]. */
  _length(mean, cv, max) {
    const z = this._normal(); // drawn even when cv = 0 so the random stream does not depend on the spread
    if (cv <= 0) return Math.max(1, Math.min(max, Math.round(mean)));
    const s2 = Math.log(1 + cv * cv);
    const x = Math.exp(Math.log(mean) - s2 / 2 + Math.sqrt(s2) * z);
    return Math.max(1, Math.min(max, Math.round(x)));
  }

  /** Materialise every arrival up to time T (ms). */
  ensure(T) {
    const p = this.p;
    while (this.next <= T && this.next <= this.horizon) {
      this.t.push(this.next);
      this.prompt.push(this._length(p.promptMean, p.promptSpread, p.promptMax));
      this.out.push(this._length(p.outMean, p.outSpread, p.outMax));
      this.next += this._gap();
    }
  }

  /** True once no further arrival can appear. */
  get finished() {
    return this.next > this.horizon;
  }

  /** The arrival rate changed at time `now`: a Poisson process is memoryless, so redraw the pending gap. */
  retime(now) {
    this.ensure(now);
    this.next = now + this._gap();
  }

  /** Number of arrivals with time <= T. T must not decrease between calls. */
  arrived(T) {
    while (this._ptr < this.t.length && this.t[this._ptr] <= T) this._ptr++;
    return this._ptr;
  }
}

/** One GPU under one scheduling policy. `advance(T)` processes every event up to simulated time T. */
export class GpuSim {
  /**
   * @param {'none'|'static'|'dynamic'|'continuous'} policy
   * @param {Trace} trace
   * @param {typeof DEFAULTS} params read live at every decision
   * @param {{bucket?: number}} [o] bucket: width of the time-series buckets, ms
   */
  constructor(policy, trace, params, o = {}) {
    this.policy = policy;
    this.trace = trace;
    this.p = params;
    this.dt = o.bucket ?? 250;

    this.slotReq = new Int32Array(MAX_SLOTS).fill(-1); // request index in each slot
    this.slotRem = new Int32Array(MAX_SLOTS); // tokens it still has to generate
    this.slotState = new Uint8Array(MAX_SLOTS);
    this.emitted = new Uint16Array(MAX_SLOTS); // tokens per slot since the renderer last cleared it

    this.now = 0; // simulated time of the last processed event
    this.head = 0; // requests admitted so far; the queue is trace[head .. arrived)
    this.busy = false; // an iteration is in progress
    this.iterStart = 0;
    this.iterEnd = 0;
    this.iterN = 0; // sequences in the current iteration (including finished ones that keep a slot)
    this.iterUseful = 0;
    this.iterHeld = 0;
    this.active = 0; // sequences still generating
    this.occupied = 0; // active + held
    this.topSlot = 0; // one past the highest occupied slot

    this.admit = []; // per request: time it left the queue
    this.first = []; // per request: time of its first token
    this.done = []; // per request: time of its last token
    this.started = []; // request indices in order of first token
    this.completed = []; // request indices in order of completion
    this.lastDone = 0;

    this.tokens = 0;
    this.steps = 0;
    this.busyMs = 0;
    this.usefulMs = 0; // slot-milliseconds spent on sequences that were generating
    this.wastedMs = 0; // slot-milliseconds held by finished sequences
    this.capMs = 0; // slot-milliseconds available (maxBatch × time)
    this._capAt = 0;

    // time series, one entry per bucket of `dt` ms
    this.qSeries = []; // queue length sampled at the start of the bucket
    this.capSeries = []; // maxBatch at that moment
    this.tokSeries = []; // tokens produced in the bucket
    this.useSeries = []; // useful slot-ms in the bucket
    this.wasteSeries = []; // wasted slot-ms in the bucket
    this._arr = 0; // arrivals seen by the scheduler
    this._sArr = 0; // arrivals seen by the sampler
    this._nextSample = 0;
  }

  /** Requests waiting at time T (>= the last advance). */
  queueAt(T) {
    const t = this.trace.t;
    let a = Math.max(this._arr, this._sArr);
    while (a < t.length && t[a] <= T) a++;
    return a - this.head;
  }

  /** Fraction of the current iteration that has elapsed at time T (0 when idle). */
  progress(T) {
    if (!this.busy) return 0;
    return Math.min(1, Math.max(0, (T - this.iterStart) / (this.iterEnd - this.iterStart)));
  }

  _sampleUntil(t, inclusive) {
    const tt = this.trace.t;
    while (this._nextSample < t || (inclusive && this._nextSample <= t)) {
      while (this._sArr < tt.length && tt[this._sArr] <= this._nextSample) this._sArr++;
      this.qSeries.push(Math.max(0, this._sArr - this.head));
      this.capSeries.push(Math.max(1, this.p.maxBatch | 0));
      this._nextSample += this.dt;
    }
  }

  _grow(b) {
    while (this.tokSeries.length <= b) {
      this.tokSeries.push(0);
      this.useSeries.push(0);
      this.wasteSeries.push(0);
    }
  }

  _limit() {
    return this.policy === 'none' ? 1 : Math.max(1, Math.min(MAX_SLOTS, this.p.maxBatch | 0));
  }

  _capacityUntil(t) {
    // capacity is what the hardware could hold: maxBatch slots, whatever the policy uses
    this.capMs += Math.max(1, this.p.maxBatch | 0) * (t - this._capAt);
    this._capAt = t;
  }

  _admit(k, lowestFree) {
    const tr = this.trace;
    this._sampleUntil(this.now, false);
    let s = 0;
    for (let i = 0; i < k; i++) {
      if (lowestFree) while (this.slotState[s] !== IDLE) s++;
      else s = i;
      const r = this.head++;
      this.slotReq[s] = r;
      this.slotRem[s] = tr.out[r];
      this.slotState[s] = PREFILL;
      this.admit[r] = this.now;
      this.active++;
      this.occupied++;
      if (s + 1 > this.topSlot) this.topSlot = s + 1;
    }
  }

  _start() {
    const tr = this.trace;
    let prompt = 0;
    let useful = 0;
    let held = 0;
    for (let s = 0; s < this.topSlot; s++) {
      const st = this.slotState[s];
      if (st === PREFILL) {
        prompt += tr.prompt[this.slotReq[s]];
        useful++;
      } else if (st === DECODE) useful++;
      else if (st === HELD) held++;
    }
    this.iterN = useful + held;
    this.iterUseful = useful;
    this.iterHeld = held;
    this.iterStart = this.now;
    this.iterEnd = this.now + stepTime(this.p, this.iterN, prompt);
    this.busy = true;
  }

  _finish() {
    const s0 = this.iterStart;
    const e = this.iterEnd;
    const dur = e - s0;
    const dt = this.dt;
    // spread the slot-time of this iteration over the buckets it overlaps
    const b0 = Math.floor(s0 / dt);
    const b1 = Math.floor(e / dt);
    this._grow(b1);
    for (let b = b0; b <= b1; b++) {
      const ov = Math.min(e, (b + 1) * dt) - Math.max(s0, b * dt);
      if (ov <= 0) continue;
      this.useSeries[b] += this.iterUseful * ov;
      this.wasteSeries[b] += this.iterHeld * ov;
    }
    this.usefulMs += this.iterUseful * dur;
    this.wastedMs += this.iterHeld * dur;
    this.busyMs += dur;
    this.steps++;

    const runToCompletion = this.policy === 'static' || this.policy === 'dynamic';
    let top = 0;
    for (let s = 0; s < this.topSlot; s++) {
      const st = this.slotState[s];
      if (st === PREFILL || st === DECODE) {
        const r = this.slotReq[s];
        this.slotRem[s]--;
        this.tokens++;
        this.tokSeries[b1]++;
        if (this.emitted[s] < 65535) this.emitted[s]++;
        if (st === PREFILL) {
          this.first[r] = e;
          this.started.push(r);
          this.slotState[s] = DECODE;
        }
        if (this.slotRem[s] <= 0) {
          this.done[r] = e;
          this.completed.push(r);
          this.lastDone = e;
          this.active--;
          if (runToCompletion) this.slotState[s] = HELD;
          else {
            this.slotState[s] = IDLE;
            this.slotReq[s] = -1;
            this.occupied--;
          }
        }
      }
      if (this.slotState[s] !== IDLE) top = s + 1;
    }
    if (runToCompletion && this.active === 0) {
      // the longest sequence has ended: the whole batch is released at once
      for (let s = 0; s < this.topSlot; s++) {
        this.slotState[s] = IDLE;
        this.slotReq[s] = -1;
      }
      this.occupied = 0;
      top = 0;
    }
    this.topSlot = top;
    this.now = e;
    this.busy = false;
  }

  /**
   * The GPU is free at `this.now`: start the next iteration, or return the time at which the
   * decision has to be revisited (the next arrival or a timeout; Infinity if nothing is pending).
   */
  _decide() {
    const tr = this.trace;
    const tt = tr.t;
    while (this._arr < tt.length && tt[this._arr] <= this.now) this._arr++;
    const avail = this._arr - this.head;
    const nextArrival = this._arr < tt.length ? tt[this._arr] : Infinity;
    const limit = this._limit();

    if (this.policy === 'static' || this.policy === 'dynamic') {
      if (this.active > 0) return this._start();
      if (avail === 0) return nextArrival;
      if (this.policy === 'dynamic' || avail >= limit) {
        this._admit(Math.min(avail, limit), false);
        return this._start();
      }
      const deadline = tt[this.head] + this.p.timeout;
      if (this.now >= deadline) {
        this._admit(avail, false);
        return this._start();
      }
      return Math.min(nextArrival, deadline);
    }

    // continuous batching, and no batching as the special case of a batch limit of one
    let free = 0;
    for (let s = 0; s < limit; s++) if (this.slotState[s] === IDLE) free++;
    const k = Math.min(avail, free);
    if (k > 0) this._admit(k, true);
    if (this.occupied > 0) return this._start();
    return nextArrival;
  }

  /** Process every event up to simulated time T (ms). Pass Infinity to run a finite trace to the end. */
  advance(T) {
    const tr = this.trace;
    tr.ensure(T);
    for (;;) {
      if (this.busy) {
        if (this.iterEnd > T) break;
        this._finish();
      }
      const wake = this._decide();
      if (wake === undefined) continue; // an iteration was started
      if (wake > T || wake === Infinity) {
        // nothing more can happen before T (or ever, when a finite trace has been served)
        if (Number.isFinite(T)) this.now = T;
        break;
      }
      this.now = wake;
    }
    const until = Number.isFinite(T) ? T : Math.max(this.now, this.lastDone);
    this._sampleUntil(until, true);
    this._capacityUntil(until);
    return this;
  }

  /** True when a finite trace has been served completely. */
  get drained() {
    return this.trace.finished && this.head >= this.trace.t.length && this.occupied === 0 && !this.busy;
  }
}

/** q-th quantile (0..1) of an ascending array, linear interpolation. NaN for an empty array. */
export function quantile(sorted, q) {
  const n = sorted.length;
  if (!n) return NaN;
  const x = (n - 1) * q;
  const i = Math.floor(x);
  const j = Math.min(n - 1, i + 1);
  return sorted[i] + (sorted[j] - sorted[i]) * (x - i);
}

const byNumber = (a, b) => a - b;

/**
 * Metrics over the window (T - W, T] of a live simulation.
 * Times are ms; throughput is tokens per simulated second.
 */
export function windowStats(sim, T, W = 20000) {
  const tr = sim.trace;
  const b0 = Math.max(0, Math.floor((T - W) / sim.dt));
  const from = b0 * sim.dt; // the window starts on a bucket boundary
  const span = Math.max(1, T - from);
  const b1 = Math.min(sim.tokSeries.length - 1, Math.floor(T / sim.dt));
  let tok = 0;
  let use = 0;
  let waste = 0;
  for (let b = b0; b <= b1; b++) {
    tok += sim.tokSeries[b];
    use += sim.useSeries[b];
    waste += sim.wasteSeries[b];
  }
  // slots the hardware offered during the window (the batch limit may have been changed live)
  let slots = 0;
  let nb = 0;
  const c1 = Math.min(sim.capSeries.length - 1, Math.floor(T / sim.dt));
  for (let b = b0; b <= c1; b++) {
    slots += sim.capSeries[b];
    nb++;
  }
  const cap = Math.max(1, (nb ? slots / nb : Math.max(1, sim.p.maxBatch | 0)) * span);
  const ttft = [];
  for (let i = sim.started.length - 1; i >= 0; i--) {
    const r = sim.started[i];
    if (sim.first[r] < from) break;
    ttft.push(sim.first[r] - tr.t[r]);
  }
  const e2e = [];
  for (let i = sim.completed.length - 1; i >= 0; i--) {
    const r = sim.completed[i];
    if (sim.done[r] < from) break;
    e2e.push(sim.done[r] - tr.t[r]);
  }
  ttft.sort(byNumber);
  e2e.sort(byNumber);
  return {
    throughput: (tok / span) * 1000,
    ttft50: quantile(ttft, 0.5),
    ttft95: quantile(ttft, 0.95),
    e2e50: quantile(e2e, 0.5),
    e2e95: quantile(e2e, 0.95),
    util: Math.min(1, use / cap),
    wasted: Math.min(1, waste / cap),
    queue: sim.queueAt(T),
    batch: sim.busy ? sim.iterN : 0,
    stepMs: sim.busy ? sim.iterEnd - sim.iterStart : 0,
    finished: e2e.length,
  };
}

/** Metrics over a whole (finished) run. */
export function summarize(sim) {
  const tr = sim.trace;
  const n = sim.completed.length;
  const ttft = new Array(n);
  const e2e = new Array(n);
  const wait = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = sim.completed[i];
    ttft[i] = sim.first[r] - tr.t[r];
    e2e[i] = sim.done[r] - tr.t[r];
    wait[i] = sim.admit[r] - tr.t[r];
  }
  ttft.sort(byNumber);
  e2e.sort(byNumber);
  wait.sort(byNumber);
  const span = Math.max(1, sim.lastDone);
  let qSum = 0;
  let qMax = 0;
  for (const q of sim.qSeries) {
    qSum += q;
    if (q > qMax) qMax = q;
  }
  return {
    policy: sim.policy,
    requests: n,
    tokens: sim.tokens,
    makespan: span,
    throughput: (sim.tokens / span) * 1000,
    ttft50: quantile(ttft, 0.5),
    ttft95: quantile(ttft, 0.95),
    e2e50: quantile(e2e, 0.5),
    e2e95: quantile(e2e, 0.95),
    wait50: quantile(wait, 0.5),
    util: sim.capMs > 0 ? sim.usefulMs / sim.capMs : 0,
    wasted: sim.capMs > 0 ? sim.wastedMs / sim.capMs : 0,
    busy: sim.capMs > 0 ? sim.busyMs / Math.max(1, sim._capAt) : 0,
    meanBatch: sim.busyMs > 0 ? (sim.usefulMs + sim.wastedMs) / sim.busyMs : 0,
    meanStep: sim.steps > 0 ? sim.busyMs / sim.steps : 0,
    queueMean: sim.qSeries.length ? qSum / sim.qSeries.length : 0,
    queueMax: qMax,
  };
}

/**
 * Serve one fixed trace (arrivals during `horizon` ms) to completion under every policy.
 * @returns {{policy:string}[]} one summary per scheduler, in SCHEDULERS order
 */
export function runFull(params, { horizon = 120000, series = false } = {}) {
  const p = { ...params };
  const trace = new Trace(p, { horizon }); // one trace, served four times
  trace.ensure(Infinity);
  const offered = trace.out.reduce((a, b) => a + b, 0) / (horizon / 1000); // tokens per second asked for
  return SCHEDULERS.map((s) => {
    const sim = new GpuSim(s.id, trace, p);
    sim.advance(Infinity);
    const out = summarize(sim);
    out.offered = offered;
    out.arrivals = trace.t.length;
    if (series) out.queue = sim.qSeries;
    return out;
  });
}
