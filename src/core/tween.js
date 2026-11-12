// Minimal tween engine. Every chapter gets its own TweenGroup so that leaving the chapter
// cancels whatever is still animating.

export const ease = {
  linear: (t) => t,
  in: (t) => t * t * t,
  out: (t) => 1 - Math.pow(1 - t, 3),
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  inOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,
  outQuint: (t) => 1 - Math.pow(1 - t, 5),
  outBack: (t) => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
  outElastic: (t) => {
    if (t === 0 || t === 1) return t;
    return Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * ((2 * Math.PI) / 3)) + 1;
  },
};

export const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, v) => {
  const t = clamp((v - a) / (b - a));
  return t * t * (3 - 2 * t);
};
/** Frame-rate independent exponential approach: value moves toward target with time constant `tau` seconds. */
export const damp = (value, target, tau, dt) => target + (value - target) * Math.exp(-dt / Math.max(tau, 1e-4));

export class TweenGroup {
  constructor() {
    this.items = new Set();
  }

  /**
   * @param {{dur?: number, delay?: number, ease?: (t:number)=>number, update?: (k:number, t:number)=>void,
   *          done?: ()=>void, loop?: boolean, yoyo?: boolean}} o
   *        update receives k = eased progress 0..1 and t = raw progress.
   * @returns {{cancel: ()=>void, promise: Promise<boolean>}} promise resolves true when finished, false if canceled.
   */
  add(o) {
    let resolve;
    const promise = new Promise((r) => (resolve = r));
    const item = {
      dur: Math.max(o.dur ?? 1, 1e-4),
      delay: o.delay ?? 0,
      ease: o.ease ?? ease.inOut,
      update: o.update,
      done: o.done,
      loop: !!o.loop,
      yoyo: !!o.yoyo,
      t: 0,
      dir: 1,
      resolve,
    };
    this.items.add(item);
    return {
      promise,
      cancel: () => {
        if (this.items.delete(item)) resolve(false);
      },
    };
  }

  /** Tween numeric properties of `target` to the values in `props`. */
  to(target, props, o = {}) {
    const keys = Object.keys(props);
    let from = null;
    return this.add({
      ...o,
      update: (k, t) => {
        if (!from) from = keys.map((key) => target[key]);
        for (let i = 0; i < keys.length; i++) target[keys[i]] = from[i] + (props[keys[i]] - from[i]) * k;
        o.update?.(k, t);
      },
    });
  }

  wait(seconds) {
    return this.add({ dur: seconds, ease: ease.linear }).promise;
  }

  update(dt) {
    for (const it of this.items) {
      if (it.delay > 0) {
        it.delay -= dt;
        continue;
      }
      it.t += (dt / it.dur) * it.dir;
      let finished = false;
      if (it.t >= 1) {
        if (it.loop && it.yoyo) {
          it.t = 1;
          it.dir = -1;
        } else if (it.loop) {
          it.t -= 1;
        } else {
          it.t = 1;
          finished = true;
        }
      } else if (it.t <= 0 && it.dir < 0) {
        it.t = 0;
        it.dir = 1;
      }
      it.update?.(it.ease(it.t), it.t);
      if (finished) {
        this.items.delete(it);
        it.done?.();
        it.resolve(true);
      }
    }
  }

  clear() {
    for (const it of this.items) it.resolve(false);
    this.items.clear();
  }
}
