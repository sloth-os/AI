// The engine: one WebGL canvas, one camera, one render loop. Chapters are mounted into `ctx.root`
// one at a time; the App owns transitions, picking, quality and the HUD.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { TweenGroup, ease } from './tween.js';
import { Background } from './background.js';
import { Hud } from './hud.js';
import { disposeDeep, v3 } from './geom.js';
import { flushLabelCache } from './labels.js';
import { matchRoute, NAV } from '../routes.js';
import { h, button } from './ui.js';

const QUALITY = {
  high: { dpr: 2, bloom: true, samples: 4, nebula: true },
  medium: { dpr: 1.5, bloom: true, samples: 0, nebula: true },
  low: { dpr: 1, bloom: false, samples: 0, nebula: false },
};
const DEFAULT_VIEW = { pos: [0, 6, 22], target: [0, 0, 0], min: 2, max: 120, minPolar: 0.05, maxPolar: Math.PI * 0.92, fov: 50 };

class Picker {
  constructor(app) {
    this.app = app;
    this.map = new Map();
    this.ray = new THREE.Raycaster();
    this.ray.params.Line.threshold = 0.15;
    this.ray.params.Points.threshold = 0.2;
    this.ndc = new THREE.Vector2();
    this.pointer = null; // last pointer event position
    this.dirty = false;
    this.hover = null; // { object, handlers, hit }
    this.down = null;
    const dom = app.renderer.domElement;
    dom.addEventListener('pointermove', (e) => {
      this.pointer = { x: e.clientX, y: e.clientY, type: e.pointerType };
      this.dirty = true;
    });
    dom.addEventListener('pointerleave', () => {
      this.pointer = null;
      this._setHover(null);
    });
    dom.addEventListener('pointerdown', (e) => {
      this.down = { x: e.clientX, y: e.clientY, t: performance.now() };
      this.pointer = { x: e.clientX, y: e.clientY, type: e.pointerType };
      this.dirty = true;
    });
    dom.addEventListener('pointerup', (e) => {
      const d = this.down;
      this.down = null;
      if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 6 || performance.now() - d.t > 600) return;
      this.pointer = { x: e.clientX, y: e.clientY, type: e.pointerType };
      const found = this._cast();
      if (found) found.handlers.click?.(found.hit, e);
      else this.onMiss?.(e);
      if (e.pointerType !== 'mouse') {
        // touch has no hover: show the tooltip briefly after a tap
        this._setHover(found);
        clearTimeout(this._touchTimer);
        this._touchTimer = setTimeout(() => this._setHover(null), 2200);
      }
    });
  }

  /** Register object(s) for hover / click. handlers: { hover(hit), out(), click(hit, event), tip: string | (hit) => string|Node, cursor } */
  add(objs, handlers) {
    const list = Array.isArray(objs) ? objs : [objs];
    for (const o of list) this.map.set(o, handlers);
    return () => list.forEach((o) => this.remove(o));
  }

  remove(o) {
    if (this.hover?.object === o) this._setHover(null);
    this.map.delete(o);
  }

  clear() {
    this._setHover(null);
    this.map.clear();
    this.onMiss = null;
  }

  _cast() {
    if (!this.pointer || this.map.size === 0) return null;
    const rect = this.app.renderer.domElement.getBoundingClientRect();
    this.ndc.set(((this.pointer.x - rect.left) / rect.width) * 2 - 1, -((this.pointer.y - rect.top) / rect.height) * 2 + 1);
    this.ray.setFromCamera(this.ndc, this.app.camera);
    const targets = [];
    for (const o of this.map.keys()) if (o.visible && o.parent) targets.push(o);
    const hits = this.ray.intersectObjects(targets, false);
    for (const hit of hits) {
      // skip objects hidden through an invisible ancestor
      let p = hit.object;
      let visible = true;
      while (p) {
        if (!p.visible) {
          visible = false;
          break;
        }
        p = p.parent;
      }
      if (!visible) continue;
      const handlers = this.map.get(hit.object);
      if (handlers.filter && !handlers.filter(hit)) continue;
      return { object: hit.object, handlers, hit };
    }
    return null;
  }

  _setHover(found) {
    const prev = this.hover;
    const same = prev && found && prev.object === found.object && prev.hit.instanceId === found.hit.instanceId;
    if (!same) {
      prev?.handlers.out?.(prev.hit);
      this.hover = found;
      found?.handlers.hover?.(found.hit);
      this.app.renderer.domElement.style.cursor = found ? (found.handlers.cursor ?? (found.handlers.click ? 'pointer' : 'default')) : '';
    } else {
      this.hover = found;
      found.handlers.move?.(found.hit);
    }
    const tip = found?.handlers.tip;
    const content = typeof tip === 'function' ? tip(found.hit) : tip;
    this.app.hud.tip(content ?? null, this.pointer?.x ?? 0, this.pointer?.y ?? 0);
  }

  update() {
    if (!this.dirty) return;
    this.dirty = false;
    if (this.down && this.pointer && Math.hypot(this.pointer.x - this.down.x, this.pointer.y - this.down.y) > 6) {
      this._setHover(null); // dragging the camera
      return;
    }
    if (this.pointer?.type !== 'mouse') return;
    this._setHover(this._cast());
  }
}

export class App {
  constructor({ canvas, router }) {
    this.router = router;
    const params = new URLSearchParams(location.search);
    this.reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    this.isTouch = window.matchMedia?.('(pointer: coarse)').matches ?? false;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.setClearColor('#05070d', 1);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2('#05070d', 0.006);
    this.camera = new THREE.PerspectiveCamera(DEFAULT_VIEW.fov, 1, 0.1, 3000);
    this.camera.position.fromArray(DEFAULT_VIEW.pos);

    this.scene.add(new THREE.HemisphereLight('#bcd3ff', '#10162a', 0.9));
    const key = new THREE.DirectionalLight('#ffffff', 1.5);
    key.position.set(6, 12, 8);
    const rim = new THREE.DirectionalLight('#6aa8ff', 0.7);
    rim.position.set(-8, 4, -10);
    this.scene.add(key, rim);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.62;
    this.controls.zoomSpeed = 0.9;
    this.controls.panSpeed = 0.7;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener('start', () => {
      this._fly?.cancel();
      this._userActive = true;
      this._userMoved = true;
      this.controls.autoRotate = false;
      clearTimeout(this._idleTimer);
    });
    this.controls.addEventListener('end', () => {
      this._userActive = false;
      clearTimeout(this._idleTimer);
      if (this._autoRotate) this._idleTimer = setTimeout(() => (this.controls.autoRotate = true), 5000);
    });

    this.css2d = new CSS2DRenderer({ element: document.getElementById('labels') });

    const stored = (() => {
      try {
        return localStorage.getItem('atlas.quality');
      } catch {
        return null;
      }
    })();
    const wanted = params.get('q') || stored;
    this.quality = QUALITY[wanted] ? wanted : this.isTouch ? 'medium' : 'high';
    this._autoQuality = !QUALITY[wanted];

    this.background = new Background(this.scene, { stars: this.quality === 'low' ? 2500 : 5200, nebula: QUALITY[this.quality].nebula });
    this.tweens = new TweenGroup();
    this.hud = new Hud(this);
    this.picker = new Picker(this);
    this.current = null;
    this._navToken = 0;
    this._home = { ...DEFAULT_VIEW };
    this.time = 0;
    this._frames = [];
    this.showFps = params.has('fps');
    if (this.showFps) {
      this.fpsEl = h('div.fps', '');
      document.body.appendChild(this.fpsEl);
    }

    this._buildComposer();
    this._resize();
    window.addEventListener('resize', () => this._resize());
    window.addEventListener('keydown', (e) => this._onKey(e));
    this.hud.setQualityLabel(this.quality);

    this._last = performance.now();
    this.renderer.setAnimationLoop(() => this._tick());

    router.onChange((r) => this._navigate(r));
  }

  // ------------------------------------------------------------------ rendering
  _buildComposer() {
    const q = QUALITY[this.quality];
    this.composer?.dispose?.();
    this.composer = null;
    if (!q.bloom) return;
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    const rt = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, samples: q.samples });
    const composer = new EffectComposer(this.renderer, rt);
    composer.addPass(new RenderPass(this.scene, this.camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(size.x, size.y), this._bloom?.strength ?? 0.62, this._bloom?.radius ?? 0.72, 1.0);
    // Selective bloom without a second scene pass: only HDR pixels (any channel above 1) glow,
    // so ordinary lit surfaces and text labels stay crisp. See colors.js `glow()`.
    bloom.materialHighPassFilter.fragmentShader = bloom.materialHighPassFilter.fragmentShader.replace(
      'float v = luminance( texel.xyz );',
      'float v = max( texel.r, max( texel.g, texel.b ) );',
    );
    bloom.materialHighPassFilter.needsUpdate = true;
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
    this.composer = composer;
    this.bloomPass = bloom;
  }

  /** Chapters may tune the glow. Defaults are restored when the chapter is left. */
  setBloom({ strength, radius } = {}) {
    this._bloom = { strength: strength ?? 0.62, radius: radius ?? 0.72 };
    if (this.bloomPass) {
      this.bloomPass.strength = this._bloom.strength;
      this.bloomPass.radius = this._bloom.radius;
    }
  }

  setFog(density = 0.006) {
    this.scene.fog.density = density;
  }

  setQuality(q, { auto = false } = {}) {
    if (!QUALITY[q] || q === this.quality) return;
    this.quality = q;
    if (!auto) {
      this._autoQuality = false;
      try {
        localStorage.setItem('atlas.quality', q);
      } catch {
        /* private mode */
      }
    }
    this.background.setDetail(QUALITY[q].nebula);
    this._buildComposer();
    this._resize();
    this.hud.setQualityLabel(q);
    if (this.current) this.current.ctx.quality = q;
  }

  cycleQuality() {
    const order = ['high', 'medium', 'low'];
    const next = order[(order.indexOf(this.quality) + 1) % order.length];
    this.setQuality(next);
    this.hud.toast(`Rendering quality: ${next}`);
  }

  _resize() {
    const w = window.innerWidth;
    const hgt = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, QUALITY[this.quality].dpr);
    this.pixelRatio = dpr;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, hgt, false);
    this.camera.aspect = w / hgt;
    // keep the horizontal field of view usable on portrait phones
    this.camera.fov = this._home.fov * (w / hgt < 0.8 ? 1.35 : 1);
    this.camera.updateProjectionMatrix();
    if (this.composer) {
      this.composer.setPixelRatio(dpr);
      this.composer.setSize(w, hgt);
    }
    this.css2d.setSize(w, hgt);
    this.layout();
  }

  /**
   * Shift the projection so the scene's center sits in the middle of the area the panels leave free
   * (called by the HUD whenever a panel appears, collapses or resizes).
   */
  layout() {
    const hud = this.hud;
    if (!hud) return;
    let sx = 0;
    let sy = 0;
    const open = (p) => !p.el.hidden && !p.el.classList.contains('collapsed') && !hud.root.classList.contains('no-panels');
    if (window.innerWidth > 860) {
      const lw = open(hud.L) ? hud.L.el.offsetWidth + 12 : 0;
      const rw = open(hud.R) ? hud.R.el.offsetWidth + 12 : 0;
      sx = (lw - rw) / 2;
    } else {
      const sheet = hud.sheet === 'left' ? hud.L : hud.sheet === 'right' ? hud.R : null;
      if (sheet && open(sheet)) sy = sheet.el.offsetHeight / 2;
    }
    this._shiftTarget = { x: sx, y: sy };
    this._shift ??= { x: sx, y: sy };
    if (this._home?.fit && !this._userMoved && this.controls) {
      this._home.pos = this._fitPos(this._home);
      if (!this._fly || !this.tweens.items.size) this.camera.position.fromArray(this._home.pos);
    }
  }

  _applyShift(dt) {
    const s = this._shift;
    const g = this._shiftTarget;
    if (!s || !g) return;
    const k = 1 - Math.exp(-dt * 7);
    s.x += (g.x - s.x) * k;
    s.y += (g.y - s.y) * k;
    const w = window.innerWidth;
    const hgt = window.innerHeight;
    if (Math.abs(s.x) < 0.5 && Math.abs(s.y) < 0.5) {
      if (this.camera.view) this.camera.clearViewOffset();
    } else this.camera.setViewOffset(w, hgt, -s.x, s.y, w, hgt);
  }

  _tick() {
    const now = performance.now();
    const raw = (now - this._last) / 1000;
    this._last = now;
    const dt = Math.min(raw, 0.05);
    // tweens follow the wall clock (up to a point) so camera moves do not drag on slow devices
    const dtTween = Math.min(raw, 0.4);
    this.time += dt;
    this.tweens.update(dtTween);
    const cur = this.current;
    if (cur) {
      cur.ctx.tweens.update(dtTween);
      try {
        cur.ctrl.update?.(dt, this.time);
      } catch (err) {
        if (!cur.failed) {
          cur.failed = true;
          console.error('[chapter update]', err);
        }
      }
    }
    this.controls.update(dt);
    this._applyShift(dt);
    this.background.update(dt, this.time, this.camera, this.pixelRatio);
    this.picker.update();
    if (this.composer) this.composer.render(dt);
    else this.renderer.render(this.scene, this.camera);
    this.css2d.render(this.scene, this.camera);
    this._measure(raw);
  }

  _measure(raw) {
    const f = this._frames;
    f.push(raw);
    if (f.length < 90) return;
    const avg = f.reduce((a, b) => a + b, 0) / f.length;
    f.length = 0;
    if (this.showFps) this.fpsEl.textContent = `${(1 / avg).toFixed(0)} fps · ${this.quality} · ${this.renderer.info.render.calls} calls`;
    if (!this._autoQuality || document.hidden || this._navigating) return;
    // step down when the device cannot keep up (two bad windows in a row)
    const slow = (this.quality === 'high' && avg > 1 / 34) || (this.quality === 'medium' && avg > 1 / 24);
    this._slowWindows = slow ? (this._slowWindows || 0) + 1 : 0;
    if (this._slowWindows >= 2) {
      this._slowWindows = 0;
      this.setQuality(this.quality === 'high' ? 'medium' : 'low', { auto: true });
      this.hud.toast('Switched to a lighter rendering mode for smoother motion');
    }
  }

  // ------------------------------------------------------------------ camera
  /**
   * Set the chapter's camera immediately and remember it as "home" (the Reset view button).
   * Pass `fit: radius` to have the distance chosen automatically so that a sphere of that radius around
   * `target` fits between the panels on any screen; `pos` then only gives the viewing direction.
   * @param {{pos:number[], target?:number[], fit?:number, min?:number, max?:number, minPolar?:number, maxPolar?:number, fov?:number, autoRotate?:number}} v
   */
  setView(v) {
    const view = { ...DEFAULT_VIEW, ...v };
    this._home = view;
    this._fly?.cancel();
    this._userMoved = false;
    this._resize();
    if (view.fit) view.pos = this._fitPos(view);
    this.camera.position.fromArray(view.pos);
    this.controls.target.fromArray(view.target);
    this.controls.minDistance = view.min;
    this.controls.maxDistance = view.max;
    this.controls.minPolarAngle = view.minPolar;
    this.controls.maxPolarAngle = view.maxPolar;
    this._autoRotate = !!view.autoRotate && !this.reduceMotion;
    this.controls.autoRotateSpeed = view.autoRotate || 0;
    this.controls.autoRotate = this._autoRotate;
    this.controls.update();
  }

  /**
   * Camera position that frames a sphere of radius `view.fit` around `view.target`, seen from the
   * direction of `view.pos`, inside the part of the viewport that the panels leave free.
   */
  _fitPos(view) {
    const w = window.innerWidth;
    const hgt = window.innerHeight;
    const hud = this.hud;
    const open = (p) => !p.el.hidden && !p.el.classList.contains('collapsed') && !hud.root.classList.contains('no-panels');
    let freeW = w;
    let freeH = hgt - 56 - 84;
    if (w > 860) freeW = w - (open(hud.L) ? hud.L.el.offsetWidth + 24 : 0) - (open(hud.R) ? hud.R.el.offsetWidth + 24 : 0);
    else {
      const sheet = hud.sheet === 'left' ? hud.L : hud.sheet === 'right' ? hud.R : null;
      freeH = hgt - 100 - (sheet && open(sheet) ? sheet.el.offsetHeight + 70 : 70);
    }
    const tanV = Math.tan((this.camera.fov * Math.PI) / 360);
    const half = Math.atan(tanV * Math.max(0.2, Math.min(freeW, freeH) / hgt));
    const dist = Math.min(view.max * 0.95, Math.max(view.min * 1.05, view.fit / Math.sin(half)));
    const dir = v3(view.pos).sub(v3(view.target)).normalize();
    return v3(view.target).addScaledVector(dir, dist).toArray();
  }

  /**
   * Fly so that a sphere of radius `fit` around `target` fills the free area, looking along `dir`
   * (a vector from the target toward the camera). Returns the flight's promise.
   */
  frame({ target, dir, fit }, dur = 1.2) {
    const t = v3(target).toArray();
    const pos = this._fitPos({ ...this._home, target: t, pos: v3(target).add(v3(dir)).toArray(), fit });
    return this.flyTo(pos, t, dur);
  }

  /** Change only the remembered home pose (used when a chapter has several sub-views). */
  setHome(pos, target) {
    this._home = { ...this._home, pos: v3(pos).toArray(), target: v3(target).toArray() };
  }

  home(dur = 1.1) {
    return this.flyTo(this._home.pos, this._home.target, dur);
  }

  /** Glide the camera on an arc around the target. Resolves true when it arrives, false if interrupted. */
  flyTo(pos, target, dur = 1.3, easing = ease.inOut) {
    this._fly?.cancel();
    const cam = this.camera;
    const ctl = this.controls;
    const t0 = ctl.target.clone();
    const t1 = v3(target ?? ctl.target);
    const s0 = new THREE.Spherical().setFromVector3(cam.position.clone().sub(t0));
    const s1 = new THREE.Spherical().setFromVector3(v3(pos).sub(t1));
    let dTheta = s1.theta - s0.theta;
    if (dTheta > Math.PI) dTheta -= Math.PI * 2;
    if (dTheta < -Math.PI) dTheta += Math.PI * 2;
    const s = new THREE.Spherical();
    const off = new THREE.Vector3();
    if (this.reduceMotion) dur = Math.min(dur, 0.01);
    this._fly = this.tweens.add({
      dur,
      ease: easing,
      update: (k) => {
        ctl.target.lerpVectors(t0, t1, k);
        s.radius = s0.radius + (s1.radius - s0.radius) * k;
        s.phi = s0.phi + (s1.phi - s0.phi) * k;
        s.theta = s0.theta + dTheta * k;
        cam.position.copy(ctl.target).add(off.setFromSpherical(s));
      },
    });
    return this._fly.promise;
  }

  toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen?.();
    else document.documentElement.requestFullscreen?.().catch(() => {});
  }

  _onKey(e) {
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target?.isContentEditable) {
      if (tag !== 'INPUT' || !['range', 'checkbox'].includes(e.target.type)) return;
      if (e.key.startsWith('Arrow')) return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const handler = this.current?.keys.get(e.key) || this.current?.keys.get(e.key.toLowerCase());
    if (handler) {
      e.preventDefault();
      handler(e);
      return;
    }
    const k = e.key.toLowerCase();
    if (k === 'h') this.home();
    else if (k === 'f') this.toggleFullscreen();
    else if (k === 'p') this.hud.togglePanels();
    else if (k === '?') this.hud.toggleHelp();
    else if (k === 'escape') this.hud.toggleHelp(false);
  }

  // ------------------------------------------------------------------ chapters
  _makeCtx(def, route) {
    const app = this;
    const root = new THREE.Group();
    root.name = 'chapter:' + def.path;
    this.scene.add(root);
    const disposers = [];
    const keys = new Map();
    const guard = (fn) => (...a) => (ctx.dead ? undefined : fn(...a));
    const ctx = {
      app,
      root,
      route,
      router: this.router,
      tweens: new TweenGroup(),
      quality: this.quality,
      reduceMotion: this.reduceMotion,
      isTouch: this.isTouch,
      dead: false,
      ui: {
        left: guard((...n) => app.hud.left(...n)),
        right: guard((...n) => app.hud.right(...n)),
        dock: guard((...n) => app.hud.dock(...n)),
        toast: guard((m, ms) => app.hud.toast(m, ms)),
        loading: guard((label) => app.hud.loading(label)),
      },
      pick: {
        add: (objs, handlers) => (ctx.dead ? () => {} : app.picker.add(objs, handlers)),
        remove: (o) => app.picker.remove(o),
        onMiss: (fn) => (app.picker.onMiss = fn),
      },
      /** Run fn when the chapter is left. */
      onDispose: (fn) => disposers.push(fn),
      /** addEventListener that is removed automatically. */
      listen(target, type, fn, opts) {
        target.addEventListener(type, fn, opts);
        disposers.push(() => target.removeEventListener(type, fn, opts));
      },
      after(ms, fn) {
        const id = setTimeout(() => !ctx.dead && fn(), ms);
        disposers.push(() => clearTimeout(id));
        return id;
      },
      every(ms, fn) {
        const id = setInterval(() => !ctx.dead && fn(), ms);
        disposers.push(() => clearInterval(id));
        return id;
      },
      /** Keyboard shortcut while this chapter is active, e.g. ctx.key(' ', togglePlay). */
      key: (k, fn) => keys.set(k, fn),
      /** Fetch a file from /public (models, data) relative to the site root. */
      asset: (path) => new URL(path.replace(/^\//, ''), document.baseURI).href,
    };
    return { ctx, disposers, keys };
  }

  _teardown() {
    const cur = this.current;
    if (!cur) return;
    this.current = null;
    cur.ctx.dead = true;
    try {
      cur.ctrl.dispose?.();
    } catch (err) {
      console.error('[chapter dispose]', err);
    }
    for (const fn of cur.disposers) {
      try {
        fn();
      } catch (err) {
        console.error(err);
      }
    }
    cur.ctx.tweens.clear();
    this.picker.clear();
    cur.ctx.root.traverse((o) => o.isCSS2DObject && o.element.remove());
    this.scene.remove(cur.ctx.root);
    disposeDeep(cur.ctx.root);
    flushLabelCache();
    this.hud.clear();
    this.setBloom();
    this.setFog();
    this._fly?.cancel();
  }

  async _navigate(route) {
    const m = matchRoute(route.parts);
    if (!m) {
      this.router.replace('');
      return this._navigate(this.router.current());
    }
    const r = { ...route, rest: m.rest };
    if (this.current && this.current.def === m.def && !this._navigating) {
      this.current.ctx.route = r;
      this.current.ctrl.onRoute?.(r);
      return;
    }
    const token = ++this._navToken;
    this._navigating = true;
    await this.hud.fade(true, this.current ? m.def.title : '');
    if (token !== this._navToken) return;
    this._teardown();
    const made = this._makeCtx(m.def, r);
    const cur = { def: m.def, ctrl: {}, ...made };
    this.current = cur;
    this.hud.setActive(m.def.nav, m.def.accent);
    this.background.setTint(m.def.accent);
    this.setView(DEFAULT_VIEW);
    document.title = m.def.path ? `${m.def.title} · AI Atlas 3D` : 'AI Atlas 3D · how AI works, in an immersive 3D lab';
    try {
      const mod = await m.def.load();
      if (token !== this._navToken) return;
      const ctrl = await mod.enter(cur.ctx);
      if (token !== this._navToken) return;
      cur.ctrl = ctrl || {};
    } catch (err) {
      console.error('[chapter enter]', err);
      if (token !== this._navToken) return;
      this._showError(cur.ctx, m.def, err);
    }
    this._navigating = false;
    // the URL may have moved on (same chapter, different sub-route) while we were loading
    const latest = this.router.current();
    const lm = matchRoute(latest.parts);
    if (lm && lm.def === m.def && (latest.path !== route.path || JSON.stringify(latest.query) !== JSON.stringify(route.query))) {
      cur.ctx.route = { ...latest, rest: lm.rest };
      cur.ctrl.onRoute?.(cur.ctx.route);
    }
    await this.hud.fade(false);
  }

  _showError(ctx, def, err) {
    ctx.ui.left(
      h('header.ph', h('div.kicker', 'Something went wrong'), h('h1', def.title)),
      h('div.callout.warn', h('strong', 'This chapter failed to load.'), h('p', String(err?.message || err))),
      button('Back to the universe', { kind: 'primary', onClick: () => this.router.go('') }),
    );
  }

  /** Path of the chapter after the given nav id (for "next chapter" buttons). */
  nextNav(navId) {
    const i = NAV.findIndex((n) => n.id === navId);
    return NAV[(i + 1) % NAV.length];
  }
}
