# Architecture

AI Atlas 3D is a static single-page app: **one WebGL canvas, one camera, one render loop**, and a
set of lazily loaded *chapters* that are mounted into the scene one at a time. A glass HUD (HTML)
floats above the canvas. There is no UI framework; three.js is the only runtime dependency.

```
index.html
src/main.js            boot: fonts -> App -> first route
src/routes.js          route table + top navigation (do not add routes without updating NAV/ROUTES)
src/core/              the engine (shared by every chapter)
  app.js               renderer, camera, bloom, picking, transitions, chapter lifecycle
  hud.js               HUD shell: top bar, left/right panels, dock, tooltip, toasts
  ui.js                DOM components for panels (h, header, section, slider, segmented, tour ...)
  charts.js            2D charts for the HUD (lineChart, barChart, heatmap, radar, sparkline)
  colors.js            palettes: PAL (scene accents), SERIES (data), STATUS, glow(), seq(), div()
  geom.js              3D helpers: glowMat, stdMat, glassMat, line, segments, edges, tube, arrow,
                       arc, rbox, ring, halo, points, instanced/setInst/flushInst, gridFloor, rng
  labels.js            textSprite, textPlane (canvas text in 3D), htmlLabel (DOM pinned to 3D)
  gallery.js           pods + galleryChapter (hub portals, lab / experiment galleries)
  icons3d.js           procedural 3D glyphs used on pods
  tween.js             TweenGroup, ease, damp, clamp, lerp
src/chapters/          one module per top-level chapter
src/labs/              one module per mechanism lab
src/perf/              one module per performance experiment
src/ml/                the actual algorithms (pure JS, no three.js, testable in Node)
src/data/              content: hand-written, sourced data files
public/models, public/data   binary / JSON assets fetched at run time
tools/                 training scripts (Python) and the headless screenshot tool
```

## The chapter contract

A route points at a module that exports one function:

```js
export async function enter(ctx) {
  // build 3D objects into ctx.root, fill the HUD through ctx.ui
  return {
    update(dt, t) {},      // every frame; dt seconds (clamped to 0.05), t = seconds since start
    onRoute(route) {},     // the URL changed but still maps to this module (route.rest, route.query)
    dispose() {},          // optional extra clean-up
  };
}
```

When the user leaves, the engine removes `ctx.root`, disposes every geometry, material and texture
under it, cancels the chapter's tweens, timers, listeners, key bindings and pick targets, and
clears the HUD. A chapter therefore never needs to clean up what it created through `ctx`.

`ctx` gives you:

| member | purpose |
| --- | --- |
| `ctx.root` | `THREE.Group` that holds everything the chapter draws |
| `ctx.app` | the engine: `setView`, `flyTo`, `home`, `setBloom`, `setFog`, `camera`, `controls`, `renderer` |
| `ctx.ui.left(...nodes)`, `.right(...)`, `.dock(...)` | fill the HUD (call again to replace; no args hides) |
| `ctx.ui.toast(msg)`, `ctx.ui.loading(label)` | notifications |
| `ctx.pick.add(obj \| obj[], {hover, out, click, tip, cursor})` | hover / click on meshes, instanced meshes, sprites. `tip` may be a string or `(hit) => string \| Node`. `hit.instanceId` identifies an instance. `ctx.pick.onMiss(fn)` fires when empty space is clicked |
| `ctx.tweens` | `TweenGroup`: `add({dur, ease, update(k)})`, `to(obj, {x: 1}, {dur})`, `wait(s)`; all return `{promise, cancel}` |
| `ctx.router.go(path, query)`, `.replace(path, query)` | navigate / update the URL silently |
| `ctx.route` | `{path, parts, rest, query}`; `rest` is what follows the module's own path |
| `ctx.key(key, fn)` | keyboard shortcut while the chapter is active |
| `ctx.listen(target, type, fn)`, `ctx.after(ms, fn)`, `ctx.every(ms, fn)`, `ctx.onDispose(fn)` | auto-cleaned side effects |
| `ctx.asset(path)` | absolute URL of a file in `public/` (works under any base path) |
| `ctx.quality`, `ctx.reduceMotion`, `ctx.isTouch` | adapt detail and motion |

### Camera

Call `ctx.app.setView({...})` once in `enter`. Prefer `fit`:

```js
app.setView({ pos: [7, 8, 11], target: [0, 1.2, 0], fit: 8.4, min: 5, max: 60, maxPolar: Math.PI * 0.49 });
```

The camera is placed on the ray from `target` through `pos` (so `pos` is still a position, not a
direction vector); `fit` is the radius of the sphere around `target` that must stay
visible; the engine picks the distance so it fits **between the panels** on any screen and keeps
the scene centered in that free area. Inside a chapter use `app.flyTo(pos, target, seconds)`, or
`app.frame({ target, dir, fit }, seconds)` to fly to a framed view of part of the scene.

## Look and feel rules

* **Glow is HDR.** The bloom pass only reacts to color channels above 1.0. `glow(color, k)` and
  `glowMat(color, {k})` multiply a color: `k ≈ 1` is a flat color, `k ≈ 2–3` is neon. Lit
  materials (`stdMat`) and text stay below 1 and therefore stay crisp. Do not make large surfaces
  glow; reserve it for edges, particles, active elements.
* **Color by job.** `SERIES` (8 fixed, color-blind-validated colors) identifies data series and
  the 3D marks that correspond to them: the same entity has the same color in the scene, the
  legend and the chart, and keeps it when others are filtered out. `seq(t)` / `seqCss(t)` is the
  single-hue ramp for magnitudes, `div(v)` / `divCss(v)` the blue-gray-red ramp for signed values.
  `STATUS` colors mean good / warning / critical only. `PAL` accents are decorative.
* **Text never wears the data color**: put a colored swatch next to neutral text.
* Charts: a legend for two or more series, thin marks, hairline grid, one y-axis, hover read-out.
  The chart helpers already do this. Do not put a number on every mark.
* Use instancing (`instanced`, `setInst`, `flushInst`) or `points` for more than ~150 similar
  objects. Keep text sprites under ~150 and DOM labels under ~60 at a time.
* Every interaction must also work without hover (touch): put the same information in a panel.
* Honesty: anything shown as a measurement is computed live by code in `src/ml/`; anything quoted
  from the outside world sits in `src/data/` with a source URL and a date; qualitative ratings are
  labeled as editorial.

## Panels

Left panel = explanation (kicker, title, lead, then a guided `tour` or reference cards).
Right panel = controls and live read-outs. Dock = the tour stepper or a short hint.
Chapter-specific CSS goes in its own file imported by the module (`import './x.css'`); shared
tokens live in `src/style.css` (`--accent`, `--ink`, `--ink-2`, `--ink-3`, `--line`, `--raise`).

`tour(ctx, steps, opts)` renders the stepper, binds ← / →, mirrors the step to `?s=N`, and calls
each step's `enter()` so the scene can reconfigure itself. See `src/labs/optim.js`, the reference
implementation of a lab: math in `src/ml/optim.js`, scene + controls in the lab file.

## Shared scene kits

Three modules carry most of the scenery, so that chapters stay about their content:

* `src/core/gallery.js`: pods on a wall or a ring, for chapters that are a menu of experiences
  (Labs, Performance).
* `src/core/flowgraph.js`: a batched 3D node-link diagram with typed nodes, directed edges, moving
  packets, floor zones and "exhibits" (token chips, tensors, images, JSON cards). The Data Flow and
  Deploy chapters are data (`src/data/pipelines.js`, `src/data/deployments.js`) drawn by this kit.
* `src/chapters/hot-vignettes.js`: one procedural scene per hot issue. A vignette is
  `build({ ctx, issue, low }) -> { group, panel, update(dt) }`; it stands on a floor at y = 0,
  faces +z, and reads every figure it displays out of the issue's dated facts with `figure()`.

## Testing

```
npm run dev                                   # http://127.0.0.1:5173
node tools/shot.mjs '#/labs/optim?s=4' shots/optim.png --q low --wait 3000
node tools/shot.mjs '#/hot' shots/hot.png --eval 'document.querySelector(".btn.primary").click()'
npm run smoke                                 # every route, in one headless session
npm run test:ml                               # numerical checks of src/ml against reference values
```

`tools/shot.mjs` drives headless Chromium (software WebGL), prints console errors, exits non-zero
if the page threw, and saves a screenshot. `--q low|medium|high` selects the rendering quality,
`--size 390x844` a viewport, `--eval` runs a snippet before the screenshot (`window.__atlas` is the
App instance). Software rendering is slow, so animations barely advance while it waits; to see a
later state, step the chapter yourself:
`--eval '(async()=>{const c=__atlas.current.ctrl;for(let i=0;i<200;i++){c.update(0.05,0);await null}})()'`.

`tools/smoke.mjs` visits every route and deep link (about 80), steps each chapter through twenty
seconds of animation and fails if anything logged an error. Both tools need Playwright, which is
not a dependency of the site; `--base http://127.0.0.1:4173/` points them at `npm run preview`.

## Regenerating the models

The tiny models under `public/models` are produced by the scripts in `tools/train` (PyTorch, CPU).
See `tools/train/README.md`.
