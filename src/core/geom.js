// Small helpers for building scenes: materials that glow, lines, tubes, arrows, instancing, floors.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { glow } from './colors.js';

export const v3 = (a, b, c) =>
  a instanceof THREE.Vector3 ? a.clone() : Array.isArray(a) ? new THREE.Vector3(a[0], a[1], a[2]) : new THREE.Vector3(a, b, c);

/** Dispose every geometry / material / texture below `root` (label textures are flushed separately). */
export function disposeDeep(root) {
  root.traverse((o) => {
    o.geometry?.dispose?.();
    const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    for (const m of mats) {
      for (const key of Object.keys(m)) {
        const val = m[key];
        if (val && val.isTexture && !val.userData?.cachedLabel && !val.userData?.shared) val.dispose();
      }
      m.dispose?.();
    }
    if (o.isInstancedMesh) o.dispose?.();
  });
}

/** Unlit HDR material: k > 1 glows through the bloom pass. */
export function glowMat(color, o = {}) {
  const opacity = o.opacity ?? 1;
  return new THREE.MeshBasicMaterial({
    color: glow(color, o.k ?? 2.2),
    transparent: opacity < 1 || !!o.transparent,
    opacity,
    side: o.side ?? THREE.FrontSide,
    wireframe: !!o.wireframe,
    depthWrite: o.depthWrite ?? opacity >= 1,
    blending: o.additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    fog: o.fog ?? true,
  });
}

/** Lit material for solid things. `emissive` + `ek` adds an inner glow. */
export function stdMat(color, o = {}) {
  const opacity = o.opacity ?? 1;
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(color),
    roughness: o.rough ?? 0.5,
    metalness: o.metal ?? 0.15,
    emissive: o.emissive ? glow(o.emissive, o.ek ?? 1) : new THREE.Color(0x000000),
    transparent: opacity < 1,
    opacity,
    depthWrite: o.depthWrite ?? opacity >= 1,
    flatShading: !!o.flat,
    side: o.side ?? THREE.FrontSide,
  });
}

/** Dark translucent "holo glass" for panels, slabs and containers. */
export function glassMat(color = '#16233f', opacity = 0.35, o = {}) {
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(color),
    roughness: 0.25,
    metalness: 0.3,
    transparent: true,
    opacity,
    depthWrite: false,
    side: o.side ?? THREE.DoubleSide,
    emissive: o.emissive ? glow(o.emissive, o.ek ?? 0.4) : new THREE.Color(0x000000),
  });
}

/** Polyline through points (arrays or Vector3s). */
export function line(points, color, o = {}) {
  const geo = new THREE.BufferGeometry().setFromPoints(points.map((p) => v3(p)));
  const mat = new THREE.LineBasicMaterial({
    color: glow(color, o.k ?? 1.6),
    transparent: (o.opacity ?? 1) < 1,
    opacity: o.opacity ?? 1,
    depthWrite: false,
  });
  return o.closed ? new THREE.LineLoop(geo, mat) : new THREE.Line(geo, mat);
}

/** Many independent segments: `pairs` is a flat list [a0, b0, a1, b1, ...] of points. */
export function segments(pairs, color, o = {}) {
  const geo = new THREE.BufferGeometry().setFromPoints(pairs.map((p) => v3(p)));
  const mat = new THREE.LineBasicMaterial({
    color: glow(color, o.k ?? 1.4),
    transparent: (o.opacity ?? 1) < 1,
    opacity: o.opacity ?? 1,
    depthWrite: false,
  });
  return new THREE.LineSegments(geo, mat);
}

/** Wireframe outline of a geometry's hard edges. */
export function edges(geometry, color, o = {}) {
  const mat = new THREE.LineBasicMaterial({
    color: glow(color, o.k ?? 1.8),
    transparent: (o.opacity ?? 1) < 1,
    opacity: o.opacity ?? 1,
    depthWrite: false,
  });
  return new THREE.LineSegments(new THREE.EdgesGeometry(geometry, o.angle ?? 30), mat);
}

/** Quadratic arc from a to b, bulging by `lift` along `up`. */
export function arc(a, b, lift = 1, up = new THREE.Vector3(0, 1, 0)) {
  const A = v3(a);
  const B = v3(b);
  const mid = A.clone().add(B).multiplyScalar(0.5).addScaledVector(up, lift);
  return new THREE.QuadraticBezierCurve3(A, mid, B);
}

/** Tube along a curve or through points. */
export function tube(pathOrPoints, radius, color, o = {}) {
  const path = Array.isArray(pathOrPoints)
    ? new THREE.CatmullRomCurve3(pathOrPoints.map((p) => v3(p)), false, 'catmullrom', o.tension ?? 0.4)
    : pathOrPoints;
  const geo = new THREE.TubeGeometry(path, o.segments ?? 48, radius, o.radial ?? 8, false);
  const mesh = new THREE.Mesh(geo, o.material ?? glowMat(color, { k: o.k ?? 1.6, opacity: o.opacity ?? 1 }));
  mesh.userData.path = path;
  return mesh;
}

const _up = new THREE.Vector3(0, 1, 0);
const _size = new THREE.Vector2();
/** Straight arrow (shaft + cone head) from a to b. */
export function arrow(a, b, color, o = {}) {
  const A = v3(a);
  const B = v3(b);
  const dir = B.clone().sub(A);
  const len = dir.length();
  const head = Math.min(o.head ?? 0.28, len * 0.6);
  const r = o.radius ?? 0.035;
  const mat = o.material ?? glowMat(color, { k: o.k ?? 1.8, opacity: o.opacity ?? 1 });
  const g = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len - head, 8), mat);
  shaft.position.y = (len - head) / 2;
  const cone = new THREE.Mesh(new THREE.ConeGeometry(r * 3.2, head, 12), mat);
  cone.position.y = len - head / 2;
  g.add(shaft, cone);
  g.position.copy(A);
  g.quaternion.setFromUnitVectors(_up, dir.normalize());
  return g;
}

export function rbox(w, h, d, r = 0.08, seg = 3) {
  return new RoundedBoxGeometry(w, h, d, seg, Math.min(r, w / 2, h / 2, d / 2));
}

/** Circle outline in the XZ plane. */
export function ring(radius, color, o = {}) {
  const n = o.segments ?? 128;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius));
  }
  return line(pts, color, { ...o, closed: true });
}

let _dotTex = null;
/** Shared soft radial-gradient texture (white core fading to transparent). */
export function dotTexture() {
  if (_dotTex) return _dotTex;
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(0.6, 'rgba(255,255,255,0.12)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  _dotTex = new THREE.CanvasTexture(c);
  _dotTex.userData.shared = true;
  return _dotTex;
}

/** Soft additive glow billboard (a cheap light halo). */
export function halo(color, size = 1, opacity = 0.6, k = 1.5) {
  const mat = new THREE.SpriteMaterial({
    map: dotTexture(),
    color: glow(color, k),
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
  });
  const s = new THREE.Sprite(mat);
  s.scale.setScalar(size);
  return s;
}

/**
 * Point cloud with per-point colour and size, round soft points.
 * positions: Float32Array(n*3); colors: Float32Array(n*3) (linear, may be HDR); sizes: Float32Array(n) in world units.
 */
export function points(positions, colors, sizes, o = {}) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  const mat = new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 1 }, uOpacity: { value: o.opacity ?? 1 } },
    vertexShader: /* glsl */ `
      attribute float size;
      attribute vec3 color;
      varying vec3 vColor;
      uniform float uScale;
      void main() {
        vColor = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = max(1.0, size * uScale / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      uniform float uOpacity;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        float a = smoothstep(1.0, ${o.soft === false ? '0.85' : '0.15'}, d);
        if (a < 0.01) discard;
        gl_FragColor = vec4(vColor, a * uOpacity);
      }`,
    transparent: true,
    depthWrite: false,
    blending: o.additive === false ? THREE.NormalBlending : THREE.AdditiveBlending,
  });
  const pts = new THREE.Points(geo, mat);
  pts.frustumCulled = false;
  // uScale converts a world-space size to pixels for the camera that is rendering
  pts.onBeforeRender = (renderer, _scene, camera) => {
    mat.uniforms.uScale.value = renderer.getDrawingBufferSize(_size).y / (2 * Math.tan((camera.fov * Math.PI) / 360));
  };
  return pts;
}

/** InstancedMesh with a per-instance colour buffer ready to use. */
export function instanced(geometry, material, count) {
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const white = new THREE.Color(1, 1, 1);
  for (let i = 0; i < count; i++) mesh.setColorAt(i, white);
  mesh.frustumCulled = false;
  return mesh;
}

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
/** Write position / scale (and optionally colour) of instance i. Call `flushInst(mesh)` when done. */
export function setInst(mesh, i, x, y, z, sx = 1, sy = sx, sz = sx, color) {
  _m.compose(_p.set(x, y, z), _q.identity(), _s.set(sx, sy, sz));
  mesh.setMatrixAt(i, _m);
  if (color) mesh.setColorAt(i, color);
}
export function flushInst(mesh) {
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

/** Infinite-looking holographic grid floor that fades out with distance from its centre. */
export function gridFloor(o = {}) {
  const size = o.size ?? 80;
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: glow(o.color ?? '#3a5a9a', o.k ?? 1.0) },
      uStep: { value: o.step ?? 1 },
      uFade: { value: o.fade ?? size * 0.45 },
      uOpacity: { value: o.opacity ?? 0.5 },
    },
    vertexShader: /* glsl */ `
      varying vec3 vPos;
      void main() {
        vPos = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      varying vec3 vPos;
      uniform vec3 uColor;
      uniform float uStep;
      uniform float uFade;
      uniform float uOpacity;
      float grid(vec2 p, float s) {
        vec2 g = abs(fract(p / s - 0.5) - 0.5) / fwidth(p / s);
        return 1.0 - min(min(g.x, g.y), 1.0);
      }
      void main() {
        vec2 p = vPos.xy;
        float a = grid(p, uStep) * 0.55 + grid(p, uStep * 5.0) * 0.6;
        float fade = 1.0 - smoothstep(uFade * 0.25, uFade, length(p));
        gl_FragColor = vec4(uColor, a * fade * uOpacity);
        if (gl_FragColor.a < 0.004) discard;
      }`,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = o.y ?? 0;
  mesh.renderOrder = -5;
  return mesh;
}

/** Deterministic pseudo-random generator (mulberry32). `const rnd = rng(42); rnd()` -> [0,1). */
export function rng(seed = 1) {
  let a = seed >>> 0;
  const f = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  f.range = (lo, hi) => lo + (hi - lo) * f();
  f.int = (n) => Math.floor(f() * n);
  f.normal = () => {
    let u = 0;
    while (u === 0) u = f();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * f());
  };
  f.pick = (arr) => arr[Math.floor(f() * arr.length)];
  return f;
}
