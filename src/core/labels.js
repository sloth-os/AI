// Text in the 3D world: billboard sprites and flat planes rasterized to canvas textures,
// plus DOM labels (CSS2D) for the few places where crisp, clickable HTML is worth it.
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

export const FONTS = {
  display: '"Space Grotesk Variable", "Inter Variable", system-ui, sans-serif',
  body: '"Inter Variable", system-ui, -apple-system, "Segoe UI", sans-serif',
  mono: '"JetBrains Mono Variable", ui-monospace, "SF Mono", Menlo, Consolas, monospace',
};

const PX = 56; // rasterization size of one em
const cache = new Map();

const DEFAULTS = {
  size: 0.4, // world height of one em (for attenuate:false: fraction of the viewport, ~0.03 is body text)
  color: '#dfe7f7',
  font: 'body',
  weight: 600,
  align: 'center',
  anchor: [0.5, 0.5], // sprite pivot: [0,0.5] = left-middle, [0.5,0] = bottom-center ...
  bg: null, // css color for a rounded backing plate
  border: null, // css color for the plate outline
  radius: 0.32, // plate corner radius in em
  attenuate: true, // false = constant size on screen
  onTop: false, // true = ignore depth (always visible)
  opacity: 1,
  k: 1, // brightness multiplier; > 1.15 or so starts to bloom
  lineHeight: 1.22,
  maxChars: 0, // wrap long text at about this many characters (0 = no wrapping)
};

function wrap(text, maxChars) {
  if (!maxChars) return String(text).split('\n');
  const out = [];
  for (const para of String(text).split('\n')) {
    let line = '';
    for (const word of para.split(' ')) {
      if (line && (line + ' ' + word).length > maxChars) {
        out.push(line);
        line = word;
      } else line = line ? line + ' ' + word : word;
    }
    out.push(line);
  }
  return out;
}

function rasterize(text, o) {
  const lines = wrap(text, o.maxChars);
  const font = `${o.weight} ${PX}px ${FONTS[o.font] || o.font}`;
  const canvas = document.createElement('canvas');
  const g = canvas.getContext('2d');
  g.font = font;
  const padX = o.bg || o.border ? PX * 0.46 : PX * 0.14;
  const padY = o.bg || o.border ? PX * 0.24 : PX * 0.1;
  const lineH = PX * o.lineHeight;
  const textW = Math.max(4, ...lines.map((l) => g.measureText(l).width));
  const w = Math.min(4096, Math.ceil(textW + padX * 2));
  const h = Math.min(4096, Math.ceil(lines.length * lineH + padY * 2));
  canvas.width = w;
  canvas.height = h;
  if (o.bg || o.border) {
    const r = Math.min(PX * o.radius, h / 2);
    g.beginPath();
    g.roundRect(2, 2, w - 4, h - 4, r);
    if (o.bg) {
      g.fillStyle = o.bg;
      g.fill();
    }
    if (o.border) {
      g.strokeStyle = o.border;
      g.lineWidth = 3;
      g.stroke();
    }
  }
  g.font = font;
  g.textBaseline = 'middle';
  g.textAlign = o.align;
  g.fillStyle = o.color;
  const x = o.align === 'left' ? padX : o.align === 'right' ? w - padX : w / 2;
  lines.forEach((l, i) => g.fillText(l, x, padY + lineH * (i + 0.5) + PX * 0.04));
  return { canvas, w, h };
}

function getTexture(text, o) {
  const key = [text, o.color, o.font, o.weight, o.align, o.bg, o.border, o.radius, o.lineHeight, o.maxChars].join('|');
  let hit = cache.get(key);
  if (!hit) {
    const { canvas, w, h } = rasterize(text, o);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    tex.userData.cachedLabel = true;
    hit = { tex, w, h };
    cache.set(key, hit);
  }
  return hit;
}

/** Called by the app when a chapter is torn down: label textures are only valid for one chapter. */
export function flushLabelCache() {
  for (const { tex } of cache.values()) tex.dispose();
  cache.clear();
}

/**
 * Billboard text. `sprite.userData.setText(str)` swaps the text later.
 * @param {string} text  may contain "\n"
 * @param {Partial<typeof DEFAULTS>} [opts]
 * @returns {THREE.Sprite}
 */
export function textSprite(text, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const mat = new THREE.SpriteMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: !o.onTop,
    sizeAttenuation: o.attenuate,
    opacity: o.opacity,
    fog: false,
  });
  mat.color.setScalar(o.k);
  const sprite = new THREE.Sprite(mat);
  sprite.center.set(o.anchor[0], o.anchor[1]);
  sprite.renderOrder = o.onTop ? 20 : 10;
  const apply = (str) => {
    const { tex, w, h } = getTexture(str, o);
    mat.map = tex;
    mat.needsUpdate = true;
    sprite.scale.set((o.size * w) / PX, (o.size * h) / PX, 1);
    sprite.userData.text = str;
  };
  apply(text);
  sprite.userData.setText = (str) => {
    if (str !== sprite.userData.text) apply(str);
  };
  return sprite;
}

/**
 * Text on a flat plane (does not turn to face the camera). Faces +Z; rotate it as needed.
 * @returns {THREE.Mesh}
 */
export function textPlane(text, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const { tex, w, h } = getTexture(text, o);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
    depthTest: !o.onTop,
    opacity: o.opacity,
    side: THREE.DoubleSide,
    fog: false,
  });
  mat.color.setScalar(o.k);
  const geo = new THREE.PlaneGeometry((o.size * w) / PX, (o.size * h) / PX);
  geo.translate(((0.5 - o.anchor[0]) * o.size * w) / PX, ((0.5 - o.anchor[1]) * o.size * h) / PX, 0);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = o.onTop ? 20 : 10;
  mesh.userData.text = text;
  return mesh;
}

/**
 * A DOM element pinned to a 3D position (always on top, never occluded). Use sparingly (< ~60 at once).
 * @param {string|HTMLElement} content
 */
export function htmlLabel(content, className = 'label3d') {
  let el = content;
  if (typeof content === 'string') {
    el = document.createElement('div');
    el.textContent = content;
  }
  el.classList.add(...className.split(' ').filter(Boolean));
  const obj = new CSS2DObject(el);
  obj.userData.el = el;
  return obj;
}
