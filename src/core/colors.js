import * as THREE from 'three';

/** Scene / brand accents. Decorative identity colors for chapters and glows. */
export const PAL = {
  bg: '#05070d',
  ink: '#e6ecf8',
  ink2: '#b4bfd6',
  ink3: '#7f8aa5',
  cyan: '#5ad7ff',
  blue: '#4d8dff',
  violet: '#9d8cff',
  magenta: '#ff6fb1',
  amber: '#ffb84d',
  mint: '#4de3b0',
  coral: '#ff7a59',
  red: '#ff5d6c',
  lime: '#b6f36a',
  steel: '#8fa3c8',
  slate: '#3a4763',
};

/**
 * Categorical series colors for data (charts AND the matching 3D marks), in fixed order.
 * Validated against the HUD surface #0b101c for color-vision-deficiency separation and contrast.
 * Never cycle past 8: fold extra series into "other" or use small multiples.
 */
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];

/** Reserved status colors: only for good / warning / serious / critical states (always with a label). */
export const STATUS = { good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b' };

const _a = new THREE.Color();
const _b = new THREE.Color();

/**
 * HDR color for anything that should glow. The bloom pass only picks up channels above 1.0,
 * so `glow(c, 1)` is a plain color and `glow(c, 2.5)` is a neon tube.
 */
export function glow(color, k = 2.2) {
  return new THREE.Color(color).multiplyScalar(k);
}

/** Linear mix of two colors (in linear working space). Returns a new THREE.Color. */
export function mix(c1, c2, t) {
  return new THREE.Color(c1).lerp(_b.set(c2), t);
}

const SEQ_STOPS = ['#0a1830', '#0d366b', '#1c5cab', '#3987e5', '#86b6ef', '#cde2fb'].map((c) => new THREE.Color(c));
const DIV_NEG = ['#383835', '#1c5cab', '#3987e5', '#9ec5f4'].map((c) => new THREE.Color(c));
const DIV_POS = ['#383835', '#a3312f', '#e66767', '#f6b3b3'].map((c) => new THREE.Color(c));

function ramp(stops, t, out) {
  const x = Math.min(0.9999, Math.max(0, t)) * (stops.length - 1);
  const i = Math.floor(x);
  return out.copy(stops[i]).lerp(stops[i + 1], x - i);
}

/** Sequential single-hue ramp for magnitude on a dark surface: 0 = recedes into the dark, 1 = bright. */
export function seq(t, out = new THREE.Color()) {
  return ramp(SEQ_STOPS, t, out);
}

/** Diverging ramp for signed values, v in [-1, 1]: blue (negative) - neutral gray - red (positive). */
export function div(v, out = new THREE.Color()) {
  return v < 0 ? ramp(DIV_NEG, -v, out) : ramp(DIV_POS, v, out);
}

/** CSS string versions for 2D canvases / DOM. */
export function seqCss(t) {
  return '#' + seq(t, _a).getHexString();
}
export function divCss(v) {
  return '#' + div(v, _a).getHexString();
}

/** sRGB css color with alpha from a css hex color (no color-space conversion: for DOM/canvas use). */
export function withAlpha(hex, a) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
