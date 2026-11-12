// "Pods": a floating 3D glyph on a glowing ring with a caption. Used for the hub portals and for
// the lab / experiment galleries. A pod is hoverable and clickable through an invisible hit proxy.
import * as THREE from 'three';
import { glowMat, glassMat, halo } from './geom.js';
import { glow } from './colors.js';
import { textSprite } from './labels.js';
import { icon } from './icons3d.js';
import { damp } from './tween.js';
import { h, header, section } from './ui.js';

/**
 * @param {object} ctx chapter context
 * @param {{title:string, sub?:string, kicker?:string, color:string, icon:string, tip?:string, onClick?:()=>void, scale?:number}} o
 * @returns {{group: THREE.Group, update: (dt:number, t:number)=>void, item: object}}
 */
export function makePod(ctx, o) {
  const s = o.scale ?? 1;
  const group = new THREE.Group();
  const inner = new THREE.Group(); // scales on hover
  group.add(inner);

  const glyph = icon(o.icon, o.color);
  glyph.scale.setScalar(s);
  glyph.position.y = 0.25 * s;
  inner.add(glyph);

  const ringMat = glowMat(o.color, { k: 1.6 });
  const ringObj = new THREE.Mesh(new THREE.TorusGeometry(1.08 * s, 0.018 * s, 8, 72), ringMat);
  ringObj.rotation.x = Math.PI / 2;
  ringObj.position.y = -0.95 * s;
  const disc = new THREE.Mesh(new THREE.CircleGeometry(1.06 * s, 48), glassMat(o.color, 0.1));
  disc.rotation.x = -Math.PI / 2;
  disc.position.y = -0.96 * s;
  const glowSprite = halo(o.color, 4.2 * s, 0.16);
  glowSprite.position.y = -0.2 * s;
  inner.add(ringObj, disc, glowSprite);

  const title = textSprite(o.title, { size: 0.36 * s, font: 'display', weight: 700, color: '#eef3ff' });
  title.position.y = -1.8 * s;
  group.add(title);
  if (o.sub) {
    const sub = textSprite(o.sub, { size: 0.215 * s, color: '#a3b0cc', weight: 500, maxChars: 30 });
    sub.center.set(0.5, 1);
    sub.position.y = -2.06 * s;
    group.add(sub);
  }
  if (o.kicker) {
    const k = textSprite(o.kicker, { size: 0.17 * s, font: 'mono', weight: 600, color: '#' + new THREE.Color(o.color).getHexString() });
    k.position.y = 1.32 * s;
    group.add(k);
  }

  const hit = new THREE.Mesh(new THREE.SphereGeometry(1.25 * s, 12, 8), new THREE.MeshBasicMaterial({ visible: false }));
  hit.position.y = 0;
  group.add(hit);
  let hovered = false;
  let k = 0;
  ctx.pick.add(hit, {
    hover: () => (hovered = true),
    out: () => (hovered = false),
    click: () => o.onClick?.(),
    tip: o.tip,
  });

  const base = glow(o.color, 1);
  const phase = Math.random() * 6.28;
  return {
    group,
    item: o,
    update(dt, t) {
      k = damp(k, hovered ? 1 : 0, 0.12, dt);
      inner.scale.setScalar(1 + k * 0.14);
      inner.position.y = Math.sin(t * 0.9 + phase) * 0.08 * s;
      glyph.rotation.y += dt * (0.35 + k * 1.2);
      ringMat.color.copy(base).multiplyScalar(1.5 + k * 2.2);
      glowSprite.material.opacity = 0.14 + k * 0.3;
      glyph.userData.tick?.(t, dt);
    },
  };
}

/**
 * Place pods on a gently curved wall facing +Z, `perRow` to a row. Returns the wall's size so the
 * caller can frame it.
 */
export function layoutWall(pods, { perRow = 4, spacing = 4.8, rowGap = 5.7, curve = 0.014 } = {}) {
  const rows = Math.ceil(pods.length / perRow);
  pods.forEach((p, i) => {
    const row = Math.floor(i / perRow);
    const col = i % perRow;
    const n = Math.min(perRow, pods.length - row * perRow);
    const x = (col - (n - 1) / 2) * spacing;
    p.group.position.set(x, ((rows - 1) / 2 - row) * rowGap, x * x * curve);
  });
  return { width: (perRow - 1) * spacing + 3.6, height: (rows - 1) * rowGap + 5.4 };
}

/** Place pods evenly on a circle in the XZ plane. */
export function layoutRing(pods, { radius = 11, y = 0, start = 0 } = {}) {
  pods.forEach((p, i) => {
    const a = start + (i / pods.length) * Math.PI * 2;
    p.group.position.set(Math.sin(a) * radius, y, Math.cos(a) * radius);
  });
}

/**
 * A whole "gallery" chapter: pods on an arc plus a matching card list in the left panel.
 * items: [{ path, title, sub, blurb, icon, color, tag? }]
 */
export function galleryChapter(ctx, { kicker, title, lead, items, footer, rows = 1 }) {
  const { app, root } = ctx;
  const pods = items.map((it, i) =>
    makePod(ctx, {
      title: it.title,
      sub: it.sub,
      kicker: it.tag ?? String(i + 1).padStart(2, '0'),
      color: it.color,
      icon: it.icon,
      tip: it.blurb,
      scale: 1.25,
      onClick: () => ctx.router.go(it.path),
    }),
  );
  // frame the wall inside the part of the viewport the left panel leaves free
  const aspect = window.innerWidth / window.innerHeight;
  const perRow = aspect < 0.8 ? 2 : aspect < 1.25 ? 3 : Math.ceil(items.length / rows);
  const { width, height } = layoutWall(pods, { perRow });
  pods.forEach((p) => root.add(p.group));
  const usable = window.innerWidth > 860 ? (window.innerWidth - 420) / window.innerHeight : aspect;
  const half = Math.tan((((aspect < 0.8 ? 1.35 : 1) * 50) * Math.PI) / 360);
  const dist = Math.max(width / 2 / (half * usable), (height / 2 / half) * (window.innerWidth > 860 ? 1.12 : 1.9)) * 1.04 + 1;
  app.setView({ pos: [0, 1.4, dist], target: [0, -0.5, 0], min: 6, max: dist * 2.2 });
  ctx.ui.left(
    header({ kicker, title, lead }),
    section(
      null,
      items.map((it) =>
        h('a.card.click.chaplink', { href: '#/' + it.path, style: { '--accent': it.color } }, h('h4', h('span.n', it.tag ?? ''), it.title), h('p', it.blurb)),
      ),
    ),
    footer,
  );
  ctx.ui.dock(h('div.dockrow', h('span.hint', 'Click a pod to open it · drag to look around')));
  return { update: (dt, t) => pods.forEach((p) => p.update(dt, t)) };
}
