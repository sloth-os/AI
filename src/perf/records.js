// Hall of Records: published reference numbers as a ring of 3D bar charts. Nothing here is
// measured on this page; every row comes from src/data/records.js with its source.
import * as THREE from 'three';
import './records.css';
import { RECORDS } from '../data/records.js';
import { AS_OF } from '../data/landscape.js';
import { PAL, SERIES, glow } from '../core/colors.js';
import { glassMat, edges, rbox, line, gridFloor, ring } from '../core/geom.js';
import { textSprite } from '../core/labels.js';
import { h, header, section, text, callout, button, legend, fmt } from '../core/ui.js';
import { damp } from '../core/tween.js';

const R = 25; // radius of the ring of exhibits (they face inward: the visitor stands inside)
const W = 9.4; // width of one chart
const H = 4.6; // height of the tallest bar

function fmtValue(rec, v) {
  if (rec.unit.startsWith('%')) return v.toFixed(v < 10 ? 2 : 1).replace(/\.?0+$/, '') + '%';
  if (rec.unit.startsWith('$')) return '$' + (v < 1 ? v.toFixed(2) : v.toFixed(v < 10 ? 2 : 0)).replace(/\.00$/, '');
  if (rec.unit === 'GB') return v >= 1000 ? (v / 1000).toFixed(1) + ' TB' : v.toFixed(v < 10 ? 1 : 0) + ' GB';
  if (rec.unit.startsWith('hours')) return v.toFixed(1) + ' h';
  return fmt.compact(v);
}

export async function enter(ctx) {
  const { app, root } = ctx;
  app.setView({ pos: [0, 40, 34], target: [0, 1.5, 0], fit: R + 7, min: 4, max: 140, maxPolar: Math.PI * 0.52 });
  root.add(gridFloor({ size: 140, y: -0.12, color: '#3a2f66', opacity: 0.32 }));
  const floorRing = ring(R, PAL.violet, { k: 1, opacity: 0.25, segments: 200 });
  root.add(floorRing);

  // ------------------------------------------------------------------ build the exhibits
  const exhibits = RECORDS.map((rec, i) => {
    const angle = (i / RECORDS.length) * Math.PI * 2;
    const g = new THREE.Group();
    g.position.set(Math.sin(angle) * R, 0, Math.cos(angle) * R);
    g.rotation.y = angle + Math.PI; // local +z points at the center of the ring
    root.add(g);

    const n = rec.rows.length;
    const step = Math.min(1.25, W / n);
    const vals = rec.rows.map((r) => r.value);
    const max = Math.max(...vals, rec.reference?.value ?? 0);
    const min = Math.min(...vals, rec.reference?.value ?? Infinity);
    const lo = rec.log ? Math.floor(Math.log10(min)) - 0.35 : 0;
    const hi = rec.log ? Math.log10(max) : max;
    const yOf = (v) => 0.1 + H * Math.max(0.01, ((rec.log ? Math.log10(v) : v) - lo) / (hi - lo));
    const groupKeys = rec.groups ? Object.keys(rec.groups) : null;
    const colorOf = (row) => (groupKeys ? SERIES[groupKeys.indexOf(String(row.group)) % SERIES.length] : SERIES[i % SERIES.length]);

    const plate = new THREE.Mesh(rbox(W + 1.4, 0.14, 2.6, 0.06), glassMat('#18233f', 0.5, { emissive: PAL.violet, ek: 0.12 }));
    plate.add(edges(plate.geometry, PAL.violet, { k: 1.1, opacity: 0.55 }));
    g.add(plate);

    const bars = rec.rows.map((row, k) => {
      const mat = new THREE.MeshBasicMaterial({ color: glow(colorOf(row), 1) });
      const geo = new THREE.BoxGeometry(step * 0.6, 1, step * 0.6 > 0.7 ? 0.7 : step * 0.6);
      geo.translate(0, 0.5, 0);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set((k - (n - 1) / 2) * step, 0.07, 0);
      mesh.scale.y = 0.02;
      g.add(mesh);
      return { mesh, mat, row, hgt: yOf(row.value), base: glow(colorOf(row), 1), boost: row.mark ? 1.9 : groupKeys ? 1.15 : 0.9 };
    });
    // logarithmic charts get one faint line per power of ten
    if (rec.log) {
      for (let e = Math.ceil(lo + 0.35); e <= hi; e++) {
        const y = yOf(Math.pow(10, e));
        g.add(line([[-W / 2 - 0.4, y, -0.5], [W / 2 + 0.4, y, -0.5]], '#8fa3c8', { k: 0.7, opacity: 0.22 }));
      }
    }
    if (rec.reference) {
      const y = yOf(rec.reference.value);
      g.add(line([[-W / 2 - 0.5, y, 0.5], [W / 2 + 0.5, y, 0.5]], '#ffe9a8', { k: 1.8, opacity: 0.9 }));
      const lab = textSprite(rec.reference.label, { size: 0.26, color: '#ffe9a8', weight: 600, anchor: [1, 0], bg: 'rgba(5,8,16,0.7)' });
      lab.position.set(W / 2 + 0.5, Math.max(y + 0.1, 1.5), 0.6);
      g.add(lab);
    }
    const title = textSprite(rec.short, { size: 0.62, font: 'display', weight: 700, color: '#eef3ff' });
    title.position.set(0, H + 1.55, 0);
    const num = textSprite(String(i + 1).padStart(2, '0'), { size: 0.3, font: 'mono', weight: 600, color: '#b6a8ff' });
    num.position.set(0, H + 2.25, 0);
    g.add(title, num);

    const labels = new THREE.Group(); // filled only while the exhibit is selected
    g.add(labels);
    const hit = new THREE.Mesh(new THREE.BoxGeometry(W + 1.6, H + 2.4, 2.8), new THREE.MeshBasicMaterial({ visible: false }));
    hit.position.y = (H + 2) / 2;
    g.add(hit);
    ctx.pick.add(hit, { click: () => select(i), tip: () => (active === i ? null : rec.title) });
    return { rec, g, bars, labels, step, n, angle, grow: 0 };
  });

  function buildLabels(ex) {
    const { rec, labels, step, n } = ex;
    const dense = n > 9;
    ex.bars.forEach((b, k) => {
      const x = (k - (n - 1) / 2) * step;
      const val = textSprite(fmtValue(rec, b.row.value), { size: dense ? 0.19 : 0.26, font: 'mono', weight: 650, color: '#eef3ff' });
      val.position.set(x, b.hgt + 0.3, 0);
      const wide = n <= 6;
      const name = textSprite(b.row.label + (b.row.sub && !dense ? '\n' + b.row.sub : ''), { size: dense ? 0.19 : wide ? 0.24 : 0.2, color: '#c3cde2', weight: 550, anchor: [0.5, 1], maxChars: dense ? 9 : wide ? 14 : 10, lineHeight: 1.12 });
      name.position.set(x, -0.12, 1.1 + (dense && k % 2 ? 0.55 : 0));
      if (dense && k % 2) name.position.y -= 0.62;
      labels.add(val, name);
    });
    const unit = textSprite(rec.unit + (rec.log ? ' (log scale)' : ''), { size: 0.25, color: '#93a1bf', weight: 500, anchor: [0, 0.5] });
    unit.position.set(-W / 2 - 0.5, H + 0.75, 0);
    labels.add(unit);
  }
  function clearLabels(ex) {
    for (const s of ex.labels.children) s.material.dispose();
    ex.labels.clear();
  }

  // ------------------------------------------------------------------ selection + HUD
  let active = -1;
  const listEl = h('div.rec-list');
  const whyEl = h('div');
  const tableEl = h('div');
  const prevBtn = button('←', { kind: 'ghost', title: 'Previous exhibit (←)', onClick: () => select((active + RECORDS.length - 1) % RECORDS.length) });
  const nextBtn = button('Next exhibit →', { kind: 'primary', onClick: () => select((active + 1) % RECORDS.length) });
  const titleEl = h('span.t', 'Seven exhibits');

  function renderList() {
    listEl.replaceChildren(
      ...RECORDS.map((rec, i) => {
        const b = h('button.rec-item' + (i === active ? '.on' : ''), { type: 'button' }, h('span.n', String(i + 1).padStart(2, '0')), rec.short);
        b.addEventListener('click', () => select(i));
        return b;
      }),
    );
  }

  function renderPanels() {
    if (active < 0) {
      whyEl.replaceChildren(text('Walk the ring: each stand is one record. Click a stand or pick an exhibit above. Unlike the rest of this chapter nothing here is measured on your machine: these are published numbers, each with its source.').firstChild);
      ctx.ui.right();
      titleEl.textContent = `${RECORDS.length} exhibits · published figures as of ${AS_OF}`;
      return;
    }
    const rec = RECORDS[active];
    const groupKeys = rec.groups ? Object.keys(rec.groups) : null;
    whyEl.replaceChildren(h('h2.steptitle', h('span.num', String(active + 1).padStart(2, '0')), rec.title), text(rec.why), callout(rec.note, 'info'), rec.link ? h('div.btnrow', button(rec.link.label + ' →', { onClick: () => (location.hash = rec.link.href) })) : '');
    const hasInput = rec.rows.some((r) => r.value2 != null);
    const rows = rec.rows.map((r) => {
      const host = r.src ? new URL(r.src).hostname.replace(/^www\./, '') : null;
      const color = groupKeys ? SERIES[groupKeys.indexOf(String(r.group)) % SERIES.length] : SERIES[active % SERIES.length];
      return h(
        'tr',
        h('td', h('i.sw', { style: { '--c': color } }), r.label, r.sub ? h('small', r.sub + (r.noteRow && !groupKeys ? ' · ' + r.noteRow : '')) : ''),
        hasInput ? h('td', '$' + r.value2) : h('td', ''),
        h('td', fmtValue(rec, r.value)),
        h('td', host ? h('a', { href: r.src, target: '_blank', rel: 'noopener', title: r.src }, host.split('.').slice(-2).join('.')) : h('span', { style: { color: 'var(--ink-3)' } }, rec.id === 'memory' ? 'computed' : rec.id === 'price' ? 'launch list price' : '')),
      );
    });
    tableEl.replaceChildren(
      groupKeys ? legend(groupKeys.map((k, j) => ({ color: SERIES[j % SERIES.length], label: rec.groups[k] }))) : '',
      h('table.tbl.compact.rec-table', h('thead', h('tr', h('th', ''), h('th', hasInput ? 'Input' : ''), h('th', hasInput ? 'Output' : rec.unit.split(' ')[0] === '%' ? rec.unit : 'Value'), h('th', 'Source'))), h('tbody', rows)),
      rec.reference?.src ? h('div.src', 'Reference line: ', h('a', { href: rec.reference.src, target: '_blank', rel: 'noopener' }, rec.reference.label)) : '',
    );
    ctx.ui.right(section(rec.unit, tableEl));
    titleEl.textContent = `${active + 1} / ${RECORDS.length} · ${rec.short}`;
  }

  function select(i, fly = true) {
    if (active >= 0) clearLabels(exhibits[active]);
    active = i;
    if (i >= 0) {
      const ex = exhibits[i];
      buildLabels(ex);
      ex.grow = 0;
      const dir = new THREE.Vector3(-Math.sin(ex.angle), 0.14, -Math.cos(ex.angle)); // from the exhibit towards the center
      if (fly) app.frame({ target: [ex.g.position.x, 2.6, ex.g.position.z], dir: dir.toArray(), fit: 7.1 }, 1.2);
      ctx.router.replace('perf/records/' + ex.rec.id);
    } else {
      if (fly) app.home();
      ctx.router.replace('perf/records');
    }
    renderList();
    renderPanels();
  }

  ctx.ui.left(
    header({ kicker: 'Experiment · Hall of Records', title: 'Numbers with receipts', lead: `The reference figures people quote, each with its source. Snapshot of ${AS_OF}.` }),
    listEl,
    whyEl,
  );
  ctx.ui.dock(h('div.tour', prevBtn, h('div.mid', titleEl), button('All', { kind: 'ghost', title: 'Back to the overview', onClick: () => select(-1) }), nextBtn));
  ctx.key('ArrowRight', () => nextBtn.click());
  ctx.key('ArrowLeft', () => prevBtn.click());
  ctx.pick.onMiss(() => {});

  const openFromRoute = (route) => {
    const idx = RECORDS.findIndex((r) => r.id === route.rest[0]);
    if (idx !== active) select(idx, true);
  };
  renderList();
  renderPanels();
  if (ctx.route.rest[0]) openFromRoute(ctx.route);

  return {
    onRoute: openFromRoute,
    update(dt, t) {
      exhibits.forEach((ex, i) => {
        const on = i === active;
        ex.grow = damp(ex.grow, 1, 0.35, dt);
        ex.bars.forEach((b, k) => {
          const want = b.hgt * Math.min(1, ex.grow * 1.25 + (active < 0 ? 1 : 0));
          b.mesh.scale.y = damp(b.mesh.scale.y, Math.max(0.02, want), 0.14 + k * 0.012, dt);
          b.mat.color.copy(b.base).multiplyScalar(on || active < 0 ? b.boost : 0.4);
        });
      });
      floorRing.rotation.y = t * 0.02;
    },
  };
}
