// DOM building blocks for the HUD. Everything returns plain elements (or {el, ...api}) so chapters
// can compose panels without a framework.

/**
 * Create an element. `h('div.card.big', {onclick}, child, 'text', [more])`.
 * Props: class, style (object, supports --vars), html (trusted markup only), text, dataset, on<Event>, any attribute.
 */
export function h(tag, props, ...kids) {
  const [name, ...cls] = tag.split('.');
  const el = document.createElement(name || 'div');
  if (cls.length) el.className = cls.join(' ');
  if (props != null && (typeof props !== 'object' || props instanceof Node || Array.isArray(props) || props.el instanceof Node)) {
    kids.unshift(props);
    props = null;
  }
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className += (el.className ? ' ' : '') + v;
    else if (k === 'style' && typeof v === 'object') {
      for (const [sk, sv] of Object.entries(v)) {
        if (sk.startsWith('--')) el.style.setProperty(sk, sv);
        else el.style[sk] = sv;
      }
    } else if (k === 'html') el.innerHTML = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  append(el, kids);
  return el;
}

function append(el, kids) {
  for (const kid of kids) {
    if (kid == null || kid === false) continue;
    if (Array.isArray(kid)) append(el, kid);
    else if (kid instanceof Node) el.appendChild(kid);
    else if (kid && kid.el instanceof Node) el.appendChild(kid.el);
    else el.appendChild(document.createTextNode(String(kid)));
  }
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

/**
 * Tiny inline formatter for content strings: **bold**, _em_, `code`, [text](https://url), blank line = new paragraph.
 * Input is escaped first, so it is safe for untrusted text too.
 * @returns {DocumentFragment}
 */
export function rich(text) {
  const frag = document.createDocumentFragment();
  for (const para of String(text).trim().split(/\n\s*\n/)) {
    let s = esc(para);
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[\s(])_([^_]+)_(?=[\s).,;:!?]|$)/g, '$1<em>$2</em>');
    s = s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    s = s.replace(/\[([^\]]+)\]\((#\/[^)\s]*)\)/g, '<a href="$2">$1</a>');
    const p = document.createElement('p');
    p.innerHTML = s.replace(/\n/g, '<br>');
    frag.appendChild(p);
  }
  return frag;
}

/** Standard panel header: small kicker, title, lead paragraph. */
export function header({ kicker, title, lead }) {
  return h('header.ph', kicker && h('div.kicker', kicker), title && h('h1', title), lead && h('div.lead', rich(lead)));
}

/** Titled block inside a panel. */
export function section(title, ...kids) {
  return h('section.sec', title && h('h3', title), ...kids);
}

/** Paragraph(s) from a content string (supports the `rich` inline syntax). */
export function text(str, cls = '') {
  return h('div.txt' + (cls ? '.' + cls : ''), rich(str));
}

/** Highlighted note. kind: 'info' | 'warn' | 'key' */
export function callout(str, kind = 'info', title) {
  return h('div.callout.' + kind, title && h('strong', title), rich(str));
}

export function button(label, o = {}) {
  const el = h(
    'button.btn' + (o.kind ? '.' + o.kind : ''),
    { type: 'button', title: o.title, onclick: o.onClick, 'aria-pressed': o.active != null ? String(!!o.active) : null },
    o.icon && h('span.ico', { html: o.icon }),
    label,
  );
  if (o.active) el.classList.add('on');
  return el;
}

/**
 * Segmented control (single choice).
 * @param {{value:any,label:string,title?:string}[]} options
 */
export function segmented(options, value, onChange, o = {}) {
  let cur = value;
  const el = h('div.seg' + (o.wrap ? '.wrap' : ''), { role: 'tablist' });
  const btns = options.map((opt) => {
    const b = h('button', { type: 'button', role: 'tab', title: opt.title }, opt.label);
    b.addEventListener('click', () => {
      if (cur === opt.value) return;
      api.set(opt.value);
      onChange?.(opt.value);
    });
    el.appendChild(b);
    return b;
  });
  const api = {
    el,
    get: () => cur,
    set(v) {
      cur = v;
      btns.forEach((b, i) => {
        const on = options[i].value === v;
        b.classList.toggle('on', on);
        b.setAttribute('aria-selected', String(on));
      });
    },
  };
  api.set(cur);
  return api;
}

/** Labeled range slider. format(v) renders the value read-out. */
export function slider(o) {
  const fmt = o.format ?? ((v) => String(+v.toFixed(3)));
  const out = h('output', fmt(o.value));
  const input = h('input', { type: 'range', min: o.min, max: o.max, step: o.step ?? 'any', 'aria-label': o.label });
  input.value = o.value;
  const paint = () => {
    const p = ((+input.value - o.min) / (o.max - o.min)) * 100;
    input.style.setProperty('--p', p + '%');
  };
  paint();
  input.addEventListener('input', () => {
    out.textContent = fmt(+input.value);
    paint();
    o.onInput?.(+input.value);
  });
  input.addEventListener('change', () => o.onChange?.(+input.value));
  const el = h('label.slider', h('span.row', h('span.lab', o.label), out), input, o.hint && h('span.hint', o.hint));
  return {
    el,
    get: () => +input.value,
    set(v) {
      input.value = v;
      out.textContent = fmt(+input.value);
      paint();
    },
  };
}

export function toggle(o) {
  const input = h('input', { type: 'checkbox' });
  input.checked = !!o.value;
  input.addEventListener('change', () => o.onChange?.(input.checked));
  const el = h('label.toggle', input, h('span.track'), h('span.lab', o.label));
  return { el, get: () => input.checked, set: (v) => (input.checked = !!v) };
}

export function select(o) {
  const sel = h('select', { 'aria-label': o.label });
  for (const opt of o.options) {
    const op = h('option', { value: opt.value }, opt.label);
    sel.appendChild(op);
  }
  sel.value = o.value;
  sel.addEventListener('change', () => o.onChange?.(sel.value));
  const el = h('label.select', o.label && h('span.lab', o.label), sel);
  return { el, get: () => sel.value, set: (v) => (sel.value = v) };
}

/** Stat tile: label, big value, optional unit and sub-line. */
export function stat(o) {
  const val = h('span.v', o.value ?? '–');
  const sub = h('div.s', o.sub ?? '');
  const el = h('div.stat', h('div.l', o.label), h('div.vr', val, o.unit && h('span.u', o.unit)), sub);
  if (o.color) el.style.setProperty('--c', o.color);
  return {
    el,
    set(value, subText) {
      val.textContent = value;
      if (subText != null) sub.textContent = subText;
    },
  };
}

export function statRow(...stats) {
  return h('div.stats', ...stats);
}

/** Legend. items: [{color, label, shape?: 'box'|'line'|'dot'}] */
export function legend(items) {
  return h(
    'div.legend',
    items.map((it) => h('span.li', h('i.' + (it.shape || 'box'), { style: { '--c': it.color } }), it.label)),
  );
}

/** Key / value rows. rows: [[key, value], ...]; value may be a Node or string. */
export function kv(rows) {
  return h(
    'dl.kv',
    rows.filter(Boolean).map(([k, v]) => [h('dt', k), h('dd', v instanceof Node ? v : rich(String(v)))]),
  );
}

/** Monospace block. Pass lang 'json' for light highlighting. */
export function code(str, lang = '') {
  const pre = h('pre.code');
  if (lang === 'json') {
    pre.innerHTML = esc(str)
      .replace(/(&quot;(?:[^&]|&(?!quot;))*?&quot;)(\s*:)?/g, (m, s, colon) => (colon ? `<span class="k">${s}</span>${colon}` : `<span class="s">${s}</span>`))
      .replace(/\b(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b/gi, '<span class="n">$1</span>')
      .replace(/\b(true|false|null)\b/g, '<span class="b">$1</span>');
  } else pre.textContent = str;
  return pre;
}

/**
 * Aligned meter rows (name, bar, figure). rows: [{label, value, max?, color?, note?, title?}]
 */
export function meterList(rows, o = {}) {
  const el = h('div.meters');
  const api = {
    el,
    update(next) {
      const max = o.max ?? Math.max(1e-9, ...next.map((r) => r.max ?? r.value));
      el.replaceChildren(
        ...next.map((r) => {
          const pct = Math.max(0, Math.min(100, (r.value / (r.max ?? max)) * 100));
          const row = h(
            'div.meter' + (r.active ? '.on' : ''),
            { title: r.title },
            h('span.ml', r.label),
            h('span.mb', h('i', { style: { width: pct + '%', '--c': r.color || o.color || 'var(--accent)' } })),
            h('span.mv', r.note ?? (o.format ? o.format(r.value) : String(r.value))),
          );
          if (r.onClick) {
            row.classList.add('click');
            row.addEventListener('click', r.onClick);
          }
          return row;
        }),
      );
    },
  };
  api.update(rows);
  return api;
}

/** Small chips, e.g. tags. */
export function chips(items, o = {}) {
  return h(
    'div.chips',
    items.map((it) => {
      const label = typeof it === 'string' ? it : it.label;
      const c = h('span.chip', { style: it.color ? { '--c': it.color } : null, title: it.title }, label);
      if (it.onClick || o.onClick) {
        c.classList.add('click');
        c.addEventListener('click', () => (it.onClick || o.onClick)(it));
      }
      return c;
    }),
  );
}

/** Simple table. head: string[], rows: (string|Node)[][] */
export function table(head, rows, o = {}) {
  return h(
    'table.tbl' + (o.compact ? '.compact' : ''),
    head && h('thead', h('tr', head.map((c) => h('th', c)))),
    h('tbody', rows.map((r) => h('tr', r.map((c) => h('td', c instanceof Node ? c : rich(String(c))))))),
  );
}

/** Tabs that swap content inside a panel. items: [{id, label, content: Node | () => Node}] */
export function tabs(items, active, onChange) {
  const body = h('div.tabbody');
  const show = (id) => {
    const it = items.find((x) => x.id === id) || items[0];
    const c = typeof it.content === 'function' ? it.content() : it.content;
    body.replaceChildren(c);
  };
  const seg = segmented(
    items.map((it) => ({ value: it.id, label: it.label })),
    active ?? items[0].id,
    (id) => {
      show(id);
      onChange?.(id);
    },
  );
  show(seg.get());
  return {
    el: h('div.tabs', seg.el, body),
    set(id) {
      seg.set(id);
      show(id);
    },
    get: seg.get,
  };
}

/**
 * Guided tour: a stepper in the dock + step text in a container you place in the left panel.
 * steps: [{ title, text?, body?: Node | () => Node, enter?(i): void, leave?(i): void }]
 * Returns { dock, body, go(i), next(), prev(), index }.
 * Arrow keys step through; the current step is mirrored to the URL as ?s=N.
 */
export function tour(ctx, steps, o = {}) {
  let index = -1;
  const body = h('div.tourbody');
  const dots = steps.map((s, i) => {
    const b = h('button.dot', { type: 'button', title: `${i + 1}. ${s.title}`, 'aria-label': `Step ${i + 1}: ${s.title}` });
    b.addEventListener('click', () => api.go(i));
    return b;
  });
  const title = h('span.t');
  const prev = h('button.btn.ghost.stepbtn', { type: 'button', title: 'Previous step (←)', onclick: () => api.prev() }, '←');
  const next = h('button.btn.primary.stepbtn', { type: 'button', title: 'Next step (→)', onclick: () => api.next() }, 'Next →');
  const dock = h('div.tour', prev, h('div.mid', title, h('div.dots', dots)), next);
  const api = {
    dock,
    body,
    get index() {
      return index;
    },
    go(i, silent = false) {
      i = Math.max(0, Math.min(steps.length - 1, i));
      if (i === index) return;
      if (index >= 0) steps[index].leave?.(index);
      index = i;
      const s = steps[i];
      title.textContent = `${i + 1} / ${steps.length} · ${s.title}`;
      dots.forEach((d, j) => {
        d.classList.toggle('on', j === i);
        d.classList.toggle('done', j < i);
      });
      prev.disabled = i === 0;
      next.textContent = i === steps.length - 1 ? (o.lastLabel ?? 'Done ✓') : 'Next →';
      const content = [h('h2.steptitle', h('span.num', String(i + 1).padStart(2, '0')), s.title)];
      if (s.text) content.push(text(s.text));
      if (s.body) content.push(typeof s.body === 'function' ? s.body() : s.body);
      // a body may be a node, a {el} component, or an array of those
      body.replaceChildren(...content.flat(Infinity).filter((n) => n != null && n !== false).map((n) => (n.el instanceof Node ? n.el : n)));
      body.parentElement?.scrollTo?.({ top: 0 });
      if (!silent && o.syncUrl !== false) {
        const r = ctx.router.current();
        ctx.router.replace(r.path, { ...r.query, s: String(i + 1) });
      }
      s.enter?.(i);
      o.onStep?.(i);
    },
    next() {
      if (index === steps.length - 1) o.onFinish?.();
      else api.go(index + 1);
    },
    prev() {
      api.go(index - 1);
    },
  };
  ctx.key('ArrowRight', () => api.next());
  ctx.key('ArrowLeft', () => api.prev());
  const start = Math.max(0, Math.min(steps.length - 1, (parseInt(ctx.route.query.s, 10) || 1) - 1));
  queueMicrotask(() => api.go(o.start ?? start, true));
  return api;
}

export const fmt = {
  int: (v) => Math.round(v).toLocaleString('en-US'),
  /** 1,284 · 12.9K · 4.2M · 1.3B */
  compact(v) {
    const a = Math.abs(v);
    if (a >= 1e12) return +(v / 1e12).toPrecision(3) + 'T';
    if (a >= 1e9) return +(v / 1e9).toPrecision(3) + 'B';
    if (a >= 1e6) return +(v / 1e6).toPrecision(3) + 'M';
    if (a >= 1e4) return +(v / 1e3).toPrecision(3) + 'K';
    if (a >= 100) return Math.round(v).toLocaleString('en-US');
    return String(+v.toPrecision(3));
  },
  pct: (v, d = 0) => (v * 100).toFixed(d) + '%',
  fixed: (d) => (v) => v.toFixed(d),
  ms: (v) => (v >= 1000 ? (v / 1000).toFixed(2) + ' s' : v >= 10 ? v.toFixed(0) + ' ms' : v.toFixed(1) + ' ms'),
  bytes(v) {
    if (v >= 1e9) return (v / 1e9).toFixed(2) + ' GB';
    if (v >= 1e6) return (v / 1e6).toFixed(2) + ' MB';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + ' kB';
    return v + ' B';
  },
};
