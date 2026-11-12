// The HUD shell: top navigation, the two glass panels, the dock, tooltip, toasts and the fade used
// between chapters. Chapters fill it through ctx.ui (left / right / dock).
import { h } from './ui.js';
import { NAV } from '../routes.js';

const ICON = {
  home: '<svg viewBox="0 0 24 24"><path d="M4 11.5 12 5l8 6.5V19a1 1 0 0 1-1 1h-4.5v-5h-5v5H5a1 1 0 0 1-1-1z"/></svg>',
  reset: '<svg viewBox="0 0 24 24"><path d="M12 5a7 7 0 1 1-6.6 4.7M5 4v5.5h5.5"/></svg>',
  full: '<svg viewBox="0 0 24 24"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  help: '<svg viewBox="0 0 24 24"><path d="M9.5 9.3a2.6 2.6 0 1 1 3.8 2.3c-.8.5-1.3 1-1.3 2M12 17.2v.3"/><circle cx="12" cy="12" r="9"/></svg>',
  gauge: '<svg viewBox="0 0 24 24"><path d="M4.5 16a8 8 0 1 1 15 0M12 13l3.5-4.5"/></svg>',
  panels: '<svg viewBox="0 0 24 24"><rect x="3.5" y="5" width="17" height="14" rx="2"/><path d="M9 5v14M15 5v14"/></svg>',
  chevron: '<svg viewBox="0 0 24 24"><path d="m14 6-6 6 6 6"/></svg>',
};

export class Hud {
  constructor(app) {
    this.app = app;
    const root = (this.root = document.getElementById('hud'));

    // ----- top bar
    this.navBtns = new Map();
    const nav = h(
      'nav.nav',
      { 'aria-label': 'Chapters' },
      NAV.map((n, i) => {
        const b = h('a.navlink', { href: '#/' + n.path, style: { '--c': n.accent } }, h('span.i', String(i).padStart(2, '0')), n.label);
        this.navBtns.set(n.id, b);
        return b;
      }),
    );
    const tool = (icon, title, onClick) => h('button.tool', { type: 'button', title, 'aria-label': title, html: icon, onclick: onClick });
    this.qualityBtn = h('button.tool.q', { type: 'button', title: 'Rendering quality', onclick: () => app.cycleQuality() }, h('span', { html: ICON.gauge }), h('b', ''));
    const tools = h(
      'div.tools',
      tool(ICON.reset, 'Reset view (H)', () => app.home()),
      tool(ICON.panels, 'Show / hide panels (P)', () => this.togglePanels()),
      this.qualityBtn,
      tool(ICON.full, 'Fullscreen (F)', () => app.toggleFullscreen()),
      tool(ICON.help, 'Help (?)', () => this.toggleHelp()),
    );
    const brand = h(
      'a.brand',
      { href: '#/', title: 'AI Atlas: home' },
      h('span.logo', { html: '<svg viewBox="0 0 32 32"><path d="M16 3 28 10v12L16 29 4 22V10z" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="16" cy="16" r="3.2" fill="currentColor"/><path d="M16 3v9.8M28 10l-9.2 4.4M28 22l-9.2-4.4M16 29v-9.8M4 22l9.2-4.4M4 10l9.2 4.4" stroke="currentColor" stroke-width="1.1" opacity=".7"/></svg>' }),
      h('span.name', 'AI ATLAS', h('em', '3D')),
    );
    this.top = h('header.topbar', brand, nav, tools);

    // ----- panels
    const mkPanel = (side) => {
      const body = h('div.panel-body');
      const toggle = h('button.panel-toggle', { type: 'button', title: 'Collapse panel', 'aria-label': 'Collapse panel', html: ICON.chevron });
      const el = h('aside.panel.' + side, { hidden: true }, toggle, body);
      toggle.addEventListener('click', () => {
        el.classList.toggle('collapsed');
        app.layout();
      });
      return { el, body };
    };
    this.L = mkPanel('left');
    this.R = mkPanel('right');
    this.dockEl = h('footer.dock', { hidden: true });

    // mobile sheet switcher
    this.sheetTabs = h(
      'div.sheet-tabs',
      { hidden: true },
      ['Explain', 'Controls', 'Hide'].map((label, i) =>
        h('button', { type: 'button', onclick: () => this.setSheet(['left', 'right', 'none'][i]) }, label),
      ),
    );

    this.tipEl = h('div.tip', { hidden: true, role: 'tooltip' });
    this.toastsEl = h('div.toasts', { 'aria-live': 'polite' });
    this.fadeEl = h('div.fade', h('div.fade-label', h('i.spin'), h('span', '')));
    this.helpEl = h(
      'div.help',
      { hidden: true },
      h(
        'div.help-card',
        h('h2', 'How to explore'),
        h(
          'dl',
          ...[
            ['Drag', 'orbit the scene'],
            ['Right-drag / two fingers', 'pan'],
            ['Scroll / pinch', 'zoom'],
            ['Click', 'select objects, open portals'],
            ['← →', 'previous / next step'],
            ['H', 'reset the camera'],
            ['P', 'show or hide the panels'],
            ['F', 'fullscreen'],
          ].flatMap(([k, v]) => [h('dt', k), h('dd', v)]),
        ),
        h('p', 'Every simulation on this site runs in your browser: the neural networks train live, the tiny GPT really predicts tokens, and the benchmarks are measured on your machine.'),
        h('button.btn.primary', { type: 'button', onclick: () => this.toggleHelp(false) }, 'Got it'),
      ),
    );
    this.helpEl.addEventListener('click', (e) => {
      if (e.target === this.helpEl) this.toggleHelp(false);
    });

    root.append(this.top, this.L.el, this.R.el, this.dockEl, this.sheetTabs, this.tipEl, this.toastsEl, this.helpEl, this.fadeEl);
    this.setSheet('left');
  }

  setActive(navId, accent) {
    for (const [id, b] of this.navBtns) b.classList.toggle('on', id === navId);
    if (accent) document.documentElement.style.setProperty('--accent', accent);
    const active = this.navBtns.get(navId);
    active?.scrollIntoView?.({ block: 'nearest', inline: 'center' });
  }

  setQualityLabel(q) {
    this.qualityBtn.querySelector('b').textContent = { high: 'HI', medium: 'MED', low: 'LOW' }[q];
    this.qualityBtn.title = `Rendering quality: ${q} (click to change)`;
  }

  _fill(panel, nodes) {
    const kids = nodes.flat(Infinity).filter((n) => n != null && n !== false).map((n) => (n.el instanceof Node ? n.el : n));
    panel.body.replaceChildren(...kids);
    panel.body.scrollTop = 0;
    panel.el.hidden = kids.length === 0;
    this._syncSheet();
    this.app.layout?.();
  }

  left(...nodes) {
    this._fill(this.L, nodes);
  }

  right(...nodes) {
    this._fill(this.R, nodes);
  }

  dock(...nodes) {
    const kids = nodes.flat(Infinity).filter(Boolean).map((n) => (n.el instanceof Node ? n.el : n));
    this.dockEl.replaceChildren(...kids);
    this.dockEl.hidden = kids.length === 0;
  }

  clear() {
    this.left();
    this.right();
    this.dock();
    this.tip(null);
    this.L.el.classList.remove('collapsed');
    this.R.el.classList.remove('collapsed');
  }

  togglePanels(force) {
    const hide = force != null ? !force : !this.root.classList.contains('no-panels');
    this.root.classList.toggle('no-panels', hide);
    this.app.layout?.();
  }

  /** Small screens show one panel at a time as a bottom sheet. */
  setSheet(which) {
    this.sheet = which;
    this.root.dataset.sheet = which;
    [...this.sheetTabs.children].forEach((b, i) => b.classList.toggle('on', ['left', 'right', 'none'][i] === which));
    this.app.layout?.();
  }

  _syncSheet() {
    const both = !this.L.el.hidden || !this.R.el.hidden;
    this.sheetTabs.hidden = !both;
    this.sheetTabs.children[1].hidden = this.R.el.hidden;
    if (this.sheet === 'right' && this.R.el.hidden) this.setSheet('left');
  }

  toggleHelp(force) {
    this.helpEl.hidden = force != null ? !force : !this.helpEl.hidden;
  }

  toast(msg, ms = 2800) {
    const el = h('div.toast', msg);
    this.toastsEl.appendChild(el);
    setTimeout(() => el.classList.add('out'), ms);
    setTimeout(() => el.remove(), ms + 400);
  }

  /** Tooltip near the pointer. content: string | Node | null */
  tip(content, x = 0, y = 0) {
    const el = this.tipEl;
    if (content == null || content === '') {
      el.hidden = true;
      return;
    }
    if (el._content !== content) {
      el._content = content;
      if (content instanceof Node) el.replaceChildren(content);
      else el.textContent = content;
    }
    el.hidden = false;
    const w = el.offsetWidth;
    const hh = el.offsetHeight;
    const px = Math.min(window.innerWidth - w - 10, Math.max(10, x + 16));
    const py = y + 18 + hh > window.innerHeight - 10 ? y - hh - 12 : y + 18;
    el.style.transform = `translate(${Math.round(px)}px, ${Math.round(Math.max(10, py))}px)`;
  }

  /** Fade the whole view to black (true) or back (false). Resolves when the transition has finished. */
  fade(on, label = '') {
    const el = this.fadeEl;
    el.querySelector('span').textContent = label;
    el.classList.toggle('labeled', !!label);
    return new Promise((resolve) => {
      if (el.classList.contains('on') === on) return resolve();
      const done = () => {
        clearTimeout(timer);
        el.removeEventListener('transitionend', done);
        resolve();
      };
      const timer = setTimeout(done, 520);
      el.addEventListener('transitionend', done);
      el.classList.toggle('on', on);
    });
  }

  loading(label) {
    this.fadeEl.querySelector('span').textContent = label || '';
    this.fadeEl.classList.toggle('labeled', !!label);
  }
}
