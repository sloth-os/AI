// Hash router: "#/labs/gpt?s=2" -> { path: 'labs/gpt', parts: ['labs','gpt'], query: { s: '2' } }

export function parseHash(hash = location.hash) {
  let h = hash.replace(/^#\/?/, '');
  let queryString = '';
  const q = h.indexOf('?');
  if (q >= 0) {
    queryString = h.slice(q + 1);
    h = h.slice(0, q);
  }
  const parts = h.split('/').map(decodeURIComponent).filter(Boolean);
  const query = {};
  for (const [k, v] of new URLSearchParams(queryString)) query[k] = v;
  return { path: parts.join('/'), parts, query };
}

export function buildHash(path, query) {
  const qs = query && Object.keys(query).length ? '?' + new URLSearchParams(query).toString() : '';
  return '#/' + String(path || '').replace(/^\/+/, '') + qs;
}

export class Router {
  constructor() {
    this.listeners = new Set();
    this._silent = false;
    window.addEventListener('hashchange', () => {
      if (this._silent) {
        this._silent = false;
        return;
      }
      this.emit();
    });
  }

  current() {
    return parseHash();
  }

  /** Navigate (adds a history entry). */
  go(path, query) {
    const next = buildHash(path, query);
    if (next === location.hash || (location.hash === '' && next === '#/')) this.emit();
    else location.hash = next;
  }

  /** Update the URL without notifying listeners (for in-chapter state such as the current step). */
  replace(path, query) {
    const next = buildHash(path, query);
    if (next === location.hash) return;
    history.replaceState(null, '', next);
  }

  onChange(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  emit() {
    const r = this.current();
    for (const fn of this.listeners) fn(r);
  }
}
