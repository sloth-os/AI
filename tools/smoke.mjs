// Headless smoke test of every route in one browser session: loads each chapter, steps its
// update loop, and reports anything the page logged as an error.
//   node tools/smoke.mjs [--base http://127.0.0.1:5173/] [--q low] [--only '#/hot'] [--size 1440x900]
// Exits non-zero if any route failed. Needs Playwright (not a dependency of the site).
import { createRequire } from 'node:module';
import { ROUTES } from '../src/routes.js';
import { HOT } from '../src/data/hot.js';
import { PIPELINES } from '../src/data/pipelines.js';
import { MODES } from '../src/data/deployments.js';
import { LEVELS } from '../src/data/problems.js';
import { RECORDS } from '../src/data/records.js';

const { chromium } = createRequire(import.meta.url)('playwright');

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args.splice(i, 2)[1] : def;
};
const base = opt('base', 'http://127.0.0.1:5173/');
const q = opt('q', 'low');
const only = opt('only', '');
const [w, h] = opt('size', '1440x900').split('x').map(Number);

const routes = [
  ...ROUTES.map((r) => '#/' + r.path),
  ...HOT.map((x) => '#/hot/' + x.id),
  ...PIPELINES.flatMap((p) => ['#/flow/' + p.id, `#/flow/${p.id}?s=${p.stages.length - 1}`]),
  ...MODES.map((m) => '#/deploy/' + m.id),
  '#/deploy?view=all',
  ...LEVELS.flatMap((l) => ['#/problems/' + l.id, `#/problems/${l.id}/${l.problems[0].id}`]),
  ...RECORDS.map((r) => '#/perf/records/' + r.id),
  '#/ecosystem/vllm',
  '#/no/such/route',
].filter((r) => r.startsWith(only));

const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'] });
const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 0.5 });
page.setDefaultTimeout(240000);
let errors = [];
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message));

const ready = () => page.waitForFunction(() => window.__atlas && !window.__atlas._navigating && window.__atlas.current, null, { timeout: 180000 });
const sep = base.includes('?') ? '&' : '?';
await page.goto(`${base}${sep}q=${q}#/`, { waitUntil: 'load' });
await ready();

let failed = 0;
for (const hash of routes) {
  errors = [];
  const t0 = Date.now();
  try {
    await page.evaluate((x) => (location.hash = x), hash);
    await page.waitForTimeout(250);
    await ready();
    await page.waitForTimeout(500);
    // step the chapter well past its opening animation without waiting for real time
    await page.evaluate(async () => {
      const c = window.__atlas.current.ctrl;
      for (let i = 0; i < 400; i++) {
        c.update?.(0.05, i * 0.05);
        if (i % 50 === 49) await new Promise((r) => setTimeout(r, 0));
      }
    });
    await page.waitForTimeout(300);
  } catch (e) {
    errors.push('SMOKE ' + e.message.split('\n')[0]);
  }
  const uniq = [...new Set(errors)];
  if (uniq.length) failed++;
  console.log(`${uniq.length ? 'FAIL' : 'ok  '} ${hash}  (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  for (const e of uniq.slice(0, 6)) console.log('     ' + e.slice(0, 400));
}
await browser.close();
console.log(failed ? `\n${failed} of ${routes.length} routes reported errors` : `\nall ${routes.length} routes clean`);
process.exit(failed ? 1 : 0);
