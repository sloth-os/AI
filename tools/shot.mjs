// Headless smoke test + screenshot of one route.
//   node tools/shot.mjs '#/labs/gpt' shots/gpt.png [--wait 2500] [--size 1440x900] [--q low] [--eval 'js'] [--after 600] [--base http://127.0.0.1:5173/]
// Prints console errors and exits non-zero if the page threw.
import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

// Playwright is resolved the CommonJS way so a globally installed copy (NODE_PATH) works too.
const { chromium } = createRequire(import.meta.url)('playwright');

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args.splice(i, 2)[1] : def;
};
const wait = +opt('wait', 2500);
const [w, h] = opt('size', '1440x900').split('x').map(Number);
const q = opt('q', 'medium');
const base = opt('base', 'http://127.0.0.1:5173/');
const evalJs = opt('eval', '');
const dpr = +opt('dpr', 1);
const after = +opt('after', 600);
const [hash = '#/', out = 'shots/out.png'] = args;

const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'] });
const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: dpr });
page.setDefaultTimeout(300000); // software rendering on a busy machine can be very slow
page.setDefaultNavigationTimeout(300000);
const errors = [];
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text());
});
page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message));
const sep = base.includes('?') ? '&' : '?';
await page.goto(`${base}${sep}q=${q}${hash}`, { waitUntil: 'load' });
await page.waitForFunction(() => window.__atlas && !window.__atlas._navigating && window.__atlas.current, null, { timeout: 240000 }).catch(() => errors.push('TIMEOUT waiting for chapter'));
await page.waitForTimeout(wait);
if (evalJs) {
  const r = await page.evaluate(evalJs).catch((e) => 'EVAL ERROR ' + e.message);
  if (r !== undefined) console.log('eval:', typeof r === 'string' ? r : JSON.stringify(r));
  await page.waitForTimeout(after);
}
mkdirSync(dirname(out), { recursive: true });
await page.screenshot({ path: out });
await browser.close();
if (errors.length) {
  console.log('ERRORS:\n' + [...new Set(errors)].slice(0, 12).join('\n'));
  process.exit(1);
}
console.log('ok', hash, '->', out);
