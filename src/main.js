import './style.css';
import { App } from './core/app.js';
import { Router } from './core/router.js';
import interUrl from '@fontsource-variable/inter/files/inter-latin-wght-normal.woff2?url';
import groteskUrl from '@fontsource-variable/space-grotesk/files/space-grotesk-latin-wght-normal.woff2?url';
import monoUrl from '@fontsource-variable/jetbrains-mono/files/jetbrains-mono-latin-wght-normal.woff2?url';

// Fonts are loaded through the FontFace API (not CSS) so we can wait for them before any 3D text
// label is rasterised to a canvas texture.
async function loadFonts() {
  const faces = [
    ['Inter Variable', interUrl, '100 900'],
    ['Space Grotesk Variable', groteskUrl, '300 700'],
    ['JetBrains Mono Variable', monoUrl, '100 800'],
  ];
  await Promise.all(
    faces.map(async ([family, url, weight]) => {
      const face = new FontFace(family, `url(${url}) format('woff2')`, { weight, style: 'normal', display: 'swap' });
      document.fonts.add(face);
      await face.load();
    }),
  );
}

async function boot() {
  await Promise.race([loadFonts().catch((err) => console.warn('font load failed', err)), new Promise((r) => setTimeout(r, 2500))]);
  const bootEl = document.getElementById('boot');
  let app;
  try {
    const router = new Router();
    app = new App({ canvas: document.getElementById('gl'), router });
    window.__atlas = app;
    await app._navigate(router.current());
  } catch (err) {
    console.error(err);
    if (!app) {
      const el = document.createElement('div');
      el.id = 'nogl';
      el.innerHTML =
        '<div><h1>AI Atlas 3D</h1><p>This site renders an interactive 3D world and needs WebGL 2, which is not available in this browser or is switched off.</p><p>Try a current version of Chrome, Edge, Firefox or Safari with hardware acceleration enabled.</p></div>';
      document.body.appendChild(el);
    }
  }
  bootEl.classList.add('done');
  setTimeout(() => bootEl.remove(), 900);
}

boot();
