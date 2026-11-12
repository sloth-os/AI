// Route table and top navigation. Every route lazy-loads one module that exports
//   async function enter(ctx) -> { update?(dt, t), onRoute?(route), dispose?() }
// The longest matching path prefix wins; whatever is left over arrives as ctx.route.rest.
import { PAL } from './core/colors.js';

export const NAV = [
  { id: 'hub', path: '', label: 'Universe', accent: PAL.cyan },
  { id: 'families', path: 'families', label: 'Families', accent: PAL.violet },
  { id: 'labs', path: 'labs', label: 'Labs', accent: PAL.cyan },
  { id: 'problems', path: 'problems', label: 'Problems', accent: PAL.amber },
  { id: 'flow', path: 'flow', label: 'Data Flow', accent: PAL.mint },
  { id: 'perf', path: 'perf', label: 'Performance', accent: PAL.coral },
  { id: 'ecosystem', path: 'ecosystem', label: 'Ecosystem', accent: PAL.blue },
  { id: 'deploy', path: 'deploy', label: 'Deploy', accent: PAL.lime },
  { id: 'hot', path: 'hot', label: 'Hot 10', accent: PAL.magenta },
];

const accentOf = (nav) => NAV.find((n) => n.id === nav).accent;
const route = (path, nav, title, load, accent) => ({ path, nav, title, load, accent: accent ?? accentOf(nav) });

export const ROUTES = [
  route('', 'hub', 'AI Atlas', () => import('./chapters/hub.js')),
  route('families', 'families', 'Families of AI', () => import('./chapters/families.js')),
  route('labs', 'labs', 'Mechanism Labs', () => import('./chapters/labs.js')),
  route('labs/gpt', 'labs', 'Inside a GPT', () => import('./labs/gpt.js'), PAL.cyan),
  route('labs/neural', 'labs', 'Neural Network Lab', () => import('./labs/neural.js'), PAL.mint),
  route('labs/optim', 'labs', 'Optimizer Arena', () => import('./labs/optim.js'), PAL.amber),
  route('labs/cnn', 'labs', 'How a CNN Sees', () => import('./labs/cnn.js'), PAL.blue),
  route('labs/diffusion', 'labs', 'Diffusion Lab', () => import('./labs/diffusion.js'), PAL.magenta),
  route('labs/world', 'labs', 'Plan vs Learn', () => import('./labs/world.js'), PAL.lime),
  route('labs/embed', 'labs', 'Embedding Space', () => import('./labs/embed.js'), PAL.violet),
  route('problems', 'problems', 'Problem Space', () => import('./chapters/problems.js')),
  route('flow', 'flow', 'Data Flow', () => import('./chapters/flow.js')),
  route('perf', 'perf', 'Performance Arena', () => import('./chapters/perf.js')),
  route('perf/quant', 'perf', 'Quantization', () => import('./perf/quant.js')),
  route('perf/kvcache', 'perf', 'KV Cache', () => import('./perf/kvcache.js')),
  route('perf/batching', 'perf', 'Batching', () => import('./perf/batching.js')),
  route('perf/bakeoff', 'perf', 'Model Bake-off', () => import('./perf/bakeoff.js')),
  route('perf/records', 'perf', 'Hall of Records', () => import('./perf/records.js')),
  route('ecosystem', 'ecosystem', 'Project Galaxy', () => import('./chapters/ecosystem.js')),
  route('deploy', 'deploy', 'Deployment Modes', () => import('./chapters/deploy.js')),
  route('hot', 'hot', 'Hot 10', () => import('./chapters/hot.js')),
];

/** @returns {{def: typeof ROUTES[number], rest: string[]} | null} */
export function matchRoute(parts) {
  for (let n = parts.length; n >= 0; n--) {
    const path = parts.slice(0, n).join('/');
    const def = ROUTES.find((r) => r.path === path);
    if (def) return { def, rest: parts.slice(n) };
  }
  return null;
}
