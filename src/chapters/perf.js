// Gallery of the performance experiments.
import { PAL, SERIES } from '../core/colors.js';
import { galleryChapter } from '../core/gallery.js';
import { button, callout } from '../core/ui.js';

export const EXPERIMENTS = [
  { path: 'perf/quant', title: 'Quantization', sub: 'Fewer bits, same model?', icon: 'quant', color: PAL.coral, tag: 'BITS', blurb: 'Squeeze the real GPT from 16-bit floats down to 8, 4, 3 and 2 bits and measure what happens to its perplexity and size.' },
  { path: 'perf/kvcache', title: 'KV Cache', sub: 'Why generation is fast', icon: 'cache', color: PAL.cyan, tag: 'CACHE', blurb: 'Generate the same text with and without the key-value cache and time both. The gap grows with every token.' },
  { path: 'perf/batching', title: 'Batching', sub: 'One GPU, many users', icon: 'batch', color: PAL.mint, tag: 'SERVE', blurb: 'A simulated GPU serving a stream of requests: no batching, static, dynamic and continuous batching compared on throughput and latency.' },
  { path: 'perf/bakeoff', title: 'Model Bake-off', sub: 'No free lunch', icon: 'podium', color: PAL.amber, tag: 'ML', blurb: 'Six classic models trained live on five datasets. The winner changes with the shape of the problem.' },
  { path: 'perf/records', title: 'Hall of Records', sub: 'Published numbers', icon: 'records', color: PAL.violet, tag: 'DATA', blurb: 'Reference figures with sources: accuracy over the years, model sizes, context windows, price per token, today’s benchmark leaders.' },
  { path: 'labs/optim', title: 'Optimisers', sub: 'in the Optimizer Arena', icon: 'optim', color: SERIES[3], tag: 'LAB', blurb: 'Steps to converge for SGD, Momentum, RMSProp and Adam on five loss landscapes.' },
  { path: 'labs/embed', title: 'Vector indexes', sub: 'in Embedding Space', icon: 'embed', color: SERIES[6], tag: 'LAB', blurb: 'Exact search against IVF, HNSW and LSH: recall versus distance computations.' },
  { path: 'labs/diffusion', title: 'Samplers', sub: 'in the Diffusion Lab', icon: 'diffusion', color: SERIES[4], tag: 'LAB', blurb: 'DDPM, DDIM and flow-matching samplers: quality against number of steps.' },
  { path: 'labs/world', title: 'Search & RL', sub: 'in Plan vs Learn', icon: 'world', color: SERIES[2], tag: 'LAB', blurb: 'Nodes expanded by BFS, Dijkstra, greedy and A*; episodes needed by Q-learning, SARSA and Dyna-Q.' },
];

export async function enter(ctx) {
  return galleryChapter(ctx, {
    kicker: 'Chapter 05 · Performance arena',
    title: 'Same problem, different solutions',
    lead: 'Performance claims age badly, so most of this chapter is **measured live on your machine**: real algorithms, real timings, real error rates. Published reference numbers are kept separate and carry their sources.',
    items: EXPERIMENTS,
    rows: 2,
    footer: [
      callout('The absolute numbers depend on your device. The ratios between solutions are the lesson.', 'key'),
      button('Next: the ecosystem →', { kind: 'ghost', onClick: () => ctx.router.go('ecosystem') }),
    ],
  });
}
