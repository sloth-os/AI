// Gallery of the mechanism labs.
import { PAL } from '../core/colors.js';
import { galleryChapter } from '../core/gallery.js';
import { button } from '../core/ui.js';

export const LABS = [
  { path: 'labs/gpt', title: 'Inside a GPT', sub: 'A real transformer, token by token', icon: 'gpt', color: PAL.cyan, tag: 'LLM', blurb: 'A 1M-parameter GPT trained on children’s stories runs in your browser. Type a prompt and follow it through tokens, embeddings, attention and the next-token lottery.' },
  { path: 'labs/neural', title: 'Neural Network Lab', sub: 'Backpropagation, live', icon: 'network', color: PAL.mint, tag: 'MLP', blurb: 'Train a small network on spirals and rings. Watch the forward pass, the gradients flowing back, and the decision surface bending to fit.' },
  { path: 'labs/optim', title: 'Optimizer Arena', sub: 'Five optimisers race downhill', icon: 'optim', color: PAL.amber, tag: 'SGD', blurb: 'SGD, Momentum, RMSProp and Adam start from the same point on the same loss landscape. See who gets stuck, who overshoots and who wins.' },
  { path: 'labs/cnn', title: 'How a CNN Sees', sub: 'Draw a digit, follow the filters', icon: 'conv', color: PAL.blue, tag: 'CNN', blurb: 'A convolutional network trained on handwritten digits. Draw one and watch every feature map light up on the way to the answer.' },
  { path: 'labs/diffusion', title: 'Diffusion Lab', sub: 'Noise becomes shape', icon: 'diffusion', color: PAL.magenta, tag: 'DIFF', blurb: 'Thousands of particles start as pure noise and are denoised into a shape. Compare samplers, step counts and guidance strength.' },
  { path: 'labs/world', title: 'Plan vs Learn', sub: 'Search and reinforcement learning', icon: 'world', color: PAL.lime, tag: 'RL', blurb: 'One maze, two philosophies: A* plans with a map; Q-learning discovers the way by trial, error and reward.' },
  { path: 'labs/embed', title: 'Embedding Space', sub: 'Meaning as geometry', icon: 'embed', color: PAL.violet, tag: 'VEC', blurb: 'Real sentence embeddings laid out in 3D. Click a word to see its neighbours, then race the vector indexes that make search fast.' },
];

export async function enter(ctx) {
  return galleryChapter(ctx, {
    kicker: 'Chapter 02 · Mechanism labs',
    title: 'Open the machines',
    lead: 'Seven working models. Each one is small enough to run on this page and real enough to behave like the big ones: same algorithms, same failure modes, about a million times fewer parameters.',
    items: LABS,
    rows: 2,
    footer: button('Next: the problem space →', { kind: 'ghost', onClick: () => ctx.router.go('problems') }),
  });
}
