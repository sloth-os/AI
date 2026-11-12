// Published reference numbers for the Hall of Records. Every row carries its source.
// Rows marked from the landscape snapshot (prices, benchmark leaders, model sizes, time horizon)
// were read from the linked pages on 12 Nov 2026; historical rows come from the original papers.
import { PRICES, BENCHMARKS, METR, FRONTIER, AS_OF } from './landscape.js';

const fromNote = (model, re) => {
  const f = FRONTIER.find((x) => x.model.startsWith(model));
  const m = f && f.note.match(re);
  return m ? { value: parseFloat(m[1]) * 1e12, src: f.src, when: f.date } : null;
};
const moe = [
  ['Mistral Large 4', /(\d\.\d+)T/],
  ['DeepSeek-V4', /(\d\.\d+)T/],
  ['Qwen3.8', /(\d\.\d+)T/],
  ['Kimi K3.1 Pro', /(\d\.\d+)T/],
]
  .map(([name, re]) => ({ name, ...fromNote(name, re) }))
  .filter((r) => r.value);

export const RECORDS = [
  {
    id: 'imagenet',
    title: 'ImageNet: error falls below human level',
    short: 'ImageNet error',
    unit: '% top-5 error',
    lowerIsBetter: true,
    why: 'The yearly ImageNet challenge (1,000 object classes) is where deep learning proved itself. In 2012 a convolutional network cut the error by ten points at once; three years later residual networks up to 152 layers deep were better than a trained human annotator. Each bar is that year’s winning entry.',
    rows: [
      { label: '2010', sub: 'hand-made features + SVM', value: 28.2, src: 'https://arxiv.org/abs/1409.0575' },
      { label: '2011', sub: 'Fisher vectors', value: 25.8, src: 'https://arxiv.org/abs/1409.0575' },
      { label: '2012', sub: 'AlexNet (CNN)', value: 15.3, mark: true, src: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html' },
      { label: '2013', sub: 'Clarifai (CNN)', value: 11.7, src: 'https://arxiv.org/abs/1409.0575' },
      { label: '2014', sub: 'GoogLeNet', value: 6.67, src: 'https://arxiv.org/abs/1409.4842' },
      { label: '2015', sub: 'ResNet (ensemble)', value: 3.57, mark: true, src: 'https://arxiv.org/abs/1512.03385' },
      { label: '2016', sub: 'ensemble', value: 2.99, src: 'https://image-net.org/challenges/LSVRC/2016/results' },
      { label: '2017', sub: 'SENet', value: 2.25, src: 'https://arxiv.org/abs/1709.01507' },
    ],
    reference: { label: 'a trained human: 5.1%', value: 5.1, src: 'https://arxiv.org/abs/1409.0575' },
    note: 'Top-5 error on the ILSVRC classification test set: the share of images whose true label is not among the model’s five guesses.',
  },
  {
    id: 'params',
    title: 'Parameters: a factor of fifty million in 28 years',
    short: 'Model size',
    unit: 'parameters',
    log: true,
    why: 'The recipe barely changed (layers of weights trained by gradient descent), the scale did. Each step right is a landmark model. The 2026 open-weight giants are mixtures of experts: they hold trillions of parameters but use only 50 to 100 billion of them for any one token.',
    rows: [
      { label: 'LeNet-5', sub: '1998', value: 60e3, src: 'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf' },
      { label: 'AlexNet', sub: '2012', value: 60e6, src: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html' },
      { label: 'Transformer', sub: '2017, big', value: 213e6, src: 'https://arxiv.org/abs/1706.03762' },
      { label: 'BERT-large', sub: '2018', value: 340e6, src: 'https://arxiv.org/abs/1810.04805' },
      { label: 'GPT-2', sub: '2019', value: 1.5e9, src: 'https://openai.com/index/better-language-models/' },
      { label: 'GPT-3', sub: '2020', value: 175e9, mark: true, src: 'https://arxiv.org/abs/2005.14165' },
      ...moe.map((m) => ({ label: m.name, sub: m.when + ', total', value: m.value, src: m.src })),
    ],
    reference: { label: 'the GPT running on this site: 1.07M', value: 1071872 },
    note: 'Logarithmic scale: each gridline is ten times the one below. Closed models do not publish their sizes.',
  },
  {
    id: 'context',
    title: 'Context windows: from a page to a library shelf',
    short: 'Context window',
    unit: 'tokens',
    log: true,
    why: 'The context window is everything a model can see at once: instructions, documents, the conversation, its own earlier output. It grew 500-fold in four years, which is what made long documents, whole code bases and long-running agents possible. A million tokens is roughly 750,000 words.',
    rows: [
      { label: 'GPT-3', sub: '2020', value: 2049, src: 'https://arxiv.org/abs/2005.14165' },
      { label: 'GPT-4', sub: 'Mar 2023', value: 8192, src: 'https://openai.com/index/gpt-4-research/' },
      { label: 'Claude 2', sub: 'Jul 2023', value: 100000, src: 'https://www.anthropic.com/news/claude-2' },
      { label: 'GPT-4 Turbo', sub: 'Nov 2023', value: 128000, src: 'https://openai.com/blog/new-models-and-developer-products-announced-at-devday' },
      { label: 'Claude 2.1', sub: 'Nov 2023', value: 200000, src: 'https://www.anthropic.com/news/claude-2-1' },
      { label: 'Gemini 1.5 Pro', sub: 'Feb 2024', value: 1000000, mark: true, src: 'https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/' },
      ...FRONTIER.filter((f) => f.ctx && ['OpenAI', 'Anthropic', 'Google DeepMind'].includes(f.lab)).map((f) => ({ label: f.model, sub: f.date, value: f.ctx, src: f.src })),
    ],
    note: 'Logarithmic scale. By late 2026 about a million tokens is the norm for flagship models.',
  },
  {
    id: 'price',
    title: 'Price: the same dollar buys far more tokens',
    short: 'Price per token',
    unit: '$ per million output tokens',
    log: true,
    lowerIsBetter: true,
    groups: { flagship: 'Flagship', mid: 'Mid-size', small: 'Small' },
    why: 'GPT-4 launched in March 2023 at $30 per million input tokens and $60 per million output tokens. Flagship models are still priced in that range, while the small tier now costs about a hundredth as much per token.',
    rows: PRICES.map((p) => ({ label: p.model, sub: p.when, value: p.out, value2: p.inp, group: p.tier, src: p.src, noteRow: p.note })),
    note: 'List prices, standard tier, at launch or as listed on the date shown; rows without a link are the vendor’s published launch price. Bars show the output price; the table also gives the input price. Logarithmic scale.',
  },
  {
    id: 'bench',
    title: `Benchmark leaders, ${AS_OF}`,
    short: 'Benchmark leaders',
    unit: '% solved',
    groups: Object.fromEntries(BENCHMARKS.map((b) => [b.name, b.name])),
    why: 'Benchmarks are how the field keeps score, and they keep running out. SWE-bench Verified (fixing real GitHub issues) and ARC-AGI-2 (abstract puzzles) are effectively solved; attention has moved to interactive tests such as ARC-AGI-3 and to operating a real computer.',
    rows: BENCHMARKS.flatMap((b) => b.rows.map(([model, v]) => ({ label: model, sub: b.name, value: v, group: b.name, src: b.src, noteRow: b.what + (b.note ? ' · ' + b.note : '') }))),
    note: 'Scores as reported on the linked pages. Harnesses differ, so compare models within one benchmark, not across benchmarks.',
  },
  {
    id: 'horizon',
    title: 'How long a task can an agent finish?',
    short: 'Agent time horizon',
    unit: 'hours of human work',
    why: `METR measures the length of software tasks (in the time a skilled human needs) that an AI agent completes on its own. The last model it measured, ${METR.latest.model} in ${METR.latest.when}, succeeds half the time on tasks of about ${(METR.latest.minutes50 / 60).toFixed(0)} hours, but needs the task to shrink to about ${(METR.latest.minutes80 / 60).toFixed(0)} hours to succeed four times out of five. ${METR.caveat}`,
    rows: [
      { label: '80% reliable', sub: METR.latest.model, value: METR.latest.minutes80 / 60, src: METR.src },
      { label: '50% reliable', sub: METR.latest.model, value: METR.latest.minutes50 / 60, mark: true, src: METR.src },
      { label: 'low estimate', sub: '95% interval, 50%', value: METR.latest.ci[0] / 60, src: METR.src },
      { label: 'high estimate', sub: '95% interval, 50%', value: METR.latest.ci[1] / 60, src: METR.src },
    ],
    note: 'The confidence interval is very wide: the benchmark ran out of tasks long enough to measure with.',
  },
  {
    id: 'memory',
    title: 'Memory: what it takes to hold the weights',
    short: 'Memory by precision',
    unit: 'GB',
    log: true,
    groups: { 16: '16-bit', 8: '8-bit', 4: '4-bit' },
    why: 'Before a model can answer anything its weights must fit in memory. This is plain arithmetic: parameters × bits ÷ 8. It is why 4-bit quantization decides whether a model runs on a laptop, one GPU or a cluster.',
    rows: [7e9, 70e9, 1e12].flatMap((p) => [16, 8, 4].map((bits) => ({ label: (p >= 1e12 ? p / 1e12 + 'T' : p / 1e9 + 'B') + ' params', sub: bits + '-bit', value: (p * bits) / 8 / 1e9, group: String(bits) }))),
    note: 'Computed, weights only. Running the model needs more: activations and the KV cache. Logarithmic scale.',
    link: { href: '#/perf/quant', label: 'Measure quantization yourself' },
  },
];
