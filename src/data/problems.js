// The problem space of AI: eight levels of the stack, the canonical problems at each level and the
// competing solutions to each problem. Used by src/chapters/problems.js.
//
// Two kinds of content live here and they must not be confused:
//
//  * `r`    Ratings from 1 (weak) to 5 (strong) on the six ANGLES, in ANGLES order. These are
//           EDITORIAL JUDGEMENTS made for this atlas, relative to the other solutions of the same
//           problem. They exist to build intuition about trade-offs; they are not measurements.
//  * `fact` One published figure or result, quoted from the source named in the text; `src` is its
//           URL. Checked against the source on 12 Nov 2026. Where no single figure could be stated
//           with confidence the field is left empty rather than guessed. Facts about the current
//           state of the field (marked `asOf`) repeat the dated snapshot in ./landscape.js.
//
// `how` is the mechanism in one sentence, `when` says when to choose the solution, `link` points at
// the part of the atlas that shows the mechanism running.

export const PROBLEMS_AS_OF = '12 Nov 2026';

/** The six requirement angles. Order matters: every `r` array follows it. */
export const ANGLES = [
  { id: 'quality', name: 'Quality', short: 'Quality', desc: 'How good the result is: accuracy, fidelity, usefulness.' },
  { id: 'speed', name: 'Speed', short: 'Speed', desc: 'Low latency: how quickly it answers or delivers a result.' },
  { id: 'cost', name: 'Cost efficiency', short: 'Cost', desc: 'Result per dollar of compute, memory and labor.' },
  { id: 'scale', name: 'Scalability', short: 'Scale', desc: 'How well it holds up with more data, more users or a bigger model.' },
  { id: 'simple', name: 'Simplicity', short: 'Simplicity', desc: 'How little there is to build, tune and operate.' },
  { id: 'control', name: 'Control', short: 'Control', desc: 'Privacy, customization and transparency: how much stays in your hands.' },
];

/** Weight words for the lens sliders (0-3). */
export const WEIGHT_WORDS = ['ignore', 'minor', 'important', 'critical'];

/** Requirement lenses: one weight (0-3) per angle, in ANGLES order. Editorial, like the ratings. */
export const PRESETS = [
  { id: 'general', name: 'General purpose', w: [2, 1, 1, 1, 1, 1], note: 'Quality counts double; the other five angles count the same.' },
  { id: 'mvp', name: 'Startup MVP', w: [2, 1, 2, 0, 3, 1], note: 'Two engineers and a deadline: simple and cheap come first.' },
  { id: 'realtime', name: 'Real-time product', w: [2, 3, 1, 2, 0, 0], note: 'A user waits on every response: latency comes first.' },
  { id: 'regulated', name: 'Regulated enterprise', w: [3, 0, 0, 1, 1, 3], note: 'Auditors and private data: control and correctness come first.' },
  { id: 'frontier', name: 'Research frontier', w: [3, 0, 0, 2, 0, 1], note: 'The best possible result at the largest scale, whatever it takes.' },
  { id: 'edge', name: 'Edge device', w: [2, 3, 3, 0, 1, 2], note: 'A phone or a sensor: fast, cheap and private on small hardware.' },
  { id: 'webscale', name: 'Web scale', w: [2, 2, 3, 3, 0, 1], note: 'Millions of users: cost per request and scalability come first.' },
];

/**
 * Score of one solution under a lens: the weighted geometric mean of its six ratings (1-5).
 * A geometric mean is used so that one very weak angle you care about cannot be bought back by
 * strength elsewhere. With all weights at zero every angle counts the same.
 */
export function lensScore(r, w) {
  let sum = 0;
  let acc = 0;
  for (let i = 0; i < r.length; i++) {
    sum += w[i];
    acc += w[i] * Math.log(r[i]);
  }
  if (sum <= 0) {
    acc = 0;
    for (let i = 0; i < r.length; i++) acc += Math.log(r[i]);
    sum = r.length;
  }
  return Math.exp(acc / sum);
}

const EVEN = [1, 1, 1, 1, 1, 1];

/**
 * Solutions of a problem ordered best first under a lens: [{ sol, index, score }].
 * Exact ties go to the better all-rounder (the score with equal weights), then to data order.
 */
export function rankSolutions(problem, w) {
  return problem.solutions
    .map((sol, index) => ({ sol, index, score: lensScore(sol.r, w), even: lensScore(sol.r, EVEN) }))
    .sort((a, b) => (Math.abs(b.score - a.score) > 1e-9 ? b.score - a.score : b.even - a.even || a.index - b.index));
}

/** Levels of the stack, bottom (1) to top (8). */
export const LEVELS = [
  // ------------------------------------------------------------------------------------ 1 · Data
  {
    id: 'data',
    name: 'Data',
    desc: 'Everything a model knows comes from data: collecting it, labeling it, encoding it and moving it.',
    problems: [
      {
        id: 'labels',
        name: 'Too few labeled examples',
        short: 'Too few labels',
        statement: 'Supervised learning needs examples with the right answer attached, and you have far fewer of them than the model needs.',
        hard: 'Labels cost expert time, annotators disagree, and the rare cases that matter most are the ones you have least of.',
        yardstick: 'Quality is the accuracy of the model you end up with; Speed is how soon you have usable training data.',
        solutions: [
          {
            id: 'manual', name: 'Manual labeling', r: [5, 1, 1, 1, 4, 5],
            how: 'People annotate every example by hand, with guidelines and several annotators per item to settle disagreements.',
            fact: 'ImageNet: more than 14 million images labeled by hand through crowdsourcing (Russakovsky et al., 2015).',
            src: 'https://arxiv.org/abs/1409.0575',
            when: 'High-stakes domains where every label must be defensible.',
          },
          {
            id: 'weak', name: 'Weak supervision', r: [3, 4, 4, 4, 2, 4],
            how: 'Write labeling functions (rules, heuristics, other models); a label model estimates their accuracies and combines their noisy votes.',
            fact: 'Snorkel user study: experts built models 2.8× faster and 45.5% better than with seven hours of hand labeling (Ratner et al., 2017).',
            src: 'https://arxiv.org/abs/1711.10160',
            when: 'Experts can state rules faster than they can label examples.',
          },
          {
            id: 'ssl', name: 'Self-supervised pretraining', r: [4, 2, 2, 5, 2, 3],
            how: 'Pretrain on unlabeled data with an invented task, such as predicting a hidden piece, then fine-tune on the few labels.',
            fact: 'SimCLR: 85.8% ImageNet top-5 accuracy after fine-tuning on only 1% of the labels (Chen et al., 2020).',
            src: 'https://arxiv.org/abs/2002.05709',
            when: 'Unlabeled data is plentiful and you can afford the pretraining.',
          },
          {
            id: 'synthetic', name: 'Synthetic data', r: [3, 4, 4, 5, 3, 4],
            how: 'Generate training examples with a simulator or a generative model, then filter out the bad ones.',
            fact: 'A detector trained only on simulated images with random textures located real objects to within 1.5 cm (Tobin et al., 2017).',
            src: 'https://arxiv.org/abs/1703.06907',
            when: 'Real data is scarce, private or dangerous to collect.',
          },
          {
            id: 'active', name: 'Active learning', r: [4, 2, 4, 2, 2, 4],
            how: 'Train on a small labeled set, let the model pick the examples it is least sure about, label those, repeat.',
            fact: '',
            when: 'Labeling is the bottleneck and annotators are available on demand.',
          },
          {
            id: 'fewshot', name: 'Few-shot foundation model', r: [4, 5, 3, 4, 5, 2],
            how: 'Skip training: describe the classes or show a handful of examples to a large pretrained model and let it label.',
            fact: 'CLIP matched the original ResNet-50 on ImageNet zero-shot, using none of its 1.28 million training examples (Radford et al., 2021).',
            src: 'https://arxiv.org/abs/2103.00020',
            when: 'The task resembles what large pretrained models have already seen.',
          },
        ],
      },
      {
        id: 'tokens',
        name: 'Turning text into numbers',
        short: 'Text into numbers',
        statement: 'A network only does arithmetic, so text must be cut into pieces and each piece mapped to a number.',
        hard: 'Small pieces make sequences long and slow; large pieces make the vocabulary huge and leave new words unknown.',
        yardstick: 'Quality is how well models trained on the pieces perform; Speed and Cost follow the number of pieces per sentence.',
        solutions: [
          {
            id: 'char', name: 'Characters', r: [2, 2, 2, 3, 5, 4],
            how: 'One token per character: a tiny vocabulary and no unknown words, but sequences several times longer.',
            fact: 'A character-aware language model matched the state of the art on Penn Treebank with 60% fewer parameters (Kim et al., 2015).',
            src: 'https://arxiv.org/abs/1508.06615',
            when: 'Small experiments, or tasks about spelling itself.',
          },
          {
            id: 'word', name: 'Words', r: [2, 4, 3, 1, 4, 3],
            how: 'Split on spaces and punctuation; each distinct word gets an id and anything unseen becomes an “unknown” token.',
            fact: 'The released word2vec vectors needed a vocabulary of 3 million words and phrases (Mikolov et al., 2013).',
            src: 'https://code.google.com/archive/p/word2vec/',
            when: 'Classic search and statistics pipelines over one well-behaved language.',
          },
          {
            id: 'bpe', name: 'Subword BPE', r: [5, 4, 4, 4, 4, 3],
            how: 'Start from characters and repeatedly merge the most frequent adjacent pair until the vocabulary reaches a chosen size.',
            fact: 'Subword units beat a back-off dictionary baseline by 1.1 and 1.3 BLEU on two WMT 15 translation tasks (Sennrich et al., 2015).',
            src: 'https://arxiv.org/abs/1508.07909',
            when: 'The default for language models: common words stay whole, rare ones split.',
          },
          {
            id: 'unigram', name: 'SentencePiece unigram', r: [5, 4, 4, 5, 3, 3],
            how: 'Learn subwords directly from raw sentences, treating the space as a normal symbol, so no language-specific word splitter is needed.',
            fact: 'T5 covers English, German, French and Romanian with one SentencePiece vocabulary of 32,000 pieces (Raffel et al., 2019).',
            src: 'https://arxiv.org/abs/1910.10683',
            when: 'Multilingual text, including languages written without spaces.',
          },
          {
            id: 'bytes', name: 'Raw bytes', r: [3, 1, 2, 5, 4, 4],
            how: 'Feed UTF-8 bytes directly: 256 values cover every language and every typo, at the price of much longer sequences.',
            fact: 'ByT5: byte-level models were competitive with token-level ones and significantly more robust to noise (Xue et al., 2021).',
            src: 'https://arxiv.org/abs/2105.13626',
            when: 'Noisy or mixed-language text where robustness beats speed.',
          },
        ],
      },
      {
        id: 'pipeline',
        name: 'Moving training data at scale',
        short: 'Moving data at scale',
        statement: 'Training reads the dataset many times over, and the storage format decides whether expensive GPUs wait for input.',
        hard: 'Parsing text is slow, billions of small files overwhelm storage, and shuffling wants random access to everything.',
        yardstick: 'Quality is fidelity (typed values, no silent corruption); Speed is samples per second delivered to the trainer.',
        solutions: [
          {
            id: 'csv', name: 'CSV / JSON lines', r: [2, 1, 2, 1, 5, 5],
            how: 'Plain text rows: readable by anything, but every value is parsed from text on every read and nothing can be skipped.',
            fact: '',
            when: 'Small datasets, hand inspection and exchange between tools.',
          },
          {
            id: 'parquet', name: 'Parquet / Arrow', r: [5, 4, 5, 4, 3, 4],
            how: 'Store each column together, typed and compressed; readers load only the columns and row groups they need.',
            fact: 'Dremel, the columnar design Parquet follows, ran aggregation queries over trillion-row tables in seconds (Melnik et al., 2010).',
            src: 'https://research.google/pubs/dremel-interactive-analysis-of-web-scale-datasets-2/',
            when: 'Tabular or text data that is also filtered and analyzed.',
          },
          {
            id: 'shards', name: 'WebDataset / TFRecord shards', r: [4, 5, 4, 5, 3, 3],
            how: 'Pack samples into large sequential shard files; workers read different shards in parallel and shuffle within a buffer.',
            fact: 'Scale reference: LAION-5B holds 5.85 billion image-text pairs (Schuhmann et al., 2022).',
            src: 'https://arxiv.org/abs/2210.08402',
            when: 'Images, audio or video at a scale of millions of samples and up.',
          },
          {
            id: 'mmap', name: 'Memory-mapped binary', r: [4, 5, 5, 3, 3, 3],
            how: 'Write samples once into one binary file with a fixed layout; the loader maps it into memory and jumps to any sample.',
            fact: 'FFCV, built on such a format, trained ResNet-50 on ImageNet to 75% accuracy in 20 minutes on one machine (Leclerc et al., 2023).',
            src: 'https://arxiv.org/abs/2306.12517',
            when: 'The dataset fits on local disks and loading speed limits training.',
          },
          {
            id: 'stream', name: 'Streaming from object storage', r: [4, 3, 4, 5, 3, 2],
            how: 'Read shards over the network while training, with no local copy: training starts at once and disks never fill up.',
            fact: '',
            when: 'Datasets larger than any local disk, or shared across many machines.',
          },
        ],
      },
    ],
  },

  // --------------------------------------------------------------------------------- 2 · Compute
  {
    id: 'compute',
    name: 'Compute',
    desc: 'The arithmetic has to run somewhere: chips, memory, and the software that spreads the work across them.',
    problems: [
      {
        id: 'hardware',
        name: 'Where to run the math',
        short: 'Where to run the math',
        statement: 'A model is billions of multiply-adds per answer. The chip that runs them sets the speed, the price and where the model can live.',
        hard: 'Chips trade flexibility for efficiency, the software stack matters as much as the silicon, and memory bandwidth often limits before raw compute does.',
        yardstick: 'Quality is how large a model the option runs well; Control includes whether data can stay on hardware you own.',
        solutions: [
          {
            id: 'cpu', name: 'CPU', r: [2, 1, 3, 2, 5, 5],
            how: 'General-purpose cores with vector instructions: runs any model with no special driver, limited by memory bandwidth.',
            fact: '',
            when: 'Small models, low traffic, or hardware you already own.',
          },
          {
            id: 'gpu', name: 'GPU', r: [5, 4, 3, 5, 4, 4],
            how: 'Thousands of simple cores run the same matrix operation in parallel, backed by a software stack every framework supports.',
            fact: 'AlexNet trained for five to six days on two GTX 580 GPUs and won ImageNet 2012 (Krizhevsky et al., 2012).',
            src: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html',
            when: 'The default for training and for serving most models.',
          },
          {
            id: 'asic', name: 'Custom ASIC (TPU-style)', r: [5, 4, 4, 5, 2, 2],
            how: 'A chip built only for tensor math: it drops general-purpose features to fit more matrix multipliers per watt.',
            fact: 'Google’s first TPU ran inference about 15–30× faster than contemporary CPUs and GPUs, with 30–80× better performance per watt (Jouppi et al., 2017).',
            src: 'https://arxiv.org/abs/1704.04760',
            when: 'Huge, stable workloads inside one cloud that justify custom silicon.',
          },
          {
            id: 'npu', name: 'On-device NPU', r: [2, 4, 5, 2, 2, 5],
            how: 'A small low-power accelerator inside a phone or laptop chip; the model runs locally and data never leaves the device.',
            fact: 'Apple’s A11 Bionic shipped a neural engine rated at up to 600 billion operations per second (Apple, 2017).',
            src: 'https://www.apple.com/newsroom/2017/09/the-future-is-here-iphone-x/',
            when: 'Private, offline or battery-powered features with small models.',
          },
          {
            id: 'wafer', name: 'Wafer-scale / inference chips', r: [4, 5, 2, 3, 2, 2],
            how: 'Use an entire wafer as one chip, or a design tuned purely for token generation, to remove the memory bottleneck.',
            fact: 'Cerebras WSE-3 puts 4 trillion transistors on a single wafer-sized chip (Cerebras, 2024).',
            src: 'https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine',
            when: 'Latency-critical serving where tokens per second is the product.',
          },
        ],
      },
      {
        id: 'precision',
        name: 'The model does not fit in memory',
        short: 'Model too big for memory',
        statement: 'Every weight is a number that takes up memory. At 32 bits each, 7 billion weights need 28 GB before a single token is processed.',
        hard: 'Fewer bits per weight means rounding error. The art is to drop bits without dropping accuracy.',
        yardstick: 'Quality is how closely outputs match the full-precision model; Cost is memory per weight.',
        solutions: [
          {
            id: 'fp32', name: 'FP32', r: [5, 2, 1, 1, 5, 5],
            how: 'Store every weight as a 32-bit float, the classic training format, with no rounding error added.',
            fact: 'IEEE 754 single precision carries a 24-bit significand, about 7 decimal digits (IEEE 754-2008).',
            when: 'Reference runs and numerically delicate research.',
          },
          {
            id: 'fp16', name: 'FP16 / BF16', r: [5, 4, 3, 3, 4, 5],
            how: 'Halve the storage with 16-bit floats; mixed-precision training keeps a 32-bit copy where rounding would hurt.',
            fact: 'Mixed-precision training cut memory use by nearly 2× across CNNs, RNNs and GANs (Micikevicius et al., 2017).',
            src: 'https://arxiv.org/abs/1710.03740',
            when: 'The standard for training and for unquantized serving.',
          },
          {
            id: 'int8', name: 'INT8', r: [4, 4, 4, 4, 3, 4],
            how: 'Map each weight to one of 256 integer levels plus a scale factor; integer math is fast on almost every chip.',
            fact: 'LLM.int8(): models up to 175B parameters ran in 8-bit with no measured loss, halving inference memory (Dettmers et al., 2022).',
            src: 'https://arxiv.org/abs/2208.07339',
            when: 'A safe first step: half the memory of FP16 for little risk.',
          },
          {
            id: 'q4', name: '4-bit', r: [4, 4, 5, 5, 3, 4],
            how: 'Sixteen levels per weight, placed per block with calibration data or learned in training so errors land where they matter least.',
            fact: 'GPTQ quantized a 175-billion-parameter model to 3–4 bits per weight in about four GPU hours with negligible accuracy loss (Frantar et al., 2022).',
            src: 'https://arxiv.org/abs/2210.17323',
            when: 'Running large models on one GPU, a laptop or a browser.',
            link: { href: '#/perf/quant', label: 'Measure quantization live' },
          },
          {
            id: 'ternary', name: 'Ternary (1.58-bit)', r: [3, 5, 5, 4, 1, 2],
            how: 'Train from scratch with every weight restricted to −1, 0 or +1, so multiplications turn into additions.',
            fact: 'BitNet b1.58 matched full-precision models of the same size and training tokens in perplexity and end-task performance (Ma et al., 2024).',
            src: 'https://arxiv.org/abs/2402.17764',
            when: 'New models designed for extreme efficiency from day one.',
          },
        ],
      },
      {
        id: 'parallel',
        name: 'One GPU is not enough',
        short: 'One GPU is not enough',
        statement: 'The model, or the data it has to learn from, is too big for one accelerator, so training must be split across many.',
        hard: 'Every split adds communication. The question is what to copy, what to slice and what to send over the wire.',
        yardstick: 'Quality is how large a model the scheme can train; Speed is throughput per GPU after communication overhead.',
        solutions: [
          {
            id: 'dp', name: 'Data parallel', r: [2, 5, 4, 3, 5, 4],
            how: 'Copy the whole model to every GPU, give each a different slice of the batch and average the gradients.',
            fact: 'ResNet-50 trained on 256 GPUs in one hour with a batch of 8,192 and no loss of accuracy (Goyal et al., 2017).',
            src: 'https://arxiv.org/abs/1706.02677',
            when: 'The model fits on one GPU and you just need more throughput.',
          },
          {
            id: 'tp', name: 'Tensor parallel', r: [4, 3, 3, 3, 2, 3],
            how: 'Split each weight matrix across GPUs so every layer is computed jointly, which needs very fast links between them.',
            fact: 'Megatron-LM trained an 8.3-billion-parameter transformer on 512 GPUs at 76% scaling efficiency (Shoeybi et al., 2019).',
            src: 'https://arxiv.org/abs/1909.08053',
            when: 'Single layers are too large for one GPU; GPUs share a fast interconnect.',
          },
          {
            id: 'pp', name: 'Pipeline parallel', r: [4, 3, 3, 4, 2, 3],
            how: 'Place consecutive layers on different GPUs and pass activations down the line, with micro-batches keeping every stage busy.',
            fact: 'GPipe trained a 557-million-parameter AmoebaNet to 84.4% ImageNet top-1 with almost linear speed-up (Huang et al., 2018).',
            src: 'https://arxiv.org/abs/1811.06965',
            when: 'Deep models across machines with slower links between them.',
          },
          {
            id: 'zero', name: 'Sharded state (FSDP / ZeRO)', r: [4, 4, 4, 4, 3, 3],
            how: 'Keep data parallelism but shard parameters, gradients and optimizer state across GPUs, gathering each layer only when needed.',
            fact: 'ZeRO trained models of over 100 billion parameters on 400 GPUs with super-linear speed-up (Rajbhandari et al., 2019).',
            src: 'https://arxiv.org/abs/1910.02054',
            when: 'The model no longer fits on one GPU and you want few code changes.',
          },
          {
            id: 'combined', name: 'Combined (3D) parallelism', r: [5, 4, 3, 5, 1, 2],
            how: 'Combine data, tensor and pipeline splits so each handles the axis it is best at.',
            fact: 'Megatron-LM ran training iterations on a 1-trillion-parameter model at 502 petaFLOP/s on 3,072 GPUs (Narayanan et al., 2021).',
            src: 'https://arxiv.org/abs/2104.04473',
            when: 'Frontier-scale runs on thousands of GPUs with a dedicated team.',
          },
        ],
      },
    ],
  },

  // ---------------------------------------------------------------------------- 3 · Architecture
  {
    id: 'arch',
    name: 'Architecture',
    desc: 'The wiring of the network: it decides which patterns can be learned, and at what cost.',
    problems: [
      {
        id: 'sequences',
        name: 'Modeling sequences',
        short: 'Modeling sequences',
        statement: 'Language, audio, code and DNA are sequences: what each element means depends on what came before, sometimes long before.',
        hard: 'Long-range dependencies need memory of distant items, but comparing every item with every other costs quadratic time.',
        yardstick: 'Quality is modeling accuracy on long sequences; Speed is the cost of processing each additional token.',
        solutions: [
          {
            id: 'ngram', name: 'n-gram / HMM', r: [1, 5, 5, 2, 5, 5],
            how: 'Count how often short runs of words occur and predict the next word from the last few.',
            fact: 'Google’s Web 1T corpus holds n-gram counts from about one trillion words of web text (Brants & Franz, 2006).',
            src: 'https://catalog.ldc.upenn.edu/LDC2006T13',
            when: 'Tiny devices, keyboards, or a transparent baseline.',
          },
          {
            id: 'rnn', name: 'RNN / LSTM', r: [3, 2, 3, 2, 3, 3],
            how: 'Read one token at a time and carry a hidden state forward; gates decide what to keep and what to forget.',
            fact: 'Google’s LSTM translation system cut translation errors by about 60% against its phrase-based predecessor (Wu et al., 2016).',
            src: 'https://arxiv.org/abs/1609.08144',
            when: 'Streaming signals with small models and strict memory limits.',
          },
          {
            id: 'transformer', name: 'Transformer', r: [5, 3, 2, 5, 4, 3],
            how: 'Every token attends to every other token, all positions in parallel; cost grows with the square of the length.',
            fact: '28.4 BLEU on WMT 2014 English-German, more than 2 BLEU above the previous best (Vaswani et al., 2017).',
            src: 'https://arxiv.org/abs/1706.03762',
            when: 'The default whenever quality matters and compute is available.',
            link: { href: '#/labs/gpt', label: 'Open a real GPT' },
          },
          {
            id: 'ssm', name: 'State-space (Mamba)', r: [3, 5, 4, 4, 2, 3],
            how: 'Compress the past into a fixed-size state updated by input-dependent dynamics; time and memory grow linearly with length.',
            fact: 'Mamba: 5× the inference throughput of Transformers; Mamba-3B matched Transformers twice its size (Gu & Dao, 2023).',
            src: 'https://arxiv.org/abs/2312.00752',
            when: 'Very long sequences such as audio, genomics or logs.',
          },
          {
            id: 'hybrid', name: 'Hybrid linear + full attention', r: [4, 4, 4, 5, 2, 3],
            how: 'Interleave cheap linear-time layers with occasional full-attention layers: most of the speed, with exact lookups where needed.',
            fact: 'Qwen3.8-27B repeats three linear-time Gated DeltaNet layers and one full-attention layer 16 times (model card, 2026).',
            src: 'https://huggingface.co/Qwen/Qwen3.8-27B',
            asOf: true,
            when: 'Long contexts where pure attention is too costly.',
          },
        ],
      },
      {
        id: 'vision',
        name: 'Seeing images',
        short: 'Seeing images',
        statement: 'An image is a grid of numbers. The same object can appear anywhere in it, at any size, in any light.',
        hard: 'The model must ignore position and lighting yet notice the small differences that separate one class from another.',
        yardstick: 'Quality is accuracy on standard benchmarks; Cost includes the data and compute needed for training.',
        solutions: [
          {
            id: 'handcrafted', name: 'Hand-crafted features + SVM', r: [2, 4, 5, 2, 2, 5],
            how: 'Engineers design descriptors of edges and gradients (SIFT, HOG); a linear classifier separates the resulting vectors.',
            fact: 'The best ImageNet 2012 entry without a deep network had 26.2% top-5 error; AlexNet had 15.3% (Krizhevsky et al., 2012).',
            src: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html',
            when: 'Fixed industrial settings with little data and strict audit needs.',
          },
          {
            id: 'cnn', name: 'CNN', r: [4, 5, 4, 4, 4, 4],
            how: 'Slide small learned filters over the image; stacked layers build edges into textures into objects, sharing weights across positions.',
            fact: 'ResNet: 3.57% top-5 error on ImageNet with an ensemble of residual networks up to 152 layers deep (He et al., 2015).',
            src: 'https://arxiv.org/abs/1512.03385',
            when: 'Limited data or compute, and real-time or embedded vision.',
            link: { href: '#/labs/cnn', label: 'Watch a CNN see' },
          },
          {
            id: 'vit', name: 'Vision Transformer', r: [5, 3, 2, 5, 3, 3],
            how: 'Cut the image into patches, treat each patch as a token and let attention relate every patch to every other.',
            fact: 'ViT reached 88.55% ImageNet top-1 after pretraining on 300 million images (Dosovitskiy et al., 2020).',
            src: 'https://arxiv.org/abs/2010.11929',
            when: 'Very large pretraining sets, or one backbone shared with text.',
          },
          {
            id: 'convnext', name: 'Modernized CNN (ConvNeXt)', r: [5, 4, 3, 4, 4, 4],
            how: 'Keep convolutions but adopt the Transformer’s design and training recipe: large kernels, fewer activations, layer normalization.',
            fact: 'ConvNeXt: 87.8% ImageNet top-1, ahead of Swin Transformers on COCO detection and ADE20K segmentation (Liu et al., 2022).',
            src: 'https://arxiv.org/abs/2201.03545',
            when: 'Transformer-level accuracy with convolutional efficiency and tooling.',
          },
          {
            id: 'clip', name: 'Pretrained image-text encoder', r: [4, 4, 4, 5, 5, 2],
            how: 'Reuse a model trained to match images with captions; classify by comparing an image with text descriptions of each class.',
            fact: 'CLIP was trained on 400 million image-text pairs collected from the internet (Radford et al., 2021).',
            src: 'https://arxiv.org/abs/2103.00020',
            when: 'New categories without labeled examples, or search across images and text.',
          },
        ],
      },
      {
        id: 'generation',
        name: 'Generating media',
        short: 'Generating media',
        statement: 'Produce new images, audio or video that look like the training data without copying it.',
        hard: 'The model must learn a distribution over millions of dimensions, cover all of its variety and still sample quickly.',
        yardstick: 'Quality is fidelity and diversity of the samples; Speed is the time to generate one sample.',
        solutions: [
          {
            id: 'gan', name: 'GAN', r: [3, 5, 4, 2, 2, 3],
            how: 'A generator makes samples and a discriminator tries to tell them from real data; each trains against the other.',
            fact: 'StyleGAN2: FID 2.84 on 1024×1024 FFHQ faces, each generated in one forward pass (Karras et al., 2019).',
            src: 'https://arxiv.org/abs/1912.04958',
            when: 'Narrow domains such as faces that need real-time generation.',
          },
          {
            id: 'vae', name: 'VAE', r: [2, 5, 5, 3, 4, 4],
            how: 'Encode data into a smooth low-dimensional latent space and decode samples from it, trained with one stable objective.',
            fact: 'Stable Diffusion’s autoencoder shrinks each image side 8-fold: 512×512 pixels become a 64×64 latent (Rombach et al., 2022).',
            src: 'https://arxiv.org/abs/2112.10752',
            when: 'As the compressor in front of another generator, or for smooth latent spaces.',
          },
          {
            id: 'ar', name: 'Autoregressive tokens', r: [4, 2, 2, 5, 3, 4],
            how: 'Turn the media into discrete tokens and predict them one at a time, exactly as a language model predicts words.',
            fact: 'Parti, a 20-billion-parameter autoregressive model, reached zero-shot FID 7.23 on MS-COCO (Yu et al., 2022).',
            src: 'https://arxiv.org/abs/2206.10789',
            when: 'One model should handle text and media in a single token stream.',
          },
          {
            id: 'diffusion', name: 'Diffusion', r: [5, 2, 2, 5, 3, 5],
            how: 'Train a network to remove a little noise; start from pure noise and apply it step by step until a sample appears.',
            fact: 'Diffusion reached FID 2.97 on ImageNet 128×128, ahead of the best GANs of the time (Dhariwal & Nichol, 2021).',
            src: 'https://arxiv.org/abs/2105.05233',
            when: 'Highest quality and diversity when seconds per sample are acceptable.',
            link: { href: '#/labs/diffusion', label: 'Denoise in the lab' },
          },
          {
            id: 'flow', name: 'Flow matching', r: [5, 3, 3, 5, 4, 4],
            how: 'Learn a velocity field that carries noise to data along nearly straight paths, so sampling means solving a simple ODE.',
            fact: 'Rectified flow outperformed established diffusion formulations in the large-scale study behind Stable Diffusion 3 (Esser et al., 2024).',
            src: 'https://arxiv.org/abs/2403.03206',
            when: 'New image and video models: simpler training, fewer sampling steps.',
          },
          {
            id: 'distilled', name: 'Few-step distilled', r: [4, 5, 4, 4, 2, 3],
            how: 'Distill a slow many-step sampler into a student that jumps from noise to a finished sample in one to four steps.',
            fact: 'Consistency models: FID 3.55 on CIFAR-10 with a single sampling step (Song et al., 2023).',
            src: 'https://arxiv.org/abs/2303.01469',
            when: 'Interactive or on-device generation where latency decides.',
          },
        ],
      },
      {
        id: 'tabular',
        name: 'Predicting from tables',
        short: 'Predicting from tables',
        statement: 'Most business data is rows and columns of mixed numbers and categories, often only a few thousand rows.',
        hard: 'Columns have no spatial or sequential structure to exploit, feature types are mixed, and datasets are small and noisy.',
        yardstick: 'Quality is predictive accuracy on typical small and medium tables; Control includes how explainable a prediction is.',
        solutions: [
          {
            id: 'linear', name: 'Linear / logistic model', r: [2, 5, 5, 5, 5, 5],
            how: 'A weighted sum of the features; each weight states how much one feature pushes the prediction.',
            fact: '',
            when: 'Regulated decisions that must be explained line by line.',
          },
          {
            id: 'forest', name: 'Random forest', r: [4, 4, 4, 3, 5, 4],
            how: 'Train many decision trees on random subsets of rows and features and average their votes.',
            fact: 'Across 179 classifiers and 121 datasets, random forests were the family most likely to be best (Fernández-Delgado et al., 2014).',
            src: 'https://jmlr.org/papers/v15/delgado14a.html',
            when: 'A strong baseline with almost no tuning.',
          },
          {
            id: 'gbdt', name: 'Gradient boosting', r: [5, 4, 4, 5, 4, 4],
            how: 'Add small trees one after another, each fitted to the errors the previous ones still make.',
            fact: '17 of the 29 winning Kaggle solutions published in 2015 used XGBoost (Chen & Guestrin, 2016).',
            src: 'https://arxiv.org/abs/1603.02754',
            when: 'The default for accuracy on medium-sized tables.',
            link: { href: '#/perf/bakeoff', label: 'Run the model bake-off' },
          },
          {
            id: 'deeptab', name: 'Deep tabular network', r: [4, 2, 2, 4, 2, 2],
            how: 'Embed every feature as a vector and apply a neural network or Transformer across the features of each row.',
            fact: 'On 45 medium-sized datasets, tree-based models still outperformed deep networks (Grinsztajn et al., 2022).',
            src: 'https://arxiv.org/abs/2207.08815',
            when: 'Very large tables, or tables combined with text and images.',
          },
          {
            id: 'tabpfn', name: 'Tabular foundation model', r: [5, 4, 3, 2, 5, 2],
            how: 'A Transformer pretrained on millions of synthetic tables predicts a new table in one forward pass, using its training rows as context.',
            fact: 'TabPFN: in 2.8 s it outperformed an ensemble of the strongest baselines tuned for 4 hours, on datasets of up to 10,000 samples (Hollmann et al., 2025).',
            src: 'https://www.nature.com/articles/s41586-024-08328-6',
            when: 'Small tables where you want a strong answer in seconds.',
          },
        ],
      },
    ],
  },

  // ------------------------------------------------------------------ 4 · Training and adaptation
  {
    id: 'train',
    name: 'Training and adaptation',
    desc: 'Turning an architecture into a model: finding the weights, then bending a general model toward a specific job.',
    problems: [
      {
        id: 'optimizers',
        name: 'Optimizing billions of parameters',
        short: 'Optimizing the weights',
        statement: 'Training nudges billions of weights so the error goes down, knowing only the slope at the current point.',
        hard: 'The loss surface is steep in some directions and flat in others, gradients are noisy, and every step on a large model is expensive.',
        yardstick: 'Quality is how good a solution it reaches; Speed is how few steps it needs; Cost is memory and compute per step.',
        solutions: [
          {
            id: 'sgd', name: 'SGD', r: [3, 2, 5, 4, 5, 4],
            how: 'Move every weight a small step against its gradient, estimated on a small random batch.',
            fact: 'The method goes back to the stochastic approximation of Robbins and Monro (1951).',
            src: 'https://doi.org/10.1214/aoms/1177729586',
            when: 'Simple models, or when its implicit regularization helps generalization.',
          },
          {
            id: 'momentum', name: 'SGD with momentum', r: [4, 3, 5, 4, 4, 4],
            how: 'Keep a running velocity of past gradients: consistent directions build speed and zig-zags cancel out.',
            fact: 'ResNet was trained with SGD, momentum 0.9 and weight decay 0.0001 (He et al., 2015).',
            src: 'https://arxiv.org/abs/1512.03385',
            when: 'Convolutional vision models with a tuned learning-rate schedule.',
          },
          {
            id: 'adam', name: 'Adam', r: [4, 4, 4, 5, 5, 4],
            how: 'Momentum plus a separate step size for each weight, scaled by the recent magnitude of that weight’s gradient.',
            fact: 'The defaults from the paper are still in use: β1 = 0.9, β2 = 0.999, ε = 1e-8 (Kingma & Ba, 2014).',
            src: 'https://arxiv.org/abs/1412.6980',
            when: 'A robust default that works without much tuning.',
            link: { href: '#/labs/optim', label: 'Race the optimizers' },
          },
          {
            id: 'adamw', name: 'AdamW + warmup + decay', r: [5, 4, 4, 5, 3, 4],
            how: 'Adam with weight decay applied directly to the weights and a learning rate that ramps up, then decays.',
            fact: 'The original Transformer warmed up its learning rate over the first 4,000 steps (Vaswani et al., 2017).',
            src: 'https://arxiv.org/abs/1706.03762',
            when: 'Training or fine-tuning Transformers: today’s standard recipe.',
          },
          {
            id: 'second', name: 'Second-order methods', r: [5, 4, 2, 2, 1, 2],
            how: 'Use curvature as well as slope to rescale each step; the full curvature matrix is too large, so it is approximated.',
            fact: 'Distributed Shampoo, a non-diagonal preconditioner, won the external-tuning track of the first AlgoPerf competition (Kasimbeg et al., 2025).',
            src: 'https://arxiv.org/abs/2502.15015',
            when: 'Each step is costly enough that fewer, smarter steps pay off.',
          },
        ],
      },
      {
        id: 'adapt',
        name: 'Adapting a foundation model to my task',
        short: 'Adapting a foundation model',
        statement: 'A general model knows a great deal, but not your task, your style or your data. How do you specialize it?',
        hard: 'Changing the weights is costly and can erase general skills; leaving them alone limits how far behavior can move.',
        yardstick: 'Quality is fit to your task; Speed is response latency once deployed; Cost includes the work of adapting.',
        solutions: [
          {
            id: 'prompt', name: 'Prompting / few-shot', r: [3, 4, 4, 4, 5, 2],
            how: 'Describe the task and show a few examples in the prompt; the weights are never touched.',
            fact: 'GPT-3 (175 billion parameters) performed new tasks from a few examples in the prompt, with no gradient updates (Brown et al., 2020).',
            src: 'https://arxiv.org/abs/2005.14165',
            when: 'The first version of anything, if the task fits in a prompt.',
          },
          {
            id: 'rag', name: 'RAG', r: [4, 3, 4, 5, 3, 5],
            how: 'Fetch relevant passages at query time and place them in the prompt, so answers come from your documents.',
            fact: 'RAG set the state of the art on three open-domain question-answering tasks (Lewis et al., 2020).',
            src: 'https://arxiv.org/abs/2005.11401',
            when: 'The gap is knowledge, and the knowledge changes or needs citations.',
            link: { href: '#/flow/rag', label: 'Follow a RAG request' },
          },
          {
            id: 'finetune', name: 'Full fine-tuning', r: [5, 5, 1, 2, 2, 4],
            how: 'Continue training all the weights on your examples; the model itself changes and each task needs its own full copy.',
            fact: 'Fine-tuned BERT set new state-of-the-art results on eleven language tasks (Devlin et al., 2018).',
            src: 'https://arxiv.org/abs/1810.04805',
            when: 'Maximum quality on one stable task with plenty of examples.',
          },
          {
            id: 'lora', name: 'LoRA / QLoRA', r: [4, 5, 5, 4, 3, 4],
            how: 'Freeze the model and train two small low-rank matrices per layer; their product is added to the frozen weights.',
            fact: 'LoRA: up to 10,000× fewer trainable parameters and 3× less GPU memory than full fine-tuning of GPT-3 175B (Hu et al., 2021).',
            src: 'https://arxiv.org/abs/2106.09685',
            when: 'One base model has to serve many tasks or customers.',
          },
          {
            id: 'distill', name: 'Distillation', r: [3, 5, 4, 5, 2, 4],
            how: 'Use a large teacher model’s outputs as training targets for a much smaller student that keeps the task skill.',
            fact: 'DistilBERT: 40% smaller and 60% faster while keeping 97% of BERT’s language-understanding performance (Sanh et al., 2019).',
            src: 'https://arxiv.org/abs/1910.01108',
            when: 'A proven task must run at high volume or on small hardware.',
          },
        ],
      },
      {
        id: 'align',
        name: 'Aligning behavior with human preferences',
        short: 'Aligning with preferences',
        statement: 'A pretrained model continues text. It does not yet follow instructions, decline harmful requests or prefer a good answer to a plausible one.',
        hard: 'What people want is hard to write as a loss function, and human feedback is slow, expensive and inconsistent.',
        yardstick: 'Quality is how closely behavior matches what people want; Speed is how fast you can iterate on it.',
        solutions: [
          {
            id: 'sysprompt', name: 'System prompt only', r: [2, 5, 5, 5, 5, 3],
            how: 'Leave the weights alone and state the desired behavior in instructions that precede every conversation.',
            fact: '',
            when: 'Tone and policy tweaks on a model that is already well behaved.',
          },
          {
            id: 'sft', name: 'Supervised instruction tuning', r: [3, 4, 3, 3, 4, 4],
            how: 'Fine-tune on pairs of instruction and ideal response written or curated by people.',
            fact: 'FLAN: instruction tuning on over 60 tasks beat zero-shot GPT-3 on 20 of 25 evaluated tasks (Wei et al., 2021).',
            src: 'https://arxiv.org/abs/2109.01652',
            when: 'You can write down examples of the behavior you want.',
          },
          {
            id: 'rlhf', name: 'RLHF', r: [5, 2, 2, 3, 1, 3],
            how: 'Train a reward model on human comparisons of outputs, then optimize the model against it with reinforcement learning.',
            fact: 'InstructGPT: people preferred outputs of the 1.3B RLHF model to those of the 100× larger 175B GPT-3 (Ouyang et al., 2022).',
            src: 'https://arxiv.org/abs/2203.02155',
            when: 'Quality is judged by taste and people can rank outputs.',
          },
          {
            id: 'dpo', name: 'DPO', r: [4, 4, 4, 4, 4, 3],
            how: 'Skip the reward model and the RL loop: a simple classification-style loss on preferred versus rejected answers.',
            fact: 'DPO matched or exceeded PPO-based RLHF on sentiment control, summarization and dialogue while being simpler to train (Rafailov et al., 2023).',
            src: 'https://arxiv.org/abs/2305.18290',
            when: 'You have preference pairs and want RLHF-level results with less machinery.',
          },
          {
            id: 'constitution', name: 'Constitution / AI feedback', r: [4, 4, 4, 5, 3, 5],
            how: 'Write the principles down; the model critiques and revises its own outputs against them, and the revisions become training data.',
            fact: 'Constitutional AI trained a harmless assistant without any human labels identifying harmful outputs (Bai et al., 2022).',
            src: 'https://arxiv.org/abs/2212.08073',
            when: 'Values must be explicit, inspectable and applied at scale.',
          },
        ],
      },
    ],
  },

  // -------------------------------------------------------------------- 5 · Inference and serving
  {
    id: 'serve',
    name: 'Inference and serving',
    desc: 'Running a trained model for real users: fast enough and cheap enough, for many people at once.',
    problems: [
      {
        id: 'fast',
        name: 'Generating tokens fast',
        short: 'Generating tokens fast',
        statement: 'A language model writes one token at a time, and every token needs a pass through the whole network.',
        hard: 'Done naively, each new token reprocesses the entire text so far, and each pass is limited by how fast weights can be read from memory.',
        yardstick: 'Quality is how closely the output matches the original model; Speed is tokens per second for one user.',
        solutions: [
          {
            id: 'recompute', name: 'Recompute everything', r: [5, 1, 1, 1, 5, 5],
            how: 'Run the whole sequence through the model again for each new token: correct, but work grows with the square of the length.',
            fact: '',
            when: 'Teaching and debugging only.',
          },
          {
            id: 'kvcache', name: 'KV cache', r: [5, 4, 4, 3, 4, 5],
            how: 'Store each layer’s keys and values for past tokens; a new token computes only its own row and attends to the rest.',
            fact: '',
            when: 'Always: it is the baseline every serving engine starts from.',
            link: { href: '#/perf/kvcache', label: 'Time it with and without' },
          },
          {
            id: 'speculative', name: 'Speculative decoding', r: [5, 4, 3, 3, 2, 4],
            how: 'A small draft model proposes several tokens; the large model checks them in one parallel pass and keeps those it agrees with.',
            fact: 'Speculative decoding made T5-XXL 2–3× faster with identical outputs (Leviathan et al., 2022).',
            src: 'https://arxiv.org/abs/2211.17192',
            when: 'You need the large model’s exact output, only faster.',
          },
          {
            id: 'quantized', name: 'Quantized, compiled kernels', r: [4, 4, 5, 4, 3, 4],
            how: 'Shrink weights to 4–8 bits and run compiled, hardware-tuned kernels, so more of the model sits in fast memory.',
            fact: 'transformers.js v4 reports a 4-bit 20B model at about 60 tokens/s in a WebGPU runtime on an M4 laptop (release notes, 2026).',
            src: 'https://github.com/huggingface/transformers.js/releases',
            asOf: true,
            when: 'Local, edge or cost-sensitive serving.',
            link: { href: '#/perf/quant', label: 'Measure quantization live' },
          },
          {
            id: 'small', name: 'Smaller distilled model', r: [3, 5, 5, 5, 3, 3],
            how: 'Answer with a model several times smaller that was trained to imitate the large one on your kind of request.',
            fact: 'TinyBERT: 7.5× smaller and 9.4× faster than BERT-base at over 96.8% of its GLUE score (Jiao et al., 2019).',
            src: 'https://arxiv.org/abs/1909.10351',
            when: 'High-volume, narrow tasks where good enough is enough.',
          },
        ],
      },
      {
        id: 'serving',
        name: 'Serving many users per GPU',
        short: 'Many users per GPU',
        statement: 'A GPU costs the same whether it serves one request or fifty. Sharing it well separates an affordable service from an unaffordable one.',
        hard: 'Requests arrive at random, differ in length and finish at different times, while users notice every extra moment of waiting.',
        yardstick: 'Quality is the same for all (none changes the output); Speed is the latency a user sees under load; Cost is throughput per GPU.',
        solutions: [
          {
            id: 'serial', name: 'One at a time', r: [5, 2, 1, 1, 5, 5],
            how: 'Process each request alone, in arrival order: the GPU is mostly idle and the queue grows with load.',
            fact: '',
            when: 'Prototypes and single-user tools.',
          },
          {
            id: 'static', name: 'Static batching', r: [5, 2, 3, 3, 4, 4],
            how: 'Wait until a fixed number of requests has arrived and run them together; everyone waits for the slowest.',
            fact: '',
            when: 'Offline jobs where nobody is waiting for a reply.',
          },
          {
            id: 'dynamic', name: 'Dynamic batching', r: [5, 3, 4, 4, 3, 4],
            how: 'Batch whatever arrives within a short time window, so light traffic is not held back waiting for a full batch.',
            fact: '',
            when: 'Fixed-size requests such as embeddings and classification.',
          },
          {
            id: 'continuous', name: 'Continuous batching', r: [5, 5, 5, 5, 2, 3],
            how: 'Rebuild the batch at every token step: finished sequences leave and waiting requests join immediately.',
            fact: 'Orca: 36.9× the throughput of NVIDIA FasterTransformer at the same latency, on a 175B GPT-3 (Yu et al., 2022).',
            src: 'https://www.usenix.org/conference/osdi22/presentation/yu',
            when: 'Any multi-user text-generation service.',
            link: { href: '#/perf/batching', label: 'Simulate the queue' },
          },
          {
            id: 'promptcache', name: 'Prompt caching', r: [5, 4, 5, 4, 3, 3],
            how: 'Keep the computed state of a shared prompt prefix and reuse it for every request that starts the same way.',
            fact: 'Anthropic launched prompt caching citing up to 90% lower cost and up to 85% lower latency for long prompts (Anthropic, 2024).',
            src: 'https://www.anthropic.com/news/prompt-caching',
            when: 'Many requests share a long system prompt or document.',
          },
        ],
      },
      {
        id: 'decoding',
        name: 'Choosing the next token',
        short: 'Choosing the next token',
        statement: 'The model outputs a probability for every token in its vocabulary. Something has to pick one, many times per second.',
        hard: 'Always taking the likeliest token gives dull, repetitive text; sampling freely gives nonsense; structured outputs must also be valid.',
        yardstick: 'Quality is how natural and useful the text is; Control is how precisely you can steer or constrain the output.',
        solutions: [
          {
            id: 'greedy', name: 'Greedy', r: [2, 5, 5, 5, 5, 4],
            how: 'Always take the single most probable token: deterministic and fast, but prone to loops and bland text.',
            fact: '',
            when: 'Short factual answers and reproducible tests.',
          },
          {
            id: 'beam', name: 'Beam search', r: [3, 2, 2, 3, 3, 3],
            how: 'Keep the k best partial sequences at every step and return the most probable complete one.',
            fact: 'The original Transformer decoded its translations with a beam size of 4 (Vaswani et al., 2017).',
            src: 'https://arxiv.org/abs/1706.03762',
            when: 'Translation and speech recognition, where one right answer exists.',
          },
          {
            id: 'temperature', name: 'Temperature sampling', r: [3, 5, 5, 5, 5, 3],
            how: 'Divide the logits by a temperature before sampling: below 1 sharpens the distribution, above 1 flattens it.',
            fact: '',
            when: 'A single dial between safe and creative.',
            link: { href: '#/labs/gpt', label: 'Try it on a real GPT' },
          },
          {
            id: 'topk', name: 'Top-k', r: [4, 5, 5, 5, 4, 3],
            how: 'Sample only among the k most probable tokens, cutting off the long tail of unlikely ones.',
            fact: 'The released GPT-2 samples used top-k sampling with k = 40 (Radford et al., 2019).',
            src: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
            when: 'A simple guard against rare, off-topic tokens.',
          },
          {
            id: 'topp', name: 'Nucleus (top-p)', r: [5, 5, 5, 5, 4, 3],
            how: 'Sample from the smallest set of tokens whose probabilities add up to p, so the cut-off adapts to the model’s confidence.',
            fact: 'Holtzman et al. (2019) showed that maximizing likelihood yields bland, repetitive text and proposed nucleus sampling in response.',
            src: 'https://arxiv.org/abs/1904.09751',
            when: 'Open-ended writing and chat: the usual default.',
          },
          {
            id: 'constrained', name: 'Constrained decoding', r: [4, 4, 4, 4, 2, 5],
            how: 'Before sampling, mask every token that would break a grammar or JSON schema, so the output is always valid.',
            fact: 'A precomputed index over the vocabulary lets regex- and grammar-guided generation add little overhead (Willard & Louf, 2023).',
            src: 'https://arxiv.org/abs/2307.09702',
            when: 'Output must parse: JSON, SQL, function arguments.',
          },
        ],
      },
    ],
  },

  // --------------------------------------------------------------------- 6 · Knowledge and memory
  {
    id: 'knowledge',
    name: 'Knowledge and memory',
    desc: 'What the model can look up: your documents, fresh facts, and what happened in earlier sessions.',
    problems: [
      {
        id: 'mydata',
        name: 'The model does not know my data',
        short: 'Model does not know my data',
        statement: 'The model was trained on public data up to a cut-off date. It has never seen your documents, tickets or code.',
        hard: 'Your data is large, changes daily and carries access rules, while the model’s attention is finite and its weights are frozen.',
        yardstick: 'Quality is answer accuracy on your data; Speed is response latency; Control includes citations and access rules.',
        solutions: [
          {
            id: 'longctx', name: 'Stuff the context', r: [4, 2, 1, 1, 5, 3],
            how: 'Paste the documents into the prompt and let attention find what matters.',
            fact: 'Models answer best when the relevant passage sits at the start or end of a long context, and worse in the middle (Liu et al., 2023).',
            src: 'https://arxiv.org/abs/2307.03172',
            when: 'A few documents, asked about occasionally.',
          },
          {
            id: 'dense', name: 'Dense-retrieval RAG', r: [4, 4, 4, 5, 3, 4],
            how: 'Embed document chunks as vectors, retrieve those nearest to the question and place them in the prompt.',
            fact: 'Dense Passage Retrieval beat BM25 by 9–19 points of top-20 passage accuracy (Karpukhin et al., 2020).',
            src: 'https://arxiv.org/abs/2004.04906',
            when: 'Large collections and questions answered by a few passages.',
            link: { href: '#/flow/rag', label: 'Follow a RAG request' },
          },
          {
            id: 'hybridrag', name: 'Hybrid search + rerank', r: [5, 3, 3, 4, 2, 4],
            how: 'Run keyword and vector search together, merge the lists and let a cross-encoder rescore the top candidates.',
            fact: 'BEIR (18 datasets): reranking and late-interaction models had the best average zero-shot results, at high computational cost (Thakur et al., 2021).',
            src: 'https://arxiv.org/abs/2104.08663',
            when: 'Exact terms, names and codes matter as much as meaning.',
          },
          {
            id: 'graphrag', name: 'GraphRAG', r: [4, 3, 2, 3, 1, 4],
            how: 'A model extracts entities and relations into a graph, summarizes its communities and retrieves those summaries for broad questions.',
            fact: 'On global questions over corpora of about 1 million tokens, GraphRAG gave more comprehensive and diverse answers than plain RAG (Edge et al., 2024).',
            src: 'https://arxiv.org/abs/2404.16130',
            when: 'Questions about themes across the whole corpus, not single facts.',
          },
          {
            id: 'ftcorpus', name: 'Fine-tune on the corpus', r: [3, 5, 2, 2, 2, 2],
            how: 'Train the model further on your documents so the knowledge moves into its weights.',
            fact: 'In a direct comparison, RAG consistently outperformed unsupervised fine-tuning at injecting knowledge (Ovadia et al., 2023).',
            src: 'https://arxiv.org/abs/2312.05934',
            when: 'Stable domain language and style, not fast-changing facts.',
          },
          {
            id: 'agentic', name: 'Agentic search', r: [5, 1, 2, 4, 3, 4],
            how: 'Give the model search tools and let it issue queries, read results and search again until it has enough.',
            fact: '',
            when: 'Hard questions that need several hops, where latency is acceptable.',
            link: { href: '#/flow/agent', label: 'See the agent loop' },
          },
        ],
      },
      {
        id: 'ann',
        name: 'Nearest vectors among millions',
        short: 'Nearest vectors',
        statement: 'Retrieval means finding the stored vectors closest to a query vector, among millions or billions, within milliseconds.',
        hard: 'In hundreds of dimensions the classic spatial indexes stop helping, and exact search must look at every vector.',
        yardstick: 'Quality is recall (how many true neighbors are found); Speed is query latency; Cost is memory per vector.',
        solutions: [
          {
            id: 'flat', name: 'Flat (exact)', r: [5, 1, 2, 1, 5, 5],
            how: 'Compare the query with every stored vector: perfect recall, and time that grows linearly with the collection.',
            fact: 'FAISS on GPUs: exact nearest-neighbor search 8.5× faster than the previous GPU state of the art (Johnson et al., 2017).',
            src: 'https://arxiv.org/abs/1702.08734',
            when: 'Small collections, and as ground truth when testing other indexes.',
          },
          {
            id: 'lsh', name: 'LSH', r: [2, 3, 3, 4, 4, 4],
            how: 'Hash vectors with random projections so near neighbors tend to share a bucket; search only the matching buckets.',
            fact: 'Locality-sensitive hashing was introduced with sub-linear query-time guarantees for approximate search in high dimensions (Indyk & Motwani, 1998).',
            when: 'Streaming data and near-duplicate detection with simple machinery.',
          },
          {
            id: 'ivf', name: 'IVF', r: [3, 4, 4, 4, 4, 4],
            how: 'Cluster the vectors with k-means; at query time search only the few clusters whose centers are nearest.',
            fact: '',
            when: 'Large collections with a tunable trade between speed and recall.',
          },
          {
            id: 'hnsw', name: 'HNSW', r: [5, 5, 3, 4, 3, 3],
            how: 'Build a layered graph that links each vector to near neighbors; search hops greedily from coarse layers down to fine ones.',
            fact: 'HNSW search cost grows logarithmically with the size of the collection (Malkov & Yashunin, 2016).',
            src: 'https://arxiv.org/abs/1603.09320',
            when: 'In-memory search where recall and latency both matter.',
            link: { href: '#/labs/embed', label: 'Race the indexes' },
          },
          {
            id: 'pq', name: 'Product quantization', r: [3, 4, 5, 5, 2, 3],
            how: 'Split each vector into sub-vectors and replace each with the id of its nearest codebook entry; distances come from lookup tables.',
            fact: 'Product quantization was validated on a set of 2 billion vectors (Jégou et al., 2011).',
            src: 'https://doi.org/10.1109/TPAMI.2010.57',
            when: 'Billions of vectors that must fit in memory.',
          },
          {
            id: 'diskann', name: 'DiskANN', r: [4, 3, 5, 5, 2, 3],
            how: 'Keep a compact graph and compressed vectors in memory, full vectors on SSD, and read a few disk pages per query.',
            fact: 'DiskANN: a billion vectors on one 64 GB machine, over 5,000 queries/s at 95%+ recall and under 3 ms mean latency (Subramanya et al., 2019).',
            src: 'https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html',
            when: 'Billion-scale search on a single machine with SSDs.',
          },
        ],
      },
      {
        id: 'memory',
        name: 'Remembering across sessions',
        short: 'Memory across sessions',
        statement: 'Every conversation starts from zero: the model keeps nothing between sessions unless something outside it stores and reloads it.',
        hard: 'Keeping everything overflows the context. Choosing what to keep, when to recall it and when to forget is itself a hard problem.',
        yardstick: 'Quality is recalling the right thing at the right time; Control includes inspecting and correcting what is remembered.',
        solutions: [
          {
            id: 'buffer', name: 'Context buffer', r: [4, 3, 2, 1, 5, 4],
            how: 'Keep the whole conversation in the prompt and resend it every turn until the window is full.',
            fact: '',
            when: 'Single sessions short enough to fit in the window.',
          },
          {
            id: 'summary', name: 'Rolling summary', r: [3, 4, 4, 3, 4, 3],
            how: 'When the history grows long, replace older turns with a model-written summary and continue from that.',
            fact: '',
            when: 'Long chats where the gist matters more than exact wording.',
          },
          {
            id: 'vector', name: 'Vector memory', r: [3, 4, 4, 5, 3, 3],
            how: 'Store past exchanges as embeddings and retrieve the most similar ones for each new message.',
            fact: 'Generative Agents gave 25 simulated characters a memory stream retrieved by recency, importance and relevance (Park et al., 2023).',
            src: 'https://arxiv.org/abs/2304.03442',
            when: 'Many past sessions and recall driven by the current topic.',
          },
          {
            id: 'kg', name: 'Knowledge-graph memory', r: [4, 3, 2, 4, 1, 5],
            how: 'Extract facts as entities and dated relations into a graph that can be queried, updated and corrected.',
            fact: 'Zep, a temporal knowledge graph, scored 94.8% against MemGPT’s 93.4% on the Deep Memory Retrieval benchmark (Rasmussen et al., 2025).',
            src: 'https://arxiv.org/abs/2501.13956',
            when: 'Facts change over time and must stay consistent and auditable.',
          },
          {
            id: 'notes', name: 'File-based notes', r: [4, 3, 4, 4, 4, 5],
            how: 'The agent writes and edits plain files (notes, to-do lists) through tools and reads them back in later sessions.',
            fact: 'MemGPT framed this as an operating system: the model pages data between its context window and external storage (Packer et al., 2023).',
            src: 'https://arxiv.org/abs/2310.08560',
            when: 'Agents on long projects whose memory people should be able to read.',
          },
        ],
      },
    ],
  },

  // ------------------------------------------------------------------ 7 · Agents and orchestration
  {
    id: 'agents',
    name: 'Agents and orchestration',
    desc: 'Models that act: calling tools, taking many steps, and coordinating with other models.',
    problems: [
      {
        id: 'multistep',
        name: 'Multi-step tasks',
        short: 'Multi-step tasks',
        statement: 'Real work takes many steps: search, read, decide, act, check. One model call is not enough.',
        hard: 'Errors compound: at 99% success per step, a 200-step chain finishes cleanly only about 13% of the time (0.99 to the power 200). Each step also adds latency and cost.',
        yardstick: 'Quality is how hard a task it can complete; Control is how predictable and inspectable each run is.',
        solutions: [
          {
            id: 'single', name: 'Single call', r: [2, 5, 5, 5, 5, 5],
            how: 'One prompt in, one answer out. No tools and no loop: the fastest, cheapest and most predictable option.',
            fact: '',
            when: 'The task fits in one response.',
          },
          {
            id: 'workflow', name: 'Fixed workflow', r: [3, 4, 4, 5, 4, 5],
            how: 'Code decides the steps: each model call feeds the next along a path written in advance.',
            fact: 'Anthropic’s guidance: find the simplest solution possible and only increase complexity when needed (Building Effective Agents, 2024).',
            src: 'https://www.anthropic.com/engineering/building-effective-agents',
            when: 'The steps are known in advance and rarely change.',
          },
          {
            id: 'react', name: 'ReAct tool loop', r: [4, 2, 2, 3, 4, 3],
            how: 'The model alternates between reasoning, calling a tool and reading the result, and decides for itself when it is done.',
            fact: 'ReAct beat imitation- and reinforcement-learning baselines by 34 and 10 points of success rate on ALFWorld and WebShop (Yao et al., 2022).',
            src: 'https://arxiv.org/abs/2210.03629',
            when: 'Open-ended tasks where the path cannot be scripted.',
            link: { href: '#/flow/agent', label: 'See the agent loop' },
          },
          {
            id: 'plan', name: 'Plan and execute', r: [4, 3, 3, 4, 3, 4],
            how: 'First write a complete plan, then carry out its steps one by one, replanning only when a step fails.',
            fact: 'Plan-and-Solve prompting consistently outperformed zero-shot chain-of-thought across ten reasoning datasets (Wang et al., 2023).',
            src: 'https://arxiv.org/abs/2305.04091',
            when: 'Long tasks where a reviewable plan is worth an extra call.',
          },
          {
            id: 'orchestrator', name: 'Orchestrator + sub-agents', r: [5, 3, 1, 5, 1, 2],
            how: 'A lead agent splits the task, hands parts to sub-agents working in parallel in their own contexts, then merges their reports.',
            fact: 'Anthropic’s multi-agent research system beat a single agent by 90.2% on an internal evaluation, using about 15× the tokens of a chat (Anthropic, 2025).',
            src: 'https://www.anthropic.com/engineering/multi-agent-research-system',
            when: 'Broad research-style tasks that split into independent parts.',
          },
        ],
      },
      {
        id: 'tools',
        name: 'Connecting models to tools and data',
        short: 'Connecting tools and data',
        statement: 'A model only emits text. To act it needs a way to discover tools, call them with valid arguments and read the results.',
        hard: 'Every model and application invented its own interface, so connecting M applications to N tools took M × N custom integrations.',
        yardstick: 'Quality is how reliably calls succeed; Scalability is how many tools and applications connect without custom work.',
        solutions: [
          {
            id: 'functions', name: 'Bespoke function calling', r: [4, 5, 4, 2, 4, 5],
            how: 'Describe each function with a JSON schema in the request; the model replies with a structured call your code executes.',
            fact: 'OpenAI added function calling to its chat API in June 2023 (OpenAI, 2023).',
            when: 'A handful of tools inside one application you control.',
          },
          {
            id: 'openapi', name: 'OpenAPI plugins', r: [3, 3, 3, 3, 3, 3],
            how: 'Publish an OpenAPI description of a web API; the model reads it and composes the HTTP requests.',
            fact: 'ChatGPT plugins launched in March 2023 and were retired in April 2024 (OpenAI, 2024).',
            when: 'Existing REST services you want to expose with little new code.',
          },
          {
            id: 'mcp', name: 'MCP', r: [4, 4, 4, 5, 3, 4],
            how: 'An open client-server protocol: wrap a tool or data source once as a server and any compliant application can use it.',
            fact: 'The 2026-07-28 revision made MCP stateless: no initialize handshake and no session id (MCP blog, 2026).',
            src: 'https://blog.modelcontextprotocol.io/posts/2026-07-28/',
            asOf: true,
            when: 'Tools that should work across many models and applications.',
            link: { href: '#/flow/agent', label: 'Follow a tool call' },
          },
          {
            id: 'a2a', name: 'A2A', r: [3, 2, 3, 5, 2, 3],
            how: 'Agents publish an Agent Card describing their skills, then exchange tasks, messages and artifacts as peers.',
            fact: 'A2A reached v1.0 in March 2026 and joined the Agentic AI Foundation in August 2026 with 150+ supporting organizations (A2A project, 2026).',
            src: 'https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/',
            asOf: true,
            when: 'Agents built by different teams or vendors must cooperate.',
          },
          {
            id: 'computeruse', name: 'Computer use (GUI)', r: [3, 1, 1, 5, 3, 2],
            how: 'The model looks at screenshots and moves the mouse and types, so it can operate any software a person can.',
            fact: 'OSWorld 2.1 (operating a real desktop): 89.3% for the leading model, lab-reported (Anthropic, Nov 2026).',
            src: 'https://www.anthropic.com/claude-fable-and-mythos-5-2',
            asOf: true,
            when: 'Legacy software with no API.',
          },
          {
            id: 'codeact', name: 'Code execution', r: [5, 4, 4, 4, 2, 3],
            how: 'Instead of one call per step, the model writes a short program that calls several tools and returns only the result.',
            fact: 'CodeAct: up to 20% higher success rate than JSON or text actions in a study of 17 models (Wang et al., 2024).',
            src: 'https://arxiv.org/abs/2402.01030',
            when: 'Many chained calls or large intermediate results.',
          },
        ],
      },
      {
        id: 'longtasks',
        name: 'Keeping long tasks on track',
        short: 'Keeping long tasks on track',
        statement: 'Over hundreds of steps an agent’s context fills with stale detail, early mistakes get built upon and the goal drifts.',
        hard: 'The context window is finite, attention degrades as it fills, and the agent cannot easily tell from the inside that it has gone wrong.',
        yardstick: 'Quality is the share of long tasks finished correctly; Speed is how little it slows the task down.',
        solutions: [
          {
            id: 'bigctx', name: 'Bigger context window', r: [3, 2, 1, 2, 5, 3],
            how: 'Use a model with a larger window so more history fits before anything must be dropped.',
            fact: 'Flagship models list context windows of about 1 million tokens (vendor model pages, Nov 2026).',
            src: 'https://platform.claude.com/docs/en/about-claude/models/overview',
            asOf: true,
            when: 'Tasks slightly too long for the current window.',
          },
          {
            id: 'compaction', name: 'Compaction', r: [3, 4, 4, 4, 4, 3],
            how: 'When the window is nearly full, summarize the history into a short brief and continue from that.',
            fact: '',
            when: 'Long single-threaded work where older detail can be dropped.',
          },
          {
            id: 'subagents', name: 'Isolated sub-agent contexts', r: [4, 4, 2, 5, 2, 3],
            how: 'Hand each sub-task to a fresh agent with a clean context; only its condensed result returns to the lead agent.',
            fact: 'In Anthropic’s analysis, token usage alone explained 80% of the performance variance on a browsing benchmark (Anthropic, 2025).',
            src: 'https://www.anthropic.com/engineering/multi-agent-research-system',
            when: 'Tasks that explore a lot but need to report only a little.',
          },
          {
            id: 'checkpoints', name: 'External files and checkpoints', r: [4, 4, 4, 4, 3, 5],
            how: 'Write plans, progress and decisions to files and commit work at milestones, so state survives outside the context window.',
            fact: '',
            when: 'Multi-hour or multi-session work that must be resumable.',
          },
          {
            id: 'verify', name: 'Verification loops', r: [5, 2, 3, 4, 3, 4],
            how: 'After each chunk of work run tests, linters or a reviewing model, and feed failures back before moving on.',
            fact: 'Reflexion: feeding test results back as reflections raised HumanEval pass@1 to 91%, from 80% for GPT-4 alone (Shinn et al., 2023).',
            src: 'https://arxiv.org/abs/2303.11366',
            when: 'Work with a checkable outcome, such as code.',
          },
          {
            id: 'human', name: 'Human in the loop', r: [5, 1, 2, 2, 4, 5],
            how: 'Pause at set checkpoints or before risky actions and ask a person to confirm, correct or redirect.',
            fact: 'METR’s last estimate: tasks of about 17 hours at 50% reliability, but about 3 hours at 80% (METR, May 2026).',
            src: 'https://metr.org/time-horizons/',
            asOf: true,
            when: 'Mistakes are costly and a person is available.',
          },
        ],
      },
    ],
  },

  // ----------------------------------------------------------------- 8 · Trust, safety and product
  {
    id: 'trust',
    name: 'Trust, safety and product',
    desc: 'Whether people can rely on the system: truthful, hard to abuse, and measurably good.',
    problems: [
      {
        id: 'hallucination',
        name: 'Hallucination',
        short: 'Hallucination',
        statement: 'A language model produces fluent text whether or not it has the facts, and it rarely signals the difference.',
        hard: 'Training and evaluation reward a confident guess over an admission of uncertainty, so the habit is built in (Kalai et al., 2025).',
        yardstick: 'Quality is the factual reliability of what reaches the user; Control is how well a reader can verify it.',
        solutions: [
          {
            id: 'nothing', name: 'Do nothing', r: [1, 5, 5, 5, 5, 1],
            how: 'Ship the raw model output: acceptable for brainstorming, risky wherever a wrong fact has a cost.',
            fact: 'Mata v. Avianca: lawyers were fined $5,000 for filing fake precedents generated by ChatGPT (S.D.N.Y., 2023).',
            src: 'https://en.wikipedia.org/wiki/Mata_v._Avianca,_Inc.',
            when: 'Brainstorming and drafts that a person will check anyway.',
          },
          {
            id: 'cite', name: 'RAG with citations', r: [4, 3, 3, 4, 3, 5],
            how: 'Retrieve sources, answer only from them and attach a citation to each claim so a reader can check it.',
            fact: 'Retrieval-based legal research tools still hallucinated on 17% to 33% of queries (Magesh et al., 2024).',
            src: 'https://arxiv.org/abs/2405.20362',
            when: 'Answers must be traceable to documents.',
            link: { href: '#/flow/rag', label: 'Follow a RAG request' },
          },
          {
            id: 'selfverify', name: 'Self-verification', r: [4, 2, 2, 4, 3, 3],
            how: 'Draft an answer, generate questions that check its claims, answer them independently, then revise the draft.',
            fact: 'Chain-of-Verification reduced hallucinations on list questions, closed-book QA and long-form generation (Dhuliawala et al., 2023).',
            src: 'https://arxiv.org/abs/2309.11495',
            when: 'No trusted source exists and extra latency is acceptable.',
          },
          {
            id: 'constrain', name: 'Constrained outputs', r: [3, 4, 4, 3, 3, 5],
            how: 'Restrict answers to a schema, a fixed list of choices or values copied from a database, so free invention is impossible.',
            fact: '',
            when: 'The answer space can be enumerated or looked up.',
          },
          {
            id: 'abstain', name: 'Calibrated abstention', r: [4, 3, 3, 4, 2, 4],
            how: 'Estimate the model’s confidence and answer “I don’t know” below a threshold, trading coverage for accuracy.',
            fact: 'Semantic entropy, computed over the meanings of sampled answers, detects confabulations without task-specific training data (Farquhar et al., 2024).',
            src: 'https://www.nature.com/articles/s41586-024-07421-0',
            when: 'A wrong answer costs more than no answer.',
          },
        ],
      },
      {
        id: 'injection',
        name: 'Prompt injection and misuse',
        short: 'Prompt injection and misuse',
        statement: 'A model cannot reliably tell instructions from data: text inside a web page, email or tool result can hijack an agent that reads it.',
        hard: 'Instructions and data share one channel and attackers can try without limit. Adaptive attacks bypassed 12 recent defenses, most at over 90% success (Nasr et al., 2025).',
        yardstick: 'Quality is how much real attack it stops; Speed and Simplicity are what it costs users and builders.',
        solutions: [
          {
            id: 'rules', name: 'System-prompt rules', r: [1, 5, 5, 5, 5, 2],
            how: 'Tell the model in its system prompt to ignore instructions found inside documents and tool results.',
            fact: 'Indirect prompt injection was demonstrated against real products such as Bing Chat (Greshake et al., 2023).',
            src: 'https://arxiv.org/abs/2302.12173',
            when: 'Never alone: a cheap first layer under real defenses.',
          },
          {
            id: 'classifiers', name: 'Input/output classifiers', r: [3, 3, 3, 5, 3, 3],
            how: 'Separate models screen incoming text for attacks and outgoing text for policy violations, and block what they flag.',
            fact: 'Constitutional Classifiers cut jailbreak success from 86% to 4.4% in automated tests, with 0.38% more refusals (Sharma et al., 2025).',
            src: 'https://www.anthropic.com/research/constitutional-classifiers',
            when: 'High-volume services facing known categories of misuse.',
          },
          {
            id: 'sandbox', name: 'Least-privilege sandbox', r: [4, 4, 4, 4, 2, 5],
            how: 'Give the agent only the permissions the task needs and run its tools in a sandbox with restricted files and network.',
            fact: 'Least privilege was set out as a design principle for secure systems by Saltzer and Schroeder (1975).',
            src: 'https://www.cs.virginia.edu/~evans/cs551/saltzer/',
            when: 'Any agent that can run code or reach the network.',
          },
          {
            id: 'approval', name: 'Human approval', r: [4, 1, 2, 1, 4, 5],
            how: 'Require a person to confirm sensitive actions such as payments, deletions and outgoing messages before they run.',
            fact: '',
            when: 'Irreversible or costly actions at low volume.',
          },
          {
            id: 'patterns', name: 'Plan/data separation', r: [5, 3, 3, 3, 1, 5],
            how: 'A privileged model plans from the trusted request only; untrusted content goes to a quarantined model whose output cannot issue commands.',
            fact: 'CaMeL solved 77% of AgentDojo tasks with provable security, against 84% for an undefended system (Debenedetti et al., 2025).',
            src: 'https://arxiv.org/abs/2503.18813',
            when: 'Agents that read untrusted content and can take consequential actions.',
          },
        ],
      },
      {
        id: 'evals',
        name: 'Knowing whether it works',
        short: 'Knowing whether it works',
        statement: 'Outputs are open-ended text and the same prompt can give different answers. How do you know a change made things better?',
        hard: 'Public benchmarks saturate and leak into training data, human judgment is slow and costly, and automated judges have biases of their own.',
        yardstick: 'Quality is how well the measurement predicts real usefulness; Speed is how quickly you get a verdict.',
        solutions: [
          {
            id: 'bench', name: 'Static benchmarks', r: [2, 5, 5, 5, 5, 3],
            how: 'Run a fixed public set of questions with known answers and report the score.',
            fact: 'SWE-bench Verified: the top three models scored 96–97% in an independent run and the leaderboard was archived as saturated (Vals AI, Sep 2026).',
            src: 'https://www.vals.ai/benchmarks/swebench',
            asOf: true,
            when: 'Comparing base models before building anything.',
            link: { href: '#/perf/records', label: 'See the record books' },
          },
          {
            id: 'humaneval', name: 'Human evaluation', r: [5, 1, 1, 2, 3, 4],
            how: 'People rate or compare outputs without knowing which system produced them.',
            fact: 'Chatbot Arena had collected over 240,000 human preference votes when its paper appeared (Chiang et al., 2024).',
            src: 'https://arxiv.org/abs/2403.04132',
            when: 'Subjective quality, and calibrating every cheaper method.',
          },
          {
            id: 'judge', name: 'LLM as judge', r: [3, 4, 4, 5, 4, 3],
            how: 'A strong model grades outputs against a rubric or picks the better of two answers.',
            fact: 'GPT-4 as judge agreed with human preferences over 80% of the time, the same level as agreement between humans (Zheng et al., 2023).',
            src: 'https://arxiv.org/abs/2306.05685',
            when: 'Many outputs to grade and a rubric you have validated against people.',
          },
          {
            id: 'tasktests', name: 'Task-based tests', r: [4, 4, 4, 4, 3, 5],
            how: 'Write your own test cases from real usage, with checks that pass or fail, and run them on every change.',
            fact: 'SWE-bench grades models on 2,294 real GitHub issues from 12 Python repositories, using each project’s own tests (Jimenez et al., 2023).',
            src: 'https://arxiv.org/abs/2310.06770',
            when: 'You are shipping a product and need regression protection.',
          },
          {
            id: 'online', name: 'Online A/B and monitoring', r: [5, 2, 3, 4, 2, 4],
            how: 'Ship the change to a fraction of real traffic and compare outcome metrics; keep watching after launch.',
            fact: '',
            when: 'Enough traffic to detect differences in real outcomes.',
          },
          {
            id: 'redteam', name: 'Red teaming', r: [4, 2, 2, 2, 3, 4],
            how: 'People and automated attackers deliberately try to make the system fail, to find problems before users do.',
            fact: 'Over 3,000 estimated hours of red teaming found no universal jailbreak against a classifier-guarded model (Sharma et al., 2025).',
            src: 'https://arxiv.org/abs/2501.18837',
            when: 'Before launch and after every major change to a risky system.',
          },
        ],
      },
    ],
  },
];

/** Look up a level and (optionally) one of its problems by id. */
export function findLevel(id) {
  return LEVELS.find((l) => l.id === id) || null;
}
export function findProblem(level, id) {
  return level?.problems.find((p) => p.id === id) || null;
}
