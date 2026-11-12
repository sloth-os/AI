// The pipelines of the Data Flow chapter, as data. Each pipeline is a small graph (nodes, edges,
// zones: see core/flowgraph.js) plus 6 to 9 stages. A stage says which node the data has reached,
// which edges it just crossed, what happened to it, its shape and dtype, an example payload, the
// wire format of the hop, where time and cost go, and an "exhibit" (a 3D depiction of the data).
//
// Honesty rules for this file: shapes and sizes are computed for the real model class named in
// `model`; token ids, scores, timings and contents that are only illustrative say "example".
// Dated protocol facts are not repeated here: a stage cites them from landscape.js with `ref`.

const N = (id, type, label, x, z, extra) => ({ id, type, label, pos: [x, z], ...extra });
const E = (id, from, to, label, extra) => ({ id, from, to, label, ...extra });
const GPU_ZONE = '#d95926';
const DEVICE_ZONE = '#3987e5';
const GPU_MEM = { name: 'Stays in GPU memory' };

// ------------------------------------------------------------------ 1. one chat request

const llm = {
  id: 'llm',
  name: 'One chat request',
  short: 'LLM chat',
  tagline: 'From the text you type to the words that stream back, through every tensor in between.',
  model: "Shapes are for a **Llama 3 8B Instruct** class model: 32 decoder blocks, width 4,096, 32 attention heads sharing 8 key/value heads, a vocabulary of 128,256 tokens, 16-bit floats. Scores and word ids marked as examples are illustrative.",
  graph: {
    nodes: [
      N('user', 'user', 'You', -10.5, 0),
      N('api', 'gateway', 'API server', -6.5, 0),
      N('tmpl', 'op', 'Chat template', -2.5, -2.5),
      N('tok', 'op', 'Tokenizer', 1.5, -2.5),
      N('detok', 'op', 'Detokenizer', 1.5, 2.5),
      N('emb', 'table', 'Embedding\ntable', 5.5, -2.5, { role: 'compute' }),
      N('blocks', 'layers', '32 decoder\nblocks', 9.5, -2.5, { size: 1.25 }),
      N('kv', 'store', 'KV cache', 13.5, -2.5),
      N('head', 'op', 'Output head', 9.5, 2.5, { role: 'compute' }),
      N('samp', 'sampler', 'Sampler', 5.5, 2.5, { role: 'compute' }),
    ],
    edges: [
      E('req', 'user', 'api', 'JSON · HTTPS', { bend: 0.1 }),
      E('sse', 'api', 'user', 'SSE', { bend: 0.1 }),
      E('msgs', 'api', 'tmpl', 'messages[]'),
      E('str', 'tmpl', 'tok', 'one string'),
      E('ids', 'tok', 'emb', 'ids · int64 [1, 16]'),
      E('x', 'emb', 'blocks', '[1, 16, 4096] f16'),
      E('kvw', 'blocks', 'kv', 'K, V per block', { both: true }),
      E('h', 'blocks', 'head', 'last row [1, 4096]', { labelLift: 0.3 }),
      E('logits', 'head', 'samp', 'logits [1, 128256]'),
      E('loop', 'samp', 'emb', 'next id', { dashed: true }),
      E('piece', 'samp', 'detok', 'token id'),
      E('out', 'detok', 'api', 'text piece'),
    ],
    zones: [
      { id: 'cpu', label: 'Server CPU', nodes: ['api', 'tmpl', 'tok', 'detok'] },
      { id: 'gpu', label: 'GPU memory', nodes: ['emb', 'blocks', 'kv', 'head', 'samp'], color: GPU_ZONE },
    ],
  },
  stages: [
    {
      title: 'The request leaves your device',
      node: 'api', from: 'user', edges: ['req'],
      what: "You typed 20 characters. The app wraps them in a small JSON body, adds the model name and the sampling settings, and sends one HTTPS request. Nothing about AI has happened yet: this is ordinary web traffic.",
      shape: '1 string · 20 characters', dtype: 'UTF-8 text inside JSON', size: 'under 200 bytes of body',
      payload: {
        label: 'POST /v1/chat/completions', lang: 'json',
        code: `{
  "model": "llama-3-8b-instruct",
  "messages": [
    { "role": "user",
      "content": "Why is the sky blue?" }
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": true
}`,
      },
      wire: { name: 'JSON over HTTPS', detail: "HTTP/2 inside TLS 1.3. This chat-completions body is the de facto request format: vLLM, SGLang, llama.cpp and Ollama all accept it." },
      cost: "A network round trip plus the TLS handshake, paid once per connection: tens of milliseconds. The model has not been touched.",
      exhibit: {
        kind: 'json', title: 'HTTP request', caption: 'JSON · UTF-8', bar: 'POST /v1/chat/completions',
        code: `{
  "model": "llama-3-8b-instruct",
  "messages": [
    { "role": "user",
      "content": "Why is the sky blue?" }
  ],
  "temperature": 0.7,
  "stream": true
}`,
      },
    },
    {
      title: 'The chat template flattens the conversation',
      node: 'tmpl', from: 'api', edges: ['msgs'],
      what: "A language model reads one flat sequence, not a list of messages. A template stored next to the tokenizer writes the conversation out as a single string and marks who is speaking with **special tokens**. It ends with an open assistant header: the cue to start answering.",
      shape: '1 string · 136 characters', dtype: 'text with special-token markers',
      payload: {
        label: 'the prompt string',
        code: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n
Why is the sky blue?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\\n\\n`,
        note: "Shown on three lines to fit. The real string breaks only where \\n is written.",
      },
      wire: { name: 'In-process function call', detail: "A Jinja template shipped in the model's `tokenizer_config.json` and run inside the server. Using the wrong template is a classic reason a good model answers badly." },
      cost: "Microseconds of string formatting on a CPU. The hidden cost: every special token and every system prompt is input you pay for on each request.",
      exhibit: {
        kind: 'text', title: 'Prompt string', caption: 'str · 136 chars', mono: true, size: 0.17, width: 4.3,
        text: '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\nWhy is the sky blue?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n',
      },
    },
    {
      title: 'Text becomes token ids',
      node: 'tok', from: 'tmpl', edges: ['str'],
      what: "The tokenizer cuts the string into pieces from a fixed vocabulary of 128,256 entries (byte-pair encoding) and replaces each piece with its row number. From here on the model never sees letters, only integers. The question itself is 6 tokens; the template around it adds 10.",
      shape: '[1, 16]', dtype: 'int64', size: '128 bytes',
      payload: {
        label: 'tokens and their ids',
        code: `<|begin_of_text|>      128000
<|start_header_id|>    128006
user                      882
<|end_header_id|>      128007
\\n\\n                      271
Why  is  the  sky  blue  ?
  10445 374 279 13180 6437 30
<|eot_id|>             128009
<|start_header_id|>    128006
assistant               78191
<|end_header_id|>      128007
\\n\\n                      271`,
        note: "The special-token ids are the real Llama 3 values. Treat the word ids as an example.",
      },
      wire: { name: 'In-process, then host to GPU', detail: "The ids form a tiny integer tensor. Copying it from host memory into GPU memory (over PCIe) is the last thing the CPU does before the model runs." },
      cost: "Well under a millisecond here. What matters is the count: tokens are the unit you are billed in and the unit the context window is measured in.",
      exhibit: {
        kind: 'tokens', title: 'Tokens and ids', caption: '[1, 16] int64', show: 'both', width: 4.6,
        tokens: [
          { t: '<|begin_of_text|>', id: 128000, special: true }, { t: '<|start_header_id|>', id: 128006, special: true }, { t: 'user', id: 882 },
          { t: '<|end_header_id|>', id: 128007, special: true }, { t: '\\n\\n', id: 271 }, { t: 'Why', id: 10445 }, { t: '·is', id: 374 }, { t: '·the', id: 279 },
          { t: '·sky', id: 13180 }, { t: '·blue', id: 6437 }, { t: '?', id: 30 }, { t: '<|eot_id|>', id: 128009, special: true },
          { t: '<|start_header_id|>', id: 128006, special: true }, { t: 'assistant', id: 78191 }, { t: '<|end_header_id|>', id: 128007, special: true }, { t: '\\n\\n', id: 271 },
        ],
      },
    },
    {
      title: 'Each id looks up a vector',
      node: 'emb', from: 'tok', edges: ['ids'],
      what: "The embedding table holds one row of 4,096 numbers for every vocabulary entry. Each id selects its row, so 16 integers become a 16 x 4,096 matrix. Nothing is computed, and position is not encoded yet: this model family injects position later, inside attention (rotary embeddings).",
      shape: '[1, 16, 4096]', dtype: 'float16', size: '128 KiB',
      payload: {
        label: 'pseudocode',
        code: `table.shape     # [128256, 4096] float16
                # = 1.05 GB
x = table[ids]  # a gather: no arithmetic
x.shape         # [1, 16, 4096]
x[0, 5, :4]     # the row of "Why":
                # [0.012, -0.009, 0.004, -0.016]`,
        note: "The four values are an example.",
      },
      wire: { ...GPU_MEM, detail: "From here to the sampler the data lives in GPU memory as float16 tensors. Nothing is serialized and nothing crosses a network." },
      cost: "A memory lookup: negligible time. The table itself is about 1.05 GB of the model's 16 GB of weights.",
      links: [{ label: 'Embedding Space lab', href: '#/labs/embed' }],
      exhibit: { kind: 'vectors', title: 'Embeddings (6 of 16 rows)', caption: '[1, 16, 4096] float16', rows: 6, cols: 26, labels: ['Why', '·is', '·the', '·sky', '·blue', '?'], more: '… 4,096' },
    },
    {
      title: 'Thirty-two blocks mix in the context',
      node: 'blocks', from: 'emb', edges: ['x'],
      what: "The matrix passes through 32 identical blocks. In each, **attention** lets every position read from the positions before it (a causal mask hides the future), then a small network transforms each position on its own. The shape never changes; what the vectors mean does. This first pass over the whole prompt is the **prefill**.",
      shape: '[1, 16, 4096] in and out of every block', dtype: 'float16',
      payload: {
        label: 'pseudocode for the whole stack',
        code: `for block in blocks:          # 32 of them
    x = x + attention(norm(x))
    #     32 heads x 128 dims, causal
    x = x + mlp(norm(x))
    #     4096 -> 14336 -> 4096

# attention weights in one block:
#     [1, 32, 16, 16]
# x is still [1, 16, 4096]`,
      },
      wire: { ...GPU_MEM, detail: "Tensors pass from block to block as pointers. When a model is split across several GPUs, the activations hop between cards over NVLink or PCIe at the split points." },
      cost: "Nearly all of it. Every token is multiplied against all 8 billion weights: about 16 billion floating-point operations per token (roughly 2 per weight). The 16 prompt tokens run in parallel, which is why input tokens are cheaper than output tokens.",
      links: [{ label: 'Inside a GPT, block by block', href: '#/labs/gpt' }, { label: 'Quantization', href: '#/perf/quant' }],
      exhibit: { kind: 'matrix', title: 'Attention, one head (illustrative)', caption: '[16, 16]', rows: 16, cols: 16, pattern: 'causal', xLabel: 'reads from earlier positions →', yLabel: 'position' },
    },
    {
      title: 'The KV cache remembers the prompt',
      node: 'kv', from: 'blocks', edges: ['kvw'],
      what: "Attention needs a **key** and a **value** vector for every earlier token, in every block. Once computed they never change, so the server keeps them. Each new token then computes only its own row and reads the rest from the cache, instead of redoing the whole prefix.",
      shape: '32 blocks x (K, V), each [1, 8, 16, 128]', dtype: 'float16', size: '2 MiB now · 128 KiB per token',
      payload: {
        label: 'the arithmetic',
        code: `K.shape = V.shape = [1, 8, 16, 128]
# batch, KV heads, tokens, head dim

per token:
  32 blocks x 2 x 8 x 128 x 2 bytes
  = 128 KiB
16 tokens      2 MiB
8,192 tokens   1 GiB  (the full context)`,
      },
      wire: { ...GPU_MEM, detail: "The cache never leaves the GPU during a request. Prompt caching in hosted APIs is the same structure kept alive between requests." },
      cost: "Memory instead of time: about 128 KiB per token buys not recomputing the prefix at every step. Cache memory, not compute, usually limits how many conversations fit on one GPU.",
      links: [{ label: 'KV cache experiment', href: '#/perf/kvcache' }],
      exhibit: { kind: 'cache', title: 'KV cache', caption: '128 KiB per token', layers: 8, slots: 26, filled: 16, first: 'block 1', last: 'block 32', filledLabel: '16 prompt tokens cached' },
    },
    {
      title: 'The last position becomes 128,256 scores',
      node: 'head', from: 'blocks', edges: ['h'],
      what: "Only the vector at the last position decides the next token. A final normalization and one multiplication against the vocabulary matrix turn its 4,096 numbers into one score, a **logit**, for every entry in the vocabulary.",
      shape: '[1, 128256]', dtype: 'float32', size: '501 KiB',
      payload: {
        label: 'pseudocode and example scores',
        code: `h = norm(x[:, -1, :])     # [1, 4096]
logits = h @ W_vocab.T    # [1, 128256]

"The"        14.2
"Because"    13.6
"Sun"        13.1
"It"         12.6
"Blue"       11.9
... 128,251 lower scores`,
      },
      wire: { ...GPU_MEM, detail: "The scores are usually converted to float32 at this point, so that the sampling arithmetic does not lose precision." },
      cost: "One 4,096 x 128,256 multiplication, about 0.5 billion multiply-adds: small change next to the 32 blocks.",
      exhibit: {
        kind: 'bars', title: 'Logits (example values)', caption: '[1, 128256] float32', barWidth: 0.62, tail: 16, tailLabel: '… 128,251 more', format: (v) => v.toFixed(1),
        items: [{ label: 'The', value: 14.2 }, { label: 'Because', value: 13.6 }, { label: 'Sun', value: 13.1 }, { label: 'It', value: 12.6 }, { label: 'Blue', value: 11.9 }],
      },
    },
    {
      title: 'Sampling picks one token, then the loop repeats',
      node: 'samp', from: 'head', also: ['emb'], edges: ['logits', 'loop'],
      what: "The scores are divided by the **temperature**, the unlikely tail is cut off (top-p), softmax turns what is left into probabilities and one token is drawn at random. Its id goes back to the embedding table and the pass repeats, one new token at a time, until the model emits its end-of-turn token.",
      shape: '[1, 1]', dtype: 'int64', size: '8 bytes',
      payload: {
        label: 'pseudocode (probabilities follow from the example scores)',
        code: `p = softmax(logits / 0.7)   # temperature
p = top_p(p, 0.9)           # cut the tail
# "The" 58%   "Because" 24%
# "Sun" 12%   "It" 6%
next_id = sample(p)         # 791, "The"
ids = concat(ids, next_id)  # [1, 17]
# ... and around again, until
# next_id == 128009  (<|eot_id|>)`,
      },
      wire: { name: 'GPU memory, then back to the start', detail: "The chosen id is appended to the sequence. Thanks to the KV cache, only this one new token travels through the 32 blocks on the next step." },
      cost: "Microseconds per draw. But this loop is why output is slow and priced higher: one full pass through the model for every generated token, strictly one after another.",
      exhibit: {
        kind: 'bars', title: 'After temperature and top-p (example)', caption: 'one id · int64', barWidth: 0.62,
        items: [{ label: 'The', value: 0.58, chosen: true }, { label: 'Because', value: 0.24 }, { label: 'Sun', value: 0.12 }, { label: 'It', value: 0.06 }, { label: 'Blue', value: 0 }],
      },
    },
    {
      title: 'Tokens stream back as text',
      node: 'api', from: 'detok', frame: ['user', 'api', 'detok'], edges: ['piece', 'out', 'sse'],
      flows: [{ path: ['piece', 'out', 'sse'], count: 6, speed: 3.2 }],
      what: "Each sampled id is turned back into bytes and decoded to text. A token can end in the middle of a multi-byte character, so the detokenizer holds bytes back until they form valid UTF-8. The server pushes every piece to you immediately as a **server-sent event**, which is why answers appear word by word.",
      shape: '1 text piece per token', dtype: 'UTF-8 text inside JSON', size: 'about 50 bytes per event',
      payload: {
        label: 'the HTTP response, as it arrives',
        code: `HTTP/1.1 200 OK
content-type: text/event-stream

data: {"choices":[{"delta":{"role":"assistant"}}]}

data: {"choices":[{"delta":{"content":"The"}}]}

data: {"choices":[{"delta":{"content":" sky"}}]}

data: [DONE]`,
      },
      wire: { name: 'Server-sent events (SSE)', detail: "One long-lived HTTP response with content type `text/event-stream`. Every event is a `data:` line followed by a blank line, and the stream ends with `data: [DONE]`." },
      cost: "Two numbers describe what you feel: **time to first token** (queueing plus prefill) and **tokens per second** after that (one decode step each).",
      links: [{ label: 'The serving path', href: '#/flow/serve' }, { label: 'Deployment modes', href: '#/deploy' }],
      exhibit: {
        kind: 'json', title: 'Response stream', caption: 'text/event-stream', bar: 'HTTP/1.1 200 OK', cols: 50,
        code: `data: {"choices":[{"delta":{"content":"The"}}]}

data: {"choices":[{"delta":{"content":" sky"}}]}

data: {"choices":[{"delta":{"content":" is"}}]}

data: [DONE]`,
      },
    },
  ],
};

// ------------------------------------------------------------------ 2. one training step

const train = {
  id: 'train',
  name: 'One training step',
  short: 'Training',
  tagline: 'Shards, batches, forward, loss, backward, all-reduce, optimizer step, checkpoint.',
  model: "Numbers follow the open recipes that reproduce **GPT-2 small** (124M parameters: 12 blocks, width 768, vocabulary 50,257) on 8 GPUs: 64 sequences of 1,024 tokens per GPU and step, 16-bit autocast, AdamW. Token ids are an example.",
  graph: {
    nodes: [
      N('data', 'database', 'Text corpus', -12, -2.5),
      N('tokz', 'op', 'Tokenize\n(once, offline)', -8, -2.5),
      N('shards', 'file', 'Token shards', -4, -2.5),
      N('loader', 'queue', 'Data loader', 0, -2.5),
      N('ckpt', 'file', 'Checkpoint', 0, 2.5),
      N('fwd', 'layers', 'Forward pass', 4.5, -2.5),
      N('loss', 'op', 'Loss', 8.5, -2.5, { role: 'compute' }),
      N('bwd', 'layers', 'Backward pass', 8.5, 2.5),
      N('opt', 'scaler', 'Optimizer\n(AdamW)', 4.5, 2.5, { role: 'compute' }),
      N('p1', 'gpu', 'GPU 1', 13.5, -2.5),
      N('p2', 'gpu', 'GPU 2', 13.5, 1),
      N('p3', 'gpu', 'GPUs 3 to 7', 13.5, 4.5),
    ],
    edges: [
      E('raw', 'data', 'tokz', 'Parquet (Arrow)'),
      E('tok', 'tokz', 'shards', 'uint16 arrays'),
      E('read', 'shards', 'loader', 'memory-mapped read'),
      E('feed', 'loader', 'fwd', 'x, y · [64, 1024] int64'),
      E('logits', 'fwd', 'loss', '[64, 1024, 50257]'),
      E('back', 'loss', 'bwd', 'scalar loss', { labelLift: 0.3 }),
      E('ar1', 'bwd', 'p1', 'NCCL all-reduce', { both: true }),
      E('ar2', 'bwd', 'p2', '', { both: true }),
      E('ar3', 'bwd', 'p3', '', { both: true }),
      E('grads', 'bwd', 'opt', 'mean gradient'),
      E('upd', 'opt', 'fwd', 'new weights', { dashed: true }),
      E('save', 'opt', 'ckpt', 'safetensors'),
    ],
    zones: [
      { id: 'cpu', label: 'CPU and storage', nodes: ['data', 'tokz', 'shards', 'loader', 'ckpt'] },
      { id: 'gpu0', label: 'GPU 0', nodes: ['fwd', 'loss', 'bwd', 'opt'], color: GPU_ZONE },
      { id: 'peers', label: 'The other 7 GPUs', nodes: ['p1', 'p2', 'p3'], color: GPU_ZONE },
    ],
  },
  stages: [
    {
      title: 'Text is tokenized once, offline',
      node: 'shards', from: 'tokz', frame: ['data', 'tokz', 'shards'], also: ['data'], edges: ['raw', 'tok'],
      flows: [{ path: ['raw', 'tok'], count: 5, speed: 2.6 }],
      what: "Training reads the same data many times, so the expensive text work is done up front. The corpus arrives as Parquet files, a tokenizer turns every document into token ids, and the ids are written to disk as flat binary shards. GPT-2's vocabulary of 50,257 fits in 16 bits, so each token costs 2 bytes.",
      shape: '[100,000,000] per shard', dtype: 'uint16', size: '200 MB per shard · 100 shards for 10B tokens',
      payload: {
        label: 'shards on disk',
        code: `shard_000001.npy  uint16 [100000000]
shard_000002.npy  uint16 [100000000]
shard_000003.npy  uint16 [100000000]
...               200 MB each

tokens[:6]
  [50256, 1026, 286, 262, 4252, 318]
# 50256 = <|endoftext|>,
# the separator between documents`,
        note: "Only 50256 is a real id here; the others are an example.",
      },
      wire: { name: 'Parquet in, binary shards out', detail: "Parquet stores Arrow columns compressed on disk: the standard way to ship text data sets. The shards are raw arrays that can be memory-mapped with no parsing at all." },
      cost: "CPU time, once: hours of tokenizing on many cores, never repeated. Skipping this step would make every pass over the data pay for it again while the GPUs wait.",
      exhibit: {
        kind: 'ids', title: 'The start of one shard', caption: 'uint16 [100,000,000]', width: 4.4,
        tokens: [{ id: 50256, special: true }, { id: 1026 }, { id: 286 }, { id: 262 }, { id: 4252 }, { id: 318 }, { id: 4171 }, { id: 13 }, { id: 50256, special: true }, { id: 464 }, { id: 2068 }, { id: '…', special: true }],
      },
    },
    {
      title: 'The loader cuts batches',
      node: 'loader', from: 'shards', edges: ['read'],
      what: "Each GPU process reads a different slice of the shards. A batch is 64 windows of 1,024 consecutive tokens. The **targets** are the same windows shifted left by one position: at every position the model must predict the token that comes next.",
      shape: 'x and y: [64, 1024] each, per GPU', dtype: 'int64', size: '0.5 MiB each · 524,288 tokens per step on 8 GPUs',
      payload: {
        label: 'one batch',
        code: `x.shape = [64, 1024]  int64  # inputs
y.shape = [64, 1024]  int64  # targets

x[0, :5] = [50256, 1026, 286, 262, 4252]
y[0, :5] = [ 1026,  286, 262, 4252, 318]
# y is x shifted left by one token`,
        note: "Example ids.",
      },
      wire: { name: 'Memory-mapped read, then host to GPU', detail: "The file is mapped into memory, slices are copied into page-locked (pinned) host memory, and from there to the GPU over PCIe." },
      cost: "The loader must deliver half a million tokens per step. If it falls behind, eight expensive GPUs sit idle: input pipelines are a common hidden bottleneck.",
      exhibit: { kind: 'tensor', mode: 'seq', title: 'One batch: inputs and targets', caption: '2 x [64, 1024] int64', slabs: 2, rows: 8, cols: 26, axes: { x: '1,024 tokens →', y: '64 sequences', depth: 'x, y' } },
    },
    {
      title: 'Forward pass: a guess at every position',
      node: 'fwd', from: 'loader', edges: ['feed'],
      what: "The batch runs through the 12 blocks, exactly as in a chat request but for all 65,536 positions at once. The output is one score per vocabulary entry at every position. Unlike inference, every intermediate activation is kept in memory: the backward pass will need it.",
      shape: '[64, 1024, 50257]', dtype: 'bfloat16 (autocast)', size: '6.6 GB of logits per GPU',
      payload: {
        label: 'pseudocode',
        code: `with autocast(dtype=bfloat16):
    logits = model(x)

logits.shape  # [64, 1024, 50257]
# 3.29 billion scores
# = 6.6 GB at 2 bytes each`,
      },
      wire: { ...GPU_MEM, detail: "Activations stay on the card. Mixed precision (autocast) computes in 16-bit floats and keeps the weights themselves in float32." },
      cost: "Roughly a third of the compute of a step. Memory is the tighter limit: logits and stored activations, not the 0.5 GB of weights, decide the largest batch that fits.",
      links: [{ label: 'Inside a GPT', href: '#/labs/gpt' }],
      exhibit: { kind: 'tensor', title: 'Logits', caption: '[64, 1024, 50257] bf16', slabs: 4, rows: 8, cols: 22, axes: { x: 'vocabulary: 50,257 →', y: '1,024 positions', depth: '64 sequences' } },
    },
    {
      title: 'Loss: one number for how wrong',
      node: 'loss', from: 'fwd', edges: ['logits'],
      what: "**Cross-entropy** compares the scores at each position with the token that really came next and averages the result over all 65,536 positions. An untrained model that spreads its guesses evenly over 50,257 tokens scores ln(50,257) = 10.82. Training is the business of pushing this number down.",
      shape: '[ ] (a scalar)', dtype: 'float32', size: '4 bytes',
      payload: {
        label: 'pseudocode',
        code: `loss = cross_entropy(
    logits.view(-1, 50257),  # [65536, 50257]
    y.view(-1))              # [65536]

loss  # tensor(10.82) at step 0
      # = ln(50257)`,
      },
      wire: { ...GPU_MEM, detail: "The loss is computed in float32 even under autocast: the logarithm of a tiny probability does not survive 16 bits." },
      cost: "Cheap to compute, and the number everyone watches: it is logged every few steps to see whether the run is healthy.",
      exhibit: { kind: 'scalar', title: 'Loss', caption: '[ ] float32', value: '10.82', label: 'cross-entropy at step 0', sub: '= ln(50,257): a uniform guess' },
    },
    {
      title: 'Backward pass: a gradient for every weight',
      node: 'bwd', from: 'loss', edges: ['back'],
      what: "Backpropagation walks the computation in reverse and works out, for each of the 124 million weights, how the loss would change if that weight moved a little. The result has exactly the shape of the model: one gradient value per weight.",
      shape: '124,439,808 values, shaped like the weights', dtype: 'float32', size: '498 MB per GPU',
      payload: {
        label: 'pseudocode',
        code: `loss.backward()

wte.weight.grad.shape
  # [50257, 768]
h[0].attn.c_attn.weight.grad.shape
  # [768, 2304]
...
# 124,439,808 gradient values = 498 MB`,
      },
      wire: { ...GPU_MEM, detail: "Gradients are written next to the weights they belong to. Nothing leaves the card until the all-reduce." },
      cost: "About two thirds of the compute of a step, and the reason training needs far more memory than inference: every activation saved in the forward pass is consumed here.",
      links: [{ label: 'Neural Network Lab', href: '#/labs/neural' }],
      exhibit: { kind: 'vectors', title: 'Gradients, one row per tensor', caption: '124M x float32', rows: 5, cols: 26, monoLabels: true, labels: ['wte', 'h.0.attn', 'h.0.mlp', 'h.11.mlp', 'ln_f'], more: '…' },
    },
    {
      title: 'All-reduce: eight GPUs agree on one gradient',
      node: 'p1', from: 'bwd', also: ['p2', 'p3'], frame: ['bwd', 'p1', 'p2', 'p3'], edges: ['ar1', 'ar2', 'ar3'], pad: 0.8,
      flows: [{ path: 'ar1' }, { path: '-ar1', phase: 0.5 }, { path: 'ar2' }, { path: '-ar2', phase: 0.5 }, { path: 'ar3' }, { path: '-ar3', phase: 0.5 }],
      what: "Each GPU saw different data, so each holds a different gradient. An **all-reduce** sums them across the cluster and hands every GPU the same average. All eight replicas then make the identical update, and their weights never drift apart.",
      shape: '8 x 124,439,808 values in, 1 average out', dtype: 'float32', size: '498 MB exchanged per GPU, every step',
      payload: {
        label: 'pseudocode (PyTorch DistributedDataParallel)',
        code: `# backend: NCCL
for bucket in gradient_buckets:  # ~25 MB each
    all_reduce(bucket, op=SUM)
    bucket /= 8

# buckets are sent as soon as they are
# ready, while backward is still running`,
      },
      wire: { name: 'NCCL all-reduce', detail: "NVIDIA's collective-communication library moves the buffers GPU to GPU: NVLink inside a server, InfiniBand or RoCE with RDMA between servers. No serialization: raw float32." },
      cost: "Pure communication. With fast links it hides behind the backward pass; with slow ones it becomes the limit on how well training scales to more GPUs.",
      exhibit: { kind: 'vectors', title: 'All-reduce: average the gradients', caption: '8 x 498 MB', rows: 4, cols: 26, labels: ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 7'], shared: 0.7, result: { label: 'mean' } },
    },
    {
      title: 'The optimizer moves the weights',
      node: 'opt', from: 'bwd', also: ['fwd'], edges: ['grads', 'upd'],
      what: "**AdamW** turns the averaged gradient into an update. It keeps two running averages per weight (of the gradient and of its square), so its state is twice the size of the model. Gradients are clipped first, and the learning rate follows a warmup and decay schedule. Then the next step begins with the new weights.",
      shape: '124,439,808 weights + 2 x as much state', dtype: 'float32', size: '0.5 GB of weights + 1 GB of optimizer state',
      payload: {
        label: 'one AdamW step',
        code: `clip_grad_norm_(params, 1.0)
m = 0.9  * m + 0.1  * g      # mean of g
v = 0.95 * v + 0.05 * g * g  # mean of g^2
w -= lr * m_hat / (sqrt(v_hat) + 1e-8)
w -= lr * 0.1 * w            # weight decay
# lr: warmup to 6e-4, then cosine decay`,
      },
      wire: { ...GPU_MEM, detail: "Each GPU applies the same update to its own copy of the weights. Because the gradients were averaged first, the copies stay identical without ever being sent." },
      cost: "A few element-wise passes over the weights: small. The real cost is how many steps you need, which is what the learning-rate schedule and the batch size are tuned for.",
      links: [{ label: 'Optimizer Arena', href: '#/labs/optim' }],
      exhibit: {
        kind: 'json', title: 'AdamW update', caption: 'lr up to 6e-4',
        code: `clip_grad_norm_(params, 1.0)
m = 0.9  * m + 0.1  * g
v = 0.95 * v + 0.05 * g * g
w -= lr * m_hat / (sqrt(v_hat) + 1e-8)
w -= lr * 0.1 * w`,
      },
    },
    {
      title: 'Checkpoint: the model becomes a file',
      node: 'ckpt', from: 'opt', edges: ['save'],
      what: "Every so often the weights are written to storage, so the run can resume after a crash and the model can be evaluated and shipped. This file is the product of the whole pipeline: all a serving stack needs is the weights plus the tokenizer.",
      shape: '148 named tensors · 124,439,808 values', dtype: 'float32', size: '498 MB (plus about 1 GB of optimizer state to resume)',
      payload: {
        label: 'model.safetensors: a JSON header, then raw bytes', lang: 'json',
        code: `{
  "wte.weight": {
    "dtype": "F32",
    "shape": [50257, 768],
    "data_offsets": [0, 154389504]
  },
  "wpe.weight": {
    "dtype": "F32",
    "shape": [1024, 768],
    "data_offsets": [154389504, 157535232]
  }
}`,
        note: "Tensor order and offsets are illustrative. **GGUF**, the format of llama.cpp-style runtimes, packs the same idea plus the tokenizer and quantized weights into one file.",
      },
      wire: { name: 'safetensors on disk or object storage', detail: "A JSON header that lists every tensor (name, dtype, shape, byte range), followed by the raw bytes. It can be memory-mapped and, unlike a Python pickle, loading it cannot run code." },
      cost: "Seconds of I/O per save. At large scale, checkpoints are written asynchronously to object storage so the GPUs do not wait for the disk.",
      links: [{ label: 'Quantization', href: '#/perf/quant' }, { label: 'Deployment modes', href: '#/deploy' }],
      exhibit: {
        kind: 'json', title: 'model.safetensors', caption: '498 MB', bar: 'JSON header + raw tensor bytes',
        code: `{ "wte.weight": {
    "dtype": "F32",
    "shape": [50257, 768],
    "data_offsets": [0, 154389504] },
  "wpe.weight": {
    "dtype": "F32",
    "shape": [1024, 768], … },
  … 146 more tensors }`,
      },
    },
  ],
};

// ------------------------------------------------------------------ 3. retrieval-augmented generation

const rag = {
  id: 'rag',
  name: 'Retrieval-augmented generation',
  short: 'RAG',
  tagline: 'Ingest documents into a vector index, then answer a question from them, with citations.',
  model: "Vector shapes are for a **bge-small-en-v1.5** class embedding model (384 dimensions, inputs up to 512 tokens). The corpus (1,000 documents cut into 12,000 chunks), the texts and every score are an example.",
  graph: {
    nodes: [
      N('docs', 'file', 'Documents', -9, -3.5),
      N('chunker', 'op', 'Parse + chunk', -4.5, -3.5),
      N('embed', 'model', 'Embedding\nmodel', 0, -3.5),
      N('index', 'database', 'Vector index', 4.5, -3.5),
      N('user', 'user', 'You', -9, 1),
      N('app', 'app', 'RAG app', -4.5, 1),
      N('rerank', 'model', 'Reranker', 4.5, 1),
      N('prompt', 'op', 'Prompt\nassembly', 0, 5.5),
      N('llm', 'model', 'Language\nmodel', -4.5, 5.5),
    ],
    edges: [
      E('parse', 'docs', 'chunker', 'PDF, HTML, Markdown'),
      E('chunks', 'chunker', 'embed', 'chunks · JSON'),
      E('vecs', 'embed', 'index', 'vectors [n, 384] f32'),
      E('q', 'user', 'app', 'question'),
      E('qtext', 'app', 'embed', 'query text'),
      E('cands', 'index', 'rerank', 'top 50 ids + scores', { labelLift: 0.3 }),
      E('top', 'rerank', 'prompt', 'top 4 chunks'),
      E('qq', 'app', 'prompt', 'question', { dashed: true }),
      E('msgs', 'prompt', 'llm', 'messages · JSON'),
      E('ans', 'llm', 'user', 'SSE + citations'),
    ],
    zones: [
      { id: 'ingest', label: 'Ingest: when documents change', nodes: ['docs', 'chunker'] },
      { id: 'store', label: 'Shared by both halves', nodes: ['embed', 'index'] },
      { id: 'query', label: 'Query: on every question', nodes: ['user', 'app'], color: DEVICE_ZONE },
    ],
  },
  stages: [
    {
      title: 'Documents are cut into chunks',
      node: 'chunker', from: 'docs', edges: ['parse'],
      extras: [{ node: 'docs', spec: { kind: 'docs', title: 'Documents', caption: '1,000 files', docs: [{ name: 'handbook.pdf', sub: '212 pages' }, { name: 'faq.html', sub: 'web page' }, { name: 'policy.md', sub: 'Markdown' }] } }],
      what: "Retrieval works on passages, not whole files. Each document is parsed to plain text and cut into overlapping **chunks**, small enough for the embedding model to read in one go. Every chunk keeps a pointer to its source, which is what makes citations possible later.",
      shape: '12,000 chunks · up to 512 tokens each', dtype: 'UTF-8 text + metadata', size: 'about 2 kB per chunk',
      payload: {
        label: 'one chunk (a line of JSON Lines)', lang: 'json',
        code: `{
  "id": 12,
  "source": "handbook.pdf",
  "page": 7,
  "text": "Refunds are accepted within 30 days of delivery. Opened items can only be exchanged ..."
}`,
      },
      wire: { name: 'Files in, JSON Lines out', detail: "PDF, HTML and Markdown go in; one JSON object per chunk comes out. Large ingest jobs move the same records as Arrow or Parquet tables." },
      cost: "Parsing is the slow, error-prone part: tables, scanned pages, layouts. Chunk size is the key decision, because a fact split across two chunks may never be retrieved whole.",
      exhibit: { kind: 'chunks', title: 'Chunks', caption: '12,000 x ≤ 512 tokens', start: 9, docs: [{ name: 'handbook.pdf', n: 4 }, { name: 'faq.html', n: 3 }, { name: 'policy.md', n: 4 }] },
    },
    {
      title: 'Each chunk becomes a vector',
      node: 'embed', from: 'chunker', edges: ['chunks'],
      what: "An embedding model reads a chunk and returns one vector of 384 numbers that places the passage in a space where **nearby means similar in meaning**. All 12,000 chunks are embedded in batches. This is the only part of ingest that needs a neural network.",
      shape: '[12000, 384]', dtype: 'float32, unit length', size: '18.4 MB',
      payload: {
        label: 'POST /v1/embeddings (request, then response)', lang: 'json',
        code: `{ "model": "bge-small-en-v1.5",
  "input": [
    "Refunds are accepted within ...",
    "Shipping takes 3 to 5 days ..."
  ] }

{ "data": [
  { "index": 0,
    "embedding": [0.0213, -0.0457, ...] },
  { "index": 1,
    "embedding": [-0.0331, 0.0094, ...] }
] }`,
        note: "Example values; each embedding has 384 of them.",
      },
      wire: { name: 'JSON over HTTPS, in batches', detail: "Thousands of floats written out as JSON text are bulky, so embedding APIs can also return each vector as base64-encoded float32 bytes." },
      cost: "One encoder pass per chunk, batched on a GPU. Paid once at ingest, then again for every chunk that changes. Switching to another embedding model later means re-embedding everything.",
      links: [{ label: 'Embedding Space lab', href: '#/labs/embed' }],
      exhibit: { kind: 'vectors', title: 'Chunk embeddings', caption: '[12000, 384] float32', rows: 5, cols: 26, monoLabels: true, labels: ['#9', '#10', '#11', '#12', '#13'], more: '… 384' },
    },
    {
      title: 'Vectors go into a search index',
      node: 'index', from: 'embed', edges: ['vecs'],
      what: "Comparing a question with all 12,000 vectors works, but does not scale to millions. An **approximate nearest-neighbor** index such as HNSW links every vector to its closest neighbors, building a graph that a search can hop through. Ingest ends here; it runs again only when documents change.",
      shape: '[12000, 384] + a neighbor graph', dtype: 'float32 vectors, integer links', size: '18.4 MB of vectors, plus the graph',
      payload: {
        label: 'upsert request (a typical JSON form)', lang: 'json',
        code: `{ "points": [ {
    "id": 12,
    "vector": [0.0213, -0.0457, ...],
    "payload": {
      "source": "handbook.pdf",
      "page": 7 }
} ] }`,
      },
      wire: { name: 'gRPC with protobuf, or JSON over HTTP', detail: "Vector databases usually offer both. gRPC sends the floats as packed binary instead of text, which matters when millions of vectors are loaded. Bulk loads often use Parquet files." },
      cost: "Building the graph costs more than storing the vectors, and the index lives in RAM for speed. Its build settings trade memory and build time against recall.",
      exhibit: { kind: 'scatter', title: 'Index: vectors linked to neighbors', caption: '12,000 x 384', n: 240, links: true, note: 'each dot is a chunk (3 of 384 dimensions)' },
    },
    {
      title: 'A question arrives and is embedded the same way',
      node: 'embed', from: 'app', also: ['user'], frame: ['user', 'app', 'embed'], edges: ['q', 'qtext'],
      flows: [{ path: ['q', 'qtext'], count: 5, speed: 2.8 }],
      what: "Now the online half. The app sends the question through the **same** embedding model that encoded the chunks. Question and passages must live in the same space, or the distances between them mean nothing.",
      shape: '[1, 384]', dtype: 'float32, unit length', size: '1.5 kB',
      payload: {
        label: 'request, then response', lang: 'json',
        code: `{ "model": "bge-small-en-v1.5",
  "input": "What is the refund window?" }

{ "data": [ { "index": 0,
  "embedding": [0.0188, -0.0412, ...] } ] }`,
      },
      wire: { name: 'JSON over HTTPS', detail: "One short text in, one vector out. Some embedding models expect a short instruction in front of a query but not in front of passages; forgetting it quietly hurts retrieval." },
      cost: "A single small forward pass: milliseconds. It sits on the critical path of every question, so it is often served close to the app.",
      exhibit: { kind: 'vectors', title: 'Query vector', caption: '[1, 384] float32', rows: 1, cols: 26, labels: ['question'], more: '… 384', seed: 12 },
    },
    {
      title: 'Nearest-neighbor search finds candidates',
      node: 'index', from: 'embed', edges: ['vecs'],
      what: "The index walks its graph from a few entry points toward the query vector and returns the 50 closest chunks by cosine similarity. It looks at only a small fraction of the 12,000 vectors. That is the “approximate” in the name: fast, at the price of sometimes missing a true neighbor.",
      shape: '50 x (id, score)', dtype: 'integer ids + float32 scores', size: 'under 1 kB',
      payload: {
        label: 'search request, then response', lang: 'json',
        code: `{ "vector": [0.0188, -0.0412, ...],
  "limit": 50,
  "filter": { "lang": "en" } }

[ { "id": 12, "score": 0.83 },
  { "id": 16, "score": 0.79 },
  { "id": 14, "score": 0.71 },
  ... ]`,
      },
      wire: { name: 'gRPC with protobuf, or JSON over HTTP', detail: "The query carries 384 floats and optional metadata filters; only ids and scores come back. The chunk text is fetched by id afterward." },
      cost: "Sub-linear: the search touches hundreds of vectors, not all of them, so it stays fast as the corpus grows. Recall below 100% is the cost, and filters can make it worse.",
      exhibit: { kind: 'scatter', title: 'Nearest neighbors of the query', caption: 'top 50 of 12,000', n: 240, k: 6, note: 'bright: the query · white: its nearest chunks' },
    },
    {
      title: 'A reranker reads question and chunk together',
      node: 'rerank', from: 'index', edges: ['cands'],
      what: "Vector distance is a blunt instrument. A **cross-encoder** reads the question and one candidate side by side and scores how well the passage answers it. It is far more accurate and far slower, so it only sees the 50 candidates. The best 4 go on.",
      shape: '50 scores in, the top 4 chunks out', dtype: 'float32', size: '4 chunks · about 2,000 tokens',
      payload: {
        label: 'POST /rerank (request, then response)', lang: 'json',
        code: `{ "model": "bge-reranker-base",
  "query": "What is the refund window?",
  "documents": [
    "Refunds are accepted within ...",
    "Shipping takes 3 to 5 days ...",
    ... ],
  "top_n": 4 }

{ "results": [
  { "index": 0, "relevance_score": 0.97 },
  { "index": 3, "relevance_score": 0.88 },
  ... ] }`,
      },
      wire: { name: 'JSON over HTTPS', detail: "The candidate texts travel with the query, so this is the largest request on the query path so far: tens of kilobytes." },
      cost: "Fifty forward passes of a small transformer. Usually the most expensive retrieval step, and the one with the best return in answer quality.",
      exhibit: {
        kind: 'bars', title: 'Reranker scores (example)', caption: '[50] float32', tail: 12, tailLabel: '… 44 more candidates', format: (v) => v.toFixed(2),
        items: [{ label: '#12', value: 0.97, chosen: true }, { label: '#16', value: 0.88, chosen: true }, { label: '#14', value: 0.81, chosen: true }, { label: '#9', value: 0.62, chosen: true }, { label: '#31', value: 0.21 }, { label: '#55', value: 0.09 }],
      },
    },
    {
      title: 'The prompt is assembled',
      node: 'prompt', from: 'rerank', also: ['app'], edges: ['top', 'qq'],
      what: "The app writes one prompt: instructions, the 4 winning chunks with numbered labels, and the question. The model is not retrained and does not look anything up: it simply reads these passages as part of its input. That is all retrieval-augmented generation is.",
      shape: 'about 2,100 tokens', dtype: 'text in a messages array', size: '4 x 512 tokens of sources + instructions + question',
      payload: {
        label: 'the request to the language model', lang: 'json',
        code: `{
  "model": "claude-opus-5-5",
  "max_tokens": 16000,
  "stream": true,
  "system": "Answer only from the numbered sources. Cite them like [1].",
  "messages": [ {
    "role": "user",
    "content": "[1] handbook.pdf, page 7\\nRefunds are accepted within 30 days ...\\n\\n[2] policy.md\\nOpened items can only be exchanged ...\\n\\n[3] ...\\n\\n[4] ...\\n\\nQuestion: What is the refund window?"
  } ]
}`,
      },
      wire: { name: 'In-process, then JSON over HTTPS', detail: "String building inside the app, then an ordinary chat request. The retrieved text is just more input tokens." },
      cost: "Every retrieved token is billed as input and adds prefill time. Four good chunks beat forty mediocre ones: retrieval quality decides whether these tokens earn their place.",
      exhibit: {
        kind: 'text', title: 'Assembled prompt', caption: '≈ 2,100 tokens', size: 0.165, width: 4.5,
        text: 'Answer only from the numbered sources. Cite them like [1].\n\n[1] handbook.pdf, p. 7: Refunds are accepted within 30 days of delivery …\n[2] policy.md: Opened items can only be exchanged …\n[3] …   [4] …\n\nQuestion: What is the refund window?',
      },
    },
    {
      title: 'The answer streams back with citations',
      node: 'llm', from: 'prompt', also: ['user'], frame: ['prompt', 'llm', 'user'], edges: ['msgs', 'ans'],
      flows: [{ path: ['msgs', 'ans'], count: 5, speed: 3 }],
      what: "The model writes an answer grounded in the passages it was handed and marks which source supports each claim. The app maps every marker back to the chunk, file and page, so a reader can check the answer instead of trusting it.",
      shape: 'about 40 tokens of text + 2 citations', dtype: 'UTF-8 text deltas', size: 'about 150 bytes per event',
      payload: {
        label: 'server-sent events from the model API',
        code: `event: content_block_delta
data: {"type":"content_block_delta",
  "index":0,"delta":{"type":"text_delta",
  "text":"Refunds are accepted"}}

event: content_block_delta
data: {"type":"content_block_delta",
  "index":0,"delta":{"type":"text_delta",
  "text":" within 30 days [1]."}}

event: message_stop
data: {"type":"message_stop"}`,
        note: "Each real `data:` payload is a single line; they are wrapped here to fit.",
      },
      wire: { name: 'Server-sent events (SSE)', detail: "The same transport as any chat answer. The app resolves [1] to handbook.pdf, page 7, chunk 12 and sends the reader the text plus the links." },
      cost: "Generation dominates the wait: retrieval takes milliseconds, the answer takes seconds. Most wrong answers trace back to retrieval, not to the model.",
      links: [{ label: 'Inside the model: one chat request', href: '#/flow/llm' }],
      exhibit: {
        kind: 'text', title: 'Answer', caption: 'text + citations', size: 0.19, width: 4.2,
        text: 'Refunds are accepted within 30 days of delivery [1]. Opened items can only be exchanged, not refunded [2].\n\n[1] handbook.pdf, page 7\n[2] policy.md',
      },
    },
  ],
};

// ------------------------------------------------------------------ 4. the tool-calling loop

const TOOL_ID = 'toolu_01A09q90qw90lq917835lq9';

const agent = {
  id: 'agent',
  name: 'The agent loop',
  short: 'Agent + MCP',
  tagline: 'How a model uses a tool: schemas in, a tool_use block out, a JSON-RPC call to an MCP server, the result back in.',
  model: "Message shapes follow the **Anthropic Messages API** (tool use) and the **Model Context Protocol** in its 2026-07-28 revision. The weather tool, its data, the ids and the token counts are an example.",
  graph: {
    nodes: [
      N('user', 'user', 'You', -8, 0),
      N('app', 'agent', 'Agent app\n(runs the loop)', -3, 0, { size: 1.15 }),
      N('model', 'model', 'Model API', 2.5, -3.2),
      N('mcp', 'tool', 'MCP server', 2.5, 3.2),
      N('ext', 'cloud', 'Weather service', 8, 3.2),
    ],
    edges: [
      E('ask', 'user', 'app', 'question', { bend: 0.1 }),
      E('ans', 'app', 'user', 'answer · SSE', { bend: 0.1 }),
      E('req', 'app', 'model', 'messages + tools · HTTPS', { bend: 0.1 }),
      E('resp', 'model', 'app', 'tool_use or text', { bend: 0.1 }),
      E('rpc', 'app', 'mcp', 'JSON-RPC request', { bend: 0.1 }),
      E('rpcr', 'mcp', 'app', 'JSON-RPC result', { bend: 0.1 }),
      E('http', 'mcp', 'ext', "the tool's own API", { bend: 0.1 }),
      E('httpr', 'ext', 'mcp', '', { bend: 0.1 }),
    ],
    zones: [
      { id: 'provider', label: 'Model provider', nodes: ['model'], color: GPU_ZONE },
      { id: 'tools', label: 'Tools: any vendor, local or remote', nodes: ['mcp', 'ext'] },
    ],
  },
  stages: [
    {
      title: 'A question the model cannot answer alone',
      node: 'app', from: 'user', edges: ['ask'],
      what: "The user asks something that needs fresh data. A language model only knows what is in its weights and in its prompt, so the **agent app**, the program that runs the loop, has to give it a way to act.",
      shape: '1 string · 45 characters', dtype: 'UTF-8 text', size: '45 bytes',
      payload: { label: 'user message', code: 'Do I need an umbrella in Lisbon this weekend?' },
      wire: { name: 'JSON over HTTPS, or a local call', detail: "From a chat window to the app's backend. In a desktop or terminal agent this hop is just a function call inside one process." },
      cost: "Nothing yet. The cost of an agent comes from how many times the loop that starts here goes around.",
      exhibit: { kind: 'text', title: 'User message', caption: 'text', text: 'Do I need an umbrella in Lisbon this weekend?', width: 3.6 },
    },
    {
      title: 'The app asks the MCP server what it can do',
      node: 'mcp', from: 'app', edges: ['rpc', 'rpcr'], ref: 'mcp',
      what: "Tools live behind **MCP servers**. The app sends `tools/list` and gets back each tool's name, a description written for the model, and a JSON Schema for its arguments. Since the 2026-07-28 revision the protocol is stateless: no handshake and no session, every request stands alone.",
      shape: '1 tool: name, description, JSON Schema', dtype: 'JSON-RPC 2.0 messages', size: 'about 400 bytes',
      payload: {
        label: 'request, then response', lang: 'json',
        code: `{ "jsonrpc": "2.0", "id": 1,
  "method": "tools/list" }

{ "jsonrpc": "2.0", "id": 1,
  "result": { "tools": [ {
    "name": "get_forecast",
    "description": "Daily weather forecast for a city.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "city": { "type": "string" },
        "days": { "type": "integer" } },
      "required": ["city"] }
  } ] } }`,
      },
      wire: { name: 'JSON-RPC 2.0 over stdio or Streamable HTTP', detail: "MCP has two transports. A local server is a child process that reads and writes JSON lines on stdin and stdout. A remote server is one HTTP endpoint that answers a POST with JSON or with an event stream." },
      cost: "Milliseconds, and usually cached. The lasting cost is tokens: every tool description is sent to the model on every turn, so fifty tools can cost more than the conversation.",
      exhibit: {
        kind: 'json', title: 'tools/list result', caption: 'JSON-RPC 2.0',
        code: `{ "tools": [ {
  "name": "get_forecast",
  "description": "Daily weather …",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": { "type": "string" },
      "days": { "type": "integer" } },
    "required": ["city"] } } ] }`,
      },
    },
    {
      title: 'The request carries the tool schemas',
      node: 'model', from: 'app', edges: ['req'],
      what: "The app calls the model API with the conversation **and** the list of tools. The tool definitions become part of the prompt: the model can read what `get_forecast` does and which arguments it takes, but it cannot run it.",
      shape: 'messages[1] + tools[1]', dtype: 'JSON', size: 'about 600 bytes · tool definitions are billed as input',
      payload: {
        label: 'POST /v1/messages', lang: 'json',
        code: `{
  "model": "claude-opus-5-5",
  "max_tokens": 16000,
  "tools": [ {
    "name": "get_forecast",
    "description": "Daily weather forecast for a city.",
    "input_schema": {
      "type": "object",
      "properties": {
        "city": { "type": "string" },
        "days": { "type": "integer" } },
      "required": ["city"] }
  } ],
  "messages": [ {
    "role": "user",
    "content": "Do I need an umbrella in Lisbon this weekend?"
  } ]
}`,
      },
      wire: { name: 'JSON over HTTPS', detail: "One POST with an API key header. MCP's `inputSchema` is copied into the API's `input_schema`: the same JSON Schema under a slightly different name." },
      cost: "Input tokens: the conversation plus every tool definition, on every call. Prompt caching exists largely to make this repeated prefix cheap.",
      exhibit: {
        kind: 'json', title: 'Model request', caption: 'JSON', bar: 'POST /v1/messages',
        code: `{ "model": "claude-opus-5-5",
  "tools": [ {
    "name": "get_forecast",
    "input_schema": { … } } ],
  "messages": [ {
    "role": "user",
    "content": "Do I need an umbrella …"
  } ] }`,
      },
    },
    {
      title: 'The model answers with a tool_use block',
      node: 'app', from: 'model', edges: ['resp'],
      what: "Instead of text, the reply contains a **tool_use** block: the name of a tool and arguments that fit its schema. The stop reason says `tool_use`. The model has not called anything. It has written a structured request and handed control back to the app.",
      shape: 'content[2]: thinking + tool_use', dtype: 'JSON', size: 'about 300 bytes · billed as output tokens',
      payload: {
        label: 'response body (shortened)', lang: 'json',
        code: `{
  "role": "assistant",
  "content": [
    { "type": "thinking",
      "thinking": "",
      "signature": "EqQB..." },
    { "type": "tool_use",
      "id": "${TOOL_ID}",
      "name": "get_forecast",
      "input": { "city": "Lisbon", "days": 3 } }
  ],
  "stop_reason": "tool_use"
}`,
        note: "Current models think before they act. By default the thinking block arrives with empty text, and it must be sent back unchanged.",
      },
      wire: { name: 'JSON over HTTPS, or SSE when streaming', detail: "The arguments are generated token by token like any other text. Marking a tool `strict` makes the API guarantee that they match the schema." },
      cost: "Output tokens, the expensive kind, plus one full model round trip. This decision step is where an agent spends most of its time.",
      exhibit: {
        kind: 'json', title: 'tool_use block', caption: 'stop_reason: tool_use',
        code: `{ "type": "tool_use",
  "id": "${TOOL_ID}",
  "name": "get_forecast",
  "input": { "city": "Lisbon",
             "days": 3 } }`,
      },
    },
    {
      title: 'The app calls the tool through MCP',
      node: 'mcp', from: 'app', edges: ['rpc'], ref: 'mcp',
      what: "The app is in charge here. It checks the request (is this tool allowed? must the user approve?), then forwards name and arguments to the MCP server as a `tools/call`. The model never talks to the server itself.",
      shape: 'name + arguments', dtype: 'JSON-RPC 2.0 request', size: 'about 150 bytes',
      payload: {
        label: 'over Streamable HTTP',
        code: `POST /mcp
content-type: application/json
accept: application/json, text/event-stream

{ "jsonrpc": "2.0", "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_forecast",
    "arguments": {
      "city": "Lisbon", "days": 3 } } }`,
      },
      wire: { name: 'JSON-RPC 2.0 over Streamable HTTP', detail: "A plain POST to the server's single endpoint. With a local server the identical JSON goes down a pipe to the child process (stdio) instead." },
      cost: "Network plus whatever the tool does. This is also the security boundary: the app decides what runs, with whose credentials, and what the model gets to see of the result.",
      exhibit: {
        kind: 'json', title: 'tools/call', caption: 'JSON-RPC 2.0', bar: 'POST /mcp',
        code: `{ "jsonrpc": "2.0", "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_forecast",
    "arguments": { "city": "Lisbon",
                   "days": 3 } } }`,
      },
    },
    {
      title: 'The server does the real work',
      node: 'ext', from: 'mcp', edges: ['http', 'httpr'],
      what: "Behind the protocol the MCP server is ordinary code. It calls a weather service with its own credentials, trims the response to what is useful, and returns it as **content blocks** that a model can read.",
      shape: 'content[1]: text', dtype: 'JSON-RPC 2.0 response', size: 'about 250 bytes',
      payload: {
        label: 'tools/call result', lang: 'json',
        code: `{ "jsonrpc": "2.0", "id": 2,
  "result": {
    "content": [ {
      "type": "text",
      "text": "Lisbon. Sat 14 Nov: 17 C, rain 80%. Sun 15 Nov: 18 C, rain 20%. Mon 16 Nov: 19 C, rain 10%."
    } ],
    "isError": false } }`,
      },
      wire: { name: "HTTPS to the tool's own API", detail: "Whatever the service speaks: REST, SQL, a file read. MCP standardizes only the hop between the app and the server, not what happens behind it." },
      cost: "Often the slowest hop in the loop, and outside your control. A failing tool comes back as a result with `isError: true`, so the model can react instead of the loop crashing.",
      exhibit: { kind: 'text', title: 'Tool result', caption: 'content[0].text', width: 3.4, text: 'Lisbon.\nSat 14 Nov: 17 °C, rain 80%\nSun 15 Nov: 18 °C, rain 20%\nMon 16 Nov: 19 °C, rain 10%' },
    },
    {
      title: 'The result goes back to the model',
      node: 'model', from: 'app', also: ['mcp'], frame: ['mcp', 'app', 'model'], edges: ['rpcr', 'req'],
      flows: [{ path: ['rpcr', 'req'], count: 5, speed: 2.8 }],
      what: "The app wraps the tool output in a **tool_result** block that points at the request it answers, and calls the model again. The API keeps no memory between calls, so the whole conversation is resent: the question, the assistant's tool request, and now the result.",
      shape: 'messages[3] + tools[1]', dtype: 'JSON', size: 'about 1.2 kB, growing with every turn',
      payload: {
        label: 'POST /v1/messages (second call)', lang: 'json',
        code: `{
  "model": "claude-opus-5-5",
  "max_tokens": 16000,
  "tools": [ ... ],
  "messages": [
    { "role": "user",
      "content": "Do I need an umbrella in Lisbon this weekend?" },
    { "role": "assistant",
      "content": [
        { "type": "thinking", ... },
        { "type": "tool_use",
          "id": "${TOOL_ID}", ... } ] },
    { "role": "user",
      "content": [ {
        "type": "tool_result",
        "tool_use_id": "${TOOL_ID}",
        "content": "Lisbon. Sat 14 Nov: 17 C, rain 80%. ..."
      } ] }
  ]
}`,
      },
      wire: { name: 'JSON over HTTPS', detail: "Tool results travel in a user-role message. The results of several parallel tool calls must all go back together in that one message." },
      cost: "The context grows with every turn of the loop and is read again each time. Long agent runs live or die by prompt caching and by trimming old tool results.",
      exhibit: {
        kind: 'json', title: 'tool_result block', caption: 'role: user',
        code: `{ "type": "tool_result",
  "tool_use_id": "toolu_01A09q90qw…",
  "content": "Lisbon. Sat 14 Nov:
     17 C, rain 80%. Sun 15 Nov: …" }`,
      },
    },
    {
      title: 'No more tools: the final answer',
      node: 'app', from: 'model', also: ['user'], frame: ['user', 'app', 'model'], edges: ['resp', 'ans'],
      flows: [{ path: ['resp', 'ans'], count: 5, speed: 2.8 }],
      what: "With the forecast in its context the model answers in plain text and stops with `end_turn`. Had it needed more, it would have emitted another tool_use block and the loop would have gone around again. **An agent is this loop**: model, tool, model, until the model stops asking.",
      shape: 'content[1]: text', dtype: 'JSON, streamed on as SSE', size: 'about 40 output tokens',
      payload: {
        label: 'response body (shortened)', lang: 'json',
        code: `{
  "role": "assistant",
  "content": [ {
    "type": "text",
    "text": "Yes, for Saturday: rain is likely (80%). Sunday and Monday look mostly dry at 18 to 19 C."
  } ],
  "stop_reason": "end_turn",
  "usage": { "input_tokens": 512,
             "output_tokens": 41 }
}`,
        note: "The token counts are an example.",
      },
      wire: { name: 'JSON over HTTPS, relayed as SSE', detail: "The app streams the text on to the user. Everything in between (two model calls, one tool call) stays invisible unless the interface chooses to show it." },
      cost: "Two model calls and one tool call for one answer. Real agents loop tens or hundreds of times, so cost and latency scale with the number of steps, not with the length of the question.",
      links: [{ label: 'Agents talking to agents (A2A)', href: '#/flow/a2a' }, { label: 'Inside one model call', href: '#/flow/llm' }],
      exhibit: { kind: 'text', title: 'Final answer', caption: 'stop_reason: end_turn', width: 3.9, text: 'Yes, for Saturday: rain is likely (80%). Sunday and Monday look mostly dry at 18 to 19 °C.' },
    },
  ],
};

// ------------------------------------------------------------------ 5. text to image

const diffusion = {
  id: 'diffusion',
  name: 'Text to image',
  short: 'Diffusion',
  tagline: 'A prompt, 77 text vectors, a small latent full of noise, a denoising loop, and a decoder that paints the pixels.',
  model: "Shapes are for a **Stable Diffusion 1.5** class model: a CLIP ViT-L/14 text encoder (77 tokens x 768), an 860M-parameter U-Net that denoises a 4 x 64 x 64 latent, and a VAE that decodes it to 512 x 512 pixels. The pictures are drawn procedurally to stand in for real outputs.",
  graph: {
    nodes: [
      N('user', 'user', 'You', -11, 0),
      N('tok', 'op', 'CLIP tokenizer', -7, -2.5),
      N('clip', 'model', 'Text encoder', -3, -2.5),
      N('noise', 'sampler', 'Random noise\n(seed)', -3, 2.5, { role: 'compute' }),
      N('unet', 'layers', 'U-Net\n(denoiser)', 2, 0, { size: 1.3 }),
      N('sched', 'op', 'Scheduler', 2, 4.6, { role: 'compute' }),
      N('vae', 'model', 'VAE decoder', 6.8, 0),
      N('img', 'image', 'Image', 11.3, 0),
    ],
    edges: [
      E('p', 'user', 'tok', 'prompt · JSON · HTTPS'),
      E('seed', 'user', 'noise', 'seed', { dashed: true }),
      E('ids', 'tok', 'clip', 'ids · int64 [1, 77]'),
      E('cond', 'clip', 'unet', '[2, 77, 768]'),
      E('z', 'noise', 'unet', '[1, 4, 64, 64]'),
      E('eps', 'unet', 'sched', 'predicted noise', { bend: 0.12 }),
      E('step', 'sched', 'unet', 'cleaner latent', { bend: 0.12, labelLift: 0.9 }),
      E('lat', 'unet', 'vae', 'latent [1, 4, 64, 64]'),
      E('px', 'vae', 'img', '[1, 3, 512, 512]'),
    ],
    zones: [{ id: 'gpu', label: 'GPU memory', nodes: ['clip', 'noise', 'unet', 'sched', 'vae'], color: GPU_ZONE }],
  },
  stages: [
    {
      title: 'The prompt is cut to exactly 77 tokens',
      node: 'tok', from: 'user', edges: ['p'],
      what: "The text encoder has a fixed window. The prompt is tokenized, a start token is put in front, and the sequence is padded with end tokens (or truncated) to exactly **77** slots. Whatever the length of your prompt, the model receives 77 ids.",
      shape: '[1, 77]', dtype: 'int64', size: '616 bytes',
      payload: {
        label: 'request, then token ids',
        code: `POST /v1/images/generations
{ "prompt": "a striped hot air balloon over snowy mountains at sunrise",
  "size": "512x512",
  "steps": 30,
  "seed": 42 }

ids = [49406, 320, 10138, 2225, 1474,
       8404, ..., 49407, 49407, ...]
# 49406 = start
# 49407 = end, also used as padding
# 77 ids in total`,
        note: "49406 and 49407 are the real start and end ids of the CLIP tokenizer; the word ids are an example.",
      },
      wire: { name: 'JSON over HTTPS', detail: "A short text and a few numbers go up. An image-generation request is tiny; the response is not." },
      cost: "Negligible. The fixed 77-token window is why very long prompts are cut off in this model family.",
      exhibit: {
        kind: 'tokens', title: 'Prompt tokens', caption: '[1, 77] int64', width: 4.4,
        tokens: [{ t: '<start>', special: true }, { t: 'a' }, { t: 'striped' }, { t: 'hot' }, { t: 'air' }, { t: 'balloon' }, { t: 'over' }, { t: 'snowy' }, { t: 'mountains' }, { t: 'at' }, { t: 'sunrise' }, { t: '<end>', special: true }, { t: '<end> × 65 (padding)', special: true }],
      },
    },
    {
      title: 'The text encoder turns tokens into conditioning',
      node: 'clip', from: 'tok', edges: ['ids'],
      what: "A 12-block transformer (the text half of CLIP) reads the 77 ids and outputs one 768-number vector per token. These vectors are all the image model will ever know about your prompt. A second, empty prompt is encoded as well, for the guidance trick in stage 5.",
      shape: '[2, 77, 768]', dtype: 'float16', size: '231 KiB',
      payload: {
        label: 'pseudocode',
        code: `cond    = text_encoder(ids)
          # [1, 77, 768]
uncond  = text_encoder(ids_of(""))
          # [1, 77, 768]
context = concat(uncond, cond)
          # [2, 77, 768]`,
      },
      wire: { name: 'Host to GPU, then GPU memory', detail: "The ids are copied to the GPU once. From here until the finished image nothing leaves GPU memory." },
      cost: "One pass of a 123M-parameter encoder, run once per image: a rounding error next to the denoising loop.",
      exhibit: { kind: 'vectors', title: 'Text conditioning (6 of 77 rows)', caption: '[2, 77, 768] float16', rows: 6, cols: 26, labels: ['<start>', 'a', 'striped', 'hot', 'air', 'balloon'], more: '… 768' },
    },
    {
      title: 'The canvas is pure noise',
      node: 'noise', from: 'user', edges: ['seed'],
      what: "Generation starts from random numbers, not from a blank image: 4 channels of 64 x 64 values drawn from a normal distribution. The **seed** fixes these numbers, which is why the same seed and prompt reproduce the same picture. The model works in this small **latent** space; a decoder blows it up to pixels at the end.",
      shape: '[1, 4, 64, 64]', dtype: 'float16', size: '32 KiB · 48 times fewer numbers than the image',
      payload: {
        label: 'pseudocode',
        code: `g = Generator("cuda").manual_seed(42)
latent = randn([1, 4, 64, 64], generator=g)

latent.mean(), latent.std()
  # about 0.0 and 1.0
# 16,384 numbers.
# The image will have 786,432.`,
      },
      wire: { name: 'Created in GPU memory', detail: "Only the seed, one integer, came from the request. The noise itself is generated on the GPU." },
      cost: "Nothing. But working at 64 x 64 instead of 512 x 512 is the central trick of latent diffusion: it is what made this affordable on consumer GPUs.",
      exhibit: { kind: 'noise', title: 'Starting latent: noise', caption: '[1, 4, 64, 64] float16', slabs: 4, rows: 16, cols: 16, axes: { x: '64 →', y: '64', depth: '4 channels' } },
    },
    {
      title: 'The U-Net predicts the noise, steered by cross-attention',
      node: 'unet', from: 'clip', also: ['noise'], frame: ['clip', 'noise', 'unet'], edges: ['cond', 'z'],
      what: "The U-Net looks at the noisy latent and predicts which part of it is noise. Inside it, **cross-attention** layers let every position of the picture read the 77 text vectors: this is where the word balloon decides which region becomes a balloon. A timestep value tells the network how noisy its input is.",
      shape: 'in [2, 4, 64, 64] + [2, 77, 768] + t · out [2, 4, 64, 64]', dtype: 'float16',
      payload: {
        label: 'one denoising step',
        code: `x = concat(latent, latent)
    # [2, 4, 64, 64]
eps = unet(x, t, context)
    # [2, 4, 64, 64]

# cross-attention at the 64 x 64 level:
#   queries (image)   [2, 4096, 320]
#   keys, values (text) [2, 77, 768]
#   attention map     [2, 8, 4096, 77]`,
      },
      wire: { ...GPU_MEM, detail: "Latent, timestep and text conditioning are tensors on the same card. The text vectors are reused unchanged at every step." },
      cost: "One U-Net pass is the unit of cost: 860M parameters applied to a batch of 2. Everything else in this pipeline is small beside it.",
      links: [{ label: 'Diffusion Lab', href: '#/labs/diffusion' }],
      exhibit: { kind: 'matrix', title: 'Cross-attention to “balloon” (illustrative)', caption: '[4096, 77] per head', rows: 16, cols: 16, pattern: 'blob', xLabel: 'image positions (shown 16 x 16)' },
    },
    {
      title: 'The denoising loop',
      node: 'sched', from: 'unet', edges: ['eps', 'step'],
      what: "A **scheduler** subtracts a measured part of the predicted noise and hands the slightly cleaner latent back to the U-Net. Repeat 20 to 50 times. Each step blends two predictions, one with the prompt and one without, and pushes the result toward the prompt: that is **classifier-free guidance**.",
      shape: '[1, 4, 64, 64] at every step', dtype: 'float16', size: '30 steps = 60 U-Net evaluations',
      payload: {
        label: 'the loop',
        code: `for t in scheduler.timesteps:  # 30 steps
    e_uncond, e_text = unet(x2, t, context)
    eps = e_uncond + 7.5 * (e_text - e_uncond)
    latent = scheduler.step(eps, t, latent)`,
      },
      wire: { ...GPU_MEM, detail: "The loop is plain Python around two GPU calls. The scheduler holds no weights: it is a formula that decides how much noise to remove at each step." },
      cost: "This loop is the bill: steps x 2 U-Net passes. Fewer steps, distilled models and skipping guidance are the main ways to make image generation faster.",
      exhibit: { kind: 'frames', title: 'The latent, decoded along the way', caption: '30 steps', warm: true, frames: [{ noise: 1, label: 'step 0' }, { noise: 0.7, label: 'step 10' }, { noise: 0.35, label: 'step 20' }, { noise: 0, label: 'step 30' }] },
    },
    {
      title: 'The VAE decodes the latent into pixels',
      node: 'vae', from: 'unet', edges: ['lat'],
      what: "The finished latent is still only 4 x 64 x 64. A convolutional decoder upsamples it by 8 in each direction into three color planes of 512 x 512. It adds no new content: it translates the compact description into pixels.",
      shape: '[1, 3, 512, 512]', dtype: 'float16, values from -1 to 1', size: '1.5 MiB',
      payload: {
        label: 'pseudocode',
        code: `image = vae.decode(latent / 0.18215)
        # [1, 3, 512, 512]
image = (image / 2 + 0.5).clamp(0, 1)
        # now in 0..1`,
      },
      wire: { ...GPU_MEM, detail: "One more model on the same card. The constant 0.18215 undoes a scaling that was applied when the latent space was trained." },
      cost: "One pass of the decoder half of an 84M-parameter autoencoder. Small next to the loop, but it holds the largest activations: at high resolutions it is often what runs out of memory.",
      exhibit: { kind: 'tensor', mode: 'image', warm: true, title: 'Decoded image tensor', caption: '[1, 3, 512, 512] float16', slabs: 3, rows: 18, cols: 18, axes: { x: '512 →', y: '512', depth: 'R, G, B' } },
    },
    {
      title: 'Pixels are encoded and sent back',
      node: 'img', from: 'vae', edges: ['px'],
      what: "The float tensor is rounded to 8-bit integers, rearranged to height x width x color, and compressed as a PNG or JPEG. The file returns as base64 text inside JSON, as raw bytes, or as a link to object storage.",
      shape: '[512, 512, 3]', dtype: 'uint8', size: '768 KiB raw · a few hundred kB as PNG',
      payload: {
        label: 'response', lang: 'json',
        code: `{
  "created": 1794441600,
  "data": [ {
    "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAA..."
  } ]
}`,
      },
      wire: { name: 'PNG as base64 in JSON, over HTTPS', detail: "Base64 inflates the file by a third. Services that produce many images return a short-lived URL to object storage instead." },
      cost: "Milliseconds of encoding, and a response thousands of times larger than the request. End to end, almost all of the time went to the denoising loop.",
      exhibit: { kind: 'image', title: 'Image', caption: '512 x 512 · RGB · uint8', n: 28, size: 2.5, warm: true, note: 'drawn procedurally: a stand-in for the output' },
    },
  ],
};

// ------------------------------------------------------------------ 6. image classification

const vision = {
  id: 'vision',
  name: 'Seeing an image',
  short: 'Vision',
  tagline: 'A photo becomes patches, patches become a sequence, and a transformer reads it like a sentence.',
  model: "Shapes are for a **ViT-B/16** class image classifier (86M parameters: 12 blocks, width 768, 16 x 16 patches, 224 x 224 input, the 1,000 ImageNet classes). The photo is a procedural stand-in and the scores are an example.",
  graph: {
    nodes: [
      N('cam', 'device', 'Your phone', -6.6, -2.5),
      N('pre', 'op', 'Decode, resize,\nnormalize', -2.2, -2.5),
      N('patch', 'op', 'Cut into\npatches', 2.2, -2.5, { role: 'compute' }),
      N('emb', 'table', 'Patch embedding\n+ position', 6.6, -2.5, { role: 'compute' }),
      N('blocks', 'layers', '12 encoder\nblocks', 6.6, 2.5),
      N('head', 'op', 'Classifier\nhead', 2.2, 2.5, { role: 'compute' }),
      N('resp', 'service', 'Response', -2.2, 2.5),
    ],
    edges: [
      E('up', 'cam', 'pre', 'JPEG · multipart upload'),
      E('px', 'pre', 'patch', 'tensor · gRPC'),
      E('pat', 'patch', 'emb', '[1, 196, 768]'),
      E('seq', 'emb', 'blocks', '[1, 197, 768]', { labelLift: 0.3 }),
      E('cls', 'blocks', 'head', 'class token [1, 768]'),
      E('lg', 'head', 'resp', 'scores [1, 1000]'),
      E('back', 'resp', 'cam', 'JSON · HTTPS'),
    ],
    zones: [
      { id: 'cpu', label: 'Preprocessing (CPU)', nodes: ['pre', 'resp'] },
      { id: 'gpu', label: 'Model server (GPU)', nodes: ['patch', 'emb', 'blocks', 'head'], color: GPU_ZONE },
    ],
  },
  stages: [
    {
      title: 'A photo is already a tensor',
      node: 'cam',
      what: "A camera sensor measures red, green and blue light on a grid. A 12-megapixel photo is 3,024 rows by 4,032 columns of three 8-bit numbers: 36 million values. JPEG compression shrinks that about tenfold before it leaves the phone.",
      shape: '[3024, 4032, 3]', dtype: 'uint8', size: '36.6 MB raw · about 3 MB as JPEG',
      payload: {
        label: 'the file',
        code: `balloon.jpg
  JPEG, 4032 x 3024, 8 bits per channel
  2.9 MB on disk        (example)
  36.6 MB once decoded to RGB`,
      },
      wire: { name: 'Sensor to file: JPEG', detail: "JPEG stores brightness at full resolution and color at half, then discards detail the eye barely notices, block by block. It is the wire format of nearly every photo." },
      cost: "On the device: nothing you wait for. The size of this file is what dominates the next hop.",
      exhibit: { kind: 'image', title: 'Photo', caption: 'JPEG · 4032 x 3024', n: 28, size: 2.4, note: 'procedural stand-in for a photo' },
    },
    {
      title: 'Upload, decode, resize, normalize',
      node: 'pre', from: 'cam', edges: ['up'],
      what: "The server decodes the JPEG back to pixels, shrinks the picture to the 224 x 224 the model was trained on, converts integers to floats and rescales them to the range -1 to 1. Color moves to the front of the shape and a batch dimension is added.",
      shape: '[1, 3, 224, 224]', dtype: 'float32', size: '588 KiB',
      payload: {
        label: 'request, then preprocessing',
        code: `POST /v1/classify
content-type: multipart/form-data; boundary=x7f3

--x7f3
content-disposition: form-data; name="image"; filename="balloon.jpg"
content-type: image/jpeg

<2.9 MB of JPEG bytes>
--x7f3--

img = decode_jpeg(body)       # [3024, 4032, 3]
img = resize(img, (224, 224)) # [224, 224, 3]
x = (img / 255 - 0.5) / 0.5   # float32, -1..1
x = x.transpose(2, 0, 1)[None]
                              # [1, 3, 224, 224]`,
      },
      wire: { name: 'multipart/form-data over HTTPS', detail: "A binary upload: the JPEG bytes travel as they are, between text boundaries. The alternative, base64 inside JSON, is simpler to handle and a third larger." },
      cost: "For a model this small, the upload and the JPEG decode usually cost more than the neural network. The resize also discards 99.6% of the pixels: fine detail is gone before the model starts.",
      exhibit: { kind: 'tensor', mode: 'image', title: 'pixel_values', caption: '[1, 3, 224, 224] float32', slabs: 3, rows: 16, cols: 16, axes: { x: '224 →', y: '224', depth: 'R, G, B' } },
    },
    {
      title: 'The image is cut into 196 patches',
      node: 'patch', from: 'pre', edges: ['px'],
      what: "A transformer needs a sequence. The picture is cut into a 14 x 14 grid of 16 x 16-pixel **patches**, and each patch is flattened into 768 numbers (16 x 16 x 3). Patches play the role that tokens play in text: the image becomes a sentence of 196 patches.",
      shape: '[1, 196, 768]', dtype: 'float32', size: '588 KiB (the same numbers, regrouped)',
      payload: {
        label: 'the tensor as it arrives (Open Inference Protocol)', lang: 'json',
        code: `{
  "inputs": [ {
    "name": "pixel_values",
    "shape": [1, 3, 224, 224],
    "datatype": "FP32",
    "data": [0.412, 0.404, 0.388, ...]
  } ]
}`,
        note: "Shown as JSON to be readable. Over gRPC the same fields are protobuf and the 150,528 floats travel as raw bytes.",
      },
      wire: { name: 'gRPC with protobuf', detail: "Between the preprocessing service and the model server: the Open Inference Protocol that KServe and Triton speak. Names, shapes and types are declared, and tensors go as packed binary over HTTP/2." },
      cost: "Cutting patches is a reshape: free. The price of this hop is one serialization and one round trip inside the data center.",
      exhibit: { kind: 'image', title: '14 x 14 patches', caption: '[1, 196, 768] float32', n: 28, size: 2.4, patch: 2, patchGap: 0.05, note: 'each tile is one 16 x 16-pixel patch' },
    },
    {
      title: 'Patches become embeddings, plus a class token',
      node: 'emb', from: 'patch', edges: ['pat'],
      what: "One learned linear layer maps every flattened patch to the model's width. A special learned vector, the **class token**, is put in front: it will collect a summary of the whole image. Because attention ignores order, a learned **position embedding** is added so the model knows where each patch came from.",
      shape: '[1, 197, 768]', dtype: 'float32', size: '591 KiB',
      payload: {
        label: 'pseudocode',
        code: `tokens = patches @ W_patch + b
         # [1, 196, 768]
x = concat(cls_token, tokens)
         # [1, 197, 768]
x = x + position_embedding
         # [1, 197, 768]`,
      },
      wire: { ...GPU_MEM, detail: "In practice the patch cutting and this projection are one operation: a convolution with a 16 x 16 kernel and a stride of 16." },
      cost: "One small matrix multiplication. Sequence length is the thing to watch: doubling the image side quadruples the patches, and attention cost grows with the square of that.",
      exhibit: { kind: 'vectors', title: 'Token sequence (5 of 197 rows)', caption: '[1, 197, 768] float32', rows: 5, cols: 26, labels: ['[class]', 'patch 1', 'patch 2', 'patch 3', 'patch 196'], more: '… 768' },
    },
    {
      title: 'Twelve blocks relate every patch to every other',
      node: 'blocks', from: 'emb', edges: ['seq'],
      what: "The same block as in a language model, with one difference: there is no causal mask. Every patch can attend to every other patch in both directions, so the balloon's patches and the sky's patches inform one another. The class token attends to all of them.",
      shape: '[1, 197, 768] in and out of every block', dtype: 'float32', size: 'attention: [1, 12, 197, 197] per block',
      payload: {
        label: 'pseudocode',
        code: `for block in blocks:      # 12 of them
    x = x + attention(norm(x))
    #     12 heads, no mask
    x = x + mlp(norm(x))
    #     768 -> 3072 -> 768`,
      },
      wire: { ...GPU_MEM, detail: "Tensors move from block to block as pointers." },
      cost: "This is the model: nearly all of the 86M parameters and of the arithmetic. It is still a single pass with no loop, which is why classifying an image is far cheaper than writing a paragraph.",
      links: [{ label: 'How a CNN sees: the other way to read images', href: '#/labs/cnn' }, { label: 'Inside a GPT', href: '#/labs/gpt' }],
      exhibit: { kind: 'matrix', title: 'What the class token attends to (illustrative)', caption: '14 x 14 patches', rows: 14, cols: 14, pattern: 'blob', xLabel: 'one tile per patch' },
    },
    {
      title: 'The class token is read out as 1,000 scores',
      node: 'head', from: 'blocks', edges: ['cls'],
      what: "After the last block only the first row is used: the class token. One linear layer maps its 768 numbers to a score for each of the 1,000 classes the model was trained on, and softmax turns the scores into probabilities.",
      shape: '[1, 1000]', dtype: 'float32', size: '4 kB',
      payload: {
        label: 'pseudocode and example result',
        code: `cls = norm(x)[:, 0]        # [1, 768]
logits = cls @ W_head + b  # [1, 1000]
probs = softmax(logits)

417  balloon     0.93
701  parachute   0.03
405  airship     0.02
... 997 more classes`,
        note: "The class numbers are the real ImageNet-1k indices; the probabilities are an example.",
      },
      wire: { ...GPU_MEM, detail: "The 196 patch rows are simply dropped here. Other tasks keep them: detection and segmentation heads read every patch." },
      cost: "A 768 x 1,000 multiplication: nothing. The fixed list of classes is the real limit, and the reason open-vocabulary models compare images with text embeddings instead.",
      exhibit: {
        kind: 'bars', title: 'Class probabilities (example)', caption: '[1, 1000] float32', barWidth: 0.68, tail: 14, tailLabel: '… 995 more classes',
        items: [{ label: 'balloon', value: 0.93, chosen: true }, { label: 'parachute', value: 0.03 }, { label: 'airship', value: 0.02 }, { label: 'alp', value: 0.01 }, { label: 'ski', value: 0.005 }],
      },
    },
    {
      title: 'A few bytes go back',
      node: 'resp', from: 'head', also: ['cam'], frame: ['head', 'resp', 'cam'], edges: ['lg', 'back'], ref: 'webgpu',
      flows: [{ path: ['lg', 'back'], count: 5, speed: 2.8 }],
      what: "The server keeps the top few classes, looks up their names and replies with a small JSON document. Megabytes went up; a hundred or so bytes come down.",
      shape: 'top 3 of 1,000', dtype: 'JSON', size: 'about 200 bytes',
      payload: {
        label: 'response', lang: 'json',
        code: `{ "predictions": [
  { "label": "balloon",
    "class_id": 417, "score": 0.93 },
  { "label": "parachute",
    "class_id": 701, "score": 0.03 },
  { "label": "airship",
    "class_id": 405, "score": 0.02 }
] }`,
      },
      wire: { name: 'JSON over HTTPS', detail: "Results are small, structured and read by programs: plain JSON is the natural fit." },
      cost: "End to end, the network and the JPEG decode usually outweigh the model. That is the argument for running small vision models on the device itself.",
      links: [{ label: 'Deployment modes', href: '#/deploy' }],
      exhibit: {
        kind: 'json', title: 'Response', caption: 'JSON',
        code: `{ "predictions": [
  { "label": "balloon",   "score": 0.93 },
  { "label": "parachute", "score": 0.03 },
  { "label": "airship",   "score": 0.02 }
] }`,
      },
    },
  ],
};

// ------------------------------------------------------------------ 7. speech to text

const speech = {
  id: 'speech',
  name: 'Speech to text',
  short: 'Speech',
  tagline: 'A waveform becomes a picture of sound, an encoder listens, a decoder writes.',
  model: "Shapes are for a **Whisper small** class model (244M parameters: 12 encoder and 12 decoder blocks, width 768, 80 mel bands, a vocabulary of 51,865). The audio is a procedural stand-in; the transcript and its word ids are an example.",
  graph: {
    nodes: [
      N('mic', 'mic', 'Microphone', -6.6, -2.5),
      N('cap', 'device', 'Your device', -2.2, -2.5),
      N('mel', 'op', 'Log-mel\nspectrogram', 2.2, -2.5),
      N('enc', 'layers', 'Audio encoder', 6.6, -2.5),
      N('dec', 'model', 'Text decoder', 11, -2.5),
      N('samp', 'sampler', 'Token search', 11, 2.5, { role: 'compute' }),
      N('txt', 'op', 'Detokenizer', 6.6, 2.5),
    ],
    edges: [
      E('pcm', 'mic', 'cap', 'PCM samples'),
      E('up', 'cap', 'mel', 'audio file · multipart upload'),
      E('spec', 'mel', 'enc', '[1, 80, 3000] f32'),
      E('mem', 'enc', 'dec', 'audio states [1, 1500, 768]'),
      E('lg', 'dec', 'samp', 'scores [1, 51865]', { bend: 0.12 }),
      E('nxt', 'samp', 'dec', 'next id', { bend: 0.12, labelLift: 0.9 }),
      E('ids', 'samp', 'txt', 'token ids'),
      E('back', 'txt', 'cap', 'JSON · text'),
    ],
    zones: [
      { id: 'dev', label: 'Your device', nodes: ['mic', 'cap'], color: DEVICE_ZONE },
      { id: 'srv', label: 'Speech server', nodes: ['mel', 'enc', 'dec', 'samp', 'txt'], color: GPU_ZONE },
    ],
  },
  stages: [
    {
      title: 'Sound becomes a list of numbers',
      node: 'cap', from: 'mic', edges: ['pcm'],
      what: "A microphone measures air pressure tens of thousands of times per second. Speech models expect **16,000 samples per second on one channel**, so the recording is downmixed and resampled: five seconds of speech is 80,000 numbers. Whisper-class models always work on a 30-second window, so the clip is later padded with silence to 480,000 samples.",
      shape: '[80000], later padded to [480000]', dtype: 'float32 (int16 in the file)', size: '160 kB as 16-bit PCM',
      payload: {
        label: 'pseudocode',
        code: `raw = record(seconds=5)
      # 48 kHz, stereo, int16
mono = raw.mean(axis=1)       # [240000]
audio = resample(mono, 48000, 16000)
                              # [80000]
audio = audio / 32768         # -1..1

audio[:4]
  # [0.0003, -0.0011, 0.0042, 0.0107]`,
        note: "The sample values are an example.",
      },
      wire: { name: 'PCM samples from the audio driver', detail: "Uncompressed pulse-code modulation: just the numbers, in order. Everything else in audio (WAV, FLAC, Opus) is a container for this or a compression of it." },
      cost: "Real time: recording five seconds takes five seconds. For voice interfaces this, not the model, sets the floor on latency, which is why live systems stream audio in small chunks.",
      exhibit: { kind: 'waveform', title: 'Waveform', caption: '16 kHz · mono · float32', left: '0 s', right: '30 s', note: 'speech, then padding' },
    },
    {
      title: 'Upload, then a picture of the sound',
      node: 'mel', from: 'cap', edges: ['up'],
      what: "The audio file is uploaded and decoded on the server. A short-time Fourier transform slides a 25 ms window along the samples in 10 ms hops and measures how much energy each frequency carries. The result is squeezed into 80 **mel** bands, spaced the way hearing works, and put on a log scale. Sound has become a picture: 80 rows by 3,000 columns.",
      shape: '[1, 80, 3000]', dtype: 'float32', size: '938 KiB',
      payload: {
        label: 'request, then feature extraction',
        code: `POST /v1/audio/transcriptions
content-type: multipart/form-data; boundary=b41c

--b41c
content-disposition: form-data; name="file"; filename="clip.flac"
content-type: audio/flac

<FLAC bytes: about half the size of the PCM>
--b41c
content-disposition: form-data; name="model"

whisper-small
--b41c--

audio = pad_or_trim(decode(file), 480000)
mel = log_mel(audio, n_fft=400,
              hop=160, n_mels=80)
mel.shape  # [80, 3000]`,
      },
      wire: { name: 'multipart/form-data over HTTPS', detail: "A compressed audio file (FLAC, Opus, MP3 or WAV) as a binary upload, with the model name as a second form field. Live transcription streams small chunks over a WebSocket or WebRTC instead." },
      cost: "The transform is cheap signal processing. The fixed window is the notable cost: a 5-second clip is processed as a full 30 seconds, mostly silence.",
      exhibit: { kind: 'spectrogram', title: 'Log-mel spectrogram', caption: '[1, 80, 3000] float32', xLabel: 'time: 3,000 frames of 10 ms →', yLabel: '80 mel bands' },
    },
    {
      title: 'The encoder listens to the whole clip at once',
      node: 'enc', from: 'mel', edges: ['spec'],
      what: "Two small convolutions halve the time axis and lift the 80 bands to the model's width, then 12 transformer blocks let every moment attend to every other. The result is 1,500 vectors, one for every 20 ms of audio: what was said, stripped of the raw signal.",
      shape: '[1, 1500, 768]', dtype: 'float32 (often float16 on the GPU)', size: '4.4 MiB',
      payload: {
        label: 'pseudocode',
        code: `x = gelu(conv1(mel))  # [1, 768, 3000]
x = gelu(conv2(x))    # [1, 768, 1500]
                      # stride 2
x = x.transpose(1, 2) + positions
                      # [1, 1500, 768]
for block in encoder_blocks:  # 12
    x = block(x)      # full attention`,
      },
      wire: { name: 'Host to GPU, then GPU memory', detail: "The spectrogram is copied to the GPU once. The encoder output stays on the card for the decoder to read." },
      cost: "A fixed price per 30-second window, however much speech is in it. The encoder runs once; the decoder that follows runs once per output token.",
      exhibit: { kind: 'vectors', title: 'Audio states (5 of 1,500 rows)', caption: '[1, 1500, 768]', rows: 5, cols: 26, monoLabels: true, labels: ['0.00 s', '0.02 s', '0.04 s', '0.06 s', '29.98 s'], more: '… 768' },
    },
    {
      title: 'The decoder is told its task with tokens',
      node: 'dec', from: 'enc', edges: ['mem'],
      what: "The decoder is a small language model that also looks at the audio. It starts from a prompt of special tokens that say what to do: begin, which language, transcribe rather than translate, with or without timestamps. In every block, **cross-attention** reads the 1,500 audio vectors.",
      shape: '[1, 4] tokens + [1, 1500, 768] audio states', dtype: 'int64 and float',
      payload: {
        label: 'the decoder prompt',
        code: `<|startoftranscript|>   50258
<|en|>                  50259
<|transcribe|>          50359
<|notimestamps|>        50363

logits = decoder(tokens, audio_states)
         # [1, 4, 51865]`,
        note: "These four ids are the real ones of the multilingual Whisper vocabulary.",
      },
      wire: { ...GPU_MEM, detail: "The audio states are computed once and reused at every step: their cross-attention keys and values are cached, like the KV cache of a chat model." },
      cost: "One model, many tasks: swapping a single token switches between transcription, translation and language detection without changing any weights.",
      links: [{ label: 'KV cache experiment', href: '#/perf/kvcache' }],
      exhibit: {
        kind: 'tokens', title: 'Decoder prompt', caption: '[1, 4] int64', show: 'both', width: 4.6,
        tokens: [{ t: '<|startoftranscript|>', id: 50258, special: true }, { t: '<|en|>', id: 50259, special: true }, { t: '<|transcribe|>', id: 50359, special: true }, { t: '<|notimestamps|>', id: 50363, special: true }],
      },
    },
    {
      title: 'Text is written one token at a time',
      node: 'samp', from: 'dec', edges: ['lg', 'nxt'],
      what: "From here it works like any language model: scores over a vocabulary of 51,865, pick a token, feed it back, repeat. Speech recognition wants the most likely transcript rather than a creative one, so decoding is greedy or a **beam search** that keeps a few candidate sentences alive.",
      shape: '[1, 51865] per step', dtype: 'float32', size: '203 KiB per step',
      payload: {
        label: 'the loop (greedy form)',
        code: `tokens = [50258, 50259, 50359, 50363]
while True:
    logits = decoder(tokens, audio_states)
    next_id = argmax(logits[:, -1])
    if next_id == 50257:   # <|endoftext|>
        break
    tokens.append(next_id)`,
      },
      wire: { ...GPU_MEM, detail: "Only token ids loop back into the decoder. With beam search the batch dimension becomes the beam width: five candidates means five rows." },
      cost: "One decoder pass per token: about 10 passes for this sentence, hundreds for 30 seconds of fast speech. The decoder, not the encoder, dominates long transcripts.",
      exhibit: {
        kind: 'bars', title: 'Next token (example)', caption: '[1, 51865] float32', barWidth: 0.6, tail: 14, tailLabel: '… 51,861 more',
        items: [{ label: 'Turn', value: 0.91, chosen: true }, { label: 'Turned', value: 0.04 }, { label: 'Tern', value: 0.02 }, { label: 'Burn', value: 0.01 }],
      },
    },
    {
      title: 'Ids become words again',
      node: 'txt', from: 'samp', edges: ['ids'],
      what: "The generated ids are mapped back to their byte sequences and joined into text, and the special tokens are dropped. With timestamps switched on, extra tokens such as `<|2.40|>` are interleaved with the words and mark where each phrase begins and ends.",
      shape: '[1, 9]', dtype: 'int64', size: '72 bytes',
      payload: {
        label: 'tokens, then text',
        code: `Turn     7956     off      766
the       264     kitchen 6525
lights   5811     ,         11
please   1767     .         13
<|endoftext|>   50257

text = "Turn off the kitchen lights, please."`,
        note: "50257 is the real end-of-text id; treat the word ids as an example.",
      },
      wire: { name: 'GPU to host, in-process', detail: "A handful of integers leave the GPU. Detokenizing is a table lookup on the CPU." },
      cost: "Negligible. Long recordings are processed as a series of 30-second windows, each conditioned on the text of the one before, and the pieces are stitched together here.",
      exhibit: {
        kind: 'tokens', title: 'Output tokens', caption: '[1, 9] int64', show: 'both', width: 4.6,
        tokens: [{ t: '·Turn', id: 7956 }, { t: '·off', id: 766 }, { t: '·the', id: 264 }, { t: '·kitchen', id: 6525 }, { t: '·lights', id: 5811 }, { t: ',', id: 11 }, { t: '·please', id: 1767 }, { t: '.', id: 13 }, { t: '<|endoftext|>', id: 50257, special: true }],
      },
    },
    {
      title: 'The transcript returns',
      node: 'cap', from: 'txt', edges: ['back'],
      what: "The server answers with the text, and optionally with segments and timestamps. What went up as tens of kilobytes of audio comes back as a few dozen bytes: speech recognition is an extreme compression down to meaning.",
      shape: '1 string · 36 characters', dtype: 'UTF-8 text in JSON', size: 'about 200 bytes',
      payload: {
        label: 'response', lang: 'json',
        code: `{
  "text": "Turn off the kitchen lights, please.",
  "language": "en",
  "duration": 5.0,
  "segments": [ {
    "start": 0.0, "end": 2.4,
    "text": "Turn off the kitchen lights, please."
  } ]
}`,
      },
      wire: { name: 'JSON over HTTPS', detail: "One small document. Streaming transcription sends partial text as server-sent events or WebSocket messages while the speaker is still talking." },
      cost: "The transcript is often not the end: in a voice assistant it becomes the user message of a chat request, and the reply goes through a text-to-speech model on the way back.",
      links: [{ label: 'What a chat model does with it', href: '#/flow/llm' }, { label: 'Run models on the device', href: '#/deploy' }],
      exhibit: { kind: 'text', title: 'Transcript', caption: 'text · 36 chars', width: 3.6, text: 'Turn off the kitchen lights, please.' },
    },
  ],
};

// ------------------------------------------------------------------ 8. the serving path

const serve = {
  id: 'serve',
  name: 'The serving path',
  short: 'Serving',
  tagline: 'What stands between a request and a GPU: gateway, tokenizer, scheduler, continuous batching, and the stream back.',
  model: "A **Llama 3 8B** class model behind an OpenAI-compatible endpoint, served by an engine of the vLLM / SGLang kind. Request sizes, batch sizes and timings are an example that shows the arithmetic; real values depend on the model, the hardware and the traffic.",
  graph: {
    nodes: [
      N('client', 'browser', 'Your app', -11, 0),
      N('gw', 'gateway', 'Gateway\n+ router', -6.5, 0),
      N('tok', 'op', 'Tokenizer\n+ admission', -2, 0),
      N('sched', 'queue', 'Scheduler', 2.5, 0),
      N('gpu', 'gpu', 'GPU', 7, 0, { size: 1.2 }),
      N('out', 'op', 'Detokenize\n+ stream', 2.5, -4.2),
      N('rb', 'service', 'Replica B', -2, 4.6),
      N('rc', 'service', 'Replica C', 2.5, 4.6),
    ],
    edges: [
      E('req', 'client', 'gw', 'JSON · HTTPS', { bend: 0.1 }),
      E('sse', 'gw', 'client', 'SSE', { bend: 0.1 }),
      E('route', 'gw', 'tok', 'HTTP/2'),
      E('toB', 'gw', 'rb', '', { dashed: true }),
      E('toC', 'gw', 'rc', '', { dashed: true }),
      E('enq', 'tok', 'sched', 'ids · int64 [1, 812]'),
      E('run', 'sched', 'gpu', 'batch of 24'),
      E('toks', 'gpu', 'out', 'new ids, every step'),
      E('back', 'out', 'gw', 'text pieces'),
    ],
    zones: [
      { id: 'engine', label: 'One engine replica', nodes: ['tok', 'sched', 'gpu', 'out'], color: GPU_ZONE },
      { id: 'more', label: 'More replicas, added under load', nodes: ['rb', 'rc'] },
    ],
  },
  stages: [
    {
      title: 'The gateway authenticates, meters and routes',
      node: 'gw', from: 'client', edges: ['req'], ref: 'engines',
      what: "Before any GPU is involved, a gateway checks the API key, applies rate limits counted in tokens per minute, and picks a replica of the model to send the request to. Smart routers prefer the replica that already holds this conversation's prefix in its KV cache.",
      shape: '1 HTTP request', dtype: 'JSON body + headers', size: 'about 4 kB in this example',
      payload: {
        label: 'request',
        code: `POST /v1/chat/completions
authorization: Bearer sk-...
content-type: application/json

{ "model": "llama-3-8b-instruct",
  "messages": [
    ... 6 earlier turns ...,
    { "role": "user",
      "content": "And in winter?" } ],
  "max_tokens": 512,
  "stream": true,
  "stream_options": {
    "include_usage": true } }`,
      },
      wire: { name: 'JSON over HTTPS', detail: "TLS ends at the gateway. Inside the cluster the request continues over HTTP/2 or gRPC on a private network." },
      cost: "About a millisecond of work. Under overload, this is where requests wait or are turned away with a 429 instead of piling up on the GPUs.",
      exhibit: {
        kind: 'json', title: 'Incoming request', caption: 'JSON', bar: 'POST /v1/chat/completions',
        code: `authorization: Bearer sk-…
{ "model": "llama-3-8b-instruct",
  "messages": [ … 7 messages … ],
  "max_tokens": 512,
  "stream": true }`,
      },
    },
    {
      title: 'Tokenize first, then decide',
      node: 'tok', from: 'gw', edges: ['route'],
      what: "The engine applies the chat template and tokenizes on the CPU, because every later decision is made in tokens: does the prompt plus `max_tokens` fit in the context window, how much KV-cache memory must be reserved, and does a cached prefix already cover part of this prompt?",
      shape: '[1, 812]', dtype: 'int64', size: '6.5 kB',
      payload: {
        label: 'what the engine now knows (example)',
        code: `prompt tokens      812
max_tokens         512
context limit     8192   fits
cached prefix      768   (48 blocks of 16)
left to prefill     44`,
      },
      wire: { name: 'HTTP/2 inside the cluster', detail: "Still the same JSON request, forwarded by the router. Tokenization happens in the engine process, next to the scheduler." },
      cost: "Around a millisecond of CPU per request. On a busy server, tokenization runs in a thread pool so that it never stalls the loop that feeds the GPU.",
      exhibit: {
        kind: 'json', title: 'Admission check (example)', caption: '[1, 812] int64',
        code: `prompt tokens      812
max_tokens         512
context limit     8192   fits
cached prefix      768
left to prefill     44`,
      },
    },
    {
      title: 'The scheduler: continuous batching',
      node: 'sched', from: 'tok', edges: ['enq'],
      what: "A GPU is fastest when it processes many sequences at once. The scheduler keeps a set of running requests and, **at every decoding step**, lets waiting requests join as soon as memory allows and removes those that have finished. Nobody waits for a batch to fill up or for the slowest request to end.",
      shape: '24 running sequences of different lengths', dtype: 'a queue + a table of KV-cache blocks',
      payload: {
        label: 'scheduler state at one step (example)',
        code: `step 4117
running  24 requests  (KV blocks: 91% used)
waiting   3 requests
 + admit  req_8c1f  812 tokens, 44 to prefill
 - finish req_77a0  164 tokens, then <|eot_id|>
next step: 23 decodes + 1 prefill, one batch`,
      },
      wire: { name: 'In-process queue', detail: "Requests wait as objects that hold token ids. Nothing is serialized; the scheduler hands the GPU index tensors that say which sequences run in this step." },
      cost: "**Queue time** is the part of the wait that grows without bound under load. Batching is what makes serving affordable, and it is the first thing a request feels when the server is full.",
      links: [{ label: 'Batching experiment', href: '#/perf/batching' }],
      exhibit: { kind: 'batch', title: 'Continuous batching (one color per request)', caption: 'slots x steps', slots: 6, steps: 28 },
    },
    {
      title: 'Prefill: the prompt in one pass',
      node: 'gpu', from: 'sched', edges: ['run'],
      what: "The new request's uncached prompt tokens go through the model together, in parallel. This fills its KV cache and produces the first output token. Engines allocate the cache in small fixed blocks, like pages of virtual memory, so that sequences of any length can share one GPU without fragmentation.",
      shape: '[1, 44, 4096] through 32 blocks', dtype: 'float16', size: 'KV cache: 812 tokens x 128 KiB = about 102 MiB',
      payload: {
        label: 'what runs (example)',
        code: `prefill req_8c1f
  reuse    48 cached KV blocks (768 tokens)
  compute  44 new tokens -> 3 more blocks
  output   1 token: the first of the answer`,
      },
      wire: { name: 'GPU memory', detail: "Token ids are copied to the GPU; hidden states and the KV cache never leave it. The blocks of a finished request are freed for the next one." },
      cost: "Prefill sets the **time to first token**: it grows with the length of the uncached prompt. A cached prefix skips most of it, which is why prompt caching cuts latency as well as price.",
      links: [{ label: 'KV cache experiment', href: '#/perf/kvcache' }],
      exhibit: { kind: 'cache', title: 'KV cache of this request', caption: '812 tokens · 102 MiB', layers: 6, slots: 26, filled: 22, first: 'block 1', last: 'block 32', filledLabel: '768 tokens reused + 44 computed', freeLabel: 'room for 512 new tokens →' },
    },
    {
      title: 'Decode: one token for everyone, every step',
      node: 'gpu', from: 'sched', edges: ['run'], ref: 'throughput',
      what: "From now on each step feeds one new token per running request through the model: a batch of 24 single-token rows, each reading its own KV cache. A step takes barely longer than it would for one request, so total throughput multiplies while each user's speed drops only a little.",
      shape: '[24, 1, 4096] in · [24, 128256] out', dtype: 'float16', size: '24 new tokens per step',
      payload: {
        label: 'one step (example batch)',
        code: `ids     [24, 1]        last token of each
hidden  [24, 1, 4096]  through 32 blocks
logits  [24, 128256]   one row per request
sample  [24]           24 new tokens`,
      },
      wire: { name: 'GPU memory', detail: "One forward pass serves all running requests. Only the 24 sampled ids come back to the CPU after each step." },
      cost: "Decoding is limited by memory bandwidth: the weights are read once per step whether 1 or 24 requests share it. That is why batching is nearly free, until the KV caches fill the card.",
      links: [{ label: 'Batching experiment', href: '#/perf/batching' }, { label: 'Quantization', href: '#/perf/quant' }],
      exhibit: { kind: 'vectors', title: 'One decode step: one new row per request', caption: '[24, 1, 4096] float16', rows: 5, cols: 26, monoLabels: true, labels: ['req_8c1f', 'req_2b90', 'req_5d07', 'req_c3aa', '+ 20 more'], more: '… 4,096' },
    },
    {
      title: 'Each token is detokenized and framed',
      node: 'out', from: 'gpu', edges: ['toks'],
      what: "After every step the engine takes each request's new id, turns it into text (holding back incomplete characters), checks the stop conditions and writes one event to that request's open response. Twenty-four independent streams advance in lockstep.",
      shape: '1 text piece per request per step', dtype: 'UTF-8 in JSON', size: 'about 60 bytes per event',
      payload: {
        label: 'events of this request',
        code: `data: {"choices":[{"delta":{"content":"In"}}]}

data: {"choices":[{"delta":{"content":" winter"}}]}

data: {"choices":[{"delta":{"content":" the"}}]}`,
      },
      wire: { name: 'Token ids from GPU to CPU', detail: "A few integers per step cross from GPU memory to the host. Text exists only from this point on." },
      cost: "Tiny per token, but it runs on the CPU for every stream at every step. A slow loop here can cap a fast GPU, which is why engines move this path to native code.",
      exhibit: {
        kind: 'json', title: 'Stream of one request', caption: 'text/event-stream', cols: 52,
        code: `data: {"choices":[{"delta":{"content":"In"}}]}

data: {"choices":[{"delta":{"content":" winter"}}]}

data: {"choices":[{"delta":{"content":" the"}}]}`,
      },
    },
    {
      title: 'The stream passes back through the gateway',
      node: 'gw', from: 'out', also: ['client'], frame: ['client', 'gw', 'out'], edges: ['back', 'sse'],
      flows: [{ path: ['back', 'sse'], count: 6, speed: 3.2 }],
      what: "The gateway relays the events to the client without buffering them, and counts tokens for billing as they pass. The last event carries the usage totals, then the stream closes. A client that disconnects early should cancel the request all the way down, freeing its slot in the batch.",
      shape: 'about 164 events', dtype: 'SSE over HTTPS', size: 'about 10 kB in total',
      payload: {
        label: 'the end of the stream',
        code: `data: {"choices":[{"delta":{},
  "finish_reason":"stop"}]}

data: {"choices":[],"usage":{
  "prompt_tokens":812,
  "completion_tokens":164,
  "total_tokens":976}}

data: [DONE]`,
        note: "Each real `data:` payload is a single line; they are wrapped here to fit.",
      },
      wire: { name: 'Server-sent events (SSE)', detail: "Proxies and load balancers must not buffer this response, or the user sees nothing and then everything at once. Buffering switched off and long timeouts are the classic configuration fixes." },
      cost: "Network only. Billing happens here: 812 input tokens and 164 output tokens, priced separately because they cost the provider very different amounts of GPU time.",
      exhibit: {
        kind: 'json', title: 'Last events', caption: 'usage totals', cols: 50,
        code: `data: {"choices":[{"finish_reason":"stop"}]}

data: {"usage":{"prompt_tokens":812,
  "completion_tokens":164}}

data: [DONE]`,
      },
    },
    {
      title: 'What the user feels: two numbers',
      node: 'client', from: 'gw', edges: ['sse'],
      what: "**Time to first token** is everything before the first event: network, queue and prefill. **Tokens per second** is the pace after that: one decode step per token. A chat feels fast when the first is short; a long document or an agent finishes sooner when the second is high. Operators tune for one or the other.",
      shape: '2 numbers', dtype: 'milliseconds, and tokens per second',
      payload: {
        label: 'how they add up (example timings)',
        code: `network + gateway       30 ms
queue                   80 ms
prefill (44 tokens)     40 ms
-----------------------------
time to first token    150 ms

164 tokens x 25 ms     4.1 s
  = 40 tokens per second
total                  4.25 s`,
        note: "Illustrative numbers that show the arithmetic, not a measurement.",
      },
      wire: { name: 'Server-sent events (SSE)', detail: "Both numbers are measured at the client, on this stream: the arrival time of the first event, and the spacing of the rest." },
      cost: "The trade-off of serving: bigger batches raise total throughput and lower the cost per token, but lengthen each step and the queue. Every deployment picks a point on that curve.",
      links: [{ label: 'Deployment modes', href: '#/deploy' }, { label: 'Batching experiment', href: '#/perf/batching' }],
      exhibit: {
        kind: 'timeline', title: 'One request on the clock (example)', caption: 'TTFT, then tokens per second', width: 4.6,
        segments: [{ label: 'net', w: 0.6, tone: 0.25 }, { label: 'queue', w: 1.1, tone: 0.4 }, { label: 'prefill', w: 0.9, tone: 0.6 }, { label: 'decode: one token per step', w: 5, tone: 0.75, ticks: 22 }],
        marks: [{ at: 2.6 / 7.6, label: 'first token' }, { at: 1, label: 'done' }],
      },
    },
  ],
};

// ------------------------------------------------------------------ 9. agent to agent

const a2a = {
  id: 'a2a',
  name: 'Agents delegating to agents',
  short: 'A2A',
  tagline: 'An orchestrator discovers other agents by their Agent Cards and exchanges tasks, messages and artifacts with them.',
  model: "Message shapes follow **Agent2Agent (A2A) 1.0** in its JSON-RPC binding, reduced to the fields that matter here. The agents, ids, prices and contents are an example.",
  graph: {
    nodes: [
      N('user', 'user', 'You', -8.5, 0),
      N('orch', 'agent', 'Orchestrator\nagent', -3.5, 0, { size: 1.15 }),
      N('card', 'file', 'Agent Card', 1.5, -4.6),
      N('travel', 'agent', 'Travel agent', 6.5, -1.6),
      N('ttools', 'tool', 'Its own tools', 11, -1.6),
      N('fin', 'agent', 'Finance agent', 6.5, 3.6),
    ],
    edges: [
      E('goal', 'user', 'orch', 'goal', { bend: 0.1 }),
      E('done', 'orch', 'user', 'answer', { bend: 0.1 }),
      E('disc', 'orch', 'card', 'GET agent-card.json', { bend: 0.08 }),
      E('cardr', 'card', 'orch', 'Agent Card · JSON', { bend: 0.08 }),
      E('send', 'orch', 'travel', 'message · JSON-RPC', { bend: 0.08 }),
      E('upd', 'travel', 'orch', 'SSE: status, artifacts', { bend: 0.08 }),
      E('priv', 'travel', 'ttools', 'private (MCP)', { both: true }),
      E('send2', 'orch', 'fin', 'message · JSON-RPC', { bend: 0.08 }),
      E('upd2', 'fin', 'orch', 'task + artifact', { bend: 0.08 }),
    ],
    zones: [
      { id: 'mine', label: 'Your side', nodes: ['user', 'orch'], color: DEVICE_ZONE },
      { id: 'travel', label: 'travel.example: another company', nodes: ['card', 'travel', 'ttools'] },
      { id: 'fin', label: 'Another team', nodes: ['fin'] },
    ],
  },
  stages: [
    {
      title: 'A goal too big for one agent',
      node: 'orch', from: 'user', edges: ['goal'],
      what: "The user gives the orchestrator a goal that needs skills it does not have: booking travel and checking a budget. Those capabilities belong to **other agents**, built by other teams on other frameworks. A2A is the protocol that lets agents work together without sharing code, memory or tools.",
      shape: '1 string · 66 characters', dtype: 'UTF-8 text', size: '66 bytes',
      payload: { label: 'user message', code: 'Plan a 3-day team offsite in Lisbon for 12 people under EUR 9,000.' },
      wire: { name: 'JSON over HTTPS', detail: "An ordinary chat request to the orchestrator. Nothing agent-to-agent has happened yet." },
      cost: "Nothing yet. Delegation multiplies cost: every agent that gets involved runs its own model loop.",
      exhibit: { kind: 'text', title: 'Goal', caption: 'text', width: 3.8, text: 'Plan a 3-day team offsite in Lisbon for 12 people under EUR 9,000.' },
    },
    {
      title: 'Discovery: reading the Agent Card',
      node: 'card', from: 'orch', edges: ['disc', 'cardr'], ref: 'a2a',
      what: "Every A2A agent publishes an **Agent Card**: a JSON document at a well-known URL that says who it is, where to reach it, how to authenticate and which **skills** it offers. The orchestrator reads cards to decide whom to delegate to, much as it reads tool descriptions.",
      shape: '1 JSON document', dtype: 'application/json', size: 'about 1 kB',
      payload: {
        label: 'GET https://travel.example/.well-known/agent-card.json', lang: 'json',
        code: `{
  "name": "Travel Planner",
  "description": "Plans and books group travel.",
  "version": "2.3.0",
  "supportedInterfaces": [ {
    "url": "https://travel.example/a2a",
    "protocolBinding": "JSONRPC" } ],
  "capabilities": { "streaming": true },
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": [
    "text/plain", "application/json"],
  "skills": [ {
    "id": "plan-trip",
    "name": "Plan a group trip",
    "description": "Flights, hotels and a day-by-day itinerary.",
    "tags": ["travel", "booking"] } ]
}`,
      },
      wire: { name: 'HTTPS GET, JSON', detail: "A plain web request to a fixed path. Cards can also come from a registry or be configured by hand. The card names the security schemes a caller must satisfy." },
      cost: "One cached web request. The cost that matters is trust: a card is a claim made by the other party, so who may delegate to whom is a policy decision, not a protocol feature.",
      exhibit: {
        kind: 'json', title: 'Agent Card', caption: 'JSON', bar: '/.well-known/agent-card.json',
        code: `{ "name": "Travel Planner",
  "supportedInterfaces": [ {
    "url": "https://travel.example/a2a",
    "protocolBinding": "JSONRPC" } ],
  "capabilities": { "streaming": true },
  "skills": [ {
    "id": "plan-trip",
    "name": "Plan a group trip" } ] }`,
      },
    },
    {
      title: 'Delegation: a message that opens a task',
      node: 'travel', from: 'orch', edges: ['send'], ref: 'a2a',
      what: "The orchestrator sends the travel agent a **message**: natural language, plus structured parts where useful. The remote agent answers with a **task**, a unit of work with an id and a state that can live for seconds or for days. The orchestrator does not learn how the work will be done.",
      shape: 'message: role + parts[1]', dtype: 'JSON-RPC 2.0 request', size: 'about 300 bytes',
      payload: {
        label: 'POST https://travel.example/a2a (request, then the first event of the reply)', lang: 'json',
        code: `{ "jsonrpc": "2.0", "id": 1,
  "method": "SendStreamingMessage",
  "params": { "message": {
    "messageId": "m-01",
    "role": "ROLE_USER",
    "parts": [ { "text": "Plan a 3-day offsite in Lisbon for 12 people, 18 to 20 March, at most EUR 6,500 for travel and hotel." } ]
  } } }

{ "jsonrpc": "2.0", "id": 1,
  "result": { "task": {
    "id": "task-7f3a",
    "contextId": "ctx-21",
    "status": {
      "state": "TASK_STATE_WORKING" }
  } } }`,
      },
      wire: { name: 'JSON-RPC 2.0 over HTTPS', detail: "A2A also defines gRPC and plain HTTP+JSON bindings over the same data model. The caller plays the user role: to the remote agent, another agent is just a client." },
      cost: "The request is cheap. What it starts is not: a whole agent loop on someone else's infrastructure, usually billed to you.",
      exhibit: {
        kind: 'json', title: 'Message', caption: 'JSON-RPC 2.0', bar: 'SendStreamingMessage',
        code: `{ "message": {
    "messageId": "m-01",
    "role": "ROLE_USER",
    "parts": [ {
      "text": "Plan a 3-day offsite …"
    } ] } }`,
      },
    },
    {
      title: 'The remote agent works, out of sight',
      node: 'ttools', from: 'travel', edges: ['priv'],
      flows: [{ path: 'priv', count: 3 }, { path: '-priv', count: 3, phase: 0.5 }],
      what: "Inside its own boundary the travel agent runs its own model, prompts and tools. None of that is visible to the caller. **MCP connects an agent to its tools; A2A connects an agent to another agent** whose insides stay private. That opacity is the point: companies can cooperate without exposing their systems.",
      shape: 'not visible to the caller', dtype: 'private',
      payload: {
        label: 'all the orchestrator sees meanwhile: a status event', lang: 'json',
        code: `{ "jsonrpc": "2.0", "id": 1,
  "result": { "statusUpdate": {
    "taskId": "task-7f3a",
    "status": {
      "state": "TASK_STATE_WORKING",
      "message": {
        "role": "ROLE_AGENT",
        "parts": [ { "text": "Comparing 3 hotels near Alfama ..." } ]
      } }
  } } }`,
      },
      wire: { name: 'Private: MCP, databases, other APIs', detail: "Whatever the remote agent uses internally. The caller sees only the task state and the progress messages the agent chooses to share." },
      cost: "Most of the time and money of the whole job is spent here, where the caller cannot see why. Long tasks are followed by streaming, by polling, or by a push notification to a webhook.",
      links: [{ label: 'The tool loop inside one agent', href: '#/flow/agent' }],
      exhibit: { kind: 'text', title: 'Inside the other agent', caption: 'opaque to the caller', width: 3.8, text: 'Its own model, prompts and tools.\nThe caller sees only:\n\ntask-7f3a · TASK_STATE_WORKING' },
    },
    {
      title: 'Updates stream back, and the agent may ask a question',
      node: 'orch', from: 'travel', edges: ['upd'],
      what: "Progress arrives as server-sent events. A task can also pause in the state **input-required**: the remote agent needs something, here a choice between two hotels. The orchestrator answers with another message in the same task and the work resumes. Agents negotiate in turns, like people.",
      shape: '1 status event', dtype: 'SSE carrying JSON-RPC', size: 'about 300 bytes per event',
      payload: {
        label: 'one event of the stream, then the reply',
        code: `data: { "jsonrpc": "2.0", "id": 1,
  "result": { "statusUpdate": {
    "taskId": "task-7f3a",
    "status": {
      "state": "TASK_STATE_INPUT_REQUIRED",
      "message": { "role": "ROLE_AGENT",
        "parts": [ { "text": "Hotel A is EUR 5,900. Hotel B is EUR 6,800 with a meeting room. Which one?" } ] } }
  } } }

{ "jsonrpc": "2.0", "id": 2,
  "method": "SendMessage",
  "params": { "message": {
    "taskId": "task-7f3a",
    "role": "ROLE_USER",
    "parts": [ { "text": "Hotel A." } ]
  } } }`,
      },
      wire: { name: 'Server-sent events (SSE)', detail: "The streaming reply to the original request. Each event wraps one JSON-RPC response: a status update, an artifact update or a message." },
      cost: "Each round trip between agents is a model call on both sides. A question the orchestrator cannot answer has to travel up to the human, and the task waits.",
      exhibit: {
        kind: 'json', title: 'Status update', caption: 'SSE event',
        code: `{ "statusUpdate": {
  "taskId": "task-7f3a",
  "status": {
    "state": "TASK_STATE_INPUT_REQUIRED",
    "message": { "parts": [ {
      "text": "Hotel A or Hotel B?"
    } ] } } } }`,
      },
    },
    {
      title: 'The result arrives as an artifact',
      node: 'travel', from: 'orch', edges: ['upd'],
      what: "When the work is done, the task's output is delivered as one or more **artifacts**: named results made of parts, which can be text, structured data or files. The task then moves to its final state. Messages are the conversation; artifacts are the deliverable.",
      shape: 'artifact: parts[2]', dtype: 'JSON: a text part and a data part', size: 'about 2 kB',
      payload: {
        label: 'the last two events of the stream',
        code: `data: { "jsonrpc": "2.0", "id": 1,
  "result": { "artifactUpdate": {
    "taskId": "task-7f3a",
    "artifact": {
      "artifactId": "a-1",
      "name": "itinerary",
      "parts": [
        { "text": "Day 1: arrive, Alfama walking tour ..." },
        { "data": { "total_eur": 5900,
            "hotel": "Hotel A",
            "nights": 2, "people": 12 } }
      ] }
  } } }

data: { "jsonrpc": "2.0", "id": 1,
  "result": { "statusUpdate": {
    "taskId": "task-7f3a",
    "status": {
      "state": "TASK_STATE_COMPLETED" }
  } } }`,
      },
      wire: { name: 'Server-sent events (SSE)', detail: "Large artifacts can be streamed in chunks, or handed over as a link to a file instead of inline bytes." },
      cost: "The artifact is the only thing the caller pays to read. Structured parts matter: a number in a data part can be checked by code, a number in prose has to be read again by a model.",
      exhibit: {
        kind: 'json', title: 'Artifact', caption: 'TASK_STATE_COMPLETED',
        code: `{ "artifactId": "a-1",
  "name": "itinerary",
  "parts": [
    { "text": "Day 1: arrive, Alfama …" },
    { "data": { "total_eur": 5900,
                "hotel": "Hotel A" } }
  ] }`,
      },
    },
    {
      title: 'A second agent, the same protocol',
      node: 'fin', from: 'orch', edges: ['send2', 'upd2'],
      what: "The orchestrator hands the itinerary's numbers to a finance agent that knows the company's travel policy. A different owner and a different framework, but the same three ideas: card, task, artifact. Independent tasks like this one can run in parallel.",
      shape: 'message: parts[2], text + data', dtype: 'JSON-RPC 2.0', size: 'about 500 bytes',
      payload: {
        label: 'request, then response', lang: 'json',
        code: `{ "jsonrpc": "2.0", "id": 5,
  "method": "SendMessage",
  "params": { "message": {
    "messageId": "m-09",
    "role": "ROLE_USER",
    "parts": [
      { "text": "Check this offsite against the travel policy." },
      { "data": { "total_eur": 5900,
          "people": 12, "nights": 2 } }
    ] } } }

{ "jsonrpc": "2.0", "id": 5,
  "result": { "task": {
    "id": "task-b210",
    "status": {
      "state": "TASK_STATE_COMPLETED" },
    "artifacts": [ {
      "artifactId": "a-1",
      "name": "policy-check",
      "parts": [ { "data": {
        "approved": true,
        "per_person_eur": 491.67,
        "limit_eur": 550 } } ]
    } ] } } }`,
      },
      wire: { name: 'JSON-RPC 2.0 over HTTPS', detail: "A short task can complete inside a single request and response, with no stream. The same call would return a working task if the check took longer." },
      cost: "Parallel delegation shortens the wait but not the bill. The orchestrator pays in context too: every artifact it receives is input on its next model call.",
      exhibit: {
        kind: 'json', title: 'Policy check', caption: 'artifact · data part',
        code: `{ "name": "policy-check",
  "parts": [ { "data": {
    "approved": true,
    "per_person_eur": 491.67,
    "limit_eur": 550 } } ] }`,
      },
    },
    {
      title: 'One answer for the human',
      node: 'orch', from: 'user', edges: ['done'], ref: 'a2a',
      what: "The orchestrator merges the two artifacts into a single reply. From the user's side it was one conversation. Behind it, three agents with different owners exchanged a dozen messages, and none of them saw inside another.",
      shape: '1 text answer', dtype: 'UTF-8 text, streamed as SSE', size: 'about 60 output tokens',
      payload: {
        label: 'what the user reads (example)',
        code: `Your offsite is planned: 18 to 20 March in Lisbon, Hotel A, 12 people, EUR 5,900 in total (EUR 491.67 per person). Finance has approved it against the EUR 550 limit. The day-by-day itinerary is attached.`,
      },
      wire: { name: 'Server-sent events (SSE)', detail: "Back on the ordinary chat transport. How much of the delegation the interface reveals is a design decision." },
      cost: "The sum of three agents' model loops. Multi-agent systems buy specialization and separate ownership at the price of more tokens, more latency and harder debugging.",
      links: [{ label: 'The loop inside each agent', href: '#/flow/agent' }],
      exhibit: { kind: 'text', title: 'Final answer', caption: 'text', width: 4, text: 'Your offsite is planned: 18 to 20 March in Lisbon, Hotel A, 12 people, EUR 5,900 in total. Finance has approved it against the EUR 550 per-person limit.' },
    },
  ],
};

// ------------------------------------------------------------------ 10. federated learning

const fed = {
  id: 'fed',
  name: 'Federated learning',
  short: 'Federated',
  tagline: 'Training on data that never leaves the device: the model travels, the weight updates come back, the server only averages.',
  model: "Sizes are for a **1.4M-parameter next-word LSTM** of the kind trained with federated learning for phone keyboards; the algorithm is federated averaging (FedAvg). The cohort of 300 phones, the example counts, settings and metric values are an example.",
  graph: {
    nodes: [
      N('glob', 'model', 'Global model', -5, -3.6),
      N('coord', 'service', 'Coordinator\nserver', 0, -3.6),
      N('agg', 'lock', 'Secure\naggregation', 5, -3.6),
      N('dataA', 'store', 'Local data\n(stays here)', -9.5, 2.6),
      N('trainA', 'device', 'Phone A:\nlocal training', -5, 2.6),
      N('phoneB', 'device', 'Phone B', 0, 2.6),
      N('phoneC', 'device', 'Phones C to N', 5, 2.6),
    ],
    edges: [
      E('w', 'glob', 'coord', 'current model'),
      E('dA', 'coord', 'trainA', 'model · gRPC, 5.6 MB', { bend: 0.08 }),
      E('dB', 'coord', 'phoneB', '', { bend: 0.08 }),
      E('dC', 'coord', 'phoneC', '', { bend: 0.08 }),
      E('local', 'dataA', 'trainA', 'on-device only'),
      E('uA', 'trainA', 'coord', 'masked update', { bend: 0.08 }),
      E('uB', 'phoneB', 'coord', '', { bend: 0.08 }),
      E('uC', 'phoneC', 'coord', '', { bend: 0.08 }),
      E('m', 'coord', 'agg', '300 masked updates'),
      E('new', 'agg', 'glob', 'new global model', { lift: 2.1 }),
    ],
    zones: [
      { id: 'server', label: 'Server: never sees raw data', nodes: ['glob', 'coord', 'agg'] },
      { id: 'phoneA', label: 'Phone A: the data never leaves', nodes: ['dataA', 'trainA'], color: DEVICE_ZONE, strong: true },
      { id: 'others', label: 'The rest of this round\'s 300 phones', nodes: ['phoneB', 'phoneC'], color: DEVICE_ZONE },
    ],
  },
  stages: [
    {
      title: 'The server sends the model to a few hundred phones',
      node: 'coord', from: 'glob', also: ['trainA', 'phoneB', 'phoneC'], frame: ['glob', 'coord', 'trainA', 'phoneB', 'phoneC'], maxFit: 7.4, edges: ['w', 'dA', 'dB', 'dC'],
      what: "In each **round** the coordinator picks a small sample of the devices that are idle, charging and on Wi-Fi, and sends each one the current global model. The model travels to the data: the opposite of ordinary training, where the data travels to the model.",
      shape: '1.4M weights (the whole model)', dtype: 'float32', size: '5.6 MB per phone',
      payload: {
        label: 'training instruction (Flower-style fields, shown as JSON)', lang: 'json',
        code: `{
  "round": 812,
  "parameters": {
    "tensor_type": "numpy.ndarray",
    "tensors": ["<5.6 MB of float32 bytes>"]
  },
  "config": {
    "local_epochs": 1,
    "batch_size": 50,
    "learning_rate": 0.3 }
}`,
      },
      wire: { name: 'gRPC with protobuf, over TLS', detail: "Weights travel as raw bytes inside protobuf messages. The phone opens the connection (it has no public address) and the server answers with work, or tells it to come back later." },
      cost: "Download bandwidth and patience: only a fraction of devices are eligible at any moment, and some drop out mid-round. Rounds are designed to succeed with whoever finishes.",
      exhibit: { kind: 'vectors', title: 'Global model, round 812', caption: '1.4M x float32 · 5.6 MB', rows: 4, cols: 26, labels: ['embedding', 'LSTM', 'projection', 'output'], more: '…' },
    },
    {
      title: 'The training data is already on the phone',
      node: 'dataA', also: ['trainA'], frame: ['dataA', 'trainA'], edges: ['local'],
      what: "What you typed, with the next word you really chose, sits in local storage. This is the sensitive part, and in federated learning it **never leaves the device**: not raw, not anonymized, not encrypted. Only the phone itself ever reads it.",
      shape: 'about 400 sentences on this phone', dtype: 'token ids, int32', size: 'tens of kilobytes · never uploaded',
      payload: {
        label: 'local examples: never sent anywhere',
        code: `input:  "see you at the"
target: "station"

input:  "running 10 min"
target: "late"

input:  "can you send me the"
target: "address"

... about 400 examples on this phone`,
      },
      wire: { name: 'Local storage: on-device only', detail: "An app-private database. This arrow of the diagram stays inside the phone: there is no network on this edge." },
      cost: "Privacy is bought with statistics. Each phone's data is small and unlike everyone else's, which makes federated training slower and less stable than training on a pooled data set.",
      exhibit: { kind: 'text', title: 'On-device examples', caption: 'never uploaded', width: 3.6, size: 0.19, text: 'see you at the → station\nrunning 10 min → late\ncan you send me the → address\n… about 400 on this phone' },
    },
    {
      title: 'Each phone trains on its own data',
      node: 'trainA', from: 'dataA', edges: ['local'],
      what: "The phone runs a few passes of ordinary gradient descent over its own examples, starting from the global model. After a minute or two it holds a slightly different model, one that has learned a little from this one user.",
      shape: '1.4M weights, locally updated', dtype: 'float32', size: '5.6 MB in memory',
      payload: {
        label: 'on the device',
        code: `w = w_global.copy()
for epoch in range(1):
    for batch in batches(local_data, 50):
        # 400 examples = 8 batches
        w -= 0.3 * grad(loss(w, batch))
# w now differs slightly from w_global`,
      },
      wire: { name: 'On-device: memory only', detail: "Training runs in a background job that the operating system allows only while the phone is idle, charging and on an unmetered network." },
      cost: "The user's battery and a minute or two of a phone processor. That is the whole compute budget of federated learning, which is why the models are small.",
      links: [{ label: 'Optimizer Arena', href: '#/labs/optim' }, { label: 'Deployment modes', href: '#/deploy' }],
      exhibit: { kind: 'vectors', title: 'Before and after local training', caption: '1.4M x float32', rows: 2, cols: 26, labels: ['global', 'local'], shared: 0.92, more: '…' },
    },
    {
      title: 'Only a weight update is uploaded',
      node: 'coord', from: 'trainA', also: ['phoneB', 'phoneC'], frame: ['coord', 'trainA', 'phoneB', 'phoneC'], maxFit: 7.4, edges: ['uA', 'uB', 'uC'],
      what: "The phone sends back the **difference** between its trained weights and the model it received, plus the number of examples it used. The update is clipped to a maximum size so that no single user can dominate, and is often compressed. An update is not the data, but it is derived from it, which is why the next step exists.",
      shape: '1.4M-value update + an example count', dtype: 'float32, often quantized to 8 bits', size: '5.6 MB, or 1.4 MB at 8 bits',
      payload: {
        label: 'the upload (Flower-style fields, shown as JSON)', lang: 'json',
        code: `{
  "round": 812,
  "parameters": {
    "tensor_type": "numpy.ndarray",
    "tensors": ["<masked update bytes>"]
  },
  "num_examples": 400,
  "metrics": { "train_loss": 3.21 }
}`,
      },
      wire: { name: 'gRPC with protobuf, over TLS', detail: "Upload is the scarce direction on mobile networks, so updates are the main target of compression: quantization, sparsification, fewer rounds." },
      cost: "Megabytes of upload per phone per round, times hundreds of phones, times thousands of rounds. Communication, not compute, is the bottleneck of federated learning.",
      links: [{ label: 'Quantization', href: '#/perf/quant' }],
      exhibit: { kind: 'vectors', title: 'Updates: local minus global', caption: '5.6 MB each', rows: 3, cols: 26, labels: ['Phone A', 'Phone B', 'Phone C'], more: '…' },
    },
    {
      title: 'Secure aggregation: the server sees only the sum',
      node: 'agg', from: 'coord', edges: ['m'],
      what: "Before uploading, every pair of phones agrees on a random mask; one adds it, the other subtracts it. Each masked update looks like noise, but when the server adds them all up the masks cancel exactly. The server learns the sum and nothing about any single phone.",
      shape: '300 masked updates in, 1 sum out', dtype: 'integers modulo a fixed range', size: '300 x 5.6 MB in, 5.6 MB out',
      payload: {
        label: 'why the masks cancel',
        code: `phone A sends   u_A + m_AB + m_AC
phone B sends   u_B - m_AB + m_BC
phone C sends   u_C - m_AC - m_BC
---------------------------------
sum             u_A + u_B + u_C`,
      },
      wire: { name: 'Masked integer vectors', detail: "The masks come from keys the phones exchange through the server. The protocol tolerates phones that drop out mid-round: the survivors help remove the masks of the missing." },
      cost: "Extra key-exchange rounds and bookkeeping that grows with the cohort. Often combined with clipping and added noise (differential privacy), so that even the sum reveals little about one person.",
      exhibit: { kind: 'vectors', title: 'Masked updates cancel in the sum', caption: '300 updates → 1', rows: 3, cols: 26, labels: ['A, masked', 'B, masked', 'C, masked'], shared: 0.15, result: { label: 'sum' } },
    },
    {
      title: 'Federated averaging makes the new global model',
      node: 'glob', from: 'agg', edges: ['new'],
      what: "The server divides the sum by the total number of examples, which gives an average in which phones with more data count more, and adds it to the global model. That is **FedAvg**. No gradient was computed on the server and no example was ever seen there.",
      shape: '1.4M weights', dtype: 'float32', size: '5.6 MB',
      payload: {
        label: 'the update rule',
        code: `n = sum(n_k)   # examples in this round
w_global += sum(n_k * update_k) / n

# 300 phones x about 400 examples:
# n = 120,000`,
      },
      wire: { name: 'In-process on the server', detail: "A weighted average of arrays. The new model is written to storage as the starting point of the next round." },
      cost: "Trivial arithmetic. The price is in rounds: thousands of them over days, each waiting for phones to become available.",
      exhibit: { kind: 'scalar', title: 'FedAvg', caption: '1.4M x float32', value: 'round 813', label: 'new global model', sub: 'old model + weighted average of 300 updates' },
    },
    {
      title: 'Repeat, evaluate, ship',
      node: 'coord', from: 'glob', edges: ['w'],
      what: "The new model goes out to a fresh sample of phones and the cycle repeats. Quality is measured in the same federated way: phones evaluate the model on their own data and report only a few numbers. When it is good enough, the model ships to everyone in an app update.",
      shape: '1 number per metric per phone', dtype: 'float32', size: 'a few bytes per phone',
      payload: {
        label: 'evaluation report from one phone (example values)', lang: 'json',
        code: `{
  "round": 813,
  "loss": 3.18,
  "num_examples": 120,
  "metrics": { "top1_accuracy": 0.24 }
}`,
      },
      wire: { name: 'gRPC with protobuf, over TLS', detail: "The same channel as training, carrying an evaluate instruction instead of a fit instruction, and metrics instead of weights." },
      cost: "Wall-clock time. A run is paced by device availability, which follows the daily rhythm of people charging their phones at night.",
      links: [{ label: 'Run models on the device', href: '#/deploy' }],
      exhibit: {
        kind: 'json', title: 'Evaluation report', caption: 'metrics only',
        code: `{ "round": 813,
  "loss": 3.18,
  "num_examples": 120,
  "metrics": {
    "top1_accuracy": 0.24 } }`,
      },
    },
  ],
};

/** All pipelines, in the order the chapter presents them. */
export const PIPELINES = [llm, train, rag, agent, diffusion, vision, speech, serve, a2a, fed];

// Twin edges (a request and its response between the same two nodes) curve apart, but from the
// usual camera their labels would still land on top of each other. The label of the twin that
// curves away from the camera (the one traveling toward +x) is raised.
for (const p of PIPELINES) {
  const pos = Object.fromEntries(p.graph.nodes.map((n) => [n.id, n.pos]));
  for (const e of p.graph.edges) {
    const twin = p.graph.edges.find((o) => o.from === e.to && o.to === e.from);
    if (!twin || e.labelLift != null || twin.labelLift != null) continue;
    (pos[e.to][0] > pos[e.from][0] ? e : twin).labelLift = 0.62;
  }
}
