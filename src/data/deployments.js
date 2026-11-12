// Ten ways to put a model in front of users: hand-written content for the Deploy chapter.
//
// * Ratings are EDITORIAL judgments on a 1-5 scale for a typical setup, not measurements.
//   5 is always the favorable end (fastest, cheapest, most private, least effort, largest model).
// * `graph` is the request path as a node-link diagram for core/flowgraph.js. Positions are [x, z]
//   on a stage about 15 units wide; the viewer looks from +z, so a larger z is closer.
// * `flows` are repeating packet streams (kind = what travels, see the legend of each mode);
//   `sim` asks the chapter to run a small live simulation instead (queue, autoscaling, routing).
// * Dated facts (model names, project status) are not repeated here: `facts` and `notes` point
//   into data/landscape.js so both chapters stay consistent.

export const RATINGS = [
  { id: 'latency', label: 'Latency', short: 'Lat', hint: '5 = fastest response' },
  { id: 'costLow', label: 'Cost at low volume', short: '$ lo', hint: '5 = cheapest when traffic is small or spiky' },
  { id: 'costHigh', label: 'Cost at high volume', short: '$ hi', hint: '5 = cheapest per token under heavy, steady load' },
  { id: 'privacy', label: 'Privacy and control', short: 'Priv', hint: '5 = data and weights stay in your hands' },
  { id: 'ops', label: 'Operational effort', short: 'Ops', hint: '5 = least work to run' },
  { id: 'size', label: 'Largest model', short: 'Size', hint: '5 = runs the largest models that exist' },
];

/** One request followed by its streamed answer along the reversed path, inside a repeating period. */
const ask = (period, at, path, o = {}) => [
  { path, kind: 'request', count: 1, size: o.size ?? 0.13, cycle: [period, at, at + (o.go ?? 0.085)] },
  { path: path.map((e) => '-' + e).reverse(), kind: 'response', count: o.tokens ?? 6, size: 0.065, gap: 0.4, cycle: [period, at + (o.go ?? 0.085) + 0.01, at + (o.go ?? 0.085) + 0.01 + (o.back ?? 0.13)] },
];
const each = (n, fn) => Array.from({ length: n }, (_, i) => fn(i + 1)).flat();

export const MODES = [
  // ------------------------------------------------------------------ 1
  {
    id: 'cloud-api',
    name: 'Hosted model API',
    short: 'Cloud API',
    tagline: 'Rent intelligence by the token',
    what: 'Your app sends each prompt over HTTPS to a model that a provider runs on its own accelerator fleet. You never see the weights or the hardware: you get an API key, rate limits and a bill per token.',
    watch: 'Every request leaves your side and crosses to the provider. Other customers share the same fleet, which is why rate limits exist.',
    path: [
      '**User** types a message.',
      '**Your app** adds the system prompt and tools, then calls the API with its key.',
      '**Provider API** checks the key, applies rate limits, meters tokens and picks a replica.',
      '**Model replica** on the shared fleet generates; tokens stream back the same way.',
    ],
    pros: ['Nothing to operate: no GPUs, no drivers, no on-call', 'The only way to reach the closed frontier models', 'No cost while idle, and instant scale inside your rate limit', 'New models arrive without any work on your side'],
    cons: ['Prompts and outputs leave your network: you rely on a contract, not on physics', 'The bill grows in step with usage and never amortizes', 'Rate limits, model retirements and outages are outside your control', 'A network round trip on every call'],
    choose: ['You are starting out, or traffic is small or unpredictable', 'You need the strongest model available', 'Your data may be processed by a third party under contract'],
    ratings: { latency: 3, costLow: 5, costHigh: 2, privacy: 2, ops: 5, size: 5 },
    stack: [
      ['Model APIs', ['OpenAI API', 'Anthropic API', 'Gemini API']],
      ['Cloud platforms', ['Amazon Bedrock', 'Google Vertex AI']],
      ['Client-side gateway', ['LiteLLM', 'OpenRouter']],
    ],
    facts: ['closed'],
    legend: [['request', 'Prompt'], ['response', 'Streamed tokens']],
    graph: {
      nodes: [
        { id: 'user', type: 'user', label: 'User', pos: [-5.2, 3.6] },
        { id: 'app', type: 'app', label: 'Your app', pos: [-5.2, -0.2], tip: 'Builds the prompt, holds the API key and calls the provider over HTTPS.' },
        { id: 'others', type: 'globe', label: 'Other customers', pos: [-5.2, -4.4], tip: 'Many other apps share the same fleet.' },
        { id: 'api', type: 'gateway', label: 'Provider API', pos: [0, -0.2], tip: 'Checks the key, applies rate limits, meters tokens and routes to a free replica.' },
        { id: 'g1', type: 'gpu', label: 'Model replica', pos: [5, -3.8], tip: 'A copy of the model on the provider’s accelerators. You never see the weights.' },
        { id: 'g2', type: 'gpu', label: 'Model replica', pos: [5, -0.2], tip: 'A copy of the model on the provider’s accelerators. You never see the weights.' },
        { id: 'g3', type: 'gpu', label: 'Model replica', pos: [5, 3.4], tip: 'A copy of the model on the provider’s accelerators. You never see the weights.' },
      ],
      edges: [
        { id: 'u-a', from: 'user', to: 'app' },
        { id: 'a-p', from: 'app', to: 'api', label: 'HTTPS + API key' },
        { id: 'o-p', from: 'others', to: 'api' },
        { id: 'p-g1', from: 'api', to: 'g1' },
        { id: 'p-g2', from: 'api', to: 'g2' },
        { id: 'p-g3', from: 'api', to: 'g3' },
      ],
      zones: [
        { id: 'you', label: 'Your side', nodes: ['user', 'app'] },
        { id: 'prov', label: 'Provider data center', nodes: ['api', 'g1', 'g2', 'g3'] },
      ],
    },
    flows: [
      { path: ['u-a', 'a-p', 'p-g2'], kind: 'request', count: 1, size: 0.13, cycle: [8, 0.02, 0.3] },
      { path: ['-p-g2', '-a-p', '-u-a'], kind: 'response', count: 7, size: 0.065, gap: 0.42, cycle: [8, 0.36, 0.86] },
      { path: ['o-p', 'p-g1'], kind: 'request', count: 1, size: 0.08, cycle: [4, 0, 0.42] },
      { path: ['-p-g1', '-o-p'], kind: 'response', count: 4, size: 0.05, gap: 0.4, cycle: [4, 0.48, 0.98] },
      { path: ['o-p', 'p-g3'], kind: 'request', count: 1, size: 0.08, cycle: [5.2, 0.1, 0.5], phase: 1.3 },
      { path: ['-p-g3', '-o-p'], kind: 'response', count: 4, size: 0.05, gap: 0.4, cycle: [5.2, 0.55, 0.99], phase: 1.3 },
    ],
    status: { at: [0.6, 4.3], y: 0.5, steps: [[0, 'Prompts leave your side']] },
  },

  // ------------------------------------------------------------------ 2
  {
    id: 'self-hosted',
    name: 'Self-hosted GPU server',
    short: 'Self-hosted',
    tagline: 'One box you run, one model you chose',
    what: 'You run an open-weight model behind an inference server on one GPU machine that you rent or own. The server speaks an OpenAI-compatible HTTP API, and continuous batching lets the single GPU work on many requests at once.',
    watch: 'Capacity is fixed. When a burst arrives, requests pile up in the queue above the server until the GPU has room. When traffic is low the GPU idles and costs the same.',
    path: [
      '**Your app** calls your own endpoint instead of a provider.',
      '**Inference server** queues the request and batches it with the others in flight.',
      '**One GPU** holds the weights in memory and generates for the whole batch.',
      'Tokens stream back. Nothing leaves your machine.',
    ],
    pros: ['Prompts, outputs and weights stay on a machine you control', 'Flat cost: every extra token is free until the GPU is full', 'Any open-weight model, any quantization, your own fine-tunes', 'Low, steady latency when the app sits next to it'],
    cons: ['No elasticity: a burst queues, a crash stops everything', 'You pay for idle hours', 'Drivers, updates, monitoring and security are your job', 'Model size is capped by the memory of one machine'],
    choose: ['Steady, predictable load that keeps a GPU busy', 'Data has to stay in your own environment', 'You need a custom or fine-tuned open-weight model', 'Internal tools and prototypes where a short outage is acceptable'],
    ratings: { latency: 4, costLow: 2, costHigh: 3, privacy: 4, ops: 3, size: 3 },
    stack: [
      ['Inference engine', ['vLLM', 'SGLang', 'TensorRT-LLM', 'llama.cpp']],
      ['Simplest start', ['Ollama']],
      ['Packaging', ['Docker', 'NVIDIA Container Toolkit']],
    ],
    facts: ['ollama'],
    notes: 'TGI was archived|No engine wins',
    legend: [['request', 'Prompt (stacked boxes: requests waiting)'], ['response', 'Streamed tokens']],
    graph: {
      nodes: [
        { id: 'user', type: 'user', label: 'Users', pos: [-5.2, 3.4] },
        { id: 'app', type: 'app', label: 'Your app', pos: [-5.2, -0.4] },
        { id: 'srv', type: 'service', label: 'Inference server', pos: [0, -0.4], tip: 'vLLM, SGLang or llama.cpp: an OpenAI-compatible endpoint with a request queue and continuous batching.' },
        { id: 'gpu', type: 'gpu', label: 'One GPU', pos: [5, -0.4], size: 1.25, tip: 'Fixed capacity: the weights sit in its memory and every request shares it.' },
        { id: 'disk', type: 'file', label: 'Model weights', pos: [5, -4.2], tip: 'An open-weight model, downloaded once and loaded at start-up.' },
      ],
      edges: [
        { id: 'u-a', from: 'user', to: 'app' },
        { id: 'a-s', from: 'app', to: 'srv', label: 'OpenAI-style API' },
        { id: 's-g', from: 'srv', to: 'gpu' },
        { id: 'd-g', from: 'disk', to: 'gpu', dashed: true },
      ],
      zones: [{ id: 'box', label: 'Your GPU server', nodes: ['srv', 'gpu', 'disk'] }],
    },
    sim: { type: 'pool', period: 16, rate: [[0, 0.6], [0.3, 2.4], [0.55, 0.25]], service: 1, min: 1, max: 1, queueAt: 'srv', inPath: ['u-a', 'a-s'], outPath: ['-a-s', '-u-a'], replicas: [{ id: 'gpu', edge: 's-g' }] },
    status: { at: [0, -0.4], y: 3.4 },
  },

  // ------------------------------------------------------------------ 3
  {
    id: 'k8s',
    name: 'Kubernetes GPU cluster',
    short: 'Kubernetes',
    tagline: 'A gateway, a pool of replicas, an autoscaler',
    what: 'Many replicas of an inference server run as pods on a pool of GPU nodes. A model gateway authenticates, applies quotas and routes each request to a good replica, and an autoscaler adds or removes replicas as load changes.',
    watch: 'When load rises the queue at the gateway grows, the autoscaler reacts and new replicas start; each needs time to pull and load its weights. When load falls, the extra replicas are removed.',
    path: [
      '**Users** reach one stable address.',
      '**Model gateway** authenticates, enforces quotas and picks a replica, often by queue depth or cached prefix.',
      '**Replica**: an inference server pod on a GPU node generates and streams back.',
      '**Autoscaler** watches the queue and changes the replica count; new pods pull weights from the **model registry**.',
    ],
    pros: ['Scales out and survives failures: the gateway routes around a lost node', 'High utilization, so the lowest cost per token at scale', 'One platform for many models, teams and quotas', 'Rolling updates, canaries and observability are built in'],
    cons: ['The most moving parts: cluster, GPU operator, gateway, autoscaler, registry', 'Needs a platform team and on-call', 'Scaling up is slow because every new replica loads many gigabytes of weights', 'Minimum replicas cost money around the clock'],
    choose: ['Sustained high volume, several models or many teams', 'You need high availability and controlled rollouts', 'You already run Kubernetes and have people to operate it'],
    ratings: { latency: 4, costLow: 1, costHigh: 5, privacy: 4, ops: 1, size: 4 },
    stack: [
      ['Orchestration', ['Kubernetes', 'KServe', 'Ray Serve', 'NVIDIA Dynamo']],
      ['Engines in the pods', ['vLLM', 'SGLang', 'TensorRT-LLM']],
      ['Routing and scaling', ['Gateway API Inference Extension', 'KEDA']],
      ['GPU plumbing', ['NVIDIA GPU Operator', 'Prometheus']],
    ],
    facts: ['open', 'vram'],
    notes: 'NVIDIA Dynamo',
    legend: [['request', 'Prompt (stacked boxes: requests waiting)'], ['response', 'Streamed tokens'], ['control', 'Metrics and scale-up signals'], ['weights', 'Weights pulled by a new replica']],
    graph: {
      nodes: [
        { id: 'users', type: 'user', label: 'Users', pos: [-6, 0.15] },
        { id: 'gw', type: 'gateway', label: 'Model gateway', pos: [-1.9, 0.15], tip: 'One address for every model: authentication, quotas, routing and load balancing.' },
        { id: 'scaler', type: 'scaler', label: 'Autoscaler', pos: [-1.9, -4.2], tip: 'Reads queue depth and GPU metrics, then adds or removes replicas.' },
        { id: 'r1', type: 'gpu', label: 'Replica 1', pos: [2.6, -4.2], tip: 'An inference server pod with its own GPU. Always on.' },
        { id: 'r2', type: 'gpu', label: 'Replica 2', pos: [2.6, -1.3], tip: 'An inference server pod with its own GPU. Always on.' },
        { id: 'r3', type: 'gpu', label: 'Replica 3', pos: [2.6, 1.6], tip: 'Started by the autoscaler when the queue grows.' },
        { id: 'r4', type: 'gpu', label: 'Replica 4', pos: [2.6, 4.5], tip: 'Started by the autoscaler when the queue grows.' },
        { id: 'reg', type: 'database', label: 'Model registry', pos: [6.8, 0.15], tip: 'Object storage or a registry that holds the weights every new pod has to load.' },
      ],
      edges: [
        { id: 'u-g', from: 'users', to: 'gw' },
        { id: 'g-r1', from: 'gw', to: 'r1' },
        { id: 'g-r2', from: 'gw', to: 'r2' },
        { id: 'g-r3', from: 'gw', to: 'r3' },
        { id: 'g-r4', from: 'gw', to: 'r4' },
        { id: 'g-s', from: 'gw', to: 'scaler', dashed: true },
        { id: 'c-r3', from: 'scaler', to: 'r3', dashed: true, lift: 1.3 },
        { id: 'c-r4', from: 'scaler', to: 'r4', dashed: true, lift: 1.5 },
        { id: 'm-r1', from: 'reg', to: 'r1', dashed: true },
        { id: 'm-r2', from: 'reg', to: 'r2', dashed: true },
        { id: 'm-r3', from: 'reg', to: 'r3', dashed: true },
        { id: 'm-r4', from: 'reg', to: 'r4', dashed: true },
      ],
      zones: [
        { id: 'cluster', label: 'Kubernetes cluster', rect: [-3.7, -5.9, 8.7, 7] },
        { id: 'pool', label: 'GPU node pool', nodes: ['r1', 'r2', 'r3', 'r4'], pad: 0.5 },
      ],
    },
    flows: [{ path: 'g-s', kind: 'control', count: 2, speed: 1.4, size: 0.05 }],
    sim: {
      type: 'pool', period: 22, rate: [[0, 1.5], [0.2, 3.7], [0.6, 1.5]], service: 1, min: 2, max: 4, boot: 2.8, idleOff: 3, upAt: 3, queueAt: 'gw', inPath: ['u-g'], outPath: ['-u-g'],
      replicas: [
        { id: 'r1', edge: 'g-r1', wts: 'm-r1' },
        { id: 'r2', edge: 'g-r2', wts: 'm-r2' },
        { id: 'r3', edge: 'g-r3', ctl: 'c-r3', wts: 'm-r3' },
        { id: 'r4', edge: 'g-r4', ctl: 'c-r4', wts: 'm-r4' },
      ],
    },
    status: { at: [-1.4, 4.3], y: 0.6 },
  },

  // ------------------------------------------------------------------ 4
  {
    id: 'serverless',
    name: 'Serverless GPU',
    short: 'Serverless GPU',
    tagline: 'Scale to zero, pay per second, wait for cold starts',
    what: 'You hand a platform a container or a model. It starts GPU replicas only when requests arrive, bills per second of use and removes them when traffic stops. You manage no machines.',
    watch: 'With nothing running, the first request waits at the endpoint while a container boots and loads the weights: the cold start. Later requests reuse the warm replica. After an idle period it scales back to zero.',
    path: [
      '**User** calls an HTTPS endpoint.',
      '**Endpoint**: the platform router holds the request if no replica is warm.',
      '**Cold start**: a GPU is scheduled, the image is pulled and the weights are loaded into memory.',
      '**GPU container** generates and streams back, then stays warm for a while.',
    ],
    pros: ['No idle cost: no traffic, almost no bill', 'Very little to operate: no nodes, drivers or autoscaler', 'Absorbs spikes by starting more replicas', 'Your own weights and code, unlike a hosted model API'],
    cons: ['Cold starts: seconds to minutes before the first token, growing with model size', 'A second of GPU costs more than on a reserved machine, so steady load is expensive', 'GPU capacity is not guaranteed at busy times', 'Your data runs on a third party’s platform'],
    choose: ['Spiky or rare traffic: internal tools, batch jobs, demos', 'Custom or fine-tuned models that no API offers', 'Users can tolerate an occasional slow first response, or you pay to keep one replica warm'],
    ratings: { latency: 2, costLow: 4, costHigh: 2, privacy: 3, ops: 4, size: 3 },
    stack: [
      ['Platforms', ['Modal', 'RunPod Serverless', 'Replicate', 'Baseten', 'Cloud Run (GPU)']],
      ['On your own cluster', ['Knative', 'KServe']],
      ['Inside the container', ['vLLM', 'SGLang']],
    ],
    facts: ['vram'],
    legend: [['request', 'Prompt (stacked boxes: requests waiting)'], ['response', 'Streamed tokens'], ['weights', 'Weights loaded during a cold start']],
    graph: {
      nodes: [
        { id: 'user', type: 'user', label: 'User', pos: [-5.6, 0.6] },
        { id: 'ep', type: 'gateway', label: 'Endpoint', pos: [-1.4, 0.6], tip: 'The platform’s router. It holds a request while a replica starts.' },
        { id: 'c1', type: 'gpu', label: 'GPU container', pos: [3.2, -1.6], tip: 'Exists only while there is work, plus a short keep-warm period.' },
        { id: 'c2', type: 'gpu', label: 'GPU container', pos: [3.2, 2.8], tip: 'A second replica, started when requests queue up.' },
        { id: 'snap', type: 'store', label: 'Image + weights', pos: [7.2, 0.6], tip: 'Container image and model weights, loaded into GPU memory on every cold start.' },
      ],
      edges: [
        { id: 'u-e', from: 'user', to: 'ep' },
        { id: 'e-c1', from: 'ep', to: 'c1' },
        { id: 'e-c2', from: 'ep', to: 'c2' },
        { id: 's-c1', from: 'snap', to: 'c1', dashed: true },
        { id: 's-c2', from: 'snap', to: 'c2', dashed: true },
      ],
      zones: [{ id: 'plat', label: 'Serverless GPU platform', nodes: ['ep', 'c1', 'c2', 'snap'] }],
    },
    sim: {
      type: 'pool', period: 24, schedule: [1.2, 7, 7.9, 8.5, 9.1, 10, 10.9, 11.8, 12.7, 13.6, 14.5], service: 1.1, min: 0, max: 2, boot: 3.4, idleOff: 3.6, upAt: 2, queueAt: 'ep', inPath: ['u-e'], outPath: ['-u-e'],
      replicas: [
        { id: 'c1', edge: 'e-c1', wts: 's-c1' },
        { id: 'c2', edge: 'e-c2', wts: 's-c2' },
      ],
    },
    status: { at: [-1.4, 0.6], y: 3.4 },
  },

  // ------------------------------------------------------------------ 5
  {
    id: 'edge',
    name: 'On-device (edge)',
    short: 'On-device',
    tagline: 'The model lives in the phone or laptop',
    what: 'A small, quantized model ships inside the app or is downloaded once, then runs on the device’s own NPU, GPU or CPU. No server is involved when it answers.',
    watch: 'After the one-time download the model hub goes dark and nothing changes: every request stays inside the device, so it works offline.',
    path: [
      '**User** talks to the app.',
      '**App** hands the prompt to a runtime inside the same process.',
      '**On-device runtime** tokenizes, keeps the KV cache and samples.',
      '**NPU / GPU** runs the quantized weights held in device memory.',
    ],
    pros: ['Prompts never leave the device, and it works with no connection', 'No network round trip before the first token', 'No serving bill: the user’s hardware does the work', 'Serves any number of users at no extra cost'],
    cons: ['Only small models fit: memory, battery and heat set the limit', 'Answers are weaker than those of large cloud models', 'A zoo of chips and OS versions to test and tune for', 'A model update means shipping gigabytes to every device'],
    choose: ['Private or offline features: keyboards, dictation, translation, on-device search', 'Responsiveness matters more than peak quality', 'Huge user counts where a cloud call per request would not pay'],
    ratings: { latency: 4, costLow: 5, costHigh: 5, privacy: 5, ops: 3, size: 2 },
    stack: [
      ['Mobile runtimes', ['ExecuTorch', 'LiteRT', 'ONNX Runtime', 'Core ML']],
      ['Laptops and desktops', ['llama.cpp', 'Ollama', 'MLX']],
    ],
    facts: ['ollama'],
    legend: [['request', 'Prompt'], ['response', 'Tokens'], ['weights', 'Model file (downloaded once)']],
    graph: {
      nodes: [
        { id: 'user', type: 'user', label: 'User', pos: [-5.6, 2.6] },
        { id: 'app', type: 'app', label: 'App', pos: [-0.8, 2.6] },
        { id: 'rt', type: 'service', label: 'On-device runtime', pos: [4.2, 2.6], tip: 'ExecuTorch, LiteRT, llama.cpp and similar: tokenizer, KV cache and sampling inside the app.' },
        { id: 'npu', type: 'gpu', label: 'NPU / GPU', pos: [4.2, -1.2], tip: 'The device’s own accelerator. Memory, battery and heat limit the model size.' },
        { id: 'wts', type: 'file', label: 'Quantized weights', pos: [-0.8, -1.2], tip: 'A small model stored on the device.' },
        { id: 'hub', type: 'cloud', label: 'Model hub', pos: [-0.8, -5.6], tip: 'Needed once, to download the model, unless it ships inside the app.', wave: { period: 16, from: 0, to: 0.2, low: 0.12 } },
      ],
      edges: [
        { id: 'u-a', from: 'user', to: 'app' },
        { id: 'a-r', from: 'app', to: 'rt' },
        { id: 'r-n', from: 'rt', to: 'npu' },
        { id: 'w-n', from: 'wts', to: 'npu', dashed: true },
        { id: 'h-w', from: 'hub', to: 'wts', dashed: true },
      ],
      zones: [{ id: 'dev', label: 'Phone or laptop', rect: [-2.7, -2.8, 6.1, 4.6], strong: true }],
    },
    flows: [
      { path: 'h-w', kind: 'weights', count: 8, size: 0.1, gap: 0.42, cycle: [16, 0.01, 0.17] },
      { path: 'w-n', kind: 'weights', count: 4, size: 0.08, gap: 0.4, cycle: [16, 0.17, 0.24] },
      ...each(3, (k) => ask(16, 0.27 + (k - 1) * 0.245, ['u-a', 'a-r', 'r-n'])),
    ],
    status: { at: [1.7, -2.6], period: 16, steps: [[0, 'One-time model download'], [0.25, 'Offline: requests stay on the device']] },
  },

  // ------------------------------------------------------------------ 6
  {
    id: 'browser',
    name: 'In the browser',
    short: 'In-browser',
    tagline: 'WebGPU or WASM inside a tab',
    what: 'The page downloads the model weights like any other asset and runs them inside the tab with WebGPU, or with WebAssembly on the CPU. The server only hosts static files and never sees a prompt.',
    watch: 'The first visit is one large download into the browser cache. After that, requests loop inside the tab between the page and the visitor’s GPU; the server sits idle.',
    path: [
      '**Static host** serves the page, the runtime and, once, the weights.',
      '**Browser cache** keeps the weights for the next visit.',
      '**Web page** passes the prompt to a model runtime, usually in a worker.',
      '**WebGPU** runs the model on the visitor’s own GPU.',
    ],
    pros: ['Zero install: a URL is the whole deployment', 'No inference servers and no cost per request', 'Prompts stay in the tab', 'Can work offline once the weights are cached'],
    cons: ['The first visit downloads the whole model', 'Small models only: tab memory and the visitor’s GPU set the limit', 'Speed varies widely between devices, and the WASM fallback is slow', 'The weights are public: anyone can copy them'],
    choose: ['Demos, privacy-preserving tools and features that must cost nothing to serve', 'Small jobs: embeddings, classification, speech-to-text, small chat models', 'You cannot ask users to install anything'],
    ratings: { latency: 3, costLow: 5, costHigh: 5, privacy: 5, ops: 4, size: 1 },
    stack: [
      ['Libraries', ['transformers.js', 'WebLLM', 'ONNX Runtime Web']],
      ['Browser APIs', ['WebGPU', 'WebAssembly', 'Cache Storage / OPFS']],
    ],
    notes: 'WebGPU now ships|transformers\\.js v4',
    legend: [['request', 'Prompt'], ['response', 'Tokens'], ['weights', 'Static files and weights (downloaded once)']],
    graph: {
      nodes: [
        { id: 'cdn', type: 'globe', label: 'Static host / CDN', pos: [-5.4, -4.8], tip: 'Serves files only. It never sees a prompt.', wave: { period: 16, from: 0, to: 0.26, low: 0.12 } },
        { id: 'user', type: 'user', label: 'User', pos: [-5.6, 2.8] },
        { id: 'page', type: 'browser', label: 'Web page', pos: [-0.6, 2.8], tip: 'JavaScript plus a library such as transformers.js or WebLLM.' },
        { id: 'rt', type: 'service', label: 'Model runtime', pos: [4.4, 2.8], tip: 'Runs in a web worker so the page stays responsive.' },
        { id: 'gpu', type: 'gpu', label: 'WebGPU', pos: [4.4, -1], tip: 'The visitor’s own GPU, reached through the browser. WebAssembly on the CPU is the fallback.' },
        { id: 'cache', type: 'store', label: 'Browser cache', pos: [-0.6, -1], tip: 'Cache Storage or the origin private file system keeps the weights between visits.' },
      ],
      edges: [
        { id: 'c-p', from: 'cdn', to: 'page', bend: -0.12 },
        { id: 'c-k', from: 'cdn', to: 'cache' },
        { id: 'k-g', from: 'cache', to: 'gpu', dashed: true },
        { id: 'u-p', from: 'user', to: 'page' },
        { id: 'p-r', from: 'page', to: 'rt' },
        { id: 'r-g', from: 'rt', to: 'gpu' },
      ],
      zones: [{ id: 'tab', label: 'Browser tab (sandbox)', rect: [-2.5, -2.6, 6.3, 4.8], strong: true }],
    },
    flows: [
      { path: 'c-p', kind: 'weights', count: 3, size: 0.07, gap: 0.4, cycle: [16, 0, 0.07] },
      { path: 'c-k', kind: 'weights', count: 10, size: 0.1, gap: 0.4, cycle: [16, 0.05, 0.22] },
      { path: 'k-g', kind: 'weights', count: 4, size: 0.08, gap: 0.4, cycle: [16, 0.22, 0.28] },
      ...each(3, (k) => ask(16, 0.31 + (k - 1) * 0.23, ['u-p', 'p-r', 'r-g'], { go: 0.08, back: 0.125 })),
    ],
    status: { at: [1.9, -2.4], period: 16, steps: [[0, 'First visit: downloading the weights'], [0.29, 'Server idle: inference runs in the tab']] },
  },

  // ------------------------------------------------------------------ 7
  {
    id: 'hybrid',
    name: 'Hybrid cascade / router',
    short: 'Hybrid router',
    tagline: 'Small model first, big model when needed',
    what: 'A router sits in front of two tiers. A small, cheap model on the device or on your own server answers what it can. Requests that a classifier marks as hard, or that the small model is not confident about, are escalated to a large cloud model.',
    watch: 'Most packets take the short path to the local model. A few, the hard ones drawn larger, cross the boundary to the cloud model and take longer to come back.',
    path: [
      '**User** sends a request.',
      '**Router** estimates the difficulty, or checks the small model’s confidence in its draft.',
      'Easy: the **small local model** answers at once.',
      'Hard: the request is escalated to the **large cloud model**.',
    ],
    pros: ['Most traffic gets the speed and cost of a small model', 'Hard requests still get frontier quality', 'Sensitive requests can be pinned to the local tier', 'Keeps working, with weaker answers, when the cloud is unreachable'],
    cons: ['Two stacks to run, plus a router to tune and evaluate', 'A wrong routing decision means a worse answer or a wasted cloud call', 'In a cascade, escalated requests pay for both tiers', 'The two tiers behave differently, which complicates testing'],
    choose: ['Difficulty varies widely and most requests are easy', 'You want to cut cloud spend without capping quality', 'On-device assistants that need a stronger fallback'],
    ratings: { latency: 4, costLow: 4, costHigh: 4, privacy: 3, ops: 2, size: 5 },
    stack: [
      ['Routing', ['RouteLLM', 'LiteLLM']],
      ['Local tier', ['llama.cpp', 'Ollama', 'MLX', 'vLLM']],
      ['Cloud tier', ['Any hosted model API']],
    ],
    facts: ['closed'],
    legend: [['request', 'Prompt (large = hard request)'], ['response', 'Streamed tokens']],
    graph: {
      nodes: [
        { id: 'user', type: 'user', label: 'User', pos: [-5.6, 1.6] },
        { id: 'router', type: 'gateway', label: 'Router', pos: [-1.2, 1.6], tip: 'A small classifier, or a confidence check on the local model’s draft, decides where each request goes.' },
        { id: 'local', type: 'model', label: 'Small local model', pos: [3.8, 3.4], tip: 'Fast and cheap. Handles the easy majority.' },
        { id: 'big', type: 'gpu', label: 'Large cloud model', pos: [3.8, -3.2], size: 1.35, tip: 'Slower and billed per token. Used only for the hard minority.' },
      ],
      edges: [
        { id: 'u-r', from: 'user', to: 'router' },
        { id: 'r-l', from: 'router', to: 'local', label: 'easy' },
        { id: 'r-c', from: 'router', to: 'big', label: 'hard' },
      ],
      zones: [
        { id: 'near', label: 'Your side', nodes: ['router', 'local'] },
        { id: 'far', label: 'Provider cloud', nodes: ['big'], pad: 0.9 },
      ],
    },
    sim: { type: 'cascade', every: 1.15, pattern: 'eeeheeeeeh', inPath: 'u-r', near: 'r-l', far: 'r-c' },
    status: { at: [-0.6, 4.7], y: 0.5 },
  },

  // ------------------------------------------------------------------ 8
  {
    id: 'federated',
    name: 'Federated / private learning',
    short: 'Federated',
    tagline: 'The model travels to the data',
    what: 'Instead of collecting user data, a server sends the current model to many devices. Each device trains briefly on its own data and returns only a weight update. The server averages the updates into the next global model (federated averaging) and the round repeats.',
    pathTitle: 'One training round',
    watch: 'Raw data only ever moves inside a device. What crosses to the server is a few small weight updates, and the improved model comes back down in the next round.',
    path: [
      '**Server** sends the current global model to a sample of devices.',
      '**Each device** trains for a few steps on its **private data**.',
      'Only the **weight update** travels back, often clipped, noised and securely aggregated.',
      '**Server** averages the updates into a new **global model**.',
    ],
    pros: ['Raw data never leaves the device', 'Learns from data that could never be pooled: keyboards, hospitals, banks', 'Uses the devices’ compute instead of a data-center bill', 'Combines with secure aggregation and differential privacy for formal guarantees'],
    cons: ['Hardest to operate: unreliable, uneven devices and skewed data', 'Updates can still leak information unless they are clipped, noised or securely aggregated', 'Slow: many rounds, each waiting for devices that are idle, charging and on Wi-Fi', 'Limited to models small enough to train on a device'],
    choose: ['Valuable data is spread over devices or organizations and may not be pooled', 'Regulation or trust rules out central collection', 'On-device personalization, or one model across several hospitals or banks'],
    ratings: { latency: 4, costLow: 3, costHigh: 4, privacy: 4, ops: 1, size: 1 },
    ratingNote: 'Latency is rated for inference, which runs on the device. Training itself takes many slow rounds.',
    stack: [
      ['Frameworks', ['Flower', 'NVIDIA FLARE', 'TensorFlow Federated', 'OpenFL']],
      ['Privacy tools', ['Opacus (differential privacy)', 'Secure aggregation']],
    ],
    legend: [['response', 'Global model, sent to the devices'], ['data', 'Raw data (never leaves a device)'], ['update', 'Weight updates, sent to the server']],
    graph: {
      nodes: [
        { id: 'srv', type: 'service', label: 'Aggregation server', pos: [0, -4.2], tip: 'Coordinates the rounds and averages the updates. It never receives raw data.' },
        { id: 'glob', type: 'model', label: 'Global model', pos: [4.8, -4.2], tip: 'The shared model that improves round by round.', wave: { period: 13, from: 0.8, to: 0.99, low: 0.55 } },
        ...each(4, (i) => [
          { id: 'd' + i, type: i % 2 ? 'device' : 'laptop', label: i % 2 ? 'Phone' : 'Laptop', pos: [-10.5 + i * 4.2, 0.4], tip: 'Trains a local copy of the model for a few steps, then sends back only the change in the weights.', wave: { period: 13, from: 0.2, to: 0.52, low: 0.5 } },
          { id: 'x' + i, type: 'table', label: 'Private data', pos: [-10.5 + i * 4.2, 3.6], tip: 'Messages, photos, records. They stay here.' },
        ]),
      ],
      edges: [
        ...each(4, (i) => [
          { id: 's-d' + i, from: 'srv', to: 'd' + i },
          { id: 'l' + i, from: 'x' + i, to: 'd' + i },
        ]),
        { id: 's-g', from: 'srv', to: 'glob', both: true },
      ],
      zones: [
        { id: 'zs', label: 'Server: sees updates, never raw data', nodes: ['srv', 'glob'], pad: 0.6 },
        ...each(4, (i) => [{ id: 'z' + i, label: 'Device ' + i, nodes: ['d' + i, 'x' + i], pad: 0.45 }]),
      ],
    },
    flows: [
      ...each(4, (i) => [
        { path: 's-d' + i, kind: 'response', count: 2, size: 0.1, gap: 0.5, cycle: [13, 0.02, 0.17] },
        { path: 'l' + i, kind: 'data', count: 3, size: 0.07, gap: 0.3, cycle: [13, 0.2, 0.34] },
        { path: 'l' + i, kind: 'data', count: 3, size: 0.07, gap: 0.3, cycle: [13, 0.36, 0.5] },
        { path: '-s-d' + i, kind: 'update', count: 3, size: 0.055, gap: 0.3, cycle: [13, 0.56, 0.74] },
      ]),
      { path: 's-g', kind: 'update', count: 3, size: 0.07, gap: 0.3, cycle: [13, 0.76, 0.86] },
    ],
    status: { at: [0, -4.2], period: 13, steps: [[0, 'The model goes to the devices'], [0.19, 'Training locally: data stays put'], [0.55, 'Only weight updates travel'], [0.76, 'The server averages the updates']] },
  },

  // ------------------------------------------------------------------ 9
  {
    id: 'training-cluster',
    name: 'Distributed training cluster',
    short: 'Training cluster',
    tagline: 'Many GPUs acting as one',
    what: 'Training a large model needs more memory and compute than one GPU has, so the work is split three ways at once. **Data parallelism**: copies of the model each take a different slice of the batch and average their gradients. **Tensor parallelism**: the matrices of one layer are split across GPUs that work on the same tokens. **Pipeline parallelism**: consecutive layers live on different GPUs and hand activations from stage to stage.',
    pathTitle: 'One training step',
    watch: 'Batches fan out to two replicas (data parallel). Inside a replica, activations flow from the left pair of GPUs to the right pair and gradients flow back (pipeline parallel), while the two GPUs of each pair keep exchanging partial results (tensor parallel). Then the replicas average their gradients.',
    path: [
      '**Data loader** cuts each batch into one piece per replica.',
      '**Forward**: activations pass through the pipeline stages; inside a stage two GPUs share every layer.',
      '**Backward**: gradients return through the stages in reverse.',
      '**All-reduce** averages the gradients across replicas, the optimizer steps, and a **checkpoint** is written now and then.',
    ],
    pros: ['The only way to train models that do not fit on one GPU', 'Close to linear speed-up when the interconnect is fast', 'The same recipe scales from eight GPUs to thousands'],
    cons: ['Communication is the bottleneck: it needs InfiniBand or RoCE class networking', 'One failed GPU can stall the whole job, so checkpoints and restarts are routine', 'An enormous fixed cost, and idle GPUs are pure waste', 'Hard to debug: hangs, stragglers, silent numerical divergence'],
    choose: ['Pre-training or large fine-tuning runs, not serving', 'The model or the batch does not fit on one GPU or one node', 'You have the data, the budget and the team to keep a cluster busy'],
    ratings: { latency: null, costLow: 1, costHigh: 3, privacy: 4, ops: 1, size: 5 },
    ratingNote: 'Latency does not apply: training is a batch job measured in throughput, not a request path.',
    stack: [
      ['Frameworks', ['PyTorch (DDP, FSDP)', 'JAX', 'DeepSpeed', 'Megatron-LM']],
      ['Scheduling', ['Slurm', 'Kubeflow Trainer', 'Ray Train']],
      ['Interconnect', ['NCCL', 'InfiniBand / RoCE']],
    ],
    facts: ['trillion'],
    legend: [['data', 'Batches and checkpoints'], ['request', 'Activations (forward)'], ['update', 'Gradients (backward, all-reduce)'], ['control', 'Tensor-parallel exchange inside a layer']],
    graph: {
      nodes: [
        { id: 'ds', type: 'database', label: 'Dataset shards', pos: [-6.4, -2.4] },
        { id: 'ld', type: 'queue', label: 'Data loader', pos: [-6.4, 1.6], tip: 'Tokenizes, shuffles and hands every replica a different slice of the batch.' },
        ...[['a', -4.8, 0], ['b', 1.3, 4]].flatMap(([r, z, n]) => [
          { id: r + '1', type: 'gpu', size: 0.8, label: 'GPU ' + n, pos: [-2.2, z], tip: 'Stage 1: the first half of the layers, shared with the GPU next to it.' },
          { id: r + '2', type: 'gpu', size: 0.8, label: 'GPU ' + (n + 1), pos: [-2.2, z + 2.4], tip: 'Stage 1: the first half of the layers, shared with the GPU next to it.' },
          { id: r + '3', type: 'gpu', size: 0.8, label: 'GPU ' + (n + 2), pos: [1.9, z], tip: 'Stage 2: the second half of the layers, shared with the GPU next to it.' },
          { id: r + '4', type: 'gpu', size: 0.8, label: 'GPU ' + (n + 3), pos: [1.9, z + 2.4], tip: 'Stage 2: the second half of the layers, shared with the GPU next to it.' },
        ]),
        { id: 'sync', type: 'op', label: 'All-reduce', pos: [6, -0.5], tip: 'A collective operation: every replica ends up with the average gradient.' },
        { id: 'ckpt', type: 'store', label: 'Checkpoints', pos: [6, 3.4], tip: 'Periodic snapshots, so a failed GPU costs minutes and not days.' },
      ],
      edges: [
        { id: 'ds-ld', from: 'ds', to: 'ld' },
        { id: 'ld-a', from: 'ld', to: 'a2' },
        { id: 'ld-b', from: 'ld', to: 'b1' },
        ...['a', 'b'].flatMap((r) => [
          { id: 't' + r + '1', from: r + '1', to: r + '2', both: true, bend: -0.5 },
          { id: 't' + r + '2', from: r + '3', to: r + '4', both: true, bend: 0.5 },
          { id: 'p' + r + '1', from: r + '1', to: r + '3' },
          { id: 'p' + r + '2', from: r + '2', to: r + '4' },
        ]),
        { id: 'a-s', from: 'a4', to: 'sync', both: true },
        { id: 'b-s', from: 'b3', to: 'sync', both: true },
        { id: 's-c', from: 'sync', to: 'ckpt', dashed: true },
      ],
      zones: [
        { id: 'ra', label: 'Replica A: stage 1 → stage 2', nodes: ['a1', 'a2', 'a3', 'a4'], pad: 0.55 },
        { id: 'rb', label: 'Replica B: second copy', nodes: ['b1', 'b2', 'b3', 'b4'], pad: 0.55 },
      ],
    },
    flows: [
      { path: 'ds-ld', kind: 'data', count: 2, size: 0.09, gap: 0.4, cycle: [10, 0, 0.1] },
      { path: 'ld-a', kind: 'data', count: 2, size: 0.09, gap: 0.4, cycle: [10, 0.1, 0.2] },
      { path: 'ld-b', kind: 'data', count: 2, size: 0.09, gap: 0.4, cycle: [10, 0.1, 0.2] },
      ...['ta1', 'ta2', 'tb1', 'tb2'].flatMap((e) => [
        { path: e, kind: 'control', count: 1, speed: 1.7, size: 0.05 },
        { path: '-' + e, kind: 'control', count: 1, speed: 1.7, size: 0.05, phase: 0.5 },
      ]),
      ...['pa1', 'pa2', 'pb1', 'pb2'].flatMap((e) => [
        { path: e, kind: 'request', count: 3, size: 0.08, gap: 0.4, cycle: [10, 0.22, 0.42] },
        { path: '-' + e, kind: 'update', count: 3, size: 0.07, gap: 0.4, cycle: [10, 0.5, 0.7] },
      ]),
      ...['a-s', 'b-s'].flatMap((e) => [
        { path: e, kind: 'update', count: 2, size: 0.07, gap: 0.4, cycle: [10, 0.72, 0.8] },
        { path: '-' + e, kind: 'update', count: 2, size: 0.07, gap: 0.4, cycle: [10, 0.82, 0.9] },
      ]),
      { path: 's-c', kind: 'data', count: 3, size: 0.08, gap: 0.4, cycle: [30, 0.3, 0.33] },
    ],
    status: { at: [0, -5], period: 10, steps: [[0, 'Data parallel: one batch per replica'], [0.21, 'Forward: stage 1 to stage 2'], [0.49, 'Backward: gradients return'], [0.71, 'All-reduce: replicas average gradients']] },
  },

  // ------------------------------------------------------------------ 10
  {
    id: 'air-gapped',
    name: 'Air-gapped on-premises',
    short: 'Air-gapped',
    tagline: 'Sovereign: no wire to the outside',
    what: 'The whole stack (users, gateway, GPUs, weights and data) sits inside a facility with no network connection to the outside. Models, containers and updates come in on signed, scanned media or through a one-way data diode.',
    watch: 'Requests circulate inside the wall. Traffic from outside stops at the wall: there is no route in, and no route out for a leak.',
    path: [
      '**Workstation** inside the facility sends the request.',
      '**Internal gateway** authenticates and writes the audit log.',
      '**GPU servers** run an open-weight model and retrieve from **internal data**.',
      'New **model weights** arrive only through the **import station**, by hand.',
    ],
    pros: ['Strongest isolation: nothing can leak over a link that does not exist', 'Full sovereignty: your hardware, your jurisdiction, your audit trail', 'Unaffected by a provider’s outages, price changes or access suspensions', 'Predictable performance on dedicated hardware'],
    cons: ['Highest fixed cost: you buy the GPUs, the room, the power and the staff', 'Every update is a manual, audited import, so models lag behind', 'Limited to open-weight models or a vendor appliance, with no burst capacity', 'You need in-house expertise for everything'],
    choose: ['Classified, defense, critical-infrastructure or highly regulated data', 'Law or policy demands sovereignty over data and models', 'A leak would cost more than running everything yourself'],
    ratings: { latency: 4, costLow: 1, costHigh: 3, privacy: 5, ops: 1, size: 4 },
    stack: [
      ['Inference', ['vLLM', 'SGLang', 'llama.cpp', 'NVIDIA NIM']],
      ['Platform', ['Kubernetes (RKE2, OpenShift)', 'Harbor registry', 'Zarf']],
      ['Front end', ['Open WebUI']],
    ],
    facts: ['open'],
    legend: [['request', 'Prompt'], ['response', 'Streamed tokens'], ['data', 'Retrieved documents, imported weights'], ['blocked', 'Outside traffic, stopped at the wall']],
    graph: {
      nodes: [
        { id: 'net', type: 'globe', label: 'Internet', pos: [-4.4, -6.5], tip: 'Not connected. Nothing from here reaches the facility.' },
        { id: 'hub', type: 'cloud', label: 'Model hub', pos: [4.4, -6.5], tip: 'New models are fetched elsewhere, checked, and carried in by hand.' },
        { id: 'imp', type: 'lock', label: 'Import station', pos: [-5, -0.2], tip: 'Signed, scanned media or a one-way data diode: the only way anything gets in.' },
        { id: 'wts', type: 'store', label: 'Model weights', pos: [-0.4, -0.2], tip: 'Open-weight models or a vendor appliance, stored inside.' },
        { id: 'docs', type: 'database', label: 'Internal data', pos: [5.2, -0.2], tip: 'The documents the model retrieves from. They never leave.' },
        { id: 'ws', type: 'laptop', label: 'Workstation', pos: [-5, 3.6] },
        { id: 'gw', type: 'gateway', label: 'Internal gateway', pos: [-0.4, 3.6], tip: 'Authentication, access rules and a full audit log.' },
        { id: 'gpu', type: 'gpu', label: 'GPU servers', pos: [5.2, 3.6], size: 1.2, tip: 'Hardware you own, in a room you control.' },
      ],
      edges: [
        { id: 'w-g', from: 'ws', to: 'gw' },
        { id: 'g-s', from: 'gw', to: 'gpu' },
        { id: 's-d', from: 'gpu', to: 'docs', both: true },
        { id: 'i-m', from: 'imp', to: 'wts', dashed: true, label: 'signed media' },
        { id: 'm-s', from: 'wts', to: 'gpu', dashed: true },
      ],
      zones: [{ id: 'site', label: 'Air-gapped facility', rect: [-7, -1.9, 7.2, 5.7], strong: true }],
    },
    flows: [
      { path: ['w-g', 'g-s'], kind: 'request', count: 1, size: 0.13, cycle: [9, 0, 0.2] },
      { path: 's-d', kind: 'request', count: 1, size: 0.08, cycle: [9, 0.21, 0.28] },
      { path: '-s-d', kind: 'data', count: 3, size: 0.07, gap: 0.3, cycle: [9, 0.3, 0.4] },
      { path: ['-g-s', '-w-g'], kind: 'response', count: 6, size: 0.065, gap: 0.4, cycle: [9, 0.42, 0.78] },
      { path: 'i-m', kind: 'data', count: 2, size: 0.1, gap: 0.5, cycle: [27, 0.02, 0.1] },
      { path: 'm-s', kind: 'data', count: 2, size: 0.1, gap: 0.5, cycle: [27, 0.1, 0.17] },
    ],
    props: { wall: { z: -2.7, x0: -7.6, x1: 7.8, label: 'AIR GAP · NO NETWORK LINK' }, blocked: ['net', 'hub'] },
    status: { at: [0, -2.7], y: 2.3, steps: [[0, 'No route in or out']] },
  },
];
