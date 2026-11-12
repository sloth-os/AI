// A dated snapshot of the fast-moving parts of the AI landscape, gathered from primary pages on
// 12 Nov 2026. Everything here carries its source so the site can show where a number came from.
// Fundamentals (how attention works, what a KV cache is) do not live here: they do not expire.

export const AS_OF = '12 Nov 2026';

/** Flagship models by lab. weights: 'closed' | 'open' (license in note). ctx = context window in tokens. */
export const FRONTIER = [
  { lab: 'OpenAI', model: 'GPT-6 Astra', date: 'Sep 2026', ctx: 1_050_000, weights: 'closed', note: 'Family: Astra, Luna (small), 6.1 Sol and Terra. 128K max output.', src: 'https://developers.openai.com/api/docs/models' },
  { lab: 'Anthropic', model: 'Claude Fable 5.2', date: 'Nov 2026', ctx: 1_000_000, weights: 'closed', note: 'Top tier above Opus 5.5 (the default), Sonnet 5.5 and Haiku 5.5.', src: 'https://platform.claude.com/docs/en/about-claude/models/overview' },
  { lab: 'Google DeepMind', model: 'Gemini 4 Carbon', date: 'Nov 2026', ctx: 1_000_000, weights: 'closed', note: 'With Gemini 4 Argon and 3.8 Flash. Open line: Gemma 4 (Apache 2.0).', src: 'https://aistudio.google.com/docs/models' },
  { lab: 'Meta', model: 'Muse Spark Open', date: 'Oct 2026', ctx: 1_000_000, weights: 'open', note: '816B-parameter mixture-of-experts, 42B active per token. Custom gated license.', src: 'https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B' },
  { lab: 'xAI', model: 'Grok 4.7', date: 'Sep 2026', ctx: 500_000, weights: 'closed', note: '', src: 'https://docs.x.ai/docs/models' },
  { lab: 'Mistral', model: 'Mistral Large 4', date: 'Oct 2026', ctx: null, weights: 'open', note: '1.05T-parameter MoE, 52B active. Custom open-weight license.', src: 'https://docs.mistral.ai/getting-started/models/models_overview/' },
  { lab: 'DeepSeek', model: 'DeepSeek-V4 Pro / V4.1 Flash', date: 'Aug–Oct 2026', ctx: 1_000_000, weights: 'open', note: 'MIT-licensed weights; Pro is listed at 1.7T parameters.', src: 'https://api-docs.deepseek.com/quick_start/pricing' },
  { lab: 'Alibaba', model: 'Qwen3.8', date: 'Aug 2026', ctx: 1_000_000, weights: 'open', note: '2.4T MoE (95B active) plus a 27B Apache-2.0 model that mixes linear and full attention.', src: 'https://www.alibabacloud.com/blog/603463' },
  { lab: 'Moonshot', model: 'Kimi K3.1 Pro', date: 'Oct 2026', ctx: 1_048_576, weights: 'open', note: '2.8T MoE, 104B active; ships 4-bit floating-point (MXFP4) weights.', src: 'https://huggingface.co/moonshotai/Kimi-K3.1-Pro' },
  { lab: 'Z.ai (Zhipu)', model: 'GLM-5.3', date: 'Aug 2026', ctx: null, weights: 'open', note: '753B parameters, custom license.', src: 'https://huggingface.co/zai-org/GLM-5.3' },
];

/** What people actually run on their own machines: Ollama pull counts (millions), Nov 2026. */
export const SELF_HOSTED = {
  src: 'https://ollama.com/library?sort=popular',
  rows: [
    ['llama3.1', 121], ['deepseek-r1', 95], ['llama3.2', 87.5], ['qwen3', 43.2], ['gemma3', 41.8], ['gemma4', 29], ['qwen3.5', 25.2], ['gpt-oss', 14.9],
  ],
};

/**
 * List price per million tokens (USD, input / output, standard tier) at launch or as listed on the date.
 * The early rows are historical list prices; the Nov 2026 rows come from the vendors' pricing pages.
 */
export const PRICES = [
  { model: 'GPT-4 (8K)', when: '2023-03', inp: 30, out: 60, tier: 'flagship' },
  { model: 'GPT-4 Turbo', when: '2023-11', inp: 10, out: 30, tier: 'flagship' },
  { model: 'Claude 3 Opus', when: '2024-03', inp: 15, out: 75, tier: 'flagship' },
  { model: 'GPT-4o', when: '2024-05', inp: 5, out: 15, tier: 'flagship' },
  { model: 'Claude 3.5 Sonnet', when: '2024-06', inp: 3, out: 15, tier: 'flagship' },
  { model: 'GPT-4o mini', when: '2024-07', inp: 0.15, out: 0.6, tier: 'small' },
  { model: 'GPT-5', when: '2025-08', inp: 1.25, out: 10, tier: 'flagship' },
  { model: 'GPT-5 nano', when: '2025-08', inp: 0.05, out: 0.4, tier: 'small' },
  { model: 'GPT-6 Astra', when: '2026-11', inp: 10, out: 50, tier: 'flagship', src: 'https://developers.openai.com/api/docs/pricing' },
  { model: 'Claude Fable 5.2', when: '2026-11', inp: 8, out: 40, tier: 'flagship', src: 'https://platform.claude.com/docs/en/about-claude/pricing' },
  { model: 'Claude Opus 5.5', when: '2026-11', inp: 4, out: 20, tier: 'flagship', src: 'https://platform.claude.com/docs/en/about-claude/pricing' },
  { model: 'Gemini 4 Carbon', when: '2026-11', inp: 4, out: 20, tier: 'flagship', note: '$2 / $10 introductory price until 30 Nov 2026', src: 'https://aistudio.google.com/docs/pricing' },
  { model: 'GPT-6.1 Sol', when: '2026-11', inp: 2, out: 10, tier: 'mid', src: 'https://developers.openai.com/api/docs/pricing' },
  { model: 'Claude Sonnet 5.5', when: '2026-11', inp: 2, out: 10, tier: 'mid', src: 'https://platform.claude.com/docs/en/about-claude/pricing' },
  { model: 'DeepSeek V4 Pro', when: '2026-11', inp: 1.32, out: 3.96, tier: 'mid', note: 'peak-hours price; half off-peak', src: 'https://api-docs.deepseek.com/quick_start/pricing' },
  { model: 'GPT-6 Luna', when: '2026-11', inp: 0.1, out: 0.5, tier: 'small', src: 'https://developers.openai.com/api/docs/pricing' },
  { model: 'Claude Haiku 5.5', when: '2026-11', inp: 0.1, out: 0.5, tier: 'small', note: 'prompts up to 100K tokens', src: 'https://platform.claude.com/docs/en/about-claude/pricing' },
];

/** Agent and tool protocols with real adoption in late 2026. */
export const PROTOCOLS = [
  {
    id: 'mcp', name: 'Model Context Protocol (MCP)', role: 'model ↔ tools and data',
    status: 'Spec revision 2026-07-28. Transports: stdio and Streamable HTTP (the old HTTP+SSE transport is deprecated). That revision made the protocol stateless: no initialize handshake, no session id. Governed by the Linux Foundation’s Agentic AI Foundation.',
    src: 'https://blog.modelcontextprotocol.io/posts/2026-07-28/',
  },
  {
    id: 'a2a', name: 'Agent2Agent (A2A)', role: 'agent ↔ agent',
    status: 'v1.0 in March 2026; joined the Agentic AI Foundation in August 2026, backed by 150+ organizations. Agents publish an Agent Card, then exchange tasks, messages and artifacts.',
    src: 'https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/',
  },
  { id: 'agui', name: 'AG-UI', role: 'agent ↔ user interface', status: 'Event stream between an agent backend and a front end (messages, tool calls, state patches). Integrated by several agent frameworks.', src: 'https://github.com/ag-ui-protocol/ag-ui' },
  { id: 'acp', name: 'Agent Client Protocol', role: 'editor ↔ coding agent', status: 'JSON-RPC over stdio between an editor and a coding agent, the way LSP connects editors to language servers.', src: 'https://agentclientprotocol.com/overview/introduction' },
  { id: 'x402', name: 'x402 / AP2', role: 'agent payments', status: 'x402 Foundation launched under the Linux Foundation in July 2026; Google donated its Agent Payments Protocol to the FIDO Alliance in April 2026.', src: 'https://www.linuxfoundation.org/press/linux-foundation-announces-operational-launch-of-x402-foundation-to-standardize-internet-native-payments-for-ai-agents-and-applications' },
];

/** Changes in the serving stack worth knowing in late 2026. */
export const SERVING_NOTES = [
  { text: 'Hugging Face TGI was archived in March 2026; its README now points to vLLM, SGLang, llama.cpp and MLX.', src: 'https://github.com/huggingface/text-generation-inference' },
  { text: 'vLLM and SGLang are the default GPU serving engines on new open-weight model cards; NVIDIA Dynamo orchestrates them (and TensorRT-LLM) across a cluster.', src: 'https://github.com/ai-dynamo/dynamo/releases' },
  { text: 'One independent test (H100, ShareGPT, 60 concurrent requests, Qwen2.5-7B): vLLM 6,927 tokens/s, SGLang 6,638, TensorRT-LLM 6,151. On a 30B mixture-of-experts model the order reversed. No engine wins everywhere.', src: 'https://jarvislabs.ai/blog/vllm-sglang-trtllm-comparison' },
  { text: 'WebGPU now ships by default in Chrome, Edge, Safari 26 and Firefox (Windows, Apple-silicon macOS), which is what makes in-browser inference practical.', src: 'https://web.dev/blog/webgpu-supported-major-browsers' },
  { text: 'transformers.js v4 (March 2026) reports a 20B model (4-bit) running at about 60 tokens/s in a WebGPU runtime on an M4 laptop.', src: 'https://github.com/huggingface/transformers.js/releases' },
];

/** Techniques that moved from papers into shipping models during 2026. */
export const TECHNIQUES_2026 = [
  { name: 'Trillion-parameter open-weight mixtures of experts', text: 'Qwen3.8 (2.4T), Kimi K3.1 Pro (2.8T), DeepSeek-V4 Pro (1.7T) and Mistral Large 4 (1.05T) activate only 50–100B parameters per token.', src: 'https://huggingface.co/moonshotai/Kimi-K3.1-Pro' },
  { name: 'Hybrid linear + full attention', text: 'Qwen3.8-27B interleaves three linear-time Gated DeltaNet layers with one full-attention layer, repeated 16 times.', src: 'https://huggingface.co/Qwen/Qwen3.8-27B' },
  { name: '4-bit floating-point weights', text: 'Models are now released in MXFP4 / NVFP4 with quantization-aware training: Muse Spark Open needs about 700 GB of VRAM in NVFP4 versus 2 TB in BF16.', src: 'https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B' },
  { name: 'Ternary (1.58-bit) models', text: 'Weights restricted to −1, 0, +1. A 27B ternary model is among the most downloaded on Hugging Face.', src: 'https://prismml.com/news/prismml-introduces-ternary-bonsai-model-family' },
  { name: 'Adaptive thinking', text: 'Reasoning is always on and steered by an effort setting instead of a manual token budget.', src: 'https://platform.claude.com/docs/en/about-claude/models/overview' },
  { name: 'Text diffusion', text: 'Open-weight diffusion language models (for example DiffusionGemma) generate all positions in parallel and refine them. Still niche.', src: 'https://arxiv.org/abs/2608.00146' },
];

/** Benchmark snapshots. Scores are as reported on the linked page; harnesses differ, so compare within a row only. */
export const BENCHMARKS = [
  { name: 'ARC-AGI-3', what: 'novel interactive puzzles (standard harness)', rows: [['Claude Fable 5.2', 81.2], ['Gemini 4 Carbon', 65.1], ['GPT-6 Astra', 62.7]], unit: '%', src: 'https://arcprize.org/leaderboard' },
  { name: 'ARC-AGI-2', what: 'abstract visual reasoning', rows: [['GPT-6 Astra', 95.0], ['Gemini 4 Carbon', 94.2], ['Claude Fable 5.2', 93.8]], unit: '%', note: 'effectively saturated', src: 'https://arcprize.org/leaderboard' },
  { name: 'SWE-bench Verified', what: 'fix real GitHub issues (independent run, Sep 2026)', rows: [['Claude Opus 5', 97.0], ['DeepSeek V4 Pro', 96.4], ['GPT-5.6 Sol', 96.2]], unit: '%', note: 'saturated: leaderboard archived', src: 'https://www.vals.ai/benchmarks/swebench' },
  { name: 'OSWorld 2.1', what: 'operate a real desktop (lab-reported)', rows: [['Claude Fable 5.2', 89.3], ['Claude Opus 5.5', 87.2], ['Gemini 4 Carbon', 81.5], ['GPT-6 Astra', 79.5]], unit: '%', src: 'https://www.anthropic.com/claude-fable-and-mythos-5-2' },
];

/** METR's "time horizon": length of task (in human time) an agent completes with 50% reliability. */
export const METR = {
  src: 'https://metr.org/time-horizons/',
  latest: { model: 'Claude Mythos Preview', when: 'May 2026', minutes50: 1044.8, ci: [508.9, 3304.3], minutes80: 185.9 },
  caveat: 'METR calls measurements above 16 hours unreliable and stopped updating the page in September 2026: the benchmark ran out of long-enough tasks.',
};

/** Open-source projects that broke out in 2026 (GitHub stars as displayed on 12 Nov 2026). */
export const BREAKOUT_2026 = [
  { name: 'OpenClaw', repo: 'openclaw/openclaw', stars: '394k', text: 'Self-hosted personal assistant that acts through your messaging apps.' },
  { name: 'DeepSeek Harness', repo: 'deepseek-ai/deepseek-harness', stars: '280k', text: 'Plugin-based agent harness (MIT).' },
  { name: 'Hermes Agent', repo: 'NousResearch/hermes-agent', stars: '263k', text: 'Agent that writes and keeps its own skills.' },
  { name: 'OpenCode', repo: 'anomalyco/opencode', stars: '226k', text: 'Open-source terminal coding agent.' },
  { name: 'Codex CLI', repo: 'openai/codex', stars: '135k', text: 'OpenAI’s local coding-agent CLI.' },
  { name: 'pi', repo: 'earendil-works/pi', stars: '122k', text: 'Minimal, extensible agent harness.' },
  { name: 'Graphify', repo: 'Graphify-Labs/graphify', stars: '146k', text: 'Turns a codebase and its docs into a queryable knowledge graph.' },
  { name: 'Paperclip', repo: 'paperclipai/paperclip', stars: '113k', text: 'Orchestration for teams of agents.' },
];
