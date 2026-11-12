// Numerical checks of the JavaScript models against values computed by PyTorch at export time.
//   node tools/test-ml.mjs [modelsDir]        (default: public/models)
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { TinyGPT, Tokenizer, softmax, topK } from '../src/ml/tinygpt.js';

const dir = process.argv[2] || 'public/models';
let failed = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!ok) failed++;
};

// ---------------------------------------------------------------- tiny GPT
{
  const buf = readFileSync(join(dir, 'tinygpt.bin'));
  const model = TinyGPT.fromBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const vocab = JSON.parse(readFileSync(join(dir, 'tinygpt-vocab.json'), 'utf8'));
  const ev = JSON.parse(readFileSync(join(dir, 'tinygpt-eval.json'), 'utf8'));
  const tok = new Tokenizer(vocab);
  console.log(`tinygpt: ${model.paramCount.toLocaleString()} parameters, config ${JSON.stringify(model.config)}`);

  for (const t of vocab.tests) {
    const ids = tok.encode(t.text);
    check(`tokenizer "${t.text.slice(0, 28).replace(/\n/g, '\\n')}"`, JSON.stringify(ids) === JSON.stringify(t.ids), `${ids.length} tokens`);
    check('  round trip', tok.decode(ids) === tok.normalize(t.text));
  }

  // probe: same top tokens and logits as PyTorch
  const out = model.forward(ev.probe.ids, { capture: true });
  const head = Array.from(out.logits.slice(0, 16));
  const maxErr = Math.max(...head.map((v, i) => Math.abs(v - ev.probe.logits_head[i])));
  check('forward logits match PyTorch', maxErr < 2e-3, `max abs error ${maxErr.toExponential(2)}`);
  const top = topK(softmax(out.logits), 8);
  check('top-8 next tokens match', JSON.stringify(top.map((t) => t.id)) === JSON.stringify(ev.probe.top_ids), top.slice(0, 5).map((t) => `${JSON.stringify(tok.vocab[t.id])} ${(t.p * 100).toFixed(1)}%`).join(', '));
  const rowSum = out.acts.attn[0].subarray(0, out.n).reduce((a, b) => a + b, 0);
  check('attention rows sum to 1', Math.abs(rowSum - 1) < 1e-4);

  // KV-cached stepping gives the same logits as the full forward pass
  const cache = model.newCache();
  let stepLogits;
  for (const id of ev.probe.ids) stepLogits = model.step(id, cache);
  let diff = 0;
  for (let i = 0; i < stepLogits.length; i++) diff = Math.max(diff, Math.abs(stepLogits[i] - out.logits[i]));
  check('KV-cached step == full forward', diff < 1e-3, `max abs diff ${diff.toExponential(2)}`);

  // held-out loss on the reference windows
  const t0 = performance.now();
  const losses = ev.windows.slice(0, 6).map((w) => model.windowLoss(w));
  const ms = performance.now() - t0;
  const err = Math.max(...losses.map((l, i) => Math.abs(l - ev.ref_window_losses[i])));
  check('window losses match PyTorch', err < 2e-3, `max abs error ${err.toExponential(2)}; ${(ms / 6).toFixed(0)} ms per 128-token window`);

  // quantization sanity: error grows as bits shrink; 8-bit is nearly lossless
  const base = losses.slice(0, 3).reduce((a, b) => a + b, 0) / 3;
  let prev = base;
  for (const bits of [8, 4, 2]) {
    const q = model.quantized(bits, { method: 'mse' });
    const l = ev.windows.slice(0, 3).map((w) => q.windowLoss(w)).reduce((a, b) => a + b, 0) / 3;
    check(`quantized to ${bits} bits`, l >= prev - 0.02 && (bits !== 8 || l - base < 0.02), `loss ${l.toFixed(3)} (base ${base.toFixed(3)}), ${(q.sizeBytes(bits) / 1e6).toFixed(2)} MB`);
    prev = Math.max(prev, l);
  }
}

// ---------------------------------------------------------------- MNIST CNN (if present)
if (existsSync(join(dir, 'mnist-cnn.json')) && existsSync('src/ml/cnn.js')) {
  try {
    const cnn = await import('../src/ml/cnn.js');
    const data = JSON.parse(readFileSync(join(dir, 'mnist-cnn.json'), 'utf8'));
    check('mnist-cnn.json loads', !!data && typeof cnn === 'object');
  } catch (err) {
    check('mnist cnn module', false, String(err.message));
  }
}

if (failed) {
  console.log(`\n${failed} check(s) failed`);
  process.exit(1);
}
console.log('\nall checks passed');
