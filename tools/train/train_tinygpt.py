"""Train the ~1M-parameter GPT that the website dissects in 3D, then export it for the browser.

  python tools/train/train_tinygpt.py --data DIR --out public/models --minutes 60

DIR must contain TinyStories text (train_head.txt and valid.txt, see tools/train/README.md).
Outputs:
  tinygpt.bin         JSON header + float16 weights
  tinygpt-vocab.json  BPE vocabulary, merges and encode() test vectors for the JS port
  tinygpt-eval.json   held-out token ids + PyTorch reference loss (used by the JS parity test
                      and by the in-browser quantisation experiment)
"""
import argparse
import json
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe import EOT, Tokenizer, normalise, pretokenise, train_bpe

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--work", default=None, help="cache dir for tokenised data / checkpoints")
p.add_argument("--minutes", type=float, default=60)
p.add_argument("--vocab", type=int, default=2048)
p.add_argument("--d", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--ctx", type=int, default=128)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--threads", type=int, default=3)
p.add_argument("--export-only", action="store_true")
args = p.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(1337)
np.random.seed(1337)
work = args.work or os.path.join(args.data, "work")
os.makedirs(work, exist_ok=True)
os.makedirs(args.out, exist_ok=True)


# ----------------------------------------------------------------------------- data
def read_stories(path, drop_first=False, drop_last=False):
    text = open(path, encoding="utf-8", errors="replace").read()
    stories = [normalise(s).strip() for s in text.split(EOT)]
    if drop_first:
        stories = stories[1:]
    if drop_last:
        stories = stories[:-1]
    return [s for s in stories if len(s) > 40]


tok_path = os.path.join(work, f"tok{args.vocab}.json")
train_path = os.path.join(work, f"train{args.vocab}.npy")
val_path = os.path.join(work, f"val{args.vocab}.npy")

if not (os.path.exists(tok_path) and os.path.exists(train_path) and os.path.exists(val_path)):
    t0 = time.time()
    train_stories = read_stories(os.path.join(args.data, "train_head.txt"), drop_last=True)
    val_stories = read_stories(os.path.join(args.data, "valid.txt"), drop_first=True)
    print(f"stories: train {len(train_stories)}, val {len(val_stories)}", flush=True)
    import collections

    freq = collections.Counter()
    for s in train_stories:
        freq.update(pretokenise(s))
    print(f"unique pre-tokens: {len(freq)} ({time.time() - t0:.0f}s)", flush=True)
    vocab, merges = train_bpe(freq, args.vocab)
    json.dump({"vocab": vocab, "merges": merges}, open(tok_path, "w"))
    tok = Tokenizer(vocab, merges)

    def encode_all(stories):
        out = []
        for s in stories:
            out.append(0)
            out.extend(tok.encode(s))
        return np.array(out, dtype=np.uint16)

    np.save(train_path, encode_all(train_stories))
    np.save(val_path, encode_all(val_stories))
    print(f"tokenised in {time.time() - t0:.0f}s", flush=True)

tj = json.load(open(tok_path))
tok = Tokenizer(tj["vocab"], tj["merges"])
train_ids = torch.from_numpy(np.load(train_path).astype(np.int64))
val_ids = torch.from_numpy(np.load(val_path).astype(np.int64))
V = len(tok.vocab)
print(f"vocab {V}, train tokens {len(train_ids):,}, val tokens {len(val_ids):,}", flush=True)


# ----------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))
        return x


class TinyGPT(nn.Module):
    def __init__(self, V, d, layers, heads, ctx):
        super().__init__()
        self.ctx = ctx
        self.wte = nn.Embedding(V, d)
        self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(b.out.weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        logits = self.ln_f(x) @ self.wte.weight.t()  # tied input/output embedding
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


model = TinyGPT(V, args.d, args.layers, args.heads, args.ctx)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}", flush=True)
ckpt_path = os.path.join(work, "ckpt.pt")


def get_batch(ids, batch):
    ix = torch.randint(len(ids) - args.ctx - 1, (batch,))
    x = torch.stack([ids[i : i + args.ctx] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + args.ctx] for i in ix])
    return x, y


@torch.no_grad()
def estimate(ids, n=24):
    model.eval()
    g = torch.Generator().manual_seed(7)
    tot = 0.0
    for _ in range(n):
        ix = torch.randint(len(ids) - args.ctx - 1, (args.batch,), generator=g)
        x = torch.stack([ids[i : i + args.ctx] for i in ix])
        y = torch.stack([ids[i + 1 : i + 1 + args.ctx] for i in ix])
        tot += model(x, y)[1].item()
    model.train()
    return tot / n


history = []
if not args.export_only:
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    budget = args.minutes * 60
    t0 = time.time()
    step = 0
    warm = 100
    while True:
        el = time.time() - t0
        frac = el / budget
        if frac >= 1:
            break
        lr = args.lr * min(1.0, (step + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_ids, args.batch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0 or step == 1:
            vl = estimate(val_ids, 8)
            history.append({"step": step, "minutes": round(el / 60, 2), "train": round(loss.item(), 4), "val": round(vl, 4)})
            print(f"step {step:5d}  {el / 60:5.1f} min  lr {lr:.5f}  train {loss.item():.3f}  val {vl:.3f}", flush=True)
            torch.save({"model": model.state_dict(), "history": history, "step": step}, ckpt_path)
    torch.save({"model": model.state_dict(), "history": history, "step": step}, ckpt_path)

ck = torch.load(ckpt_path)
model.load_state_dict(ck["model"])
history = ck["history"]
model.eval()
final_val = estimate(val_ids, 40)
print(f"final val loss {final_val:.4f}  (perplexity {math.exp(final_val):.2f})", flush=True)


# ----------------------------------------------------------------------------- samples
@torch.no_grad()
def generate(prompt, n=60, temperature=0.8, top_k=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = [0] + tok.encode(prompt)
    for _ in range(n):
        x = torch.tensor([ids[-args.ctx :]])
        logits = model(x)[0, -1] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g).item()
        if nxt == 0:
            break
        ids.append(nxt)
    return tok.decode(ids[1:])


for prm in ["Once upon a time", "The little dog", "Lily and Tom went to the park. They"]:
    print("---", generate(prm), flush=True)

# ----------------------------------------------------------------------------- export
sd = model.state_dict()
tensors = [("wte", sd["wte.weight"]), ("wpe", sd["wpe.weight"])]
for i in range(args.layers):
    pre = f"blocks.{i}."
    for short, key in [
        ("ln1.w", "ln1.weight"), ("ln1.b", "ln1.bias"), ("attn.w", "attn.weight"), ("attn.b", "attn.bias"),
        ("proj.w", "proj.weight"), ("proj.b", "proj.bias"), ("ln2.w", "ln2.weight"), ("ln2.b", "ln2.bias"),
        ("fc.w", "fc.weight"), ("fc.b", "fc.bias"), ("out.w", "out.weight"), ("out.b", "out.bias"),
    ]:
        tensors.append((f"h{i}.{short}", sd[pre + key]))
tensors += [("lnf.w", sd["ln_f.weight"]), ("lnf.b", sd["ln_f.bias"])]

blob = bytearray()
meta = []
for name, t in tensors:
    a = t.detach().numpy().astype(np.float16)
    meta.append({"name": name, "shape": list(a.shape), "offset": len(blob), "length": int(a.size)})
    blob += a.tobytes()
header = json.dumps({
    "format": "tinygpt-f16-v1",
    "config": {"vocab": V, "d": args.d, "layers": args.layers, "heads": args.heads, "ctx": args.ctx},
    "params": n_params,
    "val_loss": round(final_val, 4),
    "train_steps": ck["step"],
    "history": history,
    "tensors": meta,
}).encode()
pad = (-(8 + len(header))) % 4
header += b" " * pad
with open(os.path.join(args.out, "tinygpt.bin"), "wb") as f:
    f.write(b"TGPT")
    f.write(struct.pack("<I", len(header)))
    f.write(header)
    f.write(blob)
print(f"wrote tinygpt.bin ({(8 + len(header) + len(blob)) / 1e6:.2f} MB)", flush=True)

tests = []
for s in ["Once upon a time, there was a little girl named Lily.", 'Tom said, "I don\'t know!" He was 5 years old.',
          "The transformer predicts the next token.", "  two  spaces\nand a newline", "She's happy; they're not... we'll see."]:
    tests.append({"text": s, "ids": tok.encode(s)})
json.dump({"vocab": tok.vocab, "merges": tok.merges, "tests": tests}, open(os.path.join(args.out, "tinygpt-vocab.json"), "w"),
          ensure_ascii=True, separators=(",", ":"))

# held-out evaluation text: FP32 model loss on these exact windows, with the float16-rounded weights
with torch.no_grad():
    for name, t in tensors:
        t.copy_(torch.from_numpy(t.numpy().astype(np.float16).astype(np.float32)))
    n_eval = 16
    start = len(val_ids) // 2
    windows = []
    losses = []
    pos = start
    while len(windows) < n_eval:
        # start every window on a story boundary so the browser sees complete beginnings
        while val_ids[pos] != 0:
            pos += 1
        w = val_ids[pos : pos + args.ctx + 1]
        pos += args.ctx + 1
        x, y = w[:-1][None], w[1:][None]
        _, l = model(x, y)
        windows.append(w.tolist())
        losses.append(l.item())
    probe = [0] + tok.encode("Once upon a time, there was a little girl named")
    lg = model(torch.tensor([probe]))[0, -1]
    top = torch.topk(F.softmax(lg, -1), 8)
json.dump({
    "windows": windows,
    "ref_loss": round(float(np.mean(losses)), 5),
    "ref_window_losses": [round(x, 5) for x in losses],
    "probe": {"ids": probe, "top_ids": top.indices.tolist(), "top_probs": [round(x, 5) for x in top.values.tolist()],
              "logits_head": [round(x, 4) for x in lg[:16].tolist()]},
}, open(os.path.join(args.out, "tinygpt-eval.json"), "w"), separators=(",", ":"))
print(f"eval reference loss {np.mean(losses):.4f} over {n_eval} windows; probe top: "
      f"{[tok.vocab[i] for i in top.indices.tolist()]}", flush=True)
