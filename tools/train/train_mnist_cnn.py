"""Train the ~6k-parameter CNN that the "How a CNN Sees" lab dissects, then export it for the browser.

  python tools/train/train_mnist_cnn.py --data DIR --out public/models [--work DIR2] [--epochs 16]

DIR holds the four MNIST idx .gz files (they are downloaded from
https://ossci-datasets.s3.amazonaws.com/mnist/ when missing). The checkpoint goes to --work
(default: DIR), so `--export-only` can rebuild the JSON without training again.

Architecture (no padding anywhere):
  input 1x28x28 -> conv 5x5, 8 maps (24x24) -> ReLU -> max-pool 2 (12x12)
                -> conv 5x5, 16 maps (8x8) -> ReLU -> max-pool 2 (4x4)
                -> flatten (256) -> linear -> 10 logits

Training uses random affine augmentation (shift, rotation, scale) so that the network tolerates
digits drawn by hand in the browser, which are only roughly MNIST-shaped.

Output: mnist-cnn.json
  weights      float32, little-endian, base64 (PyTorch layouts: conv [out, in, ky, kx], linear [out, in])
  test_accuracy, test_errors   measured on the 10,000 MNIST test images with the exported weights
  samples      24 test digits (uint8): two per class plus four that the network gets wrong
  ref_logits   PyTorch logits for those samples (the JS forward pass must reproduce them)
  ref_sums     per-layer activation sums for sample 0 (catches a wrong intermediate layer in the JS port)
"""
import argparse
import base64
import gzip
import json
import math
import os
import struct
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="directory with the MNIST idx .gz files")
p.add_argument("--work", default=None, help="directory for the checkpoint (default: --data)")
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=int, default=16)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=4e-3)
p.add_argument("--wd", type=float, default=2e-2)
p.add_argument("--shift", type=float, default=2.0, help="max translation in pixels")
p.add_argument("--rot", type=float, default=12.0, help="max rotation in degrees")
p.add_argument("--scale", type=float, nargs=2, default=[0.85, 1.15])
p.add_argument("--threads", type=int, default=2)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--export-only", action="store_true")
args = p.parse_args()

torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
work = args.work or args.data
os.makedirs(args.data, exist_ok=True)
os.makedirs(work, exist_ok=True)
os.makedirs(args.out, exist_ok=True)

# ----------------------------------------------------------------------------- data
URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def read_idx(name):
    path = os.path.join(args.data, name)
    if not os.path.exists(path):
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(URL + name, path)
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = struct.unpack(">I", raw[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim])
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


train_x = torch.from_numpy(read_idx(FILES["train_x"]).copy())  # 60000 x 28 x 28, uint8
train_y = torch.from_numpy(read_idx(FILES["train_y"]).astype(np.int64))
test_x = torch.from_numpy(read_idx(FILES["test_x"]).copy())
test_y = torch.from_numpy(read_idx(FILES["test_y"]).astype(np.int64))
print(f"train {tuple(train_x.shape)}, test {tuple(test_x.shape)}", flush=True)

# How MNIST digits sit in their frame (the browser pre-processing of a drawing must match this).
with torch.no_grad():
    f = train_x[:10000].float()
    mass = f.sum((1, 2))
    ys = (f.sum(2) * torch.arange(28)).sum(1) / mass
    xs = (f.sum(1) * torch.arange(28)).sum(1) / mass
    ink = f > 0
    rows = ink.any(2)
    cols = ink.any(1)
    hgt = rows.sum(1).float()
    wid = cols.sum(1).float()
    box = torch.maximum(hgt, wid)
    print(
        f"centre of mass (pixel-centre coordinates): x {xs.mean():.3f}, y {ys.mean():.3f}; "
        f"longer side of the ink box: mean {box.mean():.2f}, max {box.max():.0f}",
        flush=True,
    )
    input_stats = {"com_x": round(float(xs.mean()), 3), "com_y": round(float(ys.mean()), 3), "box": round(float(box.mean()), 2)}


def to_input(u8):
    """uint8 images -> float tensor N x 1 x 28 x 28 in [0, 1] (no mean / std normalisation)."""
    return u8.float().div_(255.0).unsqueeze(1)


def augment(x):
    """Random affine per image: rotation, isotropic scale, translation (bilinear, zero padding)."""
    n = x.shape[0]
    ang = (torch.rand(n) * 2 - 1) * math.radians(args.rot)
    sc = args.scale[0] + torch.rand(n) * (args.scale[1] - args.scale[0])
    tx = (torch.rand(n) * 2 - 1) * args.shift * 2 / 28
    ty = (torch.rand(n) * 2 - 1) * args.shift * 2 / 28
    cos = torch.cos(ang) / sc
    sin = torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x, taps=None):
        a1 = F.relu(self.conv1(x))
        p1 = F.max_pool2d(a1, 2)
        a2 = F.relu(self.conv2(p1))
        p2 = F.max_pool2d(a2, 2)
        out = self.fc(p2.flatten(1))
        if taps is not None:
            taps.update(conv1=a1, pool1=p1, conv2=a2, pool2=p2)
        return out


model = Net()
n_params = sum(q.numel() for q in model.parameters())
print(f"parameters: {n_params:,}", flush=True)
ckpt_path = os.path.join(work, "mnist-cnn.pt")


@torch.no_grad()
def evaluate():
    model.eval()
    preds = []
    for i in range(0, len(test_x), 2000):
        preds.append(model(to_input(test_x[i : i + 2000])).argmax(1))
    model.train()
    pred = torch.cat(preds)
    return (pred == test_y).float().mean().item(), pred


history = []
seconds = 0.0
if not args.export_only:
    decay = [q for q in model.parameters() if q.dim() >= 2]
    no_decay = [q for q in model.parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = len(train_x) // args.batch
    total = args.epochs * steps_per_epoch
    warm = 200
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_x))
        run = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch : (b + 1) * args.batch]
            lr = args.lr * min(1.0, (step + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / total)))
            for g in opt.param_groups:
                g["lr"] = lr
            x = augment(to_input(train_x[idx]))
            loss = F.cross_entropy(model(x), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
        acc, _ = evaluate()
        seconds = time.time() - t0
        history.append({"epoch": epoch + 1, "train_loss": round(run / steps_per_epoch, 4), "test_acc": round(acc, 4)})
        print(f"epoch {epoch + 1:2d}  {seconds:5.0f}s  train loss {run / steps_per_epoch:.4f}  test acc {acc * 100:.2f}%", flush=True)
    torch.save({"model": model.state_dict(), "history": history, "seconds": seconds}, ckpt_path)

ck = torch.load(ckpt_path)
model.load_state_dict(ck["model"])
history = ck["history"]
seconds = ck["seconds"]
model.eval()
acc, pred = evaluate()
errors = int((pred != test_y).sum())
print(f"final test accuracy {acc * 100:.2f}%  ({errors} errors in {len(test_y)})", flush=True)

# ----------------------------------------------------------------------------- samples
wrong = (pred != test_y).nonzero().flatten().tolist()
picked = []
kinds = []
for digit in range(10):  # the first two correctly classified test images of every class
    ok = ((test_y == digit) & (pred == digit)).nonzero().flatten().tolist()[:2]
    picked += ok
    kinds += ["typical"] * len(ok)
with torch.no_grad():
    probs = F.softmax(model(to_input(test_x[wrong])), 1) if wrong else torch.zeros(0, 10)
# four mistakes, the most confidently wrong first, at most one per true class
order = sorted(range(len(wrong)), key=lambda i: -probs[i].max().item())
seen = set()
for i in order:
    lab = int(test_y[wrong[i]])
    if lab in seen:
        continue
    seen.add(lab)
    picked.append(wrong[i])
    kinds.append("error")
    if len(picked) == 24:
        break

sx = test_x[picked]
taps = {}
with torch.no_grad():
    ref = model(to_input(sx), taps)
print("sample labels:", test_y[picked].tolist())
print("sample preds: ", ref.argmax(1).tolist())


# ----------------------------------------------------------------------------- export
def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


sd = model.state_dict()
weights = {}
for short, key in [("conv1.w", "conv1.weight"), ("conv1.b", "conv1.bias"), ("conv2.w", "conv2.weight"),
                   ("conv2.b", "conv2.bias"), ("fc.w", "fc.weight"), ("fc.b", "fc.bias")]:
    a = sd[key].detach().numpy().astype("<f4")
    weights[short] = {"shape": list(a.shape), "data": b64(a)}

out = {
    "format": "mnist-cnn-v1",
    "layers": [
        {"type": "input", "shape": [1, 28, 28]},
        {"type": "conv", "name": "conv1", "kernel": 5, "in": 1, "out": 8, "shape": [8, 24, 24]},
        {"type": "relu"},
        {"type": "maxpool", "name": "pool1", "size": 2, "shape": [8, 12, 12]},
        {"type": "conv", "name": "conv2", "kernel": 5, "in": 8, "out": 16, "shape": [16, 8, 8]},
        {"type": "relu"},
        {"type": "maxpool", "name": "pool2", "size": 2, "shape": [16, 4, 4]},
        {"type": "flatten", "shape": [256]},
        {"type": "linear", "name": "fc", "in": 256, "out": 10, "shape": [10]},
    ],
    "params": n_params,
    "test_accuracy": round(acc, 4),
    "test_errors": errors,
    "test_count": int(len(test_y)),
    "train": {
        "dataset": "MNIST (60,000 training images)",
        "epochs": len(history),
        "seconds": round(seconds),
        "optimizer": f"AdamW, lr {args.lr}, cosine decay, batch {args.batch}",
        "augment": {"shift_px": args.shift, "rotate_deg": args.rot, "scale": args.scale},
        "history": history,
    },
    "input_stats": input_stats,
    "weights": weights,
    "samples": {
        "count": len(picked),
        "test_index": picked,
        "labels": test_y[picked].tolist(),
        "kinds": kinds,
        "pixels": b64(sx.numpy().astype(np.uint8)),
    },
    "ref_logits": [[round(v, 5) for v in row] for row in ref.tolist()],
    "ref_sums": {k: round(float(v[0].sum()), 4) for k, v in taps.items()},
}
path = os.path.join(args.out, "mnist-cnn.json")
with open(path, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"wrote {path} ({os.path.getsize(path) / 1e3:.1f} kB)", flush=True)
