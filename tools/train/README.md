# Training the small models

Everything under `public/models` and `public/data` is produced by the scripts in this folder.
They run on a CPU in minutes; nothing here needs a GPU.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy sentence-transformers scikit-learn
```

## The tiny GPT (`tinygpt.bin`, `tinygpt-vocab.json`, `tinygpt-eval.json`)

A 1.07M-parameter GPT (4 blocks, 4 heads, width 128, context 128) with its own 2,048-token BPE
vocabulary, trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

```bash
mkdir -p data/tinystories && cd data/tinystories
base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
curl -L -o valid.txt "$base/TinyStoriesV2-GPT4-valid.txt"
curl -L -r 0-89999999 -o train_head.txt "$base/TinyStoriesV2-GPT4-train.txt"   # first 90 MB is plenty
cd ../..
python tools/train/train_tinygpt.py --data data/tinystories --out public/models --minutes 80
npm run test:ml        # the JavaScript forward pass must reproduce PyTorch's numbers
```

`bpe.py` is the tokenizer; `src/ml/tinygpt.js` mirrors it rule for rule, and the export contains
test vectors (token ids, logits, losses) that `tools/test-ml.mjs` checks the JavaScript against.

## The other models

Each script documents its own inputs at the top of the file:

* `train_mnist_cnn.py`: the digit classifier used by the CNN lab (`public/models/mnist-cnn.json`).
* `embed_words.py`: sentence embeddings, their 3D layout and the retrieval test set used by the
  embedding lab (`public/data/embeddings.json`).
