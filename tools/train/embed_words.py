"""Embed the vocabulary, the mini corpus and the test questions of the Embedding Space lab.

  HF_HOME=/some/cache python tools/train/embed_words.py --out public/data/embeddings.json

Everything is embedded with sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, unit length).
Output (one JSON file, well under 700 kB):
  words      400 words / short phrases in 16 everyday categories, their vectors and a 3D layout
             (PCA to 50 dimensions, then t-SNE to 3; UMAP is used instead when it is installed)
  passages   a mini corpus of short passages about the AI concepts this site covers
  questions  test questions with the ids of their relevant passages and how each was written:
             "paraphrase" (no content word shared with the passage), "exact" (hinges on a rare
             term or identifier) or "plain"
Vectors are stored as int8 (each vector scaled by its own largest component, which leaves cosine
similarity unchanged) and base64-encoded. The browser never runs the embedding model; it only
does arithmetic on these vectors. BM25, cosine search and rank fusion are computed live in
src/ml/retrieval.js.
"""
import argparse
import base64
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", default="public/data/embeddings.json")
p.add_argument("--threads", type=int, default=2)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--perplexity", type=float, default=22.0)
args = p.parse_args()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch  # noqa: E402

torch.set_num_threads(args.threads)
from sentence_transformers import SentenceTransformer  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------------------------------------------------------- vocabulary
# 16 categories x 25 entries. Several words deliberately have a second meaning in another
# category (python, turkey, bass, orange, mouse, temple, cricket ...): the 384-d vector can sit
# near both, the 3D map has to pick one place.
WORDS = {
    "animals": "dog cat horse cow sheep lion tiger elephant giraffe wolf fox bear rabbit mouse dolphin whale shark eagle owl penguin frog snake butterfly bat jaguar",
    "food": "bread cheese rice pasta pizza soup salad apple banana grape strawberry tomato potato carrot onion egg butter chocolate cake cookie honey coffee tea salmon sandwich",
    "vehicles": "car truck bus train bicycle motorcycle airplane helicopter boat ship submarine tram taxi ambulance tractor scooter van rocket ferry canoe subway sailboat jeep hot_air_balloon fire_truck",
    "emotions": "happy sad angry afraid joy love hate jealous proud ashamed lonely excited bored anxious calm surprised grateful hopeful nervous confused disgust grief cheerful furious relieved",
    "professions": "doctor nurse teacher engineer lawyer farmer chef pilot firefighter police_officer carpenter plumber electrician dentist journalist accountant architect scientist librarian mechanic pharmacist judge baker waiter programmer",
    "countries": "France Germany Italy Spain Japan China India Brazil Canada Mexico Egypt Kenya Australia Russia Sweden Norway Greece Turkey Argentina Nigeria Thailand Vietnam Poland Portugal South_Korea",
    "sports": "soccer basketball tennis baseball golf swimming running cycling boxing skiing surfing hockey volleyball rugby cricket marathon gymnastics wrestling archery fencing rowing badminton karate skateboarding table_tennis",
    "music": "guitar piano violin drums flute trumpet saxophone cello harp bass melody rhythm harmony song opera jazz orchestra choir concert symphony singer composer chord lullaby hip_hop",
    "tools": "hammer screwdriver wrench saw drill pliers axe shovel rake ladder chisel tape_measure nail screw bolt scissors knife sandpaper crowbar toolbox paintbrush wheelbarrow clamp mallet glue_gun",
    "weather": "rain snow wind storm thunder lightning fog hail cloud sunshine hurricane tornado drizzle blizzard frost heat_wave humidity rainbow breeze flood drought forecast temperature monsoon sleet",
    "body": "head hand foot arm leg eye ear nose mouth heart lung brain stomach liver kidney skin bone muscle finger knee shoulder tooth spine elbow palm",
    "colors": "red blue green yellow orange purple pink brown black white gray turquoise violet crimson scarlet beige navy_blue gold silver maroon indigo teal lavender olive magenta",
    "programming": "python javascript java compiler variable function loop array database algorithm bug debugger recursion class object string pointer API server git source_code unit_test operating_system web_browser syntax_error",
    "science": "atom molecule electron gravity energy cell DNA evolution photosynthesis chemistry physics biology telescope microscope experiment hypothesis planet galaxy black_hole magnet oxygen laboratory quantum bacteria mercury",
    "family": "mother father sister brother daughter son grandmother grandfather aunt uncle cousin wife husband baby twins parents nephew niece wedding family stepmother grandchild mother-in-law godfather sibling",
    "buildings": "house castle church temple mosque school hospital library museum skyscraper bridge tower factory barn hotel stadium lighthouse palace cottage apartment cathedral warehouse airport train_station cabin",
}


def split_words(spec):
    """Entries are separated by spaces; an underscore stands for a space inside a phrase."""
    return [w.replace("_", " ") for w in spec.split()]


# ----------------------------------------------------------------------------- mini corpus
# (id, title, text). Short, factual passages about concepts explained elsewhere on the site.
PASSAGES = [
    ("tokens", "Tokens", "A language model does not read letters or whole words. Text is first cut into tokens, pieces drawn from a fixed vocabulary, and every token is replaced by an integer id."),
    ("bpe", "Byte pair encoding", "Byte pair encoding (BPE) builds a vocabulary by repeatedly merging the most frequent pair of adjacent symbols. Common words end up as a single token, while rare words are spelled out from several pieces."),
    ("token-embedding", "Token embeddings", "The first layer of a language model is a lookup table with one learned vector per token id. Tokens that appear in similar contexts end up with similar vectors."),
    ("positions", "Position information", "Self-attention by itself ignores the order of the tokens, so position information has to be injected. Older models add a learned vector per position; RoPE instead rotates queries and keys by an angle that grows with the position."),
    ("attention", "Self-attention", "In self-attention every token builds a query, compares it with the keys of all tokens, and takes a weighted average of their values. The weights come from a softmax over the query-key dot products."),
    ("transformer-block", "Transformer block", "A transformer block is self-attention followed by a small feed-forward network, each wrapped in a residual connection and layer normalization. A modern language model stacks dozens of such blocks."),
    ("next-token", "Next-token prediction", "A GPT is trained on one task: predict the next token. A causal mask hides the future, so a single pass over a text gives a training signal at every position."),
    ("temperature", "Temperature", "A language model outputs a score, called a logit, for every token in its vocabulary. Dividing the logits by a temperature before the softmax makes sampling more random above 1 and more predictable below 1."),
    ("top-p", "top_p and top_k", "Nucleus sampling, the top_p setting, keeps the smallest set of tokens whose probabilities add up to p and samples only among them. The top_k setting keeps a fixed number of candidates instead."),
    ("kv-cache", "KV cache", "While a model generates text, the keys and values computed for earlier tokens never change, so they are stored in a KV cache. Each new token then needs one pass for a single position instead of a pass over the whole prefix."),
    ("context-window", "Context window", "The context window is the largest number of tokens a model can take into account at once. Anything beyond it is cut off or has to be summarized, and memory use grows with every token that is kept."),
    ("quantization", "Quantization", "Quantization stores each weight with fewer bits, for example as an 8-bit or 4-bit integer instead of a 32-bit float. The file becomes several times smaller and inference faster, at the price of a small loss in accuracy."),
    ("gguf", "GGUF", "GGUF is the single-file format in which llama.cpp stores a model: the weights, the tokenizer and the settings needed to run it. Most GGUF files hold quantized weights, and the file name states the scheme."),
    ("gguf-q4", "Q4_K_M", "Q4_K_M is a GGUF quantization preset that stores about 4 to 5 bits per weight. It is the usual default for running a model locally: about a third of the 16-bit size for a small loss in quality."),
    ("gguf-q8", "Q8_0", "Q8_0 is a GGUF quantization preset that stores about 8 bits per weight. The file is roughly half the 16-bit size, and the output is almost indistinguishable from the original."),
    ("gguf-q2", "Q2_K", "Q2_K is a GGUF quantization preset that stores about 3 bits per weight. It is the smallest of the common presets, and the loss in quality is plain to see."),
    ("lora", "LoRA", "LoRA (low-rank adaptation) freezes the pretrained weights and learns a small update for chosen layers: the product of two thin matrices. Well under one percent of the parameters are trained, and the adapter can be merged into the weights afterward."),
    ("fine-tuning", "Fine-tuning", "Fine-tuning continues the training of a pretrained model on a smaller, task-specific dataset. It is good at changing tone, format or skill, and a poor way to store facts that change often."),
    ("distillation", "Distillation", "In knowledge distillation a compact student network is trained to reproduce the output probabilities of a much larger teacher. The student keeps most of the quality at a fraction of the cost."),
    ("rag", "Retrieval-augmented generation", "Retrieval-augmented generation, or RAG, searches a document collection for passages related to the question and pastes them into the prompt. The model answers from that text, so its knowledge can be updated without retraining."),
    ("chunking", "Chunking", "Before documents are indexed they are cut into chunks of a few hundred tokens, often with some overlap. Chunks that are too large blur the match; chunks that are too small lose their context."),
    ("sentence-embedding", "Sentence embeddings", "A sentence embedding model turns a piece of text into one fixed-length vector, placed so that texts with similar meaning lie close together. Closeness is usually measured by cosine similarity."),
    ("minilm", "all-MiniLM-L6-v2", "all-MiniLM-L6-v2 is a sentence embedding model with 6 transformer layers and about 23 million parameters. It outputs vectors with 384 dimensions and is fast enough to run on a CPU."),
    ("mpnet", "all-mpnet-base-v2", "all-mpnet-base-v2 is a sentence embedding model with 12 transformer layers and about 110 million parameters. It outputs vectors with 768 dimensions and is slower but more accurate than the small models."),
    ("bm25", "BM25", "BM25 ranks documents by the query words they contain. A rare word counts for more than a common one, repeating a word brings diminishing returns, and long documents are penalized."),
    ("rrf", "Hybrid search and RRF", "Hybrid search runs keyword search and vector search side by side and merges the two rankings. Reciprocal rank fusion (RRF) gives each document the sum of 1 / (k + rank) over the lists, with k usually set to 60."),
    ("reranker", "Reranking", "A reranker is a cross-encoder: it reads the question and one candidate passage together and outputs a relevance score. It is too slow for a whole collection, so it only reorders the top results of a faster first stage."),
    ("flat-search", "Exact search", "Exact nearest-neighbor search compares the query with every stored vector. It always returns the true neighbors, but its cost grows in step with the size of the collection."),
    ("hnsw", "HNSW", "HNSW (hierarchical navigable small world) is a graph index. Every vector is linked to a few close neighbors on several layers; a search enters at the sparse top layer and walks greedily toward the query, layer by layer."),
    ("hnsw-m", "HNSW: M", "In HNSW the parameter M is the number of links kept per node. A larger M improves recall on hard data and makes the index larger and slower to build."),
    ("hnsw-efc", "HNSW: efConstruction", "In HNSW the parameter efConstruction is the length of the candidate list while the graph is being built. A larger value gives a better graph and a longer build; it does not change the size of the index."),
    ("hnsw-efs", "HNSW: efSearch", "In HNSW the parameter efSearch is the length of the candidate list during a query. A larger value raises recall and makes every search slower; it can be changed at any time without rebuilding."),
    ("ivf", "IVF", "An IVF (inverted file) index groups the vectors into cells with k-means and stores each vector in the list of its nearest centroid. A query scans only the nprobe closest cells instead of the whole collection."),
    ("pq", "Product quantization", "Product quantization (PQ) cuts a vector into sub-vectors and replaces each piece with the id of its nearest codebook entry. A vector of several kilobytes shrinks to a few dozen bytes, and distances are estimated from small lookup tables."),
    ("lsh", "Locality-sensitive hashing", "Locality-sensitive hashing (LSH) draws random hyperplanes and records on which side of each one a vector falls. Vectors that point in similar directions tend to get the same bit pattern and land in the same bucket."),
    ("diffusion", "Diffusion models", "A diffusion model learns to remove noise. Training images are corrupted with Gaussian noise in many small steps, and the network learns to predict the noise that was added so that the corruption can be undone."),
    ("ddim", "Samplers", "Image generation with a diffusion model starts from pure noise and denoises it step by step. Samplers such as DDIM take larger, deterministic steps and cut the count from a thousand to a few dozen."),
    ("cfg", "Classifier-free guidance", "Classifier-free guidance (CFG) runs the denoiser twice, with and without the text prompt, and exaggerates the difference between the two predictions. A higher guidance scale follows the prompt more closely and reduces variety."),
    ("q-learning", "Q-learning", "Q-learning keeps one number for every state and action: the reward expected from there on. After each step the number moves toward the reward just received plus the discounted best value of the next state."),
    ("sarsa", "SARSA", "SARSA updates toward the value of the action the agent actually takes next, exploratory moves included. Next to a cliff it therefore learns a longer, safer path than Q-learning."),
    ("epsilon-greedy", "Epsilon-greedy", "An epsilon-greedy agent picks a random action with probability epsilon and the best known action otherwise. Without the random moves it would never discover an option better than its first success."),
    ("dyna-q", "Dyna-Q", "Dyna-Q records what each action did and uses this learned model of the world to replay imagined transitions. Every real step is followed by many planning updates, so far fewer real steps are needed."),
    ("a-star", "A* search", "A* search always expands the node with the lowest sum of the cost so far and a heuristic estimate of the cost still to go. If the heuristic never overestimates, the first path that reaches the goal is optimal."),
    ("alphazero", "AlphaZero", "AlphaZero combines Monte Carlo tree search with a neural network that suggests moves and evaluates positions. It learns by playing against itself, using the search results as training targets for the network."),
    ("mcp", "Model Context Protocol", "The Model Context Protocol, MCP, is an open standard that connects AI applications to external tools and data. Client and server exchange JSON-RPC 2.0 messages: tools/list announces the available tools and tools/call runs one."),
    ("tool-calling", "Tool calling", "With tool calling a model replies with a structured request, a function name plus JSON arguments, instead of plain text. The application runs the function and sends the result back as a new message."),
    ("agents", "Agents", "An agent runs a model in a loop: the model chooses an action, a tool carries it out, and the observation is appended to the context. The loop ends when the model decides that the task is done."),
    ("backprop", "Backpropagation", "Backpropagation applies the chain rule from the loss backward through every layer. One backward pass yields the gradient of the loss with respect to every weight."),
    ("adam", "Adam", "Adam keeps a running average of each weight's gradient and of its square, and divides the first by the square root of the second. Every weight thus gets its own step size."),
    ("cnn", "Convolution", "A convolutional layer slides a small filter across the image and computes the same weighted sum at every position. Early layers respond to edges, deeper layers to textures and parts of objects."),
    ("batching", "Batching", "An inference server groups requests that arrive close together and pushes them through the model as one batch. Throughput rises sharply while each request waits a little longer."),
    ("hallucination", "Hallucination", "A language model can state falsehoods fluently, because it produces likely text instead of looking facts up. Grounding answers in retrieved sources and asking for citations reduces the problem."),
    ("http-429", "HTTP 429", "An API that answers with HTTP 429 is saying that the client has exceeded its rate limit of requests or tokens per minute. The client should wait, as the Retry-After header suggests, and retry with exponential backoff."),
    ("http-401", "HTTP 401", "An API that answers with HTTP 401 is saying that the request carries no valid API key. Retrying will not help; the key or the Authorization header has to be fixed."),
    ("http-400", "HTTP 400", "An API that answers with HTTP 400 is saying that the request itself is invalid, for example because the prompt is longer than the context window. The request has to be changed before it is sent again."),
]

# (text, [relevant passage ids], kind). The set is deliberately lopsided: it is built to show where
# each retrieval method breaks, not to estimate how often that happens in real traffic.
QUESTIONS = [
    # paraphrase: shares no content word with the passage that answers it
    ("Why does my chatbot confidently make things up?", ["hallucination"], "paraphrase"),
    ("How do picture generators turn random static into a photo?", ["ddim", "diffusion"], "paraphrase"),
    ("Should a robot stick with what works or gamble on something untested?", ["epsilon-greedy"], "paraphrase"),
    ("How do I find the shortest route when I already have the whole map?", ["a-star"], "paraphrase"),
    ("How does a vision system notice outlines and shapes in a photo?", ["cnn"], "paraphrase"),
    # exact: hinges on one rare term, identifier or number
    ("Which model outputs 768 dimensions?", ["mpnet"], "exact"),
    ("What is tools/call?", ["mcp"], "exact"),
    ("Which preset stores about 3 bits per weight?", ["gguf-q2"], "exact"),
    ("What is nprobe?", ["ivf"], "exact"),
    # plain: ordinary wording, some words shared with the passage
    ("How can a tiny cheap network imitate a big expensive one?", ["distillation"], "plain"),
    ("What does a KV cache store?", ["kv-cache"], "plain"),
    ("How does quantization make a model smaller?", ["quantization"], "plain"),
]


def b64_int8(mat):
    """Scale every row by its own largest component (cosine is unchanged) and round to int8."""
    scale = np.abs(mat).max(axis=1, keepdims=True)
    q = np.rint(mat / scale * 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode("ascii"), q


def cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def layout3d(vec, seed, perplexity):
    """384-d -> 3-d. UMAP when available, otherwise PCA(50) + t-SNE(3)."""
    try:
        import umap  # type: ignore

        xyz = umap.UMAP(n_components=3, metric="cosine", n_neighbors=12, min_dist=0.25, random_state=seed).fit_transform(vec)
        method = "UMAP (cosine, 12 neighbors) to 3 dimensions"
    except ImportError:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        reduced = PCA(n_components=50, random_state=seed).fit_transform(vec)
        xyz = TSNE(n_components=3, perplexity=perplexity, init="pca", learning_rate="auto", max_iter=1500, random_state=seed).fit_transform(reduced)
        method = f"PCA to 50 dimensions, then t-SNE (perplexity {perplexity:g}) to 3"
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.abs(xyz).max()
    return xyz, method


def main():
    words, cats = [], []
    names = list(WORDS)
    for ci, name in enumerate(names):
        entries = split_words(WORDS[name])
        assert len(entries) == 25, (name, len(entries), entries)
        words += entries
        cats += [ci] * len(entries)
    assert len(set(w.lower() for w in words)) == len(words), "duplicate word"
    ids = [pid for pid, _, _ in PASSAGES]
    assert len(set(ids)) == len(ids), "duplicate passage id"
    for q, rel, kind in QUESTIONS:
        assert all(r in ids for r in rel), (q, rel)
        assert kind in ("paraphrase", "exact", "plain")

    model = SentenceTransformer(MODEL, device="cpu")
    enc = lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)  # noqa: E731
    wv = enc(words)
    pv = enc([f"{title}. {text}" for _, title, text in PASSAGES])  # the title is part of what gets indexed
    qv = enc([q for q, _, _ in QUESTIONS])
    dim = wv.shape[1]

    xyz, method = layout3d(wv, args.seed, args.perplexity)

    w64, wq = b64_int8(wv)
    p64, pq = b64_int8(pv)
    q64, qq = b64_int8(qv)

    # ---- sanity report
    err = np.abs(cosine(wq.astype(np.float32), wq.astype(np.float32)) - wv @ wv.T).max()
    print(f"{len(words)} words, {len(PASSAGES)} passages, {len(QUESTIONS)} questions, dim {dim}")
    print(f"int8 quantization: largest change of a word-word cosine {err:.4f}")
    sim = wv @ wv.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(axis=1)
    same = np.mean([cats[i] == cats[j] for i, j in enumerate(nn)])
    print(f"nearest neighbor in the same category: {same:.1%}")
    k = 8
    hi = np.argsort(-sim, axis=1)[:, :k]
    d3 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d3, np.inf)
    lo = np.argsort(d3, axis=1)[:, :k]
    keep = np.mean([len(set(a) & set(b)) / k for a, b in zip(hi, lo)])
    print(f"layout: {method}; {keep:.1%} of each word's {k} nearest neighbors survive the projection")
    dense = cosine(qq.astype(np.float32), pq.astype(np.float32))
    for qi, (q, rel, kind) in enumerate(QUESTIONS):
        order = np.argsort(-dense[qi])
        rank = min(int(np.where(order == ids.index(r))[0][0]) + 1 for r in rel)
        print(f"  dense rank {rank:2d}  [{kind:10s}] {q}  -> top: {ids[order[0]]}, {ids[order[1]]}, {ids[order[2]]}")

    out = {
        "model": MODEL,
        "dim": int(dim),
        "encoding": "int8, each vector scaled by its own largest component, base64",
        "layout": method,
        "categories": names,
        "words": {
            "text": words,
            "cat": cats,
            "xyz": [round(float(v), 4) for v in xyz.reshape(-1)],
            "vec": w64,
        },
        "passages": {
            "items": [{"id": pid, "title": title, "text": text} for pid, title, text in PASSAGES],
            "vec": p64,
        },
        "questions": {
            "items": [{"text": q, "relevant": rel, "kind": kind} for q, rel, kind in QUESTIONS],
            "vec": q64,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {args.out}: {os.path.getsize(args.out) / 1000:.1f} kB")


if __name__ == "__main__":
    main()
