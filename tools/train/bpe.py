"""Tiny byte-pair-encoding tokenizer shared by the training script and (mirrored in JS) the website.

The rules are deliberately small so that src/ml/tinygpt.js can reproduce them exactly:

1. normalize(text): map curly quotes / dashes to ASCII, drop anything outside printable ASCII + "\n".
2. pre-tokenize with PRETOKEN_RE: contractions, words/numbers/punctuation with an optional
   leading space, single whitespace characters.
3. inside each pre-token, repeatedly merge the adjacent pair with the lowest merge rank.

Token 0 is the special <|endoftext|> token, which also acts as beginning-of-story.
"""
import collections
import re

EOT = "<|endoftext|>"
PRETOKEN_RE = re.compile(r"'(?:s|t|re|ve|m|ll|d)(?![A-Za-z])| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]|\s")
_NORMALIZE = {
    "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-",
    "…": "...", " ": " ", "\x92": "'", "\x93": '"', "\x94": '"', "\t": " ", "\r": "",
}
_NORMALIZE_RE = re.compile("|".join(re.escape(k) for k in _NORMALIZE))
_DROP_RE = re.compile(r"[^\x20-\x7e\n]")


def normalize(text: str) -> str:
    text = _NORMALIZE_RE.sub(lambda m: _NORMALIZE[m.group(0)], text)
    return _DROP_RE.sub("", text)


def pretokenize(text: str):
    return PRETOKEN_RE.findall(text)


def train_bpe(word_freq: dict, vocab_size: int, verbose: bool = True):
    """Classic BPE over a {pre-token: count} table with incremental pair-count updates."""
    chars = sorted({ch for w in word_freq for ch in w})
    vocab = [EOT] + chars
    words = [list(w) for w in word_freq]
    freqs = [word_freq[w] for w in word_freq]
    pair_count = collections.Counter()
    pair_words = collections.defaultdict(set)
    for wi, sym in enumerate(words):
        f = freqs[wi]
        for a, b in zip(sym, sym[1:]):
            pair_count[(a, b)] += f
            pair_words[(a, b)].add(wi)
    merges = []
    while len(vocab) < vocab_size and pair_count:
        # deterministic tie-break: highest count, then lexicographically smallest pair
        best = min(pair_count.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if pair_count[best] < 2:
            break
        a, b = best
        merged = a + b
        merges.append((a, b))
        vocab.append(merged)
        for wi in list(pair_words[best]):
            sym = words[wi]
            f = freqs[wi]
            for x, y in zip(sym, sym[1:]):
                pair_count[(x, y)] -= f
                if pair_count[(x, y)] <= 0:
                    del pair_count[(x, y)]
                pair_words[(x, y)].discard(wi)
            out = []
            i = 0
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            words[wi] = out
            for x, y in zip(out, out[1:]):
                pair_count[(x, y)] += f
                pair_words[(x, y)].add(wi)
        if verbose and len(vocab) % 256 == 0:
            print(f"  bpe vocab {len(vocab)}: last merge {a!r}+{b!r}", flush=True)
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges):
        self.vocab = list(vocab)
        self.merges = [tuple(m) for m in merges]
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self._cache = {}

    def _bpe(self, word: str):
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        sym = list(word)
        while len(sym) > 1:
            best_rank, best_i = None, -1
            for i in range(len(sym) - 1):
                r = self.rank.get((sym[i], sym[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = sym[best_i], sym[best_i + 1]
            out = []
            i = 0
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            sym = out
        ids = [self.token_to_id[s] for s in sym if s in self.token_to_id]
        self._cache[word] = ids
        return ids

    def encode(self, text: str):
        ids = []
        for word in pretokenize(normalize(text)):
            ids.extend(self._bpe(word))
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)
