"""Tiny character-level decoder-only Transformer (the A.3/A.4 test bed).

Attention is written out by hand; Linear, LayerNorm and Embedding come from torch.nn.
Spec: openspec/changes/tiny-transformer/specs/tiny-transformer/spec.md
"""

import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LEN = 128
SIZES = {
    "target": dict(layers=4, width=128, heads=4, dropout=0.3),  # regularised harder: it overfits first
    "draft": dict(layers=2, width=64, heads=2, dropout=0.1),
}


def attention(q, k, v):
    """Causal scaled dot-product attention. q: (B,H,Tq,d); k, v: (B,H,Tk,d) with Tk >= Tq.

    The Tq queries are the last Tq positions, so with a KV cache Tq can be 1 while Tk
    covers the whole past. Returns (output, weights).
    """
    Tq, Tk = q.shape[-2], k.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B,H,Tq,Tk); /sqrt(d) keeps softmax from saturating
    # query i sits at position Tk-Tq+i and may see keys 0..Tk-Tq+i
    visible = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
    # -inf goes in before the softmax, so the hidden positions get weight exactly 0 and rows still sum to 1
    weights = scores.masked_fill(~visible, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)  # one projection for Q, K and V; plain Linear is what A.3 quantizes
        self.proj = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cache=None):
        B, T, W = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, W // self.heads).permute(2, 0, 3, 1, 4)  # each (B,H,T,d)
        if cache is not None:  # cache = (keys, values) of the earlier positions
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        out, _ = attention(q, k, v)
        return self.drop(self.proj(out.transpose(1, 2).reshape(B, T, W))), (k, v)


def sinusoidal(max_len, width):
    """Fixed position table: sin/cos(pos / 10000^(2i/width)); no learned parameters."""
    pos = torch.arange(max_len).unsqueeze(1)
    freq = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
    table = torch.zeros(max_len, width)
    table[:, 0::2], table[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return table


class Block(nn.Module):
    """Pre-norm decoder block: x + Attn(LN(x)), then x + MLP(LN(x)). Pre-norm trains without warmup."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = MultiHeadAttention(width, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x, cache=None):
        a, cache = self.attn(self.ln1(x), cache)
        x = x + a
        return x + self.mlp(self.ln2(x)), cache


class TinyTransformer(nn.Module):
    def __init__(self, vocab, layers, width, heads, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, width)
        self.register_buffer("pos", sinusoidal(max_len, width), persistent=False)  # a buffer, not a parameter
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids, cache=None):
        """ids: (B,T) new characters. cache: one (k, v) per layer, or None. Returns (logits (B,T,V), new cache).

        The cache is returned, not stored, so a speculative decoder can drop rejected positions by slicing it.
        """
        past = 0 if cache is None else cache[0][0].shape[2]
        T = ids.shape[1]
        if past + T > self.max_len:
            raise ValueError(f"sequence length {past + T} exceeds the maximum context {self.max_len}")
        x = self.drop(self.embed(ids) + self.pos[past:past + T])
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, None if cache is None else cache[i])
            new_cache.append(c)
        return self.head(self.ln_f(x)), new_cache


def make_model(kind, vocab):
    """kind is 'target' or 'draft'; both share the vocabulary and context length."""
    return TinyTransformer(vocab, **SIZES[kind])


def save_model(model, kind, path):
    tmp = Path(f"{path}.tmp")  # write beside the target, then rename, so a crash never leaves a half-written file
    torch.save({"kind": kind, "vocab": model.embed.num_embeddings, "state": model.state_dict()}, tmp)
    os.replace(tmp, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(ckpt["kind"], ckpt["vocab"])
    model.load_state_dict(ckpt["state"])
    return model.eval()


class Tokenizer:
    """Characters sorted and numbered; fitted once on the corpus and shared by both models."""

    def __init__(self, chars):
        self.chars = chars
        self.index = {c: i for i, c in enumerate(chars)}

    @classmethod
    def fit(cls, text):
        return cls("".join(sorted(set(text))))

    def encode(self, text):
        unknown = sorted({c for c in text if c not in self.index})
        if unknown:
            raise ValueError(f"character {unknown[0]!r} is not in the vocabulary")
        return [self.index[c] for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])


def _self_test():
    torch.manual_seed(0)
    # attention: agrees with the library, rows sum to 1, nothing above the diagonal, no look-ahead
    q, k, v = (torch.randn(2, 3, 10, 8) for _ in range(3))
    out, w = attention(q, k, v)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 10), atol=1e-6)
    assert (w.triu(diagonal=1) == 0).all()
    k2, v2 = k.clone(), v.clone()
    k2[..., -1, :], v2[..., -1, :] = torch.randn(8), torch.randn(8)
    assert torch.equal(attention(q, k2, v2)[0][..., :-1, :], out[..., :-1, :])

    # multi-head keeps the shape; the position table is deterministic and has no parameter
    assert MultiHeadAttention(32, 4, 0.0)(torch.randn(2, 5, 32))[0].shape == (2, 5, 32)
    assert torch.equal(sinusoidal(16, 32), sinusoidal(16, 32))

    # model: logits shape, over-long input rejected, no learned position parameter
    m = TinyTransformer(100, layers=2, width=32, heads=4, max_len=32).eval()
    assert m(torch.randint(0, 100, (2, 16)))[0].shape == (2, 16, 100)
    assert not any("pos" in name for name, _ in m.named_parameters())
    try:
        m(torch.randint(0, 100, (1, 33)))
        raise AssertionError("over-long sequence was accepted")
    except ValueError:
        pass

    # KV cache: one character at a time equals the full pass, and the cache grows by one per step
    ids = torch.randint(0, 100, (1, 20))
    full, _ = m(ids)
    cache, steps = None, []
    for t in range(ids.shape[1]):
        logits, cache = m(ids[:, t:t + 1], cache)
        steps.append(logits)
        assert all(kk.shape[2] == t + 1 and vv.shape[2] == t + 1 for kk, vv in cache)
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)

    # tokenizer: round trip, unknown character named in the error
    tok = Tokenizer.fit("Trăm năm trong cõi người ta")
    assert tok.decode(tok.encode("năm ta")) == "năm ta"
    try:
        tok.encode("z")
        raise AssertionError("unknown character was accepted")
    except ValueError as e:
        assert "'z'" in str(e)

    # target and draft are compatible; the draft is smaller
    t_model, d_model = make_model("target", 132), make_model("draft", 132)
    assert t_model.embed.num_embeddings == d_model.embed.num_embeddings and t_model.max_len == d_model.max_len
    assert sum(p.numel() for p in d_model.parameters()) < sum(p.numel() for p in t_model.parameters())
