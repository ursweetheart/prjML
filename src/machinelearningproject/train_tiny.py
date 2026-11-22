"""Train the tiny target and draft models on Truyện Kiều and sample from them.

Run: uv run python -m machinelearningproject.train_tiny [--steps N] [--models target draft]
Spec: openspec/changes/tiny-transformer/specs/tiny-transformer/spec.md
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from machinelearningproject.tiny_transformer import (
    MAX_LEN, SIZES, Tokenizer, load_model, make_model, save_model,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/raw/truyen_kieu.txt"
OUT = ROOT / "data/processed/tiny"
SEED = 0
BATCH, EVAL_EVERY, VAL_FRACTION = 64, 100, 0.1
LR = 3e-3
WEIGHT_DECAY = {"target": 0.1, "draft": 0.01}  # 0.01 is the AdamW default


@torch.no_grad()
def sample(model, tok, prompt, n, temperature=1.0, seed=0, use_cache=True):
    """Draw n characters after the prompt from the softmax, with a local generator so the seed alone decides."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    out = tok.encode(prompt)
    logits, cache = model(torch.tensor([out]))
    for _ in range(n):
        probs = (logits[0, -1] / temperature).softmax(-1)
        out.append(torch.multinomial(probs, 1, generator=g).item())
        if use_cache and len(out) <= model.max_len:
            logits, cache = model(torch.tensor([[out[-1]]]), cache)
        else:  # window full: the oldest character drops out and positions shift, so the cache cannot be reused
            logits, _ = model(torch.tensor([out[-model.max_len:]]))
    return tok.decode(out)


def batch(data, block, g, device):
    ix = torch.randint(0, len(data) - block, (BATCH,), generator=g)  # window i..i+block, target is the window shifted by one
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate(model, data, block, device, batches=20):
    model.eval()
    g = torch.Generator().manual_seed(1)  # same windows every time, so losses are comparable across steps
    losses = []
    for _ in range(batches):
        x, y = batch(data, block, g, device)
        losses.append(F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten()).item())
    return sum(losses) / len(losses)


def train(kind, vocab, train_data, val_data, steps, device):
    torch.manual_seed(SEED)
    model = make_model(kind, vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY[kind])
    block = min(MAX_LEN, len(val_data) - 1)  # shorter only when a tiny test corpus has less than a window of validation text
    assert len(train_data) > block + 1, "corpus too short"
    g = torch.Generator().manual_seed(SEED)
    best_val, best_state = float("inf"), None
    for step in range(1, steps + 1):
        model.train()
        x, y = batch(train_data, block, g, device)
        loss = F.cross_entropy(model(x)[0].flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            tl, vl = estimate(model, train_data, block, device), estimate(model, val_data, block, device)
            print(f"  {kind} step {step:5d}  train {tl:.3f}  val {vl:.3f}")
            if vl < best_val:  # overfitting is expected on ~100k characters, so keep the best-validation weights
                best_val, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu().eval(), best_val


def _self_test():
    tok = Tokenizer.fit("Trăm năm trong cõi người ta")
    torch.manual_seed(0)
    m = make_model("draft", len(tok.chars))
    a = sample(m, tok, "Trăm", 200, seed=1)  # 200 > max_len also exercises the sliding window
    assert a == sample(m, tok, "Trăm", 200, seed=1)  # same seed, same text
    assert a == sample(m, tok, "Trăm", 200, seed=1, use_cache=False)  # the cache does not change the sample
    assert len(a) == 204 and all(c in tok.chars for c in a)
    with tempfile.TemporaryDirectory() as d:
        save_model(m, "draft", Path(d) / "m.pt")
        ids = torch.randint(0, len(tok.chars), (2, 12))
        assert torch.equal(m(ids)[0], load_model(Path(d) / "m.pt")(ids)[0])  # reload gives identical logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--models", nargs="+", choices=list(SIZES), default=list(SIZES))
    args = ap.parse_args()
    if not args.corpus.exists():  # before anything is written
        sys.exit(f"corpus not found: {args.corpus} (see the Tiny Transformer section of README.md)")
    _self_test()
    text = args.corpus.read_text(encoding="utf-8")
    tok = Tokenizer.fit(text)  # one vocabulary, fitted on all the text, shared by target and draft
    data = torch.tensor(tok.encode(text))
    cut = int(len(data) * (1 - VAL_FRACTION))  # contiguous tail as validation, so no neighbouring lines leak
    train_data, val_data = data[:cut], data[cut:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    tok.save(args.out / "vocab.json")
    print(f"{len(text)} characters, vocabulary {len(tok.chars)}, uniform-guess loss ln V = {math.log(len(tok.chars)):.3f}, device {device}")
    for kind in args.models:
        model, best_val = train(kind, len(tok.chars), train_data, val_data, args.steps, device)
        save_model(model, kind, args.out / f"{kind}.pt")
        print(f"{kind}: best val {best_val:.3f}, saved {args.out / f'{kind}.pt'}")
        print(sample(model, tok, text[:8], 300, seed=0), "\n")


if __name__ == "__main__":
    main()
